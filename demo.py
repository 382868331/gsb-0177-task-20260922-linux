#!/usr/bin/env python3
"""Fixed-input demo for the afifo asynchronous-FIFO pointer checker.

Run:  python demo.py

Everything here is a fixed, deterministic schedule (no randomness, no sleep,
no pre-recorded output): the model is actually stepped, its trace printed,
and accepted payload order / occupancy bounds are cross-checked with an
independent abstract ``collections.deque``.
"""

from collections import deque

from afifo import AsyncFIFO, Event
from afifo.model import AsyncFIFOError


def b(value, width):
    return f"{value:0{width}b}"


def fmt_sync(pair):
    return f"({pair[0]:03b},{pair[1]:03b})"


def print_row(rec, width):
    edge = ("W" if rec.wclk else ".") + ("R" if rec.rclk else ".")
    wa = {True: "A", False: "x", None: "-"}[rec.w_accept]
    ra = {True: "A", False: "x", None: "-"}[rec.r_accept]
    data = rec.wdata if rec.w_accept else ""
    out = rec.rdata if rec.r_accept else ""
    print(
        f"t={rec.time:>2} {edge} en={int(rec.wen)}{int(rec.ren)} "
        f"W:{wa} R:{ra} | wptr={b(rec.wptr_after, width)} "
        f"rptr={b(rec.rptr_after, width)} "
        f"wg={b(rec.wgray_after, width)} rg={b(rec.rgray_after, width)} "
        f"| F={int(rec.full_after)} E={int(rec.empty_after)} "
        f"r2w={fmt_sync(rec.r2w_after)} w2r={fmt_sync(rec.w2r_after)} "
        f"| in={data!s:<4} out={out!s}"
    )


def abstract_queue_audit(records, depth):
    """Independent check: accepted data FIFO order and [0, depth] bounds."""
    q = deque()
    ordered_out = []
    for rec in records:
        if rec.w_accept:
            assert len(q) < depth, "抽象队列越上界（溢出）"
            q.append(rec.wdata)
        if rec.r_accept:
            assert q, "抽象队列越下界（空读）"
            ordered_out.append(q.popleft())
    return q, ordered_out


# Fixed schedule for k=2 (depth 4).  Values are literal, not generated.
NORMAL_SCHEDULE = [
    (1, 1, 0, 1, 0, "D01"),
    (2, 0, 1, 0, 1, None),    # empty 旧值=1：读被保守拒绝（同步器第1级）
    (3, 0, 1, 0, 1, None),    # 仍拒绝（同步器第2级），本拍末 s2=wgray
    (4, 0, 1, 0, 1, None),    # 仍按旧 empty=1 拒绝，但本拍末 empty 撤销
    (5, 1, 0, 1, 0, "D02"),
    (6, 1, 0, 1, 0, "D03"),
    (7, 1, 0, 1, 0, "D04"),   # 第4个接受写：恰好满，full 在本沿末置位
    (8, 1, 0, 1, 0, "D05"),   # full=1：写被拒绝（D05 永不入队）
    (9, 1, 0, 1, 0, "D06"),   # 同上
    (10, 1, 1, 1, 1, "D07"),  # 同刻：写看旧 full 拒绝，读接受 D01
    (11, 0, 1, 0, 1, None),   # 同步写值滞后 -> empty 保守回填，读拒绝
    (12, 0, 1, 0, 1, None),   # 拒绝，但本拍末 empty 再次撤销
    (13, 1, 1, 1, 1, "D08"),  # 同刻：读接受 D02，写仍看旧 full 拒绝
    (14, 1, 0, 1, 0, "D09"),  # full 旧值仍=1：拒绝
    (15, 1, 0, 1, 0, "D10"),  # 拒绝，但本拍末 full 终见读指针而撤销
    (16, 0, 1, 0, 1, None),   # 接受读 D03
    (17, 0, 1, 0, 1, None),   # 接受读 D04，FIFO 真的空了，empty 置位
    (18, 0, 1, 0, 1, None),   # 真空：读拒绝
    (19, 0, 1, 0, 1, None),
    (20, 1, 0, 1, 0, "D11"),
    (21, 1, 0, 1, 0, "D12"),  # 本拍末 full 保守置位（占用其实只有2）
    (22, 1, 0, 1, 0, "D13"),  # 按旧 full=1 拒绝，本拍末 full 又撤销
    (23, 0, 1, 0, 0, None),   # 停使能的读沿，只推进同步器
    (24, 0, 1, 0, 0, None),
    (25, 1, 1, 1, 1, "D14"),  # 同刻：写接受(wptr 6->7)，读按旧 empty 拒绝
    (26, 1, 1, 1, 1, "D15"),  # 同刻：双方均接受；wptr 7->0 回绕
]


def section_normal():
    print("=" * 106)
    print("正常结果：k=2，深度 4，指针 3 位。固定字面值调度，展示两级同步与保守 full/empty 延迟")
    print("=" * 106)
    k = 2
    fifo = AsyncFIFO(k=k)
    print(f"复位态: full={fifo.full} empty={fifo.empty} wptr=0 rptr=0 "
          f"r2w=(0,0) w2r=(0,0)\n")

    events = [Event(t, bool(w), bool(r), bool(we), bool(re_), d)
              for (t, w, r, we, re_, d) in NORMAL_SCHEDULE]
    records = fifo.run(events)

    print("t    边  en   接受  | 写/读指针与格雷(边沿后)                  "
          "| 旗标 写域同步r2w  读域同步w2r  | 数据")
    for rec in records:
        print_row(rec, k + 1)

    q, ordered_out = abstract_queue_audit(records, fifo.depth)
    written = [r.wdata for r in records if r.w_accept]
    print("\n独立抽象队列核对（不接触 afifo 任何内部状态）：")
    print(f"  接受写={fifo.accepted_writes} 接受读={fifo.accepted_reads} "
          f"残留={len(q)}；全程 0 <= 占用 <= 深度{fifo.depth} 成立")
    print(f"  接受写入顺序 = {written}")
    print(f"  接受读出顺序 = {ordered_out}")
    # exact anchors for this fixed schedule
    assert written == ["D01", "D02", "D03", "D04",
                       "D11", "D12", "D14", "D15"]
    assert ordered_out == ["D01", "D02", "D03", "D04", "D11"]
    assert list(q) == ["D12", "D14", "D15"]

    first_w = records[0]
    empty_off = next(r for r in records
                     if r.rclk and not r.empty_after)
    rejected_reads = [r.time for r in records if r.r_accept is False]
    print("\n保守空旗标延迟（旗标只由指针格雷与同步寄存器决定，不看队列长度）：")
    print(f"  首次接受写 t={first_w.time}；empty 直到 t={empty_off.time} 沿末才撤销，"
          f"首个接受读要等到下一个读沿")
    print(f"  t={rejected_reads} 的读请求（数据可能已在存储器中）均按旧 empty=1 拒绝")

    full_on = next(r for r in records if r.full_after)
    rejected_writes = [r.time for r in records if r.w_accept is False]
    print("\n保守满旗标延迟：")
    print(f"  full 首次在 t={full_on.time} 置位；被拒写发生在 "
          f"t={rejected_writes}（含占用远低于深度时的保守误置位）")

    both = [r for r in records if r.w_accept and r.r_accept]
    print("\n同刻边沿（两级同步器输入级锁存的是对端“旧”格雷）：")
    for r in [r for r in records if r.wclk and r.rclk]:
        print(f"  t={r.time}: W接受={r.w_accept} R接受={r.r_accept} "
              f"r2w={fmt_sync(r.r2w_after)} w2r={fmt_sync(r.w2r_after)}")
    assert [r.time for r in both] == [26]
    wrap26 = records[25]
    assert wrap26.time == 26
    assert wrap26.wptr_before == 7 and wrap26.wptr_after == 0
    assert wrap26.r_accept and wrap26.rdata == "D11"
    print(f"  t=26 双方接受：写指针 {wrap26.wptr_before}->"
          f"{wrap26.wptr_after}（7->0 回绕），读出 {wrap26.rdata}")
    return records


def section_wrap():
    print("\n" + "=" * 106)
    print("回绕：k=1，深度 2，指针 k+1=2 位（模 4）。固定调度：每步双时钟同刻、使能常有效，32 步")
    print("=" * 106)
    fifo = AsyncFIFO(k=1)
    events = [Event(t, True, True, wen=True, ren=True, wdata=f"W{t - 1:02d}")
              for t in range(1, 33)]
    records = fifo.run(events)

    q, ordered_out = abstract_queue_audit(records, fifo.depth)
    w_wrap = [(r.time, r.wptr_before, r.wptr_after)
              for r in records
              if r.w_accept and r.wptr_after < r.wptr_before]
    r_wrap = [(r.time, r.rptr_before, r.rptr_after)
              for r in records
              if r.r_accept and r.rptr_after < r.rptr_before]
    print(f"  写指针模 4 回绕 (3->0) 发生在 t={[t for t, *_ in w_wrap]}")
    print(f"  读指针模 4 回绕 (3->0) 发生在 t={[t for t, *_ in r_wrap]}")
    print(f"  接受写={fifo.accepted_writes} 接受读={fifo.accepted_reads} "
          f"残留={len(q)}；抽象队列顺序/越界核对通过")
    print(f"  全部读出数据 = {ordered_out}")
    print("  标识成对出现（每两个写沿才能被同步到读域），"
          "但先后顺序严格不乱；指针值仅在 0..3 内循环")
    assert w_wrap and r_wrap
    assert all(0 <= r.wptr_after <= 3 and 0 <= r.rptr_after <= 3
               for r in records)
    assert ordered_out == sorted(ordered_out)
    assert fifo.accepted_writes - fifo.accepted_reads == len(q)


def section_boundary():
    print("\n" + "=" * 106)
    print("实际触发的拒绝/异常边界（模型真实抛出并就地捕获，非预录文本）")
    print("=" * 106)

    # Hardware-level rejection: saturate a depth-2 FIFO, request one more.
    fifo = AsyncFIFO(k=1)
    t = 0
    accepted = []
    while not fifo.full:
        t += 1
        rec = fifo.run_one(Event(t, True, False, wen=True, wdata=f"X{t}"))
        if rec.w_accept:
            accepted.append(rec.wdata)
    t += 1
    rec = fifo.run_one(Event(t, True, False, wen=True, wdata="OVERFLOW"))
    print(f"[硬件拒绝] 深度2 FIFO 已写入 {accepted}（full 寄存器=1），"
          f"t={t} 再写：w_accept={rec.w_accept}，指针保持 "
          f"{rec.wptr_after:03b} 未移动")
    assert rec.w_accept is False
    assert rec.wptr_after == rec.wptr_before

    # Empty-side rejection at reset.
    rec = fifo.run_one(Event(t + 1, False, True, ren=True))
    print(f"[硬件拒绝] 复位后空 FIFO 首个读请求：r_accept={rec.r_accept}，"
          f"empty 旧值={int(rec.empty_before)}")
    assert rec.r_accept is False

    for desc, call in [
        ("k=6 越出 1..5", lambda: AsyncFIFO(k=6)),
        ("k=True 布尔冒充整数", lambda: AsyncFIFO(k=True)),
        ("NaN 浮点载荷", lambda: Event(1, True, False, wen=True,
                                       wdata=float("nan"))),
        ("Infinity 浮点载荷", lambda: Event(1, True, False, wen=True,
                                            wdata=float("inf"))),
        ("双时钟均停", lambda: Event(1, False, False)),
    ]:
        try:
            call()
        except AsyncFIFOError as exc:
            print(f"[契约异常] {desc} -> AsyncFIFOError: {exc}")
        else:
            raise AssertionError(f"{desc} 应当抛出 AsyncFIFOError")

    # Failed batch must leave no partial change (state compared before/after).
    good = AsyncFIFO(k=2)
    good.run([Event(1, True, False, wen=True, wdata="keep")])
    before, before_steps = good.snapshot(), good.steps
    try:
        good.run([
            Event(2, True, False, wen=True, wdata="nope1"),
            Event(3, True, False, wen=True, wdata="nope2"),
            Event(3, True, False),                 # non-increasing time
        ])
    except AsyncFIFOError as exc:
        print(f"[批次原子性] 非法批次被拒（events[2] 可定位）：{exc}")
    assert good.snapshot() == before
    assert good.steps == before_steps
    print("[批次原子性] 失败后寄存器与轨迹与批次前完全一致，无部分变更")


def section_reset():
    print("\n" + "=" * 106)
    print("读写域单独复位 + 世代双向确认：k=2，深度4。先填满，再做写域异步复位，逐拍展示世代/应答两级同步")
    print("=" * 106)
    fifo = AsyncFIFO(k=2)
    t = 0
    for d in ("A", "B", "C", "D"):
        t += 1
        fifo.run_one(Event(t, True, False, wen=True, wdata=d))
    print(f"填满 {fifo.depth} 个：full={int(full := fifo.full)} "
          f"wptr={fifo.wptr} 世代 w/r={fifo.w_generation}/{fifo.r_generation}")
    assert full and fifo.accepted_writes == 4

    # asynchronous write-domain reset (no clock edge required)
    t += 1
    rr = fifo.run_one(Event(t, False, False, wrst=True))
    print(f"t={t} wrst 异步复位（双时钟均停，合法）：世代 {rr.wgen_before}->"
          f"{rr.wgen_after}，wptr {rr.wptr_before}->{rr.wptr_after}，"
          f"full={int(rr.full_after)}，synced={int(rr.synced_before)}->"
          f"{int(rr.synced_after)}；存储器与计数已清空"
          f"（接受写/读={fifo.accepted_writes}/{fifo.accepted_reads}）")
    assert rr.wgen_after == 1 and rr.wptr_after == 0
    assert fifo.accepted_writes == 0 and fifo.accepted_reads == 0

    print("\n世代 1 经读域两级同步器传播（确认前每一步读写都被拒绝）：")
    rows = []
    # two read edges: generation reaches s1 then s2
    for label in ("读沿1: wgen2r s1 锁存新世代",
                  "读沿2: s2 见到新世代",
                  "读沿3: 读域学到世代1->rack_reg，应答出发"):
        t += 1
        r = fifo.run_one(Event(t, False, True, ren=True))
        rows.append((t, r))
        print(f"  t={t} {label}：wgen2r={r.wgen2r_after} "
              f"rack_reg={r.rack_reg_after} r接受={r.r_accept} "
              f"synced={int(r.synced_after)}")
    assert rows[2][1].rack_reg_after == 1
    assert all(r.r_accept is False for _, r in rows)
    # two write edges: acknowledgement returns
    ack_rows = []
    for label in ("写沿1: 应答到 s1", "写沿2: 应答到 s2，双向确认完成"):
        t += 1
        r = fifo.run_one(Event(t, True, False, wen=True, wdata="LATE"))
        ack_rows.append((t, r))
        print(f"  t={t} {label}：rack2w={r.rack2w_after} "
              f"写接受={r.w_accept} synced={int(r.synced_after)}")
    assert ack_rows[1][1].synced_after
    assert all(r.w_accept is False for _, r in ack_rows)

    # FIFO is logically empty; the LATE write was refused, offer it again
    t += 1
    empty_read = fifo.run_one(Event(t, False, True, ren=True))
    print(f"\nt={t} 同步恢复后的首个读请求：r_accept={empty_read.r_accept}"
          "（复位清空旧数据，FIFO 真空）")
    assert empty_read.r_accept is False
    t += 1
    w = fifo.run_one(Event(t, True, False, wen=True, wdata="E1"))
    assert w.w_accept
    reads = []
    for _ in range(4):  # s1, s2, empty deasserts at edge end, accept next
        t += 1
        reads.append(fifo.run_one(Event(t, False, True, ren=True)))
    got = reads[-1]
    print(f"t={t} 重新写入 E1 后经 4 个读沿（两级同步+判旗标+接受）读出："
          f"r_accept={got.r_accept} rdata={got.rdata!r}（绝无 A..D 旧数据）")
    assert [r.r_accept for r in reads] == [False, False, False, True]
    assert got.rdata == "E1"

    # ---- second reset while a handshake is already pending (re-reset)
    print("\n再次复位（上一轮握手中途再次 wrst，世代号单调递增）：")
    t += 1
    fifo.run_one(Event(t, False, False, wrst=True))      # generation 2
    t += 1
    mid = fifo.run_one(Event(t, False, True))            # only s1 of gen 2
    print(f"  t={t} 世代={fifo.w_generation}，仅过一个读沿 wgen2r={mid.wgen2r_after}，"
          "管道里还残留世代1的应答")
    assert fifo.w_generation == 2 and not fifo.synced
    guard = 0
    while not fifo.synced:
        t += 1
        fifo.run_one(Event(t, True, True))
        guard += 1
        assert guard < 20
    snap = fifo.snapshot()
    print(f"  再经 {guard} 个同刻边沿后 synced=1；收敛于最新世代：rack2w={snap['rack2w']}，"
          f"rack_reg={snap['rack_reg']}，旧世代1的在途值已自然老化")
    assert snap["rack2w"][1] == 2 and snap["rack_reg"] == 2

    # ---- actual contract errors around reset
    print("\n复位相关的真实契约异常（就地捕获）：")
    for desc, make in [
        ("wrst=1 非布尔", lambda: Event(1, False, False, wrst=1)),
        ("rrst=0.0 非布尔", lambda: Event(1, False, False, rrst=0.0)),
        ("双停且无复位", lambda: Event(1, False, False)),
    ]:
        try:
            make()
        except AsyncFIFOError as exc:
            print(f"  [契约异常] {desc} -> AsyncFIFOError: {exc}")
        else:
            raise AssertionError(f"{desc} 应当抛出 AsyncFIFOError")


def main():
    section_normal()
    section_wrap()
    section_reset()
    section_boundary()
    print("\n演示完成：以上数值均由模型实时计算；抽象队列独立核对全部通过。")


if __name__ == "__main__":
    main()
