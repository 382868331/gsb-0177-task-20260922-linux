#!/usr/bin/env python3
"""Fixed-input demo for the afifo asynchronous-FIFO pointer checker.

Run:  python demo.py

Everything here is a fixed, deterministic schedule (no randomness, no sleep,
no pre-recorded output): the model is actually stepped, its trace printed,
and accepted payload order / occupancy bounds are cross-checked with an
independent abstract ``collections.deque``.
"""

from collections import deque

from afifo import AsyncFIFO, Event, ReadyValidFIFO, RVEvent
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


# Fixed ready/valid schedule for k=1 (depth 2).  Literal values, no randomness.
# (time, wclk, rclk, wen, ready, wdata)
RV_SCHEDULE = [
    (1, 1, 0, 1, 0, "a"),     # 写 a
    (2, 1, 0, 1, 0, "b"),     # 写 b，本沿末 full 置位
    (3, 1, 0, 1, 0, "c"),     # full=1：c 被拒绝
    (4, 0, 1, 0, 0, None),    # 数据已写入但未跨域：预取请求被旧 empty 拒绝
    (5, 0, 1, 0, 0, None),    # 同上（同步器第2级）
    (6, 0, 1, 0, 0, None),    # 同上，本拍末 empty 撤销
    (7, 0, 1, 0, 0, None),    # 预取 a 装入输出寄存器，valid 置位
    (8, 0, 1, 0, 0, None),    # ready=0 背压：输出稳定，读指针不动
    (9, 0, 1, 0, 0, None),    # 持续背压
    (10, 0, 1, 0, 1, None),   # ready=1：同拍交付 a 并装入 b，valid 保持
    (11, 0, 1, 0, 1, None),   # 交付 b，内层已空无可装，valid 落下
    (12, 1, 0, 1, 0, "d"),    # 读指针尚未同步回写域：d 仍被旧 full 拒绝
    (13, 1, 0, 1, 0, "d"),    # 同上
    (14, 1, 0, 1, 0, "d"),    # 拒绝，但本拍末 full 撤销
    (15, 1, 0, 1, 0, "d"),    # 写 d 接受
    (16, 1, 0, 1, 0, "e"),    # 写 e 接受，写指针 3->0 回绕
    (17, 0, 1, 0, 1, None),   # 新写未跨域：不能凭真实队列长度预取
    (18, 0, 1, 0, 1, None),
    (19, 0, 1, 0, 1, None),
    (20, 0, 1, 0, 1, None),   # 预取 d 装入输出寄存器
    (21, 0, 1, 0, 1, None),   # 同拍交付 d 并装入 e；读指针 3->0 回绕
    (22, 0, 1, 0, 1, None),   # 交付 e，valid 落下
]


def section_ready_valid():
    print("\n" + "=" * 106)
    print("ready/valid 输出寄存器：k=1，内层深度 2（depth 参数不含输出寄存器），固定字面值调度")
    print("=" * 106)
    fifo = ReadyValidFIFO(k=1)
    print(f"复位态: valid={int(fifo.valid)} empty={int(fifo.empty)} "
          f"full={int(fifo.full)} depth={fifo.depth}\n")
    records = fifo.run(RVEvent(t, bool(w), bool(r), bool(we), bool(rd), d)
                       for (t, w, r, we, rd, d) in RV_SCHEDULE)

    print("t    边  en=wen/rdy ren | 内层W/R | load deliver | valid out "
          "| wptr rptr F E")
    for rec in records:
        edge = ("W" if rec.wclk else ".") + ("R" if rec.rclk else ".")
        ren = {True: "1", False: "0", None: "-"}[rec.ren]
        wa = {True: "A", False: "x", None: "-"}[rec.w_accept]
        ra = {True: "A", False: "x", None: "-"}[rec.r_accept]
        load = {True: "1", False: "0", None: "-"}[rec.load]
        dlv = {True: "1", False: "0", None: "-"}[rec.delivered]
        out = rec.out_after if rec.valid_after else ""
        print(f"t={rec.time:>2} {edge}  en={int(rec.wen)}{int(rec.ready)}"
              f"      ren={ren} | W:{wa} R:{ra}  | load={load} dlv={dlv}"
              f"  | v={int(rec.valid_after)} out={out!s:<4}"
              f"| wptr={rec.inner.wptr_after:02b} rptr={rec.inner.rptr_after:02b}"
              f" F={int(rec.inner.full_after)} E={int(rec.inner.empty_after)}")

    # 独立核对：被接受写入序列 vs 下游交付序列（两条序列分开维护）
    written = [r.wdata for r in records if r.w_accept]
    delivered_seq = [r.delivered_data for r in records if r.delivered]
    print("\n独立序列核对（接受写入 / 下游交付分开维护）：")
    print(f"  接受写入 = {written}")
    print(f"  下游交付 = {delivered_seq}")
    assert written == ["a", "b", "d", "e"]            # c 被 full 拒绝
    assert delivered_seq == written                    # 交付是写入的同序前缀
    assert fifo.accepted_writes == 4
    assert fifo.accepted_reads == 4                    # 内层出队=装入次数
    assert fifo.delivered == 4                         # 只计 旧valid&ready

    # 背压稳定性锚点：t=8/9 输出字与读指针冻结
    for rec in records:
        if rec.time in (8, 9):
            assert rec.ren is False and rec.out_after == "a"
            assert rec.inner.rptr_after == 1 and rec.valid_after
    print("  t=8..9 背压：ren=0，out='a' 与 rptr 均冻结，valid 保持")

    # 同拍消费+装入锚点
    rec10 = records[9]
    assert rec10.delivered and rec10.delivered_data == "a"
    assert rec10.load and rec10.out_after == "b" and rec10.valid_after
    print("  t=10 同拍：交付旧值 'a' 并装入新值 'b'，valid 保持为 1")

    # 容量口径：预取使“内层占用”与“下游在途”出现差异
    print("\n容量口径（预取导致的统计差异，输出寄存器不占 depth）：")
    print(f"  depth 参数 = {fifo.depth}（仅内层）；在途上限 = depth+1 = "
          f"{fifo.depth + 1}")
    print("  恒等式：在途 = 接受写 - 交付 = 内层占用(接受写-内层出队) + valid")
    print(f"  当前：{fifo.inflight} = {fifo.accepted_writes}-"
          f"{fifo.delivered} = "
          f"({fifo.accepted_writes}-{fifo.accepted_reads}) + "
          f"{int(fifo.valid)}")

    # 输出寄存器满（valid=1）时复位
    fifo.reset()
    fifo.run_one(RVEvent(1, True, False, wen=True, wdata="z"))
    for t in (2, 3, 4, 5):
        fifo.run_one(RVEvent(t, False, True, ready=False))
    assert fifo.valid and fifo.out_data == "z"
    fifo.reset()
    assert not fifo.valid and fifo.out_data is None and fifo.empty
    print("\n复位：输出寄存器满（valid=1, out='z'）时复位 -> "
          f"valid={int(fifo.valid)}，内层 empty={int(fifo.empty)}，"
          "计数与轨迹清零")


def main():
    section_normal()
    section_wrap()
    section_ready_valid()
    section_boundary()
    print("\n演示完成：以上数值均由模型实时计算；抽象队列独立核对全部通过。")


if __name__ == "__main__":
    main()
