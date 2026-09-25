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


def section_almost():
    print("\n" + "=" * 106)
    print("迟滞 almost_full/almost_empty：k=2，深度 4；阈值 快满 low/high=1/3，快空 low/high=0/2；")
    print("写时钟 t%2==0、读时钟 t%3==0（不等频，t=6,12,... 同刻）。估计值=本地接受后指针与旧二级")
    print("同步对端指针的模 2^(k+1) 距离，只由指针与同步寄存器推出，不看真实队列长度")
    print("=" * 106)
    fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3,
                     almost_empty_low=0, almost_empty_high=2)
    print(f"复位态: almost_full={fifo.almost_full} "
          f"almost_empty={fifo.almost_empty}（快满复位为假、快空复位为真）\n")
    # Fixed literal schedule: (t, wclk, rclk, wen, ren, wdata)
    schedule = [
        (2, 1, 0, 1, 0, "A"), (3, 0, 1, 0, 1, None), (4, 1, 0, 1, 0, "B"),
        (6, 1, 1, 1, 1, "C"), (8, 1, 0, 1, 0, "D"), (9, 0, 1, 0, 1, None),
        (10, 1, 0, 1, 0, "E"), (12, 1, 1, 1, 1, "F"), (14, 1, 0, 1, 0, "G"),
        (15, 0, 1, 0, 1, None), (16, 1, 0, 1, 0, "H"), (18, 1, 1, 1, 1, "I"),
        (20, 1, 0, 1, 0, "J"), (21, 0, 1, 0, 1, None), (22, 1, 0, 1, 0, "K"),
        (24, 1, 1, 1, 1, "L"), (26, 1, 0, 1, 0, "M"), (27, 0, 1, 0, 1, None),
        (28, 1, 0, 1, 0, "N"), (30, 1, 1, 1, 1, "O"), (32, 1, 0, 0, 0, None),
        (33, 0, 1, 0, 1, None), (34, 1, 0, 0, 0, None), (36, 1, 1, 0, 1, None),
        (38, 1, 0, 0, 0, None), (39, 0, 1, 0, 1, None), (40, 1, 0, 0, 0, None),
        (42, 1, 1, 0, 1, None),
    ]
    events = [Event(t, bool(w), bool(r), bool(we), bool(re_), d)
              for (t, w, r, we, re_, d) in schedule]
    records = fifo.run(events)

    print("t    边  en   接受  | 写域估计 读域估计 | AF AE (F E) | 备注")
    transitions = []
    for rec in records:
        edge = ("W" if rec.wclk else ".") + ("R" if rec.rclk else ".")
        wa = {True: "A", False: "x", None: "-"}[rec.w_accept]
        ra = {True: "A", False: "x", None: "-"}[rec.r_accept]
        notes = []
        if rec.almost_full_after != rec.almost_full_before:
            notes.append("AF " + ("置位" if rec.almost_full_after else "清除"))
        if rec.almost_empty_after != rec.almost_empty_before:
            notes.append("AE " + ("置位" if rec.almost_empty_after else "清除"))
        transitions.extend((rec.time, n) for n in notes)
        note = " ".join(notes)
        w_est = "-" if rec.w_used_est is None else str(rec.w_used_est)
        r_est = "-" if rec.r_avail_est is None else str(rec.r_avail_est)
        print(f"t={rec.time:>2} {edge} en={int(rec.wen)}{int(rec.ren)} "
              f"W:{wa} R:{ra} | w_used={w_est:>2} r_avail={r_est:>2} "
              f"| {int(rec.almost_full_after)}  {int(rec.almost_empty_after)}"
              f"  ({int(rec.full_after)} {int(rec.empty_after)}) | {note}")

    print("\n标志跳变（迟滞：进入阈值才动作，带内保持不抖动）：")
    for t, what in transitions:
        print(f"  t={t}: {what}")
    # anchors computed by the model above, cross-checked against the
    # hand-derived table used in tests/test_afifo.py
    at = {r.time: r for r in records}
    assert at[6].almost_full_after and at[6].w_used_est == 3      # est>=high
    assert at[38].almost_full_after and at[38].w_used_est == 2    # 带内保持
    assert at[40].almost_full_after and at[40].w_used_est == 2    # 带内保持
    assert not at[42].almost_full_after and at[42].w_used_est == 1
    assert not at[15].almost_empty_after and at[15].r_avail_est == 2
    assert at[28].wptr_after == 0 and at[28].wptr_before == 7     # 写指针回绕
    assert at[39].rptr_after == 0 and at[39].rptr_before == 7     # 读指针回绕
    print("  （t=38/40 估计值=2 落在 (1,3) 迟滞带内，AF 保持置位不抖动；"
          "t=42 估计值降到 1 才清除）")

    # 估计值是保守的：写域估计 >= 真实占用，读域估计 <= 真实可读
    occ = 0
    for rec in records:
        occ += (1 if rec.w_accept else 0) - (1 if rec.r_accept else 0)
        if rec.w_used_est is not None:
            assert rec.w_used_est >= occ
        if rec.r_avail_est is not None:
            assert rec.r_avail_est <= occ
    print("  全程核对：写域估计 >= 真实占用、读域估计 <= 真实可读（保守方向正确）")

    # 新标志只是提示：接受/拒绝仍由 full/empty 决定
    fifo2 = AsyncFIFO(k=2, almost_full_low=0, almost_full_high=1)
    recs = fifo2.run([Event(t, True, False, wen=True, wdata=t)
                      for t in range(1, 4)])
    assert all(r.w_accept for r in recs) and recs[0].almost_full_after
    print("  提示性核对：AF 已置位时写仍被接受（接受与否只看硬 full/empty）")


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
        ("快满阈值相等 low==high", lambda: AsyncFIFO(
            k=2, almost_full_low=2, almost_full_high=2)),
        ("快空阈值越界 high>depth", lambda: AsyncFIFO(
            k=2, almost_empty_low=0, almost_empty_high=5)),
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


def main():
    section_normal()
    section_wrap()
    section_almost()
    section_boundary()
    print("\n演示完成：以上数值均由模型实时计算；抽象队列独立核对全部通过。")


if __name__ == "__main__":
    main()
