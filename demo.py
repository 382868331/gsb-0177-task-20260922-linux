#!/usr/bin/env python3
"""固定输入演示：异步 FIFO 指针跨域模型（深度 8, k=3）。

运行：``python demo.py``（仅需 Python 3.14.7 标准库，无网络、无第三方依赖）。

演示内容（全部现场真实计算，无 sleep、无预录结果）：
1. 一段固定的写/读时钟事件序列（写快读慢，含同刻边沿、停钟、指针回绕段），
   打印接受/拒绝统计、同步轨迹节选与独立抽象队列核对结论；
2. 实际触发的两个硬件拒绝边界：保守满（写拒绝）与保守空（读拒绝），
   并展示 full/empty 旗标相对真实占用的跨时钟域延迟；
3. 一个实际抛出的非法输入异常（NaN 时刻），错误可定位、模型状态不变。
"""

from __future__ import annotations

import math
import time as _time

from fifo_cdc import (
    AsyncFIFOModel,
    ClockKind,
    FIFOInputError,
    TraceOracle,
    nop_edge,
    read_edge,
    write_edge,
)

K = 3
DEPTH = 1 << K


def build_scenario():
    """确定性固定事件：写时钟 1 tick/拍；读时钟较稀疏，中段停钟后恢复。"""
    edges = []

    # --- 阶段 1：写快读慢，把 FIFO 推向保守满（t=0..15）---
    t = 0
    for i in range(16):
        edges.append(write_edge(t, f"W{i:02d}"))
        if t % 4 == 0:                 # 读钟每 4 拍才来一次
            edges.append(read_edge(t))  # 同刻边沿
        t += 1

    # --- 阶段 2：读钟停 20 拍，写侧继续尝试（持续被判满拒绝）---
    stop_end = t + 20
    while t < stop_end:
        edges.append(write_edge(t, f"STOP{t}"))
        t += 1

    # --- 阶段 3：读钟恢复，连续读排空，并继续少量写（覆盖指针回绕）---
    for i in range(40):
        edges.append(read_edge(t))
        if i % 3 == 0:
            edges.append(write_edge(t, f"R{i:02d}"))  # 同刻
        t += 1

    # --- 阶段 4：写钟停，读钟空转，稳定到保守空 ---
    for _ in range(8):
        edges.append(nop_edge(ClockKind.READ, t))
        t += 1
    edges.append(read_edge(t))  # 必然被空拒绝的边界读
    return edges


def main() -> None:
    started = _time.perf_counter()

    print("=" * 72)
    print(f"异步 FIFO 指针跨域检查演示  k={K}  深度={DEPTH}  指针位宽={K+1}")
    print("=" * 72)

    model = AsyncFIFOModel(K)
    oracle = TraceOracle(K)
    edges = build_scenario()

    # 现场逐条计算，并用独立参考（另一套协议实现 + 抽象队列）同步核对。
    records = []
    for edge in edges:
        rec = model.step(edge)
        records.append((edge, rec))
        oracle.observe(edge, rec)
    report = oracle.report()

    n_w_acc = report.accepted_writes
    n_r_acc = report.accepted_reads
    print(f"事件总数            : {len(edges)}")
    print(f"写: 接受 {n_w_acc:>3}  拒绝 {report.rejected_writes:>3}")
    print(f"读: 接受 {n_r_acc:>3}  拒绝 {report.rejected_reads:>3}")
    print(f"抽象队列峰值占用    : {report.max_occupancy}（不得超过深度 {DEPTH}）")
    print(f"抽象队列残留        : {n_w_acc - n_r_acc}")
    order_ok = report.read_sequence == report.written_sequence[:n_r_acc]
    print(f"读出顺序==写入顺序前缀: {order_ok}")
    print(f"独立参考全字段核对  : {'通过' if report.ok else '失败'}")
    if not report.ok:
        for p in report.problems[:10]:
            print("  !", p)
        raise SystemExit("独立核对失败，演示中止")

    # ---- 轨迹节选：打印前 6 条边沿的两级同步链与旗标 ----
    print("-" * 72)
    print("同步轨迹节选（前 6 条边沿；wq2=写侧所见读格雷, rq2=读侧所见写格雷）:")
    print("  t  域   操作    接受  wptr/rptr(格雷)  wq1 wq2  rq1 rq2  full empty")
    for edge, rec in records[:6]:
        dom = "W" if edge.clock is ClockKind.WRITE else "R"
        op = "写" if rec.write_attempt else ("读" if rec.read_attempt else "空转")
        acc = ("是" if (rec.write_accepted or rec.read_accepted)
               else ("否" if (rec.write_attempt or rec.read_attempt) else "-"))
        print(
            f" {rec.time:>3} {dom}  {op:<4}  {acc:<3} "
            f"{rec.wptr_bin:>2}/{rec.rptr_bin:<2} "
            f"({rec.wptr_gray:03b}/{rec.rptr_gray:03b})  "
            f"{rec.wq1_rptr:03b} {rec.wq2_rptr:03b}  "
            f"{rec.rq1_wptr:03b} {rec.rq2_wptr:03b}  "
            f"{int(rec.full)}    {int(rec.empty)}"
        )

    # ---- 实际触发的保守满/空拒绝边界（从真实轨迹中提取，非预录）----
    print("-" * 72)
    first_full_rej = next(
        (r for e, r in records if e.clock is ClockKind.WRITE
         and e.write and not r.write_accepted),
        None,
    )
    # 取 t>0 的保守空拒绝：FIFO 中已有实物，但读侧 empty 尚未随同步链撤销。
    first_empty_rej = next(
        (r for e, r in records if e.clock is ClockKind.READ
         and e.read and not r.read_accepted and e.time > 0),
        None,
    )
    assert first_full_rej is not None and first_empty_rej is not None
    print(f"实际触发的保守满写拒绝: t={first_full_rej.time}, "
          f"拒绝前沿用旧 full={int(first_full_rej.full_before)}, "
          f"拒绝后 full={int(first_full_rej.full)}")
    print(f"实际触发的保守空读拒绝: t={first_empty_rej.time}, "
          f"拒绝前沿用旧 empty={int(first_empty_rej.empty_before)}, "
          f"拒绝后 empty={int(first_empty_rej.empty)}")
    print("说明: 两级同步使旗标相对另一域真实占用至少滞后 2 个本域时钟拍，")
    print("      full/empty 只会保守（可能提前判满/判空），绝不发生上溢/下漏。")
    print(f"读出数据前 5 项: {report.read_sequence[:5]}")
    print(f"回绕检查: 累计接受写 {n_w_acc} 次 (>={2*DEPTH} 即指针至少完整回绕一次)，"
          f"最终 wptr={model.wptr_bin}, rptr={model.rptr_bin}")

    # ---- 实际触发的非法输入异常边界 ----
    print("-" * 72)
    before = model.snapshot()
    try:
        model.step(write_edge(math.nan, "should-not-exist"))
    except FIFOInputError as exc:
        print(f"非法输入实际抛出 {type(exc).__name__}: {exc}")
    after = model.snapshot()
    print(f"异常后模型状态不变: {before == after}")

    # bool 整数参数同样拒绝（再展示一个可定位的构造期错误）
    try:
        AsyncFIFOModel(True)
    except FIFOInputError as exc:
        print(f"构造期非法输入 {type(exc).__name__}: {exc}")

    elapsed = _time.perf_counter() - started
    print("-" * 72)
    print(f"演示完成，实际耗时 {elapsed:.3f} 秒（确定性计算，无 sleep/网络/第三方库）。")


if __name__ == "__main__":
    main()
