"""独立参考核对器（oracle）。

本模块**不导入、不调用**被测核心 ``fifo_cdc.core`` 中的任何算法，仅复用其
枚举/数据类作为喂数载体。期望值由另一套独立写法重新推演：

1. 按题面协议逐位重算每条边沿后的二进制指针、两级同步寄存器与 full/empty
   旗标，逐字段与被测模型的轨迹记录对照；
2. 另外维护一个与硬件旗标完全无关的抽象 FIFO 队列（deque），核对：
   - 每次被接受的读都不发生下溢（读到空槽）；
   - 每次被接受的写都不发生上溢（占用不超过深度）；
   - 读出数据的顺序严格等于写入接受顺序；
3. 用累计接受计数核对保守旗标下真实占用始终在 ``[0, depth]`` 内。

同刻的写/读两边沿共享"该时刻开始时"的同一组旧寄存器，各自只更新本域；
抽象队列在同刻按"读先取旧数据、写再落新数据"提交，与喂入顺序无关。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

from .core import ClockEdge, ClockKind, EdgeRecord, FIFOError  # 仅载体，不含算法

__all__ = ["OracleReport", "TraceOracle"]


def _gray(n: int, bits: int) -> int:
    """本地独立格雷码：逐位拼 g[i]=b[i]^b[i+1]，刻意不与 core 同写法。"""
    mask = (1 << bits) - 1
    n &= mask
    g = 0
    for i in range(bits):
        bi = (n >> i) & 1
        bi1 = (n >> (i + 1)) & 1 if i + 1 < bits else 0
        g |= ((bi ^ bi1) & 1) << i
    return g


@dataclass(frozen=True)
class OracleReport:
    """核对结论。``ok`` 为真表示全部通过；否则 ``problems`` 给出可定位问题。"""

    ok: bool
    problems: tuple[str, ...]
    accepted_writes: int
    accepted_reads: int
    rejected_writes: int
    rejected_reads: int
    max_occupancy: int
    written_sequence: tuple[object, ...]
    read_sequence: tuple[object, ...]

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise FIFOError("独立核对失败:\n  - " + "\n  - ".join(self.problems))


class _Regs:
    """某一时刻开始时（或边沿后）的寄存器组。"""

    __slots__ = ("wptr", "rptr", "wq1", "wq2", "rq1", "rq2", "full", "empty")

    def __init__(self) -> None:
        self.wptr = 0
        self.rptr = 0
        self.wq1 = 0
        self.wq2 = 0
        self.rq1 = 0
        self.rq2 = 0
        self.full = False
        self.empty = True

    def copy(self) -> "_Regs":
        r = _Regs()
        r.wptr, r.rptr = self.wptr, self.rptr
        r.wq1, r.wq2 = self.wq1, self.wq2
        r.rq1, r.rq2 = self.rq1, self.rq2
        r.full, r.empty = self.full, self.empty
        return r


class TraceOracle:
    """对某一深度的 FIFO 轨迹做独立核对。

    用法::

        oracle = TraceOracle(k)
        rec = model.step(edge)      # 被测模型自己跑
        oracle.observe(edge, rec)   # 喂入同一边沿与模型记录
        oracle.report().raise_if_bad()
    """

    def __init__(self, k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int):
            raise FIFOError("TraceOracle 的 k 必须是 int 且非 bool")
        if not 1 <= k <= 5:
            raise FIFOError("TraceOracle 的 k 必须在 1..5")
        self._k = k
        self._bits = k + 1
        self._depth = 1 << k
        self._mod = 1 << self._bits
        self._top2 = 0b11 << (k - 1)

        self._committed = _Regs()       # 已提交到上一边沿后的寄存器
        self._base: Optional[_Regs] = None  # 当前时刻开始时的旧寄存器
        self._cur_time: Optional[int] = None
        self._seen_clocks: set[ClockKind] = set()

        # 抽象队列与计数（与硬件旗标完全无关）
        self._queue: deque[object] = deque()
        self._written: list[object] = []
        self._read_out: list[object] = []
        self._w_acc = self._r_acc = 0
        self._w_rej = self._r_rej = 0
        self._max_occ = 0
        self._problems: list[str] = []
        self._edge_index = 0

    # ------------------------------------------------------------------ #
    def observe(self, edge: ClockEdge, record: EdgeRecord) -> None:
        """观察一条边沿及其在被测模型上的记录，立即做协议字段与队列性质核对。"""
        idx = self._edge_index
        self._edge_index += 1
        where = f"边沿序号 {idx} t={edge.time} ({edge.clock.value})"

        t = edge.time
        if self._cur_time is not None and t < self._cur_time:
            self._problems.append(f"{where}: 时间倒流")
            return
        if t != self._cur_time:
            # 新时刻：以上一时刻提交结果为基准，抽象队列也已是上一时刻结果。
            self._base = self._committed.copy()
            self._cur_time = t
            self._seen_clocks = set()
        if edge.clock in self._seen_clocks:
            self._problems.append(f"{where}: 同一域同一时刻重复边沿")
            return
        self._seen_clocks.add(edge.clock)
        assert self._base is not None
        old = self._base

        exp_wptr, exp_rptr = old.wptr, old.rptr
        exp_wq1, exp_wq2 = old.wq1, old.wq2
        exp_rq1, exp_rq2 = old.rq1, old.rq2
        exp_full, exp_empty = old.full, old.empty
        w_accepted = r_accepted = False

        if edge.clock is ClockKind.WRITE:
            attempt = bool(edge.write)
            w_accepted = attempt and not old.full
            exp_wptr = (old.wptr + (1 if w_accepted else 0)) % self._mod
            wg = _gray(exp_wptr, self._bits)
            # full：下一写格雷 == 同步读格雷仅最高两位取反后的值（全位比较）。
            exp_full = wg == (old.wq2 ^ self._top2)
            exp_wq1 = _gray(old.rptr, self._bits)
            exp_wq2 = old.wq1
        else:
            attempt = bool(edge.read)
            r_accepted = attempt and not old.empty
            exp_rptr = (old.rptr + (1 if r_accepted else 0)) % self._mod
            rg = _gray(exp_rptr, self._bits)
            exp_empty = rg == old.rq2
            exp_rq1 = _gray(old.wptr, self._bits)
            exp_rq2 = old.rq1

        # ---- 逐字段对照被测记录 ----
        expected_fields = {
            "write_attempt": edge.clock is ClockKind.WRITE and bool(edge.write),
            "write_accepted": w_accepted if edge.clock is ClockKind.WRITE else False,
            "read_attempt": edge.clock is ClockKind.READ and bool(edge.read),
            "read_accepted": r_accepted if edge.clock is ClockKind.READ else False,
            "wptr_bin": exp_wptr,
            "rptr_bin": exp_rptr,
            "wptr_gray": _gray(exp_wptr, self._bits),
            "rptr_gray": _gray(exp_rptr, self._bits),
            "wq1_rptr": exp_wq1,
            "wq2_rptr": exp_wq2,
            "rq1_wptr": exp_rq1,
            "rq2_wptr": exp_rq2,
            "full": exp_full,
            "empty": exp_empty,
            "full_before": old.full,
            "empty_before": old.empty,
        }
        for name, expected in expected_fields.items():
            actual = getattr(record, name)
            if actual != expected:
                self._problems.append(
                    f"{where}: 字段 {name} 模型={actual!r} 独立期望={expected!r}"
                )

        # ---- 抽象队列：先读旧数据、再写新数据（与喂入顺序无关地分两步核对）----
        if edge.clock is ClockKind.READ:
            if r_accepted:
                self._r_acc += 1
                # 同刻写即使先喂入也只能 append 到队尾；若本边沿前队列空，
                # 正确协议下读不可能被接受（empty 复位即为 True）。
                if not self._queue:
                    self._problems.append(f"{where}: 接受了读但抽象队列为空（下溢）")
                else:
                    want = self._queue.popleft()
                    if record.read_data != want:
                        self._problems.append(
                            f"{where}: 读出 {record.read_data!r} 与队首期望 "
                            f"{want!r} 不符（顺序破坏）"
                        )
                        # 仍记录实际读出，供最终序列检查
                    self._read_out.append(record.read_data)
            elif edge.read:
                self._r_rej += 1
        else:
            if w_accepted:
                self._w_acc += 1
                # 占用只统计该时刻之前已接受写减已接受读；同刻读若先喂入已 pop。
                if len(self._queue) >= self._depth:
                    self._problems.append(
                        f"{where}: 接受了写但占用已达深度 {self._depth}（上溢）"
                    )
                self._queue.append(edge.data)
                self._written.append(edge.data)
                self._max_occ = max(self._max_occ, len(self._queue))
            elif edge.write:
                self._w_rej += 1

        # ---- 用累计计数核对真实占用界 ----
        occ = self._w_acc - self._r_acc
        if not 0 <= occ <= self._depth:
            self._problems.append(
                f"{where}: 真实占用 {occ} 越出 [0, {self._depth}]"
            )
        if not 0 <= len(self._queue) <= self._depth:
            self._problems.append(
                f"{where}: 抽象队列长度 {len(self._queue)} 越界"
            )

        # ---- 提交本域寄存器（另一域保持基准值不变）----
        if edge.clock is ClockKind.WRITE:
            self._committed.wptr = exp_wptr
            self._committed.wq1, self._committed.wq2 = exp_wq1, exp_wq2
            self._committed.full = exp_full
        else:
            self._committed.rptr = exp_rptr
            self._committed.rq1, self._committed.rq2 = exp_rq1, exp_rq2
            self._committed.empty = exp_empty

    def observe_many(self, pairs: Iterable[tuple[ClockEdge, EdgeRecord]]) -> OracleReport:
        for edge, record in pairs:
            self.observe(edge, record)
        return self.report()

    def report(self) -> OracleReport:
        if self._w_acc - self._r_acc != len(self._queue):
            self._problems.append(
                f"最终计数不自洽: 接受写 {self._w_acc} - 接受读 {self._r_acc} "
                f"!= 队列残留 {len(self._queue)}"
            )
        return OracleReport(
            ok=not self._problems,
            problems=tuple(self._problems),
            accepted_writes=self._w_acc,
            accepted_reads=self._r_acc,
            rejected_writes=self._w_rej,
            rejected_reads=self._r_rej,
            max_occupancy=self._max_occ,
            written_sequence=tuple(self._written),
            read_sequence=tuple(self._read_out),
        )
