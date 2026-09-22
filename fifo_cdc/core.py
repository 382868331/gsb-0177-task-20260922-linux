"""FPGA 异步 FIFO 指针跨域检查可复用模型（仅标准库）。

本模块对深度 ``2**k``（``k=1..5``）的经典二进制指针/格雷码跨域异步 FIFO
做离散事件级建模，供离线数字硬件验证复用：

* 读写指针均为 ``k+1`` 位二进制计数器，跨域传递两级同步后的格雷码；
* ``full``（写域）/ ``empty``（读域）是本域寄存器，复位为 ``False`` / ``True``；
* 时钟边沿先按旧旗标决定是否接受操作，再用“接受后的本地下一指针格雷”与
  “旧的二级同步输出”更新本域旗标；
* 读写两边沿可在同一时刻发生，彼此采样到的都是对方边沿之前的旧状态；
* 核心判定只依赖硬件旗标与同步寄存器，**不**使用全局队列长度代替旗标。

模块是确定性的整数/离散事件模拟，不产生任何浮点结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Iterable, Optional

__all__ = [
    "K_MIN",
    "K_MAX",
    "ClockKind",
    "FIFOError",
    "FIFOConfigurationError",
    "FIFOInputError",
    "FIFOScheduleError",
    "ClockEdge",
    "EdgeRecord",
    "FIFOSnapshot",
    "AsyncFIFOModel",
    "binary_to_gray",
    "gray_to_binary",
    "write_edge",
    "read_edge",
    "nop_edge",
]

K_MIN: Final[int] = 1
K_MAX: Final[int] = 5


class FIFOError(Exception):
    """本模块所有错误的基类。"""


class FIFOConfigurationError(FIFOError):
    """构造参数非法（如 k 超出 1..5）。"""


class FIFOInputError(FIFOError):
    """边沿/数值输入非法（类型、NaN/Infinity、矛盾操作等），信息可定位。"""


class FIFOScheduleError(FIFOError):
    """事件时间非法（倒流、同一域同一时刻重复边沿）。"""


class ClockKind(str, Enum):
    """时钟域种类。"""

    WRITE = "write"
    READ = "read"


def binary_to_gray(value: int, bits: int) -> int:
    """把 ``bits`` 位无符号二进制数转为格雷码（截取低 ``bits`` 位）。"""
    if isinstance(value, bool) or isinstance(bits, bool):
        raise FIFOInputError("binary_to_gray 的参数不能为 bool")
    if not isinstance(value, int) or not isinstance(bits, int):
        raise FIFOInputError("binary_to_gray 的参数必须为整数")
    if bits <= 0:
        raise FIFOConfigurationError(f"bits 必须为正整数, 得到 {bits!r}")
    if not 0 <= value < (1 << bits):
        raise FIFOInputError(f"value={value!r} 超出 {bits} 位无符号范围")
    return (value ^ (value >> 1)) & ((1 << bits) - 1)


def gray_to_binary(value: int, bits: int) -> int:
    """把 ``bits`` 位格雷码还原为二进制数（截取低 ``bits`` 位）。"""
    if isinstance(value, bool) or isinstance(bits, bool):
        raise FIFOInputError("gray_to_binary 的参数不能为 bool")
    if not isinstance(value, int) or not isinstance(bits, int):
        raise FIFOInputError("gray_to_binary 的参数必须为整数")
    if bits <= 0:
        raise FIFOConfigurationError(f"bits 必须为正整数, 得到 {bits!r}")
    if not 0 <= value < (1 << bits):
        raise FIFOInputError(f"value={value!r} 超出 {bits} 位无符号范围")
    value &= (1 << bits) - 1
    result = value
    shift = 1
    while shift < bits:
        result ^= result >> shift
        shift <<= 1
    return result


def _strict_int(value: object, what: str) -> int:
    """整数参数校验：拒绝 bool、拒绝 NaN/Infinity 浮点、不做静默取整。"""
    if isinstance(value, bool):
        raise FIFOInputError(f"{what} 不能为 bool（整数参数拒绝布尔值）")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise FIFOInputError(f"{what} 不能为 NaN")
        if math.isinf(value):
            raise FIFOInputError(f"{what} 不能为 Infinity")
        raise FIFOInputError(f"{what} 必须是整数, 不接受浮点 {value!r}（不静默取整）")
    raise FIFOInputError(f"{what} 必须是整数, 得到 {type(value).__name__}")


def _checked_time(time: object) -> int:
    t = _strict_int(time, "time")
    if t < 0:
        raise FIFOInputError(f"time 必须为非负整数, 得到 {t}")
    return t


@dataclass(frozen=True)
class ClockEdge:
    """一个时钟边沿事件。

    * ``clock``：写域或读域时钟；
    * ``time``：非负整数时刻，同一时刻允许写、读各一条边沿，表示同刻边沿；
    * ``write``：写域边沿是否尝试写（仅写域时钟可置真）；
    * ``read``：读域边沿是否尝试读（仅读域时钟可置真）；
    * ``data``：写负载，仅写域允许给出，读域必须为 ``None``。

    两者皆假表示一次空转边沿（只推进同步寄存器）。读写不得同时为真。
    """

    clock: ClockKind
    time: int
    write: bool = False
    read: bool = False
    data: object = None


def write_edge(time: int, data: object = None) -> ClockEdge:
    """构造一次写尝试边沿。``time`` 立即校验（非负整数、拒 NaN/Inf/bool）。"""
    t = _checked_time(time)
    return ClockEdge(ClockKind.WRITE, t, write=True, data=data)


def read_edge(time: int) -> ClockEdge:
    """构造一次读尝试边沿。``time`` 立即校验（非负整数、拒 NaN/Inf/bool）。"""
    return ClockEdge(ClockKind.READ, _checked_time(time), read=True)


def nop_edge(clock: ClockKind, time: int) -> ClockEdge:
    """构造一次空转边沿（不发起操作，仅让同步链前进一步）。"""
    if not isinstance(clock, ClockKind):
        raise FIFOInputError(f"clock 必须是 ClockKind 枚举, 得到 {clock!r}")
    return ClockEdge(clock, _checked_time(time))


@dataclass(frozen=True)
class EdgeRecord:
    """一条边沿提交后的寄存器/操作轨迹（域值为该边沿之后的状态）。"""

    time: int
    write_attempt: bool
    write_accepted: bool
    write_data: object
    read_attempt: bool
    read_accepted: bool
    read_data: object
    # 边沿之后的二进制/格雷指针
    wptr_bin: int
    rptr_bin: int
    wptr_gray: int
    rptr_gray: int
    # 边沿之后的两级同步寄存器
    wq1_rptr: int
    wq2_rptr: int
    rq1_wptr: int
    rq2_wptr: int
    # 边沿之后与边沿之前（旧）的本域旗标
    full: bool
    empty: bool
    full_before: bool
    empty_before: bool


@dataclass(frozen=True)
class FIFOSnapshot:
    """模型当前可观测状态的不可变快照。"""

    time: int
    wptr_bin: int
    rptr_bin: int
    wptr_gray: int
    rptr_gray: int
    wq1_rptr: int
    wq2_rptr: int
    rq1_wptr: int
    rq2_wptr: int
    full: bool
    empty: bool
    writes_accepted: int
    writes_rejected: int
    reads_accepted: int
    reads_rejected: int


@dataclass
class _State:
    """边沿之前（基准时刻）的完整内部状态。"""

    time: int = 0
    wptr: int = 0
    rptr: int = 0
    wq1: int = 0
    wq2: int = 0
    rq1: int = 0
    rq2: int = 0
    full: bool = False
    empty: bool = True
    writes_accepted: int = 0
    writes_rejected: int = 0
    reads_accepted: int = 0
    reads_rejected: int = 0
    storage: dict[int, object] = field(default_factory=dict)


class AsyncFIFOModel:
    """深度 ``2**k`` 的异步 FIFO 指针跨域离散事件模型。

    参数:
        k: FIFO 深度指数，深度 = ``2**k``，仅接受整数 ``1..5``（拒绝 bool）。

    复位后：写/读指针为 0，两级同步寄存器为 0，``full=False``、``empty=True``。
    时间从 0 开始，事件时刻必须非负且不倒流；同一时刻写域、读域各至多一条
    边沿，两条同刻边沿按“双方都采样边沿前旧状态”的原子语义处理。
    """

    def __init__(self, k: int) -> None:
        kk = _strict_int(k, "k")
        if not K_MIN <= kk <= K_MAX:
            raise FIFOConfigurationError(
                f"k 必须在 [{K_MIN}, {K_MAX}] 内（深度 2..32）, 得到 k={kk!r}"
            )
        self._k: int = kk
        self._depth: int = 1 << kk
        self._bits: int = kk + 1
        self._ptr_mask: int = (1 << self._bits) - 1
        # full 判定：把同步读格雷仅最高两位取反后与下一写格雷全位比较。
        self._full_mask: int = 0b11 << (kk - 1)
        self._state: _State = _State()
        # 当前时刻开始时的基准状态（同刻边沿对都从该快照计算）。
        self._base: _State = self._copy_state(self._state)
        self._staged: list[tuple[ClockEdge, EdgeRecord, bool]] = []
        self._staged_clocks: set[ClockKind] = set()
        self._trace: list[EdgeRecord] = []

    # ------------------------------------------------------------------ #
    # 公开属性
    # ------------------------------------------------------------------ #
    @property
    def k(self) -> int:
        return self._k

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def ptr_bits(self) -> int:
        return self._bits

    @property
    def time(self) -> int:
        return self._state.time

    @property
    def full(self) -> bool:
        return self._state.full

    @property
    def empty(self) -> bool:
        return self._state.empty

    @property
    def wptr_bin(self) -> int:
        return self._state.wptr

    @property
    def rptr_bin(self) -> int:
        return self._state.rptr

    @property
    def wptr_gray(self) -> int:
        return self._gray(self._state.wptr)

    @property
    def rptr_gray(self) -> int:
        return self._gray(self._state.rptr)

    @property
    def wq2_rptr(self) -> int:
        return self._state.wq2

    @property
    def rq2_wptr(self) -> int:
        return self._state.rq2

    @property
    def writes_accepted(self) -> int:
        return self._state.writes_accepted

    @property
    def writes_rejected(self) -> int:
        return self._state.writes_rejected

    @property
    def reads_accepted(self) -> int:
        return self._state.reads_accepted

    @property
    def reads_rejected(self) -> int:
        return self._state.reads_rejected

    @property
    def trace(self) -> tuple[EdgeRecord, ...]:
        """已提交边沿的轨迹（按提交顺序）。"""
        return tuple(self._trace)

    def snapshot(self) -> FIFOSnapshot:
        s = self._state
        return FIFOSnapshot(
            time=s.time,
            wptr_bin=s.wptr,
            rptr_bin=s.rptr,
            wptr_gray=self._gray(s.wptr),
            rptr_gray=self._gray(s.rptr),
            wq1_rptr=s.wq1,
            wq2_rptr=s.wq2,
            rq1_wptr=s.rq1,
            rq2_wptr=s.rq2,
            full=s.full,
            empty=s.empty,
            writes_accepted=s.writes_accepted,
            writes_rejected=s.writes_rejected,
            reads_accepted=s.reads_accepted,
            reads_rejected=s.reads_rejected,
        )

    # ------------------------------------------------------------------ #
    # 核心事件接口
    # ------------------------------------------------------------------ #
    def step(self, edge: ClockEdge) -> EdgeRecord:
        """提交一条时钟边沿，返回该边沿之后的 :class:`EdgeRecord`。

        同刻写/读边沿可分两次调用，第二条仍按采样旧状态的同刻语义处理。
        输入错误在任何状态修改前抛出；调用失败不会改变模型。
        """
        clock, t, do_write, do_read, data = self._validate_edge(edge)
        if t < self._base.time:
            raise FIFOScheduleError(
                f"时间倒流: 边沿时刻 {t} 早于当前时刻 {self._base.time}"
            )
        if t == self._base.time and clock in self._staged_clocks:
            raise FIFOScheduleError(
                f"{clock.value} 域在时刻 {t} 已存在边沿, 同一域同一时刻不得重复"
            )
        if t > self._base.time:
            # 上一时刻已完整提交，以当前状态作为新基准。
            self._base = self._copy_state(self._state)
            self._base.time = t
            self._staged = []
            self._staged_clocks = set()

        record = self._compute_edge(clock, t, do_write, do_read, data)
        self._staged.append((edge, record, clock == ClockKind.WRITE))
        self._staged_clocks.add(clock)
        self._commit_stage()
        self._trace.append(record)
        return record

    def run(self, edges: Iterable[ClockEdge]) -> tuple[EdgeRecord, ...]:
        """批量提交边沿（时刻非递减；同刻写读各至多一条）。

        先完整校验全部输入，再逐条提交；任一条非法则整体不产生任何状态变更。
        """
        edge_list = list(edges)
        # ---- 纯校验阶段：不触碰模型状态 ----
        normalized: list[tuple[ClockKind, int, bool, bool, object]] = []
        prev_time: Optional[int] = None
        clocks_at_time: set[ClockKind] = set()
        for index, edge in enumerate(edge_list):
            if not isinstance(edge, ClockEdge):
                raise FIFOInputError(
                    f"第 {index} 个事件不是 ClockEdge, 得到 {type(edge).__name__}"
                )
            clock, t, do_write, do_read, data = self._validate_edge(
                edge, where=f"批内序号 {index}"
            )
            if prev_time is not None and t < prev_time:
                raise FIFOScheduleError(
                    f"批内时间倒流: 序号 {index} 时刻 {t} 早于前一时刻 {prev_time}"
                )
            if prev_time is None and t < self._base.time:
                raise FIFOScheduleError(
                    f"批内时间倒流: 序号 {index} 时刻 {t} 早于模型当前时刻 "
                    f"{self._base.time}"
                )
            if t != prev_time:
                clocks_at_time = set()
                # 该时刻可能已通过 step() 单独提交过一条同刻边沿。
                if t == self._base.time:
                    clocks_at_time.update(self._staged_clocks)
                prev_time = t
            if clock in clocks_at_time:
                raise FIFOScheduleError(
                    f"{clock.value} 域在时刻 {t} 出现重复边沿（批内序号 {index}）"
                )
            clocks_at_time.add(clock)
            normalized.append((clock, t, do_write, do_read, data))

        # ---- 提交阶段：输入已全部合法 ----
        records: list[EdgeRecord] = []
        for clock, t, do_write, do_read, data in normalized:
            if t > self._base.time:
                self._base = self._copy_state(self._state)
                self._base.time = t
                self._staged = []
                self._staged_clocks = set()
            records.append(self.step(
                ClockEdge(
                    clock=clock,
                    time=t,
                    write=do_write,
                    read=do_read,
                    data=data,
                )
            ))
        return tuple(records)

    # ------------------------------------------------------------------ #
    # 硬件语义实现
    # ------------------------------------------------------------------ #
    def _gray(self, value: int) -> int:
        return (value ^ (value >> 1)) & self._ptr_mask

    def _compute_edge(
        self,
        clock: ClockKind,
        t: int,
        do_write: bool,
        do_read: bool,
        data: object,
    ) -> EdgeRecord:
        """从时刻开始时的基准状态计算一条边沿的结果（不修改模型）。"""
        b = self._base

        if clock is ClockKind.WRITE:
            # 先用旧 full 决定接受，再算下一指针/新旗标。
            w_accepted = do_write and not b.full
            wptr_next = (b.wptr + (1 if w_accepted else 0)) & self._ptr_mask
            wg_next = self._gray(wptr_next)
            # full：下一写格雷 == 同步读格雷“仅最高两位取反”后的值（全位比较，
            # 低位天然保持一致）。wq2 在 ptr_bits 位宽内，异或后不越界。
            full_next = wg_next == (b.wq2 ^ self._full_mask)
            # 写域同步链采样的是读域边沿前的旧格雷（同刻读边沿也一样）。
            wq1_next = self._gray(b.rptr)
            wq2_next = b.wq1

            return EdgeRecord(
                time=t,
                write_attempt=do_write,
                write_accepted=w_accepted,
                write_data=data if do_write else None,
                read_attempt=False,
                read_accepted=False,
                read_data=None,
                wptr_bin=wptr_next,
                rptr_bin=b.rptr,
                wptr_gray=wg_next,
                rptr_gray=self._gray(b.rptr),
                wq1_rptr=wq1_next,
                wq2_rptr=wq2_next,
                rq1_wptr=b.rq1,
                rq2_wptr=b.rq2,
                full=full_next,
                empty=b.empty,
                full_before=b.full,
                empty_before=b.empty,
            )

        # 读域
        r_accepted = do_read and not b.empty
        rptr_next = (b.rptr + (1 if r_accepted else 0)) & self._ptr_mask
        rg_next = self._gray(rptr_next)
        empty_next = rg_next == b.rq2
        rq1_next = self._gray(b.wptr)  # 采样写域边沿前旧格雷
        rq2_next = b.rq1

        read_data: object = None
        if r_accepted:
            addr = b.rptr & (self._depth - 1)
            # 已接受读必不越界（由旗标语义 + 测试中的抽象队列独立核对保证）。
            read_data = b.storage[addr]

        return EdgeRecord(
            time=t,
            write_attempt=False,
            write_accepted=False,
            write_data=None,
            read_attempt=do_read,
            read_accepted=r_accepted,
            read_data=read_data,
            wptr_bin=b.wptr,
            rptr_bin=rptr_next,
            wptr_gray=self._gray(b.wptr),
            rptr_gray=rg_next,
            wq1_rptr=b.wq1,
            wq2_rptr=b.wq2,
            rq1_wptr=rq1_next,
            rq2_wptr=rq2_next,
            full=b.full,
            empty=empty_next,
            full_before=b.full,
            empty_before=b.empty,
        )

    def _commit_stage(self) -> None:
        """依据基准状态 + 当前时刻已暂存的边沿，原子地重建当前状态。"""
        b = self._base
        s = _State(
            time=b.time,
            wptr=b.wptr,
            rptr=b.rptr,
            wq1=b.wq1,
            wq2=b.wq2,
            rq1=b.rq1,
            rq2=b.rq2,
            full=b.full,
            empty=b.empty,
            writes_accepted=b.writes_accepted,
            writes_rejected=b.writes_rejected,
            reads_accepted=b.reads_accepted,
            reads_rejected=b.reads_rejected,
            storage=dict(b.storage),
        )
        write_rec: Optional[EdgeRecord] = None
        read_rec: Optional[EdgeRecord] = None
        for _edge, record, is_write in self._staged:
            if is_write:
                write_rec = record
            else:
                read_rec = record

        if write_rec is not None:
            r = write_rec
            s.wptr = r.wptr_bin
            s.wq1 = r.wq1_rptr
            s.wq2 = r.wq2_rptr
            s.full = r.full
            if r.write_attempt:
                if r.write_accepted:
                    s.writes_accepted += 1
                else:
                    s.writes_rejected += 1
        if read_rec is not None:
            r = read_rec
            s.rptr = r.rptr_bin
            s.rq1 = r.rq1_wptr
            s.rq2 = r.rq2_wptr
            s.empty = r.empty
            if r.read_attempt:
                if r.read_accepted:
                    s.reads_accepted += 1
                else:
                    s.reads_rejected += 1

        # 同刻数据通路：读先按旧地址取旧数据（已在 record 中），写再落新数据。
        if read_rec is not None and read_rec.read_accepted:
            s.storage.pop(b.rptr & (self._depth - 1))
        if write_rec is not None and write_rec.write_accepted:
            s.storage[b.wptr & (self._depth - 1)] = write_rec.write_data

        self._state = s

    # ------------------------------------------------------------------ #
    # 校验与内部工具
    # ------------------------------------------------------------------ #
    def _validate_edge(
        self, edge: object, *, where: str = ""
    ) -> tuple[ClockKind, int, bool, bool, object]:
        prefix = f"{where}: " if where else ""
        if not isinstance(edge, ClockEdge):
            raise FIFOInputError(
                f"{prefix}事件必须是 ClockEdge, 得到 {type(edge).__name__}"
            )
        if not isinstance(edge.clock, ClockKind):
            raise FIFOInputError(
                f"{prefix}clock 必须是 ClockKind 枚举, 得到 {edge.clock!r}"
            )
        t = _strict_int(edge.time, f"{prefix}time")
        if t < 0:
            raise FIFOInputError(f"{prefix}time 必须为非负整数, 得到 {t}")
        if not isinstance(edge.write, bool) or not isinstance(edge.read, bool):
            raise FIFOInputError(f"{prefix}write/read 标志必须为 bool")
        if edge.write and edge.read:
            raise FIFOInputError(
                f"{prefix}时刻 {t}: 单边沿不得同时请求写和读"
            )
        if edge.clock is ClockKind.WRITE:
            if edge.read:
                raise FIFOInputError(
                    f"{prefix}时刻 {t}: 写域时钟边沿不能携带读请求"
                )
            return ClockKind.WRITE, t, bool(edge.write), False, edge.data
        if edge.write:
            raise FIFOInputError(
                f"{prefix}时刻 {t}: 读域时钟边沿不能携带写请求"
            )
        if edge.data is not None:
            raise FIFOInputError(
                f"{prefix}时刻 {t}: 读域边沿不能携带写数据（得到 {edge.data!r}）"
            )
        return ClockKind.READ, t, False, bool(edge.read), None

    @staticmethod
    def _copy_state(s: _State) -> _State:
        return _State(
            time=s.time,
            wptr=s.wptr,
            rptr=s.rptr,
            wq1=s.wq1,
            wq2=s.wq2,
            rq1=s.rq1,
            rq2=s.rq2,
            full=s.full,
            empty=s.empty,
            writes_accepted=s.writes_accepted,
            writes_rejected=s.writes_rejected,
            reads_accepted=s.reads_accepted,
            reads_rejected=s.reads_rejected,
            storage=dict(s.storage),
        )
