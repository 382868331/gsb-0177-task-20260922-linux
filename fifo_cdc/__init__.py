"""fifo_cdc —— FPGA 异步 FIFO 指针跨域检查模型（仅标准库）。

公开接口见 :class:`fifo_cdc.core.AsyncFIFOModel` 与
:class:`fifo_cdc.reference.TraceOracle`。
"""

from .core import (
    K_MAX,
    K_MIN,
    AsyncFIFOModel,
    ClockEdge,
    ClockKind,
    EdgeRecord,
    FIFOConfigurationError,
    FIFOError,
    FIFOInputError,
    FIFOSnapshot,
    FIFOScheduleError,
    binary_to_gray,
    gray_to_binary,
    nop_edge,
    read_edge,
    write_edge,
)
from .reference import OracleReport, TraceOracle

__all__ = [
    "K_MIN",
    "K_MAX",
    "ClockKind",
    "ClockEdge",
    "EdgeRecord",
    "FIFOSnapshot",
    "AsyncFIFOModel",
    "FIFOError",
    "FIFOConfigurationError",
    "FIFOInputError",
    "FIFOScheduleError",
    "binary_to_gray",
    "gray_to_binary",
    "write_edge",
    "read_edge",
    "nop_edge",
    "TraceOracle",
    "OracleReport",
]
