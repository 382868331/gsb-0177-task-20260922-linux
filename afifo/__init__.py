"""FPGA asynchronous-FIFO pointer cross-domain checking model.

Public surface:

* :class:`AsyncFIFO`      -- event driven hardware model.
* :class:`Event`          -- one time step: which domain clocks edge, enables.
* :class:`EdgeRecord`     -- accept/reject result + pointer/synchronizer trace.
* :class:`ReadyValidFIFO` -- AsyncFIFO + read-domain ready/valid output register.
* :class:`RVEvent`        -- one time step for the wrapper (ready instead of ren).
* :class:`RVRecord`       -- wrapper trace: load/deliver/valid + inner EdgeRecord.
* :class:`AsyncFIFOError` -- locatable contract violation (subclass of ValueError).
* :func:`binary_to_gray` / :func:`gray_to_binary` -- k+1 bit Gray helpers.
"""

from .gray import binary_to_gray, gray_to_binary
from .model import AsyncFIFO, AsyncFIFOError, EdgeRecord, Event
from .outreg import ReadyValidFIFO, RVEvent, RVRecord

__all__ = [
    "AsyncFIFO",
    "AsyncFIFOError",
    "Event",
    "EdgeRecord",
    "ReadyValidFIFO",
    "RVEvent",
    "RVRecord",
    "binary_to_gray",
    "gray_to_binary",
]
