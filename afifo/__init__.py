"""FPGA asynchronous-FIFO pointer cross-domain checking model.

Public surface:

* :class:`AsyncFIFO`      -- event driven hardware model.
* :class:`Event`          -- one time step: which domain clocks edge, enables.
* :class:`EdgeRecord`     -- accept/reject result + pointer/synchronizer trace.
* :class:`AsyncFIFOError` -- locatable contract violation (subclass of ValueError).
* :func:`binary_to_gray` / :func:`gray_to_binary` -- k+1 bit Gray helpers.
"""

from .gray import binary_to_gray, gray_to_binary
from .model import AsyncFIFO, AsyncFIFOError, EdgeRecord, Event

__all__ = [
    "AsyncFIFO",
    "AsyncFIFOError",
    "Event",
    "EdgeRecord",
    "binary_to_gray",
    "gray_to_binary",
]
