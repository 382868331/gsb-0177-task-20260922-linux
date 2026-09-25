"""FPGA asynchronous-FIFO pointer cross-domain checking model.

Public surface:

* :class:`AsyncFIFO`      -- event driven hardware model.
* :class:`ReadyValidFIFO` -- AsyncFIFO plus a read-domain ready/valid
  output register (the core is wrapped, not modified).
* :class:`Event`          -- one core time step: clocks, enables, payload.
* :class:`RVEvent`        -- one wrapped-FIFO time step (``ready`` replaces
  ``ren``; the wrapper issues the dequeue itself).
* :class:`EdgeRecord`     -- core accept/reject + pointer/synchronizer trace.
* :class:`RVRecord`       -- wrapped trace: output-register state, delivery,
  issued ``ren``, plus the inner :class:`EdgeRecord`.
* :class:`AsyncFIFOError` -- locatable contract violation (subclass of ValueError).
* :func:`binary_to_gray` / :func:`gray_to_binary` -- k+1 bit Gray helpers.
"""

from .gray import binary_to_gray, gray_to_binary
from .model import AsyncFIFO, AsyncFIFOError, EdgeRecord, Event
from .readyvalid import ReadyValidFIFO, RVEvent, RVRecord

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
