"""FPGA asynchronous-FIFO pointer cross-domain checking model.

Public surface:

* :class:`AsyncFIFO`             -- one-reader event driven hardware model.
* :class:`DualReaderAsyncFIFO`   -- two independent readers with one writer.
* :class:`Event`                 -- one-reader time step.
* :class:`DualReaderEvent`       -- one-writer/two-reader time step.
* :class:`EdgeRecord`            -- one-reader pointer/synchronizer trace.
* :class:`DualReaderEdgeRecord`  -- two-reader pointer/synchronizer trace.
* :class:`AsyncFIFOError` -- locatable contract violation (subclass of ValueError).
* :func:`binary_to_gray` / :func:`gray_to_binary` -- k+1 bit Gray helpers.
"""

from .gray import binary_to_gray, forward_distance, gray_to_binary
from .model import (
    AsyncFIFO,
    AsyncFIFOError,
    DualReaderAsyncFIFO,
    DualReaderEdgeRecord,
    DualReaderEvent,
    EdgeRecord,
    Event,
)

__all__ = [
    "AsyncFIFO",
    "DualReaderAsyncFIFO",
    "AsyncFIFOError",
    "Event",
    "DualReaderEvent",
    "EdgeRecord",
    "DualReaderEdgeRecord",
    "binary_to_gray",
    "forward_distance",
    "gray_to_binary",
]
