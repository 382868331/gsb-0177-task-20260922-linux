"""FPGA asynchronous-FIFO pointer cross-domain checking model.

Public surface:

* :class:`AsyncFIFO` / :class:`DualAsyncFIFO` -- one/two-reader models.
* :class:`Event` / :class:`DualEvent`         -- one simulation time step.
* :class:`EdgeRecord` / :class:`DualEdgeRecord` -- register/data traces.
* :class:`AsyncFIFOError` -- locatable contract violation (subclass of ValueError).
* :func:`binary_to_gray` / :func:`gray_to_binary` -- k+1 bit Gray helpers.
"""

from .gray import binary_to_gray, forward_distance, gray_to_binary
from .model import (
    AsyncFIFO,
    AsyncFIFOError,
    DualAsyncFIFO,
    DualEdgeRecord,
    DualEvent,
    EdgeRecord,
    Event,
)

__all__ = [
    "AsyncFIFO",
    "DualAsyncFIFO",
    "AsyncFIFOError",
    "Event",
    "DualEvent",
    "EdgeRecord",
    "DualEdgeRecord",
    "binary_to_gray",
    "forward_distance",
    "gray_to_binary",
]
