"""Discrete-event model of an asynchronous FIFO pointer-crossing checker.

Timing model (one :class:`Event` = one point on an integer time axis)
--------------------------------------------------------------------

* The write pointer (``wptr``) and read pointer (``rptr``) are each ``k+1``
  bits wide: k address bits plus one wrap bit.
* Only Gray-coded pointers leave a clock domain.  Each destination domain
  carries a two-stage synchronizer (``s1``/``s2``); flag logic sees the old
  ``s2`` output.  When a read and a write edge coincide, *both* sides sample
  the register values as they were before that time step -- including the
  input stage of each synchronizer.
* ``full`` is a write-domain register (reset ``False``); ``empty`` is a
  read-domain register (reset ``True``).  On a clock edge the old flag
  decides whether the enable is accepted; afterwards the flag is recomputed
  from the local *next* pointer Gray and the *old* synchronized value:

      empty_next = gray(rptr_next) == w2r_s2_old
      full_next  = gray(wptr_next) == (r2w_s2_old ^ TOP_TWO_BITS)

* A write is accepted iff ``wclk and wen and not full_old``; a read is
  accepted iff ``rclk and ren and not empty_old``.  Acceptance never consults
  a global queue length -- the data array is addressed memory, not a flag.

Disabling a clock (``wclk=False`` / ``rclk=False``) freezes every register in
that domain, synchronizers included.
"""

import math
from typing import Any, NamedTuple, Optional

from .gray import binary_to_gray, forward_distance, gray_to_binary

__all__ = [
    "AsyncFIFO",
    "DualAsyncFIFO",
    "AsyncFIFOError",
    "Event",
    "DualEvent",
    "EdgeRecord",
    "DualEdgeRecord",
]

_TIME_MAX = 2 ** 63 - 1
_BATCH_MAX = 100_000


class AsyncFIFOError(ValueError):
    """Raised for every contract violation.

    Subclasses :class:`ValueError`; the message always names the offending
    parameter (and, for batches, the event index) so the error is locatable.
    """


def _strict_int(value: Any, name: str, *, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AsyncFIFOError(
            f"{name!r} must be an int (bool rejected); got "
            f"{type(value).__name__}: {value!r}"
        )
    if not lo <= value <= hi:
        raise AsyncFIFOError(
            f"{name!r} must be within [{lo}, {hi}]; got {value!r}"
        )
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AsyncFIFOError(
            f"{name!r} must be a bool; got {type(value).__name__}: {value!r}"
        )
    return value


class Event:
    """One simulation time step.

    Fields
        time:  strictly increasing integer time stamp, ``0 <= t < 2**63``.
        wclk:  write-domain rising edge at this step.
        rclk:  read-domain rising edge at this step.  At least one must be
               ``True``.
        wen:   write enable evaluated at this step.
        ren:   read enable evaluated at this step.
        wdata: payload stored on an accepted write.  Must be ``None`` when
               ``wen`` is ``False``; a float payload must be finite.

    A plain (non-tuple) class because Python 3.14 forbids overriding
    ``__new__`` on ``typing.NamedTuple``; equality and unpacking are supported
    so instances behave like immutable value records.
    """

    __slots__ = ("time", "wclk", "rclk", "wen", "ren", "wdata")

    def __init__(
        self,
        time: int,
        wclk: bool,
        rclk: bool,
        wen: bool = False,
        ren: bool = False,
        wdata: Any = None,
    ) -> None:
        t = _strict_int(time, "time", lo=0, hi=_TIME_MAX)
        we = _strict_bool(wclk, "wclk")
        re = _strict_bool(rclk, "rclk")
        write_en = _strict_bool(wen, "wen")
        read_en = _strict_bool(ren, "ren")
        if not (we or re):
            raise AsyncFIFOError(
                f"event at time={t} has neither wclk nor rclk; at least one "
                "clock edge is required"
            )
        if not write_en and wdata is not None:
            raise AsyncFIFOError(
                f"event at time={t}: wdata must be None when wen is False"
            )
        if isinstance(wdata, float) and not math.isfinite(wdata):
            raise AsyncFIFOError(
                f"event at time={t}: float wdata must be finite; got "
                "non-finite value (NaN/Infinity is rejected)"
            )
        self.time = t
        self.wclk = we
        self.rclk = re
        self.wen = write_en
        self.ren = read_en
        self.wdata = wdata

    def __iter__(self):
        return iter((self.time, self.wclk, self.rclk,
                     self.wen, self.ren, self.wdata))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return tuple(self) == tuple(other)

    def __repr__(self) -> str:
        return (f"Event(time={self.time}, wclk={self.wclk}, "
                f"rclk={self.rclk}, wen={self.wen}, ren={self.ren}, "
                f"wdata={self.wdata!r})")


class EdgeRecord(NamedTuple):
    """Full hardware trace for one time step.

    Synchronizer snapshots are ``(stage1, stage2)`` Gray tuples.  ``*_before``
    values are the registers sampled at the edge (old state); ``*_after``
    values are the register contents after the edge.  ``w_accept`` /
    ``r_accept`` are ``None`` when the corresponding domain did not clock.
    """

    time: int
    wclk: bool
    rclk: bool
    wen: bool
    ren: bool
    w_accept: Optional[bool]
    r_accept: Optional[bool]
    wptr_before: int
    wptr_after: int
    rptr_before: int
    rptr_after: int
    wgray_before: int
    wgray_after: int
    rgray_before: int
    rgray_after: int
    full_before: bool
    full_after: bool
    empty_before: bool
    empty_after: bool
    r2w_before: tuple
    w2r_before: tuple
    r2w_after: tuple
    w2r_after: tuple
    wdata: Any
    rdata: Any


class DualEvent:
    """One time step for one writer and two independent read clock domains.

    The two read enables and clocks are independent.  There is intentionally
    no subscription operation: both consumers exist for the complete lifetime
    after a common reset, which keeps overwrite protection unambiguous.
    """

    __slots__ = (
        "time", "wclk", "rclk0", "rclk1",
        "wen", "ren0", "ren1", "wdata",
    )

    def __init__(
        self,
        time: int,
        wclk: bool,
        rclk0: bool,
        rclk1: bool,
        wen: bool = False,
        ren0: bool = False,
        ren1: bool = False,
        wdata: Any = None,
    ) -> None:
        t = _strict_int(time, "time", lo=0, hi=_TIME_MAX)
        clocks = [
            _strict_bool(wclk, "wclk"),
            _strict_bool(rclk0, "rclk0"),
            _strict_bool(rclk1, "rclk1"),
        ]
        enables = [
            _strict_bool(wen, "wen"),
            _strict_bool(ren0, "ren0"),
            _strict_bool(ren1, "ren1"),
        ]
        if not any(clocks):
            raise AsyncFIFOError(
                f"event at time={t} has wclk, rclk0, and rclk1 all False; "
                "at least one clock edge is required"
            )
        if not enables[0] and wdata is not None:
            raise AsyncFIFOError(
                f"event at time={t}: wdata must be None when wen is False"
            )
        if isinstance(wdata, float) and not math.isfinite(wdata):
            raise AsyncFIFOError(
                f"event at time={t}: float wdata must be finite; got "
                "non-finite value (NaN/Infinity is rejected)"
            )
        self.time = t
        self.wclk = clocks[0]
        self.rclk0 = clocks[1]
        self.rclk1 = clocks[2]
        self.wen = enables[0]
        self.ren0 = enables[1]
        self.ren1 = enables[2]
        self.wdata = wdata

    def __iter__(self):
        return iter((
            self.time, self.wclk, self.rclk0, self.rclk1,
            self.wen, self.ren0, self.ren1, self.wdata,
        ))

    def __eq__(self, other) -> bool:
        if not isinstance(other, DualEvent):
            return NotImplemented
        return tuple(self) == tuple(other)

    def __repr__(self) -> str:
        return (
            f"DualEvent(time={self.time}, wclk={self.wclk}, "
            f"rclk0={self.rclk0}, rclk1={self.rclk1}, wen={self.wen}, "
            f"ren0={self.ren0}, ren1={self.ren1}, wdata={self.wdata!r})"
        )


class DualEdgeRecord(NamedTuple):
    """Trace record for :class:`DualAsyncFIFO`.

    Tuple entries are indexed by read-domain number (0 or 1).  Synchronizers
    are represented per reader as ``(stage1, stage2)`` Gray pairs.
    ``release_ptr_before/after`` is an inspection value derived from actual
    read pointers; ``synced_release_ptr_before/after`` is the conservative
    value visible to the writer.  Neither uses an ordinary unsigned ``min``.
    """

    time: int
    wclk: bool
    rclks: tuple[bool, bool]
    wen: bool
    rens: tuple[bool, bool]
    w_accept: Optional[bool]
    r_accepts: tuple[Optional[bool], Optional[bool]]
    wptr_before: int
    wptr_after: int
    rptrs_before: tuple[int, int]
    rptrs_after: tuple[int, int]
    wgray_before: int
    wgray_after: int
    rgrays_before: tuple[int, int]
    rgrays_after: tuple[int, int]
    full_before: bool
    full_after: bool
    empties_before: tuple[bool, bool]
    empties_after: tuple[bool, bool]
    r2w_before: tuple[tuple[int, int], tuple[int, int]]
    w2r_before: tuple[tuple[int, int], tuple[int, int]]
    r2w_after: tuple[tuple[int, int], tuple[int, int]]
    w2r_after: tuple[tuple[int, int], tuple[int, int]]
    release_ptr_before: int
    release_ptr_after: int
    synced_release_ptr_before: int
    synced_release_ptr_after: int
    wdata: Any
    rdatas: tuple[Any, Any]


class AsyncFIFO:
    """Event-driven asynchronous-FIFO pointer cross-domain checker.

    Parameters
        k: log2 depth; ``k in {1, 2, 3, 4, 5}`` (depth 2..32).

    Usage
        >>> fifo = AsyncFIFO(k=3)
        >>> records = fifo.run([Event(0, True, False, wen=True, wdata=10)])
    """

    def __init__(self, k: int) -> None:
        self._k = _strict_int(k, "k", lo=1, hi=5)
        self._depth = 1 << k
        self._width = k + 1
        self._mask = (1 << (k + 1)) - 1
        self._addr_mask = self._depth - 1
        # Flip the top two pointer bits to test the full condition.
        self._full_mask = 0b11 << (k - 1)
        self.reset()

    def reset(self) -> None:
        """Return every register to its power-on value (clears the trace)."""
        self._wptr = 0
        self._rptr = 0
        self._full = False
        self._empty = True
        self._r2w_s1 = 0  # synchronized read Gray, stage 1 (write domain)
        self._r2w_s2 = 0  # synchronized read Gray, stage 2 (write domain)
        self._w2r_s1 = 0  # synchronized write Gray, stage 1 (read domain)
        self._w2r_s2 = 0  # synchronized write Gray, stage 2 (read domain)
        self._mem = [None] * self._depth
        self._steps = []
        self._last_time = -1
        self._accepted_writes = 0
        self._accepted_reads = 0

    # ------------------------------------------------------------------ props
    @property
    def k(self) -> int:
        return self._k

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def full(self) -> bool:
        """Current write-domain full flag (register output)."""
        return self._full

    @property
    def empty(self) -> bool:
        """Current read-domain empty flag (register output)."""
        return self._empty

    @property
    def wptr(self) -> int:
        return self._wptr

    @property
    def rptr(self) -> int:
        return self._rptr

    @property
    def steps(self) -> tuple:
        """Trace of all :class:`EdgeRecord` values produced so far."""
        return tuple(self._steps)

    @property
    def accepted_writes(self) -> int:
        return self._accepted_writes

    @property
    def accepted_reads(self) -> int:
        return self._accepted_reads

    def snapshot(self) -> dict:
        """Return a copy of every observable register (for inspection)."""
        return {
            "wptr": self._wptr,
            "rptr": self._rptr,
            "wgray": binary_to_gray(self._wptr),
            "rgray": binary_to_gray(self._rptr),
            "full": self._full,
            "empty": self._empty,
            "r2w": (self._r2w_s1, self._r2w_s2),
            "w2r": (self._w2r_s1, self._w2r_s2),
            "last_time": self._last_time,
        }

    # ------------------------------------------------------------------- run
    def run_one(self, event: Event) -> EdgeRecord:
        """Validate and apply a single event; return its trace record."""
        if not isinstance(event, Event):
            raise AsyncFIFOError(
                "run_one expects an Event; got "
                f"{type(event).__name__}: {event!r}"
            )
        if event.time <= self._last_time:
            raise AsyncFIFOError(
                f"event time {event.time} must be strictly greater than the "
                f"last processed time {self._last_time}"
            )
        return self._step(event)

    def run(self, events) -> list:
        """Apply a batch of events atomically.

        The whole batch is validated (types, strictly increasing times,
        size <= 100_000) before any register changes; a failure therefore
        leaves the FIFO exactly as it was.
        """
        if isinstance(events, Event):
            raise AsyncFIFOError(
                "run expects an iterable of Event objects; got a single "
                "Event (use run_one)"
            )
        try:
            batch = list(events)
        except TypeError as exc:
            raise AsyncFIFOError(
                "run expects an iterable of Event objects"
            ) from exc
        if len(batch) > _BATCH_MAX:
            raise AsyncFIFOError(
                f"batch size {len(batch)} exceeds the {_BATCH_MAX} event limit"
            )
        prev = self._last_time
        for i, event in enumerate(batch):
            if not isinstance(event, Event):
                raise AsyncFIFOError(
                    f"events[{i}] must be an Event; got "
                    f"{type(event).__name__}: {event!r}"
                )
            if event.time <= prev:
                raise AsyncFIFOError(
                    f"events[{i}] time {event.time} is not strictly greater "
                    f"than the previous time {prev}"
                )
            prev = event.time
        return [self._step(event) for event in batch]

    # ------------------------------------------------------------- internals
    def _step(self, event: Event) -> EdgeRecord:
        # ---- old state, sampled by both edges at this (possibly common) time
        wptr_o, rptr_o = self._wptr, self._rptr
        full_o, empty_o = self._full, self._empty
        r2w_s1_o, r2w_s2_o = self._r2w_s1, self._r2w_s2
        w2r_s1_o, w2r_s2_o = self._w2r_s1, self._w2r_s2
        wgray_o = binary_to_gray(wptr_o)
        rgray_o = binary_to_gray(rptr_o)

        w_accept = bool(event.wen and not full_o) if event.wclk else None
        r_accept = bool(event.ren and not empty_o) if event.rclk else None

        # ---- local pointers advance only on accepted accesses
        wptr_n = (wptr_o + (1 if w_accept else 0)) & self._mask
        rptr_n = (rptr_o + (1 if r_accept else 0)) & self._mask
        wgray_n = binary_to_gray(wptr_n)
        rgray_n = binary_to_gray(rptr_n)

        # ---- data memory (addressed; never used to derive flags)
        rdata = None
        if w_accept:
            self._mem[wptr_o & self._addr_mask] = event.wdata
            self._accepted_writes += 1
        if r_accept:
            rdata = self._mem[rptr_o & self._addr_mask]
            self._accepted_reads += 1

        # ---- flags: local next-pointer Gray vs OLD synchronizer output
        full_n = full_o
        if event.wclk:
            full_n = wgray_n == (r2w_s2_o ^ self._full_mask)
        empty_n = empty_o
        if event.rclk:
            empty_n = rgray_n == w2r_s2_o

        # ---- two-stage synchronizers; input stage samples the OLD Gray
        if event.wclk:
            r2w_s1_n, r2w_s2_n = rgray_o, r2w_s1_o
        else:
            r2w_s1_n, r2w_s2_n = r2w_s1_o, r2w_s2_o
        if event.rclk:
            w2r_s1_n, w2r_s2_n = wgray_o, w2r_s1_o
        else:
            w2r_s1_n, w2r_s2_n = w2r_s1_o, w2r_s2_o

        # ---- commit
        self._wptr, self._rptr = wptr_n, rptr_n
        self._full, self._empty = full_n, empty_n
        self._r2w_s1, self._r2w_s2 = r2w_s1_n, r2w_s2_n
        self._w2r_s1, self._w2r_s2 = w2r_s1_n, w2r_s2_n
        self._last_time = event.time

        record = EdgeRecord(
            time=event.time,
            wclk=event.wclk,
            rclk=event.rclk,
            wen=event.wen,
            ren=event.ren,
            w_accept=w_accept,
            r_accept=r_accept,
            wptr_before=wptr_o,
            wptr_after=wptr_n,
            rptr_before=rptr_o,
            rptr_after=rptr_n,
            wgray_before=wgray_o,
            wgray_after=wgray_n,
            rgray_before=rgray_o,
            rgray_after=rgray_n,
            full_before=full_o,
            full_after=full_n,
            empty_before=empty_o,
            empty_after=empty_n,
            r2w_before=(r2w_s1_o, r2w_s2_o),
            w2r_before=(w2r_s1_o, w2r_s2_o),
            r2w_after=(r2w_s1_n, r2w_s2_n),
            w2r_after=(w2r_s1_n, w2r_s2_n),
            wdata=event.wdata,
            rdata=rdata,
        )
        self._steps.append(record)
        return record


class DualAsyncFIFO:
    """Asynchronous FIFO with one writer and two independent read domains.

    Every accepted payload is retained until both read-domain pointers have
    moved past it.  The writer sees each read pointer only after that
    pointer's Gray code has passed its own two-stage synchronizer.  Its full
    decision therefore uses the generation-aware forward distance to every
    synchronized read pointer; it never compares wrapped values with plain
    unsigned ``min``.
    """

    def __init__(self, k: int) -> None:
        self._k = _strict_int(k, "k", lo=1, hi=5)
        self._depth = 1 << k
        self._mask = (1 << (k + 1)) - 1
        self._addr_mask = self._depth - 1
        self.reset()

    def reset(self) -> None:
        """Apply the common reset and clear all trace records."""
        self._wptr = 0
        self._rptrs = [0, 0]
        self._full = False
        self._empties = [True, True]
        # For reader i: synchronized reader Gray in the write domain.
        self._r2w = [[0, 0], [0, 0]]
        # For reader i: synchronized writer Gray in reader-i domain.
        self._w2r = [[0, 0], [0, 0]]
        self._mem = [None] * self._depth
        self._steps = []
        self._last_time = -1
        self._accepted_writes = 0
        self._accepted_reads = [0, 0]

    @property
    def k(self) -> int:
        return self._k

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def reader_count(self) -> int:
        return 2

    @property
    def wptr(self) -> int:
        return self._wptr

    @property
    def rptrs(self) -> tuple[int, int]:
        return tuple(self._rptrs)

    @property
    def full(self) -> bool:
        return self._full

    @property
    def empties(self) -> tuple[bool, bool]:
        return tuple(self._empties)

    @property
    def steps(self) -> tuple:
        return tuple(self._steps)

    @property
    def accepted_writes(self) -> int:
        return self._accepted_writes

    @property
    def accepted_reads(self) -> tuple[int, int]:
        return tuple(self._accepted_reads)

    @property
    def release_ptr(self) -> int:
        """Actual slowest-reader pointer derived with wrap generations."""
        return self._slowest_reader(self._wptr, self._rptrs)

    @property
    def synced_release_ptr(self) -> int:
        """Conservative slowest pointer visible through both synchronizers."""
        synced = [gray_to_binary(self._r2w[i][1]) for i in range(2)]
        return self._slowest_reader(self._wptr, synced)

    def snapshot(self) -> dict:
        r2w = tuple((pair[0], pair[1]) for pair in self._r2w)
        w2r = tuple((pair[0], pair[1]) for pair in self._w2r)
        return {
            "wptr": self._wptr,
            "rptrs": tuple(self._rptrs),
            "wgray": binary_to_gray(self._wptr),
            "rgrays": tuple(binary_to_gray(p) for p in self._rptrs),
            "full": self._full,
            "empties": tuple(self._empties),
            "r2w": r2w,
            "w2r": w2r,
            "release_ptr": self.release_ptr,
            "synced_release_ptr": self.synced_release_ptr,
            "last_time": self._last_time,
        }

    def run_one(self, event: DualEvent) -> DualEdgeRecord:
        if not isinstance(event, DualEvent):
            raise AsyncFIFOError(
                "run_one expects a DualEvent; got "
                f"{type(event).__name__}: {event!r}"
            )
        if event.time <= self._last_time:
            raise AsyncFIFOError(
                f"event time {event.time} must be strictly greater than the "
                f"last processed time {self._last_time}"
            )
        return self._step(event)

    def run(self, events) -> list:
        if isinstance(event := events, DualEvent):
            raise AsyncFIFOError(
                "run expects an iterable of DualEvent objects; got a single "
                "DualEvent (use run_one)"
            )
        try:
            batch = list(events)
        except TypeError as exc:
            raise AsyncFIFOError(
                "run expects an iterable of DualEvent objects"
            ) from exc
        if len(batch) > _BATCH_MAX:
            raise AsyncFIFOError(
                f"batch size {len(batch)} exceeds the {_BATCH_MAX} event limit"
            )
        prev = self._last_time
        for i, event in enumerate(batch):
            if not isinstance(event, DualEvent):
                raise AsyncFIFOError(
                    f"events[{i}] must be a DualEvent; got "
                    f"{type(event).__name__}: {event!r}"
                )
            if event.time <= prev:
                raise AsyncFIFOError(
                    f"events[{i}] time {event.time} is not strictly greater "
                    f"than the previous time {prev}"
                )
            prev = event.time
        return [self._step(event) for event in batch]

    def _distance(self, later: int, earlier: int) -> int:
        return forward_distance(later, earlier, self._k)

    def _slowest_reader(self, wptr: int, rptrs) -> int:
        slowest = rptrs[0]
        largest_distance = self._distance(wptr, slowest)
        for rptr in rptrs[1:]:
            distance = self._distance(wptr, rptr)
            if distance > largest_distance:
                slowest = rptr
                largest_distance = distance
        return slowest

    def _step(self, event: DualEvent) -> DualEdgeRecord:
        wptr_o = self._wptr
        rptrs_o = tuple(self._rptrs)
        full_o = self._full
        empties_o = tuple(self._empties)
        r2w_o = tuple((pair[0], pair[1]) for pair in self._r2w)
        w2r_o = tuple((pair[0], pair[1]) for pair in self._w2r)
        wgray_o = binary_to_gray(wptr_o)
        rgrays_o = tuple(binary_to_gray(p) for p in rptrs_o)
        synced_release_o = self._slowest_reader(
            wptr_o,
            [gray_to_binary(pair[1]) for pair in r2w_o],
        )
        release_o = self._slowest_reader(wptr_o, rptrs_o)

        rclks = (event.rclk0, event.rclk1)
        rens = (event.ren0, event.ren1)
        w_accept = bool(event.wen and not full_o) if event.wclk else None
        r_accepts = tuple(
            bool(rens[i] and not empties_o[i]) if rclks[i] else None
            for i in range(2)
        )

        wptr_n = (wptr_o + (1 if w_accept else 0)) & self._mask
        rptrs_n = tuple(
            (rptrs_o[i] + (1 if r_accepts[i] else 0)) & self._mask
            for i in range(2)
        )
        wgray_n = binary_to_gray(wptr_n)
        rgrays_n = tuple(binary_to_gray(p) for p in rptrs_n)

        rdatas = [None, None]
        for i in range(2):
            if r_accepts[i]:
                rdatas[i] = self._mem[rptrs_o[i] & self._addr_mask]
                self._accepted_reads[i] += 1
        if w_accept:
            self._mem[wptr_o & self._addr_mask] = event.wdata
            self._accepted_writes += 1

        r2w_n = [list(pair) for pair in r2w_o]
        w2r_n = [list(pair) for pair in w2r_o]
        if event.wclk:
            for i in range(2):
                r2w_n[i] = [rgrays_o[i], r2w_o[i][0]]
            synced = [gray_to_binary(pair[1]) for pair in r2w_n]
            self._full = any(
                self._distance(wptr_n, rptr) >= self._depth
                for rptr in synced
            )

        new_empties = list(empties_o)
        for i in range(2):
            if rclks[i]:
                w2r_n[i] = [wgray_o, w2r_o[i][0]]
                synced_wptr = gray_to_binary(w2r_n[i][1])
                new_empties[i] = (
                    self._distance(synced_wptr, rptrs_n[i]) == 0
                )

        release_n = self._slowest_reader(wptr_n, rptrs_n)
        synced_release_n = self._slowest_reader(
            wptr_n,
            [gray_to_binary(pair[1]) for pair in r2w_n],
        )

        self._wptr = wptr_n
        self._rptrs = list(rptrs_n)
        self._empties = new_empties
        self._r2w = r2w_n
        self._w2r = w2r_n
        self._last_time = event.time

        record = DualEdgeRecord(
            time=event.time,
            wclk=event.wclk,
            rclks=rclks,
            wen=event.wen,
            rens=rens,
            w_accept=w_accept,
            r_accepts=r_accepts,
            wptr_before=wptr_o,
            wptr_after=wptr_n,
            rptrs_before=rptrs_o,
            rptrs_after=rptrs_n,
            wgray_before=wgray_o,
            wgray_after=wgray_n,
            rgrays_before=rgrays_o,
            rgrays_after=rgrays_n,
            full_before=full_o,
            full_after=self._full,
            empties_before=empties_o,
            empties_after=tuple(new_empties),
            r2w_before=r2w_o,
            w2r_before=w2r_o,
            r2w_after=tuple((p[0], p[1]) for p in r2w_n),
            w2r_after=tuple((p[0], p[1]) for p in w2r_n),
            release_ptr_before=release_o,
            release_ptr_after=release_n,
            synced_release_ptr_before=synced_release_o,
            synced_release_ptr_after=synced_release_n,
            wdata=event.wdata,
            rdatas=tuple(rdatas),
        )
        self._steps.append(record)
        return record
