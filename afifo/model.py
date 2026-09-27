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

User-programmable warning registers ``almost_full`` (write domain) and
``almost_empty`` (read domain) fire when the free space / stored entries
*guaranteed* by the local next pointer and the OLD synchronizer contents
drop to the configured margins.  Almost-full uses the oldest synced read
pointer (``r2w_s2``, occupancy upper bound); almost-empty uses the newest
synced write pointer (``w2r_s1``, occupancy lower bound).  They never affect
acceptance, full/empty, the synchronizers, or the data path.

Disabling a clock (``wclk=False`` / ``rclk=False``) freezes every register in
that domain, synchronizers included.
"""

import math
from typing import Any, NamedTuple, Optional

from .gray import binary_to_gray, gray_to_binary

__all__ = ["AsyncFIFO", "AsyncFIFOError", "Event", "EdgeRecord"]

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
    almost_full_before: bool
    almost_full_after: bool
    almost_empty_before: bool
    almost_empty_after: bool
    r2w_before: tuple
    w2r_before: tuple
    r2w_after: tuple
    w2r_after: tuple
    wdata: Any
    rdata: Any


class AsyncFIFO:
    """Event-driven asynchronous-FIFO pointer cross-domain checker.

    Parameters
        k: log2 depth; ``k in {1, 2, 3, 4, 5}`` (depth 2..32).
        almost_full_margin: write-side warning margin in free entries,
               ``0 <= m <= depth`` (``bool`` rejected).  The write-domain
               ``almost_full`` register asserts whenever the *guaranteed*
               number of free entries is ``<= m`` under the worst pointer
               lag still possible through the two-stage synchronizer.  ``0``
               reduces the warning to the exact (conservative) full case;
               ``depth`` warns permanently.
        almost_empty_margin: read-side warning margin in stored entries,
               ``0 <= m <= depth``.  The read-domain ``almost_empty``
               register asserts whenever the guaranteed number of stored
               entries is ``<= m``.  ``0`` reduces it to the conservative
               empty case; ``depth`` warns permanently.

    Usage
        >>> fifo = AsyncFIFO(k=3)
        >>> records = fifo.run([Event(0, True, False, wen=True, wdata=10)])
    """

    def __init__(
        self,
        k: int,
        almost_full_margin: int = 1,
        almost_empty_margin: int = 1,
    ) -> None:
        self._k = _strict_int(k, "k", lo=1, hi=5)
        self._depth = 1 << k
        self._afull_margin = _strict_int(
            almost_full_margin, "almost_full_margin",
            lo=0, hi=self._depth,
        )
        self._aempty_margin = _strict_int(
            almost_empty_margin, "almost_empty_margin",
            lo=0, hi=self._depth,
        )
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
        # Almost-full is a write-domain register, almost-empty a read-domain
        # register.  At reset every pointer/synchronizer value is 0: the read
        # side can guarantee zero stored entries (almost_empty necessarily
        # on), while the write side sees the whole depth free; almost_full is
        # on only when the user margin covers the whole depth.
        self._almost_full = self._afull_margin >= self._depth
        self._almost_empty = True
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
    def almost_full(self) -> bool:
        """Write-domain almost-full warning (register output).

        Asserted when the free space that can be *guaranteed* from the local
        write pointer and the stale synchronized read pointer is no greater
        than ``almost_full_margin``.
        """
        return self._almost_full

    @property
    def almost_empty(self) -> bool:
        """Read-domain almost-empty warning (register output).

        Asserted when the number of stored entries that can be *guaranteed*
        from the local read pointer and the stale synchronized write pointer
        is no greater than ``almost_empty_margin``.
        """
        return self._almost_empty

    @property
    def almost_full_margin(self) -> int:
        return self._afull_margin

    @property
    def almost_empty_margin(self) -> int:
        return self._aempty_margin

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
            "almost_full": self._almost_full,
            "almost_empty": self._almost_empty,
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
        afull_o, aempty_o = self._almost_full, self._almost_empty
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

        # ---- almost-full / almost-empty: conservative occupancy bounds.
        #
        # The two synchronizer stages may hold two different legal Gray
        # values simultaneously.  The stage-1 value is one edge newer than
        # stage-2, but neither can be trusted as "current": the opposite
        # pointer may already have advanced one more step by this edge.  So
        # the warning uses the *least favorable* stage still in flight:
        #
        #   write side: smallest synced read ptr  -> largest possible fill
        #   read  side: largest  synced write ptr -> smallest possible fill
        #
        # Like full/empty, the local next pointer after a possible accepted
        # access and the OLD synchronizer outputs are used, so a coincident
        # read+write edge leaves both domains sampling pre-edge state.
        afull_n = afull_o
        if event.wclk:
            # r2w_s2 is the OLDEST confirmed read pointer still in the
            # synchronizer, hence the earliest position reads could be
            # guaranteed to have reached; it bounds fill from above.
            occ_max = (wptr_n - gray_to_binary(r2w_s2_o)) & (
                2 * self._depth - 1
            )
            free_guaranteed = self._depth - min(occ_max, self._depth)
            afull_n = free_guaranteed <= self._afull_margin
        aempty_n = aempty_o
        if event.rclk:
            # w2r_s1 is the NEWEST confirmed write pointer; entries up to it
            # are guaranteed to exist and bound the fill from below.
            occ_min = (gray_to_binary(w2r_s1_o) - rptr_n) & (
                2 * self._depth - 1
            )
            entries_guaranteed = min(occ_min, self._depth)
            aempty_n = entries_guaranteed <= self._aempty_margin

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
        self._almost_full = afull_n
        self._almost_empty = aempty_n
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
            almost_full_before=afull_o,
            almost_full_after=afull_n,
            almost_empty_before=aempty_o,
            almost_empty_after=aempty_n,
            r2w_before=(r2w_s1_o, r2w_s2_o),
            w2r_before=(w2r_s1_o, w2r_s2_o),
            r2w_after=(r2w_s1_n, r2w_s2_n),
            w2r_after=(w2r_s1_n, w2r_s2_n),
            wdata=event.wdata,
            rdata=rdata,
        )
        self._steps.append(record)
        return record
