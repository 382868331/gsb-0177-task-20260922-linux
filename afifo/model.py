"""Discrete-event model of an asynchronous FIFO pointer-crossing checker.

Timing model (one :class:`Event` = one point on an integer time axis)
--------------------------------------------------------------------

* The write pointer (``wptr``) and read pointer (``rptr``) are each ``k+1``
  bits wide: k address bits plus one wrap bit.
* Only Gray-coded values leave a clock domain.  Each destination domain
  carries a two-stage synchronizer (``s1``/``s2``) for every crossing:
  pointer Grays drive the flags, and *generation* Grays drive the reset
  handshake.  Flag logic sees the old ``s2`` output.  When a read and a write
  edge coincide, *both* sides sample the register values as they were before
  that time step -- including the input stage of each synchronizer -- except
  when an explicit reset is asserted at that step: resets are applied *before*
  any sampling, so the other edge latches the post-reset (zeroed) value.
* ``full`` is a write-domain register (reset ``False``); ``empty`` is a
  read-domain register (reset ``True``).  On a clock edge the old flag
  decides whether the enable is accepted; afterwards the flag is recomputed
  from the local *next* pointer Gray and the *old* synchronized value:

      empty_next = gray(rptr_next) == w2r_s2_old
      full_next  = gray(wptr_next) == (r2w_s2_old ^ TOP_TWO_BITS)

* A write is accepted iff ``wclk and wen and not full_old and both domains
  are generation-confirmed and neither domain is resetting at this step``;
  a read is accepted iff ``rclk and ren and not empty_old and the same
  handshake gate is closed (open)``.  Acceptance never consults a global
  queue length -- the data array is addressed memory, not a flag.

Independent domain reset and generation handshake
-------------------------------------------------

``Event(..., wrst=True)`` / ``rrst=True`` pulses a write-/read-domain reset
synchronously to that domain's clock edge (the matching clock must edge at
that step).  Each pulse:

1. is applied *first* at a coincident step: local pointer -> 0, local flag ->
   its reset value, every synchronizer register physically in that domain
   (pointer stages *and* generation stages) -> gray(0), the data memory is
   invalidated (write-side reset clears it directly) and the accepted-access
   counters return to 0;
2. bumps that domain's *generation* counter (power-on generation is 0);
3. crosses the new generation Gray through the ordinary two-stage synchronizer
   of the peer.  When the peer's second stage decodes a generation *newer*
   than its own, it performs the same reset actions locally, adopts that
   generation, and echoes it back through its own two-stage synchronizer;
4. a domain is ``ready`` (generation-confirmed) only when its local
   generation equals the Gray-decoded second stage coming from the peer.
   No write or read is accepted in *either* domain from the reset step until
   both domains are ready again -- this is the checker-level statement of
   the hardware reset protocol that keeps enables de-asserted for the whole
   re-synchronization sequence, including the two-edge window in which the
   peer has not yet observed the reset.

A second reset pulse before the handshake finishes just bumps the generation
again: stale in-flight acknowledgments decode to an older generation and can
never re-open the gate.  Because generations are monotonic and only a *newer*
synced generation is adopted, repeated resets always converge on the newest
one.  Generation counters are plain unbounded Python ints; adjacent
generations still differ in one Gray bit.

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
        wrst:  write-domain reset pulse at this step.  Requires
               ``wclk is True``; applied before the edge samples anything, so
               a coincident read edge latches the post-reset write domain.
               Bumps the write generation.  ``wen`` may stay ``True``; it is
               rejected at the reset step regardless.
        rrst:  read-domain reset pulse; symmetric to ``wrst``.

    A plain (non-tuple) class because Python 3.14 forbids overriding
    ``__new__`` on ``typing.NamedTuple``; equality and unpacking are supported
    so instances behave like immutable value records.
    """

    __slots__ = ("time", "wclk", "rclk", "wen", "ren", "wdata", "wrst", "rrst")

    def __init__(
        self,
        time: int,
        wclk: bool,
        rclk: bool,
        wen: bool = False,
        ren: bool = False,
        wdata: Any = None,
        wrst: bool = False,
        rrst: bool = False,
    ) -> None:
        t = _strict_int(time, "time", lo=0, hi=_TIME_MAX)
        we = _strict_bool(wclk, "wclk")
        re = _strict_bool(rclk, "rclk")
        write_en = _strict_bool(wen, "wen")
        read_en = _strict_bool(ren, "ren")
        write_rst = _strict_bool(wrst, "wrst")
        read_rst = _strict_bool(rrst, "rrst")
        if not (we or re):
            raise AsyncFIFOError(
                f"event at time={t} has neither wclk nor rclk; at least one "
                "clock edge is required"
            )
        if write_rst and not we:
            raise AsyncFIFOError(
                f"event at time={t}: wrst requires a write clock edge "
                "(wclk must be True); a reset without its clock cannot take "
                "effect"
            )
        if read_rst and not re:
            raise AsyncFIFOError(
                f"event at time={t}: rrst requires a read clock edge "
                "(rclk must be True); a reset without its clock cannot take "
                "effect"
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
        self.wrst = write_rst
        self.rrst = read_rst

    def __iter__(self):
        return iter((self.time, self.wclk, self.rclk,
                     self.wen, self.ren, self.wdata,
                     self.wrst, self.rrst))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return tuple(self) == tuple(other)

    def __repr__(self) -> str:
        return (f"Event(time={self.time}, wclk={self.wclk}, "
                f"rclk={self.rclk}, wen={self.wen}, ren={self.ren}, "
                f"wdata={self.wdata!r}, wrst={self.wrst}, rrst={self.rrst})")


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
    wrst: bool
    rrst: bool
    wgen_before: int
    wgen_after: int
    rgen_before: int
    rgen_after: int
    rgen2w_before: tuple
    wgen2r_before: tuple
    rgen2w_after: tuple
    wgen2r_after: tuple
    wready_before: bool
    rready_before: bool
    wready_after: bool
    rready_after: bool


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
        """Return every register to its power-on value (clears the trace).

        This is the coarse *simulation* reset: both generations return to 0,
        both pointer/generation synchronizer pairs clear, the memory empties
        and all statistics counters zero.  In-stream per-domain resets are
        requested with ``Event(wrst=True)`` / ``Event(rrst=True)`` and do not
        clear the trace.
        """
        self._wptr = 0
        self._rptr = 0
        self._full = False
        self._empty = True
        self._r2w_s1 = 0  # synchronized read pointer Gray, stage 1 (write domain)
        self._r2w_s2 = 0  # synchronized read pointer Gray, stage 2 (write domain)
        self._w2r_s1 = 0  # synchronized write pointer Gray, stage 1 (read domain)
        self._w2r_s2 = 0  # synchronized write pointer Gray, stage 2 (read domain)
        self._wgen = 0    # write-domain generation (power-on generation 0)
        self._rgen = 0    # read-domain generation (power-on generation 0)
        self._rgen2w_s1 = 0  # synchronized read generation Gray, stage 1
        self._rgen2w_s2 = 0  # synchronized read generation Gray, stage 2
        self._wgen2r_s1 = 0  # synchronized write generation Gray, stage 1
        self._wgen2r_s2 = 0  # synchronized write generation Gray, stage 2
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
    def wgen(self) -> int:
        """Current write-domain generation (0 at power-on, bumps on reset)."""
        return self._wgen

    @property
    def rgen(self) -> int:
        """Current read-domain generation (0 at power-on, bumps on reset)."""
        return self._rgen

    @property
    def wready(self) -> bool:
        """Write domain is generation-confirmed with the peer (may transfer)."""
        return self._wgen == gray_to_binary(self._rgen2w_s2)

    @property
    def rready(self) -> bool:
        """Read domain is generation-confirmed with the peer (may transfer)."""
        return self._rgen == gray_to_binary(self._wgen2r_s2)

    @property
    def synchronized(self) -> bool:
        """Both domains confirmed with each other (transfer gate open)."""
        return self.wready and self.rready

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
            "wgen": self._wgen,
            "rgen": self._rgen,
            "rgen2w": (self._rgen2w_s1, self._rgen2w_s2),
            "wgen2r": (self._wgen2r_s1, self._wgen2r_s2),
            "wready": self.wready,
            "rready": self.rready,
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
        # ---- old state as sampled at the edge (before this step's resets)
        wptr_o, rptr_o = self._wptr, self._rptr
        full_o, empty_o = self._full, self._empty
        r2w_s1_o, r2w_s2_o = self._r2w_s1, self._r2w_s2
        w2r_s1_o, w2r_s2_o = self._w2r_s1, self._w2r_s2
        wgen_o, rgen_o = self._wgen, self._rgen
        rgen2w_s1_o, rgen2w_s2_o = self._rgen2w_s1, self._rgen2w_s2
        wgen2r_s1_o, wgen2r_s2_o = self._wgen2r_s1, self._wgen2r_s2
        wgray_o = binary_to_gray(wptr_o)
        rgray_o = binary_to_gray(rptr_o)
        wready_o = wgen_o == gray_to_binary(rgen2w_s2_o)
        rready_o = rgen_o == gray_to_binary(wgen2r_s2_o)

        # ============================================================
        # Phase 1: explicit, clock-synchronous local resets come FIRST.
        # "same-time events: reset before sampling": a coincident peer edge
        # below therefore latches the post-reset (zeroed) domain.
        # ============================================================
        if event.wrst:
            self._wptr = 0
            self._full = False
            self._r2w_s1 = self._r2w_s2 = 0
            self._rgen2w_s1 = self._rgen2w_s2 = 0
            self._wgen += 1
        if event.rrst:
            self._rptr = 0
            self._empty = True
            self._w2r_s1 = self._w2r_s2 = 0
            self._wgen2r_s1 = self._wgen2r_s2 = 0
            self._rgen += 1
        if event.wrst or event.rrst:
            for i in range(self._depth):
                self._mem[i] = None
            self._accepted_writes = 0
            self._accepted_reads = 0

        # live (post-explicit-reset) local values used by the edge logic
        wptr_l = self._wptr
        rptr_l = self._rptr
        full_l = self._full
        empty_l = self._empty
        wgen_l = self._wgen
        rgen_l = self._rgen
        rgen2w_s1_l, rgen2w_s2_l = self._rgen2w_s1, self._rgen2w_s2
        wgen2r_s1_l, wgen2r_s2_l = self._wgen2r_s1, self._wgen2r_s2
        r2w_s1_l, r2w_s2_l = self._r2w_s1, self._r2w_s2
        w2r_s1_l, w2r_s2_l = self._w2r_s1, self._w2r_s2

        # ============================================================
        # Phase 2: transfer gate.  Both domains must be generation-confirmed
        # and neither may be resetting at this step.  This closes the gate
        # from the very reset step through the full two-way handshake, even
        # on the domain that has not yet observed the peer's reset.
        # ============================================================
        w_ready = wgen_l == gray_to_binary(rgen2w_s2_l)
        r_ready = rgen_l == gray_to_binary(wgen2r_s2_l)
        gate_open = w_ready and r_ready and not event.wrst and not event.rrst

        w_accept = (bool(event.wen and not full_l) and gate_open
                    if event.wclk else None)
        r_accept = (bool(event.ren and not empty_l) and gate_open
                    if event.rclk else None)

        # ---- local pointers advance only on accepted accesses
        wptr_n = (wptr_l + (1 if w_accept else 0)) & self._mask
        rptr_n = (rptr_l + (1 if r_accept else 0)) & self._mask
        wgray_n = binary_to_gray(wptr_n)
        rgray_n = binary_to_gray(rptr_n)

        # ---- data memory (addressed; never used to derive flags)
        rdata = None
        if w_accept:
            self._mem[wptr_l & self._addr_mask] = event.wdata
            self._accepted_writes += 1
        if r_accept:
            rdata = self._mem[rptr_l & self._addr_mask]
            self._accepted_reads += 1

        # ============================================================
        # Phase 3: synchronizer shifts, with peer-triggered reactive resets.
        # At an edge WITHOUT an explicit reset the domain shifts both chains
        # (pointer Gray and generation Gray).  If the freshly shifted second
        # generation stage decodes to a generation NEWER than its own, the
        # domain adopts it and performs its reset actions; the generation
        # chain is retained (it carries the adopted generation), while the
        # pointer chain is cleared.  Only strictly newer generations are
        # adopted, so stale/echoed old codes can never roll a domain back.
        # ============================================================
        w_reactive = False
        r_reactive = False

        if event.wclk:
            if event.wrst:
                # synchronous reset edge: both chains stay cleared
                r2w_s1_n = r2w_s2_n = 0
                rgen2w_s1_n = rgen2w_s2_n = 0
            else:
                # input stages sample the peer's values AFTER explicit resets
                # but BEFORE any reactive reset produced at this same edge
                r2w_s1_n = binary_to_gray(rptr_l)
                r2w_s2_n = r2w_s1_l
                rgen2w_s1_n = binary_to_gray(rgen_l)
                rgen2w_s2_n = rgen2w_s1_l
                peer_gen = gray_to_binary(rgen2w_s2_n)
                if peer_gen > wgen_l:
                    # peer's reset observed: adopt generation, reset locally
                    w_reactive = True
                    self._wgen = peer_gen
                    wptr_n = 0
                    wgray_n = binary_to_gray(0)
                    full_l = False
                    r2w_s1_n = r2w_s2_n = 0
                    for i in range(self._depth):
                        self._mem[i] = None
                    self._accepted_writes = 0
                    self._accepted_reads = 0
        else:
            r2w_s1_n, r2w_s2_n = r2w_s1_l, r2w_s2_l
            rgen2w_s1_n, rgen2w_s2_n = rgen2w_s1_l, rgen2w_s2_l

        if event.rclk:
            if event.rrst:
                w2r_s1_n = w2r_s2_n = 0
                wgen2r_s1_n = wgen2r_s2_n = 0
            else:
                w2r_s1_n = binary_to_gray(wptr_l)
                w2r_s2_n = w2r_s1_l
                wgen2r_s1_n = binary_to_gray(wgen_l)
                wgen2r_s2_n = wgen2r_s1_l
                peer_gen = gray_to_binary(wgen2r_s2_n)
                if peer_gen > rgen_l:
                    r_reactive = True
                    self._rgen = peer_gen
                    rptr_n = 0
                    rgray_n = binary_to_gray(0)
                    empty_l = True
                    w2r_s1_n = w2r_s2_n = 0
                    for i in range(self._depth):
                        self._mem[i] = None
                    self._accepted_writes = 0
                    self._accepted_reads = 0
        else:
            w2r_s1_n, w2r_s2_n = w2r_s1_l, w2r_s2_l
            wgen2r_s1_n, wgen2r_s2_n = wgen2r_s1_l, wgen2r_s2_l

        # ---- flags: held at their reset values until the local generation
        # is confirmed by the peer.  The pointer and generation synchronizers
        # shift on the same edges, so an s2 generation/s2-pointer pair always
        # comes from one coherent peer state; evaluating the pointer equation
        # against an older generation's pair (which happens for an edge or
        # two around a reactive peer reset) could falsely deassert empty and
        # admit one stale read.  Once confirmed, the ordinary equation uses
        # the OLD synchronized pointer value, as without resets.
        if event.wclk:
            full_n = False if (event.wrst or w_reactive or not w_ready) else (
                wgray_n == (r2w_s2_l ^ self._full_mask))
        else:
            full_n = full_l
        if event.rclk:
            empty_n = True if (event.rrst or r_reactive or not r_ready) else (
                rgray_n == w2r_s2_l)
        else:
            empty_n = empty_l

        # ---- commit
        self._wptr, self._rptr = wptr_n, rptr_n
        self._full, self._empty = full_n, empty_n
        self._r2w_s1, self._r2w_s2 = r2w_s1_n, r2w_s2_n
        self._w2r_s1, self._w2r_s2 = w2r_s1_n, w2r_s2_n
        self._rgen2w_s1, self._rgen2w_s2 = rgen2w_s1_n, rgen2w_s2_n
        self._wgen2r_s1, self._wgen2r_s2 = wgen2r_s1_n, wgen2r_s2_n
        self._last_time = event.time

        wready_n = self._wgen == gray_to_binary(self._rgen2w_s2)
        rready_n = self._rgen == gray_to_binary(self._wgen2r_s2)

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
            wrst=event.wrst,
            rrst=event.rrst,
            wgen_before=wgen_o,
            wgen_after=self._wgen,
            rgen_before=rgen_o,
            rgen_after=self._rgen,
            rgen2w_before=(rgen2w_s1_o, rgen2w_s2_o),
            wgen2r_before=(wgen2r_s1_o, wgen2r_s2_o),
            rgen2w_after=(rgen2w_s1_n, rgen2w_s2_n),
            wgen2r_after=(wgen2r_s1_n, wgen2r_s2_n),
            wready_before=wready_o,
            rready_before=rready_o,
            wready_after=wready_n,
            rready_after=rready_n,
        )
        self._steps.append(record)
        return record
