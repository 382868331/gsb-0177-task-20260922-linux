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

Per-domain reset and generation handshaking
-------------------------------------------

Either domain may be reset independently at any step (``wrst`` / ``rrst``;
asynchronous, so a clock edge is not even required).  Reset is processed
*before* edge sampling: the partner domain's synchronizer input stage samples
the post-reset Gray 0 / new generation when its clock edge coincides with the
reset.  At a step in which either reset is asserted neither direction passes
data; a reset clears the memory and the accepted-access counters, advances the
resetting domain's *generation* token, forces that domain's pointer / flag /
synchronizers to their reset values, and restarts a bidirectional handshake:

    new generation --2 partner edges--> partner learns (s2) and zeroes its
        own pointer/flag, records an acknowledgement --2 local edges-->
        the resetting domain sees the acknowledgement; once BOTH directions
        are acknowledged (rack2w_s2 == w_gen and wack2r_s2 == r_gen) the
        FIFO is ``synced`` again and accepts/rejects on flag merit alone.

A second reset while resynchronization is pending simply starts a new
generation; stale in-flight tokens age out of the synchronizer pipelines.

Disabling a clock (``wclk=False`` / ``rclk=False``) freezes every register in
that domain, synchronizers included.
"""

import math
from typing import Any, NamedTuple, Optional

from .gray import binary_to_gray

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
               ``True`` unless a reset (``wrst`` / ``rrst``) is asserted.
        wen:   write enable evaluated at this step.
        ren:   read enable evaluated at this step.
        wrst:  asynchronous write-domain reset at this step (active high).
        rrst:  asynchronous read-domain reset at this step (active high).
        wdata: payload stored on an accepted write.  Must be ``None`` when
               ``wen`` is ``False``; a float payload must be finite.

    Reset is processed before edge sampling.  On a reset step neither
    direction passes data regardless of the enables.

    A plain (non-tuple) class because Python 3.14 forbids overriding
    ``__new__`` on ``typing.NamedTuple``; equality and unpacking are supported
    so instances behave like immutable value records.
    """

    __slots__ = ("time", "wclk", "rclk", "wen", "ren", "wrst", "rrst",
                 "wdata")

    def __init__(
        self,
        time: int,
        wclk: bool,
        rclk: bool,
        wen: bool = False,
        ren: bool = False,
        wdata: Any = None,
        *,
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
        if not (we or re or write_rst or read_rst):
            raise AsyncFIFOError(
                f"event at time={t} has neither wclk, rclk, wrst nor rrst; "
                "at least one clock edge or one reset is required"
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
        self.wrst = write_rst
        self.rrst = read_rst
        self.wdata = wdata

    def __iter__(self):
        return iter((self.time, self.wclk, self.rclk,
                     self.wen, self.ren, self.wdata))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return (tuple(self) == tuple(other)
                and self.wrst == other.wrst
                and self.rrst == other.rrst)

    def __repr__(self) -> str:
        rst = ""
        if self.wrst:
            rst += ", wrst=True"
        if self.rrst:
            rst += ", rrst=True"
        return (f"Event(time={self.time}, wclk={self.wclk}, "
                f"rclk={self.rclk}, wen={self.wen}, ren={self.ren}, "
                f"wdata={self.wdata!r}{rst})")


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
    # --- per-domain reset / generation handshake (appended; old fields
    # keep their positions) ---
    wrst: bool = False
    rrst: bool = False
    wgen_before: int = 0
    wgen_after: int = 0
    rgen_before: int = 0
    rgen_after: int = 0
    # write generation synchronized INTO the read domain (stage1, stage2)
    wgen2r_before: tuple = (0, 0)
    wgen2r_after: tuple = (0, 0)
    # read generation synchronized INTO the write domain (stage1, stage2)
    rgen2w_before: tuple = (0, 0)
    rgen2w_after: tuple = (0, 0)
    # synchronized read-domain acknowledgement of the write generation:
    # (stage1, stage2) living in the write domain
    rack2w_before: tuple = (0, 0)
    rack2w_after: tuple = (0, 0)
    # synchronized write-domain acknowledgement of the read generation:
    # (stage1, stage2) living in the read domain
    wack2r_before: tuple = (0, 0)
    wack2r_after: tuple = (0, 0)
    # locally latched acknowledgement registers
    wack_reg_before: int = 0
    wack_reg_after: int = 0
    rack_reg_before: int = 0
    rack_reg_after: int = 0
    synced_before: bool = True
    synced_after: bool = True


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
        """Reset BOTH domains: return every register to its power-on value.

        Mirrors simultaneous ``wrst``/``rrst`` at construction time: both
        generation counters restart at 0, both handshake directions are
        pre-acknowledged (the FIFO is ``synced`` immediately), the memory is
        cleared, counters and trace are cleared.
        """
        self._wptr = 0
        self._rptr = 0
        self._full = False
        self._empty = True
        self._r2w_s1 = 0  # synchronized read Gray, stage 1 (write domain)
        self._r2w_s2 = 0  # synchronized read Gray, stage 2 (write domain)
        self._w2r_s1 = 0  # synchronized write Gray, stage 1 (read domain)
        self._w2r_s2 = 0  # synchronized write Gray, stage 2 (read domain)
        # generation handshake state (see module docstring)
        self._w_gen = 0          # live write-domain generation
        self._r_gen = 0          # live read-domain generation
        self._wgen2r_s1 = 0      # w_gen -> read domain, stage 1
        self._wgen2r_s2 = 0      # w_gen -> read domain, stage 2
        self._rgen2w_s1 = 0      # r_gen -> write domain, stage 1
        self._rgen2w_s2 = 0      # r_gen -> write domain, stage 2
        self._rack_reg = 0       # latest w_gen learned in the read domain
        self._wack_reg = 0       # latest r_gen learned in the write domain
        self._rack2w_s1 = 0      # read ack -> write domain, stage 1
        self._rack2w_s2 = 0      # read ack -> write domain, stage 2
        self._wack2r_s1 = 0      # write ack -> read domain, stage 1
        self._wack2r_s2 = 0      # write ack -> read domain, stage 2
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

    @property
    def w_generation(self) -> int:
        """Write-domain generation token; each write reset increments it."""
        return self._w_gen

    @property
    def r_generation(self) -> int:
        """Read-domain generation token; each read reset increments it."""
        return self._r_gen

    @property
    def synced(self) -> bool:
        """Both handshake directions acknowledged at the end of last step.

        ``False`` while the bidirectional generation handshake started by a
        per-domain reset is still in flight; data cannot pass in either
        direction until it becomes ``True`` again.
        """
        return self._rack2w_s2 == self._w_gen and \
            self._wack2r_s2 == self._r_gen

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
            "w_generation": self._w_gen,
            "r_generation": self._r_gen,
            "wgen2r": (self._wgen2r_s1, self._wgen2r_s2),
            "rgen2w": (self._rgen2w_s1, self._rgen2w_s2),
            "rack2w": (self._rack2w_s1, self._rack2w_s2),
            "wack2r": (self._wack2r_s1, self._wack2r_s2),
            "wack_reg": self._wack_reg,
            "rack_reg": self._rack_reg,
            "synced": self.synced,
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
        # ---- old register outputs (end of the previous step)
        wptr_o, rptr_o = self._wptr, self._rptr
        full_o, empty_o = self._full, self._empty
        r2w_s1_o, r2w_s2_o = self._r2w_s1, self._r2w_s2
        w2r_s1_o, w2r_s2_o = self._w2r_s1, self._w2r_s2
        wgen_o, rgen_o = self._w_gen, self._r_gen
        wgen2r_s1_o, wgen2r_s2_o = self._wgen2r_s1, self._wgen2r_s2
        rgen2w_s1_o, rgen2w_s2_o = self._rgen2w_s1, self._rgen2w_s2
        rack_reg_o, wack_reg_o = self._rack_reg, self._wack_reg
        rack2w_s1_o, rack2w_s2_o = self._rack2w_s1, self._rack2w_s2
        wack2r_s1_o, wack2r_s2_o = self._wack2r_s1, self._wack2r_s2
        wgen_pre_reset, rgen_pre_reset = wgen_o, rgen_o
        synced_pre = (rack2w_s2_o == wgen_o and wack2r_s2_o == rgen_o)

        # Trace ``*_before`` fields describe the registers as they were at the
        # end of the previous step -- i.e. BEFORE this step's asynchronous
        # reset -- while the edge logic below samples post-reset values.
        before = {
            "wptr": wptr_o, "rptr": rptr_o,
            "wgray": binary_to_gray(wptr_o),
            "rgray": binary_to_gray(rptr_o),
            "full": full_o, "empty": empty_o,
            "r2w": (r2w_s1_o, r2w_s2_o),
            "w2r": (w2r_s1_o, w2r_s2_o),
            "wgen2r": (wgen2r_s1_o, wgen2r_s2_o),
            "rgen2w": (rgen2w_s1_o, rgen2w_s2_o),
            "rack2w": (rack2w_s1_o, rack2w_s2_o),
            "wack2r": (wack2r_s1_o, wack2r_s2_o),
            "wack_reg": wack_reg_o, "rack_reg": rack_reg_o,
            "synced": synced_pre,
        }

        # ---- asynchronous resets FIRST (reset before edge sampling).
        # A coincident partner edge therefore samples post-reset Gray 0 and
        # the new generation via its synchronizer input stages.
        wrst, rrst = event.wrst, event.rrst
        if wrst:
            wptr_o, full_o = 0, False
            r2w_s1_o = r2w_s2_o = 0
            rgen2w_s1_o = rgen2w_s2_o = 0
            rack2w_s1_o = rack2w_s2_o = 0
            wack_reg_o = 0
            wgen_o += 1
        if rrst:
            rptr_o, empty_o = 0, True
            w2r_s1_o = w2r_s2_o = 0
            wgen2r_s1_o = wgen2r_s2_o = 0
            wack2r_s1_o = wack2r_s2_o = 0
            rack_reg_o = 0
            rgen_o += 1
        wgray_o = binary_to_gray(wptr_o)
        rgray_o = binary_to_gray(rptr_o)
        synced_o = (rack2w_s2_o == wgen_o and wack2r_s2_o == rgen_o)

        # Reset clears stale data; nothing passes in either direction on a
        # reset step, nor while the bidirectional handshake is unacknowledged.
        if wrst or rrst:
            self._mem = [None] * self._depth
            self._accepted_writes = 0
            self._accepted_reads = 0
        pass_ok = not (wrst or rrst) and synced_o

        # ---- accept decisions use the sampled (post-reset) old flags
        w_accept = bool(pass_ok and event.wen and not full_o) \
            if event.wclk else None
        r_accept = bool(pass_ok and event.ren and not empty_o) \
            if event.rclk else None

        wptr_n, rptr_n = wptr_o, rptr_o
        full_n, empty_n = full_o, empty_o
        r2w_s1_n, r2w_s2_n = r2w_s1_o, r2w_s2_o
        w2r_s1_n, w2r_s2_n = w2r_s1_o, w2r_s2_o
        wgen2r_s1_n, wgen2r_s2_n = wgen2r_s1_o, wgen2r_s2_o
        rgen2w_s1_n, rgen2w_s2_n = rgen2w_s1_o, rgen2w_s2_o
        rack_reg_n, wack_reg_n = rack_reg_o, wack_reg_o
        rack2w_s1_n, rack2w_s2_n = rack2w_s1_o, rack2w_s2_o
        wack2r_s1_n, wack2r_s2_n = wack2r_s1_o, wack2r_s2_o

        # ---- data memory (addressed; never used to derive flags)
        rdata = None
        if w_accept:
            self._mem[wptr_o & self._addr_mask] = event.wdata
            self._accepted_writes += 1
            wptr_n = (wptr_o + 1) & self._mask
        if r_accept:
            rdata = self._mem[rptr_o & self._addr_mask]
            self._accepted_reads += 1
            rptr_n = (rptr_o + 1) & self._mask

        # ---------------- write-domain clocked registers
        if event.wclk and not wrst:
            # pointer Gray and read generation cross into the write domain;
            # input stages sample the (post-reset) old values.
            r2w_s1_n, r2w_s2_n = rgray_o, r2w_s1_o
            rgen2w_s1_n, rgen2w_s2_n = rgen_o, rgen2w_s1_o
            rack2w_s1_n, rack2w_s2_n = rack_reg_o, rack2w_s1_o
            if rgen2w_s2_o != wack_reg_o:
                # first edge at which a new read generation reaches s2: the
                # FIFO restarts empty, so the write view restarts at ptr 0
                wack_reg_n = rgen2w_s2_o
                wptr_n = 0
                full_n = False
            elif synced_o:
                wgray_next = binary_to_gray(wptr_n)
                full_n = wgray_next == (r2w_s2_o ^ self._full_mask)
            else:
                # handshake in flight (including the edge at which it is
                # about to complete): hold the safe reset value so the first
                # post-sync edge cannot see a stale "full"
                full_n = False

        # ---------------- read-domain clocked registers
        if event.rclk and not rrst:
            w2r_s1_n, w2r_s2_n = wgray_o, w2r_s1_o
            wgen2r_s1_n, wgen2r_s2_n = wgen_o, wgen2r_s1_o
            wack2r_s1_n, wack2r_s2_n = wack_reg_o, wack2r_s1_o
            if wgen2r_s2_o != rack_reg_o:
                rack_reg_n = wgen2r_s2_o
                rptr_n = 0
                empty_n = True
            elif synced_o:
                rgray_next = binary_to_gray(rptr_n)
                empty_n = rgray_next == w2r_s2_o
            else:
                # hold "empty" through the resync window and its release edge
                empty_n = True

        wgray_n = binary_to_gray(wptr_n)
        rgray_n = binary_to_gray(rptr_n)
        synced_n = (rack2w_s2_n == wgen_o and wack2r_s2_n == rgen_o)

        # ---- commit
        self._wptr, self._rptr = wptr_n, rptr_n
        self._full, self._empty = full_n, empty_n
        self._r2w_s1, self._r2w_s2 = r2w_s1_n, r2w_s2_n
        self._w2r_s1, self._w2r_s2 = w2r_s1_n, w2r_s2_n
        self._w_gen, self._r_gen = wgen_o, rgen_o
        self._wgen2r_s1, self._wgen2r_s2 = wgen2r_s1_n, wgen2r_s2_n
        self._rgen2w_s1, self._rgen2w_s2 = rgen2w_s1_n, rgen2w_s2_n
        self._rack_reg, self._wack_reg = rack_reg_n, wack_reg_n
        self._rack2w_s1, self._rack2w_s2 = rack2w_s1_n, rack2w_s2_n
        self._wack2r_s1, self._wack2r_s2 = wack2r_s1_n, wack2r_s2_n
        self._last_time = event.time

        record = EdgeRecord(
            time=event.time,
            wclk=event.wclk,
            rclk=event.rclk,
            wen=event.wen,
            ren=event.ren,
            w_accept=w_accept,
            r_accept=r_accept,
            wptr_before=before["wptr"],
            wptr_after=wptr_n,
            rptr_before=before["rptr"],
            rptr_after=rptr_n,
            wgray_before=before["wgray"],
            wgray_after=wgray_n,
            rgray_before=before["rgray"],
            rgray_after=rgray_n,
            full_before=before["full"],
            full_after=full_n,
            empty_before=before["empty"],
            empty_after=empty_n,
            r2w_before=before["r2w"],
            w2r_before=before["w2r"],
            r2w_after=(r2w_s1_n, r2w_s2_n),
            w2r_after=(w2r_s1_n, w2r_s2_n),
            wdata=event.wdata,
            rdata=rdata,
            wrst=wrst,
            rrst=rrst,
            wgen_before=wgen_pre_reset,
            wgen_after=wgen_o,
            rgen_before=rgen_pre_reset,
            rgen_after=rgen_o,
            wgen2r_before=before["wgen2r"],
            rgen2w_before=before["rgen2w"],
            wgen2r_after=(wgen2r_s1_n, wgen2r_s2_n),
            rgen2w_after=(rgen2w_s1_n, rgen2w_s2_n),
            rack2w_before=before["rack2w"],
            wack2r_before=before["wack2r"],
            rack2w_after=(rack2w_s1_n, rack2w_s2_n),
            wack2r_after=(wack2r_s1_n, wack2r_s2_n),
            wack_reg_before=before["wack_reg"],
            wack_reg_after=wack_reg_n,
            rack_reg_before=before["rack_reg"],
            rack_reg_after=rack_reg_n,
            synced_before=before["synced"],
            synced_after=synced_n,
        )
        self._steps.append(record)
        return record
