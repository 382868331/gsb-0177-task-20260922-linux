"""Read-domain ready/valid output register wrapping :class:`AsyncFIFO`.

The downstream side of an async FIFO often speaks ready/valid instead of a
bare ``ren``.  This module wraps the *unmodified* :class:`~afifo.model.AsyncFIFO`
(write-domain protocol, Gray pointers and both two-stage synchronizers are
exactly the core's) with one read-domain output register:

* ``out_valid`` / ``out_data`` are read-domain registers.  ``reset()`` clears
  ``out_valid`` (and the delivered count); ``out_data`` is then ``None``.
* The wrapper -- not the caller -- drives the core's ``ren``:

      ren = (not out_valid_old) or ready

  i.e. a dequeue is requested only when the output register is empty or its
  current word is being consumed by ``ready`` on this same read edge.  The
  dequeue itself is still gated by the core's *old* read-domain ``empty``
  flag, so a write that has been accepted but has not yet crossed into the
  read domain is never prefetched -- the true queue length is never consulted.
* On a read edge, consuming the old word (``out_valid_old and ready``) and
  loading the next word (accepted dequeue) happen in the same cycle.  The
  delivered count increments only for ``out_valid_old and ready``.
* While ``out_valid and not ready`` the wrapper issues ``ren = 0``: the read
  pointer does not move and ``out_data`` is bit-stable.
* ``out_data`` only changes on an accepted dequeue load; when ``out_valid``
  falls the register keeps its last bits (don't-care while invalid).

Capacity accounting: the output register is *not* folded into ``depth``.
``depth`` stays the core FIFO depth (2**k); the wrapped system can hold at
most ``depth + 1`` accepted-but-undelivered words (one sitting in the output
register after a prefetch dequeue).
"""

from typing import Any, NamedTuple, Optional

import math

from .model import (
    _BATCH_MAX,
    _TIME_MAX,
    AsyncFIFO,
    AsyncFIFOError,
    EdgeRecord,
    Event,
    _strict_bool,
    _strict_int,
)

__all__ = ["ReadyValidFIFO", "RVEvent", "RVRecord"]


class RVEvent:
    """One time step for the ready/valid-wrapped FIFO.

    Same contract as :class:`afifo.Event` except the read side carries
    ``ready`` (downstream consume) instead of ``ren`` -- the wrapper computes
    the dequeue request itself from the output-register state.

    Fields
        time:  strictly increasing integer time stamp, ``0 <= t < 2**63``.
        wclk:  write-domain rising edge at this step.
        rclk:  read-domain rising edge at this step.  At least one of
               ``wclk``/``rclk`` must be ``True``.
        wen:   write enable evaluated at this step.
        ready: downstream ready, sampled on the read edge.  Meaningless (but
               still must be a bool) on steps without ``rclk``.
        wdata: payload stored on an accepted write; ``None`` when ``wen`` is
               ``False``; floats must be finite.
    """

    __slots__ = ("time", "wclk", "rclk", "wen", "ready", "wdata")

    def __init__(
        self,
        time: int,
        wclk: bool,
        rclk: bool,
        wen: bool = False,
        ready: bool = False,
        wdata: Any = None,
    ) -> None:
        t = _strict_int(time, "time", lo=0, hi=_TIME_MAX)
        wc = _strict_bool(wclk, "wclk")
        rc = _strict_bool(rclk, "rclk")
        we = _strict_bool(wen, "wen")
        rd = _strict_bool(ready, "ready")
        if not (wc or rc):
            raise AsyncFIFOError(
                f"event at time={t} has neither wclk nor rclk; at least one "
                "clock edge is required"
            )
        if not we and wdata is not None:
            raise AsyncFIFOError(
                f"event at time={t}: wdata must be None when wen is False"
            )
        if isinstance(wdata, float) and not math.isfinite(wdata):
            raise AsyncFIFOError(
                f"event at time={t}: float wdata must be finite; got "
                "non-finite value (NaN/Infinity is rejected)"
            )
        self.time = t
        self.wclk = wc
        self.rclk = rc
        self.wen = we
        self.ready = rd
        self.wdata = wdata

    def __iter__(self):
        return iter((self.time, self.wclk, self.rclk,
                     self.wen, self.ready, self.wdata))

    def __eq__(self, other) -> bool:
        if not isinstance(other, RVEvent):
            return NotImplemented
        return tuple(self) == tuple(other)

    def __repr__(self) -> str:
        return (f"RVEvent(time={self.time}, wclk={self.wclk}, "
                f"rclk={self.rclk}, wen={self.wen}, ready={self.ready}, "
                f"wdata={self.wdata!r})")


class RVRecord(NamedTuple):
    """Full trace of one wrapped-FIFO time step.

    ``ren_issued`` is the dequeue request the wrapper drove into the core
    (``None`` when the read domain did not clock).  ``delivered`` is
    ``out_valid_before and ready`` on a read edge -- the only condition that
    increments the delivered count.  ``inner`` is the core
    :class:`afifo.EdgeRecord` for the same step (pointers, flags,
    synchronizers, accepts).
    """

    time: int
    wclk: bool
    rclk: bool
    wen: bool
    ready: bool
    wdata: Any
    w_accept: Optional[bool]
    ren_issued: Optional[bool]
    r_accept: Optional[bool]
    delivered: Optional[bool]
    out_valid_before: bool
    out_valid_after: bool
    out_data_before: Any
    out_data_after: Any
    inner: EdgeRecord


class ReadyValidFIFO:
    """Async FIFO plus a read-domain ready/valid output register.

    Parameters
        k: log2 of the *core* FIFO depth; ``k in {1, 2, 3, 4, 5}``.  The
           output register is extra and is never counted in ``depth``.

    Usage
        >>> fifo = ReadyValidFIFO(k=2)
        >>> rec = fifo.run_one(RVEvent(1, True, False, wen=True, wdata=7))
    """

    def __init__(self, k: int) -> None:
        self._fifo = AsyncFIFO(k)
        self._out_valid = False
        self._out_data = None
        self._delivered = 0
        self._records = []
        self._last_time = -1

    def reset(self) -> None:
        """Power-on state: core reset, output register invalidated, trace cleared."""
        self._fifo.reset()
        self._out_valid = False
        self._out_data = None
        self._delivered = 0
        self._records = []
        self._last_time = -1

    # ------------------------------------------------------------------ props
    @property
    def k(self) -> int:
        return self._fifo.k

    @property
    def depth(self) -> int:
        """Core FIFO depth (2**k).  The output register is NOT included."""
        return self._fifo.depth

    @property
    def full(self) -> bool:
        """Write-domain full flag of the core (register output)."""
        return self._fifo.full

    @property
    def empty(self) -> bool:
        """Read-domain empty flag of the core (register output)."""
        return self._fifo.empty

    @property
    def out_valid(self) -> bool:
        """Read-domain output-register valid bit (reset ``False``)."""
        return self._out_valid

    @property
    def out_data(self) -> Any:
        """Output-register payload (meaningful only while ``out_valid``)."""
        return self._out_data

    @property
    def delivered(self) -> int:
        """Words consumed downstream (``out_valid_old and ready`` edges)."""
        return self._delivered

    @property
    def accepted_writes(self) -> int:
        return self._fifo.accepted_writes

    @property
    def accepted_reads(self) -> int:
        """Core dequeues (output-register loads), not downstream deliveries."""
        return self._fifo.accepted_reads

    @property
    def in_flight(self) -> int:
        """Accepted writes not yet delivered; bounded by ``depth + 1``."""
        return self._fifo.accepted_writes - self._delivered

    @property
    def steps(self) -> tuple:
        """Trace of all :class:`RVRecord` values produced so far."""
        return tuple(self._records)

    @property
    def inner(self) -> AsyncFIFO:
        """The wrapped core FIFO (for register inspection; do not drive it)."""
        return self._fifo

    def snapshot(self) -> dict:
        """Copy of every observable register, core plus output register."""
        snap = self._fifo.snapshot()
        snap.update({
            "out_valid": self._out_valid,
            "out_data": self._out_data,
            "delivered": self._delivered,
        })
        return snap

    # ------------------------------------------------------------------- run
    def run_one(self, event: RVEvent) -> RVRecord:
        """Validate and apply a single event; return its trace record."""
        if not isinstance(event, RVEvent):
            raise AsyncFIFOError(
                "run_one expects an RVEvent; got "
                f"{type(event).__name__}: {event!r}"
            )
        if event.time <= self._last_time:
            raise AsyncFIFOError(
                f"event time {event.time} must be strictly greater than the "
                f"last processed time {self._last_time}"
            )
        return self._step(event)

    def run(self, events) -> list:
        """Apply a batch of events atomically (same contract as the core)."""
        if isinstance(events, RVEvent):
            raise AsyncFIFOError(
                "run expects an iterable of RVEvent objects; got a single "
                "RVEvent (use run_one)"
            )
        try:
            batch = list(events)
        except TypeError as exc:
            raise AsyncFIFOError(
                "run expects an iterable of RVEvent objects"
            ) from exc
        if len(batch) > _BATCH_MAX:
            raise AsyncFIFOError(
                f"batch size {len(batch)} exceeds the {_BATCH_MAX} event limit"
            )
        prev = self._last_time
        for i, event in enumerate(batch):
            if not isinstance(event, RVEvent):
                raise AsyncFIFOError(
                    f"events[{i}] must be an RVEvent; got "
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
    def _step(self, event: RVEvent) -> RVRecord:
        out_valid_o, out_data_o = self._out_valid, self._out_data

        if event.rclk:
            # Dequeue only when the register is empty or is being consumed.
            ren = (not out_valid_o) or event.ready
            consume = out_valid_o and event.ready
        else:
            ren = False
            consume = False

        inner = self._fifo.run_one(Event(
            event.time, event.wclk, event.rclk,
            wen=event.wen, ren=ren, wdata=event.wdata,
        ))

        if event.rclk:
            if inner.r_accept:
                # Load (possibly the same edge the old word is consumed).
                out_valid_n, out_data_n = True, inner.rdata
            elif consume:
                out_valid_n, out_data_n = False, out_data_o
            else:
                out_valid_n, out_data_n = out_valid_o, out_data_o
            if consume:
                self._delivered += 1
            delivered = consume
            ren_issued = ren
        else:
            out_valid_n, out_data_n = out_valid_o, out_data_o
            delivered = None
            ren_issued = None

        self._out_valid, self._out_data = out_valid_n, out_data_n
        self._last_time = event.time

        record = RVRecord(
            time=event.time,
            wclk=event.wclk,
            rclk=event.rclk,
            wen=event.wen,
            ready=event.ready,
            wdata=event.wdata,
            w_accept=inner.w_accept,
            ren_issued=ren_issued,
            r_accept=inner.r_accept,
            delivered=delivered,
            out_valid_before=out_valid_o,
            out_valid_after=out_valid_n,
            out_data_before=out_data_o,
            out_data_after=out_data_n,
            inner=inner,
        )
        self._records.append(record)
        return record
