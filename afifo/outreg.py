"""Read-domain output register (ready/valid) wrapper around :class:`AsyncFIFO`.

The downstream side of an asynchronous FIFO is commonly expected to speak
ready/valid: data must be presented *before* it is consumed, and a consumer
that keeps ``ready`` low must see a stable word.  A bare FIFO cannot do that
(a read only happens on ``ren``), so this module wraps one read-domain
output register around :class:`~afifo.model.AsyncFIFO`:

* ``valid``/``out_data`` are read-domain registers; ``valid`` resets to
  ``False`` and the output word is held (don't-care) while invalid.
* On every read-clock edge the wrapper issues an inner dequeue request
  ``ren = (not valid_old) or ready`` -- the output register can accept a new
  word exactly when it is empty or its current word is being consumed this
  same edge.  The inner FIFO still accepts that request only when its old
  ``empty`` flag is deasserted, so a word that has been written but has not
  yet crossed the two-stage synchronizer is *never* prefetched on the basis
  of any true queue length.
* Consuming the old word and loading the next one complete in the same read
  edge: when ``valid_old and ready`` and the inner dequeue is accepted, the
  old word is delivered and the new word is loaded, ``valid`` stays high.
* When ``valid and not ready`` no dequeue is requested, the read pointer
  does not move, and ``out_data`` is bit-stable across the stall.
* A delivery (downstream handshake) is counted exactly when a read edge
  samples ``valid_old and ready`` -- never on a load.

The write-domain protocol (``wen``/``full``) and both two-stage Gray
synchronizers are the inner FIFO's own, untouched.  ``depth`` still means
the inner FIFO depth; the output register is one *extra* slot on top of it
(see README "容量口径"), never folded into the ``depth`` parameter.
"""

import math
from typing import Any, NamedTuple, Optional

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
    """One time step for the ready/valid wrapper.

    Fields
        time:  strictly increasing integer time stamp, ``0 <= t < 2**63``.
        wclk:  write-domain rising edge at this step.
        rclk:  read-domain rising edge at this step.  At least one must be
               ``True``.
        wen:   write enable evaluated at this step (write domain unchanged).
        ready: downstream ready level sampled on the read edge.  Ignored on
               steps without ``rclk``.
        wdata: payload stored on an accepted write; same contract as
               :class:`afifo.Event` (``None`` when ``wen`` is ``False``,
               floats must be finite).
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
        write_en = _strict_bool(wen, "wen")
        rdy = _strict_bool(ready, "ready")
        if not (wc or rc):
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
        self.wclk = wc
        self.rclk = rc
        self.wen = write_en
        self.ready = rdy
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
    """Full trace of one wrapper time step.

    ``ren`` is the dequeue request the wrapper issued to the inner FIFO
    (``None`` when the read domain did not clock).  ``load`` marks the read
    edge that moved an inner word into the output register; ``delivered``
    marks a downstream handshake (old ``valid`` and ``ready``).  Both are
    ``None`` on steps without ``rclk``.  ``inner`` is the wrapped FIFO's own
    :class:`EdgeRecord` for the same step.
    """

    time: int
    wclk: bool
    rclk: bool
    wen: bool
    ready: bool
    wdata: Any
    ren: Optional[bool]
    w_accept: Optional[bool]
    r_accept: Optional[bool]
    load: Optional[bool]
    delivered: Optional[bool]
    delivered_data: Any
    valid_before: bool
    valid_after: bool
    out_before: Any
    out_after: Any
    inner: EdgeRecord


class ReadyValidFIFO:
    """Asynchronous FIFO with a read-domain ready/valid output register.

    Parameters
        k: log2 of the *inner* FIFO depth; ``k in {1, 2, 3, 4, 5}`` exactly
           as for :class:`AsyncFIFO`.  The output register is one additional
           slot and is not counted in ``depth``.

    Usage
        >>> fifo = ReadyValidFIFO(k=2)
        >>> rec = fifo.run_one(RVEvent(0, True, True, wen=True, ready=False,
        ...                            wdata=1))
        >>> rec.w_accept, rec.valid_after
        (True, False)
    """

    def __init__(self, k: int) -> None:
        self._fifo = AsyncFIFO(k)
        self._valid = False
        self._out = None
        self._delivered = 0
        self._steps = []
        self._last_time = -1

    def reset(self) -> None:
        """Return every register (inner FIFO included) to its power-on value.

        Clears ``valid`` even when the output register was holding an
        undelivered word, and clears the trace.
        """
        self._fifo.reset()
        self._valid = False
        self._out = None
        self._delivered = 0
        self._steps = []
        self._last_time = -1

    # ------------------------------------------------------------------ props
    @property
    def k(self) -> int:
        return self._fifo.k

    @property
    def depth(self) -> int:
        """Inner FIFO depth (2**k).  The output register is NOT included."""
        return self._fifo.depth

    @property
    def valid(self) -> bool:
        """Output-register valid flag (read-domain register)."""
        return self._valid

    @property
    def out_data(self) -> Any:
        """Output-register word.  Don't-care (last loaded) while invalid."""
        return self._out

    @property
    def full(self) -> bool:
        """Inner write-domain full flag (register output)."""
        return self._fifo.full

    @property
    def empty(self) -> bool:
        """Inner read-domain empty flag (register output)."""
        return self._fifo.empty

    @property
    def wptr(self) -> int:
        return self._fifo.wptr

    @property
    def rptr(self) -> int:
        return self._fifo.rptr

    @property
    def steps(self) -> tuple:
        """Trace of all :class:`RVRecord` values produced so far."""
        return tuple(self._steps)

    @property
    def accepted_writes(self) -> int:
        return self._fifo.accepted_writes

    @property
    def accepted_reads(self) -> int:
        """Inner dequeues so far == words loaded into the output register."""
        return self._fifo.accepted_reads

    @property
    def delivered(self) -> int:
        """Downstream handshakes so far (read edges with old valid & ready)."""
        return self._delivered

    @property
    def inflight(self) -> int:
        """Accepted writes not yet delivered downstream.

        Equals inner occupancy (``accepted_writes - accepted_reads``) plus
        the output-register slot (``1`` while ``valid``); bounded by
        ``depth + 1``.
        """
        return self._fifo.accepted_writes - self._delivered

    @property
    def inner(self) -> AsyncFIFO:
        """The wrapped FIFO (read-only inspection; do not drive it)."""
        return self._fifo

    def snapshot(self) -> dict:
        """Return a copy of every observable register (for inspection)."""
        snap = self._fifo.snapshot()
        snap.update({
            "valid": self._valid,
            "out_data": self._out,
            "delivered": self._delivered,
            "last_time": self._last_time,
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
        """Apply a batch of events atomically.

        The whole batch is validated (types, strictly increasing times,
        size <= 100_000) before any register changes; a failure therefore
        leaves the wrapper exactly as it was.
        """
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
        valid_o, out_o = self._valid, self._out

        # Dequeue request: the output register can accept a new word exactly
        # when it is empty or its current word is consumed this same edge.
        # Whether the request is *accepted* is decided by the inner FIFO's
        # old empty flag alone -- never by any true queue length.
        ren = bool((not valid_o) or event.ready) if event.rclk else False
        inner_rec = self._fifo.run_one(
            Event(event.time, event.wclk, event.rclk,
                  event.wen, ren, event.wdata)
        )

        load = delivered = None
        delivered_data = None
        if event.rclk:
            delivered = bool(valid_o and event.ready)
            if delivered:
                delivered_data = out_o
                self._delivered += 1
            load = bool(inner_rec.r_accept)
            if load:
                # New word loaded; on a same-edge consume+load valid stays.
                self._out = inner_rec.rdata
                self._valid = True
            elif delivered:
                # Old word consumed, nothing to replace it with.
                self._valid = False
            # else: stalled (valid & ~ready) or still empty -- hold state.

        self._last_time = event.time
        record = RVRecord(
            time=event.time,
            wclk=event.wclk,
            rclk=event.rclk,
            wen=event.wen,
            ready=event.ready,
            wdata=event.wdata,
            ren=ren if event.rclk else None,
            w_accept=inner_rec.w_accept,
            r_accept=inner_rec.r_accept,
            load=load,
            delivered=delivered,
            delivered_data=delivered_data,
            valid_before=valid_o,
            valid_after=self._valid,
            out_before=out_o,
            out_after=self._out,
            inner=inner_rec,
        )
        self._steps.append(record)
        return record
