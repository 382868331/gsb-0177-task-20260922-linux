"""Independent golden model + abstract-queue checker (test support only).

This module deliberately does NOT import :mod:`afifo`.  The expected pointer,
flag, synchronizer and accept/reject values are re-derived here straight from
the task contract, and accepted payload order/occupancy bounds are checked
with an abstract FIFO queue (:class:`collections.deque`) -- never with the
core's addressed memory or flags.
"""

from collections import deque


def ref_gray(binary: int) -> int:
    """Independent re-statement of the reflected Gray formula."""
    return binary ^ (binary >> 1)


class RefFIFO:
    """Hand-written reference mirroring the contracted hardware semantics."""

    def __init__(self, k: int) -> None:
        assert 1 <= k <= 5
        self.k = k
        self.depth = 1 << k
        self.mask = (1 << (k + 1)) - 1
        self.full_mask = 0b11 << (k - 1)
        # monotonic accepted-access counters (wrap-free occupancy measure)
        self.n_w = 0
        self.n_r = 0
        self.reset()

    def reset(self) -> None:
        self.wptr = 0
        self.rptr = 0
        self.full = False
        self.empty = True
        self.r2w = [0, 0]  # synchronized read Gray, stages 1/2, write domain
        self.w2r = [0, 0]  # synchronized write Gray, stages 1/2, read domain
        self.queue = deque()  # abstract data queue (ordering oracle only)

    def step(self, wclk, rclk, wen, ren, wdata=None):
        w_o, r_o = self.wptr, self.rptr
        full_o, empty_o = self.full, self.empty
        r2w_1_o, r2w_2_o = self.r2w
        w2r_1_o, w2r_2_o = self.w2r

        w_accept = bool(wen and not full_o) if wclk else None
        r_accept = bool(ren and not empty_o) if rclk else None

        w_n = (w_o + (1 if w_accept else 0)) & self.mask
        r_n = (r_o + (1 if r_accept else 0)) & self.mask

        # Abstract queue: bounds + order, independent of any hardware flag.
        popped = None
        if w_accept:
            assert len(self.queue) < self.depth, "abstract overflow"
            self.queue.append(wdata)
            self.n_w += 1
        if r_accept:
            assert self.queue, "abstract underflow"
            popped = self.queue.popleft()
            self.n_r += 1

        if wclk:
            self.full = ref_gray(w_n) == (r2w_2_o ^ self.full_mask)
            self.r2w = [ref_gray(r_o), r2w_1_o]  # samples OLD read Gray
        if rclk:
            self.empty = ref_gray(r_n) == w2r_2_o
            self.w2r = [ref_gray(w_o), w2r_1_o]  # samples OLD write Gray

        self.wptr, self.rptr = w_n, r_n
        return {
            "w_accept": w_accept,
            "r_accept": r_accept,
            "wptr": w_n,
            "rptr": r_n,
            "wgray": ref_gray(w_n),
            "rgray": ref_gray(r_n),
            "full": self.full,
            "empty": self.empty,
            "r2w": tuple(self.r2w),
            "w2r": tuple(self.w2r),
            "occupancy": len(self.queue),
            "rdata": popped,
        }


# Deterministic schedule generator: fixed LCG, no random-module seeding quirks.
def lcg_stream(seed=0xC0FFEE):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state


class RefOutReg:
    """Independent reference for the ready/valid output-register wrapper.

    Re-stated straight from the wrapper contract (never imports ``afifo``):

    * dequeue request ``ren = (not valid_old) or ready`` on read edges;
    * the inner reference still gates that request with its own old ``empty``
      flag, so nothing is prefetched on true queue length;
    * a read edge with ``valid_old and ready`` delivers the old word; a
      same-edge accepted dequeue loads the next word and keeps ``valid``;
    * ``valid and not ready`` freezes the output word and the read pointer.

    Two independent sequences are maintained: ``written`` (accepted writes,
    in order) and ``delivered_seq`` (downstream handshakes, in order).
    """

    def __init__(self, k: int) -> None:
        self.inner = RefFIFO(k)
        self.valid = False
        self.out = None
        self.n_delivered = 0
        self.written = []        # accepted writes, in order
        self.delivered_seq = []  # downstream deliveries, in order

    def step(self, wclk, rclk, wen, ready, wdata=None):
        valid_o = self.valid
        ren = bool(rclk and ((not valid_o) or ready))
        exp = self.inner.step(wclk, rclk, wen, ren, wdata)

        delivered = load = None
        delivered_data = None
        if rclk:
            delivered = bool(valid_o and ready)
            if delivered:
                delivered_data = self.out
                self.n_delivered += 1
                self.delivered_seq.append(self.out)
            load = bool(exp["r_accept"])
            if load:
                self.out = exp["rdata"]
                self.valid = True
            elif delivered:
                self.valid = False
        if exp["w_accept"]:
            self.written.append(wdata)

        return {
            "ren": ren if rclk else None,
            "w_accept": exp["w_accept"],
            "r_accept": exp["r_accept"],
            "load": load,
            "delivered": delivered,
            "delivered_data": delivered_data,
            "valid": self.valid,
            "out": self.out,
            "wptr": exp["wptr"],
            "rptr": exp["rptr"],
            "full": exp["full"],
            "empty": exp["empty"],
            "inner_occupancy": exp["occupancy"],
        }


def make_rv_schedule(k, n_steps, *, w_period, r_period, seed=1,
                     stop_windows=(), r_stop_windows=(),
                     backpressure_windows=()):
    """Build ``(t, wclk, rclk, wen, ready, payload)`` tuples.

    Same clock/stop scheme as :func:`make_schedule`, but the read side is a
    ``ready`` level (75% high from the deterministic bit stream) that is
    forced low inside ``backpressure_windows`` ``(lo, hi)`` intervals to
    exercise sustained downstream stalls.
    """
    rnd = lcg_stream(seed)
    events = []
    t = 0
    payload = 0
    stops = stop_windows
    rstops = r_stop_windows
    bps = backpressure_windows
    while len(events) < n_steps:
        t += 1
        wclk = (t % w_period == 0)
        rclk = (t % r_period == 0)
        if any(lo <= t < hi for lo, hi in stops):
            wclk = False
        if any(lo <= t < hi for lo, hi in rstops):
            rclk = False
        if not (wclk or rclk):
            continue
        wen = bool(wclk and (next(rnd) & 3))        # 75% write requests
        ready = bool(next(rnd) & 3)                 # 75% downstream ready
        if any(lo <= t < hi for lo, hi in bps):
            ready = False
        data = payload if wen else None
        if wen:
            payload += 1
        events.append((t, wclk, rclk, wen, ready, data))
    return events


def make_schedule(k, n_steps, *, w_period, r_period, seed=1, stop_windows=(),
                  r_stop_windows=()):
    """Build ``(t, wclk, rclk, wen, ren, payload)`` tuples.

    Clock edges are pure periodic patterns on a common integer time axis;
    enables come from a deterministic bit stream.  ``stop_windows`` /
    ``r_stop_windows`` are iterables of ``(lo, hi)`` time intervals in which
    the write / read clock is held low, to exercise stopped clocks.
    """
    rnd = lcg_stream(seed)
    events = []
    t = 0
    payload = 0
    stops = stop_windows
    rstops = r_stop_windows
    while len(events) < n_steps:
        t += 1
        wclk = (t % w_period == 0)
        rclk = (t % r_period == 0)
        if any(lo <= t < hi for lo, hi in stops):
            wclk = False
        if any(lo <= t < hi for lo, hi in rstops):
            rclk = False
        if not (wclk or rclk):
            continue
        wen = bool(wclk and (next(rnd) & 3))        # 75% write requests
        ren = bool(rclk and (next(rnd) & 3))        # 75% read requests
        data = payload if wen else None
        if wen:
            payload += 1
        events.append((t, wclk, rclk, wen, ren, data))
    return events
