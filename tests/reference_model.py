"""Independent golden model + abstract-queue checker (test support only).

This module deliberately does NOT import :mod:`afifo`.  The expected pointer,
flag, synchronizer, generation/acknowledgement handshake and accept/reject
values are re-derived here straight from the task contract, and accepted
payload order/occupancy bounds are checked with an abstract FIFO queue
(:class:`collections.deque`) -- never with the core's addressed memory or
flags.
"""

from collections import deque


def ref_gray(binary: int) -> int:
    """Independent re-statement of the reflected Gray formula."""
    return binary ^ (binary >> 1)


class RefFIFO:
    """Hand-written reference mirroring the contracted hardware semantics.

    Includes the per-domain reset generation handshake: a reset increments the
    domain generation; the partner learns it after two of its own clock edges
    (synchronizer s2), latches an acknowledgement and zeroes its local view;
    the acknowledgement reaches the resetting domain after two of its edges.
    Data passes only when both directions are acknowledged
    (``rack2w_s2 == w_gen`` and ``wack2r_s2 == r_gen``).
    """

    def __init__(self, k: int) -> None:
        assert 1 <= k <= 5
        self.k = k
        self.depth = 1 << k
        self.mask = (1 << (k + 1)) - 1
        self.full_mask = 0b11 << (k - 1)
        # monotonic accepted-access counters (wrap-free occupancy measure)
        self.n_w = 0
        self.n_r = 0
        self.reset_all()

    def reset_all(self) -> None:
        self.wptr = 0
        self.rptr = 0
        self.full = False
        self.empty = True
        self.r2w = [0, 0]  # synchronized read Gray, stages 1/2, write domain
        self.w2r = [0, 0]  # synchronized write Gray, stages 1/2, read domain
        # generation handshake
        self.w_gen = 0
        self.r_gen = 0
        self.wgen2r = [0, 0]
        self.rgen2w = [0, 0]
        self.rack_reg = 0   # latest w_gen learned in the read domain
        self.wack_reg = 0   # latest r_gen learned in the write domain
        self.rack2w = [0, 0]
        self.wack2r = [0, 0]
        self.queue = deque()  # abstract data queue (ordering oracle only)

    def _synced(self, rack2w_s2, wack2r_s2, w_gen=None, r_gen=None):
        if w_gen is None:
            w_gen = self.w_gen
        if r_gen is None:
            r_gen = self.r_gen
        return rack2w_s2 == w_gen and wack2r_s2 == r_gen

    def step(self, wclk, rclk, wen, ren, wdata=None,
             wrst=False, rrst=False):
        w_o, r_o = self.wptr, self.rptr
        full_o, empty_o = self.full, self.empty
        r2w_1_o, r2w_2_o = self.r2w
        w2r_1_o, w2r_2_o = self.w2r
        wg, rg = self.w_gen, self.r_gen
        wgen2r_1_o, wgen2r_2_o = self.wgen2r
        rgen2w_1_o, rgen2w_2_o = self.rgen2w
        rack_o, wack_o = self.rack_reg, self.wack_reg
        rack2w_1_o, rack2w_2_o = self.rack2w
        wack2r_1_o, wack2r_2_o = self.wack2r
        wg_pre, rg_pre = wg, rg

        # asynchronous reset precedes edge sampling
        if wrst:
            w_o, full_o = 0, False
            r2w_1_o = r2w_2_o = 0
            rgen2w_1_o = rgen2w_2_o = 0
            rack2w_1_o = rack2w_2_o = 0
            wack_o = 0
            wg += 1
        if rrst:
            r_o, empty_o = 0, True
            w2r_1_o = w2r_2_o = 0
            wgen2r_1_o = wgen2r_2_o = 0
            wack2r_1_o = wack2r_2_o = 0
            rack_o = 0
            rg += 1

        synced_o = self._synced(rack2w_2_o, wack2r_2_o, wg, rg)
        if wrst or rrst:
            self.queue.clear()
            self.n_w = 0
            self.n_r = 0
        pass_ok = (not (wrst or rrst)) and synced_o

        w_accept = bool(pass_ok and wen and not full_o) if wclk else None
        r_accept = bool(pass_ok and ren and not empty_o) if rclk else None

        w_n, r_n = w_o, r_o
        full_n, empty_n = full_o, empty_o
        r2w_n = [r2w_1_o, r2w_2_o]
        w2r_n = [w2r_1_o, w2r_2_o]
        wgen2r_n = [wgen2r_1_o, wgen2r_2_o]
        rgen2w_n = [rgen2w_1_o, rgen2w_2_o]
        rack_n, wack_n = rack_o, wack_o
        rack2w_n = [rack2w_1_o, rack2w_2_o]
        wack2r_n = [wack2r_1_o, wack2r_2_o]

        # Abstract queue: bounds + order, independent of any hardware flag.
        popped = None
        if w_accept:
            assert len(self.queue) < self.depth, "abstract overflow"
            self.queue.append(wdata)
            self.n_w += 1
            w_n = (w_o + 1) & self.mask
        if r_accept:
            assert self.queue, "abstract underflow"
            popped = self.queue.popleft()
            self.n_r += 1
            r_n = (r_o + 1) & self.mask

        if wclk and not wrst:
            r2w_n = [ref_gray(r_o), r2w_1_o]       # samples OLD read Gray
            rgen2w_n = [rg, rgen2w_1_o]
            rack2w_n = [rack_o, rack2w_1_o]
            if rgen2w_2_o != wack_o:
                # write domain learns a new read generation: restart view
                wack_n = rgen2w_2_o
                w_n = 0
                full_n = False
            elif synced_o:
                full_n = ref_gray(w_n) == (r2w_2_o ^ self.full_mask)
            else:
                full_n = False  # hold safe value across the resync window

        if rclk and not rrst:
            w2r_n = [ref_gray(w_o), w2r_1_o]       # samples OLD write Gray
            wgen2r_n = [wg, wgen2r_1_o]
            wack2r_n = [wack_o, wack2r_1_o]
            if wgen2r_2_o != rack_o:
                rack_n = wgen2r_2_o
                r_n = 0
                empty_n = True
            elif synced_o:
                empty_n = ref_gray(r_n) == w2r_2_o
            else:
                empty_n = True  # hold safe value across the resync window

        self.wptr, self.rptr = w_n, r_n
        self.full, self.empty = full_n, empty_n
        self.r2w, self.w2r = r2w_n, w2r_n
        self.w_gen, self.r_gen = wg, rg
        self.wgen2r, self.rgen2w = wgen2r_n, rgen2w_n
        self.rack_reg, self.wack_reg = rack_n, wack_n
        self.rack2w, self.wack2r = rack2w_n, wack2r_n
        synced_n = self._synced(rack2w_n[1], wack2r_n[1])

        return {
            "w_accept": w_accept,
            "r_accept": r_accept,
            "wptr": w_n,
            "rptr": r_n,
            "wgray": ref_gray(w_n),
            "rgray": ref_gray(r_n),
            "full": full_n,
            "empty": empty_n,
            "r2w": tuple(r2w_n),
            "w2r": tuple(w2r_n),
            "wgen_before": wg_pre,
            "wgen": wg,
            "rgen_before": rg_pre,
            "rgen": rg,
            "wgen2r": tuple(wgen2r_n),
            "rgen2w": tuple(rgen2w_n),
            "rack2w": tuple(rack2w_n),
            "wack2r": tuple(wack2r_n),
            "rack_reg": rack_n,
            "wack_reg": wack_n,
            "synced": synced_n,
            "occupancy": len(self.queue),
            "rdata": popped,
        }


# Deterministic schedule generator: fixed LCG, no random-module seeding quirks.
def lcg_stream(seed=0xC0FFEE):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state


def make_schedule(k, n_steps, *, w_period, r_period, seed=1, stop_windows=(),
                  r_stop_windows=()):
    """Build ``(t, wclk, rclk, wen, ren, payload)`` tuples.

    Clock edges are pure periodic patterns on a common integer time axis;
    enables come from a deterministic bit stream.  ``stop_windows`` /
    ``r_stop_windows`` are iterables of ``(lo, hi)`` time intervals in which
    the write / read clock is held low, to exercise stopped clocks.
    """
    return [(t, wc, rc, we, re_, data)
            for (t, wc, rc, we, re_, data, _wrst, _rrst)
            in make_schedule_rst(
                k, n_steps, w_period=w_period, r_period=r_period, seed=seed,
                stop_windows=stop_windows, r_stop_windows=r_stop_windows)]


def make_schedule_rst(k, n_steps, *, w_period, r_period, seed=1,
                      stop_windows=(), r_stop_windows=(),
                      wrst_times=(), rrst_times=()):
    """Like :func:`make_schedule` but emits 8-tuples with reset flags.

    ``wrst_times`` / ``rrst_times`` are time points carrying a one-step
    asynchronous reset (they need no clock edge).
    """
    rnd = lcg_stream(seed)
    events = []
    t = 0
    payload = 0
    stops = stop_windows
    rstops = r_stop_windows
    wrst_set = set(wrst_times)
    rrst_set = set(rrst_times)
    while len(events) < n_steps:
        t += 1
        wrst = t in wrst_set
        rrst = t in rrst_set
        wclk = (t % w_period == 0)
        rclk = (t % r_period == 0)
        if any(lo <= t < hi for lo, hi in stops):
            wclk = False
        if any(lo <= t < hi for lo, hi in rstops):
            rclk = False
        if not (wclk or rclk or wrst or rrst):
            continue
        wen = bool(wclk and (next(rnd) & 3))        # 75% write requests
        ren = bool(rclk and (next(rnd) & 3))        # 75% read requests
        data = payload if wen else None
        if wen:
            payload += 1
        events.append((t, wclk, rclk, wen, ren, data, wrst, rrst))
    return events
