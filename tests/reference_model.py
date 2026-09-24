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
    """Hand-written reference mirroring the contracted hardware semantics.

    Independently restates the two-stage pointer/Gray timing *and* the
    per-domain reset generation handshake: a reset bumps the local
    generation, the peer adopts a strictly newer synced generation and
    echoes it, and transfers are gated until both second stages agree.
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
        """Power-on / coarse reset: generations back to zero, queue empty."""
        self.wptr = 0
        self.rptr = 0
        self.full = False
        self.empty = True
        self.r2w = [0, 0]  # synchronized read pointer Gray, write domain
        self.w2r = [0, 0]  # synchronized write pointer Gray, read domain
        self.wgen = 0
        self.rgen = 0
        self.rgen2w = [0, 0]  # synced read generation Gray, write domain
        self.wgen2r = [0, 0]  # synced write generation Gray, read domain
        self.queue = deque()

    def _ref_gray_inv(self, g):
        bits = g
        while bits:
            bits >>= 1
            g ^= bits
        return g

    def _purge(self):
        self.queue.clear()
        self.n_w = 0
        self.n_r = 0

    def step(self, wclk, rclk, wen, ren, wdata=None, wrst=False, rrst=False):
        w_o, r_o = self.wptr, self.rptr
        full_o, empty_o = self.full, self.empty
        r2w_1_o, r2w_2_o = self.r2w
        w2r_1_o, w2r_2_o = self.w2r
        wg_o, rg_o = self.wgen, self.rgen
        rgen_1_o, rgen_2_o = self.rgen2w
        wgen_1_o, wgen_2_o = self.wgen2r
        wready_o = (wg_o == self._ref_gray_inv(rgen_2_o))
        rready_o = (rg_o == self._ref_gray_inv(wgen_2_o))

        # ---- phase 1: explicit synchronous resets first
        if wrst:
            self.wptr = 0
            self.full = False
            self.r2w = [0, 0]
            self.rgen2w = [0, 0]
            self.wgen += 1
        if rrst:
            self.rptr = 0
            self.empty = True
            self.w2r = [0, 0]
            self.wgen2r = [0, 0]
            self.rgen += 1
        if wrst or rrst:
            self._purge()

        w_l, r_l = self.wptr, self.rptr
        full_l, empty_l = self.full, self.empty
        wg_l, rg_l = self.wgen, self.rgen
        r2w_1_l, r2w_2_l = self.r2w
        w2r_1_l, w2r_2_l = self.w2r
        rgen_1_l, rgen_2_l = self.rgen2w
        wgen_1_l, wgen_2_l = self.wgen2r

        w_ready = wg_l == self._ref_gray_inv(rgen_2_l)
        r_ready = rg_l == self._ref_gray_inv(wgen_2_l)
        gate = w_ready and r_ready and not wrst and not rrst

        w_accept = (bool(wen and not full_l) and gate) if wclk else None
        r_accept = (bool(ren and not empty_l) and gate) if rclk else None

        w_n = (w_l + (1 if w_accept else 0)) & self.mask
        r_n = (r_l + (1 if r_accept else 0)) & self.mask

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

        w_react = r_react = False
        if wclk:
            if wrst:
                self.r2w = [0, 0]
                self.rgen2w = [0, 0]
            else:
                self.r2w = [ref_gray(r_l), r2w_1_l]
                self.rgen2w = [ref_gray(rg_l), rgen_1_l]
                peer = self._ref_gray_inv(self.rgen2w[1])
                if peer > wg_l:
                    w_react = True
                    self.wgen = peer
                    w_n = 0
                    full_l = False
                    self.r2w = [0, 0]
                    self._purge()
        if rclk:
            if rrst:
                self.w2r = [0, 0]
                self.wgen2r = [0, 0]
            else:
                self.w2r = [ref_gray(w_l), w2r_1_l]
                self.wgen2r = [ref_gray(wg_l), wgen_1_l]
                peer = self._ref_gray_inv(self.wgen2r[1])
                if peer > rg_l:
                    r_react = True
                    self.rgen = peer
                    r_n = 0
                    empty_l = True
                    self.w2r = [0, 0]
                    self._purge()

        if wclk:
            self.full = (False if (wrst or w_react or not w_ready)
                         else ref_gray(w_n) == (r2w_2_l ^ self.full_mask))
        else:
            self.full = full_l
        if rclk:
            self.empty = (True if (rrst or r_react or not r_ready)
                          else ref_gray(r_n) == w2r_2_l)
        else:
            self.empty = empty_l

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
            "wgen": self.wgen,
            "rgen": self.rgen,
            "rgen2w": tuple(self.rgen2w),
            "wgen2r": tuple(self.wgen2r),
            "wready": self.wgen == self._ref_gray_inv(self.rgen2w[1]),
            "rready": self.rgen == self._ref_gray_inv(self.wgen2r[1]),
            "w_reactive": w_react,
            "r_reactive": r_react,
            "wgen_before": wg_o,
            "rgen_before": rg_o,
            "wready_before": wready_o,
            "rready_before": rready_o,
        }


# Deterministic schedule generator: fixed LCG, no random-module seeding quirks.
def lcg_stream(seed=0xC0FFEE):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state


def make_schedule(k, n_steps, *, w_period, r_period, seed=1, stop_windows=(),
                  r_stop_windows=(), wrst_times=(), rrst_times=()):
    """Build ``(t, wclk, rclk, wen, ren, payload, wrst, rrst)`` tuples.

    Clock edges are pure periodic patterns on a common integer time axis;
    enables come from a deterministic bit stream.  ``stop_windows`` /
    ``r_stop_windows`` are iterables of ``(lo, hi)`` time intervals in which
    the write / read clock is held low, to exercise stopped clocks.
    ``wrst_times`` / ``rrst_times`` are explicit time stamps at which a
    write-/read-domain reset pulse is issued; the corresponding domain clock
    is forced high at that step (a reset must edge its own clock).
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
        wclk = (t % w_period == 0)
        rclk = (t % r_period == 0)
        if any(lo <= t < hi for lo, hi in stops):
            wclk = False
        if any(lo <= t < hi for lo, hi in rstops):
            rclk = False
        wrst = t in wrst_set
        rrst = t in rrst_set
        if wrst:
            wclk = True   # a reset pulse edges its own clock despite a stop
        if rrst:
            rclk = True
        if not (wclk or rclk):
            continue
        wen = bool(wclk and (next(rnd) & 3))        # 75% write requests
        ren = bool(rclk and (next(rnd) & 3))        # 75% read requests
        data = payload if wen else None
        if wen:
            payload += 1
        events.append((t, wclk, rclk, wen, ren, data, wrst, rrst))
    return events
