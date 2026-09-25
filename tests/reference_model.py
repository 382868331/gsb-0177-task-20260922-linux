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

    def __init__(self, k: int, *, af_low=None, af_high=None,
                 ae_low=None, ae_high=None) -> None:
        assert 1 <= k <= 5
        self.k = k
        self.depth = 1 << k
        self.mask = (1 << (k + 1)) - 1
        self.full_mask = 0b11 << (k - 1)
        self.af_low = self.depth - 1 if af_low is None else af_low
        self.af_high = self.depth if af_high is None else af_high
        self.ae_low = 0 if ae_low is None else ae_low
        self.ae_high = 1 if ae_high is None else ae_high
        assert 0 <= self.af_low < self.af_high <= self.depth
        assert 0 <= self.ae_low < self.ae_high <= self.depth
        # monotonic accepted-access counters (wrap-free occupancy measure)
        self.n_w = 0
        self.n_r = 0
        self.reset()

    def reset(self) -> None:
        self.wptr = 0
        self.rptr = 0
        self.full = False
        self.empty = True
        self.almost_full = False
        self.almost_empty = True
        self.r2w = [0, 0]  # synchronized read Gray, stages 1/2, write domain
        self.w2r = [0, 0]  # synchronized write Gray, stages 1/2, read domain
        self.queue = deque()  # abstract data queue (ordering oracle only)

    @staticmethod
    def _gray_decode(code):
        """Independent Gray inverse (XOR fold), restated locally."""
        bits = code
        out = code
        while bits:
            bits >>= 1
            out ^= bits
        return out

    def step(self, wclk, rclk, wen, ren, wdata=None):
        w_o, r_o = self.wptr, self.rptr
        full_o, empty_o = self.full, self.empty
        af_o, ae_o = self.almost_full, self.almost_empty
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

        # Advisory estimates, derived independently from the SAME contract:
        # local post-accept binary pointer minus the OLD stage-2 Gray value of
        # the peer (decoded here), modulo 2**(k+1).  Never touches the queue.
        w_occ_b = w_occ_a = None
        r_read_b = r_read_a = None
        if wclk:
            peer_r = self._gray_decode(r2w_2_o)
            w_occ_b = (w_o - peer_r) & self.mask
            w_occ_a = (w_n - peer_r) & self.mask
            if w_occ_a >= self.af_high:
                self.almost_full = True
            elif w_occ_a <= self.af_low:
                self.almost_full = False
        if rclk:
            peer_w = self._gray_decode(w2r_2_o)
            r_read_b = (peer_w - r_o) & self.mask
            r_read_a = (peer_w - r_n) & self.mask
            if r_read_a <= self.ae_low:
                self.almost_empty = True
            elif r_read_a >= self.ae_high:
                self.almost_empty = False

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
            "almost_full": self.almost_full if wclk else af_o,
            "almost_empty": self.almost_empty if rclk else ae_o,
            "w_occ_before": w_occ_b,
            "w_occ_after": w_occ_a,
            "r_readable_before": r_read_b,
            "r_readable_after": r_read_a,
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
