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


def ref_ungray(gray: int) -> int:
    """Independently re-stated Gray decoder."""
    bits = gray
    while bits:
        bits >>= 1
        gray ^= bits
    return gray


class RefFIFO:
    """Hand-written reference mirroring the contracted hardware semantics."""

    def __init__(self, k: int, af_margin=1, ae_margin=1) -> None:
        assert 1 <= k <= 5
        self.k = k
        self.depth = 1 << k
        self.mask = (1 << (k + 1)) - 1
        self.full_mask = 0b11 << (k - 1)
        self.af_margin = af_margin
        self.ae_margin = ae_margin
        # monotonic accepted-access counters (wrap-free occupancy measure)
        self.n_w = 0
        self.n_r = 0
        self.reset()

    def reset(self) -> None:
        self.wptr = 0
        self.rptr = 0
        self.full = False
        self.empty = True
        self.almost_full = self.af_margin >= self.depth
        self.almost_empty = True
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
            # Oldest confirmed read pointer (s2) -> worst-case fill.
            occ_hi = (w_n - ref_ungray(r2w_2_o)) % (2 * self.depth)
            free_sure = self.depth - min(occ_hi, self.depth)
            self.almost_full = free_sure <= self.af_margin
        if rclk:
            self.empty = ref_gray(r_n) == w2r_2_o
            self.w2r = [ref_gray(w_o), w2r_1_o]  # samples OLD write Gray
            # Newly latched (newest confirmed) write pointer s1 -> entries
            # guaranteed present for the local next read pointer r_n.
            occ_lo = (ref_ungray(w2r_1_o) - r_n) % (2 * self.depth)
            entries_sure = min(occ_lo, self.depth)
            self.almost_empty = entries_sure <= self.ae_margin

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
            "almost_full": self.almost_full,
            "almost_empty": self.almost_empty,
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
