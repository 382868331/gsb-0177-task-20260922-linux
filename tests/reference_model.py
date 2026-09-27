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


class RefDualReaderFIFO:
    """Independent oracle for one writer and two independent readers."""

    def __init__(self, k: int) -> None:
        assert 1 <= k <= 5
        self.k = k
        self.depth = 1 << k
        self.modulus = 1 << (k + 1)
        self.mask = self.modulus - 1
        self.reset()

    def reset(self):
        self.wptr = 0
        self.rptrs = [0, 0]
        self.full = False
        self.empties = [True, True]
        self.r2w_s1 = [0, 0]
        self.r2w_s2 = [0, 0]
        self.w2r_s1 = [0, 0]
        self.w2r_s2 = [0, 0]
        self.queues = [deque(), deque()]
        self.n_w = 0
        self.n_r = [0, 0]

    @staticmethod
    def _decode(gray_value):
        bits = gray_value
        result = gray_value
        while bits:
            bits >>= 1
            result ^= bits
        return result

    @staticmethod
    def _distance(leader, follower, modulus):
        return (leader - follower) % modulus

    def step(self, wclk, r0clk, r1clk, wen, r0en, r1en, wdata=None):
        w_o = self.wptr
        r_o = (self.rptrs[0], self.rptrs[1])
        full_o = self.full
        empty_o = (self.empties[0], self.empties[1])
        r2w1_o = (self.r2w_s1[0], self.r2w_s1[1])
        r2w2_o = (self.r2w_s2[0], self.r2w_s2[1])
        w2r1_o = (self.w2r_s1[0], self.w2r_s1[1])
        w2r2_o = (self.w2r_s2[0], self.w2r_s2[1])
        clocks = (r0clk, r1clk)
        enables = (r0en, r1en)

        w_accept = bool(wen and not full_o) if wclk else None
        r_accept = tuple(
            bool(enables[i] and not empty_o[i]) if clocks[i] else None
            for i in range(2)
        )
        w_n = (w_o + (1 if w_accept else 0)) & self.mask
        r_n = tuple(
            (r_o[i] + (1 if r_accept[i] else 0)) & self.mask
            for i in range(2)
        )

        popped = [None, None]
        if w_accept:
            assert all(len(q) < self.depth for q in self.queues)
            self.queues[0].append(wdata)
            self.queues[1].append(wdata)
            self.n_w += 1
        for i in range(2):
            if r_accept[i]:
                assert self.queues[i]
                popped[i] = self.queues[i].popleft()
                self.n_r[i] += 1

        if wclk:
            d0 = self._distance(w_n, self._decode(r2w2_o[0]), self.modulus)
            d1 = self._distance(w_n, self._decode(r2w2_o[1]), self.modulus)
            self.full = max(d0, d1) >= self.depth
            self.r2w_s1 = [ref_gray(r_o[0]), ref_gray(r_o[1])]
            self.r2w_s2 = [r2w1_o[0], r2w1_o[1]]
        for i in range(2):
            if clocks[i]:
                self.empties[i] = ref_gray(r_n[i]) == w2r2_o[i]
                self.w2r_s1[i] = ref_gray(w_o)
                self.w2r_s2[i] = w2r1_o[i]

        self.wptr = w_n
        self.rptrs = list(r_n)
        return {
            "w_accept": w_accept,
            "r_accept": r_accept,
            "wptr": w_n,
            "rptrs": r_n,
            "full": self.full,
            "empties": tuple(self.empties),
            "r2w": (
                (self.r2w_s1[0], self.r2w_s2[0]),
                (self.r2w_s1[1], self.r2w_s2[1]),
            ),
            "w2r": (
                (self.w2r_s1[0], self.w2r_s2[0]),
                (self.w2r_s1[1], self.w2r_s2[1]),
            ),
            "rdata": tuple(popped),
            "outstanding": (len(self.queues[0]), len(self.queues[1])),
        }


def make_dual_schedule(n_steps, *, w_period, r0_period, r1_period, seed=1,
                       stop_windows=(), r0_stop_windows=(),
                       r1_stop_windows=()):
    rnd = lcg_stream(seed)
    events = []
    t = 0
    payload = 0
    while len(events) < n_steps:
        t += 1
        wclk = t % w_period == 0
        r0clk = t % r0_period == 0
        r1clk = t % r1_period == 0
        if any(lo <= t < hi for lo, hi in stop_windows):
            wclk = False
        if any(lo <= t < hi for lo, hi in r0_stop_windows):
            r0clk = False
        if any(lo <= t < hi for lo, hi in r1_stop_windows):
            r1clk = False
        if not (wclk or r0clk or r1clk):
            continue
        wen = bool(wclk and (next(rnd) & 3))
        r0en = bool(r0clk and (next(rnd) & 3))
        r1en = bool(r1clk and (next(rnd) & 3))
        data = payload if wen else None
        if wen:
            payload += 1
        events.append((t, wclk, r0clk, r1clk, wen, r0en, r1en, data))
    return events
