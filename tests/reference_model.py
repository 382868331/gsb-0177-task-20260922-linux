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


class RefDualFIFO:
    """Independent dual-reader golden model; deliberately imports no afifo."""

    def __init__(self, k):
        assert 1 <= k <= 5
        self.k = k
        self.depth = 1 << k
        self.mask = (1 << (k + 1)) - 1
        self.reset()

    def reset(self):
        self.wptr = 0
        self.rptrs = [0, 0]
        self.full = False
        self.empties = [True, True]
        self.r2w = [[0, 0], [0, 0]]
        self.w2r = [[0, 0], [0, 0]]
        self.queues = [deque(), deque()]
        self.slot_owner = [None] * self.depth
        self.n_w = 0
        self.n_r = [0, 0]

    def _distance(self, later, earlier):
        return (later - earlier) & self.mask

    def _slowest(self, wptr, rptrs):
        slowest = rptrs[0]
        largest = self._distance(wptr, slowest)
        for rptr in rptrs[1:]:
            distance = self._distance(wptr, rptr)
            if distance > largest:
                slowest, largest = rptr, distance
        return slowest

    def step(self, wclk, rclks, wen, rens, wdata=None):
        w_o = self.wptr
        rs_o = list(self.rptrs)
        full_o = self.full
        empties_o = list(self.empties)
        r2w_o = [list(pair) for pair in self.r2w]
        w2r_o = [list(pair) for pair in self.w2r]

        w_accept = bool(wen and not full_o) if wclk else None
        r_accepts = tuple(
            bool(rens[i] and not empties_o[i]) if rclks[i] else None
            for i in range(2)
        )

        if w_accept:
            addr = w_o & (self.depth - 1)
            old_owner = self.slot_owner[addr]
            if old_owner is not None:
                # Every entry at this address must have been consumed by both
                # independent logical queues before the slot is reusable.
                assert old_owner < min(self.n_r), "slot overwritten too early"
            seq = self.n_w
            self.slot_owner[addr] = seq
            self.n_w += 1
            for q in self.queues:
                q.append((seq, wdata))

        rdatas = [None, None]
        for i in range(2):
            if r_accepts[i]:
                assert self.queues[i], f"reader {i} abstract underflow"
                seq, payload = self.queues[i].popleft()
                expected_addr = rs_o[i] & (self.depth - 1)
                assert self.slot_owner[expected_addr] == seq
                rdatas[i] = payload
                self.n_r[i] += 1

        w_n = (w_o + (1 if w_accept else 0)) & self.mask
        rs_n = [
            (rs_o[i] + (1 if r_accepts[i] else 0)) & self.mask
            for i in range(2)
        ]

        r2w_n = [list(pair) for pair in r2w_o]
        w2r_n = [list(pair) for pair in w2r_o]
        if wclk:
            for i in range(2):
                r2w_n[i] = [ref_gray(rs_o[i]), r2w_o[i][0]]
            self.full = any(
                self._distance(w_n, gray_inv(pair[1])) >= self.depth
                for pair in r2w_n
            )

        for i in range(2):
            if rclks[i]:
                w2r_n[i] = [ref_gray(w_o), w2r_o[i][0]]
                synced_w = gray_inv(w2r_n[i][1])
                self.empties[i] = self._distance(synced_w, rs_n[i]) == 0

        self.wptr = w_n
        self.rptrs = rs_n
        self.r2w = r2w_n
        self.w2r = w2r_n
        release = self._slowest(w_n, rs_n)
        synced_release = self._slowest(
            w_n, [gray_inv(pair[1]) for pair in r2w_n]
        )
        return {
            "w_accept": w_accept,
            "r_accepts": r_accepts,
            "wptr": w_n,
            "rptrs": tuple(rs_n),
            "wgray": ref_gray(w_n),
            "rgrays": tuple(ref_gray(p) for p in rs_n),
            "full": self.full,
            "empties": tuple(self.empties),
            "r2w": tuple((p[0], p[1]) for p in r2w_n),
            "w2r": tuple((p[0], p[1]) for p in w2r_n),
            "release_ptr": release,
            "synced_release_ptr": synced_release,
            "rdatas": tuple(rdatas),
            "queues": tuple(tuple(q) for q in self.queues),
        }


def gray_inv(gray):
    bits = gray
    value = gray
    while bits:
        bits >>= 1
        value ^= bits
    return value


def make_dual_schedule(k, n_steps, *, w_period, r0_period, r1_period, seed=1,
                       w_stops=(), r0_stops=(), r1_stops=()):
    rnd = lcg_stream(seed)
    events = []
    t = 0
    payload = 0
    while len(events) < n_steps:
        t += 1
        clocks = [
            t % w_period == 0,
            t % r0_period == 0,
            t % r1_period == 0,
        ]
        windows = (w_stops, r0_stops, r1_stops)
        for idx, stops in enumerate(windows):
            if any(lo <= t < hi for lo, hi in stops):
                clocks[idx] = False
        if not any(clocks):
            continue
        wen = bool(clocks[0] and (next(rnd) & 3))
        ren0 = bool(clocks[1] and (next(rnd) & 3))
        ren1 = bool(clocks[2] and (next(rnd) & 3))
        data = f"D{payload}" if wen else None
        if wen:
            payload += 1
        events.append(
            (t, clocks[0], clocks[1], clocks[2], wen, ren0, ren1, data)
        )
    return events
