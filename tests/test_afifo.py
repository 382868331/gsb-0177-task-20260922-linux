"""Necessary tests for the afifo package.

Expected values come from:
1. Hand-computed Gray tables and a hand-derived k=1 fill/drain waveform.
2. ``reference_model.RefFIFO`` -- an independent reimplementation that never
   imports ``afifo`` -- plus an abstract ``deque`` for data order / bounds.
Round-trip (Gray inverse) checks are supplementary only.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_model import RefFIFO, make_schedule  # noqa: E402

from afifo import AsyncFIFO, Event, binary_to_gray, gray_to_binary  # noqa: E402
from afifo.model import AsyncFIFOError  # noqa: E402


# Hand-computed reflected Gray table for 3..6 bit widths.
GRAY_TABLE = {
    0: 0, 1: 1, 2: 3, 3: 2, 4: 6, 5: 7, 6: 5, 7: 4,
    8: 12, 9: 13, 10: 15, 11: 14, 12: 10, 13: 11, 14: 9, 15: 8,
}


class TestGray(unittest.TestCase):
    def test_hand_computed_table(self):
        for binary, gray in GRAY_TABLE.items():
            self.assertEqual(binary_to_gray(binary), gray, f"binary={binary}")

    def test_adjacent_codes_differ_in_one_bit(self):
        for width in range(1, 7):
            for v in range((1 << width) - 1):
                diff = binary_to_gray(v) ^ binary_to_gray(v + 1)
                self.assertEqual(diff & (diff - 1), 0)  # exactly one bit

    def test_round_trip_supplementary(self):
        for v in range(2 ** 7):
            self.assertEqual(gray_to_binary(binary_to_gray(v)), v)

    def test_gray_input_contract(self):
        with self.assertRaisesRegex(TypeError, "int"):
            binary_to_gray(1.0)
        with self.assertRaisesRegex(TypeError, "bool"):
            binary_to_gray(True)
        with self.assertRaisesRegex(ValueError, ">="):
            binary_to_gray(-1)


class TestReset(unittest.TestCase):
    def test_power_on_state(self):
        fifo = AsyncFIFO(k=3)
        self.assertEqual(fifo.depth, 8)
        self.assertEqual(fifo.wptr, 0)
        self.assertEqual(fifo.rptr, 0)
        self.assertFalse(fifo.full)       # full resets to False
        self.assertTrue(fifo.empty)       # empty resets to True
        self.assertEqual(fifo.accepted_writes, 0)
        self.assertEqual(fifo.accepted_reads, 0)
        self.assertEqual(fifo.snapshot()["r2w"], (0, 0))
        self.assertEqual(fifo.snapshot()["w2r"], (0, 0))

    def test_k_bounds_and_full_mask(self):
        for k in range(1, 6):
            fifo = AsyncFIFO(k=k)
            self.assertEqual(fifo.depth, 1 << k)
        self.assertEqual(AsyncFIFO(1)._full_mask, 0b11)
        self.assertEqual(AsyncFIFO(5)._full_mask, 0b11 << 4)


# Hand-derived k=1 waveform.  Each row:
# (time, wclk, rclk, wen, ren, w_accept, r_accept, wptr, rptr,
#  full, empty, r2w, w2r, expected rdata)
HAND_WAVE = [
    (1, 1, 0, 1, 0, True,  None, 1, 0, False, True,  (0, 0), (0, 0), None),
    (2, 1, 0, 1, 0, True,  None, 2, 0, True,  True,  (0, 0), (0, 0), None),
    (3, 1, 0, 1, 0, False, None, 2, 0, True,  True,  (0, 0), (0, 0), None),
    (4, 0, 1, 0, 1, None,  False, 2, 0, True, True,  (0, 0), (3, 0), None),
    (5, 0, 1, 0, 1, None,  False, 2, 0, True, True,  (0, 0), (3, 3), None),
    (6, 0, 1, 0, 1, None,  False, 2, 0, True, False, (0, 0), (3, 3), None),
    (7, 1, 1, 1, 1, False, True,  2, 1, True, False, (0, 0), (3, 3), "a"),
    (8, 0, 1, 0, 1, None,  True,  2, 2, True, True,  (0, 0), (3, 3), "b"),
    (9, 1, 0, 1, 0, False, None, 2, 2, True,  True,  (3, 0), (3, 3), None),
    (10, 1, 0, 1, 0, False, None, 2, 2, True, True,  (3, 3), (3, 3), None),
    (11, 1, 0, 1, 0, False, None, 2, 2, False, True, (3, 3), (3, 3), None),
    (12, 1, 0, 1, 0, True,  None, 3, 2, False, True, (3, 3), (3, 3), None),
]


class TestHandWaveform(unittest.TestCase):
    def test_exact_register_trace(self):
        fifo = AsyncFIFO(k=1)
        events = []
        for i, (t, wc, rc, we, re, *_rest) in enumerate(HAND_WAVE):
            data = None
            if we:
                data = "a" if i == 0 else "b"
            events.append(Event(t, bool(wc), bool(rc), bool(we), bool(re), data))
        records = fifo.run(events)

        for rec, row in zip(records, HAND_WAVE):
            (t, wc, rc, we, re, wacc, racc, wp, rp,
             full, empty, r2w, w2r, rdata) = row
            with self.subTest(time=t):
                self.assertEqual(rec.w_accept, wacc)
                self.assertEqual(rec.r_accept, racc)
                self.assertEqual(rec.wptr_after, wp)
                self.assertEqual(rec.rptr_after, rp)
                self.assertEqual(rec.wgray_after, binary_to_gray(wp))
                self.assertEqual(rec.rgray_after, binary_to_gray(rp))
                self.assertIs(rec.full_after, full)
                self.assertIs(rec.empty_after, empty)
                self.assertEqual(rec.r2w_after, r2w)
                self.assertEqual(rec.w2r_after, w2r)
                self.assertEqual(rec.rdata, rdata)

    def test_conservative_latency_visible(self):
        """At t=7 space is made but full stays asserted (old sync value)."""
        fifo = AsyncFIFO(k=1)
        # fill to full
        for t, data in ((1, "a"), (2, "b")):
            fifo.run_one(Event(t, True, False, wen=True, wdata=data))
        self.assertTrue(fifo.full)
        # bring empty down without reading: three rclk edges sync wgray
        for t in (4, 5, 6):
            fifo.run_one(Event(t, False, True, ren=True))
        self.assertFalse(fifo.empty)
        self.assertTrue(fifo.full)
        # coincident edge: read accepted, write rejected, full held stale
        rec = fifo.run_one(Event(7, True, True, wen=True, ren=True))
        self.assertTrue(rec.r_accept)
        self.assertFalse(rec.w_accept)
        self.assertTrue(rec.full_after)  # conservative: deasserts 2+ edges later


class TestCoincidentEdgesSampleOldState(unittest.TestCase):
    def test_reset_coincident_write_accepted_read_rejected(self):
        fifo = AsyncFIFO(k=2)
        rec = fifo.run_one(Event(1, True, True, wen=True, ren=True, wdata=7))
        self.assertTrue(rec.w_accept)
        self.assertFalse(rec.r_accept)       # old empty flag is True
        self.assertEqual(rec.wptr_after, 1)
        self.assertEqual(rec.rptr_after, 0)
        # both synchronizer input stages sample the OTHER domain's old Gray
        self.assertEqual(rec.r2w_after, (0, 0))
        self.assertEqual(rec.w2r_after, (0, 0))
        self.assertTrue(rec.empty_after)     # rgray=0 == old w2r_s2=0

    def test_coincident_accept_both_advances_both_pointers(self):
        fifo = AsyncFIFO(k=2)
        # one write, then let read domain see it, then coincident R+W
        fifo.run_one(Event(1, True, False, wen=True, wdata="x"))
        for t in (2, 3, 4):
            fifo.run_one(Event(t, False, True))
        self.assertFalse(fifo.empty)
        rec = fifo.run_one(Event(5, True, True, wen=True, ren=True, wdata="y"))
        self.assertTrue(rec.w_accept)
        self.assertTrue(rec.r_accept)
        self.assertEqual(rec.wptr_after, 2)
        self.assertEqual(rec.rptr_after, 1)
        self.assertEqual(rec.rdata, "x")
        # input stages latch old opposite Grays, not post-edge values
        self.assertEqual(rec.r2w_after[0], rec.rgray_before)
        self.assertEqual(rec.w2r_after[0], rec.wgray_before)


class TestStoppedClock(unittest.TestCase):
    def test_write_clock_stop_freezes_write_domain(self):
        fifo = AsyncFIFO(k=2)
        fifo.run([
            Event(1, True, False, wen=True, wdata=1),
            Event(2, True, False, wen=True, wdata=2),
        ])
        snap = fifo.snapshot()
        # only read clocks for a while: write domain (full, wptr, r2w) frozen
        for t in (3, 4, 5, 6):
            rec = fifo.run_one(Event(t, False, True))
            self.assertEqual(rec.full_after, snap["full"])
            self.assertEqual(rec.wptr_after, snap["wptr"])
            self.assertEqual(rec.r2w_after, snap["r2w"])

    def test_read_clock_stop_freezes_read_domain(self):
        fifo = AsyncFIFO(k=2)
        fifo.run_one(Event(1, True, False, wen=True, wdata=1))
        snap = fifo.snapshot()
        for t in (2, 3, 4):
            rec = fifo.run_one(Event(t, True, False))
            self.assertEqual(rec.empty_after, snap["empty"])
            self.assertEqual(rec.rptr_after, snap["rptr"])
            self.assertEqual(rec.w2r_after, snap["w2r"])


class TestDifferentialAgainstIndependentModel(unittest.TestCase):
    """Compare every register against RefFIFO over deterministic schedules."""

    def _diff_run(self, k, wp, rp, n, seed=1, w_stops=(), r_stops=()):
        schedule = make_schedule(
            k, n, w_period=wp, r_period=rp, seed=seed,
            stop_windows=w_stops, r_stop_windows=r_stops,
        )
        dut = AsyncFIFO(k=k)
        ref = RefFIFO(k=k)
        events = [Event(t, wc, rc, we, re_, data)
                  for (t, wc, rc, we, re_, data) in schedule]
        records = dut.run(events)
        read_out = []
        written = []  # only ACCEPTED writes, observed from the reference
        for ev, rec in zip(schedule, records):
            t, wc, rc, we, re_, data = ev
            exp = ref.step(wc, rc, we, re_, data)
            with self.subTest(k=k, wp=wp, rp=rp, t=t):
                self.assertEqual(rec.w_accept, exp["w_accept"])
                self.assertEqual(rec.r_accept, exp["r_accept"])
                self.assertEqual(rec.wptr_after, exp["wptr"])
                self.assertEqual(rec.rptr_after, exp["rptr"])
                self.assertEqual(rec.wgray_after, exp["wgray"])
                self.assertEqual(rec.rgray_after, exp["rgray"])
                self.assertIs(rec.full_after, exp["full"])
                self.assertIs(rec.empty_after, exp["empty"])
                self.assertEqual(rec.r2w_after, exp["r2w"])
                self.assertEqual(rec.w2r_after, exp["w2r"])
                self.assertEqual(rec.rdata, exp["rdata"])
                self.assertGreaterEqual(exp["occupancy"], 0)
                self.assertLessEqual(exp["occupancy"], 1 << k)
                if exp["w_accept"]:
                    written.append(data)
                if rec.r_accept:
                    read_out.append(rec.rdata)
        # accepted payloads leave in the order written (abstract queue check)
        self.assertEqual(dut.accepted_writes, ref.n_w)
        self.assertEqual(dut.accepted_reads, ref.n_r)
        self.assertEqual(ref.n_w - ref.n_r, len(ref.queue))
        self.assertEqual(read_out, written[: ref.n_r])
        self.assertEqual(list(ref.queue), written[ref.n_r:])
        self.assertEqual(dut.accepted_reads, len(read_out))

    def test_ratios_and_coincidence(self):
        ratios = [(1, 1), (2, 3), (3, 2), (2, 4), (4, 2),
                  (2, 5), (5, 2), (3, 5), (5, 3)]
        for k in (1, 2, 3, 4, 5):
            for wp, rp in ratios:
                self._diff_run(k, wp, rp, n=300, seed=100 * k + wp * 7 + rp)

    def test_stopped_clocks(self):
        self._diff_run(2, 2, 3, 240, seed=9,
                       w_stops=((40, 80),), r_stops=((140, 190),))
        self._diff_run(3, 3, 2, 240, seed=11,
                       w_stops=((10, 60), (120, 150)))

    def test_pointer_wrap_many_times(self):
        # k=1 pointers wrap every 2 accepted accesses; 400 steps => wraps.
        self._diff_run(1, 1, 1, 400, seed=42)
        # k=5 (depth 32) needs 64 accepts per wrap; long run covers it.
        self._diff_run(5, 2, 3, 4000, seed=77)


class TestSafetyInvariants(unittest.TestCase):
    def test_never_read_above_write_and_no_overflow(self):
        """Independent occupancy can never leave [0, depth] on any schedule."""
        for k, wp, rp in [(1, 1, 1), (2, 2, 3), (3, 3, 2), (5, 2, 5)]:
            schedule = make_schedule(k, 500, w_period=wp, r_period=rp,
                                     seed=k * 31 + wp + rp)
            ref = RefFIFO(k=k)
            dut = AsyncFIFO(k=k)
            for (t, wc, rc, we, re_, data) in schedule:
                dut.run_one(Event(t, wc, rc, we, re_, data))
                exp = ref.step(wc, rc, we, re_, data)
                self.assertTrue(0 <= exp["occupancy"] <= (1 << k))

    def test_flags_derive_from_pointers_not_queue_length(self):
        """Structural guarantee: full/empty equations never read the memory."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "afifo", "model.py"), encoding="utf-8") as f:
            source = f.read()
        for line in source.splitlines():
            if "full_n" in line or "empty_n" in line:
                self.assertNotIn("_mem", line)
                self.assertNotIn("len(", line)


class TestInputContract(unittest.TestCase):
    def test_k_rejects_bad_values(self):
        for bad in (0, 6, -1, True, 1.0, "2", None, 2.0):
            with self.subTest(bad=bad), self.assertRaises(AsyncFIFOError):
                AsyncFIFO(bad)

    def test_event_time_contract(self):
        with self.assertRaises(AsyncFIFOError):
            Event(True, True, False)                 # bool time rejected
        with self.assertRaises(AsyncFIFOError):
            Event(-1, True, False)
        with self.assertRaises(AsyncFIFOError):
            Event(1.0, True, False)                  # non-int time
        Event(0, True, False)                        # boundary allowed

    def test_event_clock_and_enable_contract(self):
        with self.assertRaisesRegex(AsyncFIFOError, "neither"):
            Event(1, False, False)
        with self.assertRaises(AsyncFIFOError):
            Event(1, 1, 0)                           # int instead of bool
        with self.assertRaises(AsyncFIFOError):
            Event(1, 1, 1, wen=1)
        with self.assertRaises(AsyncFIFOError):
            Event(1, 1, 1, ren=0)

    def test_non_finite_float_payload_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(AsyncFIFOError, "finite"):
                    Event(1, True, False, wen=True, wdata=bad)

    def test_wdata_rules(self):
        with self.assertRaisesRegex(AsyncFIFOError, "wdata"):
            Event(1, True, False, wen=False, wdata=9)
        # finite float and arbitrary objects are fine when wen is True
        Event(1, True, False, wen=True, wdata=1.5)
        Event(1, True, False, wen=True, wdata=object())

    def test_non_increasing_time_rejected(self):
        fifo = AsyncFIFO(k=2)
        fifo.run_one(Event(5, True, False))
        with self.assertRaisesRegex(AsyncFIFOError, "strictly greater"):
            fifo.run_one(Event(5, True, False))
        with self.assertRaisesRegex(AsyncFIFOError, "strictly greater"):
            fifo.run_one(Event(4, True, False))

    def test_run_rejects_single_event_and_non_events(self):
        fifo = AsyncFIFO(k=2)
        with self.assertRaises(AsyncFIFOError):
            fifo.run(Event(1, True, False))
        with self.assertRaises(AsyncFIFOError):
            fifo.run([Event(1, True, False), "nope"])
        with self.assertRaises(AsyncFIFOError):
            fifo.run(42)

    def test_batch_size_limit_locatable(self):
        fifo = AsyncFIFO(k=2)
        big = [Event(i + 1, True, False) for i in range(100_001)]
        with self.assertRaisesRegex(AsyncFIFOError, "100000"):
            fifo.run(big)

    def test_run_accepts_generator(self):
        fifo = AsyncFIFO(k=2)
        fifo.run(Event(t, True, False) for t in range(1, 4))
        self.assertEqual(len(fifo.steps), 3)


class TestBatchAtomicity(unittest.TestCase):
    def test_failed_batch_leaves_no_partial_change(self):
        fifo = AsyncFIFO(k=2)
        fifo.run([Event(1, True, False, wen=True, wdata="q")])
        before = fifo.snapshot()
        before_steps = fifo.steps
        bad_batch = [
            Event(2, True, False, wen=True, wdata="ok"),
            Event(3, True, False, wen=True, wdata="ok2"),
            Event(3, True, False),                     # duplicate time
        ]
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[2\]"):
            fifo.run(bad_batch)
        self.assertEqual(fifo.snapshot(), before)
        self.assertEqual(fifo.steps, before_steps)
        self.assertEqual(fifo.accepted_writes, 1)

    def test_bad_event_in_batch_reports_index(self):
        fifo = AsyncFIFO(k=2)
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[2\]"):
            fifo.run([Event(1, True, False), Event(2, True, False), 7])


# ===========================================================================
# Almost-full / almost-empty with hysteresis
# ===========================================================================
#
# Hand-derived k=2 (depth 4) waveform, write/read edges at UNEQUAL rates
# (bursts of four write edges then read edges then interleaved edges).
# Thresholds: almost_full (low=1, high=3), almost_empty (low=1, high=3),
# so an estimate of exactly 2 is the hysteresis band (flag must hold).
#
# Estimate contracts (k+1 = 3 bit pointers, modulo 8):
#   w_occ      = (wptr_after - gray_decode(r2w_s2_OLD)) mod 8   (W edges)
#   r_readable = (gray_decode(w2r_s2_OLD) - rptr_after) mod 8   (R edges)
# Gray table (0..7): 0:000 1:001 2:011 3:010 4:110 5:111 6:101 7:100.
# Reset: AF=0, AE=1.  Accepted writes D0..D7 in address order.
#
# Columns:
# (t, wclk, rclk, wen, ren, wacc, racc, wp, rp,
#  r2w1, r2w2, w2r1, w2r2, full, empty, w_occ, r_read, AF, AE, rdata)
# w_occ / r_read are None on steps where that domain does not clock.
# All wclk rows request a write; the only disabled read edge is t=22.
HAND_WAVE_AF = [
    (1,  1, 0, 1, 0, True,  None, 1, 0, 0, 0, 0, 0, 0, 1, 1, None, 0, 1, None),
    (2,  1, 0, 1, 0, True,  None, 2, 0, 0, 0, 0, 0, 0, 1, 2, None, 0, 1, None),
    (3,  1, 0, 1, 0, True,  None, 3, 0, 0, 0, 0, 0, 0, 1, 3, None, 1, 1, None),
    (4,  1, 0, 1, 0, True,  None, 4, 0, 0, 0, 0, 0, 1, 1, 4, None, 1, 1, None),
    (5,  0, 1, 0, 1, None, False, 4, 0, 0, 0, 6, 0, 1, 1, None, 0, 1, 1, None),
    (6,  1, 0, 1, 0, False, None, 4, 0, 0, 0, 6, 0, 1, 1, 4, None, 1, 1, None),
    (7,  0, 1, 0, 1, None, False, 4, 0, 0, 0, 6, 6, 1, 1, None, 0, 1, 1, None),
    (8,  1, 0, 1, 0, False, None, 4, 0, 0, 0, 6, 6, 1, 1, 4, None, 1, 1, None),
    (9,  0, 1, 0, 1, None, False, 4, 0, 0, 0, 6, 6, 1, 0, None, 4, 1, 0, None),
    (10, 0, 1, 0, 1, None, True,  4, 1, 0, 0, 6, 6, 1, 0, None, 3, 1, 0, "D0"),
    (11, 0, 1, 0, 1, None, True,  4, 2, 0, 0, 6, 6, 1, 0, None, 2, 1, 0, "D1"),
    (12, 0, 1, 0, 1, None, True,  4, 3, 0, 0, 6, 6, 1, 0, None, 1, 1, 1, "D2"),
    (13, 1, 0, 1, 0, False, None, 4, 3, 2, 0, 6, 6, 1, 0, 4, None, 1, 1, None),
    (14, 1, 0, 1, 0, False, None, 4, 3, 2, 2, 6, 6, 1, 0, 4, None, 1, 1, None),
    (15, 1, 0, 1, 0, False, None, 4, 3, 2, 2, 6, 6, 0, 0, 1, None, 0, 1, None),
    (16, 0, 1, 0, 1, None, True,  4, 4, 2, 2, 6, 6, 0, 1, None, 0, 0, 1, "D3"),
    (17, 1, 0, 1, 0, True,  None, 5, 4, 6, 2, 6, 6, 0, 1, 2, None, 0, 1, None),
    (19, 1, 0, 1, 0, True,  None, 6, 4, 6, 6, 6, 6, 0, 1, 3, None, 1, 1, None),
    (20, 0, 1, 0, 1, None, False, 6, 4, 6, 6, 5, 6, 0, 1, None, 0, 1, 1, None),
    (21, 0, 1, 0, 1, None, False, 6, 4, 6, 6, 5, 5, 0, 1, None, 0, 1, 1, None),
    (22, 0, 1, 0, 0, None, False, 6, 4, 6, 6, 5, 5, 0, 0, None, 2, 1, 1, None),
    (23, 1, 0, 1, 0, True,  None, 7, 4, 6, 6, 5, 5, 0, 0, 3, None, 1, 1, None),
    (24, 0, 1, 0, 1, None, True,  7, 5, 6, 6, 4, 5, 0, 0, None, 1, 1, 1, "D4"),
    (25, 0, 1, 0, 1, None, True,  7, 6, 6, 6, 4, 4, 0, 1, None, 0, 1, 1, "D5"),
    (26, 1, 0, 1, 0, True,  None, 0, 6, 5, 6, 4, 4, 1, 1, 4, None, 1, 1, None),
]


def _events_from_hand_af():
    """Build the Event stream for HAND_WAVE_AF; payloads D0.. go to accepted
    writes in acceptance order (address order here)."""
    events = []
    wi = 0
    for (t, wc, rc, we, re_, wacc, *_rest) in HAND_WAVE_AF:
        wdata = None
        if wc and we and wacc:
            wdata = f"D{wi}"
            wi += 1
        events.append(Event(t, bool(wc), bool(rc), bool(we), bool(re_), wdata))
    return events


class TestHandWaveformAlmostFlags(unittest.TestCase):
    """Beat-by-beat comparison against the hand-derived table above."""

    def _build(self):
        return AsyncFIFO(k=2,
                         almost_full_low=1, almost_full_high=3,
                         almost_empty_low=1, almost_empty_high=3)

    def test_exact_trace(self):
        fifo = self._build()
        records = fifo.run(_events_from_hand_af())
        self.assertEqual(len(records), len(HAND_WAVE_AF))
        for rec, row in zip(records, HAND_WAVE_AF):
            (t, wc, rc, we, re_, wacc, racc, wp, rp,
             r2w1, r2w2, w2r1, w2r2, full, empty, wocc, rread,
             af, ae, rdata) = row
            with self.subTest(t=t):
                self.assertEqual(rec.time, t)
                self.assertEqual((rec.wclk, rec.rclk), (bool(wc), bool(rc)))
                self.assertEqual((rec.wen, rec.ren), (bool(we), bool(re_)))
                self.assertEqual(rec.w_accept, wacc)
                self.assertEqual(rec.r_accept, racc)
                self.assertEqual((rec.wptr_after, rec.rptr_after), (wp, rp))
                self.assertEqual(rec.r2w_after, (r2w1, r2w2))
                self.assertEqual(rec.w2r_after, (w2r1, w2r2))
                self.assertEqual((rec.full_after, rec.empty_after),
                                 (bool(full), bool(empty)))
                self.assertEqual(rec.w_occ_after, wocc)
                self.assertEqual(rec.r_readable_after, rread)
                self.assertIs(rec.almost_full_after, bool(af))
                self.assertIs(rec.almost_empty_after, bool(ae))
                self.assertEqual(rec.rdata, rdata)

    def test_before_fields_and_estimate_arithmetic(self):
        """Flag registers carry over from the previous step (a non-clocking
        domain keeps its flag); each estimate is independently recomputed in
        this test from the sampled OLD stage-2 Gray and the local pointer,
        modulo 8 -- the model's flag helpers are never used for the expected
        value."""
        gray_to_bin = {g: b for b, g in GRAY_TABLE.items()}  # 0..15, k=2 uses 0..7
        records = self._build().run(_events_from_hand_af())

        # reset flag values on the very first edge
        self.assertFalse(records[0].almost_full_before)
        self.assertTrue(records[0].almost_empty_before)

        prev = records[0]
        for rec in records[1:]:
            with self.subTest(t=rec.time):
                # advisory flags are registers: each *_before equals the
                # register value left by the preceding time step
                self.assertIs(rec.almost_full_before, prev.almost_full_after)
                self.assertIs(rec.almost_empty_before,
                              prev.almost_empty_after)
            prev = rec

        # independently recompute the estimates for every edge
        for rec in records:
            with self.subTest(t=rec.time, side="W"):
                if rec.wclk:
                    s2_gray = rec.r2w_before[1]          # OLD stage 2
                    exp_b = (rec.wptr_before - gray_to_bin[s2_gray]) & 7
                    exp_a = (rec.wptr_after - gray_to_bin[s2_gray]) & 7
                    self.assertEqual(rec.w_occ_before, exp_b)
                    self.assertEqual(rec.w_occ_after, exp_a)
                else:
                    self.assertIsNone(rec.w_occ_before)
                    self.assertIsNone(rec.w_occ_after)
            with self.subTest(t=rec.time, side="R"):
                if rec.rclk:
                    s2_gray = rec.w2r_before[1]          # OLD stage 2
                    exp_b = (gray_to_bin[s2_gray] - rec.rptr_before) & 7
                    exp_a = (gray_to_bin[s2_gray] - rec.rptr_after) & 7
                    self.assertEqual(rec.r_readable_before, exp_b)
                    self.assertEqual(rec.r_readable_after, exp_a)
                else:
                    self.assertIsNone(rec.r_readable_before)
                    self.assertIsNone(rec.r_readable_after)

    def test_reset_values(self):
        fifo = self._build()
        self.assertFalse(fifo.almost_full)
        self.assertTrue(fifo.almost_empty)
        rec = fifo.run_one(Event(1, True, True))
        self.assertFalse(rec.almost_full_after)
        self.assertTrue(rec.almost_empty_after)

    def test_default_thresholds(self):
        fifo = AsyncFIFO(k=2)            # depth 4
        self.assertEqual((fifo.almost_full_low, fifo.almost_full_high), (3, 4))
        self.assertEqual((fifo.almost_empty_low, fifo.almost_empty_high),
                         (0, 1))

    def test_hysteresis_band_keeps_rising_flag_set(self):
        """AF=1 must stay asserted while w_occ sits in the band (estimate 2),
        and clear only once the estimate reaches low (1).  Fully hand-derived
        sequence for k=2, AF thresholds (1, 3); disabled write edges are used
        to advance the r2w synchronizer without accepting more data."""
        fifo = self._build()
        # fill to full: AF sets at estimate 3 (t3), full sets at t4
        for t in (1, 2, 3, 4):
            fifo.run_one(Event(t, True, False, wen=True, wdata=f"D{t-1}"))
        self.assertTrue(fifo.almost_full)
        self.assertTrue(fifo.full)
        # empty stays high through two sync stages: t5..t7 reject, t8/t9 accept
        for t in (5, 6, 7):
            self.assertFalse(
                fifo.run_one(Event(t, False, True, ren=True)).r_accept)
        for t, addr in ((8, 1), (9, 2)):
            r = fifo.run_one(Event(t, False, True, ren=True))
            self.assertTrue(r.r_accept)
            self.assertEqual(r.rptr_after, addr)
        self.assertEqual(fifo.rptr, 2)          # true occupancy now 2
        # disabled write edges march the stale read Gray through r2w;
        # the estimate uses the OLD stage-2 value, so it lags two edges
        r10 = fifo.run_one(Event(10, True, False))
        self.assertEqual(r10.r2w_after, (3, 0))
        self.assertEqual(r10.w_occ_after, 4)
        self.assertTrue(r10.full_after)
        self.assertTrue(r10.almost_full_after)
        r11 = fifo.run_one(Event(11, True, False))
        self.assertEqual(r11.r2w_after, (3, 3))
        self.assertEqual(r11.w_occ_after, 4)    # s2 still old on THIS edge
        self.assertTrue(r11.full_after)
        r12 = fifo.run_one(Event(12, True, False))
        self.assertEqual(r12.w_occ_after, 2)    # enters the hysteresis band
        self.assertFalse(r12.full_after)        # exact full deasserts
        self.assertTrue(r12.almost_full_after)  # ... but the hint HOLDS
        # one more accepted read -> true occupancy 1
        r13 = fifo.run_one(Event(13, False, True, ren=True))
        self.assertTrue(r13.r_accept)
        self.assertEqual(r13.rptr_after, 3)
        # rgray(3)=2 needs another two write edges to reach stage 2
        r14 = fifo.run_one(Event(14, True, False))
        self.assertEqual(r14.r2w_after, (2, 3))
        self.assertEqual(r14.w_occ_after, 2)    # band: still held
        self.assertTrue(r14.almost_full_after)
        r15 = fifo.run_one(Event(15, True, False))
        self.assertEqual(r15.r2w_after, (2, 2))
        self.assertEqual(r15.w_occ_after, 2)    # old s2 still decodes r=2
        self.assertTrue(r15.almost_full_after)
        r16 = fifo.run_one(Event(16, True, False))
        self.assertEqual(r16.w_occ_after, 1)    # finally <= low
        self.assertFalse(r16.almost_full_after)  # and only now it clears


class TestEstimateIsLocalAndConservative(unittest.TestCase):
    def test_write_estimate_overstates_when_reads_lag(self):
        """After real occupancy drops, stale r2w keeps w_occ high (pessimistic);
        it never understates relative to the synchronized value."""
        fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3,
                         almost_empty_low=1, almost_empty_high=3)
        for t in (1, 2, 3, 4):
            fifo.run_one(Event(t, True, False, wen=True, wdata=t))
        # one read accepted: true occupancy 3, write domain still estimates 4
        for t in (5, 6, 7):
            fifo.run_one(Event(t, False, True, ren=True))
        rec = fifo.run_one(Event(8, True, False))
        self.assertEqual(rec.w_occ_after, 4)       # stale s2 still decodes r=0

    def test_read_estimate_understates_when_writes_lag(self):
        fifo = AsyncFIFO(
            k=2, almost_full_low=1, almost_full_high=3,
            almost_empty_low=1, almost_empty_high=3)
        fifo.run_one(Event(1, True, False, wen=True, wdata="a"))
        # first read edge: data exists but w2r_s2 is still 0 -> estimate 0
        rec = fifo.run_one(Event(2, False, True, ren=True))
        self.assertEqual(rec.r_readable_after, 0)
        self.assertTrue(rec.almost_empty_after)

    def test_estimate_lines_never_read_memory_or_queue(self):
        """Structural guarantee for the new hint logic."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "afifo", "model.py"), encoding="utf-8") as f:
            source = f.read()
        for line in source.splitlines():
            if "w_occ" in line or "r_read" in line:
                self.assertNotIn("_mem", line)
                self.assertNotIn("len(", line)
                self.assertNotIn("queue", line)


class TestAlmostFlagsAdvisoryOnly(unittest.TestCase):
    def test_thresholds_do_not_change_acceptance(self):
        """Two FIFO instances with wildly different thresholds must accept and
        reject identically; only the advisory flags may differ."""
        schedule = make_schedule(2, 400, w_period=2, r_period=3, seed=2026)
        events = [Event(t, wc, rc, we, re_, data)
                  for (t, wc, rc, we, re_, data) in schedule]
        a = AsyncFIFO(k=2)
        b = AsyncFIFO(k=2, almost_full_low=0, almost_full_high=4,
                      almost_empty_low=0, almost_empty_high=4)
        ra, rb = a.run(events), b.run([
            Event(t, wc, rc, we, re_, data)
            for (t, wc, rc, we, re_, data) in schedule])
        differ = False
        for x, y in zip(ra, rb):
            self.assertEqual(x.w_accept, y.w_accept)
            self.assertEqual(x.r_accept, y.r_accept)
            self.assertEqual(x.full_after, y.full_after)
            self.assertEqual(x.empty_after, y.empty_after)
            self.assertEqual(x.wptr_after, y.wptr_after)
            self.assertEqual(x.rptr_after, y.rptr_after)
            differ |= (x.almost_full_after != y.almost_full_after or
                       x.almost_empty_after != y.almost_empty_after)
        self.assertTrue(differ)  # otherwise the test configuration is vacuous

    def test_almost_full_does_not_block_write(self):
        # (1,2): AF asserts at estimate 2, but the write is still accepted.
        fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=2)
        r1 = fifo.run_one(Event(1, True, False, wen=True, wdata=1))
        self.assertFalse(r1.almost_full_after)
        r2 = fifo.run_one(Event(2, True, False, wen=True, wdata=2))
        self.assertTrue(r2.almost_full_after)
        self.assertTrue(r2.w_accept)             # hint, not a gate
        self.assertFalse(r2.full_after)

    def test_almost_empty_does_not_block_read(self):
        # AE (low=2, high=3): AE sets while readable data still exists, but
        # acceptance is governed solely by empty.
        fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3,
                         almost_empty_low=2, almost_empty_high=3)
        for t, d in ((1, "a"), (2, "b"), (3, "c")):
            fifo.run_one(Event(t, True, False, wen=True, wdata=d))
        # empty deasserts only at the END of the third sync edge (t6); the
        # first three read requests are rejected even with data in memory
        for t in (4, 5, 6):
            self.assertFalse(
                fifo.run_one(Event(t, False, True, ren=True)).r_accept)
        out = []
        for t in (7, 8, 9):
            r = fifo.run_one(Event(t, False, True, ren=True))
            self.assertTrue(r.r_accept)
            out.append(r.rdata)
        self.assertEqual(out, ["a", "b", "c"])
        # AE asserted while readable data still remains, never gating a read
        self.assertTrue(fifo.almost_empty)
        # the NEXT edge: empty (not AE) is what finally stops the read
        r10 = fifo.run_one(Event(10, False, True, ren=True))
        self.assertFalse(r10.r_accept)


class TestThresholdContract(unittest.TestCase):
    def test_equal_thresholds_rejected(self):
        with self.assertRaisesRegex(AsyncFIFOError, "almost_full_low"):
            AsyncFIFO(k=2, almost_full_low=2, almost_full_high=2)
        with self.assertRaisesRegex(AsyncFIFOError, "almost_empty_low"):
            AsyncFIFO(k=2, almost_empty_low=2, almost_empty_high=2)

    def test_low_above_high_rejected(self):
        with self.assertRaisesRegex(AsyncFIFOError, "strictly less"):
            AsyncFIFO(k=2, almost_full_low=3, almost_full_high=2)
        with self.assertRaisesRegex(AsyncFIFOError, "strictly less"):
            AsyncFIFO(k=2, almost_empty_low=3, almost_empty_high=2)

    def test_endpoint_thresholds_accepted(self):
        f = AsyncFIFO(k=2, almost_full_low=0, almost_full_high=4,
                      almost_empty_low=0, almost_empty_high=4)
        self.assertEqual(f.almost_full_low, 0)
        self.assertEqual(f.almost_full_high, 4)
        # k=1 (depth 2) endpoints still leave a valid low<high pair
        AsyncFIFO(k=1, almost_full_low=0, almost_full_high=2,
                  almost_empty_low=0, almost_empty_high=2)
        AsyncFIFO(k=1, almost_full_low=0, almost_full_high=1,
                  almost_empty_low=1, almost_empty_high=2)

    def test_out_of_range_rejected(self):
        for kw in (
            dict(almost_full_low=-1, almost_full_high=3),
            dict(almost_full_low=1, almost_full_high=5),   # > depth 4
            dict(almost_empty_low=-1, almost_empty_high=1),
            dict(almost_empty_low=0, almost_empty_high=5),
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, **kw)

    def test_bools_and_non_ints_rejected(self):
        for kw in (
            dict(almost_full_low=True, almost_full_high=3),
            dict(almost_full_low=1.0, almost_full_high=3),
            dict(almost_empty_low=0, almost_empty_high="2"),
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, **kw)

    def test_threshold_groups_independent(self):
        # bad full pair, valid empty pair
        with self.assertRaisesRegex(AsyncFIFOError, "almost_full"):
            AsyncFIFO(k=2, almost_full_low=3, almost_full_high=3,
                      almost_empty_low=0, almost_empty_high=1)
        # valid full pair, bad empty pair
        with self.assertRaisesRegex(AsyncFIFOError, "almost_empty"):
            AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3,
                      almost_empty_low=2, almost_empty_high=2)


class TestAlmostFlagsStoppedClock(unittest.TestCase):
    def _fifo(self):
        return AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3,
                         almost_empty_low=1, almost_empty_high=3)

    def test_write_stop_freezes_hint_and_estimate_is_none(self):
        fifo = self._fifo()
        for t, d in ((1, 1), (2, 2), (3, 3)):
            fifo.run_one(Event(t, True, False, wen=True, wdata=d))
        self.assertTrue(fifo.almost_full)
        snap = fifo.snapshot()
        for t in range(4, 12):
            rec = fifo.run_one(Event(t, False, True, ren=True))
            self.assertIsNone(rec.w_occ_after)
            self.assertIsNone(rec.w_occ_before)
            self.assertEqual(rec.almost_full_after, snap["almost_full"])
            self.assertEqual(rec.wptr_after, snap["wptr"])
            self.assertEqual(rec.r2w_after, snap["r2w"])

    def test_read_stop_freezes_hint_and_estimate_is_none(self):
        fifo = self._fifo()
        fifo.run_one(Event(1, True, False, wen=True, wdata=1))
        snap = fifo.snapshot()
        self.assertTrue(fifo.almost_empty)
        for t in range(2, 10):
            rec = fifo.run_one(Event(t, True, False))
            self.assertIsNone(rec.r_readable_after)
            self.assertIsNone(rec.r_readable_before)
            self.assertEqual(rec.almost_empty_after, snap["almost_empty"])
            self.assertEqual(rec.rptr_after, snap["rptr"])
            self.assertEqual(rec.w2r_after, snap["w2r"])


class TestAlmostFlagsDifferential(unittest.TestCase):
    """Beat-by-beat differential vs the independent RefFIFO (which never
    imports afifo), including estimates and hysteresis flags."""

    def _pairs(self, d):
        candidates = [(d - 1, d), (0, 1), (0, d), (1, min(3, d))]
        out = []
        for lo, hi in candidates:
            if 0 <= lo < hi <= d and (lo, hi) not in out:
                out.append((lo, hi))
        return out

    def _diff(self, k, wp, rp, n, seed, af, ae, w_stops=(), r_stops=()):
        schedule = make_schedule(
            k, n, w_period=wp, r_period=rp, seed=seed,
            stop_windows=w_stops, r_stop_windows=r_stops)
        dut = AsyncFIFO(k, almost_full_low=af[0], almost_full_high=af[1],
                        almost_empty_low=ae[0], almost_empty_high=ae[1])
        ref = RefFIFO(k, af_low=af[0], af_high=af[1],
                      ae_low=ae[0], ae_high=ae[1])
        records = dut.run([Event(t, wc, rc, we, re_, data)
                           for (t, wc, rc, we, re_, data) in schedule])
        for ev, rec in zip(schedule, records):
            t, wc, rc, we, re_, data = ev
            exp = ref.step(wc, rc, we, re_, data)
            with self.subTest(k=k, af=af, ae=ae, t=t):
                self.assertIs(rec.almost_full_after, exp["almost_full"])
                self.assertIs(rec.almost_empty_after, exp["almost_empty"])
                self.assertEqual(rec.w_occ_after, exp["w_occ_after"])
                self.assertEqual(rec.w_occ_before, exp["w_occ_before"])
                self.assertEqual(rec.r_readable_after,
                                 exp["r_readable_after"])
                self.assertEqual(rec.r_readable_before,
                                 exp["r_readable_before"])
                # old registers must still match
                self.assertIs(rec.full_after, exp["full"])
                self.assertIs(rec.empty_after, exp["empty"])
                self.assertEqual(rec.w_accept, exp["w_accept"])
                self.assertEqual(rec.r_accept, exp["r_accept"])
                self.assertEqual(rec.r2w_after, exp["r2w"])
                self.assertEqual(rec.w2r_after, exp["w2r"])

    def test_threshold_matrix_ratios_and_wrap(self):
        for k in (1, 2, 3, 4, 5):
            d = 1 << k
            pairs = self._pairs(d)
            for af in pairs:
                for ae in pairs:
                    self._diff(k, 2, 3, 200, seed=1000 + 31 * k + af[1] * 5
                               + ae[1], af=af, ae=ae)

    def test_long_wrap_k1(self):
        # k=1 pointers wrap every 2 accepts; estimates modulo 4 throughout.
        self._diff(1, 1, 1, 600, seed=4242, af=(0, 2), ae=(0, 2))
        self._diff(1, 2, 3, 600, seed=4243, af=(0, 1), ae=(1, 2))

    def test_long_wrap_k5_with_stops(self):
        self._diff(5, 2, 3, 4000, seed=505, af=(7, 24), ae=(3, 9),
                   w_stops=((500, 900),), r_stops=((2000, 2600),))

    def test_no_chatter_inside_band(self):
        """Across long runs a flag edge may occur only when the estimate
        crosses its own set/clear threshold -- never from a band value."""
        for k, af, ae in [
            (2, (1, 3), (1, 3)),
            (3, (2, 6), (1, 5)),
            (4, (3, 9), (2, 7)),
        ]:
            schedule = make_schedule(k, 1500, w_period=2, r_period=3,
                                     seed=7 * k + 1)
            fifo = AsyncFIFO(k, almost_full_low=af[0],
                             almost_full_high=af[1],
                             almost_empty_low=ae[0],
                             almost_empty_high=ae[1])
            records = fifo.run([Event(t, wc, rc, we, re_, data)
                                for (t, wc, rc, we, re_, data) in schedule])
            prev = records[0]
            for rec in records[1:]:
                with self.subTest(k=k, t=rec.time):
                    if rec.wclk and (prev.almost_full_after
                                     != rec.almost_full_after):
                        if rec.almost_full_after:
                            self.assertGreaterEqual(rec.w_occ_after, af[1])
                        else:
                            self.assertLessEqual(rec.w_occ_after, af[0])
                    if rec.rclk and (prev.almost_empty_after
                                     != rec.almost_empty_after):
                        if rec.almost_empty_after:
                            self.assertLessEqual(rec.r_readable_after, ae[0])
                        else:
                            self.assertGreaterEqual(rec.r_readable_after,
                                                    ae[1])
                    # an estimate strictly inside the band MUST hold the flag
                    if rec.wclk and af[0] < rec.w_occ_after < af[1]:
                        self.assertEqual(rec.almost_full_after,
                                         prev.almost_full_after)
                    if rec.rclk and ae[0] < rec.r_readable_after < ae[1]:
                        self.assertEqual(rec.almost_empty_after,
                                         prev.almost_empty_after)
                prev = rec


class TestLegacyInterfaceUnchanged(unittest.TestCase):
    def test_old_positional_edge_record_fields(self):
        fifo = AsyncFIFO(k=2)
        rec = fifo.run_one(Event(1, True, False, wen=True, wdata="z"))
        # all original 25 fields still exist under their original names
        old_names = [
            "time", "wclk", "rclk", "wen", "ren", "w_accept", "r_accept",
            "wptr_before", "wptr_after", "rptr_before", "rptr_after",
            "wgray_before", "wgray_after", "rgray_before", "rgray_after",
            "full_before", "full_after", "empty_before", "empty_after",
            "r2w_before", "w2r_before", "r2w_after", "w2r_after",
            "wdata", "rdata",
        ]
        values = [getattr(rec, n) for n in old_names]
        self.assertEqual(values[0], 1)
        self.assertEqual(values[-2], "z")
        # tuple length grew but old positions/indices are untouched
        self.assertEqual(len(rec), 33)
        self.assertEqual(rec[0], 1)
        self.assertEqual(rec[23], "z")       # wdata keeps its old index
        self.assertIsNone(rec[24])           # rdata: nothing read

    def test_legacy_constructor_and_calls(self):
        fifo = AsyncFIFO(3)
        recs = fifo.run([Event(1, True, False), Event(2, False, True)])
        self.assertEqual(len(recs), 2)
        # new snapshot keys are additive
        for key in ("almost_full", "almost_empty"):
            self.assertIn(key, fifo.snapshot())


if __name__ == "__main__":
    unittest.main(verbosity=2)
