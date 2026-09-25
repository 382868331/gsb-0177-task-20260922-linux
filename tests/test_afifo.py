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
        self.assertFalse(fifo.almost_full)    # almost_full resets to False
        self.assertTrue(fifo.almost_empty)    # almost_empty resets to True
        self.assertEqual(fifo.accepted_writes, 0)
        self.assertEqual(fifo.accepted_reads, 0)
        self.assertEqual(fifo.snapshot()["r2w"], (0, 0))
        self.assertEqual(fifo.snapshot()["w2r"], (0, 0))

    def test_reset_restores_almost_flags_after_activity(self):
        fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=2,
                         almost_empty_low=0, almost_empty_high=1)
        fifo.run([Event(t, True, False, wen=True, wdata=t)
                  for t in range(1, 4)])
        self.assertTrue(fifo.almost_full)
        fifo.reset()
        self.assertFalse(fifo.almost_full)
        self.assertTrue(fifo.almost_empty)
        self.assertEqual(fifo.steps, ())

    def test_default_thresholds(self):
        for k in range(1, 6):
            fifo = AsyncFIFO(k=k)
            d = 1 << k
            self.assertEqual((fifo.almost_full_low, fifo.almost_full_high),
                             (d - 1, d))
            self.assertEqual((fifo.almost_empty_low, fifo.almost_empty_high),
                             (0, 1))

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

    def _diff_run(self, k, wp, rp, n, seed=1, w_stops=(), r_stops=(),
                  thresholds=None):
        thresholds = thresholds or {}
        schedule = make_schedule(
            k, n, w_period=wp, r_period=rp, seed=seed,
            stop_windows=w_stops, r_stop_windows=r_stops,
        )
        dut = AsyncFIFO(k=k, **thresholds)
        ref = RefFIFO(k=k, **thresholds)
        af_low, af_high = ref.af_low, ref.af_high
        ae_low, ae_high = ref.ae_low, ref.ae_high
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
                # advisory almost-* interface
                self.assertEqual(rec.w_used_est, exp["w_est"])
                self.assertEqual(rec.r_avail_est, exp["r_est"])
                self.assertIs(rec.almost_full_after, exp["almost_full"])
                self.assertIs(rec.almost_empty_after, exp["almost_empty"])
                # hysteresis: strictly inside the band the flag must hold
                if exp["w_est"] is not None and \
                        af_low < exp["w_est"] < af_high:
                    self.assertIs(rec.almost_full_after,
                                  rec.almost_full_before)
                if exp["r_est"] is not None and \
                        ae_low < exp["r_est"] < ae_high:
                    self.assertIs(rec.almost_empty_after,
                                  rec.almost_empty_before)
                # estimates never come from the abstract queue length, but
                # they must stay consistent with its [0, depth] bound
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

    def test_custom_thresholds_differential(self):
        # non-default hysteresis windows, both narrow and wide bands
        self._diff_run(2, 2, 3, 300, seed=5, thresholds={
            "almost_full_low": 1, "almost_full_high": 3,
            "almost_empty_low": 0, "almost_empty_high": 2})
        self._diff_run(3, 3, 2, 300, seed=6, thresholds={
            "almost_full_low": 2, "almost_full_high": 6,
            "almost_empty_low": 1, "almost_empty_high": 5})
        self._diff_run(1, 1, 1, 300, seed=7, thresholds={
            "almost_full_low": 0, "almost_full_high": 2,
            "almost_empty_low": 0, "almost_empty_high": 2})
        self._diff_run(4, 2, 5, 400, seed=8, thresholds={
            "almost_full_low": 3, "almost_full_high": 13,
            "almost_empty_low": 2, "almost_empty_high": 11})

    def test_stopped_clocks(self):
        self._diff_run(2, 2, 3, 240, seed=9,
                       w_stops=((40, 80),), r_stops=((140, 190),))
        self._diff_run(3, 3, 2, 240, seed=11,
                       w_stops=((10, 60), (120, 150)))
        # stopped clocks with custom thresholds: hints freeze with the domain
        self._diff_run(2, 2, 3, 240, seed=12,
                       w_stops=((40, 80),), r_stops=((140, 190),),
                       thresholds={"almost_full_low": 1,
                                   "almost_full_high": 3,
                                   "almost_empty_low": 0,
                                   "almost_empty_high": 2})

    def test_pointer_wrap_many_times(self):
        # k=1 pointers wrap every 2 accepted accesses; 400 steps => wraps.
        self._diff_run(1, 1, 1, 400, seed=42)
        # k=5 (depth 32) needs 64 accepts per wrap; long run covers it.
        self._diff_run(5, 2, 3, 4000, seed=77)
        # wrap with non-default thresholds
        self._diff_run(1, 1, 1, 400, seed=43, thresholds={
            "almost_full_low": 0, "almost_full_high": 1,
            "almost_empty_low": 0, "almost_empty_high": 1})


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


# Hand-derived k=2 (depth 4) waveform with hysteresis thresholds
# almost_full low/high = 1/3, almost_empty low/high = 0/2.  Write clock on
# even t, read clock on t divisible by 3 (unequal frequencies, coincident
# edges at t = 6, 12, ...).  Each row:
# (time, wclk, rclk, wen, ren, wdata, w_accept, r_accept, wptr, rptr,
#  full, empty, almost_full, almost_empty, w_used_est, r_avail_est,
#  r2w, w2r, expected rdata)
AF_HAND_WAVE = [
    (2,  1, 0, 1, 0, "A",  True,  None,  1, 0, 0, 1, 0, 1, 1,    None, (0, 0), (0, 0), None),
    (3,  0, 1, 0, 1, None, None,  False, 1, 0, 0, 1, 0, 1, None, 0,    (0, 0), (1, 0), None),
    (4,  1, 0, 1, 0, "B",  True,  None,  2, 0, 0, 1, 0, 1, 2,    None, (0, 0), (1, 0), None),
    (6,  1, 1, 1, 1, "C",  True,  False, 3, 0, 0, 1, 1, 1, 3,    0,    (0, 0), (3, 1), None),
    (8,  1, 0, 1, 0, "D",  True,  None,  4, 0, 1, 1, 1, 1, 4,    None, (0, 0), (3, 1), None),
    (9,  0, 1, 0, 1, None, None,  False, 4, 0, 1, 0, 1, 1, None, 1,    (0, 0), (6, 3), None),
    (10, 1, 0, 1, 0, "E",  False, None,  4, 0, 1, 0, 1, 1, 4,    None, (0, 0), (6, 3), None),
    (12, 1, 1, 1, 1, "F",  False, True,  4, 1, 1, 0, 1, 1, 4,    1,    (0, 0), (6, 6), "A"),
    (14, 1, 0, 1, 0, "G",  False, None,  4, 1, 1, 0, 1, 1, 4,    None, (1, 0), (6, 6), None),
    (15, 0, 1, 0, 1, None, None,  True,  4, 2, 1, 0, 1, 0, None, 2,    (1, 0), (6, 6), "B"),
    (16, 1, 0, 1, 0, "H",  False, None,  4, 2, 1, 0, 1, 0, 4,    None, (3, 1), (6, 6), None),
    (18, 1, 1, 1, 1, "I",  False, True,  4, 3, 0, 0, 1, 0, 3,    1,    (3, 3), (6, 6), "C"),
    (20, 1, 0, 1, 0, "J",  True,  None,  5, 3, 0, 0, 1, 0, 3,    None, (2, 3), (6, 6), None),
    (21, 0, 1, 0, 1, None, None,  True,  5, 4, 0, 1, 1, 1, None, 0,    (2, 3), (7, 6), "D"),
    (22, 1, 0, 1, 0, "K",  True,  None,  6, 4, 1, 1, 1, 1, 4,    None, (6, 2), (7, 6), None),
    (24, 1, 1, 1, 1, "L",  False, False, 6, 4, 0, 1, 1, 1, 3,    0,    (6, 6), (5, 7), None),
    (26, 1, 0, 1, 0, "M",  True,  None,  7, 4, 0, 1, 1, 1, 3,    None, (6, 6), (5, 7), None),
    (27, 0, 1, 0, 1, None, None,  False, 7, 4, 0, 0, 1, 1, None, 1,    (6, 6), (4, 5), None),
    (28, 1, 0, 1, 0, "N",  True,  None,  0, 4, 1, 0, 1, 1, 4,    None, (6, 6), (4, 5), None),
    (30, 1, 1, 1, 1, "O",  False, True,  0, 5, 1, 0, 1, 1, 4,    1,    (6, 6), (0, 4), "J"),
    (32, 1, 0, 0, 0, None, False, None,  0, 5, 1, 0, 1, 1, 4,    None, (7, 6), (0, 4), None),
    (33, 0, 1, 0, 1, None, None,  True,  0, 6, 1, 0, 1, 1, None, 1,    (7, 6), (0, 0), "K"),
    (34, 1, 0, 0, 0, None, False, None,  0, 6, 1, 0, 1, 1, 4,    None, (5, 7), (0, 0), None),
    (36, 1, 1, 0, 1, None, False, True,  0, 7, 0, 0, 1, 1, 3,    1,    (5, 5), (0, 0), "M"),
    (38, 1, 0, 0, 0, None, False, None,  0, 7, 0, 0, 1, 1, 2,    None, (4, 5), (0, 0), None),
    (39, 0, 1, 0, 1, None, None,  True,  0, 0, 0, 1, 1, 1, None, 0,    (4, 5), (0, 0), "N"),
    (40, 1, 0, 0, 0, None, False, None,  0, 0, 0, 1, 1, 1, 2,    None, (0, 4), (0, 0), None),
    (42, 1, 1, 0, 1, None, False, False, 0, 0, 0, 1, 0, 1, 1,    0,    (0, 0), (0, 0), None),
]


class TestAlmostHandWaveform(unittest.TestCase):
    """Per-cycle comparison against the hand-derived k=2 hysteresis wave."""

    def test_exact_trace_with_estimates_and_flags(self):
        fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3,
                         almost_empty_low=0, almost_empty_high=2)
        events = [Event(t, bool(wc), bool(rc), bool(we), bool(re), data)
                  for (t, wc, rc, we, re, data, *_rest) in AF_HAND_WAVE]
        records = fifo.run(events)
        self.assertEqual(len(records), len(AF_HAND_WAVE))
        for rec, row in zip(records, AF_HAND_WAVE):
            (t, wc, rc, we, re, data, wacc, racc, wp, rp, full, empty,
             af, ae, w_est, r_est, r2w, w2r, rdata) = row
            with self.subTest(time=t):
                self.assertEqual(rec.w_accept, wacc)
                self.assertEqual(rec.r_accept, racc)
                self.assertEqual(rec.wptr_after, wp)
                self.assertEqual(rec.rptr_after, rp)
                self.assertIs(rec.full_after, bool(full))
                self.assertIs(rec.empty_after, bool(empty))
                self.assertIs(rec.almost_full_after, bool(af))
                self.assertIs(rec.almost_empty_after, bool(ae))
                self.assertEqual(rec.w_used_est, w_est)
                self.assertEqual(rec.r_avail_est, r_est)
                self.assertEqual(rec.r2w_after, r2w)
                self.assertEqual(rec.w2r_after, w2r)
                self.assertEqual(rec.rdata, rdata)

    def test_hysteresis_edges_anchored(self):
        """af sets at est=3 (t=6), holds at est=2 (t=38/40), clears at
        est=1 (t=42); ae clears at est=2 (t=15), holds at est=1 (t=18)."""
        fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3,
                         almost_empty_low=0, almost_empty_high=2)
        events = [Event(t, bool(wc), bool(rc), bool(we), bool(re), data)
                  for (t, wc, rc, we, re, data, *_rest) in AF_HAND_WAVE]
        records = fifo.run(events)
        at = {r.time: r for r in records}
        self.assertFalse(at[2].almost_full_after)    # est=1 <= low
        self.assertFalse(at[4].almost_full_after)    # est=2 in band, holds
        self.assertTrue(at[6].almost_full_after)     # est=3 >= high: set
        self.assertTrue(at[38].almost_full_after)    # est=2 in band, holds
        self.assertTrue(at[40].almost_full_after)    # est=2 in band, holds
        self.assertFalse(at[42].almost_full_after)   # est=1 <= low: clear
        self.assertTrue(at[12].almost_empty_after)   # est=1 in band, holds
        self.assertFalse(at[15].almost_empty_after)  # est=2 >= high: clear
        self.assertFalse(at[18].almost_empty_after)  # est=1 in band, holds
        self.assertTrue(at[21].almost_empty_after)   # est=0 <= low: set

    def test_estimates_are_conservative_not_queue_length(self):
        """Write domain overestimates / read domain underestimates vs the
        true occupancy because each uses the stale synchronized peer."""
        fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3,
                         almost_empty_low=0, almost_empty_high=2)
        events = [Event(t, bool(wc), bool(rc), bool(we), bool(re), data)
                  for (t, wc, rc, we, re, data, *_rest) in AF_HAND_WAVE]
        records = fifo.run(events)
        # independent true occupancy from accepted operations only
        occ = 0
        for rec in records:
            occ += (1 if rec.w_accept else 0) - (1 if rec.r_accept else 0)
            if rec.w_used_est is not None:
                self.assertGreaterEqual(rec.w_used_est, occ)
            if rec.r_avail_est is not None:
                self.assertLessEqual(rec.r_avail_est, occ)


class TestAlmostThresholdContract(unittest.TestCase):
    def test_equal_thresholds_rejected(self):
        with self.assertRaisesRegex(AsyncFIFOError, "almost_full"):
            AsyncFIFO(k=2, almost_full_low=2, almost_full_high=2)
        with self.assertRaisesRegex(AsyncFIFOError, "almost_empty"):
            AsyncFIFO(k=2, almost_empty_low=1, almost_empty_high=1)

    def test_inverted_thresholds_rejected(self):
        with self.assertRaisesRegex(AsyncFIFOError, "almost_full"):
            AsyncFIFO(k=2, almost_full_low=3, almost_full_high=1)
        with self.assertRaisesRegex(AsyncFIFOError, "almost_empty"):
            AsyncFIFO(k=2, almost_empty_low=4, almost_empty_high=0)

    def test_out_of_range_rejected(self):
        for bad in (-1, 5, 100):
            with self.subTest(bad=bad):
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, almost_full_low=bad)
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, almost_full_high=bad)
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, almost_empty_low=bad)
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, almost_empty_high=bad)

    def test_non_int_and_bool_rejected(self):
        for bad in (True, False, 1.0, "1", 2.5):
            with self.subTest(bad=bad):
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, almost_full_low=bad)
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, almost_empty_high=bad)

    def test_threshold_endpoints_allowed(self):
        # 0 and depth are both legal threshold values
        fifo = AsyncFIFO(k=2, almost_full_low=0, almost_full_high=4,
                         almost_empty_low=0, almost_empty_high=4)
        self.assertEqual((fifo.almost_full_low, fifo.almost_full_high),
                         (0, 4))
        self.assertEqual((fifo.almost_empty_low, fifo.almost_empty_high),
                         (0, 4))
        # k=1 endpoints: high == depth == 2
        fifo = AsyncFIFO(k=1, almost_full_low=0, almost_full_high=2,
                         almost_empty_low=0, almost_empty_high=2)
        self.assertEqual(fifo.almost_full_high, 2)

    def test_partial_defaults(self):
        fifo = AsyncFIFO(k=3, almost_full_low=2)
        self.assertEqual((fifo.almost_full_low, fifo.almost_full_high),
                         (2, 8))
        fifo = AsyncFIFO(k=3, almost_empty_high=5)
        self.assertEqual((fifo.almost_empty_low, fifo.almost_empty_high),
                         (0, 5))


class TestAlmostStoppedClock(unittest.TestCase):
    def test_write_clock_stop_freezes_almost_full(self):
        fifo = AsyncFIFO(k=2, almost_full_low=1, almost_full_high=3)
        fifo.run([Event(t, True, False, wen=True, wdata=t)
                  for t in (1, 2, 3)])
        self.assertTrue(fifo.almost_full)
        # long read-only stretch: write-domain hint and estimate freeze
        for t in range(4, 40):
            rec = fifo.run_one(Event(t, False, True, ren=(t % 2 == 0)))
            self.assertIsNone(rec.w_used_est)
            self.assertTrue(rec.almost_full_after)
            self.assertTrue(rec.almost_full_before)
        self.assertTrue(fifo.almost_full)

    def test_read_clock_stop_freezes_almost_empty(self):
        fifo = AsyncFIFO(k=2, almost_empty_low=0, almost_empty_high=2)
        fifo.run_one(Event(1, True, False, wen=True, wdata="x"))
        self.assertTrue(fifo.almost_empty)
        # long write-only stretch: read-domain hint and estimate freeze
        for t in range(2, 40):
            rec = fifo.run_one(Event(t, True, False, wen=(t % 3 == 0),
                                     wdata=t if t % 3 == 0 else None))
            self.assertIsNone(rec.r_avail_est)
            self.assertTrue(rec.almost_empty_after)
        self.assertTrue(fifo.almost_empty)


class TestAlmostHysteresisNoDither(unittest.TestCase):
    def test_occupancy_bouncing_in_band_never_toggles_almost_full(self):
        # k=3, band (2, 5): drive w_used_est 3 -> 4 -> 3 -> 4 ... and check
        # the flag never changes while the estimate stays inside the band.
        fifo = AsyncFIFO(k=3, almost_full_low=2, almost_full_high=5)
        t = 0
        # warmup: 3 accepted writes (est 1, 2, 3), then clear empty's view
        # (two-stage sync plus the old-flag edge: three read edges)
        for _ in range(3):
            t += 1
            fifo.run_one(Event(t, True, False, wen=True, wdata=t))
        for _ in range(3):
            t += 1
            fifo.run_one(Event(t, False, True))
        est_seen = set()
        for _ in range(6):
            t += 1
            rec = fifo.run_one(Event(t, True, False, wen=True, wdata=t))
            est_seen.add(rec.w_used_est)             # est rises to 4
            t += 1
            fifo.run_one(Event(t, False, True, ren=True))   # drain one
            for _ in range(3):                       # sync rptr into r2w
                t += 1                               # (old s2 => 3 edges)
                rec = fifo.run_one(Event(t, True, False))
                est_seen.add(rec.w_used_est)         # est falls back to 3
        self.assertEqual(est_seen, {3, 4})           # oscillated in band
        self.assertFalse(fifo.almost_full)           # and never set
        self.assertTrue(all(
            r.almost_full_after == r.almost_full_before
            for r in fifo.steps
            if r.w_used_est is not None and 2 < r.w_used_est < 5))

    def test_readable_bouncing_in_band_never_toggles_almost_empty(self):
        # k=3, band (2, 5): fill 6, let the read domain see all 6, then
        # drain one per read edge: r_avail_est walks 5, 4, 3, 2 and the flag
        # must hold while the estimate is strictly inside (2, 5).
        fifo = AsyncFIFO(k=3, almost_empty_low=2, almost_empty_high=5)
        t = 0
        for _ in range(6):
            t += 1
            fifo.run_one(Event(t, True, False, wen=True, wdata=t))
        for _ in range(3):                           # sync wptr into w2r
            t += 1
            fifo.run_one(Event(t, False, True))
        rec = fifo.run_one(Event(10, False, True, ren=True))
        self.assertEqual(rec.r_avail_est, 5)
        self.assertFalse(rec.almost_empty_after)     # est >= high: cleared
        rec = fifo.run_one(Event(11, False, True, ren=True))
        self.assertEqual(rec.r_avail_est, 4)         # in band: hold
        self.assertFalse(rec.almost_empty_after)
        rec = fifo.run_one(Event(12, False, True, ren=True))
        self.assertEqual(rec.r_avail_est, 3)         # in band: hold
        self.assertFalse(rec.almost_empty_after)
        rec = fifo.run_one(Event(13, False, True, ren=True))
        self.assertEqual(rec.r_avail_est, 2)
        self.assertTrue(rec.almost_empty_after)      # est <= low: set


class TestAlmostWrap(unittest.TestCase):
    def test_estimates_use_modular_distance_across_wrap(self):
        # k=1 (depth 2, pointers mod 4): after wptr wraps 3 -> 0 the write
        # estimate must wrap too, not go negative.
        fifo = AsyncFIFO(k=1, almost_full_low=0, almost_full_high=2,
                         almost_empty_low=0, almost_empty_high=1)
        fifo.run([Event(1, True, False, wen=True, wdata="a"),
                  Event(2, True, False, wen=True, wdata="b")])
        self.assertTrue(fifo.full)
        # drain both through the read domain (3 sync edges, then 2 reads)
        fifo.run_one(Event(3, False, True))
        fifo.run_one(Event(4, False, True))
        fifo.run_one(Event(5, False, True))
        fifo.run_one(Event(6, False, True, ren=True))
        fifo.run_one(Event(7, False, True, ren=True))
        # let the write domain see the drained read pointer, then wrap wptr
        # (two-stage sync plus the old-flag edge: three write edges)
        fifo.run_one(Event(8, True, False))
        fifo.run_one(Event(9, True, False))
        fifo.run_one(Event(10, True, False))
        rec = fifo.run_one(Event(11, True, False, wen=True, wdata="c"))
        self.assertTrue(rec.w_accept)
        self.assertEqual((rec.wptr_before, rec.wptr_after), (2, 3))
        rec = fifo.run_one(Event(12, True, False, wen=True, wdata="d"))
        self.assertEqual((rec.wptr_before, rec.wptr_after), (3, 0))  # wrap
        self.assertEqual(rec.w_used_est,
                         (0 - 2) % 4)                # modular, not negative
        self.assertGreaterEqual(rec.w_used_est, 0)


class TestAlmostFlagsAreAdvisoryOnly(unittest.TestCase):
    def test_accept_decisions_ignore_almost_flags(self):
        # almost_full asserted long before full: writes must still be
        # accepted until the hard full flag fires.
        fifo = AsyncFIFO(k=2, almost_full_low=0, almost_full_high=1)
        records = fifo.run([Event(t, True, False, wen=True, wdata=t)
                            for t in range(1, 5)])
        self.assertTrue(records[0].almost_full_after)   # hint already on
        self.assertTrue(all(r.w_accept for r in records[:3]))
        self.assertEqual(fifo.accepted_writes, 4)       # hard full only at 4
        self.assertTrue(fifo.full)
        # almost_empty deasserted does not gate reads either
        fifo2 = AsyncFIFO(k=2, almost_empty_low=0, almost_empty_high=1)
        rec = fifo2.run_one(Event(1, False, True, ren=True))
        self.assertFalse(rec.r_accept)                  # empty still gates
        self.assertTrue(rec.almost_empty_after)

    def test_old_record_fields_and_callers_unaffected(self):
        fifo = AsyncFIFO(k=2)
        rec = fifo.run_one(Event(1, True, False, wen=True, wdata=1))
        # leading positional fields keep their order (old unpacking works)
        self.assertEqual(rec[:23], (1, True, False, True, False, True, None,
                                    0, 1, 0, 0, 0, 1, 0, 0, False, False,
                                    True, True, (0, 0), (0, 0), (0, 0),
                                    (0, 0)))
        self.assertEqual(rec[23], 1)                    # wdata
        self.assertIsNone(rec[24])                      # rdata
        self.assertEqual(rec.w_used_est, 1)             # new fields appended
        self.assertIsNone(rec.r_avail_est)
        self.assertFalse(rec.almost_full_after)
        self.assertTrue(rec.almost_empty_after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
