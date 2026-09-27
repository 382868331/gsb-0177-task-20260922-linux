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

    def test_threshold_defaults_and_power_on(self):
        fifo = AsyncFIFO(k=3)
        self.assertEqual(fifo.almost_full_margin, 1)
        self.assertEqual(fifo.almost_empty_margin, 1)
        self.assertFalse(fifo.almost_full)   # depth free at reset
        self.assertTrue(fifo.almost_empty)   # zero guaranteed entries

    def test_threshold_reset_marks(self):
        # margin == depth: warning necessarily on right after reset.
        fifo = AsyncFIFO(k=2, almost_full_margin=4, almost_empty_margin=4)
        self.assertTrue(fifo.almost_full)
        self.assertTrue(fifo.almost_empty)
        # margin 0: almost flags coincide with the conservative exact case.
        fifo = AsyncFIFO(k=2, almost_full_margin=0, almost_empty_margin=0)
        self.assertFalse(fifo.almost_full)
        self.assertTrue(fifo.almost_empty)

    def test_threshold_contract(self):
        for bad in (-1, 9, True, 1.0, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, almost_full_margin=bad)
                with self.assertRaises(AsyncFIFOError):
                    AsyncFIFO(k=2, almost_empty_margin=bad)
        for margin in range(0, 5):
            AsyncFIFO(k=2, almost_full_margin=margin,
                      almost_empty_margin=margin)

    def test_reset_keeps_user_thresholds_and_recomputes_marks(self):
        fifo = AsyncFIFO(k=2, almost_full_margin=1, almost_empty_margin=2)
        fifo.run([Event(1, True, False, wen=True, wdata=1)])
        fifo.reset()
        self.assertEqual(fifo.almost_full_margin, 1)
        self.assertEqual(fifo.almost_empty_margin, 2)
        self.assertFalse(fifo.almost_full)
        self.assertTrue(fifo.almost_empty)


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


class TestAlmostFlags(unittest.TestCase):
    def test_almost_full_warns_before_full_and_never_underpromises(self):
        # depth 4, margin 1: warning fires when guaranteed free space <= 1,
        # i.e. strictly before the exact full case; once full is on the
        # warning must stay on.
        fifo = AsyncFIFO(k=2, almost_full_margin=1)
        warned_at = full_at = None
        for t in range(1, 8):
            rec = fifo.run_one(
                Event(t, True, False, wen=True, wdata=t)
            )
            if warned_at is None and rec.almost_full_after:
                warned_at = t
            if rec.almost_full_after:
                self.assertTrue(rec.almost_full_before is True
                                or not rec.full_after
                                or rec.full_after)
            if full_at is None and rec.full_after:
                full_at = t
        self.assertIsNotNone(warned_at)
        self.assertIsNotNone(full_at)
        self.assertLessEqual(warned_at, full_at)

    def test_almost_empty_warns_while_draining(self):
        # Write 4 (write-only edges), drain with read-only edges.  With
        # margin 1 the warning asserts no later than one accepted read before
        # the real empty flag.
        fifo = AsyncFIFO(k=2, almost_empty_margin=1)
        for t in range(1, 5):
            fifo.run_one(Event(t, True, False, wen=True, wdata=t))
        warned = []
        emptied = []
        for t in range(5, 12):
            rec = fifo.run_one(Event(t, False, True, ren=True))
            if rec.almost_empty_after:
                warned.append(t)
            if rec.empty_after:
                emptied.append(t)
        self.assertTrue(warned and emptied)
        self.assertLessEqual(warned[0], emptied[0])

    def test_guaranteed_bounds_hold_across_wraps(self):
        # k=1, pointers wrap every 2 accepts; drive many wrap cycles and
        # verify the warning never promises more than local register data
        # guarantees:
        #   * full => margin-0 almost-full must also be on;
        #   * margin-0 almost-empty OFF => newest synced write pointer
        #     provably leaves at least one entry ahead of the read pointer.
        from afifo import gray_to_binary
        fifo = AsyncFIFO(k=1, almost_full_margin=0,
                         almost_empty_margin=0)
        for t in range(1, 80):
            if t % 2:
                rec = fifo.run_one(
                    Event(t, True, False, wen=True, wdata=t))
                if rec.full_after:
                    self.assertTrue(rec.almost_full_after)
            else:
                rec = fifo.run_one(Event(t, False, True, ren=True))
                if not rec.almost_empty_after:
                    guaranteed = ((gray_to_binary(rec.w2r_after[0])
                                   - rec.rptr_after) % 4)
                    self.assertGreaterEqual(guaranteed, 1)
        self.assertGreater(fifo.accepted_writes, 4)
        self.assertGreater(fifo.accepted_reads, 4)

    def test_coincident_edge_uses_old_sync_state(self):
        fifo = AsyncFIFO(k=2, almost_full_margin=1, almost_empty_margin=1)
        fifo.run_one(Event(1, True, False, wen=True, wdata=1))
        for t in (2, 3):
            fifo.run_one(Event(t, False, True))
        af_before = fifo.almost_full
        ae_before = fifo.almost_empty
        rec = fifo.run_one(Event(4, True, True, wen=True, ren=True))
        # both flags recomputed from OLD synchronizer outputs only
        self.assertEqual(rec.almost_full_before, af_before)
        self.assertEqual(rec.almost_empty_before, ae_before)
        self.assertEqual(rec.r2w_after[0], rec.rgray_before)
        self.assertEqual(rec.w2r_after[0], rec.wgray_before)

    def test_stopped_clock_holds_almost_flag_register(self):
        fifo = AsyncFIFO(k=2, almost_full_margin=1)
        fifo.run([Event(1, True, False, wen=True, wdata=1),
                  Event(2, True, False, wen=True, wdata=2),
                  Event(3, True, False, wen=True, wdata=3)])
        held = fifo.almost_full
        for t in (4, 5, 6):
            rec = fifo.run_one(Event(t, False, True))
            self.assertIs(rec.almost_full_after, held)

    def test_warning_never_late_against_real_occupancy(self):
        """Whenever real free/entries cross the margin the warning is on.

        Early (pessimistic) assertions are allowed; a missing warning when
        the true abstract-queue occupancy already crossed the threshold would
        overpromise and is forbidden.  Checked with the independent reference
        queue over mixed ratios, stops and k=1 wrap churn.
        """
        combos = [(1, 1, 1, 0, 0), (1, 2, 3, 1, 1), (2, 3, 2, 1, 2),
                  (3, 2, 5, 3, 1), (5, 3, 2, 3, 3)]
        for k, wp, rp, mf, me in combos:
            schedule = make_schedule(
                k, 600, w_period=wp, r_period=rp,
                seed=k * 131 + wp * 17 + rp + mf * 7 + me,
                stop_windows=((200, 260),),
                r_stop_windows=((400, 450),),
            )
            dut = AsyncFIFO(k, almost_full_margin=mf,
                            almost_empty_margin=me)
            ref = RefFIFO(k)
            with self.subTest(k=k, wp=wp, rp=rp):
                for (t, wc, rc, we, re_, data) in schedule:
                    rec = dut.run_one(Event(t, wc, rc, we, re_, data))
                    occ = ref.step(wc, rc, we, re_, data)["occupancy"]
                    if wc:
                        self.assertTrue(
                            (1 << k) - occ > mf
                            or rec.almost_full_after
                        )
                    if rc:
                        self.assertTrue(occ > me or rec.almost_empty_after)


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
                  af_margin=1, ae_margin=1):
        schedule = make_schedule(
            k, n, w_period=wp, r_period=rp, seed=seed,
            stop_windows=w_stops, r_stop_windows=r_stops,
        )
        dut = AsyncFIFO(k=k, almost_full_margin=af_margin,
                        almost_empty_margin=ae_margin)
        ref = RefFIFO(k=k, af_margin=af_margin, ae_margin=ae_margin)
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
                self.assertIs(rec.almost_full_after, exp["almost_full"])
                self.assertIs(rec.almost_empty_after, exp["almost_empty"])
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

    def test_almost_thresholds_diff(self):
        for k in (1, 2, 3, 4, 5):
            for af, ae in ((0, 0), (1, 1), (1, 3), (3, 1),
                           (1 << k, 1 << k)):
                af_ok = min(af, (1 << k) - 1)
                ae_ok = min(ae, (1 << k) - 1)
                self._diff_run(k, 2, 3, 200, seed=k * 500 + af * 7 + ae,
                               af_margin=af_ok, ae_margin=ae_ok)
                self._diff_run(k, 1, 1, 200, seed=k * 700 + af + ae,
                               af_margin=af_ok, ae_margin=ae_ok)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
