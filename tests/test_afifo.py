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

    def _diff_run(self, k, wp, rp, n, seed=1, w_stops=(), r_stops=(),
                  wrst_times=(), rrst_times=()):
        schedule = make_schedule(
            k, n, w_period=wp, r_period=rp, seed=seed,
            stop_windows=w_stops, r_stop_windows=r_stops,
            wrst_times=wrst_times, rrst_times=rrst_times,
        )
        dut = AsyncFIFO(k=k)
        ref = RefFIFO(k=k)
        events = [Event(t, wc, rc, we, re_, data, wrst=wrs, rrst=rrs)
                  for (t, wc, rc, we, re_, data, wrs, rrs) in schedule]
        records = dut.run(events)
        read_out = []
        written = []  # only ACCEPTED writes, observed from the reference
        for ev, rec in zip(schedule, records):
            t, wc, rc, we, re_, data, wrs, rrs = ev
            # explicit pulse or a peer-triggered reactive reset purges all
            # pre-reset data from both implementations
            if wrs or rrs:
                written.clear()
                read_out.clear()
            exp = ref.step(wc, rc, we, re_, data, wrst=wrs, rrst=rrs)
            if exp["w_reactive"] or exp["r_reactive"]:
                written.clear()
                read_out.clear()
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
                self.assertEqual(rec.wgen_after, exp["wgen"])
                self.assertEqual(rec.rgen_after, exp["rgen"])
                self.assertEqual(rec.rgen2w_after, exp["rgen2w"])
                self.assertEqual(rec.wgen2r_after, exp["wgen2r"])
                self.assertIs(rec.wready_after, exp["wready"])
                self.assertIs(rec.rready_after, exp["rready"])
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
            for (t, wc, rc, we, re_, data, wrs, rrs) in schedule:
                dut.run_one(Event(t, wc, rc, we, re_, data,
                                  wrst=wrs, rrst=rrs))
                exp = ref.step(wc, rc, we, re_, data, wrst=wrs, rrst=rrs)
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


# ---------------------------------------------------------------------------
# Independent-domain reset + generation handshake.
#
# Hand-derived waveform below (k=2, depth 4; g(v)=v xor (v>>1)).
# Fill, then write-domain reset, clocks alternating:
#   t=1..4 W wen A..D: four accepted writes; wptr 0->4; full sets at t=4
#                      (g(4)=6 == 0 ^ 0b110)
#   t=5 W wrst, wen "E": wptr 4->0, full->0, r2w/rgen2w cleared, wgen 0->1,
#       memory purged, counters zeroed; gate closed -> "E" rejected
#   t=6 R ren: gate closed -> rejected; wgen2r (0,0)->(g(1)=1, 0)
#   t=7 R: wgen2r ->(1,1); decoded peer gen 1 > rgen 0 -> reactive reset:
#       rgen 0->1, rptr->0, empty stays 1, w2r pointer chain cleared, purge
#   t=8 W: rgen2w (0,0)->(1,0); wready still False
#   t=9 W: rgen2w ->(1,1); wgen 1 == decode(1): both domains ready
#   t=10 W wen "E": gate open -> accepted, wptr 0->1
#   t=11..12 R: pointer Gray propagates; empty deasserts at the end of t=12
#   t=13 R: "E" is the one and only readable payload, then empty again
# ---------------------------------------------------------------------------
RESET_WAVE = [
    # t, wc, rc, we, re, wrst, rrst, data
    (1, 1, 0, 1, 0, 0, 0, "A"),
    (2, 1, 0, 1, 0, 0, 0, "B"),
    (3, 1, 0, 1, 0, 0, 0, "C"),
    (4, 1, 0, 1, 0, 0, 0, "D"),
    (5, 1, 0, 1, 0, 1, 0, "E"),
    (6, 0, 1, 0, 1, 0, 0, None),
    (7, 0, 1, 0, 0, 0, 0, None),
    (8, 1, 0, 0, 0, 0, 0, None),
    (9, 1, 0, 0, 0, 0, 0, None),
    (10, 1, 0, 1, 0, 0, 0, "E"),
]


class TestResetHandshakeHandWave(unittest.TestCase):
    def setUp(self):
        self.fifo = AsyncFIFO(k=2)
        self.records = self.fifo.run([
            Event(t, bool(wc), bool(rc), bool(we), bool(re_), d,
                  wrst=bool(ws), rrst=bool(rs))
            for (t, wc, rc, we, re_, ws, rs, d) in RESET_WAVE
        ])
        self.by_time = {r.time: r for r in self.records}

    def test_power_on_generations_ready(self):
        fifo = AsyncFIFO(k=2)
        self.assertEqual(fifo.wgen, 0)
        self.assertEqual(fifo.rgen, 0)
        self.assertTrue(fifo.wready and fifo.rready and fifo.synchronized)
        rec = self.by_time[1]
        self.assertTrue(rec.wready_before and rec.rready_before)

    def test_fill_reaches_full_before_reset(self):
        r4 = self.by_time[4]
        self.assertTrue(all(self.by_time[t].w_accept for t in (1, 2, 3, 4)))
        self.assertEqual(r4.wptr_after, 4)
        self.assertTrue(r4.full_after)

    def test_reset_step_clears_write_domain(self):
        r = self.by_time[5]
        self.assertTrue(r.wrst)
        # before values are the old running state
        self.assertEqual(r.wptr_before, 4)
        self.assertEqual(r.wgen_before, 0)
        self.assertTrue(r.full_before)
        # reset committed before sampling
        self.assertEqual(r.wptr_after, 0)
        self.assertEqual(r.wgen_after, 1)
        self.assertFalse(r.full_after)
        self.assertEqual(r.r2w_after, (0, 0))
        self.assertEqual(r.rgen2w_after, (0, 0))
        # enable at the reset step is rejected (gate closed before confirm)
        self.assertFalse(r.w_accept)
        self.assertIsNone(r.r_accept)
        # read domain frozen this step
        self.assertEqual(r.rgen_after, 0)
        self.assertEqual(r.wgen2r_after, (0, 0))
        self.assertFalse(r.wready_after)
        self.assertTrue(r.rready_after)
        self.assertFalse(r.wready_after and r.rready_after)

    def test_reset_purges_old_data_and_counters(self):
        # Run only through t=6 so the statistics counters reflect the reset,
        # not the later post-handshake write at t=10.
        fifo = AsyncFIFO(k=2)
        partial = []
        for (t, wc, rc, we, re_, ws, rs, d) in RESET_WAVE[:6]:
            partial.append(Event(t, bool(wc), bool(rc), bool(we), bool(re_),
                                 d, wrst=bool(ws), rrst=bool(rs)))
        fifo.run(partial)
        r6 = fifo.steps[-1]
        self.assertFalse(r6.r_accept)
        self.assertIsNone(r6.rdata)
        self.assertEqual(fifo.accepted_writes, 0)
        self.assertEqual(fifo.accepted_reads, 0)
        # first generation Gray stage is on its way at t=6, not yet adopted
        self.assertEqual(r6.wgen2r_after, (binary_to_gray(1), 0))
        self.assertEqual(r6.rgen_after, 0)
        # read domain is locally still ready (its own gen matches old s2);
        # the read is refused by the GLOBAL gate (write domain not confirmed)
        self.assertTrue(r6.rready_after)
        self.assertFalse(r6.wready_after)
        self.assertFalse(fifo.synchronized)

    def test_peer_adopts_generation_reactively(self):
        r7 = self.by_time[7]
        self.assertEqual(r7.rgen_before, 0)
        self.assertEqual(r7.rgen_after, 1)
        self.assertEqual(r7.rptr_after, 0)
        self.assertTrue(r7.empty_after)
        self.assertEqual(r7.w2r_after, (0, 0))       # pointer chain cleared
        self.assertTrue(r7.rready_after)
        self.assertFalse(r7.wready_after)

    def test_echo_opens_gate_two_edges_later(self):
        r8 = self.by_time[8]
        self.assertEqual(r8.rgen2w_after, (binary_to_gray(1), 0))
        self.assertFalse(r8.wready_after)
        r9 = self.by_time[9]
        self.assertEqual(r9.rgen2w_after,
                         (binary_to_gray(1), binary_to_gray(1)))
        self.assertTrue(r9.wready_after and r9.rready_after)
        self.assertTrue(self.fifo.synchronized)

    def test_transfers_resume_and_old_data_gone(self):
        r10 = self.by_time[10]
        self.assertTrue(r10.w_accept)
        self.assertEqual(r10.wptr_after, 1)
        self.assertEqual(self.fifo.accepted_writes, 1)
        # drain: the only readable payload is the post-reset "E"
        reads = []
        t = 10
        while t < 20:
            t += 1
            rec = self.fifo.run_one(Event(t, False, True, ren=True))
            if rec.r_accept:
                reads.append(rec.rdata)
        self.assertEqual(reads, ["E"])
        self.assertEqual(self.fifo.accepted_reads, 1)


class TestResetSecondPulse(unittest.TestCase):
    """A second reset before the handshake finishes: stale acks cannot pass."""

    def test_double_write_reset_before_peer_observes(self):
        fifo = AsyncFIFO(k=2)
        fifo.run_one(Event(1, True, False, wen=True, wdata="A"))
        fifo.run_one(Event(2, True, False, wrst=True))      # wgen 0->1
        r3 = fifo.run_one(Event(3, True, False, wrst=True,
                                wen=True, wdata="B"))        # wgen 1->2
        self.assertEqual(r3.wgen_after, 2)
        self.assertEqual(r3.rgen2w_after, (0, 0))
        self.assertFalse(r3.w_accept)
        # first two read edges: gen gray 2 (binary 10 -> gray 11) propagates
        r4 = fifo.run_one(Event(4, False, True, ren=True))
        self.assertEqual(r4.wgen2r_after, (0b11, 0))
        self.assertEqual(r4.rgen_after, 0)
        self.assertFalse(r4.r_accept)
        r5 = fifo.run_one(Event(5, False, True, ren=True))
        self.assertEqual(r5.rgen_after, 2)                  # adopts newest
        self.assertFalse(r5.r_accept)
        # echo back: two write edges; everything in between is rejected
        r6 = fifo.run_one(Event(6, True, False, wen=True, wdata="B"))
        self.assertFalse(r6.w_accept)
        r7 = fifo.run_one(Event(7, True, False, wen=True, wdata="B"))
        self.assertTrue(fifo.synchronized)
        self.assertFalse(r7.w_accept)  # gate opens at END of t=7
        r8 = fifo.run_one(Event(8, True, False, wen=True, wdata="B"))
        self.assertTrue(r8.w_accept)
        self.assertEqual(fifo.accepted_writes, 1)

    def test_reset_while_older_echo_in_flight(self):
        # gen1 fully acknowledged, then reset to gen2 while the gen1 echo is
        # still traveling: gen1 values must not be mistaken for gen2 ack.
        fifo = AsyncFIFO(k=1)
        fifo.run_one(Event(1, True, False, wrst=True))       # wgen 1
        fifo.run([Event(2, False, True), Event(3, False, True)])  # r adopts
        fifo.run([Event(4, True, False), Event(5, True, False)])  # echo done
        self.assertTrue(fifo.synchronized)
        fifo.run_one(Event(6, True, False, wen=True, wdata="z"))
        fifo.run_one(Event(7, True, False, wrst=True))       # wgen 2
        r8 = fifo.run_one(Event(8, False, True))
        self.assertEqual(r8.rgen_after, 1)   # old gen-1 echo, not yet adopted
        self.assertFalse(fifo.synchronized)
        r9 = fifo.run_one(Event(9, False, True))
        self.assertEqual(r9.rgen_after, 2)                  # adopts newest
        fifo.run_one(Event(10, True, False))
        self.assertFalse(fifo.synchronized)                  # echo of 2 pending
        fifo.run_one(Event(11, True, False))
        self.assertTrue(fifo.synchronized)
        # old payload "z" is unreachable
        rec = fifo.run_one(Event(12, False, True, ren=True))
        self.assertFalse(rec.r_accept)


class TestReadDomainReset(unittest.TestCase):
    def test_read_reset_symmetric_handwave(self):
        fifo = AsyncFIFO(k=2)
        fifo.run_one(Event(1, True, False, wen=True, wdata="A"))
        # empty needs three read edges to drop: s2 reaches g1 at edge 2, the
        # flag recomputes from old s2 at edge 3, the request accepts at edge 4
        for t in (2, 3, 4):
            fifo.run_one(Event(t, False, True))
        r5 = fifo.run_one(Event(5, False, True, ren=True))
        self.assertTrue(r5.r_accept and r5.rdata == "A")
        fifo.run_one(Event(6, True, False, wen=True, wdata="B"))
        # read-domain reset
        r7 = fifo.run_one(Event(7, False, True, rrst=True, ren=True))
        self.assertTrue(r7.rrst)
        self.assertEqual(r7.rptr_after, 0)
        self.assertEqual(r7.rgen_after, 1)
        self.assertTrue(r7.empty_after)
        self.assertEqual(r7.w2r_after, (0, 0))
        self.assertEqual(r7.wgen2r_after, (0, 0))
        self.assertFalse(r7.r_accept)
        # write side first sees rgen gray stage 1, then adopts
        r8 = fifo.run_one(Event(8, True, False, wen=True, wdata="C"))
        self.assertEqual(r8.rgen2w_after, (binary_to_gray(1), 0))
        self.assertFalse(r8.w_accept)
        r9 = fifo.run_one(Event(9, True, False, wen=True, wdata="C"))
        self.assertEqual(r9.wgen_after, 1)
        self.assertEqual(r9.wptr_after, 0)
        self.assertFalse(r9.full_after)
        self.assertTrue(r9.wready_after)
        self.assertFalse(r9.rready_after)
        self.assertFalse(r9.w_accept)
        # read domain clocks the echo of wgen=1 through
        r10 = fifo.run_one(Event(10, False, True))
        self.assertEqual(r10.wgen2r_after, (binary_to_gray(1), 0))
        self.assertFalse(fifo.synchronized)
        r11 = fifo.run_one(Event(11, False, True))
        self.assertEqual(r11.wgen2r_after,
                         (binary_to_gray(1), binary_to_gray(1)))
        self.assertTrue(fifo.synchronized)
        r12 = fifo.run_one(Event(12, True, False, wen=True, wdata="C"))
        self.assertTrue(r12.w_accept)
        self.assertEqual(fifo.accepted_writes, 1)


class TestCoincidentResetBeforeSampling(unittest.TestCase):
    def test_coincident_reset_is_seen_by_peer_same_step(self):
        fifo = AsyncFIFO(k=2)
        fifo.run_one(Event(1, True, False, wen=True, wdata="A"))
        # write reset + read edge at the SAME time: reset first, then sample
        r2 = fifo.run_one(Event(2, True, True, ren=True,
                                wen=True, wdata="B", wrst=True))
        self.assertEqual(r2.wptr_after, 0)
        self.assertEqual(r2.wgen_after, 1)
        # read side latches the POST-reset pointer Gray (0), not stale g(1)
        self.assertEqual(r2.w2r_after, (0, 0))
        # generation stage 1 already carries the new generation
        self.assertEqual(r2.wgen2r_after, (binary_to_gray(1), 0))
        self.assertFalse(r2.w_accept)
        self.assertFalse(r2.r_accept)
        self.assertIsNone(r2.rdata)
        self.assertEqual(r2.rgen_after, 0)       # adopts one edge later
        r3 = fifo.run_one(Event(3, False, True))
        self.assertEqual(r3.rgen_after, 1)

    def test_both_domains_reset_at_same_time(self):
        fifo = AsyncFIFO(k=2)
        fifo.run([Event(1, True, False, wen=True, wdata="A"),
                  Event(2, True, False, wen=True, wdata="B")])
        r3 = fifo.run_one(Event(3, True, True, wrst=True, rrst=True))
        self.assertEqual((r3.wgen_after, r3.rgen_after), (1, 1))
        self.assertEqual(r3.wptr_after, 0)
        self.assertEqual(r3.rptr_after, 0)
        self.assertFalse(r3.full_after)
        self.assertTrue(r3.empty_after)
        # both chains cleared on both sides
        self.assertEqual(r3.r2w_after, (0, 0))
        self.assertEqual(r3.w2r_after, (0, 0))
        self.assertEqual(r3.rgen2w_after, (0, 0))
        self.assertEqual(r3.wgen2r_after, (0, 0))
        self.assertFalse(fifo.synchronized)
        # with coincident clocks each step, each side sees the peer's new
        # generation after the standard two synchronizer stages
        r4 = fifo.run_one(Event(4, True, True, wen=True, ren=True))
        self.assertFalse(r4.w_accept and r4.r_accept)
        self.assertFalse(fifo.synchronized)
        r5 = fifo.run_one(Event(5, True, True, wen=True, ren=True,
                                wdata="C"))
        self.assertTrue(fifo.synchronized)
        self.assertFalse(r5.w_accept)   # gate opens at the END of this edge
        r6 = fifo.run_one(Event(6, True, True, wen=True, ren=True,
                                wdata="C"))
        self.assertTrue(r6.w_accept)
        self.assertFalse(r6.r_accept)   # empty still asserted (ptr latency)
        self.assertEqual(fifo.accepted_writes, 1)


class TestResetStateSemantics(unittest.TestCase):
    def test_instream_reset_keeps_trace_but_coarse_reset_clears(self):
        fifo = AsyncFIFO(k=2)
        fifo.run([Event(1, True, False, wen=True, wdata="A"),
                  Event(2, True, False, wrst=True)])
        self.assertEqual(len(fifo.steps), 2)
        fifo.reset()
        self.assertEqual(fifo.steps, ())
        self.assertEqual((fifo.wgen, fifo.rgen), (0, 0))
        self.assertTrue(fifo.synchronized)
        self.assertEqual(fifo.accepted_writes, 0)
        self.assertEqual(fifo.snapshot()["wgen2r"], (0, 0))

    def test_reset_then_normal_operation_matches_power_on(self):
        # after a completed handshake the register state equals a fresh FIFO
        # (apart from generation counters, which sit one bump above zero)
        fresh = AsyncFIFO(k=2)
        fifo = AsyncFIFO(k=2)
        fifo.run([Event(1, True, False, wen=True, wdata="A"),
                  Event(2, True, False, wen=True, wdata="B"),
                  Event(3, True, False, wrst=True)])
        fifo.run([Event(t, False, True) for t in (4, 5)])
        fifo.run([Event(t, True, False) for t in (6, 7)])
        self.assertTrue(fifo.synchronized)
        self.assertEqual(fifo.wptr, fresh.wptr)
        self.assertEqual(fifo.rptr, fresh.rptr)
        self.assertIs(fifo.full, fresh.full)
        self.assertIs(fifo.empty, fresh.empty)
        self.assertEqual(fifo.snapshot()["r2w"], (0, 0))
        self.assertEqual(fifo.snapshot()["w2r"], (0, 0))
        self.assertEqual(fifo.accepted_writes, 0)
        self.assertEqual(fifo.accepted_reads, 0)

    def test_reset_batch_failure_leaves_no_partial_change(self):
        fifo = AsyncFIFO(k=2)
        fifo.run([Event(1, True, False, wen=True, wdata="A")])
        before = fifo.snapshot()
        bad = [
            Event(2, True, False, wrst=True),
            Event(3, True, False, wen=True, wdata="B"),
            "not-an-event",
        ]
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[2\]"):
            fifo.run(bad)
        self.assertEqual(fifo.snapshot(), before)
        self.assertEqual(fifo.accepted_writes, 1)


class TestNoTransferBeforeConfirmed(unittest.TestCase):
    def test_peer_unaware_window_rejects_even_with_data_present(self):
        # Fill 3, let the read side see them so empty is False, then pulse a
        # WRITE reset while the read domain has no edge: the very next read
        # edge is the peer-unaware window (generation only at sync stage 1);
        # the edge after that performs the reactive reset.
        fifo = AsyncFIFO(k=2)
        for t, d in ((1, "a"), (2, "b"), (3, "c")):
            fifo.run_one(Event(t, True, False, wen=True, wdata=d))
        for t in (4, 5, 6):
            fifo.run_one(Event(t, False, True))
        self.assertFalse(fifo.empty)
        fifo.run_one(Event(7, True, False, wrst=True))     # write resets
        # read edge 1: generation reaches stage 1 only; domain unaware, yet
        # the global gate forces the read down and no old datum can appear
        r8 = fifo.run_one(Event(8, False, True, ren=True))
        self.assertIs(r8.r_accept, False)
        self.assertIsNone(r8.rdata)
        self.assertEqual(r8.rgen_after, 0)
        self.assertEqual(r8.wgen2r_after, (binary_to_gray(1), 0))
        # read edge 2: reactive reset adopts the generation; still no read
        r9 = fifo.run_one(Event(9, False, True, ren=True))
        self.assertEqual(r9.rgen_after, 1)
        self.assertFalse(r9.r_accept)
        self.assertIsNone(r9.rdata)
        self.assertTrue(r9.empty_after)
        self.assertEqual(fifo.accepted_reads, 0)
        # echo back over two write edges; first request still rejected,
        # the next one is the first legal post-reset transfer
        r10 = fifo.run_one(Event(10, True, False, wen=True, wdata="n1"))
        self.assertFalse(r10.w_accept)
        r11 = fifo.run_one(Event(11, True, False, wen=True, wdata="n1"))
        self.assertTrue(fifo.synchronized)
        self.assertFalse(r11.w_accept)   # gate opens at the END of this edge
        r12 = fifo.run_one(Event(12, True, False, wen=True, wdata="n1"))
        self.assertTrue(r12.w_accept)

    def test_write_unaware_window_after_read_reset(self):
        fifo = AsyncFIFO(k=2)
        for t, d in ((1, "a"), (2, "b")):
            fifo.run_one(Event(t, True, False, wen=True, wdata=d))
        for t in (3, 4, 5):
            fifo.run_one(Event(t, False, True))
        fifo.run_one(Event(6, False, True, ren=True))     # drains "a"
        fifo.run_one(Event(7, False, True, rrst=True))    # read resets
        # write edge 1: write domain still unaware, full not set, yet refused
        r8 = fifo.run_one(Event(8, True, False, wen=True, wdata="x"))
        self.assertFalse(r8.w_accept)
        self.assertEqual(r8.wgen_after, 0)
        # write edge 2: reactive adoption of the read generation
        r9 = fifo.run_one(Event(9, True, False, wen=True, wdata="x"))
        self.assertEqual(r9.wgen_after, 1)
        self.assertEqual(r9.wptr_after, 0)
        self.assertFalse(r9.w_accept)
        self.assertEqual(fifo.accepted_writes, 0)
        # echo over two read edges reopens the gate
        r10 = fifo.run_one(Event(10, False, True))
        self.assertFalse(fifo.synchronized)
        r11 = fifo.run_one(Event(11, False, True))
        self.assertTrue(fifo.synchronized)
        r12 = fifo.run_one(Event(12, True, False, wen=True, wdata="x"))
        self.assertTrue(r12.w_accept)


class TestResetContract(unittest.TestCase):
    def test_reset_requires_its_own_clock(self):
        with self.assertRaisesRegex(AsyncFIFOError, "wrst requires"):
            Event(1, False, True, wrst=True)
        with self.assertRaisesRegex(AsyncFIFOError, "rrst requires"):
            Event(1, True, False, rrst=True)

    def test_reset_must_be_bool(self):
        with self.assertRaises(AsyncFIFOError):
            Event(1, True, False, wrst=1)
        with self.assertRaises(AsyncFIFOError):
            Event(1, True, False, rrst=0)

    def test_legacy_positional_construction_still_valid(self):
        # old call sites used at most 6 positional args; they keep working
        ev = Event(1, True, False, True, False, "payload")
        self.assertFalse(ev.wrst or ev.rrst)
        self.assertEqual(ev.wdata, "payload")


class TestResetDifferential(unittest.TestCase):
    """Every register vs the independent model over randomized reset runs."""

    def _diff(self, k, wp, rp, n, seed, wrst_times=(), rrst_times=(),
              w_stops=(), r_stops=()):
        TestDifferentialAgainstIndependentModel._diff_run(
            self, k, wp, rp, n, seed=seed,
            wrst_times=wrst_times, rrst_times=rrst_times,
            w_stops=w_stops, r_stops=r_stops,
        )

    def test_single_write_reset_all_ratios(self):
        for k in (1, 2, 3, 5):
            self._diff(k, 2, 3, 300, seed=1000 + k,
                       wrst_times=(60, 200))

    def test_single_read_reset_all_ratios(self):
        for k in (1, 2, 4, 5):
            self._diff(k, 3, 2, 300, seed=2000 + k,
                       rrst_times=(50, 220))

    def test_repeated_alternating_resets(self):
        self._diff(2, 2, 3, 420, seed=3001,
                   wrst_times=(30, 90, 150, 300),
                   rrst_times=(60, 120, 260))
        # rapid double pulses before the peer can observe
        self._diff(3, 5, 4, 400, seed=3002,
                   wrst_times=(40, 45, 50), rrst_times=(200, 205))

    def test_resets_around_pointer_wrap_k1(self):
        # k=1 wraps constantly; resets must reconcile with wrap state
        self._diff(1, 1, 1, 500, seed=3003,
                   wrst_times=(7, 130, 260), rrst_times=(70, 330))

    def test_reset_while_opposite_clock_stopped(self):
        self._diff(3, 2, 3, 400, seed=3004,
                   wrst_times=(50, 130),
                   r_stops=((45, 120),))
        self._diff(3, 3, 2, 400, seed=3005,
                   rrst_times=(60, 200),
                   w_stops=((55, 180),))

    def test_coincident_resets(self):
        self._diff(2, 2, 2, 300, seed=3006,
                   wrst_times=(40, 160), rrst_times=(40, 100, 160))


class TestResetExhaustiveSmallEnumeration(unittest.TestCase):
    """Exhaustive small-state enumeration: every event word of length 6
    over a 6-symbol alphabet from power-on, k=1.  Symbols:

      W edge / R edge / WR edge (enables on) and the same three with a
      reset pulse on every clocking domain.  Expectations come solely from
      the independent RefFIFO (never from the DUT).
    """

    ALPHABET = (
        # (wclk, rclk, wrst, rrst)
        (True, False, False, False),
        (False, True, False, False),
        (True, True, False, False),
        (True, False, True, False),
        (False, True, False, True),
        (True, True, True, True),
    )

    def test_all_words_length_six(self):
        import itertools
        k = 1
        for word in itertools.product(range(6), repeat=6):
            dut = AsyncFIFO(k=k)
            ref = RefFIFO(k=k)
            for i, sym in enumerate(word, start=1):
                wc, rc, wrs, rrs = self.ALPHABET[sym]
                data = f"d{i}" if wc else None
                rec = dut.run_one(Event(i, wc, rc, wen=wc, ren=rc,
                                        wdata=data, wrst=wrs, rrst=rrs))
                exp = ref.step(wc, rc, wc, rc, data,
                               wrst=wrs, rrst=rrs)
                self.assertEqual(rec.wptr_after, exp["wptr"])
                self.assertEqual(rec.rptr_after, exp["rptr"])
                self.assertEqual(rec.w_accept, exp["w_accept"])
                self.assertEqual(rec.r_accept, exp["r_accept"])
                self.assertIs(rec.full_after, exp["full"])
                self.assertIs(rec.empty_after, exp["empty"])
                self.assertEqual(rec.r2w_after, exp["r2w"])
                self.assertEqual(rec.w2r_after, exp["w2r"])
                self.assertEqual(rec.wgen_after, exp["wgen"])
                self.assertEqual(rec.rgen_after, exp["rgen"])
                self.assertEqual(rec.rgen2w_after, exp["rgen2w"])
                self.assertEqual(rec.wgen2r_after, exp["wgen2r"])
                self.assertIs(rec.wready_after, exp["wready"])
                self.assertIs(rec.rready_after, exp["rready"])
                self.assertEqual(rec.rdata, exp["rdata"])
                self.assertTrue(0 <= exp["occupancy"] <= (1 << k))
            self.assertEqual(dut.accepted_writes, ref.n_w)
            self.assertEqual(dut.accepted_reads, ref.n_r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
