"""Necessary tests for the ready/valid output-register wrapper.

Expected values come from:
1. A hand-derived k=1 (depth 2) waveform segment, checked cycle by cycle.
2. ``reference_model.RefReadyValid`` -- an independent reimplementation that
   never imports ``afifo`` -- maintaining two reference sequences: accepted
   writes and downstream deliveries.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_model import RefReadyValid, make_rv_schedule  # noqa: E402

from afifo import ReadyValidFIFO, RVEvent  # noqa: E402
from afifo.model import AsyncFIFOError  # noqa: E402


# Hand-derived k=1 (depth 2, 2-bit pointers) wrapper waveform.  Both clocks
# edge every step.  Each row:
# (time, wen, ready, wdata, w_accept, ren_issued, r_accept, delivered,
#  out_valid_after, out_data_after, wptr_after, rptr_after, full_after,
#  empty_after)
#
# t=2..4: two writes are already accepted (real queue length 2) but the write
#         Gray has not crossed into the read domain yet -- the wrapper must
#         NOT prefetch from the true queue length: ren is issued (register
#         empty) yet r_accept stays False and out_valid stays 0.
# t=5:    empty fell at t=4; the dequeue loads "A" into the output register.
# t=6..7: ready=0 backpressure -- ren=0, read pointer frozen, "A" bit-stable.
# t=8:    ready=1 -- old word "A" delivered and "B" loaded on the same edge.
# t=9:    ready=1 but the core is empty -- no load, out_valid falls.
HAND_RV_WAVE = [
    (1, 1, 0, "A", True,  True,  False, False, False, None, 1, 0, False, True),
    (2, 1, 0, "B", True,  True,  False, False, False, None, 2, 0, True,  True),
    (3, 0, 0, None, False, True,  False, False, False, None, 2, 0, True,  True),
    (4, 0, 0, None, False, True,  False, False, False, None, 2, 0, True,  False),
    (5, 0, 0, None, False, True,  True,  False, True,  "A",  2, 1, True,  False),
    (6, 0, 0, None, False, False, False, False, True,  "A",  2, 1, True,  False),
    (7, 0, 0, None, False, False, False, False, True,  "A",  2, 1, True,  False),
    (8, 0, 1, None, False, True,  True,  True,  True,  "B",  2, 2, False, True),
    (9, 0, 1, None, False, True,  False, True,  False, "B",  2, 2, False, True),
]


class TestHandWaveformRV(unittest.TestCase):
    def test_exact_register_trace(self):
        fifo = ReadyValidFIFO(k=1)
        events = [RVEvent(t, True, True, bool(we), bool(rd), data)
                  for (t, we, rd, data, *_rest) in HAND_RV_WAVE]
        records = fifo.run(events)

        for rec, row in zip(records, HAND_RV_WAVE):
            (t, we, rd, data, wacc, ren, racc, deliv, ov, od,
             wp, rp, full, empty) = row
            with self.subTest(time=t):
                self.assertIs(rec.w_accept, wacc)
                self.assertIs(rec.ren_issued, ren)
                self.assertIs(rec.r_accept, racc)
                self.assertIs(rec.delivered, deliv)
                self.assertIs(rec.out_valid_after, ov)
                self.assertEqual(rec.out_data_after, od)
                self.assertEqual(rec.inner.wptr_after, wp)
                self.assertEqual(rec.inner.rptr_after, rp)
                self.assertIs(rec.inner.full_after, full)
                self.assertIs(rec.inner.empty_after, empty)

        # delivery count: only t=8 and t=9 (old valid & ready)
        self.assertEqual(fifo.delivered, 2)
        self.assertEqual(fifo.accepted_writes, 2)
        self.assertEqual(fifo.accepted_reads, 2)   # core dequeues (loads)

    def test_no_prefetch_from_true_queue_length(self):
        """Writes accepted but not yet cross-domain visible: no load."""
        fifo = ReadyValidFIFO(k=1)
        fifo.run([
            RVEvent(1, True, True, wen=True, wdata="A"),
            RVEvent(2, True, True, wen=True, wdata="B"),
            RVEvent(3, True, True),
            RVEvent(4, True, True),
        ])
        # real queue held 2 words from t=2 on, yet nothing was loaded early
        for rec in fifo.steps:
            self.assertFalse(rec.out_valid_after)
            self.assertIs(rec.r_accept, False)
        self.assertEqual(fifo.accepted_writes, 2)   # writes did land
        rec5 = fifo.run_one(RVEvent(5, True, True))
        self.assertTrue(rec5.r_accept)              # only now visible
        self.assertEqual(rec5.out_data_after, "A")

    def test_backpressure_holds_data_and_pointer(self):
        """valid && !ready: out_data stable, read pointer frozen."""
        fifo = ReadyValidFIFO(k=1)
        fifo.run([
            RVEvent(1, True, True, wen=True, wdata="A"),
            RVEvent(2, True, True, wen=True, wdata="B"),
            RVEvent(3, True, True),
            RVEvent(4, True, True),
            RVEvent(5, True, True),               # loads "A"
        ])
        self.assertTrue(fifo.out_valid)
        for t in (6, 7, 8, 9):
            rec = fifo.run_one(RVEvent(t, True, True, ready=False))
            self.assertFalse(rec.ren_issued)      # no dequeue requested
            self.assertIs(rec.r_accept, False)
            self.assertEqual(rec.out_data_after, "A")
            self.assertEqual(rec.inner.rptr_after, 1)   # frozen
            self.assertEqual(rec.inner.rptr_before, 1)
            self.assertEqual(fifo.delivered, 0)


class TestSameCycleConsumeAndLoad(unittest.TestCase):
    def test_consume_old_and_load_new_same_edge(self):
        fifo = ReadyValidFIFO(k=1)
        fifo.run([
            RVEvent(1, True, True, wen=True, wdata="A"),
            RVEvent(2, True, True, wen=True, wdata="B"),
            RVEvent(3, True, True),
            RVEvent(4, True, True),
            RVEvent(5, True, True),               # out = "A", valid
        ])
        rec = fifo.run_one(RVEvent(6, True, True, ready=True))
        self.assertTrue(rec.delivered)            # old "A" consumed
        self.assertEqual(rec.out_data_before, "A")
        self.assertTrue(rec.r_accept)             # "B" dequeued same edge
        self.assertEqual(rec.out_data_after, "B")
        self.assertTrue(rec.out_valid_after)      # valid never dropped
        self.assertEqual(fifo.delivered, 1)


class TestResetWhileOutputRegisterFull(unittest.TestCase):
    def test_reset_clears_valid_mid_stream(self):
        fifo = ReadyValidFIFO(k=1)
        fifo.run([
            RVEvent(1, True, True, wen=True, wdata="A"),
            RVEvent(2, True, True, wen=True, wdata="B"),
            RVEvent(3, True, True),
            RVEvent(4, True, True),
            RVEvent(5, True, True),               # out_valid=1, holds "A"
        ])
        self.assertTrue(fifo.out_valid)
        fifo.reset()
        self.assertFalse(fifo.out_valid)          # reset clears valid
        self.assertIsNone(fifo.out_data)
        self.assertEqual(fifo.delivered, 0)
        self.assertEqual(fifo.accepted_writes, 0)
        self.assertTrue(fifo.empty)
        self.assertFalse(fifo.full)
        self.assertEqual(fifo.steps, ())
        # and the FIFO works again from power-on state
        rec = fifo.run_one(RVEvent(1, True, True, wen=True, wdata="Z"))
        self.assertTrue(rec.w_accept)
        self.assertFalse(rec.out_valid_after)


class TestCapacityAccounting(unittest.TestCase):
    def test_prefetch_adds_one_beyond_core_depth(self):
        """Output register is NOT hidden in depth: in-flight can be depth+1."""
        fifo = ReadyValidFIFO(k=1)                # core depth 2
        self.assertEqual(fifo.depth, 2)
        # ready stuck at 0; poke a write on every step (both clocks edge)
        for t in range(1, 10):
            fifo.run_one(RVEvent(t, True, True, wen=True,
                                 ready=False, wdata=f"W{t}"))
        # t=1,2 fill the core; t=5 prefetches one word into the output
        # register; once full deasserts (t=9) a third write lands.
        self.assertEqual(fifo.accepted_writes, 3)
        self.assertEqual(fifo.delivered, 0)
        self.assertEqual(fifo.in_flight, 3)       # depth(2) + register(1)
        self.assertTrue(fifo.out_valid)
        self.assertEqual(fifo.accepted_writes - fifo.accepted_reads, 2)
        # the core still reports depth 2 -- the register was never folded in
        self.assertEqual(fifo.depth, 2)


class TestDifferentialRV(unittest.TestCase):
    """Cycle-by-cycle diff against the independent RefReadyValid."""

    def _diff_run(self, k, wp, rp, n, seed=1, w_stops=(), r_stops=(), bp=()):
        schedule = make_rv_schedule(
            k, n, w_period=wp, r_period=rp, seed=seed,
            stop_windows=w_stops, r_stop_windows=r_stops,
            backpressure_windows=bp,
        )
        dut = ReadyValidFIFO(k=k)
        ref = RefReadyValid(k)
        events = [RVEvent(t, wc, rc, we, rd, data)
                  for (t, wc, rc, we, rd, data) in schedule]
        records = dut.run(events)
        for (t, wc, rc, we, rd, data), rec in zip(schedule, records):
            exp = ref.step(wc, rc, we, rd, data)
            inner = exp["inner"]
            with self.subTest(k=k, wp=wp, rp=rp, t=t):
                self.assertIs(rec.ren_issued, exp["ren"])
                self.assertIs(rec.delivered, exp["consume"])
                self.assertIs(rec.out_valid_after, exp["out_valid"])
                self.assertEqual(rec.out_data_after, exp["out_data"])
                self.assertEqual(rec.w_accept, inner["w_accept"])
                self.assertEqual(rec.r_accept, inner["r_accept"])
                self.assertEqual(rec.inner.wptr_after, inner["wptr"])
                self.assertEqual(rec.inner.rptr_after, inner["rptr"])
                self.assertIs(rec.inner.full_after, inner["full"])
                self.assertIs(rec.inner.empty_after, inner["empty"])
                self.assertEqual(rec.inner.r2w_after, inner["r2w"])
                self.assertEqual(rec.inner.w2r_after, inner["w2r"])
                # data stability under backpressure
                if (rec.rclk and rec.out_valid_before and not rec.ready):
                    self.assertFalse(rec.ren_issued)
                    self.assertEqual(rec.out_data_after, rec.out_data_before)
                    self.assertEqual(rec.inner.rptr_after,
                                     rec.inner.rptr_before)
                # occupancy of the abstract queue stays in bounds
                self.assertTrue(0 <= inner["occupancy"] <= (1 << k))
        # independent reference sequences: deliveries are a prefix of writes
        self.assertEqual(dut.delivered, ref.n_delivered)
        self.assertEqual(dut.accepted_writes, ref.inner.n_w)
        self.assertEqual(ref.delivered_seq, ref.written[: ref.n_delivered])
        self.assertEqual(dut.in_flight,
                         len(ref.inner.queue) + (1 if ref.out_valid else 0))
        self.assertLessEqual(dut.in_flight, (1 << k) + 1)
        self.assertEqual(dut.in_flight, dut.accepted_writes - dut.delivered)

    def test_unequal_clock_ratios(self):
        for k in (1, 2, 3, 5):
            for wp, rp in [(1, 1), (2, 3), (3, 2), (2, 5), (5, 2)]:
                self._diff_run(k, wp, rp, n=300, seed=100 * k + wp * 7 + rp)

    def test_continuous_backpressure(self):
        # ready held low for long stretches while traffic continues
        self._diff_run(1, 1, 1, 300, seed=5, bp=((20, 120), (200, 260)))
        self._diff_run(2, 2, 3, 300, seed=6, bp=((10, 250),))
        self._diff_run(3, 3, 2, 300, seed=7, bp=((50, 90), (150, 290)))

    def test_stopped_clocks_with_backpressure(self):
        self._diff_run(2, 2, 3, 240, seed=9, w_stops=((40, 80),),
                       r_stops=((140, 190),), bp=((60, 160),))

    def test_pointer_wrap(self):
        # k=1 wraps every 2 accepted accesses; long run wraps many times
        self._diff_run(1, 1, 1, 400, seed=42)
        self._diff_run(1, 2, 3, 600, seed=43, bp=((100, 200),))
        self._diff_run(5, 2, 3, 3000, seed=77)


class TestRVInputContract(unittest.TestCase):
    def test_event_contract(self):
        with self.assertRaises(AsyncFIFOError):
            RVEvent(True, True, False)                # bool time
        with self.assertRaises(AsyncFIFOError):
            RVEvent(-1, True, False)
        with self.assertRaisesRegex(AsyncFIFOError, "neither"):
            RVEvent(1, False, False)
        with self.assertRaises(AsyncFIFOError):
            RVEvent(1, True, True, ready=1)           # ready must be bool
        with self.assertRaisesRegex(AsyncFIFOError, "wdata"):
            RVEvent(1, True, False, wen=False, wdata=9)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(AsyncFIFOError, "finite"):
                RVEvent(1, True, False, wen=True, wdata=bad)
        RVEvent(0, True, False)                       # boundary allowed

    def test_time_ordering_and_types(self):
        fifo = ReadyValidFIFO(k=2)
        fifo.run_one(RVEvent(5, True, False))
        with self.assertRaisesRegex(AsyncFIFOError, "strictly greater"):
            fifo.run_one(RVEvent(5, True, False))
        with self.assertRaises(AsyncFIFOError):
            fifo.run_one("nope")
        with self.assertRaises(AsyncFIFOError):
            fifo.run(RVEvent(6, True, False))         # single event, not iter
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[1\]"):
            fifo.run([RVEvent(6, True, False), 7])

    def test_failed_batch_leaves_no_partial_change(self):
        fifo = ReadyValidFIFO(k=2)
        fifo.run([RVEvent(1, True, False, wen=True, wdata="q")])
        before = fifo.snapshot()
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[2\]"):
            fifo.run([
                RVEvent(2, True, False, wen=True, wdata="a"),
                RVEvent(3, True, False, wen=True, wdata="b"),
                RVEvent(3, True, False),              # duplicate time
            ])
        self.assertEqual(fifo.snapshot(), before)
        self.assertEqual(fifo.accepted_writes, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
