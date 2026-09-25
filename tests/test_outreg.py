"""Necessary tests for the ready/valid output-register wrapper.

Expected values come from:
1. A hand-derived k=1 (depth 2) waveform, checked cycle by cycle.
2. ``reference_model.RefOutReg`` -- an independent reimplementation that
   never imports ``afifo`` -- plus two abstract sequences maintained
   separately: accepted writes vs. downstream deliveries.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_model import RefOutReg, make_rv_schedule  # noqa: E402

from afifo import ReadyValidFIFO, RVEvent  # noqa: E402
from afifo.model import AsyncFIFOError  # noqa: E402


# Hand-derived k=1 (depth 2, 2-bit pointers) wrapper waveform.
# Columns per row:
#   (time, wclk, rclk, wen, ready, wdata,
#    w_accept, ren, r_accept, load, delivered, delivered_data,
#    valid_after, out_after, wptr, rptr, full, empty, r2w, w2r)
# None in w_accept/ren/r_accept/load/delivered means "domain not clocked".
HAND_RV_WAVE = [
    # fill the inner FIFO to full
    (1,  1, 0, 1, 0, "a", True,  None, None, None, None, None,
     False, None, 1, 0, False, True,  (0, 0), (0, 0)),
    (2,  1, 0, 1, 0, "b", True,  None, None, None, None, None,
     False, None, 2, 0, True,  True,  (0, 0), (0, 0)),
    (3,  1, 0, 1, 0, "c", False, None, None, None, None, None,  # full: reject
     False, None, 2, 0, True,  True,  (0, 0), (0, 0)),
    # write crosses the two-stage synchronizer: no prefetch yet, even though
    # two words really sit in the memory (queue length must not be used)
    (4,  0, 1, 0, 0, None, None, True,  False, False, False, None,
     False, None, 2, 0, True,  True,  (0, 0), (3, 0)),
    (5,  0, 1, 0, 0, None, None, True,  False, False, False, None,
     False, None, 2, 0, True,  True,  (0, 0), (3, 3)),
    (6,  0, 1, 0, 0, None, None, True,  False, False, False, None,
     False, None, 2, 0, True,  False, (0, 0), (3, 3)),
    # old empty finally low: prefetch "a" into the output register
    (7,  0, 1, 0, 0, None, None, True,  True,  True,  False, None,
     True,  "a", 2, 1, True,  False, (0, 0), (3, 3)),
    # continuous backpressure: valid & ~ready -> no ren, data stable
    (8,  0, 1, 0, 0, None, None, False, False, False, False, None,
     True,  "a", 2, 1, True,  False, (0, 0), (3, 3)),
    (9,  0, 1, 0, 0, None, None, False, False, False, False, None,
     True,  "a", 2, 1, True,  False, (0, 0), (3, 3)),
    # ready=1: consume "a" and load "b" in the SAME read edge; valid stays
    (10, 0, 1, 0, 1, None, None, True,  True,  True,  True,  "a",
     True,  "b", 2, 2, True,  True,  (0, 0), (3, 3)),
    # inner now empty: deliver "b", nothing to load, valid falls
    (11, 0, 1, 0, 1, None, None, True,  False, False, True,  "b",
     False, "b", 2, 2, True,  True,  (0, 0), (3, 3)),
    # read pointer crosses back; full deasserts only after sync latency
    (12, 1, 0, 1, 0, "d", False, None, None, None, None, None,
     False, "b", 2, 2, True,  True,  (3, 0), (3, 3)),
    (13, 1, 0, 1, 0, "d", False, None, None, None, None, None,
     False, "b", 2, 2, True,  True,  (3, 3), (3, 3)),
    (14, 1, 0, 1, 0, "d", False, None, None, None, None, None,
     False, "b", 2, 2, False, True,  (3, 3), (3, 3)),
    (15, 1, 0, 1, 0, "d", True,  None, None, None, None, None,
     False, "b", 3, 2, False, True,  (3, 3), (3, 3)),
    (16, 1, 0, 1, 0, "e", True,  None, None, None, None, None,  # wptr wraps
     False, "b", 0, 2, True,  True,  (3, 3), (3, 3)),
    # new writes cross to the read domain (again: no queue-length prefetch)
    (17, 0, 1, 0, 1, None, None, True,  False, False, False, None,
     False, "b", 0, 2, True,  True,  (3, 3), (0, 3)),
    (18, 0, 1, 0, 1, None, None, True,  False, False, False, None,
     False, "b", 0, 2, True,  True,  (3, 3), (0, 0)),
    (19, 0, 1, 0, 1, None, None, True,  False, False, False, None,
     False, "b", 0, 2, True,  False, (3, 3), (0, 0)),
    # prefetch "d", then same-edge consume+load of "e"; rptr wraps at t=21
    (20, 0, 1, 0, 1, None, None, True,  True,  True,  False, None,
     True,  "d", 0, 3, True,  False, (3, 3), (0, 0)),
    (21, 0, 1, 0, 1, None, None, True,  True,  True,  True,  "d",
     True,  "e", 0, 0, True,  True,  (3, 3), (0, 0)),
    (22, 0, 1, 0, 1, None, None, True,  False, False, True,  "e",
     False, "e", 0, 0, True,  True,  (3, 3), (0, 0)),
]


class TestHandWaveformOutReg(unittest.TestCase):
    def test_exact_cycle_by_cycle_trace(self):
        fifo = ReadyValidFIFO(k=1)
        events = [RVEvent(t, bool(wc), bool(rc), bool(we), bool(rd), data)
                  for (t, wc, rc, we, rd, data, *_rest) in HAND_RV_WAVE]
        records = fifo.run(events)

        for rec, row in zip(records, HAND_RV_WAVE):
            (t, wc, rc, we, rd, data, wacc, ren, racc, load, deliv, ddata,
             valid, out, wp, rp, full, empty, r2w, w2r) = row
            with self.subTest(time=t):
                self.assertEqual(rec.w_accept, wacc)
                self.assertEqual(rec.ren, ren)
                self.assertEqual(rec.r_accept, racc)
                self.assertEqual(rec.load, load)
                self.assertEqual(rec.delivered, deliv)
                self.assertEqual(rec.delivered_data, ddata)
                self.assertIs(rec.valid_after, valid)
                self.assertEqual(rec.out_after, out)
                self.assertEqual(rec.inner.wptr_after, wp)
                self.assertEqual(rec.inner.rptr_after, rp)
                self.assertIs(rec.inner.full_after, full)
                self.assertIs(rec.inner.empty_after, empty)
                self.assertEqual(rec.inner.r2w_after, r2w)
                self.assertEqual(rec.inner.w2r_after, w2r)

        # counters: 4 accepted writes, 4 loads (inner reads), 4 deliveries
        self.assertEqual(fifo.accepted_writes, 4)
        self.assertEqual(fifo.accepted_reads, 4)
        self.assertEqual(fifo.delivered, 4)
        self.assertEqual(fifo.inflight, 0)

    def test_waveform_anchors(self):
        fifo = ReadyValidFIFO(k=1)
        records = fifo.run(
            RVEvent(t, bool(wc), bool(rc), bool(we), bool(rd), data)
            for (t, wc, rc, we, rd, data, *_rest) in HAND_RV_WAVE
        )
        by_time = {r.time: r for r in records}

        # t=8/9 continuous backpressure: output word and read pointer frozen
        for t in (8, 9):
            rec = by_time[t]
            self.assertFalse(rec.ren)
            self.assertEqual(rec.out_after, "a")
            self.assertEqual(rec.out_before, "a")
            self.assertEqual(rec.inner.rptr_after, 1)
            self.assertTrue(rec.valid_after)

        # t=10 same-edge consume+load: delivered "a", loaded "b", valid held
        rec = by_time[10]
        self.assertTrue(rec.delivered)
        self.assertEqual(rec.delivered_data, "a")
        self.assertTrue(rec.load)
        self.assertEqual(rec.out_after, "b")
        self.assertTrue(rec.valid_after)

        # t=16 write pointer wraps 3 -> 0; t=21 read pointer wraps 3 -> 0
        self.assertEqual(by_time[16].inner.wptr_before, 3)
        self.assertEqual(by_time[16].inner.wptr_after, 0)
        self.assertEqual(by_time[21].inner.rptr_before, 3)
        self.assertEqual(by_time[21].inner.rptr_after, 0)

        # deliveries are a strict prefix of accepted writes, in order
        delivered_seq = [r.delivered_data for r in records if r.delivered]
        written = [r.wdata for r in records if r.w_accept]
        self.assertEqual(written, ["a", "b", "d", "e"])
        self.assertEqual(delivered_seq, written)


class TestPrefetchOnlyOnEmptyFlag(unittest.TestCase):
    def test_no_prefetch_from_true_queue_length(self):
        """A written word not yet across the synchronizer must not be
        prefetched, even though the abstract queue length is already 1."""
        fifo = ReadyValidFIFO(k=2)
        rec = fifo.run_one(RVEvent(1, True, False, wen=True, wdata="x"))
        self.assertTrue(rec.w_accept)          # word really stored now
        self.assertEqual(fifo.inflight, 1)     # abstract backlog is 1 ...
        # ... but the read domain cannot see it for two sync stages plus the
        # registered empty flag: three read edges must refuse to load.
        for t in (2, 3, 4):
            rec = fifo.run_one(RVEvent(t, False, True, ready=True))
            self.assertFalse(rec.load, f"t={t} must not prefetch")
            self.assertFalse(fifo.valid)
        rec = fifo.run_one(RVEvent(5, False, True, ready=True))
        self.assertTrue(rec.load)              # old empty finally low
        self.assertTrue(fifo.valid)
        self.assertEqual(fifo.out_data, "x")

    def test_dequeue_request_follows_valid_and_ready_only(self):
        fifo = ReadyValidFIFO(k=2)
        fifo.run_one(RVEvent(1, True, False, wen=True, wdata=1))
        for t in (2, 3, 4, 5):
            fifo.run_one(RVEvent(t, False, True, ready=False))
        self.assertTrue(fifo.valid)
        # valid & ~ready -> ren must be low (structural stall)
        rec = fifo.run_one(RVEvent(6, False, True, ready=False))
        self.assertFalse(rec.ren)
        # invalid -> ren high regardless of ready
        fifo2 = ReadyValidFIFO(k=2)
        rec = fifo2.run_one(RVEvent(1, False, True, ready=False))
        self.assertTrue(rec.ren)
        rec = fifo2.run_one(RVEvent(2, False, True, ready=True))
        self.assertTrue(rec.ren)


class TestBackpressureHold(unittest.TestCase):
    def test_sustained_stall_keeps_output_stable(self):
        fifo = ReadyValidFIFO(k=2)
        fifo.run_one(RVEvent(1, True, False, wen=True, wdata="hold-me"))
        for t in (2, 3, 4, 5):
            fifo.run_one(RVEvent(t, False, True, ready=False))
        self.assertTrue(fifo.valid)
        rptr = fifo.rptr
        # 20 consecutive stalled read edges: word, valid, rptr all frozen
        for t in range(6, 26):
            rec = fifo.run_one(RVEvent(t, False, True, ready=False))
            self.assertFalse(rec.ren)
            self.assertFalse(rec.r_accept)
            self.assertEqual(rec.out_after, "hold-me")
            self.assertTrue(rec.valid_after)
            self.assertEqual(rec.inner.rptr_after, rptr)
            self.assertEqual(fifo.delivered, 0)
        # release: the very same word is delivered, not a re-fetched one
        rec = fifo.run_one(RVEvent(26, False, True, ready=True))
        self.assertTrue(rec.delivered)
        self.assertEqual(rec.delivered_data, "hold-me")
        self.assertEqual(fifo.delivered, 1)


class TestResetWhileValid(unittest.TestCase):
    def test_reset_clears_valid_and_inner_state(self):
        fifo = ReadyValidFIFO(k=1)
        fifo.run_one(RVEvent(1, True, False, wen=True, wdata="z"))
        for t in (2, 3, 4, 5):
            fifo.run_one(RVEvent(t, False, True, ready=False))
        self.assertTrue(fifo.valid)            # output register full
        self.assertEqual(fifo.out_data, "z")
        fifo.reset()
        self.assertFalse(fifo.valid)           # valid cleared by reset
        self.assertIsNone(fifo.out_data)
        self.assertTrue(fifo.empty)
        self.assertFalse(fifo.full)
        self.assertEqual(fifo.wptr, 0)
        self.assertEqual(fifo.rptr, 0)
        self.assertEqual(fifo.accepted_writes, 0)
        self.assertEqual(fifo.delivered, 0)
        self.assertEqual(fifo.steps, ())
        # and the wrapper is fully usable afterwards
        fifo.run_one(RVEvent(1, True, False, wen=True, wdata="again"))
        for t in (2, 3, 4, 5):
            fifo.run_one(RVEvent(t, False, True, ready=True))
        self.assertTrue(fifo.valid)
        rec = fifo.run_one(RVEvent(6, False, True, ready=True))
        self.assertEqual(rec.delivered_data, "again")


class TestCapacityAccounting(unittest.TestCase):
    def test_output_register_is_extra_to_depth(self):
        """depth stays the inner depth; the wrapper can hold depth+1 words."""
        fifo = ReadyValidFIFO(k=1)
        self.assertEqual(fifo.depth, 2)        # NOT 3: register not hidden
        fifo.run_one(RVEvent(1, True, False, wen=True, wdata="a"))
        fifo.run_one(RVEvent(2, True, False, wen=True, wdata="b"))
        self.assertTrue(fifo.full)
        for t in (3, 4, 5, 6):
            fifo.run_one(RVEvent(t, False, True, ready=False))
        self.assertTrue(fifo.valid)            # "a" prefetched into register
        # let the write domain learn the freed slot, then refill it
        for t in (7, 8, 9):
            fifo.run_one(RVEvent(t, True, False))
        self.assertFalse(fifo.full)
        fifo.run_one(RVEvent(10, True, False, wen=True, wdata="c"))
        # inner FIFO holds 2 ("b","c") + output register holds 1 ("a")
        self.assertEqual(fifo.accepted_writes, 3)
        self.assertEqual(fifo.delivered, 0)
        self.assertEqual(fifo.inflight, 3)
        self.assertEqual(fifo.inflight, fifo.depth + 1)
        # accounting identity: backlog = inner occupancy + valid slot
        inner_occ = fifo.accepted_writes - fifo.accepted_reads
        self.assertEqual(inner_occ, 2)
        self.assertEqual(fifo.inflight, inner_occ + int(fifo.valid))


class TestDifferentialOutReg(unittest.TestCase):
    """Compare every wrapper register against RefOutReg over schedules."""

    def _diff_run(self, k, wp, rp, n, seed=1, w_stops=(), r_stops=(), bps=()):
        schedule = make_rv_schedule(
            k, n, w_period=wp, r_period=rp, seed=seed,
            stop_windows=w_stops, r_stop_windows=r_stops,
            backpressure_windows=bps,
        )
        dut = ReadyValidFIFO(k=k)
        ref = RefOutReg(k)
        for (t, wc, rc, we, rd, data) in schedule:
            rec = dut.run_one(RVEvent(t, wc, rc, we, rd, data))
            exp = ref.step(wc, rc, we, rd, data)
            with self.subTest(k=k, wp=wp, rp=rp, t=t):
                self.assertEqual(rec.ren, exp["ren"])
                self.assertEqual(rec.w_accept, exp["w_accept"])
                self.assertEqual(rec.r_accept, exp["r_accept"])
                self.assertEqual(rec.load, exp["load"])
                self.assertEqual(rec.delivered, exp["delivered"])
                self.assertEqual(rec.delivered_data, exp["delivered_data"])
                self.assertIs(rec.valid_after, exp["valid"])
                self.assertEqual(rec.out_after, exp["out"])
                self.assertEqual(rec.inner.wptr_after, exp["wptr"])
                self.assertEqual(rec.inner.rptr_after, exp["rptr"])
                self.assertIs(rec.inner.full_after, exp["full"])
                self.assertIs(rec.inner.empty_after, exp["empty"])
                # capacity accounting, every single cycle:
                #   backlog = inner occupancy + output-register slot
                self.assertEqual(
                    dut.inflight,
                    exp["inner_occupancy"] + int(exp["valid"]),
                )
                self.assertLessEqual(exp["inner_occupancy"], 1 << k)
                self.assertLessEqual(dut.inflight, (1 << k) + 1)
        # independent sequences: deliveries are a prefix of accepted writes
        self.assertEqual(ref.delivered_seq, ref.written[: ref.n_delivered])
        self.assertEqual(dut.delivered, ref.n_delivered)
        self.assertEqual(dut.accepted_writes, len(ref.written))
        # remainder = output-register word (oldest) + inner queue, in order
        remainder = ([ref.out] if ref.valid else []) + list(ref.inner.queue)
        self.assertEqual(remainder, ref.written[ref.n_delivered:])

    def test_clock_ratios_and_coincidence(self):
        ratios = [(1, 1), (2, 3), (3, 2), (2, 5), (5, 2)]
        for k in (1, 2, 3, 5):
            for wp, rp in ratios:
                self._diff_run(k, wp, rp, n=300, seed=100 * k + wp * 7 + rp)

    def test_continuous_backpressure_windows(self):
        self._diff_run(2, 2, 3, 300, seed=5, bps=((30, 120), (200, 260)))
        self._diff_run(1, 1, 1, 300, seed=6, bps=((10, 290),))

    def test_stopped_clocks_with_backpressure(self):
        self._diff_run(2, 2, 3, 240, seed=9, w_stops=((40, 80),),
                       r_stops=((140, 190),), bps=((60, 100),))

    def test_pointer_wrap_many_times(self):
        self._diff_run(1, 1, 1, 400, seed=42)
        self._diff_run(5, 2, 3, 4000, seed=77, bps=((1000, 1500),))


class TestRVEventContract(unittest.TestCase):
    def test_ready_and_clock_contract(self):
        with self.assertRaises(AsyncFIFOError):
            RVEvent(1, False, False)
        with self.assertRaises(AsyncFIFOError):
            RVEvent(1, True, False, ready=1)       # non-bool ready
        with self.assertRaises(AsyncFIFOError):
            RVEvent(1, True, False, wen=0)
        with self.assertRaises(AsyncFIFOError):
            RVEvent(True, True, False)             # bool time rejected
        with self.assertRaises(AsyncFIFOError):
            RVEvent(-1, True, False)

    def test_wdata_rules(self):
        with self.assertRaisesRegex(AsyncFIFOError, "wdata"):
            RVEvent(1, True, False, wen=False, wdata=9)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(AsyncFIFOError, "finite"):
                    RVEvent(1, True, False, wen=True, wdata=bad)

    def test_non_increasing_time_rejected(self):
        fifo = ReadyValidFIFO(k=2)
        fifo.run_one(RVEvent(5, True, False))
        with self.assertRaisesRegex(AsyncFIFOError, "strictly greater"):
            fifo.run_one(RVEvent(5, True, False))

    def test_run_rejects_single_event_and_non_events(self):
        fifo = ReadyValidFIFO(k=2)
        with self.assertRaises(AsyncFIFOError):
            fifo.run(RVEvent(1, True, False))
        with self.assertRaises(AsyncFIFOError):
            fifo.run([RVEvent(1, True, False), "nope"])
        with self.assertRaises(AsyncFIFOError):
            fifo.run(42)

    def test_failed_batch_leaves_no_partial_change(self):
        fifo = ReadyValidFIFO(k=2)
        fifo.run([RVEvent(1, True, False, wen=True, wdata="q")])
        for t in (2, 3, 4, 5):
            fifo.run_one(RVEvent(t, False, True, ready=False))
        self.assertTrue(fifo.valid)
        before = fifo.snapshot()
        before_steps = fifo.steps
        bad_batch = [
            RVEvent(6, True, False, wen=True, wdata="ok"),
            RVEvent(6, False, True),               # duplicate time
        ]
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[1\]"):
            fifo.run(bad_batch)
        self.assertEqual(fifo.snapshot(), before)
        self.assertEqual(fifo.steps, before_steps)
        self.assertTrue(fifo.valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
