"""Tests for per-domain reset, generation tokens and bidirectional
re-synchronization acknowledgement.

Expected values come from:
1. A hand-derived k=2 reset waveform (key register anchors per step).
2. ``reference_model.RefFIFO`` -- the independent reimplementation that never
   imports ``afifo`` -- extended with the same generation handshake, plus an
   abstract ``deque`` for accepted-data order / occupancy bounds.
The reset schedule assertions below are literal hand computations, never read
out of the model under test.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_model import RefFIFO, make_schedule_rst  # noqa: E402

from afifo import AsyncFIFO, Event  # noqa: E402
from afifo.model import AsyncFIFOError  # noqa: E402


def fill(fifo, start_t, items, period=1):
    """Accepted writes on write-only edges; return the next time stamp."""
    t = start_t
    for d in items:
        rec = None
        while True:
            rec = fifo.run_one(Event(t, True, False, wen=True, wdata=d))
            t += 1
            if rec.w_accept:
                break
    return t


def clock(fifo, t, w=False, r=False, **kw):
    return fifo.run_one(Event(t, bool(w), bool(r), **kw))


def resync(fifo, t):
    """Advance alternating edges until synced; return next time stamp.

    Four partner/self edge pairs always suffice (two each direction); the
    helper simply steps until the handshake completes so the test body reads
    as intent rather than bookkeeping.
    """
    guard = 0
    while not fifo.synced:
        clock(fifo, t, w=True, r=True)
        t += 1
        guard += 1
        assert guard < 50, "handshake never completed"
    return t


def drain_one(fifo, t, max_edges=8):
    """Request reads on consecutive read edges until one is accepted.

    Returns ``(record, next_t)``.  The accepted data emerges after the normal
    two-stage synchronizer plus one flag-evaluation edge (never sooner than
    the fourth read edge after its write) -- this helper counts real steps.
    """
    last = None
    for _ in range(max_edges):
        last = fifo.run_one(Event(t, False, True, ren=True))
        t += 1
        if last.r_accept:
            return last, t
    raise AssertionError("read never accepted within bound")


class TestResetContract(unittest.TestCase):
    def test_rst_flags_must_be_bool(self):
        with self.assertRaises(AsyncFIFOError):
            Event(1, False, False, wrst=1)
        with self.assertRaises(AsyncFIFOError):
            Event(1, False, False, rrst=1.0)

    def test_reset_step_needs_no_clock(self):
        # asynchronous: a reset with both clocks stopped is legal
        Event(1, False, False, wrst=True)
        Event(2, False, False, rrst=True)
        # but a bare step with no clock and no reset is still rejected
        with self.assertRaisesRegex(AsyncFIFOError, "wrst"):
            Event(3, False, False)

    def test_global_reset_restarts_generations(self):
        f = AsyncFIFO(k=2)
        f.run_one(Event(1, False, False, wrst=True))
        self.assertEqual(f.w_generation, 1)
        f.reset()
        self.assertEqual((f.w_generation, f.r_generation), (0, 0))
        self.assertTrue(f.synced)
        self.assertEqual(f.steps, ())
        self.assertEqual((f.accepted_writes, f.accepted_reads), (0, 0))


class TestWriteDomainResetHandwave(unittest.TestCase):
    """Hand-derived anchors for one write-domain reset on k=2."""

    def setUp(self):
        self.f = AsyncFIFO(k=2)
        # fill depth 4 with writes on t=1..4
        for t, d in zip(range(1, 5), "ABCD"):
            self.f.run_one(Event(t, True, False, wen=True, wdata=d))
        self.assertTrue(self.f.full)

    def test_reset_step_state(self):
        rec = self.f.run_one(Event(5, False, False, wrst=True))
        self.assertTrue(rec.wrst)
        self.assertEqual(rec.wgen_before, 0)
        self.assertEqual(rec.wgen_after, 1)
        self.assertEqual(rec.rgen_after, 0)
        self.assertEqual((rec.wptr_after, rec.rptr_after), (0, 0))
        self.assertFalse(rec.full_after)
        self.assertTrue(rec.empty_after)
        self.assertFalse(rec.synced_after)
        self.assertTrue(rec.synced_before)
        self.assertIsNone(rec.w_accept)
        self.assertIsNone(rec.r_accept)
        # stale counters are cleared immediately
        self.assertEqual(self.f.accepted_writes, 0)
        self.assertEqual(self.f.accepted_reads, 0)

    def test_handshake_waveform(self):
        f = self.f
        f.run_one(Event(5, False, False, wrst=True))

        # two read-domain edges propagate the new generation to s2 ...
        r6 = f.run_one(Event(6, False, True, ren=True))
        self.assertEqual(r6.wgen2r_after, (1, 0))
        self.assertEqual(r6.rack_reg_after, 0)
        self.assertFalse(r6.r_accept)          # not synced -> no passage
        r7 = f.run_one(Event(7, False, True, ren=True))
        self.assertEqual(r7.wgen2r_after, (1, 1))
        self.assertEqual(r7.rack_reg_after, 0)
        self.assertFalse(r7.r_accept)
        # ... the third read edge LEARNS it: read view restarts at ptr 0
        r8 = f.run_one(Event(8, False, True, ren=True))
        self.assertEqual(r8.rack_reg_before, 0)
        self.assertEqual(r8.rack_reg_after, 1)
        self.assertEqual(r8.rptr_after, 0)
        self.assertTrue(r8.empty_after)
        self.assertFalse(r8.r_accept)          # acknowledgement not back yet
        self.assertFalse(r8.synced_after)

        # two write-domain edges carry the acknowledgement back
        w9 = f.run_one(Event(9, True, False, wen=True))
        self.assertEqual(w9.rack2w_after, (1, 0))
        self.assertFalse(w9.w_accept)
        w10 = f.run_one(Event(10, True, False, wen=True))
        self.assertEqual(w10.rack2w_after, (1, 1))
        self.assertTrue(w10.synced_after)
        self.assertFalse(w10.w_accept)         # decision used old synced=False

        # only AFTER sync does data move again; FIFO is logically empty
        r11 = f.run_one(Event(11, False, True, ren=True))
        self.assertFalse(r11.r_accept)         # genuinely empty
        w12 = f.run_one(Event(12, True, False, wen=True, wdata="E"))
        self.assertTrue(w12.w_accept)
        self.assertEqual(f.accepted_writes, 1)
        # drain E through the normal synchronizer + flag latency: empty
        # deasserts at the end of the third read edge after the write, so the
        # fourth edge accepts (same latency as the pre-reset model)
        r13 = f.run_one(Event(13, False, True, ren=True))
        r14 = f.run_one(Event(14, False, True, ren=True))
        r15 = f.run_one(Event(15, False, True, ren=True))
        self.assertFalse(r13.r_accept)
        self.assertFalse(r14.r_accept)
        self.assertFalse(r15.r_accept)
        r16 = f.run_one(Event(16, False, True, ren=True))
        self.assertTrue(r16.r_accept)
        self.assertEqual(r16.rdata, "E")       # no stale A..D survives
        self.assertEqual(f.accepted_reads, 1)


class TestResetProperties(unittest.TestCase):
    def test_write_reset_clears_full_and_memory(self):
        f = AsyncFIFO(k=1)
        f.run([Event(1, True, False, wen=True, wdata="a"),
               Event(2, True, False, wen=True, wdata="b")])
        self.assertTrue(f.full)
        f.run_one(Event(3, False, False, wrst=True))
        self.assertFalse(f.full)
        self.assertEqual(f.wptr, 0)
        t = resync(f, 4)
        # nothing old is readable: empty until a fresh write
        rec = f.run_one(Event(t, False, True, ren=True))
        self.assertFalse(rec.r_accept)
        f.run_one(Event(t + 1, True, False, wen=True, wdata="new"))
        rec, _ = drain_one(f, t + 2)
        self.assertEqual(rec.rdata, "new")

    def test_read_reset_clears_data_and_restarts_empty(self):
        f = AsyncFIFO(k=2)
        fill(f, 1, ["a", "b", "c"])
        f.run_one(Event(10, False, False, rrst=True))
        self.assertTrue(f.empty)
        self.assertEqual(f.rptr, 0)
        self.assertEqual(f.r_generation, 1)
        self.assertFalse(f.synced)
        t = resync(f, 11)
        rec = f.run_one(Event(t, False, True, ren=True))
        self.assertFalse(rec.r_accept)         # old a/b/c are gone
        f.run_one(Event(t + 1, True, False, wen=True, wdata="z"))
        rec, _ = drain_one(f, t + 2)
        self.assertEqual(rec.rdata, "z")

    def test_both_resets_same_step(self):
        f = AsyncFIFO(k=2)
        fill(f, 1, ["a", "b"])
        rec = f.run_one(Event(20, True, True, wen=True, ren=True,
                              wrst=True, rrst=True))
        self.assertIs(rec.w_accept, False)
        self.assertIs(rec.r_accept, False)
        self.assertEqual((rec.wgen_after, rec.rgen_after), (1, 1))
        self.assertFalse(rec.synced_after)     # handshake not yet acknowledged
        self.assertEqual((f.wptr, f.rptr), (0, 0))
        self.assertFalse(f.full)
        self.assertTrue(f.empty)
        # even the next edge cannot pass data before acknowledgement
        early = f.run_one(Event(21, True, True, wen=True, ren=True,
                                wdata="q"))
        self.assertFalse(early.w_accept)
        self.assertFalse(early.r_accept)
        t = resync(f, 22)
        # after the handshake completes the FIFO behaves like power-on
        nxt = f.run_one(Event(t, True, True, wen=True, ren=True, wdata="q"))
        self.assertTrue(nxt.w_accept)
        self.assertFalse(nxt.r_accept)

    def test_reset_coincident_edge_samples_post_reset(self):
        """Reset-before-sampling: partner s1 latches Gray 0 / new gen."""
        f = AsyncFIFO(k=2)
        fill(f, 1, ["a", "b", "c", "d"])
        self.assertTrue(f.full)
        rec = f.run_one(Event(20, False, True, ren=True, wrst=True))
        # read input stage sampled the write Gray AFTER the write reset
        self.assertEqual(rec.w2r_after, (0, 0))
        self.assertEqual(rec.wgen2r_after, (1, 0))
        self.assertFalse(rec.r_accept)
        self.assertEqual(rec.wptr_after, 0)

    def test_reset_coincident_with_its_own_clock(self):
        """wrst on a wclk edge still acts as reset (no edge logic in domain);
        the coincident partner edge samples post-reset values."""
        f = AsyncFIFO(k=2)
        fill(f, 1, ["a", "b"])
        rec = f.run_one(Event(20, True, True, wen=True, ren=True,
                              wdata="x", wrst=True))
        self.assertFalse(rec.w_accept)
        self.assertFalse(rec.r_accept)
        self.assertEqual(rec.wptr_after, 0)
        self.assertFalse(rec.full_after)
        self.assertEqual(rec.w2r_after, (0, 0))
        self.assertEqual(rec.wgen2r_after, (1, 0))

        g = AsyncFIFO(k=1)
        g.run_one(Event(1, True, False, wen=True, wdata="a"))
        rec = g.run_one(Event(2, True, True, ren=True, rrst=True))
        self.assertFalse(rec.r_accept)
        self.assertEqual(rec.rptr_after, 0)
        self.assertEqual(rec.rgen2w_after, (1, 0))
        self.assertTrue(rec.empty_after)

    def test_event_backward_compatible_tuple_and_equality(self):
        e = Event(7, True, False, True, False, "x")
        self.assertEqual(tuple(e), (7, True, False, True, False, "x"))
        self.assertEqual(e, Event(7, True, False, True, False, "x"))
        self.assertNotEqual(e, Event(7, True, False, True, False, "x",
                                     wrst=True))
        self.assertEqual(Event(1, False, False, rrst=True),
                         Event(1, False, False, rrst=True))


class TestNoPassageBeforeAcknowledged(unittest.TestCase):
    def test_enables_asserted_throughout_resync_are_rejected(self):
        f = AsyncFIFO(k=3)
        fill(f, 1, ["x", "y"])
        t = 10
        f.run_one(Event(t, False, False, wrst=True))
        t += 1
        rejected = 0
        while not f.synced:
            rec = f.run_one(Event(t, True, True, wen=True, ren=True,
                                  wdata="nope"))
            if rec.w_accept is False:
                rejected += 1
            self.assertFalse(rec.r_accept)
            t += 1
        self.assertGreaterEqual(rejected, 1)
        # the data offered during resync never entered
        self.assertEqual(f.accepted_writes, 0)
        # first synced step is the earliest possible accepted access
        rec = f.run_one(Event(t, True, False, wen=True, wdata="ok"))
        self.assertTrue(rec.w_accept)
        self.assertEqual(f.accepted_writes, 1)

    def test_synced_property_tracks_record(self):
        f = AsyncFIFO(k=1)
        f.run_one(Event(1, False, False, rrst=True))
        states = []
        t = 2
        while True:
            rec = f.run_one(Event(t, True, True))
            states.append(rec.synced_after)
            t += 1
            if f.synced:
                break
        self.assertFalse(states[0])
        self.assertTrue(states[-1])


class TestReReset(unittest.TestCase):
    def test_second_reset_between_edges_takes_new_generation(self):
        f = AsyncFIFO(k=2)
        fill(f, 1, ["a", "b"])
        f.run_one(Event(10, False, False, wrst=True))         # gen 1
        f.run_one(Event(11, False, True))                     # s1=1
        f.run_one(Event(12, False, False, wrst=True))         # gen 2 again
        self.assertEqual(f.w_generation, 2)
        self.assertFalse(f.synced)
        t = resync(f, 13)
        self.assertTrue(f.synced)
        # the converged acknowledgement is for the NEWEST generation
        self.assertEqual(f.snapshot()["rack2w"][1], 2)
        self.assertEqual(f.snapshot()["rack_reg"], 2)
        rec = f.run_one(Event(t, True, False, wen=True, wdata="g2"))
        self.assertTrue(rec.w_accept)

    def test_opposite_domain_reset_during_pending_handshake(self):
        f = AsyncFIFO(k=2)
        fill(f, 1, ["a", "b", "c"])
        f.run_one(Event(10, False, False, wrst=True))
        f.run_one(Event(11, False, True))
        f.run_one(Event(12, False, False, rrst=True))         # other side too
        self.assertEqual((f.w_generation, f.r_generation), (1, 1))
        t = resync(f, 13)
        self.assertTrue(f.synced)
        self.assertEqual((f.wptr, f.rptr), (0, 0))
        # empty FIFO, fresh traffic works in both directions
        self.assertFalse(f.run_one(Event(t, False, True, ren=True)).r_accept)
        f.run_one(Event(t + 1, True, False, wen=True, wdata="v"))
        rec, _ = drain_one(f, t + 2)
        self.assertEqual(rec.rdata, "v")

    def test_repeated_resets_generations_monotone(self):
        f = AsyncFIFO(k=1)
        t = 1
        for g in range(1, 6):
            f.run_one(Event(t, False, False, wrst=True))
            t += 1
            self.assertEqual(f.w_generation, g)
            t = resync(f, t)
            t += 1
        self.assertTrue(f.synced)


class TestFullAndWrapAroundReset(unittest.TestCase):
    def test_full_then_reset_allows_writes_again(self):
        f = AsyncFIFO(k=1)
        f.run([Event(1, True, False, wen=True, wdata=1),
               Event(2, True, False, wen=True, wdata=2)])
        self.assertTrue(f.full)
        rej = f.run_one(Event(3, True, False, wen=True, wdata=3))
        self.assertFalse(rej.w_accept)
        f.run_one(Event(4, False, False, wrst=True))
        t = resync(f, 5)
        outs = []
        for d in (10, 11):
            rec = f.run_one(Event(t, True, False, wen=True, wdata=d))
            self.assertTrue(rec.w_accept)
            t += 1
        self.assertTrue(f.full)

    def test_pointer_wrap_after_many_resets(self):
        """k=1: pointers wrap mod 4; resets must keep the peers consistent."""
        f = AsyncFIFO(k=1)
        t = 1
        for cycle in range(6):
            # write until the write side reports full
            while not f.full:
                f.run_one(Event(t, True, False, wen=True, wdata=(cycle, t)))
                t += 1
            f.run_one(Event(t, False, False,
                            wrst=(cycle % 2 == 0),
                            rrst=(cycle % 2 == 1)))
            t += 1
            t = resync(f, t)
            self.assertEqual((f.wptr, f.rptr), (0, 0))
            self.assertFalse(f.full)
            self.assertTrue(f.empty)


class TestDifferentialResetSchedules(unittest.TestCase):
    """Every register vs RefFIFO over deterministic schedules with resets."""

    def _diff(self, k, wp, rp, n, wrst_times=(), rrst_times=(), seed=1,
              w_stops=(), r_stops=()):
        schedule = make_schedule_rst(
            k, n, w_period=wp, r_period=rp, seed=seed,
            wrst_times=wrst_times, rrst_times=rrst_times,
            stop_windows=w_stops, r_stop_windows=r_stops)
        dut = AsyncFIFO(k=k)
        ref = RefFIFO(k=k)
        written, read_out = [], []
        for (t, wc, rc, we, re_, data, wrst, rrst) in schedule:
            rec = dut.run_one(Event(t, wc, rc, we, re_, data,
                                    wrst=wrst, rrst=rrst))
            exp = ref.step(wc, rc, we, re_, data, wrst=wrst, rrst=rrst)
            # a reset discards all previously accepted data on both sides
            if wrst or rrst:
                written.clear()
                read_out.clear()
            with self.subTest(k=k, t=t):
                self.assertEqual(rec.w_accept, exp["w_accept"])
                self.assertEqual(rec.r_accept, exp["r_accept"])
                self.assertEqual(rec.wptr_after, exp["wptr"])
                self.assertEqual(rec.rptr_after, exp["rptr"])
                self.assertIs(rec.full_after, exp["full"])
                self.assertIs(rec.empty_after, exp["empty"])
                self.assertEqual(rec.r2w_after, exp["r2w"])
                self.assertEqual(rec.w2r_after, exp["w2r"])
                self.assertEqual(rec.wgen_after, exp["wgen"])
                self.assertEqual(rec.rgen_after, exp["rgen"])
                self.assertEqual(rec.wgen2r_after, exp["wgen2r"])
                self.assertEqual(rec.rgen2w_after, exp["rgen2w"])
                self.assertEqual(rec.rack2w_after, exp["rack2w"])
                self.assertEqual(rec.wack2r_after, exp["wack2r"])
                self.assertEqual(rec.rack_reg_after, exp["rack_reg"])
                self.assertEqual(rec.wack_reg_after, exp["wack_reg"])
                self.assertIs(rec.synced_after, exp["synced"])
                self.assertTrue(0 <= exp["occupancy"] <= 1 << k)
                if rec.w_accept:
                    written.append(data)
                if rec.r_accept:
                    read_out.append(rec.rdata)
        self.assertEqual(dut.accepted_writes, ref.n_w)
        self.assertEqual(dut.accepted_reads, ref.n_r)
        self.assertEqual(read_out, written[: ref.n_r])
        self.assertEqual(list(ref.queue), written[ref.n_r:])

    def test_single_write_resets_all_k(self):
        for k in (1, 2, 3, 4, 5):
            self._diff(k, 2, 3, 220, wrst_times=(40, 150), seed=10 + k)

    def test_single_read_resets_all_k(self):
        for k in (1, 2, 3, 4, 5):
            self._diff(k, 3, 2, 220, rrst_times=(30, 120, 200), seed=20 + k)

    def test_simultaneous_resets(self):
        for k in (1, 2, 3, 5):
            self._diff(k, 2, 2, 200,
                       wrst_times=(25, 110), rrst_times=(25, 110),
                       seed=30 + k)

    def test_re_reset_while_pending(self):
        # second reset lands before four edges can complete the handshake
        self._diff(2, 5, 7, 300, wrst_times=(30, 33, 90, 92), seed=41)
        self._diff(3, 7, 5, 300, rrst_times=(20, 22, 130, 133), seed=42)
        self._diff(1, 3, 3, 300,
                   wrst_times=(15, 17), rrst_times=(16, 18), seed=43)

    def test_resets_with_stopped_clocks(self):
        self._diff(3, 2, 3, 320, wrst_times=(50, 200),
                   w_stops=((60, 110),), r_stops=((150, 190),), seed=51)
        self._diff(2, 3, 2, 320, rrst_times=(70, 210),
                   w_stops=((30, 90),), seed=52)

    def test_wrap_with_resets_k1_and_k5(self):
        # k=1 wraps pointers constantly with frequent resets
        self._diff(1, 1, 1, 600, wrst_times=(70, 200, 400),
                   rrst_times=(120, 300), seed=61)
        # k=5 long run crosses a 64-accept wrap plus resets
        self._diff(5, 2, 3, 5000, wrst_times=(900, 3000),
                   rrst_times=(2000,), seed=62)

    def test_reset_without_any_clock_edge(self):
        self._diff(2, 2, 3, 180, wrst_times=(37, 141), seed=71)


class TestSmallStateEnumeration(unittest.TestCase):
    """Exhaustive k=1 action sequences vs RefFIFO (independent oracle).

    Each step both clocks edge and one of four actions occurs: write, read,
    write reset, read reset.  Every register and both flags are compared to
    the independent reference, and every accepted payload is checked to be
    strictly newer than the latest reset (stale data may never be read).
    """

    ACTIONS = (0, 1, 2, 3)  # write / read / wrst / rrst

    def _run_sequence(self, seq):
        dut = AsyncFIFO(k=1)
        ref = RefFIFO(k=1)
        fresh = set()
        t = 0
        for a in seq:
            t += 1
            we, re_ = a == 0, a == 1
            wr, rr = a == 2, a == 3
            data = t if we else None
            rec = dut.run_one(Event(t, True, True, we, re_, data,
                                    wrst=wr, rrst=rr))
            exp = ref.step(True, True, we, re_, data, wrst=wr, rrst=rr)
            if wr or rr:
                fresh.clear()
            if rec.w_accept:
                fresh.add(data)
            if rec.r_accept:
                self.assertIn(rec.rdata, fresh)  # no pre-reset data survives
            self.assertEqual(rec.wptr_after, exp["wptr"], seq)
            self.assertEqual(rec.rptr_after, exp["rptr"], seq)
            self.assertIs(rec.full_after, exp["full"], seq)
            self.assertIs(rec.empty_after, exp["empty"], seq)
            self.assertEqual(rec.r2w_after, exp["r2w"], seq)
            self.assertEqual(rec.w2r_after, exp["w2r"], seq)
            self.assertEqual(rec.rack2w_after, exp["rack2w"], seq)
            self.assertEqual(rec.wack2r_after, exp["wack2r"], seq)
            self.assertIs(rec.synced_after, exp["synced"], seq)

    def test_all_sequences_up_to_length_seven(self):
        import itertools
        total = 0
        for depth in range(1, 8):
            for seq in itertools.product(self.ACTIONS, repeat=depth):
                self._run_sequence(seq)
                total += 1
        self.assertEqual(total, sum(4 ** d for d in range(1, 8)))


class TestBatchAtomicityWithResets(unittest.TestCase):
    def test_failed_batch_after_reset_leaves_state_intact(self):
        f = AsyncFIFO(k=2)
        f.run_one(Event(1, False, False, wrst=True))
        before = f.snapshot()
        before_steps = f.steps
        bad = [
            Event(2, True, False, wen=True, wdata="a"),
            Event(3, False, False, wrst=True),
            Event(3, True, False),                     # duplicate time
        ]
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[2\]"):
            f.run(bad)
        self.assertEqual(f.snapshot(), before)
        self.assertEqual(f.steps, before_steps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
