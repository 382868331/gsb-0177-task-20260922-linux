"""Tests for the one-writer/two-independent-reader FIFO."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_model import RefDualFIFO, make_dual_schedule  # noqa: E402

from afifo import DualAsyncFIFO, DualEvent, forward_distance  # noqa: E402
from afifo.model import AsyncFIFOError  # noqa: E402


def event_from_row(row):
    t, wclk, rclk0, rclk1, wen, ren0, ren1, wdata = row
    return DualEvent(
        t, bool(wclk), bool(rclk0), bool(rclk1),
        bool(wen), bool(ren0), bool(ren1), wdata,
    )


class TestGenerationDistance(unittest.TestCase):
    def test_wrapped_pointers_use_generation_not_unsigned_min(self):
        fifo = DualAsyncFIFO(k=2)
        self.assertEqual(forward_distance(1, 0, 2), 1)
        self.assertEqual(forward_distance(1, 5, 2), 4)
        self.assertEqual(fifo._slowest_reader(1, (0, 5)), 5)

    def test_full_distance_boundary(self):
        self.assertEqual(forward_distance(4, 0, 2), 4)
        self.assertEqual(forward_distance(0, 4, 2), 4)
        self.assertEqual(forward_distance(1, 5, 2), 4)


class TestDualDifferential(unittest.TestCase):
    def _diff(self, k, wp, p0, p1, n, seed, stops=None):
        stops = stops or ((), (), ())
        schedule = make_dual_schedule(
            k, n, w_period=wp, r0_period=p0, r1_period=p1,
            seed=seed, w_stops=stops[0], r0_stops=stops[1],
            r1_stops=stops[2],
        )
        dut = DualAsyncFIFO(k=k)
        ref = RefDualFIFO(k=k)
        consumed = [set(), set()]
        slot_payload = [None] * dut.depth

        for row in schedule:
            (t, wclk, rclk0, rclk1, wen, ren0, ren1, wdata) = row
            rec = dut.run_one(event_from_row(row))
            exp = ref.step(
                wclk, (rclk0, rclk1), wen, (ren0, ren1), wdata
            )

            if rec.w_accept:
                addr = rec.wptr_before & (dut.depth - 1)
                prior = slot_payload[addr]
                if prior is not None:
                    self.assertIn(prior, consumed[0])
                    self.assertIn(prior, consumed[1])
                slot_payload[addr] = rec.wdata
            for reader in range(2):
                if rec.r_accepts[reader]:
                    payload = rec.rdatas[reader]
                    consumed[reader].add(payload)

            with self.subTest(k=k, t=t):
                self.assertEqual(rec.w_accept, exp["w_accept"])
                self.assertEqual(rec.r_accepts, exp["r_accepts"])
                self.assertEqual(rec.wptr_after, exp["wptr"])
                self.assertEqual(rec.rptrs_after, exp["rptrs"])
                self.assertEqual(rec.full_after, exp["full"])
                self.assertEqual(rec.empties_after, exp["empties"])
                self.assertEqual(rec.r2w_after, exp["r2w"])
                self.assertEqual(rec.w2r_after, exp["w2r"])
                self.assertEqual(rec.release_ptr_after, exp["release_ptr"])
                self.assertEqual(
                    rec.synced_release_ptr_after,
                    exp["synced_release_ptr"],
                )
                self.assertEqual(rec.rdatas, exp["rdatas"])
                for q in exp["queues"]:
                    self.assertLessEqual(len(q), dut.depth)

    def test_clock_ratios_and_coincidence(self):
        ratios = [
            (1, 1, 1), (2, 3, 5), (5, 2, 3), (3, 5, 2),
            (2, 4, 6), (7, 3, 5),
        ]
        for k in (1, 2, 3, 5):
            for wp, p0, p1 in ratios:
                seed = 1000 + k * 31 + wp + p0 + p1
                self._diff(k, wp, p0, p1, 420, seed=seed)

    def test_wrapping_and_long_run(self):
        self._diff(1, 1, 1, 1, 500, seed=71)
        self._diff(2, 2, 3, 5, 900, seed=72)

    def test_one_reader_stops_then_writer_backpressures(self):
        self._diff(
            2, 2, 3, 5, 700, seed=83,
            stops=((), (), ((180, 520),)),
        )


class TestDualBehavior(unittest.TestCase):
    def test_each_payload_consumed_once_by_both_readers(self):
        fifo = DualAsyncFIFO(k=2)
        for seq in range(4):
            fifo.run_one(DualEvent(seq + 1, True, False, False,
                                   wen=True, wdata=f"D{seq}"))
        self.assertTrue(fifo.full)
        outputs = [[], []]
        t = 4
        for reader, clock in enumerate((lambda e: (False, e, False),
                                        lambda e: (False, False, e))):
            for edge in range(3):
                t += 1
                rec = fifo.run_one(DualEvent(t, *clock(True), ren0=True,
                                             ren1=True))
                if rec.r_accepts[reader]:
                    outputs[reader].append(rec.rdatas[reader])
        self.assertEqual(outputs[0], ["D0"])
        self.assertEqual(outputs[1], ["D0"])
        self.assertEqual(fifo.accepted_reads, (1, 1))
        self.assertEqual(fifo.release_ptr, 1)

    def test_fast_reader_cannot_release_slow_reader_slot(self):
        fifo = DualAsyncFIFO(k=1)
        fifo.run([
            DualEvent(1, True, False, False, wen=True, wdata="A"),
            DualEvent(2, True, False, False, wen=True, wdata="B"),
        ])
        for t in range(3, 9):
            fifo.run_one(DualEvent(t, False, True, False, ren0=True))
        self.assertEqual(fifo.rptrs, (2, 0))
        self.assertTrue(fifo.full)
        for t in range(9, 13):
            rec = fifo.run_one(DualEvent(t, True, False, False,
                                         wen=True, wdata="C"))
            self.assertFalse(rec.w_accept)
        self.assertEqual(fifo.wptr, 2)
        self.assertEqual(fifo.accepted_writes, 2)

    def test_coincident_edges_sample_old_state(self):
        fifo = DualAsyncFIFO(k=2)
        rec = fifo.run_one(DualEvent(1, True, True, True,
                                     wen=True, ren0=True, ren1=True,
                                     wdata="x"))
        self.assertTrue(rec.w_accept)
        self.assertEqual(rec.r_accepts, (False, False))
        self.assertEqual(rec.r2w_after, ((0, 0), (0, 0)))
        self.assertEqual(rec.w2r_after, ((0, 0), (0, 0)))

    def test_common_reset_clears_both_domains(self):
        fifo = DualAsyncFIFO(k=2)
        fifo.run_one(DualEvent(1, True, False, False, wen=True, wdata="x"))
        fifo.reset()
        self.assertEqual((fifo.wptr, fifo.rptrs, fifo.full, fifo.empties),
                         (0, (0, 0), False, (True, True)))
        self.assertEqual(fifo.steps, ())


class TestDualContracts(unittest.TestCase):
    def test_all_three_clocks_stopped_rejected(self):
        with self.assertRaisesRegex(AsyncFIFOError, "all False"):
            DualEvent(1, False, False, False)

    def test_wrong_event_type_and_time_are_locatable(self):
        fifo = DualAsyncFIFO(k=2)
        with self.assertRaises(AsyncFIFOError):
            fifo.run_one("bad")
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[1\]"):
            fifo.run([
                DualEvent(1, True, False, False),
                "bad",
            ])
        fifo.run_one(DualEvent(2, True, False, False))
        with self.assertRaisesRegex(AsyncFIFOError, "strictly greater"):
            fifo.run_one(DualEvent(2, False, True, False))

    def test_core_does_not_use_plain_pointer_min(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "afifo", "model.py"),
                  encoding="utf-8") as source_file:
            source = source_file.read()
        self.assertIn("forward_distance", source)
        self.assertNotIn("min(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
