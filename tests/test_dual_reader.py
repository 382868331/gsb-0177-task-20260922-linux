"""Tests for one writer and two independently clocked readers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_model import RefDualReaderFIFO, make_dual_schedule

from afifo import (
    DualReaderAsyncFIFO,
    DualReaderEvent,
    gray_to_binary,
    forward_distance,
)
from afifo.model import AsyncFIFOError


def ev(row):
    t, wclk, r0clk, r1clk, wen, r0en, r1en, wdata = row
    return DualReaderEvent(
        t, bool(wclk), bool(r0clk), bool(r1clk),
        bool(wen), bool(r0en), bool(r1en), wdata,
    )


class TestGenerationDistance(unittest.TestCase):
    def test_wrapped_pointer_distance_never_uses_plain_min(self):
        self.assertEqual(forward_distance(0, 3, 4), 1)
        self.assertEqual(forward_distance(1, 3, 4), 2)
        self.assertEqual(forward_distance(3, 0, 4), 3)
        self.assertEqual(forward_distance(0, 0, 8), 0)

    def test_contract(self):
        with self.assertRaises(TypeError):
            forward_distance(True, 0, 4)
        with self.assertRaises(ValueError):
            forward_distance(-1, 0, 4)
        with self.assertRaises(ValueError):
            forward_distance(0, 0, 0)


class TestSlowReaderRetention(unittest.TestCase):
    def test_stopped_reader_backpressures_writer_while_other_continues(self):
        fifo = DualReaderAsyncFIFO(k=2)

        for t in range(1, 5):
            rec = fifo.run_one(DualReaderEvent(
                t, True, False, False, wen=True, wdata=f"D{t}"
            ))
            self.assertTrue(rec.w_accept)
        self.assertTrue(fifo.full)

        rejected = []
        for t in range(5, 13):
            wclk = t in (8, 12)
            rec = fifo.run_one(DualReaderEvent(
                t, wclk, False, True,
                wen=wclk,
                r1en=True,
                wdata=f"W{t}" if wclk else None,
            ))
            if wclk:
                rejected.append(rec.w_accept)
        self.assertEqual(rejected, [False, False])
        self.assertEqual(fifo.accepted_reads, (0, 4))
        self.assertEqual(fifo.r1ptr, 4)
        self.assertTrue(fifo.full)
        self.assertEqual(fifo.wptr, 4)

        for t in (13, 14, 15):
            fifo.run_one(DualReaderEvent(t, False, True, False))
        rec = fifo.run_one(DualReaderEvent(16, False, True, False, r0en=True))
        self.assertTrue(rec.r0_accept)
        self.assertEqual(rec.r0data, "D1")
        self.assertEqual(fifo.r0ptr, 1)

        for t in (17, 18, 19):
            rec = fifo.run_one(DualReaderEvent(t, True, False, False))
        self.assertFalse(rec.full_after)
        rec = fifo.run_one(DualReaderEvent(
            20, True, False, False, wen=True, wdata="D5"
        ))
        self.assertTrue(rec.w_accept)
        self.assertEqual(rec.wptr_after, 5)
        self.assertTrue(rec.full_after)

    def test_slot_is_released_only_after_both_readers_consume_it(self):
        fifo = DualReaderAsyncFIFO(k=1)
        fifo.run([
            DualReaderEvent(1, True, False, False, wen=True, wdata="A"),
            DualReaderEvent(2, True, False, False, wen=True, wdata="B"),
        ])
        self.assertTrue(fifo.full)
        blocked = fifo.run_one(DualReaderEvent(
            3, True, False, False, wen=True, wdata="C"
        ))
        self.assertFalse(blocked.w_accept)

        for t in (4, 5, 6):
            fifo.run_one(DualReaderEvent(t, False, True, False))
        r0_a = fifo.run_one(DualReaderEvent(7, False, True, False, r0en=True))
        r0_b = fifo.run_one(DualReaderEvent(8, False, True, False, r0en=True))
        self.assertEqual((r0_a.r0data, r0_b.r0data), ("A", "B"))

        for t in (9, 10, 11):
            fifo.run_one(DualReaderEvent(t, False, False, True))
        r1_a = fifo.run_one(DualReaderEvent(12, False, False, True, r1en=True))
        r1_b = fifo.run_one(DualReaderEvent(13, False, False, True, r1en=True))
        self.assertEqual((r1_a.r1data, r1_b.r1data), ("A", "B"))

        still_blocked = fifo.run_one(DualReaderEvent(
            14, True, False, False, wen=True, wdata="C"
        ))
        self.assertFalse(still_blocked.w_accept)
        for t in (15, 16):
            fifo.run_one(DualReaderEvent(t, True, False, False))
        released = fifo.run_one(DualReaderEvent(
            17, True, False, False, wen=True, wdata="C"
        ))
        self.assertTrue(released.w_accept)
        self.assertEqual((released.wptr_before, released.wptr_after), (2, 3))


class TestIndependentQueues(unittest.TestCase):
    def test_every_payload_is_consumed_once_by_each_reader(self):
        schedule = make_dual_schedule(
            900,
            w_period=2,
            r0_period=3,
            r1_period=5,
            seed=808,
            r1_stop_windows=((250, 430),),
        )
        dut = DualReaderAsyncFIFO(k=3)
        ref = RefDualReaderFIFO(k=3)
        written = []
        outputs = ([], [])

        for row in schedule:
            record = dut.run_one(ev(row))
            expected = ref.step(*row[1:])
            t = row[0]
            with self.subTest(time=t):
                if record.w_accept:
                    modulus = 1 << (dut.k + 1)
                    depth = 1 << dut.k
                    safe_d0 = forward_distance(
                        record.wptr_before,
                        gray_to_binary(record.r02w_before[1]),
                        modulus,
                    )
                    safe_d1 = forward_distance(
                        record.wptr_before,
                        gray_to_binary(record.r12w_before[1]),
                        modulus,
                    )
                    self.assertLess(safe_d0, depth)
                    self.assertLess(safe_d1, depth)
                self.assertEqual(record.w_accept, expected["w_accept"])
                self.assertEqual(record.r0_accept, expected["r_accept"][0])
                self.assertEqual(record.r1_accept, expected["r_accept"][1])
                self.assertEqual(record.wptr_after, expected["wptr"])
                self.assertEqual(record.r0ptr_after, expected["rptrs"][0])
                self.assertEqual(record.r1ptr_after, expected["rptrs"][1])
                self.assertIs(record.full_after, expected["full"])
                self.assertEqual(record.empty0_after, expected["empties"][0])
                self.assertEqual(record.empty1_after, expected["empties"][1])
                self.assertEqual(record.r02w_after, expected["r2w"][0])
                self.assertEqual(record.r12w_after, expected["r2w"][1])
                self.assertEqual(record.w2r0_after, expected["w2r"][0])
                self.assertEqual(record.w2r1_after, expected["w2r"][1])
                self.assertEqual(record.r0data, expected["rdata"][0])
                self.assertEqual(record.r1data, expected["rdata"][1])
            if record.w_accept:
                written.append(row[7])
            for i in range(2):
                if (record.r0_accept, record.r1_accept)[i]:
                    outputs[i].append((record.r0data, record.r1data)[i])

        self.assertEqual(len(written), ref.n_w)
        self.assertEqual(dut.accepted_reads, (ref.n_r[0], ref.n_r[1]))
        self.assertEqual(outputs[0], written[:ref.n_r[0]])
        self.assertEqual(outputs[1], written[:ref.n_r[1]])
        self.assertEqual(list(ref.queues[0]), written[ref.n_r[0]:])
        self.assertEqual(list(ref.queues[1]), written[ref.n_r[1]:])

    def test_arbitrary_clock_ratios_wraps_and_common_edges(self):
        ratios = ((1, 1, 1), (2, 3, 5), (5, 3, 2), (3, 4, 6), (7, 2, 3))
        for k in (1, 2, 3, 4, 5):
            for index, (wp, r0p, r1p) in enumerate(ratios):
                with self.subTest(k=k, ratio=(wp, r0p, r1p)):
                    schedule = make_dual_schedule(
                        500 if k > 1 else 250,
                        w_period=wp,
                        r0_period=r0p,
                        r1_period=r1p,
                        seed=k * 100 + index,
                    )
                    dut = DualReaderAsyncFIFO(k)
                    ref = RefDualReaderFIFO(k)
                    for row in schedule:
                        record = dut.run_one(ev(row))
                        expected = ref.step(*row[1:])
                        self.assertEqual(record.full_after, expected["full"])
                        self.assertEqual(record.r0ptr_after,
                                         expected["rptrs"][0])
                        self.assertEqual(record.r1ptr_after,
                                         expected["rptrs"][1])
                        self.assertTrue(
                            all(0 <= n <= (1 << k)
                                for n in expected["outstanding"])
                        )

    def test_coincident_edges_sample_old_state(self):
        fifo = DualReaderAsyncFIFO(k=2)
        rec = fifo.run_one(DualReaderEvent(
            1, True, True, True, wen=True, r0en=True, r1en=True, wdata=9
        ))
        self.assertTrue(rec.w_accept)
        self.assertFalse(rec.r0_accept)
        self.assertFalse(rec.r1_accept)
        self.assertEqual(rec.r02w_after, (0, 0))
        self.assertEqual(rec.r12w_after, (0, 0))
        self.assertEqual(rec.w2r0_after, (0, 0))
        self.assertEqual(rec.w2r1_after, (0, 0))


class TestDualReaderContracts(unittest.TestCase):
    def test_common_reset(self):
        fifo = DualReaderAsyncFIFO(k=2)
        fifo.run_one(DualReaderEvent(1, True, False, False, wen=True, wdata=1))
        fifo.reset()
        self.assertEqual(fifo.snapshot(), {
            "wptr": 0,
            "r0ptr": 0,
            "r1ptr": 0,
            "wgray": 0,
            "r0gray": 0,
            "r1gray": 0,
            "full": False,
            "empty0": True,
            "empty1": True,
            "r02w": (0, 0),
            "r12w": (0, 0),
            "w2r0": (0, 0),
            "w2r1": (0, 0),
            "last_time": -1,
        })
        self.assertEqual(fifo.steps, ())
        self.assertEqual(fifo.accepted_reads, (0, 0))

    def test_invalid_events_are_locatable_and_batch_is_atomic(self):
        fifo = DualReaderAsyncFIFO(k=2)
        with self.assertRaisesRegex(AsyncFIFOError, "r0clk"):
            DualReaderEvent(1, True, 1, False)
        with self.assertRaisesRegex(AsyncFIFOError, "all False"):
            DualReaderEvent(1, False, False, False)
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[1\]"):
            fifo.run([
                DualReaderEvent(1, True, False, False),
                "bad",
            ])
        before = fifo.snapshot()
        with self.assertRaisesRegex(AsyncFIFOError, r"events\[2\]"):
            fifo.run([
                DualReaderEvent(1, True, False, False, wen=True, wdata="a"),
                DualReaderEvent(2, True, False, False, wen=True, wdata="b"),
                DualReaderEvent(2, False, True, False),
            ])
        self.assertEqual(fifo.snapshot(), before)
        self.assertEqual(fifo.steps, ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
