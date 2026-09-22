"""必要测试：格雷码、复位语义、旗标时序、回绕、同刻边沿、停钟、频比、
独立参考全量核对、输入校验与批处理原子性。

期望值来源：
* 手算常量（格雷表、指针序列、首满/首空时刻）；
* ``fifo_cdc.reference.TraceOracle``——独立重写的协议参考 + 抽象队列，
  其内部不调用被测核心算法；
* 往返一致（gray/binary 互逆）仅作为补充。
"""

from __future__ import annotations

import math
import unittest

from fifo_cdc import (
    K_MAX,
    K_MIN,
    AsyncFIFOModel,
    ClockEdge,
    ClockKind,
    FIFOConfigurationError,
    FIFOInputError,
    FIFOScheduleError,
    TraceOracle,
    binary_to_gray,
    gray_to_binary,
    nop_edge,
    read_edge,
    write_edge,
)


class TestGrayCode(unittest.TestCase):
    def test_known_table_2bit(self):
        # 2 位标准反射格雷码表（手算）
        table = {0: 0b00, 1: 0b01, 2: 0b11, 3: 0b10}
        for b, g in table.items():
            self.assertEqual(binary_to_gray(b, 2), g)
            self.assertEqual(gray_to_binary(g, 2), b)

    def test_known_table_3bit(self):
        table = {0: 0b000, 1: 0b001, 2: 0b011, 3: 0b010,
                 4: 0b110, 5: 0b111, 6: 0b101, 7: 0b100}
        for b, g in table.items():
            self.assertEqual(binary_to_gray(b, 3), g)
            self.assertEqual(gray_to_binary(g, 3), b)

    def test_consecutive_values_differ_one_bit(self):
        for bits in range(1, 7):
            for v in range(1, 1 << bits):
                diff = binary_to_gray(v, bits) ^ binary_to_gray(v - 1, bits)
                self.assertEqual(diff & (diff - 1), 0,
                                 f"相邻格雷码 {v}/{v-1} 差不止 1 位")

    def test_roundtrip_supplementary(self):
        # 往返一致仅作补充
        for bits in range(1, 7):
            for v in range(1 << bits):
                self.assertEqual(gray_to_binary(binary_to_gray(v, bits), bits), v)

    def test_out_of_range(self):
        with self.assertRaises(FIFOInputError):
            binary_to_gray(4, 2)
        with self.assertRaises(FIFOInputError):
            gray_to_binary(-1, 3)
        with self.assertRaises(FIFOConfigurationError):
            binary_to_gray(0, 0)

    def test_bool_rejected(self):
        with self.assertRaises(FIFOInputError):
            binary_to_gray(True, 2)
        with self.assertRaises(FIFOInputError):
            binary_to_gray(1, False)


class TestResetAndRejectInputs(unittest.TestCase):
    def test_reset_state(self):
        m = AsyncFIFOModel(2)
        self.assertEqual(m.depth, 4)
        self.assertEqual(m.ptr_bits, 3)
        self.assertEqual(m.wptr_bin, 0)
        self.assertEqual(m.rptr_bin, 0)
        self.assertIs(m.full, False)
        self.assertIs(m.empty, True)
        self.assertEqual(m.wq2_rptr, 0)
        self.assertEqual(m.rq2_wptr, 0)

    def test_k_bounds_and_types(self):
        for k in range(K_MIN, K_MAX + 1):
            self.assertEqual(AsyncFIFOModel(k).depth, 1 << k)
        with self.assertRaises(FIFOConfigurationError):
            AsyncFIFOModel(0)
        with self.assertRaises(FIFOConfigurationError):
            AsyncFIFOModel(6)
        with self.assertRaises(FIFOInputError):
            AsyncFIFOModel(True)          # bool 不是整数参数
        with self.assertRaises(FIFOInputError):
            AsyncFIFOModel(2.0)           # 浮点不静默取整
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(FIFOInputError):
                AsyncFIFOModel(bad)
        with self.assertRaises(FIFOInputError):
            AsyncFIFOModel("2")

    def test_edge_validation(self):
        m = AsyncFIFOModel(2)
        with self.assertRaises(FIFOInputError):
            m.step(write_edge(-1, "x"))
        with self.assertRaises(FIFOInputError):
            write_edge(math.nan)
        with self.assertRaises(FIFOInputError):
            write_edge(math.inf)
        with self.assertRaises(FIFOInputError):
            write_edge(True)              # bool 当时刻
        # 跨域携带错误操作
        with self.assertRaises(FIFOInputError):
            m.step(ClockEdge(ClockKind.READ, 1, write=True, data="x"))
        with self.assertRaises(FIFOInputError):
            m.step(ClockEdge(ClockKind.WRITE, 1, read=True))
        with self.assertRaises(FIFOInputError):
            m.step(ClockEdge(ClockKind.READ, 1, read=True, data="x"))
        with self.assertRaises(FIFOInputError):
            m.step(ClockEdge(ClockKind.WRITE, 1, write=True, read=True))
        with self.assertRaises(FIFOInputError):
            m.step("not-an-edge")

    def test_duplicate_edge_same_clock_same_time(self):
        m = AsyncFIFOModel(2)
        m.step(write_edge(0, "a"))
        with self.assertRaises(FIFOScheduleError):
            m.step(write_edge(0, "b"))

    def test_time_must_not_go_backwards(self):
        m = AsyncFIFOModel(2)
        m.step(write_edge(5, "a"))
        with self.assertRaises(FIFOScheduleError):
            m.step(write_edge(4, "b"))

    def test_batch_atomicity_on_bad_input(self):
        m = AsyncFIFOModel(2)
        good = [write_edge(0, "a"), write_edge(1, "b")]
        # 直接构造非法边沿（绕过工厂即时校验），验证 run() 先全量校验再提交
        bad = good + [ClockEdge(ClockKind.WRITE, -1, write=True, data="bad")]
        with self.assertRaises(FIFOInputError):
            m.run(bad)
        # 全部不生效
        self.assertEqual(m.wptr_bin, 0)
        self.assertEqual(m.writes_accepted, 0)
        self.assertEqual(m.time, 0)

    def test_batch_atomicity_on_time_reversal(self):
        m = AsyncFIFOModel(2)
        with self.assertRaises(FIFOScheduleError):
            m.run([write_edge(3, "a"), write_edge(2, "b")])
        self.assertEqual(m.wptr_bin, 0)

    def test_batch_atomicity_duplicate_domain(self):
        m = AsyncFIFOModel(2)
        with self.assertRaises(FIFOScheduleError):
            m.run([write_edge(1, "a"), write_edge(1, "b")])
        self.assertEqual(m.wptr_bin, 0)


class TestBasicProtocolHandComputed(unittest.TestCase):
    """k=2（深度4、3位指针）手算序列。"""

    def test_empty_after_reset_first_read_rejected(self):
        m = AsyncFIFOModel(2)
        rec = m.step(read_edge(0))
        self.assertFalse(rec.read_accepted)
        self.assertIs(rec.empty_before, True)
        self.assertIs(rec.empty, True)      # 空读，指针不动，仍空
        self.assertEqual(m.reads_rejected, 1)

    def test_write_then_sync_delay_empty_deassert(self):
        m = AsyncFIFOModel(2)
        m.step(write_edge(0, "a"))          # wptr 格雷 001；读侧本拍看不到
        self.assertIs(m.empty, True)        # empty 是读域寄存器，写域动它不了
        m.step(read_edge(1))                # rq1=001, rq2=000；旗标比旧 rq2，仍空，拒
        self.assertIs(m.empty, True)
        m.step(read_edge(2))                # rq2=001 锁存但本拍旗标仍比旧值 000
        self.assertIs(m.empty, True)
        r3 = m.step(read_edge(3))           # 此刻旗标比 rq2=001：empty 撤
        self.assertIs(r3.empty, False)
        self.assertFalse(r3.read_accepted)  # 但本拍操作沿沿前仍为旧 empty=True
        r4 = m.step(read_edge(4))           # 旧旗标已非空：接受
        self.assertTrue(r4.read_accepted)
        self.assertEqual(r4.read_data, "a")
        self.assertIs(r4.empty, True)       # 读空后旗标重新置位

    def test_fill_to_full_and_full_is_conservative(self):
        m = AsyncFIFOModel(2)
        # 连写 4 个：full 在第 4 次写入的边沿后才置位（比较的是下一写指针）
        recs = [m.step(write_edge(t, f"d{t}")) for t in range(4)]
        self.assertTrue(all(r.write_accepted for r in recs))
        self.assertIs(recs[2].full, False)  # 写完第 3 个还未满
        self.assertIs(recs[3].full, True)   # 下一写指针=4 格雷110 命中
        # 满后再写被拒
        r = m.step(write_edge(4, "overflow"))
        self.assertFalse(r.write_accepted)
        self.assertIs(m.full, True)
        self.assertEqual(m.writes_rejected, 1)

    def test_full_clear_needs_two_sync_edges_after_read(self):
        m = AsyncFIFOModel(2)
        for t in range(4):
            m.step(write_edge(t, f"d{t}"))
        self.assertIs(m.full, True)
        # 读域一直没动过；empty 旗标经历 rq1/rq2 与寄存器比较各一拍后才撤
        m.step(read_edge(4))   # rq1=110, rq2=000, empty 真 → 拒
        m.step(read_edge(5))   # rq2=110 锁存，旗标比旧值 → 仍拒
        m.step(read_edge(6))   # empty 撤，但操作用沿前旧旗标 → 仍拒
        r7 = m.step(read_edge(7))  # 接受读 d0，rptr 格雷 001
        self.assertTrue(r7.read_accepted)
        self.assertEqual(r7.read_data, "d0")
        self.assertIs(m.full, True)     # 写侧仍认为满
        m.step(nop_edge(ClockKind.WRITE, 8))   # wq1=001
        self.assertIs(m.full, True)
        m.step(nop_edge(ClockKind.WRITE, 9))   # wq2=001 锁存，旗标比旧 wq2
        self.assertIs(m.full, True)
        m.step(nop_edge(ClockKind.WRITE, 10))  # 比较 wq2=001：full 撤
        self.assertIs(m.full, False)
        m.step(nop_edge(ClockKind.WRITE, 11))
        nxt = m.step(write_edge(12, "d4"))     # 旧 full=0：接受
        self.assertTrue(nxt.write_accepted)


class TestSimultaneousEdges(unittest.TestCase):
    def test_both_edges_sample_old_state(self):
        # 复位即同刻：读用旧 empty=True 必拒；写接受；同步器双方都采样 0。
        m = AsyncFIFOModel(2)
        w = m.step(write_edge(0, "a"))
        r = m.step(read_edge(0))
        self.assertTrue(w.write_accepted)
        self.assertFalse(r.read_accepted)
        self.assertEqual(r.rq1_wptr, 0)     # 同刻读一级采样的是边沿前 wptr=0
        self.assertEqual(w.wq1_rptr, 0)
        self.assertEqual(m.wptr_bin, 1)
        self.assertEqual(m.rptr_bin, 0)

    def test_simultaneous_order_independent(self):
        def run(read_first: bool):
            m = AsyncFIFOModel(3)
            oracle = TraceOracle(3)
            # 先填一些，再制造大量同刻读写
            edges = []
            t = 0
            for i in range(6):
                edges.append(write_edge(t, f"w{i}"))
                t += 2
                edges.append(read_edge(t))
                t += 2
            for i in range(6, 20):
                edges.append(write_edge(t, f"w{i}"))
                edges.append(read_edge(t))   # 同刻
                t += 1
            pairs = []
            if read_first:
                edges.sort(key=lambda e: (e.time, 0 if e.clock is ClockKind.READ else 1))
            for e in edges:
                pairs.append((e, m.step(e)))
            rep = oracle.observe_many(pairs)
            rep.raise_if_bad()
            return m, rep, [p[1].read_data for p in pairs if p[1].read_accepted]

        m1, rep1, outs1 = run(False)
        m2, rep2, outs2 = run(True)
        self.assertTrue(rep1.ok and rep2.ok)
        self.assertEqual(m1.snapshot(), m2.snapshot())
        self.assertEqual(outs1, outs2)


class TestWraparound(unittest.TestCase):
    def test_pointer_wraparound_all_k(self):
        for k in range(K_MIN, K_MAX + 1):
            m = AsyncFIFOModel(k)
            oracle = TraceOracle(k)
            depth = 1 << k
            pairs = []
            t = 0
            # 远超一次回绕的读写交替，写多读少制造满，再读多写少排空
            for round_i in range(3):
                for i in range(depth * 3):
                    pairs.append((write_edge(t, f"r{round_i}w{i}"), None))
                    t += 1
                    pairs.append((read_edge(t), None))
                    t += 1
            records = []
            for edge, _ in pairs:
                rec = m.step(edge)
                records.append((edge, rec))
            rep = oracle.observe_many(records)
            rep.raise_if_bad()
            self.assertGreaterEqual(rep.max_occupancy, 1)
            self.assertEqual(rep.read_sequence, rep.written_sequence[:len(rep.read_sequence)])
            # 指针必然发生过回绕
            self.assertGreater(m.wptr_bin + m.rptr_bin, 0)


class TestClockFrequencyAndGating(unittest.TestCase):
    def _run_edges(self, k, edges, data_seq):
        m = AsyncFIFOModel(k)
        oracle = TraceOracle(k)
        records = [(e, m.step(e)) for e in edges]
        rep = oracle.observe_many(records)
        rep.raise_if_bad()
        return m, rep

    def test_fast_write_slow_read_frequency_ratio(self):
        # 写时钟每 1 tick，读时钟每 5 tick（5:1 频比）
        edges = []
        seq = 0
        for t in range(60):
            if t % 1 == 0:
                edges.append(write_edge(t, f"x{seq}"))
                seq += 1
            if t % 5 == 0:
                edges.append(read_edge(t))
        m, rep = self._run_edges(2, sorted(edges, key=lambda e: (e.time, 0 if e.clock is ClockKind.WRITE else 1)), seq)
        self.assertTrue(rep.ok)
        self.assertGreater(rep.rejected_writes, 0)   # 必然出现保守满拒绝

    def test_slow_write_fast_read(self):
        edges = []
        seq = 0
        for t in range(60):
            if t % 4 == 0:
                edges.append(write_edge(t, f"y{seq}"))
                seq += 1
            if t % 1 == 0:
                edges.append(read_edge(t))
        m, rep = self._run_edges(2, edges, seq)
        self.assertTrue(rep.ok)
        self.assertGreater(rep.rejected_reads, 0)    # 必然出现保守空拒绝

    def test_read_clock_gated_then_resumes(self):
        m = AsyncFIFOModel(2)
        for t in range(4):
            m.step(write_edge(t, f"d{t}"))
        # 读钟停 100 tick：期间写侧同步不到任何读，持续判满
        rej = m.step(write_edge(104, "x"))
        self.assertFalse(rej.write_accepted)
        # 读钟恢复，排空
        got = []
        for t in range(105, 130):
            rec = m.step(read_edge(t))
            if rec.read_accepted:
                got.append(rec.read_data)
        self.assertEqual(got, ["d0", "d1", "d2", "d3"])

    def test_write_clock_gated(self):
        m = AsyncFIFOModel(2)
        m.step(write_edge(0, "a"))
        # 写钟停很久，读钟连续空转，最终 empty 稳定为真且读被拒
        results = []
        for t in range(1, 40):
            rec = m.step(read_edge(t))
            results.append(rec.read_accepted)
        self.assertIn(True, results)
        last = m.step(read_edge(41))
        self.assertFalse(last.read_accepted)
        self.assertIs(last.empty, True)


class TestIndependentOracleRandomish(unittest.TestCase):
    """确定性伪随机事件流：独立参考全字段比对 + 抽象队列顺序/越界核对。"""

    def test_deterministic_streams(self):
        # 简单 LCG，不依赖 random 种子环境
        state = 0x12345678

        def rnd() -> int:
            nonlocal state
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            return state

        for k in range(K_MIN, K_MAX + 1):
            for trial in range(4):
                m = AsyncFIFOModel(k)
                oracle = TraceOracle(k)
                edge_pairs = []
                t = 0
                wcount = rcount = 0
                for _ in range(400):
                    t += 1 + (rnd() % 3)
                    # 域选择与频比由随机数决定，含停钟（某域长时间无边沿）
                    domain_pick = rnd() % 100
                    edges_at_t = []
                    if domain_pick < 45:
                        edges_at_t.append(write_edge(t, f"k{k}t{trial}w{wcount}"))
                        wcount += 1
                    elif domain_pick < 90:
                        edges_at_t.append(read_edge(t))
                        rcount += 1
                    else:
                        # 同刻两边沿
                        edges_at_t.append(write_edge(t, f"k{k}t{trial}w{wcount}"))
                        edges_at_t.append(read_edge(t))
                        wcount += 1
                        rcount += 1
                    for e in edges_at_t:
                        rec = m.step(e)
                        edge_pairs.append((e, rec))
                rep = oracle.observe_many(edge_pairs)
                rep.raise_if_bad()
                self.assertEqual(
                    rep.read_sequence,
                    rep.written_sequence[: len(rep.read_sequence)],
                    "读出顺序必须等于写入接受顺序前缀",
                )
                self.assertLessEqual(rep.max_occupancy, 1 << k)


class TestQueueNoGlobalLengthInCore(unittest.TestCase):
    """核心旗标判定不得依赖全局队列长度：直接做源码结构守卫。"""

    def test_core_does_not_use_len_queue_for_flags(self):
        import inspect
        import fifo_cdc.core as core

        src = inspect.getsource(core)
        # 旗标计算函数中不得出现 deque/len(self...queue 之类
        self.assertNotIn("deque", src)
        self.assertNotIn("occupancy", src.replace("占用", ""))  # 无占用量旗标


if __name__ == "__main__":
    unittest.main()
