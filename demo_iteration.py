#!/usr/bin/env python3
"""Offline deterministic demo for the dual-reader asynchronous FIFO."""

from collections import deque

from afifo import DualAsyncFIFO, DualEvent


# (time, wclk, rclk0, rclk1, wen, ren0, ren1, data)
SCHEDULE = [
    (1, 1, 0, 0, 1, 0, 0, "A"),
    (2, 1, 0, 0, 1, 0, 0, "B"),
    (3, 1, 0, 0, 1, 0, 0, "C"),
    (4, 1, 0, 0, 1, 0, 0, "D"),
    *[(t, 0, 1, 0, 0, 1, 0, None) for t in range(5, 11)],
    *[(t, 1, 0, 0, 1, 0, 0, "STOPPED_READER_REJECTED")
      for t in range(11, 15)],
    *[(t, 0, 0, 1, 0, 0, 1, None) for t in range(15, 21)],
    (21, 1, 0, 0, 1, 0, 0, "SYNC_REJECTED_1"),
    (22, 1, 0, 0, 1, 0, 0, "SYNC_REJECTED_2"),
    (23, 1, 0, 0, 1, 0, 0, "E"),
    (24, 1, 0, 0, 1, 0, 0, "F"),
    (25, 1, 0, 0, 1, 0, 0, "G"),
    (26, 1, 0, 0, 1, 0, 0, "H"),
    (27, 1, 0, 0, 1, 0, 0, "FULL_REJECTED_1"),
    (28, 1, 0, 0, 1, 0, 0, "FULL_REJECTED_2"),
    *[(t, 0, 1, 0, 0, 1, 0, None) for t in range(29, 36)],
    *[(t, 0, 0, 1, 0, 0, 1, None) for t in range(36, 44)],
    (44, 1, 0, 0, 1, 0, 0, "SYNC_REJECTED_3"),
    (45, 1, 0, 0, 1, 0, 0, "SYNC_REJECTED_4"),
    (46, 1, 0, 0, 1, 0, 0, "I"),
    (47, 1, 0, 0, 1, 0, 0, "J"),
    (48, 1, 0, 0, 1, 0, 0, "K"),
    (49, 1, 0, 0, 1, 0, 0, "L"),
    (50, 1, 0, 0, 1, 0, 0, "FULL_REJECTED_3"),
    (51, 1, 0, 0, 1, 0, 0, "FULL_REJECTED_4"),
    *[(t, 0, 1, 0, 0, 1, 0, None) for t in range(52, 59)],
    *[(t, 0, 0, 1, 0, 0, 1, None) for t in range(59, 66)],
]


def audit_and_run():
    fifo = DualAsyncFIFO(k=2)
    logical_queues = [deque(), deque()]
    slot_payload = [None] * fifo.depth
    consumed = [set(), set()]
    records = []

    for row in SCHEDULE:
        t, wclk, rclk0, rclk1, wen, ren0, ren1, data = row
        rec = fifo.run_one(DualEvent(
            t, bool(wclk), bool(rclk0), bool(rclk1),
            bool(wen), bool(ren0), bool(ren1), data,
        ))

        if rec.w_accept:
            addr = rec.wptr_before & (fifo.depth - 1)
            previous = slot_payload[addr]
            if previous is not None:
                assert previous in consumed[0], f"{previous} unread by R0"
                assert previous in consumed[1], f"{previous} unread by R1"
            slot_payload[addr] = data
            logical_queues[0].append(data)
            logical_queues[1].append(data)

        for reader in range(2):
            if rec.r_accepts[reader]:
                payload = logical_queues[reader].popleft()
                assert payload == rec.rdatas[reader]
                consumed[reader].add(payload)

        records.append(rec)

    assert not logical_queues[0] and not logical_queues[1]
    assert fifo.rptrs == (4, 4) and fifo.wptr == 4
    return fifo, records


def print_record(rec):
    clocks = ("W" if rec.wclk else ".")
    clocks += "0" if rec.rclks[0] else "."
    clocks += "1" if rec.rclks[1] else "."
    w = {True: "A", False: "x", None: "-"}[rec.w_accept]
    r0 = {True: "A", False: "x", None: "-"}[rec.r_accepts[0]]
    r1 = {True: "A", False: "x", None: "-"}[rec.r_accepts[1]]
    outputs = [rec.rdatas[i] if rec.r_accepts[i] else "-" for i in range(2)]
    print(
        f"t={rec.time:2d} {clocks} W:{w} R0:{r0} R1:{r1} "
        f"wptr={rec.wptr_after} rptrs={rec.rptrs_after} F={int(rec.full_after)} "
        f"E={tuple(int(x) for x in rec.empties_after)} "
        f"release={rec.release_ptr_after} syncRelease={rec.synced_release_ptr_after} "
        f"in={rec.wdata if rec.w_accept else '-'} out={outputs}"
    )


def main():
    print("双读端异步 FIFO：深度4，共同复位，无运行中取消订阅；固定调度实时计算")
    print("-" * 112)
    fifo, records = audit_and_run()
    for rec in records:
        print_record(rec)

    rejected_while_stopped = [
        rec.time for rec in records
        if rec.wdata == "STOPPED_READER_REJECTED" and rec.w_accept is False
    ]
    print("-" * 112)
    print(f"读端0先读完 A-D；读端1停钟期间，t={rejected_while_stopped} 的写请求全部背压拒绝")
    print("随后读端1独立读完 A-D；写端通过各自两级同步指针后才回收槽位")
    print("E-H 写入时写端在 t=26 从指针7回绕到0；两个读端读完后再写入 I-L 并再次复用槽位")
    print(f"接受写={fifo.accepted_writes}，两个读端接受数={fifo.accepted_reads}")
    print("两条独立 deque 逻辑队列、每次覆盖前的槽位消费集合、E-L 回绕顺序均已断言通过")


if __name__ == "__main__":
    main()
