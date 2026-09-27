#!/usr/bin/env python3
"""Offline one-writer/two-independent-reader FIFO demonstration.

The schedule is fixed in this file and calculated by the real model.  It uses
no network, credentials, randomness, sleep, or pre-recorded result.
"""

from collections import deque

from afifo import DualReaderAsyncFIFO, DualReaderEvent, forward_distance


def build_events():
    events = []
    for t in range(1, 5):
        events.append(DualReaderEvent(
            t, True, False, False, wen=True, wdata=f"W{t:02d}"
        ))

    # Reader 0 is stopped.  Reader 1 drains all four entries, while writer
    # attempts at t=8 and t=12 remain rejected because storage is retained.
    for t in range(5, 13):
        wclk = t in (8, 12)
        events.append(DualReaderEvent(
            t, wclk, False, True,
            wen=wclk,
            r1en=True,
            wdata=f"W{t:02d}" if wclk else None,
        ))

    # Wake reader 0.  The first three edges synchronize the write Gray value;
    # the request on the fourth edge is the first accepted read.  Write edges
    # let the write domain independently observe reader 0's progress.
    for t in range(13, 20):
        events.append(DualReaderEvent(
            t, t in (17, 18, 19), True, False, r0en=True
        ))

    # Both readers are active at independent phase offsets.  Reader 0 is
    # deliberately slower, so it remains the slowest storage-release pointer.
    for t in range(20, 41):
        wclk = t % 3 == 0
        r0clk = t % 4 == 0
        r1clk = t % 2 == 0
        if not (wclk or r0clk or r1clk):
            continue
        events.append(DualReaderEvent(
            t, wclk, r0clk, r1clk,
            wen=wclk,
            r0en=r0clk,
            r1en=r1clk,
            wdata=f"W{t:02d}" if wclk else None,
        ))

    # A final common-clock burst wraps all k+1=3 bit pointers many times.
    for t in range(41, 83):
        events.append(DualReaderEvent(
            t, True, True, True, wen=True, r0en=True, r1en=True,
            wdata=f"W{t:02d}",
        ))
    for t in range(83, 85):
        events.append(DualReaderEvent(
            t, True, True, True, wen=False, r0en=True, r1en=True
        ))
    return events


def audit(records, depth, modulus):
    queues = (deque(), deque())
    outputs = ([], [])
    last_wptr = 0
    for rec in records:
        if rec.w_accept:
            assert all(len(q) < depth for q in queues)
            queues[0].append(rec.wdata)
            queues[1].append(rec.wdata)
            # Verify the slot immediately before the new write pointer was
            # released by both synchronized readers.
            assert (
                forward_distance(rec.wptr_after, rec.r0ptr_after, modulus)
                <= depth
            )
            assert (
                forward_distance(rec.wptr_after, rec.r1ptr_after, modulus)
                <= depth
            )
            last_wptr = rec.wptr_after
        for i, accept in enumerate((rec.r0_accept, rec.r1_accept)):
            if accept:
                outputs[i].append(queues[i].popleft())
    return queues, outputs, last_wptr


def print_rows(records, times, fifo):
    print("t   edge   W/R0/R1 accept   wptr r0ptr r1ptr  full empty  outputs")
    selected = {rec.time: rec for rec in records}
    for t in times:
        rec = selected[t]
        edge = f"{'W' if rec.wclk else '.'}{'0' if rec.r0clk else '.'}" \
               f"{'1' if rec.r1clk else '.'}"
        accept = (
            f"{'-' if rec.w_accept is None else int(rec.w_accept)}/"
            f"{'-' if rec.r0_accept is None else int(rec.r0_accept)}/"
            f"{'-' if rec.r1_accept is None else int(rec.r1_accept)}"
        )
        outs = ",".join(
            f"{i}:{data}"
            for i, data in enumerate((rec.r0data, rec.r1data))
            if data is not None
        )
        print(
            f"{rec.time:>2}  {edge:<4} {accept:<14} "
            f"{rec.wptr_after:^4} {rec.r0ptr_after:^5} {rec.r1ptr_after:^5}  "
            f"{int(rec.full_after):^4} "
            f"{int(rec.empty0_after)},{int(rec.empty1_after):<5} {outs}"
        )


def main():
    fifo = DualReaderAsyncFIFO(k=2)
    records = fifo.run(build_events())
    queues, outputs, final_wptr = audit(
        records, fifo.depth, 1 << (fifo.k + 1)
    )

    print("共同复位后的一写两读异步 FIFO（k=2，深度4，指针3位）")
    print("每条接受写入进入两条独立逻辑队列；存储器槽位由较慢读者释放。\n")
    print_rows(records, [1, 4, 8, 9, 12, 16, 17, 19, 27, 40, 43, 82, 84], fifo)

    rejected = [rec.time for rec in records if rec.w_accept is False]
    stopped_phase_rejected = [t for t in rejected if t <= 12]
    r0_out = [(rec.time, rec.r0data) for rec in records if rec.r0_accept]
    r1_out = [(rec.time, rec.r1data) for rec in records if rec.r1_accept]
    w_wraps = [
        (rec.time, rec.wptr_before, rec.wptr_after)
        for rec in records
        if rec.w_accept and rec.wptr_after < rec.wptr_before
    ]
    r0_wraps = [
        (rec.time, rec.r0ptr_before, rec.r0ptr_after)
        for rec in records
        if rec.r0_accept and rec.r0ptr_after < rec.r0ptr_before
    ]

    print("\n核对结果：")
    print(f"  接受写={fifo.accepted_writes}；接受读 R0/R1={fifo.accepted_reads}")
    print(f"  R0 前四项={r0_out[:4]}")
    print(f"  R1 前四项={r1_out[:4]}")
    print(f"  R0 停钟期间被拒绝的写请求={stopped_phase_rejected}；此时 R1 指针=4，R0 仍=0")
    print(f"  后续慢读端/同步延迟导致的保守写拒绝={rejected[len(stopped_phase_rejected):]}")
    print(f"  写指针回绕={[t for t, *_ in w_wraps]}")
    print(f"  R0 指针回绕={[t for t, *_ in r0_wraps]}")
    print(f"  两条逻辑队列剩余长度=({len(queues[0])},{len(queues[1])})")
    print(f"  最终指针 w/r0/r1={final_wptr}/{fifo.r0ptr}/{fifo.r1ptr}")

    assert stopped_phase_rejected == [8, 12]
    assert r0_out[:4] == [(16, "W01"), (17, "W02"), (18, "W03"), (19, "W04")]
    assert r1_out[:4] == [
        (8, "W01"), (9, "W02"), (10, "W03"), (11, "W04")
    ]
    assert outputs[0] == outputs[1]
    assert not queues[0] and not queues[1]
    assert w_wraps and r0_wraps
    print("\n演示完成：未覆盖任何仍被读者需要的槽；两条队列顺序和回绕距离均通过。")


if __name__ == "__main__":
    main()
