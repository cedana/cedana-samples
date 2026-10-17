#!/usr/bin/env python3
"""
A producer and a consumer passing messages through shared memory, as processes of an
inference server do: the producer writes each message's payload into a POSIX shared memory
slot, then sends its sequence number through a multiprocessing.Queue (a pipe and POSIX
semaphores in /dev/shm). The consumer requires every sequence number, in order, with the
payload that goes with it, and logs its progress. Many messages are in flight at any time,
so a checkpoint catches some in the queue, the pipe, and the slots.

With --gate PATH, once the file exists the consumer must receive more messages still, then
both stop and PASS is logged (exit 0), so a test can checkpoint and restore it mid-run, then
create the gate. Without it, it runs until killed.

Usage: shm-messages.py <log> [--gate PATH]
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import shared_memory

SLOTS, SLOT = 64, 4096


def payload(seq):
    return (seq.to_bytes(8, "little") * (SLOT // 8))


def producer(name, queue, free):
    shm = shared_memory.SharedMemory(name=name)
    seq = 0
    while True:
        free.acquire()  # a slot the consumer is done with
        slot = seq % SLOTS
        shm.buf[slot * SLOT:(slot + 1) * SLOT] = payload(seq)
        queue.put(seq)
        seq += 1


def consumer(name, queue, free, log_path, received):
    shm = shared_memory.SharedMemory(name=name)
    log = open(log_path, "a", buffering=1)
    expected = 0
    last_log = 0.0
    while True:
        seq = queue.get()
        slot = seq % SLOTS
        if seq != expected or bytes(shm.buf[slot * SLOT:(slot + 1) * SLOT]) != payload(seq):
            log.write(f"MISMATCH got {seq}, expected {expected}\n")
            sys.exit(1)
        free.release()
        expected += 1
        received.value = expected
        if time.monotonic() - last_log > 0.2:
            log.write(f"SEQ {seq}\n")
            last_log = time.monotonic()


def gate(path, procs, received, log_path):
    """Run until the gate exists, then require more messages, stop both and log PASS"""
    while not os.path.exists(path):
        for p in procs:
            if p.exitcode is not None:
                sys.exit(p.exitcode or 1)
        time.sleep(0.1)
    seen, deadline = received.value, time.monotonic() + 30
    while received.value <= seen:
        if time.monotonic() > deadline or any(p.exitcode is not None for p in procs):
            with open(log_path, "a") as log:
                log.write(f"MISMATCH no messages received after the gate, stuck at {seen}\n")
            sys.exit(1)
        time.sleep(0.1)
    for p in procs:
        p.terminate()
        p.join()
    with open(log_path, "a") as log:
        log.write(f"PASS {received.value} messages\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--gate", help="once this file exists: check progress, log PASS and exit 0")
    args = p.parse_args()
    log_path = args.log
    mp.set_start_method("fork")
    shm = shared_memory.SharedMemory(create=True, size=SLOTS * SLOT)
    queue = mp.Queue()
    free = mp.Semaphore(SLOTS)
    received = mp.Value("q", 0, lock=False)  # messages the consumer has checked
    procs = [
        mp.Process(target=producer, args=(shm.name, queue, free)),
        mp.Process(target=consumer, args=(shm.name, queue, free, log_path, received)),
    ]
    for p in procs:
        p.start()
    with open(log_path, "a") as log:
        log.write("READY\n")
    if args.gate:
        gate(args.gate, procs, received, log_path)
        shm.close()
        shm.unlink()
        return
    for p in procs:
        p.join()
        if p.exitcode:
            sys.exit(p.exitcode)


if __name__ == "__main__":
    main()
