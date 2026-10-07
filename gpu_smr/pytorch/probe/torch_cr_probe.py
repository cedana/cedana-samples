"""Minimal multi-GPU C/R probe: one process per GPU (spawned like vLLM TP workers), NCCL over a
file rendezvous, GPU state that must survive restore, and pinned/registered host memory.

Every second rank 0 prints `STEP <n> state=<v> ...`. State is checked on every step, so a restore
that resumes with lost or corrupted GPU/host state, or broken NCCL, prints FAIL and exits non-zero.
"""
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MiB = 1 << 20


def fail(rank, msg):
    print(f"FAIL rank={rank} {msg}", flush=True)
    sys.exit(1)


def worker(rank, world, init):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=world)

    state = torch.zeros(64 * MiB // 4, device="cuda")  # 64 MiB of GPU state, incremented per step
    pinned = torch.empty(4 * MiB, dtype=torch.uint8, pin_memory=True)  # cudaHostAlloc
    registered = torch.zeros(4 * MiB, dtype=torch.uint8)  # cudaHostRegister, as offload/hicache do
    rc = torch.cuda.cudart().cudaHostRegister(registered.data_ptr(), registered.numel(), 0)
    rc = int(getattr(rc, "value", rc))
    if rc != 0:
        fail(rank, f"cudaHostRegister rc={rc}")

    step = 0
    while True:
        step += 1
        state += 1
        total = torch.ones(1, device="cuda") * step
        dist.all_reduce(total)  # NCCL on every step
        for buf in (pinned, registered):
            buf.fill_(step % 256)
            if not bool((buf.to("cuda", non_blocking=True) == step % 256).all()):
                fail(rank, f"step={step} host->device copy mismatch")
        torch.cuda.synchronize()
        if state[0].item() != step or state[-1].item() != step:
            fail(rank, f"step={step} gpu state={state[0].item()}")
        if total.item() != step * world:
            fail(rank, f"step={step} all_reduce={total.item()} want={step * world}")
        if rank == 0:
            print(f"STEP {step} state={int(state[0].item())} world={world}", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    world = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    init = f"file://{tempfile.mkdtemp()}/rendezvous"
    mp.spawn(worker, args=(world, init), nprocs=world, join=True)
