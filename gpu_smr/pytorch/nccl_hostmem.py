"""Multi-GPU C/R check for NCCL and host memory mappings: one process per GPU (spawned like vLLM
TP workers), NCCL over a file rendezvous, GPU state that must survive restore, and pinned and
registered host memory.

Rank 0 first sweeps host memory registration across sizes that broke before (the 8 GiB boundary,
vLLM's ~12 GB CPU offload buffer): cudaHostRegister on torch buffers, cuMemHostRegister on mmap'd
memory, pin_memory, and pinned allocations after a register attempt. It then keeps a 9 GiB
registered buffer live, so a checkpoint has to carry registered host memory.

Every second rank 0 prints `STEP <n> state=<v> ...`. State is checked on every step, so a restore
that resumes with lost or corrupted GPU/host state, or broken NCCL, prints FAIL and exits non-zero.
"""
import ctypes
import mmap
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MiB = 1 << 20
GiB = 1 << 30
SWEEP = [4096, MiB, 64 * MiB, GiB, 4 * GiB, 7 * GiB, 8 * GiB - 2 * MiB, 8 * GiB, 8 * GiB + 2 * MiB,
         9 * GiB, 12001886208, 16 * GiB]
LIVE_REGISTERED = 9 * GiB  # past the 8 GiB boundary


def fail(rank, msg):
    print(f"FAIL rank={rank} {msg}", flush=True)
    sys.exit(1)


def rc_of(rc):
    return int(getattr(rc, "value", rc))


def copy_ok(t, value):
    """Pinned DMA actually works: fill on the host, copy to the GPU asynchronously, compare."""
    n = min(t.numel(), 64 * MiB)
    t[:n].fill_(value)
    d = t[:n].to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    return bool((d == value).all())


def register(t):
    return rc_of(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0))


def sweep(rank):
    cudart = torch.cuda.cudart()
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    cuda.cuMemHostUnregister.argtypes = [ctypes.c_void_p]

    for size in SWEEP:
        t = torch.zeros(size, dtype=torch.uint8)
        if (rc := register(t)) != 0:
            fail(rank, f"cudaHostRegister {size} B rc={rc}")
        if not copy_ok(t, 7):
            fail(rank, f"cudaHostRegister {size} B copy mismatch")
        cudart.cudaHostUnregister(t.data_ptr())
        del t

        m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        buf = (ctypes.c_char * size).from_buffer(m)
        if (rc := cuda.cuMemHostRegister_v2(ctypes.addressof(buf), size, 0)) != 0:
            fail(rank, f"cuMemHostRegister mmap {size} B rc={rc}")
        cuda.cuMemHostUnregister(ctypes.addressof(buf))
        del buf
        m.close()

        try:
            t = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        except Exception as e:  # noqa: BLE001
            fail(rank, f"pin_memory {size} B: {str(e).splitlines()[0][:120]}")
        if not copy_ok(t, 9):
            fail(rank, f"pin_memory {size} B copy mismatch")
        del t
        torch.cuda.empty_cache()
        print(f"SWEEP {size} B ok", flush=True)

    # Pinned allocations must keep working after a register attempt, whatever it returned
    t = torch.zeros(MiB, dtype=torch.uint8)
    if register(t) == 0:
        cudart.cudaHostUnregister(t.data_ptr())
    for attempt in (1, 2):
        try:
            p = torch.empty(2 * MiB, dtype=torch.uint8, pin_memory=True)
        except Exception as e:  # noqa: BLE001
            fail(rank, f"pin_memory after register attempt {attempt}: {str(e).splitlines()[0][:120]}")
        if not copy_ok(p, 11):
            fail(rank, f"pin_memory after register attempt {attempt}: copy mismatch")
        del p


def worker(rank, world, init):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=world)

    if rank == 0:
        sweep(rank)
    dist.barrier()

    state = torch.zeros(64 * MiB // 4, device="cuda")  # 64 MiB of GPU state, incremented per step
    pinned = torch.empty(4 * MiB, dtype=torch.uint8, pin_memory=True)  # cudaHostAlloc
    registered = torch.zeros(LIVE_REGISTERED if rank == 0 else 4 * MiB, dtype=torch.uint8)
    if (rc := register(registered)) != 0:  # as vLLM CPU offload and SGLang HiCache do
        fail(rank, f"cudaHostRegister live {registered.numel()} B rc={rc}")

    step = 0
    while True:
        step += 1
        state += 1
        total = torch.ones(1, device="cuda") * step
        dist.all_reduce(total)  # NCCL on every step
        for buf in (pinned, registered):
            if not copy_ok(buf, step % 256):
                fail(rank, f"step={step} host->device copy mismatch ({buf.numel()} B buffer)")
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
