# cedana-samples
Samples and example workloads for our documentation. See docs.cedana.ai for examples on how to use!

For Kubernetes examples, see examples in our helm-chart repo: https://github.com/cedana/cedana-helm-charts. 

You can also use the cedana/cedana-samples docker container to run the arbitrary samples in here (which is the recommended approach - this container has everything!). 

## Setup 
Highly recommend using the requirements.txt to set up a virtualenv. 

``` sh
python3 -m env venv 
```

``` sh
source env/bin/active && pip install -r requirements.txt 
```


## llama.cpp 
Llama.cpp is downloaded directly from its github repo and unzipped into /usr/local/bin. To run arbitrary models, run `llama-cli -hf HUGGINGFACEMODEL`. 

## Host memory and fd samples (`cpu_smr/`)
Checkpoint/restore regression workloads for shared memory, ghost files (files unlinked while still mapped), IPC and the file descriptors servers hold. Python 3 stdlib only, no root needed. They are in both images, and in the slim `smr` image (with a bare `python3`) at `/cedana-samples/cpu_smr/...`.

The gate convention, shared with the GPU samples below: each logs `READY` once set up, progress lines while running, and `MISMATCH ...` (then exits non-zero) when state is lost or corrupted (`FAIL ...` when a call fails). With a gate file (`--gate PATH` here, the last argument for the GPU samples) each runs until that file exists, checks once more, logs `PASS` and exits 0, so a test can checkpoint and restore it mid-run and then create the gate. Without one they run until killed.

| Sample | What it exercises |
|---|---|
| `hostmem/hostmem.py` | Memory regions of every kind: private anonymous, `MAP_SHARED` anonymous, a named `/dev/shm` file (kept, or unlinked as a ghost), a memfd, a `MAP_SHARED` regular file (kept or unlinked), and a sparse file over 4 GiB. Each has data pages, pages written as zeros, holes, zero pages with a few non-zero bytes at either end, and a partly written last page (zero-page elision, short reads at EOF, ghost files over 4 GiB, parallel shmem dumps). `--procs` children keep verifying every region and counting in each shared region's comm page. The parent logs `OK <count per child>` and checks the counts keep advancing (the mappings are still shared after a restore) and that writes to private memory stay private. |
| `hostmem/shm-messages.py` | A producer and a consumer passing messages through POSIX shared memory slots and a `multiprocessing.Queue` (a pipe and POSIX semaphores in `/dev/shm`), as an inference server's processes do. Every message must arrive, in order, with its payload. Logs `SEQ <n>`. |
| `hostmem/ghost-busy-name.py` | Processes mapping a file that is then replaced under the same name (`os.replace`), as tensor-parallel ranks do with a JIT-compiled library: a ghost file whose name is busy with another file. |
| `fd_zoo.py` | One process tree holding the fd kinds servers use, each with state in flight: an epoll set (its registrations and which fds are ready), an eventfd count, an armed periodic timerfd, signals pending for a signalfd, queued inotify events, unread bytes in a pipe, a FIFO, `SOCK_STREAM` and `SOCK_SEQPACKET` socketpairs and a raw-mode pty (both directions), queued UDP datagrams, and an abstract unix socket and a TCP connection on 127.0.0.1 to a child process, with unread bytes both ways, whose listeners must still accept. Logs `FD <kind> ...` with the fd numbers, then `OK <round>` every 0.5 s. `tcp` needs established TCP connections checkpointed (CRIU's `--tcp-established`); `--skip` leaves kinds out. File locks are not covered. |


``` sh
# Every kind of region, small; or --preset large|sparse|hicache, or your own:
cpu_smr/hostmem/hostmem.py --preset small
cpu_smr/hostmem/hostmem.py --regions anon:64M,shm:256M,ghost-shm:1G,memfd:128M,ghost-sparse:5G \
    --procs 4 --dir /tmp --gate /tmp/gate --log /tmp/hostmem.log
# KIND:SIZE:sparse or :dense picks the layout (sparse kinds are sparse by default)
cpu_smr/hostmem/hostmem.py --regions ghost-shm:32M:sparse

cpu_smr/hostmem/shm-messages.py /tmp/msgs.log [--gate /tmp/gate]
cpu_smr/hostmem/ghost-busy-name.py /tmp/dir /tmp/ghost.log [--gate /tmp/gate]
cpu_smr/fd_zoo.py [--gate /tmp/gate] [--log /tmp/fd_zoo.log] [--dir /tmp] [--skip tcp,unix]
```

`/dev/shm` regions that don't fit in `/dev/shm` (64 MiB by default in Docker) are shrunk to fit, or skipped, with a `NOTE` line. `fd_zoo.py` skips a kind the Python or libc lacks with a `NOTE` line too.

## GPU checkpoint/restore samples (`gpu_smr/`)
Built with the other `gpu_smr/*.cu` samples, so in every image (`/app/gpu_smr/...`, `/cedana-samples/gpu_smr/...` in `smr`). Same gate convention, with the gate path as the last argument. Each checks its own results exactly; they link only libcudart (static), and those using the driver API load `libcuda.so.1` at run time.

| Sample | What it exercises |
|---|---|
| `host_register` | One GPU. A `malloc` buffer and an anonymous `MAP_SHARED` buffer (as KV-cache offload does), registered with `cudaHostRegister(..., cudaHostRegisterMapped)`; a kernel increments every element of both through their device pointers each iteration (zero-copy), then a copy through device memory must round-trip. 20 iterations without a gate. |
| `hostmem_ipc` | A customer's pinned-memory probe. One process per GPU (exec'd, like vLLM TP workers), each reading its neighbour's device memory through CUDA IPC every step. Rank 0 first sweeps `cudaHostRegister` (aligned and not), `cuMemHostRegister`, and `cudaHostAlloc` over sizes from 4 KiB to 16 GiB, across the 8 GiB boundary, then keeps a 9 GiB registered buffer live. Logs `STEP <n>` every second. Needs about 48 GiB of RAM; runs on one GPU too (`world` 1), without the IPC. |
| `cuda_ipc` | Two processes (fork) on one GPU sharing device buffers and interprocess events through `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle`/`cudaIpcGetEventHandle`. After the gate each exports a fresh buffer, whose handle must not collide with any exported before. A checkpoint must take both processes. |
| `memcpy_async_pinned` | `cudaMemcpyAsync` on four streams between pinned host memory (`cudaHostAlloc`, `cudaMallocHost`, `cudaHostRegister` of malloc and of `MAP_SHARED` memory) and device buffers allocated at new addresses after earlier ones were freed; copies are still in flight while gated. About 128 MiB of pinned memory. |
| `device_globals` | `__device__`/`__constant__` scalars and arrays written by `cudaMemcpyToSymbol(Async)` and by kernels, read back by symbol and through cached `cudaGetSymbolAddress` pointers, plus a `__device__` pointer to a `cudaMalloc` buffer. The symbols must not move. |
| `jit_modules` | Driver-API modules JIT-compiled from embedded PTX: one image loaded twice and another once with `cuModuleLoadDataEx` (identical images must not share globals), `cuLibraryLoadData`/CUkernel, per-module globals (`cuModuleGetGlobal`), 64 KiB of dynamic shared memory set by `cuFuncSetAttribute`/`cuKernelSetAttribute`, launched with `cuLaunchKernel`. |
| `vmm_pools` | VMM (`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`/`cuMemSetAccess`), with an aliased mapping and a chunk remapped every iteration, and stream-ordered allocation (`cudaMallocAsync`/`cudaFreeAsync`) from the default pool and from a `cudaMemPoolCreate` pool with a release threshold. |
| `replaced_module` | Driver-API modules whose image file is replaced by rename after they're loaded: image A loaded by path (`cuModuleLoad`) and from a mapping of another file (`cuModuleLoadData`), then both files replaced with image B (same kernel name, different result), and the path loaded again. Each module must keep running the image it was loaded from, and after the gate the path loads B and the mapping A. |
| `cumem_ipc_fd` | Two processes sharing VMM allocations through exported file descriptors, as NCCL does with cuMem buffers: `cuMemCreate` with POSIX-fd handles, `cuMemExportToShareableHandle`, the fds sent over a unix socket (`SCM_RIGHTS`), `cuMemImportFromShareableHandle`. The child closes one fd after importing and keeps the other open; after the gate a fresh allocation is exported and imported, and the kept fd is imported again. |

``` sh
gpu_smr/host_register [/tmp/gate]
gpu_smr/hostmem_ipc [world] [/tmp/gate]   # world 0 or none: all visible GPUs
gpu_smr/cuda_ipc [/tmp/gate]               # likewise memcpy_async_pinned, device_globals, jit_modules, vmm_pools, replaced_module, cumem_ipc_fd
kubectl create -f kubernetes/gpu/cuda-2xGPU-hostmem-ipc.yaml
```
