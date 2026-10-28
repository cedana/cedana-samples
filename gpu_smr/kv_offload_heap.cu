/**
  @file kv_offload_heap.cu
  @brief vLLM's CPU KV-cache offload (SimpleCPUOffloadConnector), with one CPU region on the brk heap.

  A customer's prefill worker, with vLLM's CPU KV offload (76 GB per GPU), failed at startup
  under cedana with "cuMemHostRegister(..., 0x46f78): memory on the brk heap cannot be registered
  under cedana": malloc had served one of the CPU tensors the connector pins from the top chunk of
  the brk heap, and vLLM raises when the registration fails. This sample makes the same
  CUDA calls, in the same order and with the same arguments, as vLLM does, with no Python, torch or
  vLLM, so it runs from cedana-samples' lite image. Each step below names the vLLM code it stands
  for (vLLM main, October 2026; vllm/v1/simple_kv_offload/):

  1. Startup. The model runner has allocated the GPU KV cache, one storage that the worker views
     as regions of [num_blocks, block_bytes] (worker.py, SimpleCPUOffloadWorker.register_kv_caches).
     Here: one cudaMalloc carved into NUM_REGIONS regions, with block sizes that differ between
     regions, as they do in hybrid models ("Tensors may have different page_size_bytes").
  2. The CPU side of each region (worker.py, _init_cpu_mode), one region at a time:
       tensor = torch.zeros((num_cpu_blocks, block_bytes), dtype=int8, device="cpu")
       pin_tensor(tensor)  ->  torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)
     torch's CPU allocator is posix_memalign(64) (c10::alloc_cpu), zero-filled; pin_tensor raises
     RuntimeError on any error, so a refused registration kills the worker. Here: posix_memalign(64)
     + memset, then cudaHostRegister(ptr, nbytes, cudaHostRegisterDefault), failing the same way.
     One region is exactly 0x46f78 bytes, the size in the customer's error, and as in the
     customer's worker, malloc serves it from the main arena's top chunk: malloc tries the top
     chunk before it considers mmap, so no mmap threshold (cedana's CEDANA_GPU_MMAP_THRESHOLD, or
     glibc's own) keeps it off the heap. The top chunk only stays that large when glibc's trim
     threshold is above the default 128 KiB (glibc raises it itself, up to 64 MiB, while its mmap
     threshold adapts); the sample raises it with mallopt(M_TRIM_THRESHOLD), then grows the heap
     with small objects on the main thread and frees them, as a large Python process does. The
     other regions are large and get mmap'd. That mix is the point: the heap region is registered
     unpinned while the others are pinned for real, and every transfer takes all of them in one
     batch. Free chunks elsewhere on the heap that could hold the tensor would be used before the
     top chunk; in the customer's worker there were none, and here probes of the tensor's size
     occupy any there are. The sample checks that the region is on [heap] (/proc/self/maps) and
     inside the top chunk, failing with SETUP otherwise rather than test the wrong path.
  3. The copy backend (copy_backend.py, DmaCopyBackend.init; cuda_mem_ops.py, build_params):
     cuMemcpyBatchAsync from cuGetProcAddress("cuMemcpyBatchAsync", 12080, 0) (the 12.8 entry point,
     with failIdx), a load stream and a store stream at the lowest priority
     (torch.cuda.Stream(priority=low_pri), non-blocking), one attribute for the whole batch:
     srcAccessOrder STREAM for stores (they read the live KV cache), ANY for loads. Transfers run on
     a background thread fed by a queue (DmaCopyBackend._copy_loop), which makes the stream wait
     on the store's wait_event, issues one batch for all regions with the descriptors layer-major
     (cuda_mem_ops.py, copy_blocks: every block of region 0, then of region 1, ...), and records a
     fresh event (torch.Event(), no timing) on the stream.
  4. Each engine step (worker.py):
       - the forward pass writes the new tokens' KV into GPU blocks on the compute stream (here a
         kernel, after a 20 ms spin so a copy that reads too early sees old data);
       - start_load_kv(): loads CPU blocks (prefix-cache hits) into free GPU blocks;
       - wait_for_save(): records one reused compute-done event on the compute stream and queues
         the store with it as wait_event;
       - get_finished() -> _poll_stream_events(): polls each direction's events, oldest first,
         with cudaEventQuery, the CPU and GPU blocks being used only once their event is done.
     Here every step checks the stored CPU blocks byte by byte and the loaded GPU blocks with a
     kernel, against a model of which tag every block holds.

  Beyond vLLM, for cedana: the heap region is cut from the middle of one allocation, so that 4 KiB
  of other live data (canaries, where other Python objects would be) sits on its first and last
  pages, and a thread rewrites and checks them for the whole run: nothing done to the region's
  pages may lose a neighbour's write. At the end
  one more 0x46f78-byte buffer is registered (after a restore, through the new GPU controller),
  registered again (must fail with cudaErrorHostMemoryAlreadyRegistered, 712, as natively), copied
  through and unregistered.

  The driver API is reached through dlopen("libcuda.so.1"), so the binary links only libcudart.
  A driver older than CUDA 12.8 has no cuMemcpyBatchAsync, and vLLM's connector refuses to start
  there; the sample says so (NOTE) and issues each descriptor with cudaMemcpyAsync instead.

  Each transfer moves 64 blocks of every region (448 copies per batch); KV_OFFLOAD_BLOCKS sets the
  number of blocks, up to 256, to bring a batch closer to a large model's.

  With no argument the work is 20 steps. With a gate-file path as argv[1] it prints READY after the
  first step and steps until that file exists, letting a test checkpoint and restore it mid-work,
  then create the gate; every CPU and GPU block is then checked, and it takes one more step. A
  difference prints MISMATCH. Prints the transfer throughput (THROUGHPUT; stores timed from the end
  of the forward pass they wait on; compare runs without a gate, native against cedana: an unpinned
  region in a batch makes cedana run its copies one by one), then PASS, and exits 0; non-zero on
  failure.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <malloc.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int NUM_REGIONS = 7;
// Bytes per block of each region; region SMALL is the one that lands on the heap
static const size_t REGION_BPB[NUM_REGIONS] = {32768, 16384, 172, 32768, 16384, 32768, 16384};
static const int SMALL = 2;
static const size_t NUM_GPU_BLOCKS = 512;
static const size_t NUM_CPU_BLOCKS = 1690;  // 1690 * 172 = 290680 = 0x46f78, the customer's size
static const size_t CUSTOMER_BYTES = 0x46f78;
static size_t blocks_per_transfer = 64;  // KV_OFFLOAD_BLOCKS sets it, up to NUM_GPU_BLOCKS / 2
static const int STEPS = 20;
static const unsigned long long FORWARD_NS = 20ull * 1000 * 1000;
static const int TRIM_THRESHOLD = 64 << 20;  // glibc's dynamic maximum
static const size_t HEAP_OBJECTS = 4096;
static const size_t HEAP_OBJECT_BYTES = 2000;  // 8 MiB of small objects
static const size_t CANARY_WORDS = 512;     // 4 KiB on each side of the heap region
static const unsigned POISON = 0xffffffffu;

// cuda_mem_ops.py: _CUmemcpyAttributes, CU_MEMCPY_SRC_ACCESS_ORDER_*, _BATCH_MEMCPY_FUNC_TYPE.
// Declared here because cuda.h before 12.8 has none of it.
struct MemLocation {
  unsigned type;
  int id;
};
struct MemcpyAttributes {
  unsigned srcAccessOrder;
  MemLocation srcLocHint;
  MemLocation dstLocHint;
  unsigned flags;
};
static const unsigned SRC_ACCESS_ORDER_STREAM = 1;
static const unsigned SRC_ACCESS_ORDER_ANY = 3;
typedef CUresult (*BatchMemcpyFn)(CUdeviceptr *dsts, CUdeviceptr *srcs, size_t *sizes, size_t count,
                                  MemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs,
                                  size_t *failIdx, CUstream stream);
typedef CUresult (*GetProcAddressFn)(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                                     CUdriverProcAddressQueryResult *status);

static BatchMemcpyFn batch_memcpy = nullptr;

// The KV bytes a block holds when tagged `tag`, per region; tag 0 is torch.zeros
__host__ __device__ static inline unsigned char kv_byte(unsigned tag, unsigned region, size_t i) {
  if (tag == 0) return 0;
  unsigned x = tag * 2654435761u ^ (region + 1) * 0x85ebca6bu ^ (unsigned)i * 0x9e3779b1u;
  x ^= x >> 15;
  return (unsigned char)(x | 1);
}

// The forward pass writing KV into GPU blocks (and the poisoning of blocks about to be loaded)
__global__ void write_blocks(unsigned char *base, size_t bpb, const unsigned *blocks, const unsigned *tags,
                             unsigned region) {
  unsigned char *p = base + (size_t)blocks[blockIdx.x] * bpb;
  for (size_t i = threadIdx.x; i < bpb; i += blockDim.x) p[i] = kv_byte(tags[blockIdx.x], region, i);
}

__global__ void check_blocks(const unsigned char *base, size_t bpb, const unsigned *blocks, const unsigned *tags,
                             unsigned region, unsigned *errors) {
  const unsigned char *p = base + (size_t)blocks[blockIdx.x] * bpb;
  for (size_t i = threadIdx.x; i < bpb; i += blockDim.x)
    if (p[i] != kv_byte(tags[blockIdx.x], region, i)) atomicAdd(errors, 1u);
}

__global__ void spin(unsigned long long ns) {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < ns);
}

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The /proc/self/maps name of the mapping holding p: "[heap]", "anonymous", a path, or "unmapped"
static std::string mapping_of(const void *p) {
  FILE *f = fopen("/proc/self/maps", "r");
  if (!f) return "unknown";
  uintptr_t a = (uintptr_t)p;
  char line[4096];
  std::string name = "unmapped";
  while (fgets(line, sizeof(line), f)) {
    unsigned long start, end;
    int pathAt = 0;
    if (sscanf(line, "%lx-%lx %*s %*s %*s %*s %n", &start, &end, &pathAt) < 2) continue;
    if (a < start || a >= end) continue;
    std::string path = pathAt ? line + pathAt : "";
    while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
    name = path.empty() ? "anonymous" : path;
    break;
  }
  fclose(f);
  return name;
}

// cuda_mem_ops.py, _resolve_batch_memcpy: cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
static int resolve_batch_memcpy() {
  void *h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) {
    printf("FAIL dlopen libcuda.so.1: %s\n", dlerror());
    return 1;
  }
  GetProcAddressFn getProcAddress = (GetProcAddressFn)dlsym(h, "cuGetProcAddress_v2");
  if (!getProcAddress) {
    printf("FAIL dlsym cuGetProcAddress_v2\n");
    return 1;
  }
  void *fn = nullptr;
  CUdriverProcAddressQueryResult status;
  CUresult r = getProcAddress("cuMemcpyBatchAsync", &fn, 12080, 0, &status);
  if (r != CUDA_SUCCESS || fn == nullptr) {
    printf("NOTE cuMemcpyBatchAsync is unavailable (driver older than CUDA 12.8, %d): vLLM's connector "
           "refuses to start here; copying each descriptor with cudaMemcpyAsync instead\n", (int)r);
    return 0;
  }
  batch_memcpy = (BatchMemcpyFn)fn;
  printf("VLLM cuGetProcAddress(cuMemcpyBatchAsync, 12080) = %p\n", fn);
  return 0;
}

// cuda_mem_ops.py, BatchMemcpyParams: per region a base on each side and the bytes per block
struct BatchParams {
  std::vector<CUdeviceptr> src_bases, dst_bases;
  std::vector<size_t> bpb;
  MemcpyAttributes attrs;
  cudaStream_t stream;
};

// cuda_mem_ops.py, copy_blocks: one batch, descriptors layer-major, one attribute for all
static int copy_blocks(const BatchParams &p, const std::vector<unsigned> &src_ids,
                       const std::vector<unsigned> &dst_ids) {
  size_t n = src_ids.size(), total = n * p.bpb.size();
  std::vector<CUdeviceptr> srcs(total), dsts(total);
  std::vector<size_t> sizes(total);
  for (size_t l = 0, k = 0; l < p.bpb.size(); l++)
    for (size_t j = 0; j < n; j++, k++) {
      srcs[k] = p.src_bases[l] + src_ids[j] * p.bpb[l];
      dsts[k] = p.dst_bases[l] + dst_ids[j] * p.bpb[l];
      sizes[k] = p.bpb[l];
    }
  if (!batch_memcpy) {
    for (size_t k = 0; k < total; k++)
      if (check(cudaMemcpyAsync((void *)dsts[k], (void *)srcs[k], sizes[k], cudaMemcpyDefault, p.stream),
                "cudaMemcpyAsync (descriptor)"))
        return 1;
    return 0;
  }
  MemcpyAttributes attrs = p.attrs;
  size_t attrsIdx = 0, failIdx = 0;
  CUresult r = batch_memcpy(dsts.data(), srcs.data(), sizes.data(), total, &attrs, &attrsIdx, 1, &failIdx,
                            (CUstream)p.stream);
  if (r != CUDA_SUCCESS) {
    printf("FAIL cuMemcpyBatchAsync(%zu descriptors): %d, failIdx %zu (vLLM raises \"batch memcpy failed\")\n",
           total, (int)r, failIdx);
    return 1;
  }
  return 0;
}

// worker.py: _load_events / _store_events, (event_idx, event) oldest first, and the high-water mark
struct EventList {
  std::mutex m;
  std::deque<std::pair<long, cudaEvent_t>> events;
  long hwm = -1;
};

// worker.py, _poll_stream_events
static int poll_stream_events(EventList &l, long *hwm) {
  std::lock_guard<std::mutex> lock(l.m);
  while (!l.events.empty()) {
    auto [idx, event] = l.events.front();
    cudaError_t e = cudaEventQuery(event);
    if (e == cudaErrorNotReady) break;
    if (check(e, "cudaEventQuery")) return 1;
    l.hwm = idx;
    cudaEventDestroy(event);
    l.events.pop_front();
  }
  *hwm = l.hwm;
  return 0;
}

// copy_backend.py, DmaCopyBackend: launch_copy() queues, _copy_loop() runs on its own thread
struct CopyItem {
  const BatchParams *params;
  std::vector<unsigned> src, dst;
  long event_idx;
  EventList *events;
  cudaEvent_t wait_event;
};

struct DmaCopyBackend {
  std::mutex m;
  std::condition_variable cv;
  std::deque<CopyItem *> queue;  // nullptr stops the loop
  std::thread thread;
  std::atomic<bool> failed{false};

  void launch_copy(CopyItem *item) {
    std::lock_guard<std::mutex> lock(m);
    queue.push_back(item);
    cv.notify_one();
  }

  void copy_loop() {
    cudaSetDevice(0);  // current_platform.set_device(device)
    while (true) {
      CopyItem *item;
      {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return !queue.empty(); });
        item = queue.front();
        queue.pop_front();
      }
      if (!item) return;
      const BatchParams &p = *item->params;
      if (item->wait_event && check(cudaStreamWaitEvent(p.stream, item->wait_event, 0), "stream.wait_event"))
        failed = true;
      if (copy_blocks(p, item->src, item->dst)) failed = true;
      cudaEvent_t event;
      if (check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "torch.Event()") ||
          check(cudaEventRecord(event, p.stream), "event.record(stream)")) {
        failed = true;
      } else {
        std::lock_guard<std::mutex> lock(item->events->m);
        item->events->events.emplace_back(item->event_idx, event);
      }
      delete item;
    }
  }
};

// Two neighbours of the heap region, as other objects on its first and last pages: a thread keeps
// rewriting them, checking each time that its last write is still there
struct Canaries {
  uint64_t *pre = nullptr, *post = nullptr;
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  std::thread thread;

  void run() {
    uint64_t gen = 0;
    while (!stop) {
      for (uint64_t *c : {pre, post})
        for (size_t i = 0; i < CANARY_WORDS; i++) {
          uint64_t got = __atomic_load_n(&c[i], __ATOMIC_RELAXED);
          if (got != gen) {
            printf("MISMATCH heap neighbour %s[%zu] = %llu, its last write was %llu\n", c == pre ? "before" : "after",
                   i, (unsigned long long)got, (unsigned long long)gen);
            failed = true;
            return;
          }
          __atomic_store_n(&c[i], gen + 1, __ATOMIC_RELAXED);
        }
      gen++;
      usleep(200);
    }
  }
};

struct State {
  unsigned char *gpu_base[NUM_REGIONS];
  unsigned char *cpu[NUM_REGIONS];
  std::vector<unsigned> gpu_tag, cpu_tag;  // what each block holds
  cudaStream_t compute, load_stream, store_stream;
  cudaEvent_t store_compute_done;  // worker.py: _store_compute_done, reused across steps
  BatchParams store_params, load_params;
  DmaCopyBackend backend;
  EventList load_events, store_events;
  unsigned *d_blocks, *d_tags, *d_errors;
  std::mt19937 rng{1234};
  double store_s = 0, load_s = 0;
  size_t store_bytes = 0, load_bytes = 0;
};

static size_t block_bytes() {
  size_t b = 0;
  for (size_t x : REGION_BPB) b += x;
  return b;
}

// Runs kernel launches for every region over `blocks` on the compute stream; with errors, returns
// how many bytes differ from their tags
static int for_gpu_blocks(State &s, const std::vector<unsigned> &blocks, const std::vector<unsigned> &tags,
                          bool checking, unsigned *errors) {
  if (blocks.empty()) return 0;
  size_t n = blocks.size();
  if (check(cudaMemcpyAsync(s.d_blocks, blocks.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice, s.compute),
            "upload block ids") ||
      check(cudaMemcpyAsync(s.d_tags, tags.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice, s.compute),
            "upload tags"))
    return 1;
  if (checking && check(cudaMemsetAsync(s.d_errors, 0, sizeof(unsigned), s.compute), "cudaMemsetAsync")) return 1;
  for (unsigned r = 0; r < NUM_REGIONS; r++) {
    if (checking)
      check_blocks<<<n, 256, 0, s.compute>>>(s.gpu_base[r], REGION_BPB[r], s.d_blocks, s.d_tags, r, s.d_errors);
    else
      write_blocks<<<n, 256, 0, s.compute>>>(s.gpu_base[r], REGION_BPB[r], s.d_blocks, s.d_tags, r);
  }
  if (check(cudaGetLastError(), checking ? "check_blocks launch" : "write_blocks launch")) return 1;
  if (checking) {
    if (check(cudaMemcpyAsync(errors, s.d_errors, sizeof(unsigned), cudaMemcpyDeviceToHost, s.compute),
              "read errors") ||
        check(cudaStreamSynchronize(s.compute), "check_blocks"))
      return 1;
  }
  return 0;
}

static int check_gpu(State &s, const std::vector<unsigned> &blocks, const char *what) {
  std::vector<unsigned> tags;
  for (unsigned b : blocks) tags.push_back(s.gpu_tag[b]);
  unsigned errors = 0;
  if (for_gpu_blocks(s, blocks, tags, true, &errors)) return 2;
  if (errors) {
    printf("MISMATCH %s: %u bytes of %zu GPU blocks differ\n", what, errors, blocks.size());
    return 3;
  }
  return 0;
}

static int check_cpu(State &s, const std::vector<unsigned> &blocks, const char *what) {
  for (unsigned r = 0; r < NUM_REGIONS; r++)
    for (unsigned c : blocks) {
      const unsigned char *p = s.cpu[r] + (size_t)c * REGION_BPB[r];
      for (size_t i = 0; i < REGION_BPB[r]; i++) {
        unsigned char e = kv_byte(s.cpu_tag[c], r, i);
        if (p[i] != e) {
          printf("MISMATCH %s: CPU region %u%s block %u [%zu] = 0x%02x, expected 0x%02x (tag 0x%x)\n", what, r,
                 r == SMALL ? " (heap)" : "", c, i, p[i], e, s.cpu_tag[c]);
          return 3;
        }
      }
    }
  return 0;
}

static std::vector<unsigned> all_blocks(size_t n) {
  std::vector<unsigned> v(n);
  for (size_t i = 0; i < n; i++) v[i] = i;
  return v;
}

// get_finished(), called until step k's load and store are both done. Times the load from its
// launch and the store from the end of the forward pass it waits on
static int get_finished(State &s, long k, double launched) {
  double deadline = now_s() + 120, forward_done = 0, load_done = 0, store_done = 0;
  while (!load_done || !store_done) {
    if (!forward_done) {
      cudaError_t e = cudaEventQuery(s.store_compute_done);
      if (e == cudaSuccess)
        forward_done = now_s();
      else if (e != cudaErrorNotReady && check(e, "cudaEventQuery(_store_compute_done)"))
        return 2;
    }
    long hwm;
    if (!load_done) {
      if (poll_stream_events(s.load_events, &hwm)) return 2;
      if (hwm >= k) load_done = now_s();
    }
    if (!store_done) {
      if (poll_stream_events(s.store_events, &hwm)) return 2;
      if (hwm >= k) store_done = now_s();
    }
    if (s.backend.failed) return 2;
    if (now_s() > deadline) {
      printf("FAIL the %s of step %ld never completed\n", load_done ? "store" : "load", k);
      return 2;
    }
    usleep(100);
  }
  s.load_s += load_done - launched;
  s.store_s += store_done - (forward_done ? forward_done : launched);
  return 0;
}

// One engine step: forward, start_load_kv(), wait_for_save(), then get_finished() until both
// transfers are done
static int step(State &s, long k, bool gated) {
  std::vector<unsigned> gpu = all_blocks(NUM_GPU_BLOCKS), cpu = all_blocks(NUM_CPU_BLOCKS);
  std::shuffle(gpu.begin(), gpu.end(), s.rng);
  std::shuffle(cpu.begin(), cpu.end(), s.rng);
  const size_t n = blocks_per_transfer;
  // Scheduler: GPU blocks this step computes and stores; free GPU blocks that CPU hits load into
  std::vector<unsigned> store_gpu(gpu.begin(), gpu.begin() + n), load_gpu(gpu.begin() + n, gpu.begin() + 2 * n);
  std::vector<unsigned> load_cpu(cpu.begin(), cpu.begin() + n), store_cpu(cpu.begin() + n, cpu.begin() + 2 * n);

  // The load's destinations are free blocks: poisoned, so a load that doesn't land shows
  std::vector<unsigned> poison(n, POISON);
  if (for_gpu_blocks(s, load_gpu, poison, false, nullptr) || check(cudaStreamSynchronize(s.compute), "poison"))
    return 2;
  for (unsigned b : load_gpu) s.gpu_tag[b] = POISON;

  // Forward: KV for this step's tokens, written after a spin so an early read sees old data
  std::vector<unsigned> tags;
  for (unsigned b : store_gpu) tags.push_back(((unsigned)(k + 1) << 12) | b);
  spin<<<1, 1, 0, s.compute>>>(FORWARD_NS);
  if (for_gpu_blocks(s, store_gpu, tags, false, nullptr)) return 2;
  for (size_t j = 0; j < n; j++) s.gpu_tag[store_gpu[j]] = tags[j];

  // start_load_kv(): CPU -> GPU, no wait_event
  double t0 = now_s();
  s.backend.launch_copy(new CopyItem{&s.load_params, load_cpu, load_gpu, k, &s.load_events, nullptr});
  // wait_for_save(): compute-done recorded on the current stream, then GPU -> CPU waiting on it
  if (check(cudaEventRecord(s.store_compute_done, s.compute), "_store_compute_done.record")) return 2;
  s.backend.launch_copy(
      new CopyItem{&s.store_params, store_gpu, store_cpu, k, &s.store_events, s.store_compute_done});
  if (gated) usleep(50 * 1000);  // copies in flight for a checkpoint to land on

  if (get_finished(s, k, t0)) return 2;
  s.load_bytes += n * block_bytes();
  s.store_bytes += n * block_bytes();
  for (size_t j = 0; j < n; j++) {
    s.gpu_tag[load_gpu[j]] = s.cpu_tag[load_cpu[j]];
    s.cpu_tag[store_cpu[j]] = s.gpu_tag[store_gpu[j]];
  }

  int rc;
  if ((rc = check_cpu(s, store_cpu, "stored blocks"))) return rc;
  if ((rc = check_gpu(s, load_gpu, "loaded blocks"))) return rc;
  printf("STEP %ld stored %zu blocks, loaded %zu, %zu descriptors per batch\n", k, n, n, n * NUM_REGIONS);
  return 0;
}

// cedana: one more registration of heap-sized memory late in the run (after a restore, through the
// new GPU controller), registered twice, copied through and unregistered
static int late_registration(State &s) {
  void *p = nullptr;
  if (posix_memalign(&p, 64, CUSTOMER_BYTES) != 0) {
    printf("FAIL posix_memalign\n");
    return 1;
  }
  for (size_t i = 0; i < CUSTOMER_BYTES; i++) ((unsigned char *)p)[i] = kv_byte(7, 0, i);
  printf("LATE buffer %p (0x%zx bytes) on %s\n", p, CUSTOMER_BYTES, mapping_of(p).c_str());
  if (check(cudaHostRegister(p, CUSTOMER_BYTES, cudaHostRegisterDefault), "late cudaHostRegister")) return 1;
  cudaError_t again = cudaHostRegister(p, CUSTOMER_BYTES, cudaHostRegisterDefault);
  cudaGetLastError();
  if (again != cudaErrorHostMemoryAlreadyRegistered) {
    printf("FAIL registering it again: %d %s, expected %d (cudaErrorHostMemoryAlreadyRegistered)\n", (int)again,
           cudaGetErrorString(again), (int)cudaErrorHostMemoryAlreadyRegistered);
    return 1;
  }
  unsigned char *d = nullptr;
  if (check(cudaMalloc(&d, CUSTOMER_BYTES), "cudaMalloc") ||
      check(cudaMemcpyAsync(d, p, CUSTOMER_BYTES, cudaMemcpyHostToDevice, s.load_stream), "late HtoD") ||
      check(cudaStreamSynchronize(s.load_stream), "late HtoD sync"))
    return 1;
  memset(p, 0, CUSTOMER_BYTES);
  if (check(cudaMemcpyAsync(p, d, CUSTOMER_BYTES, cudaMemcpyDeviceToHost, s.store_stream), "late DtoH") ||
      check(cudaStreamSynchronize(s.store_stream), "late DtoH sync"))
    return 1;
  for (size_t i = 0; i < CUSTOMER_BYTES; i++)
    if (((unsigned char *)p)[i] != kv_byte(7, 0, i)) {
      printf("MISMATCH late buffer [%zu] after a round trip\n", i);
      return 3;
    }
  if (check(cudaHostUnregister(p), "late cudaHostUnregister") || check(cudaFree(d), "cudaFree")) return 1;
  free(p);
  printf("LATE registered, 712 on the second registration, round trip exact, unregistered\n");
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;
  static State s;
  if (const char *env = getenv("KV_OFFLOAD_BLOCKS"))
    blocks_per_transfer = std::min<size_t>(std::max(atoi(env), 1), NUM_GPU_BLOCKS / 2);

  // The process has its CUDA context before the connector is set up
  if (check(cudaSetDevice(0), "cudaSetDevice") || check(cudaFree(0), "cudaFree(0)")) return 1;
  cudaDeviceProp prop;
  if (check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties")) return 1;
  printf("Device name: %s\n", prop.name);

  // The model runner's GPU KV cache: one allocation, viewed as [num_blocks, block_bytes] regions
  size_t gpu_bytes = 0, offsets[NUM_REGIONS];
  for (int r = 0; r < NUM_REGIONS; r++) {
    offsets[r] = gpu_bytes;
    gpu_bytes += (NUM_GPU_BLOCKS * REGION_BPB[r] + 255) & ~(size_t)255;
  }
  unsigned char *gpu_kv = nullptr;
  if (check(cudaMalloc(&gpu_kv, gpu_bytes), "cudaMalloc (KV cache)") ||
      check(cudaMemset(gpu_kv, 0, gpu_bytes), "cudaMemset (KV cache)"))
    return 1;
  for (int r = 0; r < NUM_REGIONS; r++) s.gpu_base[r] = gpu_kv + offsets[r];
  s.gpu_tag.assign(NUM_GPU_BLOCKS, 0);
  s.cpu_tag.assign(NUM_CPU_BLOCKS, 0);
  if (check(cudaMalloc(&s.d_blocks, NUM_GPU_BLOCKS * sizeof(unsigned)), "cudaMalloc") ||
      check(cudaMalloc(&s.d_tags, NUM_GPU_BLOCKS * sizeof(unsigned)), "cudaMalloc") ||
      check(cudaMalloc(&s.d_errors, sizeof(unsigned)), "cudaMalloc") ||
      check(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking), "compute stream"))
    return 1;

  // register_kv_caches: load and store streams at the lowest priority
  int low_pri, high_pri;
  if (check(cudaDeviceGetStreamPriorityRange(&low_pri, &high_pri), "priority_range") ||
      check(cudaStreamCreateWithPriority(&s.load_stream, cudaStreamNonBlocking, low_pri), "load_stream") ||
      check(cudaStreamCreateWithPriority(&s.store_stream, cudaStreamNonBlocking, low_pri), "store_stream"))
    return 1;

  // The trim threshold raised, standing in for whatever raised it in the customer's worker
  if (mallopt(M_TRIM_THRESHOLD, TRIM_THRESHOLD) != 1) {
    printf("FAIL mallopt(M_TRIM_THRESHOLD)\n");
    return 1;
  }

  // _init_cpu_mode: per region, torch.zeros then pin_tensor (cudaHostRegister(ptr, nbytes, 0))
  Canaries canaries;
  uintptr_t top_start = 0, top_block = 0;
  for (int r = 0; r < NUM_REGIONS; r++) {
    size_t nbytes = NUM_CPU_BLOCKS * REGION_BPB[r];
    void *p = nullptr;
    if (r == SMALL) {
      // The heap as the customer's worker has it when this tensor is allocated: a large top chunk
      // (small objects that grew the heap on the main thread, freed, merged back into it) and no
      // other free chunk that could hold the tensor
      size_t need = nbytes + 2 * CANARY_WORDS * sizeof(uint64_t);
      std::vector<void *> objects;
      for (size_t i = 0; i < HEAP_OBJECTS; i++) {
        objects.push_back(malloc(HEAP_OBJECT_BYTES));
        if (!objects.back()) {
          printf("FAIL malloc\n");
          return 1;
        }
        memset(objects.back(), 1, HEAP_OBJECT_BYTES);
      }
      for (void *o : objects) free(o);
      // The top chunk now starts at the lowest of the newest objects that sat back to back up to
      // the heap's end (the oldest few freed went to tcache instead and stay apart)
      size_t i = HEAP_OBJECTS - 1;
      uintptr_t stride = (uintptr_t)objects[i] - (uintptr_t)objects[i - 1];
      while (i > 8 && (uintptr_t)objects[i] - (uintptr_t)objects[i - 1] == stride) i--;
      top_start = (uintptr_t)objects[i] - 2 * sizeof(size_t);
      // A free chunk elsewhere that fits would be used first: probes of the tensor's size occupy
      // such chunks, as live objects do in the customer's worker, until one comes from the top
      size_t probes = 0;
      while (true) {
        void *q = malloc(need);
        if (!q) {
          printf("FAIL malloc\n");
          return 1;
        }
        if ((uintptr_t)q >= top_start) {
          free(q);
          break;
        }
        if (++probes > 100000) {
          printf("FAIL SETUP no allocation of 0x%zx bytes comes from the top chunk\n", need);
          return 1;
        }
      }
      printf("HEAP top chunk of %zu KiB at 0x%lx (from %zu freed objects of %zu bytes); %zu free chunks "
             "elsewhere occupied\n", ((uintptr_t)sbrk(0) - top_start) >> 10, (unsigned long)top_start,
             HEAP_OBJECTS, HEAP_OBJECT_BYTES, probes);
      // The heap region is cut from the middle of one allocation, so that live bytes of other
      // objects (the canaries) are certain to sit on its first and last pages
      void *block = nullptr;
      if (posix_memalign(&block, 64, need) != 0) {
        printf("FAIL torch.zeros (posix_memalign %zu)\n", nbytes);
        return 1;
      }
      top_block = (uintptr_t)block;
      canaries.pre = (uint64_t *)block;
      p = canaries.pre + CANARY_WORDS;
      canaries.post = (uint64_t *)((unsigned char *)p + nbytes);
      memset(canaries.pre, 0, CANARY_WORDS * sizeof(uint64_t));
      memset(canaries.post, 0, CANARY_WORDS * sizeof(uint64_t));
    } else if (posix_memalign(&p, 64, nbytes) != 0) {
      printf("FAIL torch.zeros (posix_memalign %zu)\n", nbytes);
      return 1;
    }
    memset(p, 0, nbytes);
    s.cpu[r] = (unsigned char *)p;
    std::string where = mapping_of(p);
    printf("VLLM torch.zeros((%zu, %zu), int8) region %d = %p, 0x%zx bytes, on %s\n", NUM_CPU_BLOCKS,
           REGION_BPB[r], r, p, nbytes, where.c_str());
    cudaError_t e = cudaHostRegister(p, nbytes, cudaHostRegisterDefault);
    if (e != cudaSuccess) {
      printf("FAIL pin_tensor: cudaHostRegister(%p, 0x%zx, 0) = %d %s; vLLM raises RuntimeError "
             "(\"cudaHostRegister failed\") here and the worker exits\n",
             p, nbytes, (int)e, cudaGetErrorString(e));
      return 1;
    }
  }
  if (mapping_of(s.cpu[SMALL]) != "[heap]") {
    printf("FAIL SETUP region %d (0x%zx bytes) is on %s, not [heap]: this run doesn't reproduce the customer's "
           "layout\n", SMALL, NUM_CPU_BLOCKS * REGION_BPB[SMALL], mapping_of(s.cpu[SMALL]).c_str());
    return 1;
  }
  for (int r = 0; r < NUM_REGIONS; r++)
    if (r != SMALL && mapping_of(s.cpu[r]) == "[heap]") {
      printf("FAIL SETUP region %d is on [heap] too: every batch must mix a heap region with mmap'd ones\n", r);
      return 1;
    }
  if (top_block < top_start) {
    printf("FAIL SETUP region %d was not served from the top chunk (0x%lx, below its start 0x%lx)\n", SMALL,
           (unsigned long)top_block, (unsigned long)top_start);
    return 1;
  }
  printf("HEAP region %d served from the top chunk, with canaries on its first and last pages at %p and %p\n",
         SMALL, canaries.pre, canaries.post);
  canaries.thread = std::thread(&Canaries::run, &canaries);

  // DmaCopyBackend.init: stores read GPU in stream order, loads read stable pinned host memory
  if (resolve_batch_memcpy()) return 1;
  for (int r = 0; r < NUM_REGIONS; r++) {
    CUdeviceptr g = (CUdeviceptr)s.gpu_base[r], c = (CUdeviceptr)s.cpu[r];
    s.store_params.src_bases.push_back(g);
    s.store_params.dst_bases.push_back(c);
    s.store_params.bpb.push_back(REGION_BPB[r]);
    s.load_params.src_bases.push_back(c);
    s.load_params.dst_bases.push_back(g);
    s.load_params.bpb.push_back(REGION_BPB[r]);
  }
  s.store_params.attrs = MemcpyAttributes{SRC_ACCESS_ORDER_STREAM, {0, 0}, {0, 0}, 0};
  s.store_params.stream = s.store_stream;
  s.load_params.attrs = MemcpyAttributes{SRC_ACCESS_ORDER_ANY, {0, 0}, {0, 0}, 0};
  s.load_params.stream = s.load_stream;
  if (check(cudaEventCreateWithFlags(&s.store_compute_done, cudaEventDisableTiming), "_store_compute_done"))
    return 1;
  s.backend.thread = std::thread(&DmaCopyBackend::copy_loop, &s.backend);

  int rc = 0;
  bool gated_done = false;
  for (long k = 0;; k++) {
    if ((rc = step(s, k, gate != nullptr))) break;
    if (canaries.failed) {
      rc = 3;
      break;
    }
    if (k == 0 && gate) printf("READY\n");
    if (gate) {
      if (gated_done) break;
      if (access(gate, F_OK) == 0) {
        if ((rc = check_cpu(s, all_blocks(NUM_CPU_BLOCKS), "every CPU block after the gate")) ||
            (rc = check_gpu(s, all_blocks(NUM_GPU_BLOCKS), "every GPU block after the gate")))
          break;
        gated_done = true;
      }
    } else if (k + 1 >= STEPS) {
      break;
    }
  }

  s.backend.launch_copy(nullptr);
  s.backend.thread.join();
  canaries.stop = true;
  canaries.thread.join();
  if (rc) return rc;
  if (canaries.failed) return 3;
  if ((rc = check_cpu(s, all_blocks(NUM_CPU_BLOCKS), "every CPU block at the end")) ||
      (rc = check_gpu(s, all_blocks(NUM_GPU_BLOCKS), "every GPU block at the end")))
    return rc;
  if ((rc = late_registration(s))) return rc;

  if (gate)
    printf("THROUGHPUT not measured with a gate (copies are left in flight on purpose)\n");
  else
    printf("THROUGHPUT store %.1f MB/s, load %.1f MB/s, %s\n", s.store_bytes / s.store_s / 1e6,
           s.load_bytes / s.load_s / 1e6, batch_memcpy ? "cuMemcpyBatchAsync" : "cudaMemcpyAsync per descriptor");
  for (int r = 0; r < NUM_REGIONS; r++)
    if (check(cudaHostUnregister(s.cpu[r]), "cudaHostUnregister")) return 1;
  printf("PASS\n");
  return 0;
}
