/**
  @file vmm_pools.cu
  @brief Virtual memory management (cuMemCreate/cuMemMap) with remapping, and stream-ordered pools.

  The VMM half reserves a virtual range, backs it with four physical chunks (cuMemCreate,
  cuMemMap, cuMemSetAccess, at the allocation granularity) and maps chunk 0 a second time at
  another address, as an alias. Kernels write the range and the alias; every iteration one chunk is
  unmapped, released, and replaced by a new physical chunk mapped at the same address, as
  PyTorch's expandable segments and vLLM's sleep mode do. The pool half keeps buffers from the
  device's default pool (cudaMallocAsync) and from a cudaMemPoolCreate pool with a release
  threshold (cudaMallocFromPoolAsync), and allocates, fills, and frees (cudaFreeAsync) more of
  both every iteration, some living across iterations. An interceptor that restores VMM ranges as
  plain allocations (losing the alias, or the current physical chunk behind an address), or does
  not rebuild the pools and their live allocations, corrupts the contents or faults.

  The driver API is reached through dlopen("libcuda.so.1"), so the binary links only libcudart.

  With no argument the work is 30 iterations. With a gate-file path as argv[1] it prints READY
  after the first iteration and iterates until that file exists, letting a test checkpoint and
  restore it mid-work, then create the gate; it then remaps and allocates once more. Every
  iteration, and the phase after it, checks every word of the range, the alias and the long-lived
  pool buffers before changing, remapping or freeing any of them (so a checkpoint between
  iterations that loses a chunk about to be replaced is still caught), and again after: a
  difference prints MISMATCH. Prints PASS and exits 0; non-zero on failure.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

// The driver API, from libcuda.so.1 (cuda.h maps names such as cuMemsetD32 to their _v2 symbols)
#define XSTR(x) STR(x)
#define STR(x) #x
#define DRIVER_FNS(X)                                                                      \
  X(cuGetErrorString) X(cuCtxGetDevice) X(cuMemGetAllocationGranularity) X(cuMemAddressReserve) \
  X(cuMemAddressFree) X(cuMemCreate) X(cuMemRelease) X(cuMemMap) X(cuMemUnmap) X(cuMemSetAccess)
#define DECLARE(fn) static decltype(&fn) d_##fn;
DRIVER_FNS(DECLARE)

static int load_driver() {
  void *h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) {
    printf("FAIL dlopen libcuda.so.1: %s\n", dlerror());
    return 1;
  }
#define LOAD(fn)                                       \
  if (!(d_##fn = (decltype(&fn))dlsym(h, XSTR(fn)))) { \
    printf("FAIL dlsym %s\n", XSTR(fn));               \
    return 1;                                          \
  }
  DRIVER_FNS(LOAD)
  return 0;
}

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

static int check(CUresult r, const char *what) {
  if (r != CUDA_SUCCESS) {
    const char *s = nullptr;
    if (!d_cuGetErrorString || d_cuGetErrorString(r, &s) != CUDA_SUCCESS || !s) s = "unknown";
    printf("FAIL %s: %d %s\n", what, (int)r, s);
    return 1;
  }
  return 0;
}

// p[i] = base + i
__global__ void fill(unsigned *p, size_t n, unsigned base) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = base + (unsigned)i;
}
// p[i] = v
__global__ void set(unsigned *p, size_t n, unsigned v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = v;
}
// p[i] += v
__global__ void add(unsigned *p, size_t n, unsigned v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] += v;
}
// p[i] += q[i % m]
__global__ void accumulate(unsigned *p, size_t n, const unsigned *q, size_t m) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] += q[i % m];
}

static const int CHUNKS = 4;
static const size_t POOL_WORDS = 1 << 20;              // 4 MiB per long-lived pool buffer
static const uint64_t THRESHOLD = 64ull << 20;         // the created pool's release threshold
static const int HELD = 3;                             // pool buffers kept for this many iterations

static int verify(const char *name, const unsigned *got, size_t n, unsigned base) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != base + (unsigned)i) {
      printf("MISMATCH %s[%zu] = %u, expected %u\n", name, i, got[i], base + (unsigned)i);
      return 3;
    }
  }
  return 0;
}

struct Vmm {
  CUdevice dev;
  size_t gran = 0, words = 0; // bytes and words per chunk
  CUdeviceptr base = 0, alias = 0;
  CUmemGenericAllocationHandle handles[CHUNKS];
  unsigned bases[CHUNKS]; // chunk c's word i holds bases[c] + i
  CUmemAllocationProp prop = {};
  CUmemAccessDesc access = {};

  unsigned *chunk(int c) { return (unsigned *)(base + c * gran); }

  int map_chunk(int c) {
    if (check(d_cuMemCreate(&handles[c], gran, &prop, 0), "cuMemCreate")) return 1;
    if (check(d_cuMemMap(base + c * gran, gran, 0, handles[c], 0), "cuMemMap")) return 1;
    if (check(d_cuMemSetAccess(base + c * gran, gran, &access, 1), "cuMemSetAccess")) return 1;
    return 0;
  }

  int init() {
    if (check(d_cuCtxGetDevice(&dev), "cuCtxGetDevice")) return 1;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (check(d_cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity"))
      return 1;
    words = gran / sizeof(unsigned);
    if (check(d_cuMemAddressReserve(&base, CHUNKS * gran, 0, 0, 0), "cuMemAddressReserve")) return 1;
    if (check(d_cuMemAddressReserve(&alias, gran, 0, 0, 0), "cuMemAddressReserve (alias)")) return 1;
    for (int c = 0; c < CHUNKS; c++)
      if (map_chunk(c)) return 1;
    // The alias: chunk 0's physical memory at a second address
    if (check(d_cuMemMap(alias, gran, 0, handles[0], 0), "cuMemMap (alias)")) return 1;
    if (check(d_cuMemSetAccess(alias, gran, &access, 1), "cuMemSetAccess (alias)")) return 1;
    for (int c = 0; c < CHUNKS; c++) {
      bases[c] = 1000000u * (c + 1);
      fill<<<64, 256>>>(chunk(c), words, bases[c]);
    }
    return check(cudaDeviceSynchronize(), "fill");
  }

  // Replaces chunk c's physical memory with a new chunk at the same address
  int remap(int c, unsigned new_base) {
    if (check(d_cuMemUnmap(base + c * gran, gran), "cuMemUnmap")) return 1;
    if (check(d_cuMemRelease(handles[c]), "cuMemRelease")) return 1;
    if (map_chunk(c)) return 1;
    fill<<<64, 256>>>(chunk(c), words, new_base);
    bases[c] = new_base;
    return check(cudaDeviceSynchronize(), "fill after remap");
  }

  int check_contents(std::vector<unsigned> &got) {
    got.resize(CHUNKS * words);
    if (check(cudaMemcpy(got.data(), (void *)base, CHUNKS * gran, cudaMemcpyDeviceToHost), "cudaMemcpy VMM range"))
      return 2;
    char name[32];
    for (int c = 0; c < CHUNKS; c++) {
      snprintf(name, sizeof(name), "vmm chunk %d", c);
      if (int rc = verify(name, got.data() + c * words, words, bases[c])) return rc;
    }
    if (check(cudaMemcpy(got.data(), (void *)alias, gran, cudaMemcpyDeviceToHost), "cudaMemcpy VMM alias"))
      return 2;
    return verify("vmm alias", got.data(), words, bases[0]);
  }

  int destroy() {
    if (check(d_cuMemUnmap(alias, gran), "cuMemUnmap (alias)")) return 1;
    if (check(d_cuMemUnmap(base, CHUNKS * gran), "cuMemUnmap")) return 1;
    for (int c = 0; c < CHUNKS; c++)
      if (check(d_cuMemRelease(handles[c]), "cuMemRelease")) return 1;
    if (check(d_cuMemAddressFree(alias, gran), "cuMemAddressFree (alias)")) return 1;
    return check(d_cuMemAddressFree(base, CHUNKS * gran), "cuMemAddressFree");
  }
};

// A pool allocation and what it holds
struct Held {
  unsigned *p = nullptr;
  size_t words = 0;
  unsigned base = 0;
};

static int read_and_verify(const char *name, const unsigned *d, size_t words, unsigned base, cudaStream_t s,
                           std::vector<unsigned> &got) {
  got.resize(words);
  if (check(cudaMemcpyAsync(got.data(), d, words * sizeof(unsigned), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync"))
    return 2;
  if (check(cudaStreamSynchronize(s), "cudaStreamSynchronize")) return 2;
  return verify(name, got.data(), words, base);
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;
  std::vector<unsigned> got;

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaFree(0), "cudaFree(0)")) return 1; // makes the primary context current
  if (load_driver()) return 1;
  int pools_supported = 0;
  if (check(cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, 0), "cudaDeviceGetAttribute"))
    return 1;
  if (!pools_supported) {
    printf("FAIL the device does not support memory pools\n");
    return 1;
  }

  Vmm vmm;
  if (vmm.init()) return 1;
  printf("ok   VMM granularity %zu B\n", vmm.gran);

  cudaStream_t s;
  if (check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate")) return 1;
  cudaMemPool_t def_pool, pool;
  if (check(cudaDeviceGetDefaultMemPool(&def_pool, 0), "cudaDeviceGetDefaultMemPool")) return 1;
  cudaMemPoolProps props = {};
  props.allocType = cudaMemAllocationTypePinned;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = 0;
  if (check(cudaMemPoolCreate(&pool, &props), "cudaMemPoolCreate")) return 1;
  uint64_t threshold = THRESHOLD;
  if (check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold), "cudaMemPoolSetAttribute"))
    return 1;

  // Long-lived buffers, one from each pool: word i holds base + i
  unsigned *from_def = nullptr, *from_pool = nullptr;
  unsigned def_base = 7, pool_base = 11;
  if (check(cudaMallocAsync((void **)&from_def, POOL_WORDS * sizeof(unsigned), s), "cudaMallocAsync")) return 1;
  if (check(cudaMallocFromPoolAsync((void **)&from_pool, POOL_WORDS * sizeof(unsigned), pool, s),
            "cudaMallocFromPoolAsync"))
    return 1;
  fill<<<64, 256, 0, s>>>(from_def, POOL_WORDS, def_base);
  fill<<<64, 256, 0, s>>>(from_pool, POOL_WORDS, pool_base);
  if (check(cudaStreamSynchronize(s), "fill pool buffers")) return 1;

  Held held[HELD];
  // The VMM range and alias, the long-lived pool buffers and the held ones, as last left
  auto check_all = [&]() -> int {
    if (int rc = vmm.check_contents(got)) return rc;
    if (int rc = read_and_verify("default pool buffer", from_def, POOL_WORDS, def_base, s, got)) return rc;
    if (int rc = read_and_verify("created pool buffer", from_pool, POOL_WORDS, pool_base, s, got)) return rc;
    for (Held &h : held)
      if (h.p)
        if (int rc = read_and_verify("held pool buffer", h.p, h.words, h.base, s, got)) return rc;
    return 0;
  };
  long iterations = 0;
  while (true) {
    unsigned k = (unsigned)iterations + 1;

    // What the previous iteration left, before anything below changes, remaps or frees it
    if (int rc = check_all()) return rc;

    // VMM: everything +1, chunk 0 +10 more through the alias, then a chunk remapped
    add<<<64, 256>>>(vmm.chunk(0), CHUNKS * vmm.words, 1);
    add<<<64, 256>>>((unsigned *)vmm.alias, vmm.words, 10);
    if (check(cudaDeviceSynchronize(), "add")) return 2;
    for (int c = 0; c < CHUNKS; c++) vmm.bases[c] += 1;
    vmm.bases[0] += 10;
    if (vmm.remap(1 + k % (CHUNKS - 1), 5000000u + 1000u * k)) return 2;
    if (int rc = vmm.check_contents(got)) return rc;

    // Pools: short-lived buffers of varying sizes holding k and 2k, added to the long-lived ones
    size_t tw = (size_t)(1 + k % 4) << 18; // 1-4 MiB
    unsigned *t1 = nullptr, *t2 = nullptr;
    if (check(cudaMallocAsync((void **)&t1, tw * sizeof(unsigned), s), "cudaMallocAsync")) return 2;
    if (check(cudaMallocFromPoolAsync((void **)&t2, tw * sizeof(unsigned), pool, s), "cudaMallocFromPoolAsync"))
      return 2;
    set<<<64, 256, 0, s>>>(t1, tw, k);
    set<<<64, 256, 0, s>>>(t2, tw, 2 * k);
    accumulate<<<64, 256, 0, s>>>(from_def, POOL_WORDS, t1, tw);
    accumulate<<<64, 256, 0, s>>>(from_pool, POOL_WORDS, t2, tw);
    if (check(cudaFreeAsync(t1, s), "cudaFreeAsync")) return 2;
    if (check(cudaFreeAsync(t2, s), "cudaFreeAsync")) return 2;
    def_base += k;
    pool_base += 2 * k;
    if (int rc = read_and_verify("default pool buffer", from_def, POOL_WORDS, def_base, s, got)) return rc;
    if (int rc = read_and_verify("created pool buffer", from_pool, POOL_WORDS, pool_base, s, got)) return rc;

    // A buffer from the created pool that lives for HELD iterations, checked when it is freed
    Held &h = held[k % HELD];
    if (h.p) {
      if (int rc = read_and_verify("held pool buffer", h.p, h.words, h.base, s, got)) return rc;
      if (check(cudaFreeAsync(h.p, s), "cudaFreeAsync held")) return 2;
    }
    h.words = (size_t)(1 + k % 3) << 17;
    h.base = 0xa0000000u + k;
    if (check(cudaMallocFromPoolAsync((void **)&h.p, h.words * sizeof(unsigned), pool, s), "cudaMallocFromPoolAsync"))
      return 2;
    fill<<<64, 256, 0, s>>>(h.p, h.words, h.base);
    if (check(cudaStreamSynchronize(s), "fill held")) return 2;

    iterations++;
    if (iterations == 1 && gate) printf("READY\n");
    if (gate ? access(gate, F_OK) == 0 : iterations >= 30) break;
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  printf("ok   %ld iterations\n", iterations);

  // The pool's attribute must still be set
  uint64_t got_threshold = 0;
  if (check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &got_threshold), "cudaMemPoolGetAttribute"))
    return 4;
  if (got_threshold != THRESHOLD) {
    printf("MISMATCH release threshold %llu, expected %llu\n", (unsigned long long)got_threshold,
           (unsigned long long)THRESHOLD);
    return 3;
  }
  // After a restore, when gated: everything as the last iteration left it, before it is remapped
  if (int rc = check_all()) return rc;
  // Then remap every chunk but 0, trim the pool, allocate from both again
  for (int c = 1; c < CHUNKS; c++)
    if (vmm.remap(c, 9000000u + c)) return 4;
  if (int rc = vmm.check_contents(got)) return rc;
  if (check(cudaMemPoolTrimTo(pool, 0), "cudaMemPoolTrimTo")) return 4;
  unsigned *n1 = nullptr, *n2 = nullptr;
  const size_t nw = 3 << 18;
  if (check(cudaMallocAsync((void **)&n1, nw * sizeof(unsigned), s), "cudaMallocAsync after")) return 4;
  if (check(cudaMallocFromPoolAsync((void **)&n2, nw * sizeof(unsigned), pool, s), "cudaMallocFromPoolAsync after"))
    return 4;
  fill<<<64, 256, 0, s>>>(n1, nw, 0x51u);
  fill<<<64, 256, 0, s>>>(n2, nw, 0x52u);
  if (int rc = read_and_verify("new default pool buffer", n1, nw, 0x51u, s, got)) return rc;
  if (int rc = read_and_verify("new created pool buffer", n2, nw, 0x52u, s, got)) return rc;
  // Everything from before is still intact
  if (int rc = read_and_verify("default pool buffer", from_def, POOL_WORDS, def_base, s, got)) return rc;
  if (int rc = read_and_verify("created pool buffer", from_pool, POOL_WORDS, pool_base, s, got)) return rc;
  for (Held &h : held)
    if (h.p)
      if (int rc = read_and_verify("held pool buffer", h.p, h.words, h.base, s, got)) return rc;
  printf("ok   remap and allocations after\n");

  for (unsigned *p : {n1, n2, from_def, from_pool})
    if (check(cudaFreeAsync(p, s), "cudaFreeAsync")) return 5;
  for (Held &h : held)
    if (h.p && check(cudaFreeAsync(h.p, s), "cudaFreeAsync held")) return 5;
  if (check(cudaStreamSynchronize(s), "cudaFreeAsync")) return 5;
  if (check(cudaMemPoolDestroy(pool), "cudaMemPoolDestroy")) return 5;
  if (check(cudaStreamDestroy(s), "cudaStreamDestroy")) return 5;
  if (vmm.destroy()) return 5;
  printf("PASS\n");
  return 0;
}
