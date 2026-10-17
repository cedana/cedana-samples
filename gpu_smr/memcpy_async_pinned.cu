/**
  @file memcpy_async_pinned.cu
  @brief cudaMemcpyAsync between pinned host memory and reallocated device memory, on four streams.

  Each stream has a pinned source and destination on the host, pinned a different way:
  cudaHostAlloc, cudaMallocHost, cudaHostRegister of malloc memory, and cudaHostRegister of an
  anonymous MAP_SHARED mapping. Before the steady state each stream copies into a device buffer
  that is then freed, and the work uses device buffers allocated after that, at other addresses.
  Every iteration copies new host data in, transforms it with a kernel, waits on the device for
  ~20 ms, and copies it back, all asynchronously; with a gate the process sleeps while that work
  is queued, so a checkpoint lands with copies in flight. An interceptor that replays a copy to a
  freed (stale) device pointer, loses the pinning, or drops queued copies corrupts the round trip.

  With no argument the work is 20 iterations. With a gate-file path as argv[1] it prints READY
  after the first iteration and iterates until that file exists, letting a test checkpoint and
  restore it mid-work, then create the gate. Every iteration checks every element and a checksum
  of each stream's destination: a difference prints MISMATCH. Prints PASS and exits 0; non-zero on
  failure.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>

static const int NS = 4;                 // streams
static const size_t N = 2 << 20;         // 8 MiB of words per stream and direction
static const unsigned long long SPIN_NS = 20ull * 1000 * 1000;

__host__ __device__ static inline unsigned source(unsigned k, unsigned s, size_t i) {
  return ((unsigned)i * 2654435761u) ^ (k * 40503u) ^ (s << 28);
}
__host__ __device__ static inline unsigned transformed(unsigned x, unsigned k, unsigned s) {
  return x * 3u + k + s;
}

__global__ void transform(unsigned *d, size_t n, unsigned k, unsigned s) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    d[i] = transformed(d[i], k, s);
}

// Holds the stream for ns nanoseconds, so the copy back is still queued for a while
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

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;
  const size_t size = N * sizeof(unsigned);
  const char *kinds[NS] = {"cudaHostAlloc", "cudaMallocHost", "cudaHostRegister (malloc)",
                           "cudaHostRegister (MAP_SHARED)"};

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;

  // Pinned host memory, four ways: a source and a destination per stream
  unsigned *src[NS], *dst[NS];
  if (check(cudaHostAlloc((void **)&src[0], size, cudaHostAllocDefault), "cudaHostAlloc")) return 1;
  if (check(cudaHostAlloc((void **)&dst[0], size, cudaHostAllocPortable), "cudaHostAlloc")) return 1;
  if (check(cudaMallocHost((void **)&src[1], size), "cudaMallocHost")) return 1;
  if (check(cudaMallocHost((void **)&dst[1], size), "cudaMallocHost")) return 1;
  for (unsigned **p : {&src[2], &dst[2]}) {
    if (posix_memalign((void **)p, 4096, size) != 0) {
      printf("FAIL posix_memalign\n");
      return 1;
    }
    memset(*p, 0, size);
    if (check(cudaHostRegister(*p, size, cudaHostRegisterDefault), "cudaHostRegister (malloc)")) return 1;
  }
  for (unsigned **p : {&src[3], &dst[3]}) {
    *p = (unsigned *)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (*p == MAP_FAILED) {
      printf("FAIL mmap\n");
      return 1;
    }
    if (check(cudaHostRegister(*p, size, cudaHostRegisterPortable), "cudaHostRegister (MAP_SHARED)")) return 1;
  }

  cudaStream_t streams[NS];
  for (int s = 0; s < NS; s++)
    if (check(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking), "cudaStreamCreate")) return 1;

  // Copies into device buffers that are then freed; the work uses buffers allocated after them
  unsigned *old[NS], *dev[NS];
  for (int s = 0; s < NS; s++) {
    for (size_t i = 0; i < N; i++) src[s][i] = source(0, s, i);
    if (check(cudaMalloc(&old[s], size), "cudaMalloc (to be freed)")) return 1;
    if (check(cudaMemcpyAsync(old[s], src[s], size, cudaMemcpyHostToDevice, streams[s]),
              "cudaMemcpyAsync to the freed buffer"))
      return 1;
  }
  if (check(cudaDeviceSynchronize(), "copies to the freed buffers")) return 1;
  // Allocated while the old ones are live, so at other addresses; then the old ones are freed
  for (int s = 0; s < NS; s++)
    if (check(cudaMalloc(&dev[s], size + (s + 1) * (1 << 20)), "cudaMalloc")) return 1;
  for (int s = 0; s < NS; s++)
    if (check(cudaFree(old[s]), "cudaFree")) return 1;

  long iterations = 0;
  while (true) {
    unsigned k = (unsigned)iterations + 1;
    for (int s = 0; s < NS; s++) {
      for (size_t i = 0; i < N; i++) src[s][i] = source(k, s, i);
      memset(dst[s], 0, size);
    }
    for (int s = 0; s < NS; s++) {
      if (check(cudaMemcpyAsync(dev[s], src[s], size, cudaMemcpyHostToDevice, streams[s]), "cudaMemcpyAsync H2D"))
        return 2;
      transform<<<64, 256, 0, streams[s]>>>(dev[s], N, k, s);
      spin<<<1, 1, 0, streams[s]>>>(SPIN_NS);
      if (check(cudaGetLastError(), "kernel launch")) return 2;
      if (check(cudaMemcpyAsync(dst[s], dev[s], size, cudaMemcpyDeviceToHost, streams[s]), "cudaMemcpyAsync D2H"))
        return 2;
    }
    // With a gate, the copies are still queued while the process sleeps
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (int s = 0; s < NS; s++)
      if (check(cudaStreamSynchronize(streams[s]), "cudaStreamSynchronize")) return 2;
    iterations++;

    for (int s = 0; s < NS; s++) {
      uint64_t sum = 0, want = 0;
      for (size_t i = 0; i < N; i++) {
        unsigned e = transformed(source(k, s, i), k, s);
        if (dst[s][i] != e) {
          printf("MISMATCH %s stream %d iteration %u [%zu] = 0x%x, expected 0x%x\n", kinds[s], s, k, i,
                 dst[s][i], e);
          return 3;
        }
        sum += dst[s][i];
        want += e;
      }
      if (sum != want) {
        printf("MISMATCH %s stream %d checksum 0x%llx, expected 0x%llx\n", kinds[s], s,
               (unsigned long long)sum, (unsigned long long)want);
        return 3;
      }
    }
    if (iterations == 1 && gate) printf("READY\n");
    if (gate ? access(gate, F_OK) == 0 : iterations >= 20) break;
  }
  printf("ok   %ld iterations\n", iterations);

  for (int s = 0; s < NS; s++) {
    if (check(cudaStreamDestroy(streams[s]), "cudaStreamDestroy")) return 4;
    if (check(cudaFree(dev[s]), "cudaFree")) return 4;
  }
  for (int b = 0; b < 2; b++) {
    if (check(cudaFreeHost(b ? dst[0] : src[0]), "cudaFreeHost")) return 4;
    if (check(cudaFreeHost(b ? dst[1] : src[1]), "cudaFreeHost")) return 4;
    if (check(cudaHostUnregister(b ? dst[2] : src[2]), "cudaHostUnregister (malloc)")) return 4;
    if (check(cudaHostUnregister(b ? dst[3] : src[3]), "cudaHostUnregister (MAP_SHARED)")) return 4;
  }
  printf("PASS\n");
  return 0;
}
