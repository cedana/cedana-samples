/**
  @file host_register.cu
  @brief Pinned host memory from cudaHostRegister(), used zero-copy by kernels.

  Registers two host buffers as mapped (pinned) memory: one from malloc, as frameworks pin
  their own buffers, and one anonymous MAP_SHARED mapping, as KV-cache offload (SGLang
  HiCache) does. Each iteration a kernel increments every element of both through their
  device pointers, reading and writing host memory directly (zero-copy). An interceptor
  that doesn't restore the registration, or the mapping behind it, faults or loses updates.

  With no argument the work is 20 iterations. With a gate-file path as argv[1] it prints READY
  after the first iteration and iterates until that file exists, letting a test checkpoint and
  restore it mid-work, then create the gate. Every element must equal the iteration count, and
  a copy through device memory must round-trip: a difference prints MISMATCH. Prints PASS and
  exits 0; non-zero on failure.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>

static const size_t N = 1 << 20; // 4 MiB of ints per buffer

__global__ void increment(int *a, int *b, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    a[i] += 1;
    b[i] += 1;
  }
}

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

static int verify(const char *name, const int *buf, long expected) {
  for (size_t i = 0; i < N; i++) {
    if (buf[i] != expected) {
      printf("MISMATCH %s[%zu] = %d, expected %ld\n", name, i, buf[i], expected);
      return 1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;
  const size_t size = N * sizeof(int);

  int *heap = nullptr;
  if (posix_memalign((void **)&heap, 4096, size) != 0) {
    printf("FAIL posix_memalign\n");
    return 1;
  }
  int *shared = (int *)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (shared == MAP_FAILED) {
    printf("FAIL mmap\n");
    return 1;
  }
  memset(heap, 0, size);
  memset(shared, 0, size);

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaHostRegister(heap, size, cudaHostRegisterMapped), "cudaHostRegister (malloc)")) return 1;
  if (check(cudaHostRegister(shared, size, cudaHostRegisterMapped), "cudaHostRegister (MAP_SHARED)")) return 1;
  int *d_heap = nullptr, *d_shared = nullptr;
  if (check(cudaHostGetDevicePointer((void **)&d_heap, heap, 0), "cudaHostGetDevicePointer (malloc)")) return 1;
  if (check(cudaHostGetDevicePointer((void **)&d_shared, shared, 0), "cudaHostGetDevicePointer (MAP_SHARED)")) return 1;

  long iterations = 0;
  while (true) {
    increment<<<64, 256>>>(d_heap, d_shared, N);
    if (check(cudaDeviceSynchronize(), "kernel")) return 2;
    iterations++;
    if (iterations == 1 && gate) printf("READY\n");
    if (gate ? access(gate, F_OK) == 0 : iterations >= 20) break;
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (verify("malloc", heap, iterations) || verify("MAP_SHARED", shared, iterations)) return 3;
  printf("ok   %ld iterations\n", iterations);

  // A round trip through device memory, from and into pinned memory
  int *d = nullptr;
  if (check(cudaMalloc(&d, size), "cudaMalloc")) return 2;
  if (check(cudaMemcpy(d, heap, size, cudaMemcpyHostToDevice), "cudaMemcpy to device")) return 2;
  memset(shared, 0, size);
  if (check(cudaMemcpy(shared, d, size, cudaMemcpyDeviceToHost), "cudaMemcpy from device")) return 2;
  if (verify("round trip", shared, iterations)) return 3;
  printf("ok   round trip\n");

  if (check(cudaFree(d), "cudaFree")) return 2;
  if (check(cudaHostUnregister(heap), "cudaHostUnregister (malloc)")) return 4;
  if (check(cudaHostUnregister(shared), "cudaHostUnregister (MAP_SHARED)")) return 4;
  printf("PASS\n");
  return 0;
}
