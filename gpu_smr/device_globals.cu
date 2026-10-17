/**
  @file device_globals.cu
  @brief __device__ and __constant__ globals that change every iteration, from the host and kernels.

  Module globals live in the CUDA runtime's own module, not in cudaMalloc memory: a scalar and
  an array in __constant__ memory written with cudaMemcpyToSymbol, a scalar and arrays in
  __device__ memory accumulated by kernels, one array also written through the pointer from
  cudaGetSymbolAddress (cached at startup, as frameworks do) and by cudaMemcpyToSymbolAsync, and a
  __device__ pointer to a cudaMalloc buffer that kernels dereference. An interceptor that does not
  save the globals' contents, or puts them at other addresses on restore (so the cached pointer
  and the stored device pointer go stale), loses or corrupts the values.

  With no argument the work is 100 iterations. With a gate-file path as argv[1] it prints READY
  after the first iteration and iterates until that file exists, letting a test checkpoint and
  restore it mid-work, then create the gate. Every iteration first checks that every global
  still holds what the previous iteration left (so a checkpoint between iterations that loses one
  is caught before it is overwritten), and again after its own changes: each is read back with
  cudaMemcpyFromSymbol and through its cached address, the constants also through a kernel, and
  the addresses must not have moved. Everything is checked once more after the last iteration. A
  difference prints MISMATCH. Prints PASS and exits 0; non-zero on failure.
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>

static const int NC = 256;  // __constant__ array elements
static const int ND = 4096; // __device__ array elements

__constant__ unsigned c_scalar;
__constant__ unsigned c_arr[NC];
__device__ unsigned long long g_scalar;
__device__ unsigned g_arr[ND];
__device__ unsigned g_ptr_arr[ND]; // written through the cudaGetSymbolAddress pointer
__device__ unsigned *g_buf;        // points at a cudaMalloc buffer

// g_scalar += c_scalar; g_arr[i] += c_arr[i % NC] + i; g_buf[i] ^= g_arr[i]
__global__ void accumulate() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ND; i += gridDim.x * blockDim.x) {
    g_arr[i] += c_arr[i % NC] + i;
    g_buf[i] ^= g_arr[i];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) g_scalar += c_scalar;
}

// out[0] = c_scalar, out[1 + j] = c_arr[j]: the constants as kernels see them
__global__ void read_constants(unsigned *out) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < NC; j += gridDim.x * blockDim.x) out[1 + j] = c_arr[j];
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = c_scalar;
}

// p[i] = p[i] * 5 + k, through a pointer to g_ptr_arr
__global__ void through_pointer(unsigned *p, unsigned k) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ND; i += gridDim.x * blockDim.x) p[i] = p[i] * 5 + k;
}

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

static const char *when = ""; // which check is running, for the messages

static int compare(const char *name, const unsigned *got, const unsigned *want, int n) {
  for (int i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("MISMATCH %s[%d] = %u, expected %u (%s)\n", name, i, got[i], want[i], when);
      return 3;
    }
  }
  return 0;
}

// The host's copies of what every global must hold
static unsigned h_cs, h_c_arr[NC], h_arr[ND], h_ptr_arr[ND], h_buf[ND];
static unsigned long long h_scalar;

// Where the globals and the buffer are, cached once, and a buffer for read_constants
static unsigned *buf, *ptr_arr, *arr, *d_consts;
static void *scalar;
static cudaStream_t stream;

// Reads every global back, by symbol, through the cached addresses and (the constants) through a
// kernel, and checks it against the host's copies and that no symbol has moved
static int verify_all(const char *w) {
  when = w;
  static unsigned got[ND];
  unsigned long long s = 0;
  unsigned c = 0;
  if (check(cudaMemcpyFromSymbol(&s, g_scalar, sizeof(s)), "cudaMemcpyFromSymbol g_scalar")) return 2;
  if (s != h_scalar) {
    printf("MISMATCH g_scalar = %llu, expected %llu (%s)\n", s, h_scalar, when);
    return 3;
  }
  if (check(cudaMemcpy(&s, scalar, sizeof(s), cudaMemcpyDeviceToHost), "cudaMemcpy g_scalar")) return 2;
  if (s != h_scalar) {
    printf("MISMATCH g_scalar through its address = %llu, expected %llu (%s)\n", s, h_scalar, when);
    return 3;
  }
  if (check(cudaMemcpyFromSymbol(&c, c_scalar, sizeof(c)), "cudaMemcpyFromSymbol c_scalar")) return 2;
  if (c != h_cs) {
    printf("MISMATCH c_scalar = %u, expected %u (%s)\n", c, h_cs, when);
    return 3;
  }
  if (check(cudaMemcpyFromSymbol(got, c_arr, sizeof(h_c_arr)), "cudaMemcpyFromSymbol c_arr")) return 2;
  if (int rc = compare("c_arr", got, h_c_arr, NC)) return rc;
  read_constants<<<1, 256>>>(d_consts);
  if (check(cudaGetLastError(), "read_constants launch")) return 2;
  if (check(cudaMemcpy(got, d_consts, (NC + 1) * sizeof(unsigned), cudaMemcpyDeviceToHost), "read_constants"))
    return 2;
  if (got[0] != h_cs) {
    printf("MISMATCH c_scalar in a kernel = %u, expected %u (%s)\n", got[0], h_cs, when);
    return 3;
  }
  if (int rc = compare("c_arr in a kernel", got + 1, h_c_arr, NC)) return rc;
  if (check(cudaMemcpyFromSymbol(got, g_arr, sizeof(h_arr)), "cudaMemcpyFromSymbol g_arr")) return 2;
  if (int rc = compare("g_arr", got, h_arr, ND)) return rc;
  if (check(cudaMemcpy(got, arr, sizeof(h_arr), cudaMemcpyDeviceToHost), "cudaMemcpy g_arr")) return 2;
  if (int rc = compare("g_arr through its address", got, h_arr, ND)) return rc;
  if (check(cudaMemcpyFromSymbolAsync(got, g_ptr_arr, sizeof(h_ptr_arr), 0, cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyFromSymbolAsync g_ptr_arr"))
    return 2;
  if (check(cudaStreamSynchronize(stream), "cudaMemcpyFromSymbolAsync")) return 2;
  if (int rc = compare("g_ptr_arr", got, h_ptr_arr, ND)) return rc;
  if (check(cudaMemcpy(got, buf, sizeof(h_buf), cudaMemcpyDeviceToHost), "cudaMemcpy buf")) return 2;
  if (int rc = compare("*g_buf", got, h_buf, ND)) return rc;

  // The symbols must still be where they were
  unsigned *p = nullptr, *q = nullptr, *stored = nullptr;
  if (check(cudaGetSymbolAddress((void **)&p, g_ptr_arr), "cudaGetSymbolAddress g_ptr_arr")) return 2;
  if (check(cudaGetSymbolAddress((void **)&q, g_arr), "cudaGetSymbolAddress g_arr")) return 2;
  if (p != ptr_arr || q != arr) {
    printf("MISMATCH symbols moved: g_ptr_arr %p -> %p, g_arr %p -> %p (%s)\n", ptr_arr, p, arr, q, when);
    return 3;
  }
  if (check(cudaMemcpyFromSymbol(&stored, g_buf, sizeof(stored)), "cudaMemcpyFromSymbol g_buf")) return 2;
  if (stored != buf) {
    printf("MISMATCH g_buf = %p, expected %p (%s)\n", stored, buf, when);
    return 3;
  }
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) return 1;

  // Initial values: every global starts from a known pattern
  for (int i = 0; i < ND; i++) {
    h_arr[i] = 3u * i;
    h_ptr_arr[i] = 7u * i + 1;
    h_buf[i] = 0x9e3779b9u * i;
  }
  h_scalar = 1ull << 40; // past 32 bits, so a truncated copy shows
  h_cs = 1; // the constants as iteration 0 would leave them
  for (int j = 0; j < NC; j++) h_c_arr[j] = 1;
  if (check(cudaMemcpyToSymbol(c_scalar, &h_cs, sizeof(h_cs)), "cudaMemcpyToSymbol c_scalar")) return 1;
  if (check(cudaMemcpyToSymbol(c_arr, h_c_arr, sizeof(h_c_arr)), "cudaMemcpyToSymbol c_arr")) return 1;
  if (check(cudaMemcpyToSymbol(g_arr, h_arr, sizeof(h_arr)), "cudaMemcpyToSymbol g_arr")) return 1;
  if (check(cudaMemcpyToSymbol(g_ptr_arr, h_ptr_arr, sizeof(h_ptr_arr)), "cudaMemcpyToSymbol g_ptr_arr")) return 1;
  if (check(cudaMemcpyToSymbol(g_scalar, &h_scalar, sizeof(h_scalar)), "cudaMemcpyToSymbol g_scalar")) return 1;
  if (check(cudaMalloc(&d_consts, (NC + 1) * sizeof(unsigned)), "cudaMalloc")) return 1;
  if (check(cudaMalloc(&buf, sizeof(h_buf)), "cudaMalloc")) return 1;
  if (check(cudaMemcpy(buf, h_buf, sizeof(h_buf), cudaMemcpyHostToDevice), "cudaMemcpy to device")) return 1;
  if (check(cudaMemcpyToSymbol(g_buf, &buf, sizeof(buf)), "cudaMemcpyToSymbol g_buf")) return 1;

  // The addresses, cached once
  if (check(cudaGetSymbolAddress((void **)&ptr_arr, g_ptr_arr), "cudaGetSymbolAddress g_ptr_arr")) return 1;
  if (check(cudaGetSymbolAddress((void **)&arr, g_arr), "cudaGetSymbolAddress g_arr")) return 1;
  if (check(cudaGetSymbolAddress(&scalar, g_scalar), "cudaGetSymbolAddress g_scalar")) return 1;

  long iterations = 0;
  while (true) {
    unsigned k = (unsigned)iterations + 1;

    // What the previous iteration left, before anything replaces it
    if (int rc = verify_all("before the next iteration")) return rc;

    // The host writes the constants
    unsigned cs = h_cs = k * 1000u + 1;
    for (int j = 0; j < NC; j++) h_c_arr[j] = k * j + 1;
    if (check(cudaMemcpyToSymbol(c_scalar, &cs, sizeof(cs)), "cudaMemcpyToSymbol c_scalar")) return 2;
    if (check(cudaMemcpyToSymbol(c_arr, h_c_arr, sizeof(h_c_arr)), "cudaMemcpyToSymbol c_arr")) return 2;

    // Kernels accumulate into the __device__ globals and the buffer g_buf points at
    accumulate<<<8, 256>>>();
    if (check(cudaGetLastError(), "accumulate launch")) return 2;
    h_scalar += cs;
    for (int i = 0; i < ND; i++) {
      h_arr[i] += h_c_arr[i % NC] + i;
      h_buf[i] ^= h_arr[i];
    }

    // g_ptr_arr: a kernel through the cached pointer, then one element from the host, on a stream
    if (check(cudaDeviceSynchronize(), "accumulate")) return 2;
    through_pointer<<<8, 256, 0, stream>>>(ptr_arr, k);
    if (check(cudaGetLastError(), "through_pointer launch")) return 2;
    for (int i = 0; i < ND; i++) h_ptr_arr[i] = h_ptr_arr[i] * 5 + k;
    int at = (int)(k * 37 % ND);
    h_ptr_arr[at] = 0xc0de0000u + k;
    if (check(cudaMemcpyToSymbolAsync(g_ptr_arr, &h_ptr_arr[at], sizeof(unsigned), at * sizeof(unsigned),
                                      cudaMemcpyHostToDevice, stream),
              "cudaMemcpyToSymbolAsync g_ptr_arr"))
      return 2;
    if (check(cudaStreamSynchronize(stream), "through_pointer")) return 2;
    iterations++;

    // Read everything back, by symbol and through the cached addresses
    if (int rc = verify_all("after the iteration")) return rc;

    if (iterations == 1 && gate) printf("READY\n");
    if (gate ? access(gate, F_OK) == 0 : iterations >= 100) break;
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  printf("ok   %ld iterations\n", iterations);
  // After a restore, when gated: everything still holds the last iteration
  if (int rc = verify_all("after the iterations")) return rc;

  if (check(cudaStreamDestroy(stream), "cudaStreamDestroy")) return 4;
  if (check(cudaFree(buf), "cudaFree")) return 4;
  if (check(cudaFree(d_consts), "cudaFree")) return 4;
  printf("PASS\n");
  return 0;
}
