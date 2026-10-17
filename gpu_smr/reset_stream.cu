/**
  @file reset_stream.cu
  @brief Use of the device after cudaDeviceReset().

  Creates a stream, runs a kernel, calls cudaDeviceReset(), then creates a stream again and runs
  a kernel on it, as the device check of GROMACS does at startup. Under an interceptor that keeps
  the runtime's per-context state after the reset, the stream after the reset fails with
  cudaErrorContextIsDestroyed (709). Prints one ok/FAIL line per step and PASS at the end; exits
  0 on PASS, 1 before the reset, 2 after it.
 */

#include <cstdio>
#include <cuda_runtime.h>

#define N 32

__global__ void fill(int *p, int v) { p[threadIdx.x] = v; }

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  printf("ok   %s\n", what);
  return 0;
}

int main() {
  setbuf(stdout, NULL);
  cudaStream_t s;
  int *d = nullptr;
  int h[N] = {0};

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaStreamCreate(&s), "cudaStreamCreate before reset")) return 1;
  if (check(cudaMalloc(&d, sizeof(h)), "cudaMalloc before reset")) return 1;
  fill<<<1, N, 0, s>>>(d, 1);
  if (check(cudaStreamSynchronize(s), "kernel before reset")) return 1;
  if (check(cudaStreamDestroy(s), "cudaStreamDestroy")) return 1;
  if (check(cudaDeviceReset(), "cudaDeviceReset")) return 1;

  if (check(cudaStreamCreate(&s), "cudaStreamCreate after reset")) return 2;
  if (check(cudaMalloc(&d, sizeof(h)), "cudaMalloc after reset")) return 2;
  fill<<<1, N, 0, s>>>(d, 7);
  if (check(cudaGetLastError(), "kernel launch after reset")) return 2;
  if (check(cudaMemcpyAsync(h, d, sizeof(h), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync")) return 2;
  if (check(cudaStreamSynchronize(s), "cudaStreamSynchronize")) return 2;
  if (h[N - 1] != 7) {
    printf("FAIL value %d, want 7\n", h[N - 1]);
    return 2;
  }
  printf("PASS\n");
  return 0;
}
