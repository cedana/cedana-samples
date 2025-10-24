/**
  @file mem-throughput-saxpy.cu
  @brief A simple CUDA measure to measure computational throughput using CUDA
  events

  This program simply exercises SAXPY: Single-Precision A*X Plus Y on a large
  number of elements. Typically, doing this operation repeatedly will saturate
  the memory bandwidth of the GPU, giving us effective bandwidth, which we call
  throughput. This effective bandwidth will always be less than the advertised
  theoretical bandwidth of the GPU.

  The output is in GFLOP/s
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "cuda_error.cuh"

#define MILLION 1024 * 1024

int N = 64 * MILLION; // Number of elements
int NThreads = 256;   // Threads per block

// Essentially only 2 FLOPS
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}
int flops = 2;

void usage(const char *progname) {
  fprintf(stderr, "Usage: %s [No. threads/block] [Nelemes]\n", progname);
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  if (argc > 1) {
    N = atoi(argv[1]);
    if (N == 0) {
      usage(argv[0]);
    }
  }
  if (argc > 2) {
    NThreads = atoi(argv[2]);
    if (NThreads == 0) {
      usage(argv[0]);
    }
  }

  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  cudaErrChk(cudaMalloc(&d_x, N * sizeof(float)));
  cudaErrChk(cudaMalloc(&d_y, N * sizeof(float)));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaErrChk(cudaEventCreate(&start));
  cudaErrChk(cudaEventCreate(&stop));

  cudaErrChk(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

  int NBlocks = (N + (NThreads - 1)) / NThreads;

  cudaErrChk(cudaEventRecord(start));
  saxpy<<<NBlocks, NThreads>>>(N, 2.0f, d_x, d_y);
  cudaErrChk(cudaEventRecord(stop));

  cudaErrChk(cudaPeekAtLastError());

  cudaErrChk(cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaErrChk(cudaEventSynchronize(stop));
  float milliseconds = 0;
  cudaErrChk(cudaEventElapsedTime(&milliseconds, start, stop));

  // Check if computations are correct

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = std::max(maxError, std::abs(y[i] - 4.0f));
  }
  if (maxError != 0) {
    fprintf(stderr, "%f error in output\n", maxError);
    exit(EXIT_FAILURE);
  }

  cudaErrChk(cudaFree(d_x));
  cudaErrChk(cudaFree(d_y));
  free(x);
  free(y);

  float gflops = N * flops / milliseconds / 1e6;

  printf("%f\n", gflops);
}
