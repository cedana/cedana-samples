/**
  @file cuda_graph_siblings.cu
  @brief Multiple live CUDA graphs of one topology at mixed warmth, for C/R.

  Builds K graphs of identical topology, each writing its own buffer: graphs
  [0,WARM) are launched every iteration (warm), [WARM,K) are never launched before
  the gate (cold). A template restore keeps one as the template and buckets the
  rest, deferring the cold ones until first launch. A test checkpoints during the
  warm phase, restores, opens the gate, and the cold siblings launch for the first
  time and must compute correctly. Buffer k must equal (#launches of k)*INCREMENT;
  drift prints MISMATCH and exits non-zero.
 */

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>

#include "cuda_error.cuh"

#define N 4096
#define INCREMENT 1
#define THREADS 256
#define K 4    // total graphs of the same topology
#define WARM 2 // graphs [0, WARM) are launched before the checkpoint

__global__ void addKernel(int *buf, int n, int increment) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] += increment;
}

static int *d_buf[K];
static cudaStream_t stream;
static cudaGraph_t graph[K];
static cudaGraphExec_t exec[K];

void handleSignal(int signal) { exit(signal); }

// Read back buf[k] and require every element == expected.
void check(int k, long long expected, const char *phase) {
  static int h_buf[N];
  cudaErrChk(cudaMemcpy(h_buf, d_buf[k], N * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    if (h_buf[i] != expected) {
      fprintf(stderr, "siblings MISMATCH phase=%s graph=%d index=%d value=%d expected=%lld\n",
              phase, k, i, h_buf[i], expected);
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);
  setbuf(stdout, NULL);

  const char *gate = (argc > 1) ? argv[1] : nullptr;
  const int blocks = (N + THREADS - 1) / THREADS;

  cudaErrChk(cudaStreamCreate(&stream));
  for (int k = 0; k < K; k++) {
    cudaErrChk(cudaMalloc(&d_buf[k], N * sizeof(int)));
    cudaErrChk(cudaMemset(d_buf[k], 0, N * sizeof(int)));
    // Identical capture sequence for every graph => identical topology key.
    cudaErrChk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    addKernel<<<blocks, THREADS, 0, stream>>>(d_buf[k], N, INCREMENT);
    cudaErrChk(cudaStreamEndCapture(stream, &graph[k]));
    cudaErrChk(cudaGraphInstantiate(&exec[k], graph[k], 0));
  }
  printf("siblings READY K=%d warm=%d cold=%d gate=%s\n", K, WARM, K - WARM,
         gate ? gate : "(none)");

  long long warmLaunches = 0;
  bool coldLaunched = false;

  while (true) {
    // Launch the warm siblings; their buffers track warmLaunches.
    for (int k = 0; k < WARM; k++) {
      cudaErrChk(cudaGraphLaunch(exec[k], stream));
    }
    cudaErrChk(cudaStreamSynchronize(stream));
    warmLaunches++;
    for (int k = 0; k < WARM; k++) check(k, warmLaunches, "warm");

    // After the gate opens (test does this post-restore), launch the cold
    // siblings for the first time and verify they compute correctly.
    if (gate && !coldLaunched && access(gate, F_OK) == 0) {
      for (int k = WARM; k < K; k++) {
        cudaErrChk(cudaGraphLaunch(exec[k], stream));
      }
      cudaErrChk(cudaStreamSynchronize(stream));
      for (int k = WARM; k < K; k++) check(k, INCREMENT, "cold");
      coldLaunched = true;
      printf("siblings COLD LAUNCHED OK (%d deferred graphs)\n", K - WARM);
    }

    printf("siblings warm_launches=%lld cold_done=%d OK\n", warmLaunches,
           coldLaunched ? 1 : 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}
