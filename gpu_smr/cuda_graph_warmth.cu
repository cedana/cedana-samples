/**
  @file cuda_graph_warmth.cu
  @brief CUDA graph C/R at controllable warmth (launches before checkpoint).

  Captures + instantiates a graph, prints a READY marker before the first launch,
  and (if given a gate-file path as argv[1]) blocks until that file exists before
  launching -- letting a test checkpoint COLD (zero launches, fresh KV cache),
  restore, then create the gate to proceed. With no gate it launches immediately,
  so checkpointing after many iters gives the WARM case. Same self-check as
  cuda_graph_loop; drift prints MISMATCH and exits non-zero.
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

__global__ void addKernel(int *buf, int n, int increment) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] += increment;
}

void handleSignal(int signal) { exit(signal); }

int main(int argc, char **argv) {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);
  setbuf(stdout, NULL);

  const char *gate = (argc > 1) ? argv[1] : nullptr;

  int *d_buf = nullptr;
  cudaErrChk(cudaMalloc(&d_buf, N * sizeof(int)));
  cudaErrChk(cudaMemset(d_buf, 0, N * sizeof(int)));

  cudaStream_t stream;
  cudaErrChk(cudaStreamCreate(&stream));
  const int blocks = (N + THREADS - 1) / THREADS;

  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cudaErrChk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  addKernel<<<blocks, THREADS, 0, stream>>>(d_buf, N, INCREMENT);
  cudaErrChk(cudaStreamEndCapture(stream, &graph));
  cudaErrChk(cudaGraphInstantiateWithFlags(&exec, graph, 0));

  // Captured + instantiated, zero launches: the "cold" checkpoint window.
  printf("graph-warmth READY captured unlaunched gate=%s\n", gate ? gate : "(none)");

  if (gate) {
    while (access(gate, F_OK) != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    printf("graph-warmth gate released, launching\n");
  }

  long long expected = 0;
  long long iter = 0;
  int h_buf[N];

  while (true) {
    cudaErrChk(cudaGraphLaunch(exec, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    expected += INCREMENT;
    iter++;

    cudaErrChk(cudaMemcpy(h_buf, d_buf, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
      if (h_buf[i] != expected) {
        fprintf(stderr,
                "graph-warmth MISMATCH iter=%lld index=%d value=%d expected=%lld\n",
                iter, i, h_buf[i], expected);
        exit(EXIT_FAILURE);
      }
    }

    printf("graph-warmth iter=%lld value=%lld OK\n", iter, expected);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}
