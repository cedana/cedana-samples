/**
  @file cuda_graph_loop.cu
  @brief Self-validating stream-captured CUDA graph in a loop, for C/R testing.

  Each graph launch adds INCREMENT to every buffer element, so after K launches
  every element must equal K*INCREMENT. A host counter shadows this; buffer and
  counter are both process state, so a correct C/R keeps them in lockstep. Drift
  prints MISMATCH and exits non-zero.
 */

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#include "cuda_error.cuh"

#define N 4096            // elements in the buffer
#define INCREMENT 1       // added to every element per graph launch
#define THREADS 256

__global__ void addKernel(int *buf, int n, int increment) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] += increment;
}

void handleSignal(int signal) { exit(signal); }

int main() {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);
  setbuf(stdout, NULL); // unbuffered: the job log must see each line promptly

  int *d_buf = nullptr;
  cudaErrChk(cudaMalloc(&d_buf, N * sizeof(int)));
  cudaErrChk(cudaMemset(d_buf, 0, N * sizeof(int)));

  cudaStream_t stream;
  cudaErrChk(cudaStreamCreate(&stream));

  const int blocks = (N + THREADS - 1) / THREADS;

  // Capture a single-kernel graph once, reuse the instantiated exec every launch.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cudaErrChk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  addKernel<<<blocks, THREADS, 0, stream>>>(d_buf, N, INCREMENT);
  cudaErrChk(cudaStreamEndCapture(stream, &graph));
  cudaErrChk(cudaGraphInstantiate(&exec, graph, 0));

  long long expected = 0; // host shadow of every buffer element
  long long iter = 0;
  int h_buf[N];

  while (true) {
    cudaErrChk(cudaGraphLaunch(exec, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    expected += INCREMENT;
    iter++;

    // Validate: read back and confirm the whole buffer tracks the host shadow.
    cudaErrChk(cudaMemcpy(h_buf, d_buf, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
      if (h_buf[i] != expected) {
        fprintf(stderr,
                "graph MISMATCH iter=%lld index=%d value=%d expected=%lld\n",
                iter, i, h_buf[i], expected);
        exit(EXIT_FAILURE);
      }
    }

    printf("graph iter=%lld value=%lld OK\n", iter, expected);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}
