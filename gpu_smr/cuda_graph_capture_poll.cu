/**
  @file cuda_graph_capture_poll.cu
  @brief Self-validating captured CUDA graph whose capture is polled, for C/R testing.

  Frameworks ask a stream for its capture status far more often than they add
  nodes to it (PyTorch does before most operations), so a checkpoint records
  long runs of identical capture-info queries. This workload does the same:
  between each of the STAGES kernels it captures, it queries the capture
  POLLS times, and it checks that every query names the node added just
  before it. Each graph launch adds STAGES*INCREMENT to every buffer element.
  A host counter shadows this; buffer and counter are both process state, so a
  correct C/R keeps them in lockstep. Drift prints MISMATCH and exits non-zero.

  With a gate-file argument, the first launch waits for that file: a
  checkpoint taken before then restores a graph that has never been launched.
 */

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>

#include "cuda_error.cuh"

#define N 4096      // elements in the buffer
#define INCREMENT 1 // added to every element per kernel
#define THREADS 256
#define STAGES 8    // kernels captured, each depending on the one before
#define POLLS 2000  // capture-info queries after each captured kernel

__global__ void addKernel(int *buf, int n, int increment) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] += increment;
}

void handleSignal(int signal) { exit(signal); }

int main(int argc, char **argv) {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);
  setbuf(stdout, NULL); // unbuffered: the job log must see each line promptly
  const char *gate = argc > 1 ? argv[1] : nullptr;

  int *d_buf = nullptr;
  cudaErrChk(cudaMalloc(&d_buf, N * sizeof(int)));
  cudaErrChk(cudaMemset(d_buf, 0, N * sizeof(int)));

  cudaStream_t stream;
  cudaErrChk(cudaStreamCreate(&stream));

  const int blocks = (N + THREADS - 1) / THREADS;

  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cudaErrChk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  cudaGraphNode_t lastNode = nullptr;
  for (int stage = 0; stage < STAGES; stage++) {
    addKernel<<<blocks, THREADS, 0, stream>>>(d_buf, N, INCREMENT);
    cudaGraphNode_t stageNode = nullptr;
    for (int poll = 0; poll < POLLS; poll++) {
      cudaStreamCaptureStatus status;
      unsigned long long id = 0;
      cudaGraph_t capturing = nullptr;
      const cudaGraphNode_t *deps = nullptr;
      size_t numDeps = 0;
#if CUDART_VERSION >= 13000
      cudaErrChk(cudaStreamGetCaptureInfo(stream, &status, &id, &capturing, &deps, nullptr,
                                          &numDeps));
#else
      cudaErrChk(cudaStreamGetCaptureInfo(stream, &status, &id, &capturing, &deps, &numDeps));
#endif
      // The stream's only dependency is the kernel just captured: a new node each stage,
      // the same node for every query within it
      if (status != cudaStreamCaptureStatusActive || numDeps != 1 || deps[0] == lastNode ||
          (poll > 0 && deps[0] != stageNode)) {
        fprintf(stderr, "capture MISMATCH stage=%d poll=%d status=%d deps=%zu\n", stage, poll,
                (int)status, numDeps);
        exit(EXIT_FAILURE);
      }
      stageNode = deps[0];
    }
    lastNode = stageNode;
  }
  cudaErrChk(cudaStreamEndCapture(stream, &graph));
  cudaErrChk(cudaGraphInstantiateWithFlags(&exec, graph, 0));
  printf("graph captured: %d kernels, %d capture queries\n", STAGES, STAGES * POLLS);

  while (gate != nullptr && access(gate, F_OK) != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  long long expected = 0; // host shadow of every buffer element
  long long iter = 0;
  int h_buf[N];

  while (true) {
    cudaErrChk(cudaGraphLaunch(exec, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    expected += STAGES * INCREMENT;
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
