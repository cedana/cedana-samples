/**
  @file cuda_graph_events.cu
  @brief cuda_graph_loop plus CUDA event lifecycle stress across C/R.

  Adds three things on top of the self-validating loop:
    - graph-node events: cudaEventRecord() into the capturing stream (evStart/evEnd).
    - teardown/rebuild every REBUILD_EVERY iters: destroys the graph, exec, and its
      graph-node events, then rebuilds -- exercising graph-node event destroy.
    - event-pool churn: per-iter throwaway timing events created/recorded/destroyed.

  Drift prints MISMATCH and exits non-zero.
 */

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#include "cuda_error.cuh"

#define N 4096
#define INCREMENT 1
#define THREADS 256
#define REBUILD_EVERY 10 // rebuild graph + graph-node events this often

__global__ void addKernel(int *buf, int n, int increment) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] += increment;
}

static int *d_buf = nullptr;
static cudaStream_t stream;
static cudaGraph_t graph;
static cudaGraphExec_t exec;
static cudaEvent_t evStart, evEnd; // graph-node events (recorded during capture)

void buildGraph() {
  const int blocks = (N + THREADS - 1) / THREADS;
  cudaErrChk(cudaEventCreate(&evStart));
  cudaErrChk(cudaEventCreate(&evEnd));
  cudaErrChk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  cudaErrChk(cudaEventRecord(evStart, stream)); // becomes a graph node
  addKernel<<<blocks, THREADS, 0, stream>>>(d_buf, N, INCREMENT);
  cudaErrChk(cudaEventRecord(evEnd, stream));    // becomes a graph node
  cudaErrChk(cudaStreamEndCapture(stream, &graph));
  cudaErrChk(cudaGraphInstantiateWithFlags(&exec, graph, 0));
}

void teardownGraph() {
  // Destroy exec and graph before the events they reference, then the events.
  cudaErrChk(cudaGraphExecDestroy(exec));
  cudaErrChk(cudaGraphDestroy(graph));
  cudaErrChk(cudaEventDestroy(evStart));
  cudaErrChk(cudaEventDestroy(evEnd));
}

void handleSignal(int signal) { exit(signal); }

int main() {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);
  setbuf(stdout, NULL);

  cudaErrChk(cudaMalloc(&d_buf, N * sizeof(int)));
  cudaErrChk(cudaMemset(d_buf, 0, N * sizeof(int)));
  cudaErrChk(cudaStreamCreate(&stream));
  buildGraph();

  long long expected = 0;
  long long iter = 0;
  int h_buf[N];

  while (true) {
    // Off-capture throwaway events: create/record/query/destroy -> pool churn.
    cudaEvent_t t0, t1;
    cudaErrChk(cudaEventCreate(&t0));
    cudaErrChk(cudaEventCreate(&t1));

    cudaErrChk(cudaEventRecord(t0, stream));
    cudaErrChk(cudaGraphLaunch(exec, stream));
    cudaErrChk(cudaEventRecord(t1, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    expected += INCREMENT;
    iter++;

    (void)cudaEventQuery(t0); // just exercise the query path
    float wallMs = 0.0f;
    cudaErrChk(cudaEventElapsedTime(&wallMs, t0, t1)); // ordinary (non-graph) events

    cudaErrChk(cudaEventDestroy(t0)); // recycled through the pool
    cudaErrChk(cudaEventDestroy(t1));

    // Validate the whole buffer against the host shadow.
    cudaErrChk(cudaMemcpy(h_buf, d_buf, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
      if (h_buf[i] != expected) {
        fprintf(stderr,
                "graph-events MISMATCH iter=%lld index=%d value=%d expected=%lld\n",
                iter, i, h_buf[i], expected);
        exit(EXIT_FAILURE);
      }
    }

    printf("graph-events iter=%lld value=%lld wall_ms=%.3f OK\n", iter, expected,
           wallMs);

    // Periodically destroy and rebuild the graph and its graph-node events.
    if (iter % REBUILD_EVERY == 0) {
      teardownGraph();
      buildGraph();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}
