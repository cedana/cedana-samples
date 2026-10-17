/**
  @file deep_queue.cu
  @brief Keeps seconds of kernels queued on the legacy default stream, for C/R testing.

  A GPU-bound PyTorch job (ESMFold, for one) runs ahead of the GPU, so at any
  moment the default stream can hold more queued work than a freeze waits for.
  This reproduces that: it keeps DEPTH kernels of KERNEL_MS each in flight on
  stream 0, about 2 s with the defaults.

  The host never blocks in a CUDA call. It polls the event of the oldest kernel
  with cudaEventQuery and sleeps 1 ms between polls. A thread inside a
  synchronizing call holds its freeze barrier for the whole call, which would
  test the barrier step of a freeze instead of the wait for the GPU queue.

  Each kernel writes its sequence number to device memory, and an async copy
  behind it moves the value to a pinned host slot. The host checks the slot
  after the event completes, with no CUDA call that waits for the queue. A
  completion out of order, or a gap after a restore, prints MISMATCH and exits
  non-zero.

  Environment: DEEP_QUEUE_DEPTH (default 20), DEEP_QUEUE_KERNEL_MS (default 100).
 */

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#include "cuda_error.cuh"

#define MAX_DEPTH 256

// Spin for `cycles` GPU clock cycles, then record this kernel's sequence number
__global__ void spinKernel(long long cycles, long long *done, long long seq) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
  *done = seq;
}

static int envInt(const char *name, int fallback) {
  const char *v = getenv(name);
  return v != nullptr && atoi(v) > 0 ? atoi(v) : fallback;
}

void handleSignal(int signal) { exit(signal); }

int main() {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);
  setbuf(stdout, NULL); // unbuffered: the job log must see each line promptly

  int depth = envInt("DEEP_QUEUE_DEPTH", 20);
  if (depth > MAX_DEPTH) depth = MAX_DEPTH;
  const int kernelMs = envInt("DEEP_QUEUE_KERNEL_MS", 100);

  int clockKHz = 0;
  cudaErrChk(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0));
  const long long cycles = static_cast<long long>(clockKHz) * kernelMs;

  long long *d_done = nullptr;
  cudaErrChk(cudaMalloc(&d_done, sizeof(long long)));
  cudaErrChk(cudaMemset(d_done, 0, sizeof(long long)));
  long long *h_done = nullptr; // one pinned slot per ring entry
  cudaErrChk(cudaMallocHost(&h_done, MAX_DEPTH * sizeof(long long)));

  cudaEvent_t events[MAX_DEPTH];
  long long seqs[MAX_DEPTH];
  for (int i = 0; i < depth; i++) {
    cudaErrChk(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
  }

  printf("deep_queue depth=%d kernel_ms=%d queued_ms=%d\n", depth, kernelMs,
         depth * kernelMs);

  long long launched = 0;  // sequence number of the newest kernel
  long long completed = 0; // sequence number of the newest completion seen
  int head = 0;            // ring slot of the oldest kernel in flight
  int inflight = 0;

  while (true) {
    if (inflight < depth) {
      const int slot = (head + inflight) % depth;
      launched++;
      h_done[slot] = 0;
      spinKernel<<<1, 1>>>(cycles, d_done, launched); // stream 0, legacy
      cudaErrChk(cudaGetLastError());
      cudaErrChk(cudaMemcpyAsync(&h_done[slot], d_done, sizeof(long long),
                                 cudaMemcpyDeviceToHost, 0));
      cudaErrChk(cudaEventRecord(events[slot], 0));
      seqs[slot] = launched;
      inflight++;
      continue;
    }

    const cudaError_t q = cudaEventQuery(events[head]);
    if (q == cudaErrorNotReady) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    cudaErrChk(q);

    // Stream order: the copy behind the oldest kernel holds its own number
    const long long seq = seqs[head];
    const long long seen = h_done[head];
    if (seen != seq || seq != completed + 1) {
      fprintf(stderr, "deep_queue MISMATCH seq=%lld seen=%lld completed=%lld\n",
              seq, seen, completed);
      exit(EXIT_FAILURE);
    }
    completed = seq;
    head = (head + 1) % depth;
    inflight--;
    printf("deep_queue step=%lld OK\n", completed);
  }
}
