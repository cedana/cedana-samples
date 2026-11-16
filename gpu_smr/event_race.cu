/**
  @file event_race.cu
  @brief Cross-thread event ordering, as vLLM's CPU KV offload uses it (CED-2567).

  Natively, once cudaEventRecord returns on one thread, a cudaStreamWaitEvent or
  cudaEventQuery on any other thread sees that record. An interceptor that posts
  records asynchronously per thread can break this: the other thread's wait binds
  to the event's previous record (already complete), or its query says "done"
  before the record has run.

  Two threads, every iteration:
   wait:  main queues DEPTH no-op kernels, a kernel that writes the iteration
          number, and records a reused event; it then tells the copy thread,
          which makes its own stream wait on the event and copies the value
          back. Anything but the iteration number means the wait was stale.
   query: the copy thread queues a 2 ms kernel that writes a flag, records a
          fresh event and tells main; main queries the event once. If the
          query says done, the flag must already hold the iteration number.

  Prints PASS at the end if neither check ever failed, FAIL otherwise. With a
  gate path as the argument it runs half the iterations, prints READY, waits
  for the gate file to exist and runs the rest, so a test can checkpoint and
  restore it in between.

  Environment: EVENT_RACE_ITERS (default 2000), EVENT_RACE_DEPTH (default 64).
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "cuda_error.cuh"

__global__ void nop() {}
__global__ void put(unsigned *p, unsigned v) { *p = v; }
__global__ void spinThenPut(unsigned *p, unsigned v, unsigned long long ns) {
  unsigned long long s, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(s));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - s < ns);
  *p = v;
}

static int envInt(const char *name, int fallback) {
  const char *v = getenv(name);
  return v != nullptr && atoi(v) > 0 ? atoi(v) : fallback;
}

static void waitForGate(const char *gate) {
  struct stat st;
  printf("READY\n");
  while (stat(gate, &st) != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL); // unbuffered: the job log must see each line promptly
  const char *gate = argc > 1 ? argv[1] : nullptr;
  const int iters = envInt("EVENT_RACE_ITERS", 2000);
  const int depth = envInt("EVENT_RACE_DEPTH", 64);
  const int gateAt = gate != nullptr ? iters / 2 : 0;

  cudaErrChk(cudaSetDevice(0));
  cudaStream_t a, b;
  cudaErrChk(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
  cudaErrChk(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
  cudaEvent_t done;
  cudaErrChk(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  unsigned *d = nullptr, *h = nullptr;
  cudaErrChk(cudaMalloc(&d, 2 * sizeof(unsigned)));
  cudaErrChk(cudaMemset(d, 0, 2 * sizeof(unsigned)));
  cudaErrChk(cudaHostAlloc(&h, sizeof(unsigned), 0));

  std::atomic<int> go{0}, ack{0}, qready{0};
  std::atomic<unsigned> waitBad{0}, queryBad{0}, queryEarly{0};
  cudaEvent_t qev = nullptr;

  std::thread copier([&] {
    cudaErrChk(cudaSetDevice(0));
    for (int it = 1; it <= iters; it++) {
      while (go.load() != it) {
        std::this_thread::yield();
      }
      // wait: must bind to the record main just made, not the previous one
      cudaErrChk(cudaStreamWaitEvent(b, done, 0));
      cudaErrChk(cudaMemcpyAsync(h, d, sizeof(unsigned), cudaMemcpyDeviceToHost, b));
      cudaErrChk(cudaStreamSynchronize(b));
      if (h[0] != (unsigned)it) waitBad++;
      // query: record behind a slow kernel, then hand the event to main
      cudaEvent_t e;
      cudaErrChk(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      spinThenPut<<<1, 1, 0, b>>>(d + 1, it, 2000000ull);
      cudaErrChk(cudaGetLastError());
      cudaErrChk(cudaEventRecord(e, b));
      qev = e;
      qready.store(it);
      ack.store(it);
    }
  });

  for (int it = 1; it <= iters; it++) {
    if (gateAt != 0 && it == gateAt + 1) waitForGate(gate);
    for (int i = 0; i < depth; i++) nop<<<1, 1, 0, a>>>();
    put<<<1, 1, 0, a>>>(d, it);
    cudaErrChk(cudaGetLastError());
    cudaErrChk(cudaEventRecord(done, a));
    go.store(it);
    while (qready.load() != it) {
      std::this_thread::yield();
    }
    cudaEvent_t e = qev;
    if (cudaEventQuery(e) == cudaSuccess) {
      queryEarly++;
      unsigned v = 0;
      cudaErrChk(cudaMemcpy(&v, d + 1, sizeof(unsigned), cudaMemcpyDeviceToHost));
      if (v != (unsigned)it) queryBad++;
    }
    cudaErrChk(cudaEventSynchronize(e));
    cudaErrChk(cudaEventDestroy(e));
    while (ack.load() != it) {
      std::this_thread::yield();
    }
    cudaErrChk(cudaStreamSynchronize(a));
    if (it % 500 == 0) printf("event_race step=%d\n", it);
  }
  copier.join();

  printf("event_race iterations=%d depth=%d stale_wait=%u query_done=%u query_early=%u\n",
         iters, depth, waitBad.load(), queryEarly.load(), queryBad.load());
  const bool race = waitBad.load() || queryBad.load();
  printf("%s\n", race ? "FAIL" : "PASS");
  return race ? EXIT_FAILURE : EXIT_SUCCESS;
}
