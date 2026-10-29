/**
  @file stream_race.cu
  @brief Cross-thread stream ordering and launches posted just before thread exit (CED-2567).

  Natively, once a launch returns on one thread, work another thread queues
  later on the same stream runs after it, and launches a thread queued before
  exiting still run. An interceptor that posts launches asynchronously per
  thread can break both: the other thread's call overtakes the posted launch,
  or the launches still posted when the thread exits are dropped.

   shared: main queues DEPTH no-op kernels and a kernel that writes the
           iteration number on a stream both threads use, then tells the copy
           thread, which copies the value back on that same stream. Anything
           but the iteration number means the copy overtook the kernel.
   exit:   a fresh thread queues a slow kernel, then as many no-op kernels as
           can be queued behind it without blocking (measured once up front),
           then a kernel that writes the iteration number, and exits without
           synchronizing; main joins it, synchronizes the stream and reads the
           value. Whoever executes the launches is still blocked behind the
           slow kernel when the thread exits, so anything but the iteration
           number means the last launches were lost. More iterations than the
           interceptor's 512 thread slots, so slots are reused.

  Prints PASS at the end if neither check ever failed, FAIL otherwise. With a
  gate path as the argument it runs half the shared iterations, prints READY,
  waits for the gate file to exist and runs the rest, so a test can checkpoint
  and restore it in between.

  Environment: STREAM_RACE_ITERS (default 2000), STREAM_RACE_DEPTH (default
  64), STREAM_RACE_EXIT_ITERS (default 600).
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
__global__ void spin(unsigned long long ns) {
  unsigned long long s, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(s));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - s < ns);
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
  const int iters = envInt("STREAM_RACE_ITERS", 2000);
  const int depth = envInt("STREAM_RACE_DEPTH", 64);
  const int exitIters = envInt("STREAM_RACE_EXIT_ITERS", 600);
  const int gateAt = gate != nullptr ? iters / 2 : 0;

  cudaErrChk(cudaSetDevice(0));
  cudaStream_t s;
  cudaErrChk(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  unsigned *d = nullptr, *h = nullptr;
  cudaErrChk(cudaMalloc(&d, 2 * sizeof(unsigned)));
  cudaErrChk(cudaMemset(d, 0, 2 * sizeof(unsigned)));
  cudaErrChk(cudaHostAlloc(&h, sizeof(unsigned), 0));

  std::atomic<int> go{0}, ack{0};
  std::atomic<unsigned> sharedBad{0};

  std::thread copier([&] {
    cudaErrChk(cudaSetDevice(0));
    for (int it = 1; it <= iters; it++) {
      while (go.load() != it) {
        std::this_thread::yield();
      }
      // must run behind the kernels main queued on s before telling us
      cudaErrChk(cudaMemcpyAsync(h, d, sizeof(unsigned), cudaMemcpyDeviceToHost, s));
      cudaErrChk(cudaStreamSynchronize(s));
      if (h[0] != (unsigned)it) sharedBad++;
      ack.store(it);
    }
  });

  for (int it = 1; it <= iters; it++) {
    if (gateAt != 0 && it == gateAt + 1) waitForGate(gate);
    for (int i = 0; i < depth; i++) nop<<<1, 1, 0, s>>>();
    put<<<1, 1, 0, s>>>(d, it);
    cudaErrChk(cudaGetLastError());
    go.store(it);
    while (ack.load() != it) {
      std::this_thread::yield();
    }
    if (it % 500 == 0) printf("stream_race shared step=%d shared_stale=%u\n", it, sharedBad.load());
  }
  copier.join();

  // How many launches a thread can queue behind a slow kernel before one blocks, and how long
  // queueing them takes: past the GPU's pending launch queue (plus anything an interceptor
  // buffers in front of it) the launch waits for the slow kernel to finish.
  int fill = 0;
  long long fillNs = 0;
  std::thread([&] {
    cudaErrChk(cudaSetDevice(0));
    spin<<<1, 1, 0, s>>>(2000000000ull);
    const auto start = std::chrono::steady_clock::now();
    while (fill < 1 << 16) {
      const auto t = std::chrono::steady_clock::now();
      nop<<<1, 1, 0, s>>>();
      fill++;
      if (std::chrono::steady_clock::now() - t > std::chrono::milliseconds(10)) break;
      fillNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - start)
                   .count();
    }
    cudaErrChk(cudaGetLastError());
  }).join();
  cudaErrChk(cudaStreamSynchronize(s));
  // Margin below the blocking launch, so all of them are queued while the slow kernel (three
  // times the queueing time) is still running.
  const int exitDepth = fill - 8;
  const unsigned long long spinNs = 3 * (unsigned long long)fillNs + 1000000ull;
  printf("stream_race exit fill=%d fill_ms=%.1f\n", fill, fillNs / 1e6);

  unsigned exitBad = 0;
  for (int it = 1; it <= exitIters; it++) {
    std::thread worker([&] {
      cudaErrChk(cudaSetDevice(0));
      // A kernel's first launch on a thread can need a synchronous call (an interceptor looking
      // up its parameters), which would wait out the slow kernel: launch each one once first.
      put<<<1, 1, 0, s>>>(d + 1, 0);
      spin<<<1, 1, 0, s>>>(spinNs);
      for (int i = 0; i < exitDepth; i++) nop<<<1, 1, 0, s>>>();
      put<<<1, 1, 0, s>>>(d + 1, it);
      cudaErrChk(cudaGetLastError());
    });
    worker.join();
    cudaErrChk(cudaStreamSynchronize(s));
    unsigned v = 0;
    cudaErrChk(cudaMemcpy(&v, d + 1, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (v != (unsigned)it) exitBad++;
    if (it % 100 == 0) printf("stream_race exit step=%d exit_lost=%u\n", it, exitBad);
  }

  printf("stream_race iterations=%d depth=%d shared_stale=%u exit_iterations=%d exit_depth=%d "
         "exit_lost=%u\n",
         iters, depth, sharedBad.load(), exitIters, exitDepth, exitBad);
  const bool race = sharedBad.load() || exitBad;
  printf("%s\n", race ? "FAIL" : "PASS");
  return race ? EXIT_FAILURE : EXIT_SUCCESS;
}
