/**
  @file reset_exit.cu
  @brief cudaDeviceReset() at startup and again just before exit, as GROMACS does.

  A device check that ends in cudaDeviceReset(), then work: a counter on the device that a
  kernel increments once per iteration, then cudaDeviceReset() again before the process exits.
  The runtime's global destructor runs after that second reset, so an interceptor that does not
  run the runtime's context-local storage destructors at a reset crashes the process at exit.

  With no argument the work is 100 iterations. With a gate-file path as argv[1] it prints READY
  after the first iteration and iterates until that file exists, letting a test checkpoint and
  restore it mid-work, then create the gate. The counter must equal the iteration count: a
  mismatch prints MISMATCH. Prints PASS after the second reset and exits 0; non-zero on failure.
 */

#include <chrono>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>

__global__ void increment(int *counter) { *counter += 1; }

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;
  int *d = nullptr;

  // The device check: a dummy kernel, then a reset.
  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaMalloc(&d, sizeof(int)), "cudaMalloc in the check")) return 1;
  increment<<<1, 1>>>(d);
  if (check(cudaDeviceSynchronize(), "kernel in the check")) return 1;
  if (check(cudaDeviceReset(), "cudaDeviceReset after the check")) return 1;
  printf("ok   startup reset\n");

  // The work.
  if (check(cudaMalloc(&d, sizeof(int)), "cudaMalloc")) return 2;
  if (check(cudaMemset(d, 0, sizeof(int)), "cudaMemset")) return 2;
  long iterations = 0;
  while (true) {
    increment<<<1, 1>>>(d);
    if (check(cudaDeviceSynchronize(), "kernel")) return 2;
    iterations++;
    if (iterations == 1 && gate) printf("READY\n");
    if (gate ? access(gate, F_OK) == 0 : iterations >= 100) break;
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  int counter = 0;
  if (check(cudaMemcpy(&counter, d, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy")) return 2;
  if (counter != iterations) {
    printf("MISMATCH counter %d, iterations %ld\n", counter, iterations);
    return 3;
  }
  printf("ok   %ld iterations\n", iterations);
  if (check(cudaFree(d), "cudaFree")) return 2;

  // The reset before exit; the runtime's global destructor runs after it.
  if (check(cudaDeviceReset(), "cudaDeviceReset before exit")) return 4;
  printf("PASS\n");
  return 0;
}
