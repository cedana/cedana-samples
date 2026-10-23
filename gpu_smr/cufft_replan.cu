/**
  @file cufft_replan.cu
  @brief cuFFT plans destroyed and created again before a checkpoint, as the PME tuning of
  GROMACS does.

  GROMACS does PME on the GPU with one pair of 3D cuFFT plans per PME grid: an R2C plan and a C2R
  plan from cufftPlanMany(). At the start of a run it tunes the grid, and at each change it
  destroys the plan pair and creates a new one. This sample does the same over the grid sizes of
  the PME tuning of a GROMACS run (44, 40, 36, 32, 28, then 42), with one forward and one inverse
  FFT per plan pair. A GPU interceptor that keeps a cuFFT module call in its checkpoint after the
  module of that call is unloaded replays the call with a stale module handle at the restore.

  With --control the sample makes one plan pair, for the last size only, and destroys nothing.

  Each round trip, R2C then C2R, must give the input times the number of grid points. The sample
  computes the input and the expected values on the CPU and compares every element.

  With no gate argument the sample does 100 round trips on the last plan pair. With a gate-file
  path it prints READY after the first round trip on the last plan pair and does round trips until
  that file exists, so a test can checkpoint and restore it mid-work, then create the gate. Prints
  one ok/FAIL line per step and PASS at the end. Exits 0 on PASS, 1 on a failure before the last
  plan pair, 2 on a failure after it, and 3 on a wrong FFT result.

  Usage: cufft_replan [--control] [gate-file]
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

static int checkFft(cufftResult r, const char *what) {
  if (r != CUFFT_SUCCESS) {
    printf("FAIL %s: cufftResult %d\n", what, (int)r);
    return 1;
  }
  return 0;
}

// One PME grid: a cubic real grid of n^3 points, with the R2C and C2R plans of GROMACS.
struct Grid {
  int n = 0;
  cufftHandle r2c = 0;
  cufftHandle c2r = 0;
  cufftReal *real = nullptr;
  cufftComplex *complex = nullptr;
  std::vector<float> input;

  size_t realCount() const { return (size_t)n * n * n; }
  size_t complexCount() const { return (size_t)n * n * (n / 2 + 1); }
};

static int createGrid(Grid &g, int n, cudaStream_t stream) {
  g.n = n;
  int dims[3] = {n, n, n};
  int realEmbed[3] = {n, n, n};
  int complexEmbed[3] = {n, n, n / 2 + 1};
  char what[64];
  snprintf(what, sizeof(what), "cufftPlanMany R2C %d", n);
  if (checkFft(cufftPlanMany(&g.r2c, 3, dims, realEmbed, 1, (int)g.realCount(), complexEmbed, 1,
                             (int)g.complexCount(), CUFFT_R2C, 1),
               what))
    return 1;
  snprintf(what, sizeof(what), "cufftPlanMany C2R %d", n);
  if (checkFft(cufftPlanMany(&g.c2r, 3, dims, complexEmbed, 1, (int)g.complexCount(), realEmbed,
                             1, (int)g.realCount(), CUFFT_C2R, 1),
               what))
    return 1;
  if (checkFft(cufftSetStream(g.r2c, stream), "cufftSetStream R2C")) return 1;
  if (checkFft(cufftSetStream(g.c2r, stream), "cufftSetStream C2R")) return 1;
  if (check(cudaMalloc(&g.real, g.realCount() * sizeof(cufftReal)), "cudaMalloc real grid"))
    return 1;
  if (check(cudaMalloc(&g.complex, g.complexCount() * sizeof(cufftComplex)),
            "cudaMalloc complex grid"))
    return 1;
  g.input.resize(g.realCount());
  for (size_t i = 0; i < g.input.size(); i++) {
    // Small integers: the round trip stays exact enough to compare with a tight tolerance.
    g.input[i] = (float)((i * 7 + n) % 13) - 6.0f;
  }
  return 0;
}

static int destroyGrid(Grid &g) {
  char what[64];
  snprintf(what, sizeof(what), "cufftDestroy R2C %d", g.n);
  if (checkFft(cufftDestroy(g.r2c), what)) return 1;
  snprintf(what, sizeof(what), "cufftDestroy C2R %d", g.n);
  if (checkFft(cufftDestroy(g.c2r), what)) return 1;
  if (check(cudaFree(g.real), "cudaFree real grid")) return 1;
  if (check(cudaFree(g.complex), "cudaFree complex grid")) return 1;
  g = Grid{};
  return 0;
}

// R2C then C2R on the grid; returns 0 when every element equals input * n^3.
static int roundTrip(Grid &g, cudaStream_t stream, std::vector<float> &out) {
  size_t bytes = g.realCount() * sizeof(cufftReal);
  if (check(cudaMemcpyAsync(g.real, g.input.data(), bytes, cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync input"))
    return 1;
  if (checkFft(cufftExecR2C(g.r2c, g.real, g.complex), "cufftExecR2C")) return 1;
  if (checkFft(cufftExecC2R(g.c2r, g.complex, g.real), "cufftExecC2R")) return 1;
  out.resize(g.realCount());
  if (check(cudaMemcpyAsync(out.data(), g.real, bytes, cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync output"))
    return 1;
  if (check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) return 1;
  const double scale = (double)g.realCount();
  for (size_t i = 0; i < out.size(); i++) {
    double want = g.input[i] * scale;
    if (std::fabs(out[i] - want) > 1e-3 * scale) {
      printf("MISMATCH grid %d element %zu: %f, want %f\n", g.n, i, out[i], want);
      return 3;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  bool control = false;
  const char *gate = nullptr;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--control") == 0) {
      control = true;
    } else {
      gate = argv[i];
    }
  }

  const std::vector<int> tuning = {44, 40, 36, 32, 28};
  const int last = 42;
  cudaStream_t stream;
  std::vector<float> out;
  Grid g;

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaStreamCreate(&stream), "cudaStreamCreate")) return 1;
  printf("ok   mode %s\n", control ? "control" : "replan");

  if (!control) {
    for (int n : tuning) {
      if (createGrid(g, n, stream)) return 1;
      int r = roundTrip(g, stream, out);
      if (r) return r == 3 ? 3 : 1;
      if (destroyGrid(g)) return 1;
      printf("ok   grid %d planned, round trip, destroyed\n", n);
    }
  }

  if (createGrid(g, last, stream)) return 2;
  long iterations = 0;
  while (true) {
    int r = roundTrip(g, stream, out);
    if (r) return r == 3 ? 3 : 2;
    iterations++;
    if (iterations == 1) {
      printf("ok   grid %d planned, round trip\n", last);
      if (gate) printf("READY\n");
    }
    if (gate ? access(gate, F_OK) == 0 : iterations >= 100) break;
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  printf("ok   %ld round trips on grid %d\n", iterations, last);
  if (destroyGrid(g)) return 2;
  if (check(cudaStreamDestroy(stream), "cudaStreamDestroy")) return 2;
  printf("PASS\n");
  return 0;
}
