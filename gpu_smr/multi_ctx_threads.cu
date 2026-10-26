/**
  @file multi_ctx_threads.cu
  @brief Two contexts on one device, each used by its own thread.

  A process with one thread per GPU uses its devices this way. Thread T makes a created context B
  current and runs kernels on it. The main thread uses the primary context P. cudaDeviceReset() on
  the main thread ends P only, so T must still use B after it. Then cuCtxDestroy(B) ends B only,
  so P must still work. cuda.h makes cuCtxDestroy cuCtxDestroy_v2, which the binary imports at
  link time, as ctx_destroy does.

  KEEP_T=1 keeps T alive until the main thread is done. NO_RESET=1 leaves out both resets, for an
  interceptor that does not yet set up the runtime again after a reset. Prints one ok/FAIL line per
  step and PASS at the end. Exits 0 on PASS, else with the part that failed: 1 the setup, 2 the
  reset of P, 3 thread T, 4 P after the reset, 5 the destroy of B or P after it, 6 the last reset.
 */

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <thread>

#if CUDA_VERSION >= 13000
// CUDA 13 makes cuCtxCreate cuCtxCreate_v4, with creation parameters. The driver still exports
// the CUDA 12 form, which an interceptor is more likely to hook.
extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define createContext cuCtxCreate_v2
#else
#define createContext cuCtxCreate
#endif

#define N 32

__global__ void fill(int *p, int v) { p[threadIdx.x] = v; }

static std::mutex m;
static std::condition_variable cv;
static int stage = 0; // 1: T used B, 2: main reset P, 3: T used B again, 4: main done
static bool keep_t = getenv("KEEP_T") != nullptr;
static bool no_reset = getenv("NO_RESET") != nullptr;
static int failed = 0;

static void wait_for(int s) {
  std::unique_lock<std::mutex> lock(m);
  cv.wait(lock, [s] { return stage >= s || failed; });
}

static void set_stage(int s) {
  {
    std::lock_guard<std::mutex> lock(m);
    stage = s;
  }
  cv.notify_all();
}

static void fail() {
  {
    std::lock_guard<std::mutex> lock(m);
    failed = 1;
  }
  cv.notify_all();
}

static int rt(cudaError_t e, const char *who, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL [%s] %s: %d %s\n", who, what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  printf("ok   [%s] %s\n", who, what);
  return 0;
}

static int drv(CUresult e, const char *who, const char *what) {
  if (e != CUDA_SUCCESS) {
    const char *s = nullptr;
    cuGetErrorString(e, &s);
    printf("FAIL [%s] %s: %d %s\n", who, what, (int)e, s ? s : "?");
    return 1;
  }
  printf("ok   [%s] %s\n", who, what);
  return 0;
}

// A kernel that writes v, a copy back, and a check of the value.
static int use(int *p, int v, const char *who, const char *what) {
  int h[N] = {0};
  fill<<<1, N>>>(p, v);
  if (rt(cudaGetLastError(), who, what)) return 1;
  if (rt(cudaMemcpy(h, p, sizeof(h), cudaMemcpyDeviceToHost), who, what)) return 1;
  if (h[N - 1] != v) {
    printf("FAIL [%s] %s: value %d, want %d\n", who, what, h[N - 1], v);
    return 1;
  }
  return 0;
}

static void thread_b(CUcontext b) {
  int *db = nullptr;
  if (drv(cuCtxSetCurrent(b), "T", "cuCtxSetCurrent B") ||
      rt(cudaMalloc(&db, N * sizeof(int)), "T", "cudaMalloc on B") ||
      use(db, 1, "T", "kernel on B")) {
    fail();
    return;
  }
  set_stage(1);
  wait_for(2);
  if (failed) return;
  if (use(db, 3, "T", "kernel on B after the reset of P") ||
      rt(cudaFree(db), "T", "cudaFree on B")) {
    fail();
    return;
  }
  set_stage(3);
  if (keep_t) wait_for(4);
}

int main() {
  setbuf(stdout, NULL);
  printf("keep_t=%d no_reset=%d\n", keep_t, no_reset);
  CUdevice dev;
  CUcontext b, popped;
  int *dp = nullptr;

  if (drv(cuInit(0), "main", "cuInit")) return 1;
  if (drv(cuDeviceGet(&dev, 0), "main", "cuDeviceGet")) return 1;
  if (drv(createContext(&b, 0, dev), "main", "cuCtxCreate B")) return 1;
  if (drv(cuCtxPopCurrent(&popped), "main", "cuCtxPopCurrent B")) return 1;
  std::thread t(thread_b, b);
  wait_for(1);
  if (failed) {
    t.join();
    return 1;
  }
  if (rt(cudaSetDevice(0), "main", "cudaSetDevice 0, P")) return 1;
  if (rt(cudaMalloc(&dp, N * sizeof(int)), "main", "cudaMalloc on P")) return 1;
  if (use(dp, 2, "main", "kernel on P")) return 1;
  if (!no_reset && rt(cudaDeviceReset(), "main", "cudaDeviceReset, ends P")) {
    fail();
    t.join();
    return 2;
  }
  set_stage(2);
  wait_for(3);
  if (!keep_t) t.join();
  if (failed) return 3;
  if (rt(cudaMalloc(&dp, N * sizeof(int)), "main", "cudaMalloc on P after the reset")) return 4;
  if (use(dp, 4, "main", "kernel on P after the reset")) return 4;
  if (drv(cuCtxDestroy(b), "main", "cuCtxDestroy B")) return 5;
  if (use(dp, 5, "main", "kernel on P after the destroy of B")) return 5;
  set_stage(4);
  if (keep_t) t.join();
  if (!no_reset && rt(cudaDeviceReset(), "main", "cudaDeviceReset before exit")) return 6;
  printf("PASS\n");
  return 0;
}
