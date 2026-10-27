/**
  @file ctx_destroy.cu
  @brief A created context, destroyed through the driver API, on one thread.

  Calls cuInit, cuDeviceGet, cuCtxCreate and cuCtxDestroy, and nothing of the runtime. cuda.h makes
  cuCtxDestroy cuCtxDestroy_v2, so the binary imports that name at link time, and the dynamic
  loader binds it at startup. A GPU interceptor that does not export cuCtxDestroy_v2 leaves the
  name to the real libcuda.so.1, which the process never initialized, and the destroy fails or
  crashes.

  With a gate-file path as argv[1] it prints READY after cuCtxCreate and waits until that file
  exists before it destroys the context, letting a test checkpoint and restore it with the context
  live. Prints one ok/FAIL line per step and PASS at the end; exits 0 on PASS, 1 before the
  destroy, 2 at it.
 */

#include <chrono>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <cuda.h>

#if CUDA_VERSION >= 13000
// CUDA 13 makes cuCtxCreate cuCtxCreate_v4, with creation parameters. The driver still exports
// the CUDA 12 form, which an interceptor is more likely to hook.
extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define createContext cuCtxCreate_v2
#else
#define createContext cuCtxCreate
#endif

static int check(CUresult e, const char *what) {
  if (e != CUDA_SUCCESS) {
    const char *s = nullptr;
    cuGetErrorString(e, &s);
    printf("FAIL %s: %d %s\n", what, (int)e, s ? s : "?");
    return 1;
  }
  printf("ok   %s\n", what);
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;
  CUdevice dev;
  CUcontext c;

  if (check(cuInit(0), "cuInit")) return 1;
  if (check(cuDeviceGet(&dev, 0), "cuDeviceGet")) return 1;
  if (check(createContext(&c, 0, dev), "cuCtxCreate")) return 1;
  if (gate) {
    printf("READY\n");
    while (access(gate, F_OK) != 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (check(cuCtxDestroy(c), "cuCtxDestroy")) return 2;
  printf("PASS\n");
  return 0;
}
