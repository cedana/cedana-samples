/**
  @file jit_modules.cu
  @brief Modules JIT-compiled from PTX images through the driver API, as Triton and NVRTC users do.

  Loads one PTX image twice with cuModuleLoadDataEx (two modules from identical bytes), a second,
  different image once, and the first image a third time as a library (cuLibraryLoadData, whose
  kernels are CUkernel handles). Every module has its own __device__ globals, found with
  cuModuleGetGlobal / cuLibraryGetGlobal and written from the host. The kernels use 64 KiB of
  dynamic shared memory, more than the 48 KiB default, so each launch needs the attribute set with
  cuFuncSetAttribute / cuKernelSetAttribute. An interceptor that shares the globals of identical
  images between modules, moves a module's globals on restore, does not replay a function's
  attributes (or replays them against the wrong handle), or loses a JIT image, computes wrong
  values or fails the launch.

  The driver API is reached through dlopen("libcuda.so.1"), so the binary links only libcudart.

  With no argument the work is 50 iterations. With a gate-file path as argv[1] it prints READY
  after the first iteration and iterates until that file exists, letting a test checkpoint and
  restore it mid-work, then create the gate; it then loads both images again and runs them once
  more. Every iteration checks every buffer, every launch counter and that every global is where
  it was: a difference prints MISMATCH. Prints PASS and exits 0; non-zero on failure.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Image A, kernel "step": every element += g_add; one launch adds 1 to g_calls. The dynamic shared
// memory is filled and read back, so the launch really uses it. (nvcc -ptx -arch=compute_70 of a
// small kernel, with .version lowered so that older drivers JIT it.)
#define PTX_BODY(NAME, UPDATE)                                     \
  ".version 7.0\n"                                                 \
  ".target sm_70\n"                                                \
  ".address_size 64\n"                                             \
  ".global .align 4 .u32 g_add;\n"                                 \
  ".global .align 4 .u32 g_calls;\n"                               \
  ".extern .shared .align 16 .b8 scratch[];\n"                     \
  ".visible .entry " NAME "(\n"                                    \
  "  .param .u64 p_buf,\n"                                         \
  "  .param .u32 p_n,\n"                                           \
  "  .param .u32 p_words\n"                                        \
  ")\n"                                                            \
  "{\n"                                                            \
  "  .reg .pred %p<6>;\n"                                          \
  "  .reg .b32 %r<32>;\n"                                          \
  "  .reg .b64 %rd<6>;\n"                                          \
  "  ld.param.u64 %rd2, [p_buf];\n"                                \
  "  ld.param.u32 %r12, [p_n];\n"                                  \
  "  ld.param.u32 %r13, [p_words];\n"                              \
  "  mov.u32 %r1, %tid.x;\n"                                       \
  "  setp.lt.u32 %p1, %r1, %r13;\n"                                \
  "  @%p1 bra L1;\n"                                               \
  "  bra.uni L3;\n"                                                \
  "L1:\n"                                                          \
  "  mov.u32 %r2, %ntid.x;\n"                                      \
  "  mov.u32 %r30, %r1;\n"                                         \
  "L2:\n"                                                          \
  "  shl.b32 %r14, %r30, 2;\n"                                     \
  "  mov.u32 %r15, scratch;\n"                                     \
  "  add.s32 %r16, %r15, %r14;\n"                                  \
  "  st.shared.u32 [%r16], %r30;\n"                                \
  "  add.s32 %r30, %r30, %r2;\n"                                   \
  "  setp.lt.u32 %p2, %r30, %r13;\n"                               \
  "  @%p2 bra L2;\n"                                               \
  "L3:\n"                                                          \
  "  mov.u32 %r5, %ntid.x;\n"                                      \
  "  bar.sync 0;\n"                                                \
  "  mov.u32 %r6, %ctaid.x;\n"                                     \
  "  mad.lo.s32 %r31, %r6, %r5, %r1;\n"                            \
  "  setp.ge.u32 %p3, %r31, %r12;\n"                               \
  "  @%p3 bra L6;\n"                                               \
  "  mov.u32 %r17, 1;\n"                                           \
  "  sub.s32 %r18, %r17, %r13;\n"                                  \
  "  shl.b32 %r19, %r13, 2;\n"                                     \
  "  mov.u32 %r20, scratch;\n"                                     \
  "  add.s32 %r21, %r20, %r19;\n"                                  \
  "  ld.global.u32 %r22, [g_add];\n"                               \
  "  add.s32 %r23, %r22, %r18;\n"                                  \
  "  ld.shared.u32 %r24, [%r21+-4];\n"                             \
  "  add.s32 %r8, %r23, %r24;\n"                                   \
  "  mov.u32 %r25, %nctaid.x;\n"                                   \
  "  mul.lo.s32 %r9, %r25, %r5;\n"                                 \
  "  cvta.to.global.u64 %rd1, %rd2;\n"                             \
  "L5:\n"                                                          \
  "  mul.wide.u32 %rd3, %r31, 4;\n"                                \
  "  add.s64 %rd4, %rd1, %rd3;\n"                                  \
  "  ld.global.u32 %r26, [%rd4];\n"                                \
  "  " UPDATE "\n"                                                 \
  "  st.global.u32 [%rd4], %r27;\n"                                \
  "  add.s32 %r31, %r31, %r9;\n"                                   \
  "  setp.lt.u32 %p4, %r31, %r12;\n"                               \
  "  @%p4 bra L5;\n"                                               \
  "L6:\n"                                                          \
  "  or.b32 %r28, %r6, %r1;\n"                                     \
  "  setp.ne.s32 %p5, %r28, 0;\n"                                  \
  "  @%p5 bra L8;\n"                                               \
  "  mov.u64 %rd5, g_calls;\n"                                     \
  "  red.global.add.u32 [%rd5], 1;\n"                              \
  "L8:\n"                                                          \
  "  ret;\n"                                                       \
  "}\n"

// Image A: x += g_add. Image B, kernel "mix": x = 3x + g_add.
static const char PTX_A[] = PTX_BODY("step", "add.s32 %r27, %r8, %r26;");
static const char PTX_B[] = PTX_BODY("mix", "mad.lo.s32 %r27, %r26, 3, %r8;");

// The driver API, from libcuda.so.1 (cuda.h maps names such as cuMemcpyHtoD to their _v2 symbols)
#define XSTR(x) STR(x)
#define STR(x) #x
#define DRIVER_FNS(X)                                                                          \
  X(cuGetErrorString) X(cuCtxGetDevice) X(cuModuleLoadDataEx) X(cuModuleGetFunction)           \
  X(cuModuleGetGlobal) X(cuModuleUnload) X(cuFuncSetAttribute) X(cuFuncGetAttribute)           \
  X(cuLaunchKernel) X(cuMemcpyHtoD) X(cuMemcpyDtoH) X(cuCtxSynchronize) X(cuLibraryLoadData) \
  X(cuLibraryGetKernel) X(cuLibraryGetGlobal) X(cuKernelSetAttribute) X(cuKernelGetAttribute) \
  X(cuLibraryUnload)
#define DECLARE(fn) static decltype(&fn) d_##fn;
DRIVER_FNS(DECLARE)

static int load_driver() {
  void *h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) {
    printf("FAIL dlopen libcuda.so.1: %s\n", dlerror());
    return 1;
  }
#define LOAD(fn)                                                \
  if (!(d_##fn = (decltype(&fn))dlsym(h, XSTR(fn)))) {          \
    printf("FAIL dlsym %s\n", XSTR(fn));                        \
    return 1;                                                   \
  }
  DRIVER_FNS(LOAD)
  return 0;
}

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

static int check(CUresult r, const char *what) {
  if (r != CUDA_SUCCESS) {
    const char *s = nullptr;
    if (!d_cuGetErrorString || d_cuGetErrorString(r, &s) != CUDA_SUCCESS || !s) s = "unknown";
    printf("FAIL %s: %d %s\n", what, (int)r, s);
    return 1;
  }
  return 0;
}

static const unsigned N = 1 << 14;           // elements per buffer
static const unsigned SMEM = 64 * 1024;      // dynamic shared memory per block, > the 48 KiB default
static const unsigned SMEM_WORDS = SMEM / 4;

// One kernel and its module's globals, plus a device buffer and its expected contents.
struct Work {
  const char *name;
  bool mix = false; // image B: x = 3x + add; image A: x += add
  CUfunction fn = nullptr;
  CUdeviceptr g_add = 0, g_calls = 0; // where the module's globals were when first looked up
  unsigned *buf = nullptr;
  std::vector<unsigned> expected = {};
  unsigned calls = 0; // launches so far
};

static CUresult load_module(const char *image, CUmodule *mod) {
  char log[4096] = {0};
  CUjit_option opts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                         CU_JIT_OPTIMIZATION_LEVEL};
  void *vals[] = {log, (void *)(uintptr_t)sizeof(log), (void *)(uintptr_t)3};
  CUresult r = d_cuModuleLoadDataEx(mod, image, 3, opts, vals);
  if (r != CUDA_SUCCESS && log[0]) printf("JIT log: %s\n", log);
  return r;
}

static int init_module_work(Work &w, CUmodule mod, const char *fn) {
  size_t bytes = 0;
  if (check(d_cuModuleGetFunction(&w.fn, mod, fn), "cuModuleGetFunction")) return 1;
  if (check(d_cuFuncSetAttribute(w.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM),
            "cuFuncSetAttribute"))
    return 1;
  if (check(d_cuModuleGetGlobal(&w.g_add, &bytes, mod, "g_add"), "cuModuleGetGlobal g_add")) return 1;
  if (check(d_cuModuleGetGlobal(&w.g_calls, &bytes, mod, "g_calls"), "cuModuleGetGlobal g_calls"))
    return 1;
  return 0;
}

static int init_buffer(Work &w, unsigned seed) {
  w.expected.resize(N);
  for (unsigned i = 0; i < N; i++) w.expected[i] = seed * 100003u + i;
  if (check(cudaMalloc(&w.buf, N * sizeof(unsigned)), "cudaMalloc")) return 1;
  if (check(cudaMemcpy(w.buf, w.expected.data(), N * sizeof(unsigned), cudaMemcpyHostToDevice),
            "cudaMemcpy to device"))
    return 1;
  unsigned zero = 0;
  if (check(d_cuMemcpyHtoD(w.g_calls, &zero, sizeof(zero)), "cuMemcpyHtoD g_calls")) return 1;
  w.calls = 0;
  return 0;
}

static int set_add(Work &w, unsigned add) {
  return check(d_cuMemcpyHtoD(w.g_add, &add, sizeof(add)), "cuMemcpyHtoD g_add");
}

// Launches w on its buffer and updates the expected contents for the given g_add
static int launch(Work &w, unsigned add) {
  void *buf = w.buf;
  unsigned n = N, words = SMEM_WORDS;
  void *args[] = {&buf, &n, &words};
  char what[64];
  snprintf(what, sizeof(what), "cuLaunchKernel %s", w.name);
  if (check(d_cuLaunchKernel(w.fn, 32, 1, 1, 256, 1, 1, SMEM, 0, args, nullptr), what)) return 1;
  for (unsigned i = 0; i < N; i++) w.expected[i] = w.mix ? 3u * w.expected[i] + add : w.expected[i] + add;
  w.calls++;
  return 0;
}

static int verify(Work &w) {
  std::vector<unsigned> got(N);
  if (check(cudaMemcpy(got.data(), w.buf, N * sizeof(unsigned), cudaMemcpyDeviceToHost),
            "cudaMemcpy from device"))
    return 2;
  for (unsigned i = 0; i < N; i++) {
    if (got[i] != w.expected[i]) {
      printf("MISMATCH %s buf[%u] = %u, expected %u\n", w.name, i, got[i], w.expected[i]);
      return 3;
    }
  }
  unsigned calls = 0;
  if (check(d_cuMemcpyDtoH(&calls, w.g_calls, sizeof(calls)), "cuMemcpyDtoH g_calls")) return 2;
  if (calls != w.calls) {
    printf("MISMATCH %s g_calls = %u, expected %u\n", w.name, calls, w.calls);
    return 3;
  }
  return 0;
}

// The module's globals must still be where they were first found
static int verify_globals(Work &w, CUmodule mod) {
  CUdeviceptr add = 0, calls = 0;
  size_t bytes = 0;
  if (check(d_cuModuleGetGlobal(&add, &bytes, mod, "g_add"), "cuModuleGetGlobal g_add")) return 2;
  if (check(d_cuModuleGetGlobal(&calls, &bytes, mod, "g_calls"), "cuModuleGetGlobal g_calls")) return 2;
  if (add != w.g_add || calls != w.g_calls) {
    printf("MISMATCH %s globals moved: g_add 0x%llx -> 0x%llx, g_calls 0x%llx -> 0x%llx\n", w.name,
           (unsigned long long)w.g_add, (unsigned long long)add, (unsigned long long)w.g_calls,
           (unsigned long long)calls);
    return 3;
  }
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaFree(0), "cudaFree(0)")) return 1; // makes the primary context current
  if (load_driver()) return 1;
  CUdevice dev;
  if (check(d_cuCtxGetDevice(&dev), "cuCtxGetDevice")) return 1;
  int optin = 0;
  if (check(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0),
            "cudaDeviceGetAttribute")) return 1;
  if (optin < (int)SMEM) {
    printf("FAIL device allows %d B of shared memory per block, %u needed\n", optin, SMEM);
    return 1;
  }

  // a1 and a2 from identical bytes; b from a different image; lib from image A again, as a library
  CUmodule mod_a1, mod_a2, mod_b;
  if (check(load_module(PTX_A, &mod_a1), "cuModuleLoadDataEx A (1)")) return 1;
  if (check(load_module(PTX_A, &mod_a2), "cuModuleLoadDataEx A (2)")) return 1;
  if (check(load_module(PTX_B, &mod_b), "cuModuleLoadDataEx B")) return 1;
  CUlibrary lib;
  if (check(d_cuLibraryLoadData(&lib, PTX_A, nullptr, nullptr, 0, nullptr, nullptr, 0),
            "cuLibraryLoadData A")) return 1;

  Work a1{"a1"}, a2{"a2"}, b{"b", true}, l{"lib"};
  if (init_module_work(a1, mod_a1, "step") || init_module_work(a2, mod_a2, "step") ||
      init_module_work(b, mod_b, "mix"))
    return 1;
  CUkernel kernel;
  size_t bytes = 0;
  if (check(d_cuLibraryGetKernel(&kernel, lib, "step"), "cuLibraryGetKernel")) return 1;
  if (check(d_cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM, kernel, dev),
            "cuKernelSetAttribute")) return 1;
  l.fn = (CUfunction)kernel; // a CUkernel launches through cuLaunchKernel as a CUfunction
  if (check(d_cuLibraryGetGlobal(&l.g_add, &bytes, lib, "g_add"), "cuLibraryGetGlobal g_add")) return 1;
  if (check(d_cuLibraryGetGlobal(&l.g_calls, &bytes, lib, "g_calls"), "cuLibraryGetGlobal g_calls"))
    return 1;
  if (init_buffer(a1, 1) || init_buffer(a2, 2) || init_buffer(b, 3) || init_buffer(l, 4)) return 1;
  if (a1.g_add == a2.g_add || a1.g_add == l.g_add) {
    printf("MISMATCH modules from one image share g_add at 0x%llx\n", (unsigned long long)a1.g_add);
    return 3;
  }
  // a2, b and lib keep their g_add; a1's changes every iteration
  if (set_add(a2, 7) || set_add(b, 11) || set_add(l, 13)) return 1;

  long iterations = 0;
  unsigned add1 = 0;
  while (true) {
    add1 = (unsigned)(iterations % 5) + 1;
    if (set_add(a1, add1)) return 2;
    if (launch(a1, add1) || launch(a2, 7) || launch(b, 11) || launch(l, 13)) return 2;
    if (check(d_cuCtxSynchronize(), "kernels")) return 2;
    iterations++;
    if (int rc = verify(a1)) return rc;
    if (int rc = verify(a2)) return rc;
    if (int rc = verify(b)) return rc;
    if (int rc = verify(l)) return rc;
    if (int rc = verify_globals(a1, mod_a1)) return rc;
    if (int rc = verify_globals(a2, mod_a2)) return rc;
    if (int rc = verify_globals(b, mod_b)) return rc;
    if (iterations == 1 && gate) printf("READY\n");
    if (gate ? access(gate, F_OK) == 0 : iterations >= 50) break;
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  printf("ok   %ld iterations\n", iterations);

  // The attributes must still be set
  int v = 0;
  if (check(d_cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, a1.fn),
            "cuFuncGetAttribute")) return 4;
  if (v != (int)SMEM) {
    printf("MISMATCH a1 max dynamic shared memory %d, expected %u\n", v, SMEM);
    return 3;
  }
  if (check(d_cuKernelGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel, dev),
            "cuKernelGetAttribute")) return 4;
  if (v != (int)SMEM) {
    printf("MISMATCH lib max dynamic shared memory %d, expected %u\n", v, SMEM);
    return 3;
  }
  printf("ok   attributes\n");

  // Both images loaded afresh (after a restore, when gated), next to the modules from before
  CUmodule mod_a3, mod_b2;
  if (check(load_module(PTX_A, &mod_a3), "cuModuleLoadDataEx A (3)")) return 4;
  if (check(load_module(PTX_B, &mod_b2), "cuModuleLoadDataEx B (2)")) return 4;
  Work a3{"a3"}, b2{"b2", true};
  if (init_module_work(a3, mod_a3, "step") || init_module_work(b2, mod_b2, "mix")) return 4;
  if (init_buffer(a3, 5) || init_buffer(b2, 6)) return 4;
  if (set_add(a3, 17) || set_add(b2, 19)) return 4;
  // a1's g_add still holds its value from the last iteration
  if (launch(a3, 17) || launch(b2, 19) || launch(a1, add1) || launch(b, 11)) return 4;
  if (check(d_cuCtxSynchronize(), "kernels after the fresh loads")) return 4;
  for (Work *w : {&a1, &a2, &b, &l, &a3, &b2})
    if (int rc = verify(*w)) return rc;
  printf("ok   fresh loads\n");

  for (Work *w : {&a1, &a2, &b, &l, &a3, &b2})
    if (check(cudaFree(w->buf), "cudaFree")) return 5;
  for (CUmodule m : {mod_a1, mod_a2, mod_b, mod_a3, mod_b2})
    if (check(d_cuModuleUnload(m), "cuModuleUnload")) return 5;
  if (check(d_cuLibraryUnload(lib), "cuLibraryUnload")) return 5;
  printf("PASS\n");
  return 0;
}
