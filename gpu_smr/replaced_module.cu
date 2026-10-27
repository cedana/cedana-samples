/**
  @file replaced_module.cu
  @brief Modules whose image file is replaced under the same path after they're loaded, as when a
  late rank's JIT (Triton, a kernel cache) rewrites a file other ranks loaded or mapped.

  Writes PTX image A to two files: loads one by path (cuModuleLoad), and maps the other and loads
  the image from the mapping (cuModuleLoadData), as an app that maps a cubin does. Both files are
  then replaced, by rename, with image B: the same kernel name, computing something else. The path
  is loaded again, now B. So the path names A for two modules and B for a third, and the mapping
  (of the replaced file) still holds A. An interceptor that keys images by path, or that reopens a
  path or a mapped file on restore instead of keeping the image a module was loaded from, runs B
  where A was loaded (or A where B was), and computes wrong values.

  The driver API is reached through dlopen("libcuda.so.1"), so the binary links only libcudart.

  With no argument the work is 50 iterations. With a gate-file path as argv[1] it prints READY
  after the first iteration and iterates until that file exists, letting a test checkpoint and
  restore it mid-work, then create the gate. Every iteration first checks that every buffer still
  holds what the previous iteration left (so a checkpoint between iterations that loses one is
  caught), and checks them again after its launches. After the last iteration it loads the path
  (B) and the mapping (A) once more, and runs both. A difference prints MISMATCH. Prints PASS and
  exits 0; non-zero on failure.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel "step" over buf[0..n): image A does x += 1, image B does x = 3x + 5
#define PTX_BODY(UPDATE)                                    \
  ".version 7.0\n"                                          \
  ".target sm_70\n"                                         \
  ".address_size 64\n"                                      \
  ".visible .entry step(\n"                                 \
  "  .param .u64 p_buf,\n"                                  \
  "  .param .u32 p_n\n"                                     \
  ")\n"                                                     \
  "{\n"                                                     \
  "  .reg .pred %p<2>;\n"                                   \
  "  .reg .b32 %r<8>;\n"                                    \
  "  .reg .b64 %rd<5>;\n"                                   \
  "  ld.param.u64 %rd1, [p_buf];\n"                         \
  "  ld.param.u32 %r1, [p_n];\n"                            \
  "  mov.u32 %r2, %ctaid.x;\n"                              \
  "  mov.u32 %r3, %ntid.x;\n"                               \
  "  mov.u32 %r4, %tid.x;\n"                                \
  "  mad.lo.s32 %r5, %r2, %r3, %r4;\n"                      \
  "  setp.ge.u32 %p1, %r5, %r1;\n"                          \
  "  @%p1 bra DONE;\n"                                      \
  "  cvta.to.global.u64 %rd2, %rd1;\n"                      \
  "  mul.wide.u32 %rd3, %r5, 4;\n"                          \
  "  add.s64 %rd4, %rd2, %rd3;\n"                           \
  "  ld.global.u32 %r6, [%rd4];\n"                          \
  "  " UPDATE "\n"                                          \
  "  st.global.u32 [%rd4], %r7;\n"                          \
  "DONE:\n"                                                 \
  "  ret;\n"                                                \
  "}\n"

static const char PTX_A[] = PTX_BODY("add.s32 %r7, %r6, 1;");
static const char PTX_B[] = PTX_BODY("mad.lo.s32 %r7, %r6, 3, 5;");

// The driver API, from libcuda.so.1
#define XSTR(x) STR(x)
#define STR(x) #x
#define DRIVER_FNS(X) \
  X(cuGetErrorString) X(cuModuleLoad) X(cuModuleLoadData) X(cuModuleGetFunction) X(cuLaunchKernel) X(cuCtxSynchronize)
#define DECLARE(fn) static decltype(&fn) d_##fn;
DRIVER_FNS(DECLARE)

static int load_driver() {
  void *h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) {
    printf("FAIL dlopen libcuda.so.1: %s\n", dlerror());
    return 1;
  }
#define LOAD(fn)                                       \
  if (!(d_##fn = (decltype(&fn))dlsym(h, XSTR(fn)))) { \
    printf("FAIL dlsym %s\n", XSTR(fn));               \
    return 1;                                          \
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

// Writes an image (with its terminating NUL, as the driver reads PTX up to it) to path, replacing
// any file there by rename, as a cache that rewrites an entry does
static int write_image(const std::string &path, const char *image) {
  std::string tmp = path + ".tmp";
  int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    printf("FAIL open %s: %s\n", tmp.c_str(), strerror(errno));
    return 1;
  }
  size_t len = strlen(image) + 1;
  bool ok = write(fd, image, len) == (ssize_t)len;
  if (close(fd) != 0 || !ok || rename(tmp.c_str(), path.c_str()) != 0) {
    printf("FAIL write %s: %s\n", path.c_str(), strerror(errno));
    return 1;
  }
  return 0;
}

static const unsigned N = 1 << 14; // elements per buffer

// A module's kernel, a device buffer it updates, and the buffer's expected contents
struct Work {
  const char *name;
  bool b; // image B (x = 3x + 5), else image A (x += 1)
  CUfunction fn = nullptr;
  unsigned *buf = nullptr;
  std::vector<unsigned> expected = {};
};

static int init(Work &w, CUmodule mod, unsigned seed) {
  if (check(d_cuModuleGetFunction(&w.fn, mod, "step"), "cuModuleGetFunction")) return 1;
  w.expected.resize(N);
  for (unsigned i = 0; i < N; i++) w.expected[i] = seed * 100003u + i;
  if (check(cudaMalloc(&w.buf, N * sizeof(unsigned)), "cudaMalloc")) return 1;
  return check(cudaMemcpy(w.buf, w.expected.data(), N * sizeof(unsigned), cudaMemcpyHostToDevice),
               "cudaMemcpy to device");
}

static int launch(Work &w) {
  void *buf = w.buf;
  unsigned n = N;
  void *args[] = {&buf, &n};
  char what[64];
  snprintf(what, sizeof(what), "cuLaunchKernel %s", w.name);
  if (check(d_cuLaunchKernel(w.fn, N / 256, 1, 1, 256, 1, 1, 0, 0, args, nullptr), what)) return 1;
  for (unsigned i = 0; i < N; i++) w.expected[i] = w.b ? 3u * w.expected[i] + 5u : w.expected[i] + 1u;
  return 0;
}

static int verify(Work &w, const char *when) {
  std::vector<unsigned> got(N);
  if (check(cudaMemcpy(got.data(), w.buf, N * sizeof(unsigned), cudaMemcpyDeviceToHost),
            "cudaMemcpy from device"))
    return 2;
  for (unsigned i = 0; i < N; i++) {
    if (got[i] != w.expected[i]) {
      printf("MISMATCH %s (image %c) buf[%u] = %u, expected %u (%s)\n", w.name, w.b ? 'B' : 'A', i, got[i],
             w.expected[i], when);
      return 3;
    }
  }
  return 0;
}

// Loads a module, runs its kernel once, and checks it computes what its image does
static int run_once(const char *name, bool b, CUmodule mod, unsigned seed) {
  Work w{name, b};
  if (init(w, mod, seed) || launch(w)) return 2;
  if (check(d_cuCtxSynchronize(), name)) return 2;
  return verify(w, "loaded after the last iteration");
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaFree(0), "cudaFree(0)")) return 1; // makes the primary context current
  if (load_driver()) return 1;

  char tmpl[] = "/tmp/replaced_module-XXXXXX";
  if (!mkdtemp(tmpl)) {
    printf("FAIL mkdtemp: %s\n", strerror(errno));
    return 1;
  }
  std::string dir = tmpl, file = dir + "/file.ptx", mapped = dir + "/mapped.ptx";

  // Image A, loaded by path, and from a mapping of another file
  CUmodule mod_file, mod_mapped, mod_new;
  if (write_image(file, PTX_A) || write_image(mapped, PTX_A)) return 1;
  if (check(d_cuModuleLoad(&mod_file, file.c_str()), "cuModuleLoad A")) return 1;
  int fd = open(mapped.c_str(), O_RDONLY);
  size_t len = sizeof(PTX_A);
  void *map = fd < 0 ? MAP_FAILED : mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    printf("FAIL map %s: %s\n", mapped.c_str(), strerror(errno));
    return 1;
  }
  close(fd);
  if (check(d_cuModuleLoadData(&mod_mapped, map), "cuModuleLoadData A (mapped)")) return 1;

  // Both files replaced with image B; the path loaded again is B
  if (write_image(file, PTX_B) || write_image(mapped, PTX_B)) return 1;
  if (check(d_cuModuleLoad(&mod_new, file.c_str()), "cuModuleLoad B")) return 1;

  Work w_file{"file", false}, w_mapped{"mapped", false}, w_new{"new", true};
  if (init(w_file, mod_file, 1) || init(w_mapped, mod_mapped, 2) || init(w_new, mod_new, 3)) return 1;

  long iterations = 0;
  while (true) {
    for (Work *w : {&w_file, &w_mapped, &w_new})
      if (int rc = verify(*w, "before the next iteration")) return rc;
    for (Work *w : {&w_file, &w_mapped, &w_new})
      if (launch(*w)) return 2;
    if (check(d_cuCtxSynchronize(), "kernels")) return 2;
    iterations++;
    for (Work *w : {&w_file, &w_mapped, &w_new})
      if (int rc = verify(*w, "after the iteration")) return rc;
    if (iterations == 1 && gate) printf("READY\n");
    if (gate ? access(gate, F_OK) == 0 : iterations >= 50) break;
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  printf("ok   %ld iterations\n", iterations);

  // After a restore, when gated: the path is B, and the mapping still A
  CUmodule mod_again, mod_remapped;
  if (check(d_cuModuleLoad(&mod_again, file.c_str()), "cuModuleLoad B (again)")) return 2;
  if (check(d_cuModuleLoadData(&mod_remapped, map), "cuModuleLoadData A (mapped, again)")) return 2;
  if (int rc = run_once("again", true, mod_again, 4)) return rc;
  if (int rc = run_once("remapped", false, mod_remapped, 5)) return rc;

  unlink(file.c_str());
  unlink(mapped.c_str());
  rmdir(dir.c_str());
  printf("PASS\n");
  return 0;
}
