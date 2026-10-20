/**
  @file cumem_ipc_fd.cu
  @brief Two processes on one GPU sharing VMM allocations through exported file descriptors.

  As NCCL does with cuMem buffers (NCCL_CUMEM_ENABLE): the process forks before any CUDA call;
  the parent creates two allocations whose handles are shareable as POSIX file descriptors
  (cuMemCreate), maps them, exports each as an fd (cuMemExportToShareableHandle), and sends the
  fds to the child over a unix socket (SCM_RIGHTS). The child imports and maps both
  (cuMemImportFromShareableHandle); it closes the fd of A once imported, as NCCL does, and keeps
  the fd of B open. Every iteration both sides first check that both allocations still hold what
  the previous iteration left (so a checkpoint between iterations that loses either is caught
  before it's overwritten). Then the parent writes a new pattern into A; the child checks it
  through its mapping and writes a value derived from it into B, which the parent checks through
  its own mapping. After the iterations the parent exports a third allocation, C, which the
  child imports and checks, and the child imports B again from the fd it kept, which must map
  B's memory. An interceptor that doesn't restore the allocations, their mappings in both
  processes, or the exported fds (the one passed over, and the one kept open), reads stale or
  wrong memory, or fails the import.

  The driver API is reached through dlopen("libcuda.so.1"), so the binary links only libcudart.

  With no argument the work is 50 iterations. With a gate-file path as argv[1] the parent prints
  READY after the first iteration and both iterate until that file exists, letting a test
  checkpoint and restore both processes mid-work, then create the gate. A difference prints
  MISMATCH. The parent prints PASS and exits 0 once the child has exited 0; non-zero on failure.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <dlfcn.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const size_t N = 1 << 20; // 4 MiB of words per allocation

// What the shared page holds; accessed with __atomic builtins
struct Shared {
  long begin;    // the last iteration the parent has started
  long verified; // the last iteration the child has checked the previous state for
  long step[2];  // the last iteration each side has finished its part of
  int stop;      // the parent has finished its last iteration
  int fresh_sent, fresh_checked;
  int failed; // 1 + the role of the side that failed first
};

static Shared *sh;
static int role; // 0 the parent, 1 the child
static const char *who[2] = {"parent", "child"};
static pid_t peer;
static int sock; // the parent's or the child's end of the socket pair

// The driver API, from libcuda.so.1
#define XSTR(x) STR(x)
#define STR(x) #x
#define DRIVER_FNS(X)                                                                                   \
  X(cuGetErrorString) X(cuCtxGetDevice) X(cuMemGetAllocationGranularity) X(cuMemCreate) X(cuMemRelease) \
  X(cuMemAddressReserve) X(cuMemAddressFree) X(cuMemMap) X(cuMemUnmap) X(cuMemSetAccess)                \
  X(cuMemExportToShareableHandle) X(cuMemImportFromShareableHandle)
#define DECLARE(fn) static decltype(&fn) d_##fn;
DRIVER_FNS(DECLARE)

static void set_failed() {
  int none = 0;
  __atomic_compare_exchange_n(&sh->failed, &none, 1 + role, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

static int load_driver() {
  void *h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) {
    printf("FAIL dlopen libcuda.so.1 (%s): %s\n", who[role], dlerror());
    set_failed();
    return 1;
  }
#define LOAD(fn)                                                  \
  if (!(d_##fn = (decltype(&fn))dlsym(h, XSTR(fn)))) {            \
    printf("FAIL dlsym %s (%s)\n", XSTR(fn), who[role]);           \
    set_failed();                                                 \
    return 1;                                                     \
  }
  DRIVER_FNS(LOAD)
  return 0;
}

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s (%s): %d %s\n", what, who[role], (int)e, cudaGetErrorString(e));
    set_failed();
    return 1;
  }
  return 0;
}

static int check(CUresult r, const char *what) {
  if (r != CUDA_SUCCESS) {
    const char *s = nullptr;
    if (!d_cuGetErrorString || d_cuGetErrorString(r, &s) != CUDA_SUCCESS || !s) s = "unknown";
    printf("FAIL %s (%s): %d %s\n", what, who[role], (int)r, s);
    set_failed();
    return 1;
  }
  return 0;
}

static int load(int *p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
static long load(long *p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
template <typename T> static void store(T *p, T v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }

// Waits until done() holds; fails if the other process has failed or gone
template <typename F> static int wait_for(F done, const char *what) {
  while (!done()) {
    if (load(&sh->failed)) return 1;
    if (role == 0) {
      int status;
      if (waitpid(peer, &status, WNOHANG) == peer) {
        printf("FAIL child exited (status %d) while the parent waited for %s\n", status, what);
        peer = 0;
        return 1;
      }
    } else if (getppid() != peer) {
      printf("FAIL parent exited while the child waited for %s\n", what);
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return 0;
}

static int send_fd(int fd) {
  char byte = 0;
  struct iovec iov = {&byte, 1};
  char buf[CMSG_SPACE(sizeof(int))] = {};
  struct msghdr msg = {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);
  struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(c), &fd, sizeof(int));
  if (sendmsg(sock, &msg, 0) != 1) {
    printf("FAIL sendmsg (%s): %s\n", who[role], strerror(errno));
    set_failed();
    return 1;
  }
  return 0;
}

static int recv_fd(int *fd) {
  char byte;
  struct iovec iov = {&byte, 1};
  char buf[CMSG_SPACE(sizeof(int))] = {};
  struct msghdr msg = {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);
  if (recvmsg(sock, &msg, 0) != 1) {
    printf("FAIL recvmsg (%s): %s\n", who[role], strerror(errno));
    set_failed();
    return 1;
  }
  struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
  if (!c || c->cmsg_type != SCM_RIGHTS) {
    printf("FAIL recvmsg (%s): no fd\n", who[role]);
    set_failed();
    return 1;
  }
  memcpy(fd, CMSG_DATA(c), sizeof(int));
  return 0;
}

__host__ __device__ static inline unsigned pattern(long k, size_t i) {
  return (unsigned)i * 2654435761u + (unsigned)k * 97u;
}

__global__ void write_pattern(unsigned *p, size_t n, long k) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = pattern(k, i);
}

// Counts the elements of p that are not pattern(k, i) * mul + add
__global__ void check_pattern(const unsigned *p, size_t n, long k, unsigned mul, unsigned add, unsigned *errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    if (p[i] != pattern(k, i) * mul + add) atomicAdd(errors, 1u);
}

// dst[i] = src[i] * 3 + 1
__global__ void derive(unsigned *dst, const unsigned *src, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i] * 3u + 1u;
}

static size_t bytes; // of each allocation, rounded up to the granularity
static CUmemAllocationProp prop = {};
static CUmemAccessDesc access_desc = {};

// Reserves an address range for handle h, maps it there, and makes it readable and writable
static int map(CUmemGenericAllocationHandle h, unsigned **p, const char *what) {
  CUdeviceptr ptr;
  if (check(d_cuMemAddressReserve(&ptr, bytes, 0, 0, 0), what)) return 1;
  if (check(d_cuMemMap(ptr, bytes, 0, h, 0), what)) return 1;
  if (check(d_cuMemSetAccess(ptr, bytes, &access_desc, 1), what)) return 1;
  *p = (unsigned *)ptr;
  return 0;
}

static int unmap(unsigned *p) {
  CUdeviceptr ptr = (CUdeviceptr)p;
  if (check(d_cuMemUnmap(ptr, bytes), "cuMemUnmap")) return 1;
  return check(d_cuMemAddressFree(ptr, bytes), "cuMemAddressFree");
}

// A new allocation shareable as a POSIX fd, mapped, and exported
static int create(CUmemGenericAllocationHandle *h, unsigned **p, int *fd, const char *what) {
  if (check(d_cuMemCreate(h, bytes, &prop, 0), what)) return 1;
  if (map(*h, p, what)) return 1;
  return check(d_cuMemExportToShareableHandle(fd, *h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), what);
}

static int import(int fd, CUmemGenericAllocationHandle *h, unsigned **p, const char *what) {
  if (check(d_cuMemImportFromShareableHandle(h, (void *)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            what))
    return 1;
  return map(*h, p, what);
}

static int read_errors(unsigned *d_errors, cudaStream_t s, const char *what, long k) {
  unsigned errors = 0;
  if (check(cudaMemcpyAsync(&errors, d_errors, sizeof(errors), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync errors"))
    return 2;
  if (check(cudaStreamSynchronize(s), what)) return 2;
  if (errors) {
    printf("MISMATCH %s (%s) iteration %ld: %u of %zu elements wrong\n", what, who[role], k, errors, N);
    set_failed();
    return 3;
  }
  return 0;
}

static int check_buf(const unsigned *p, long k, unsigned mul, unsigned add, unsigned *d_errors, cudaStream_t s,
                     const char *what) {
  if (check(cudaMemsetAsync(d_errors, 0, sizeof(unsigned), s), "cudaMemsetAsync")) return 2;
  check_pattern<<<64, 256, 0, s>>>(p, N, k, mul, add, d_errors);
  return read_errors(d_errors, s, what, k);
}

// A after iteration k holds the pattern, B the pattern * 3 + 1 (both all zero before the first)
static int check_both(unsigned *a, unsigned *b, long k, unsigned *d_errors, cudaStream_t s, const char *when) {
  char what[96];
  snprintf(what, sizeof(what), "A %s", when);
  if (int rc = check_buf(a, k, k ? 1u : 0u, 0u, d_errors, s, what)) return rc;
  snprintf(what, sizeof(what), "B %s", when);
  return check_buf(b, k, k ? 3u : 0u, k ? 1u : 0u, d_errors, s, what);
}

static int run(const char *gate) {
  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaFree(0), "cudaFree(0)")) return 1; // makes the primary context current
  if (load_driver()) return 1;
  CUdevice dev;
  if (check(d_cuCtxGetDevice(&dev), "cuCtxGetDevice")) return 1;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  access_desc.location = prop.location;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  size_t gran = 0;
  if (check(d_cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            "cuMemGetAllocationGranularity"))
    return 1;
  bytes = (N * sizeof(unsigned) + gran - 1) / gran * gran;

  cudaStream_t s;
  if (check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate")) return 1;
  unsigned *d_errors = nullptr;
  if (check(cudaMalloc(&d_errors, sizeof(unsigned)), "cudaMalloc errors")) return 1;

  CUmemGenericAllocationHandle ha, hb;
  unsigned *a = nullptr, *b = nullptr;
  int fd_b = -1; // the child keeps it open
  if (role == 0) {
    int fd_a;
    if (create(&ha, &a, &fd_a, "export A") || create(&hb, &b, &fd_b, "export B")) return 1;
    if (check(cudaMemset(a, 0, bytes), "cudaMemset A") || check(cudaMemset(b, 0, bytes), "cudaMemset B")) return 1;
    if (send_fd(fd_a) || send_fd(fd_b)) return 1;
    close(fd_a);
    close(fd_b);
  } else {
    int fd_a;
    if (recv_fd(&fd_a) || recv_fd(&fd_b)) return 1;
    if (import(fd_a, &ha, &a, "import A") || import(fd_b, &hb, &b, "import B")) return 1;
    close(fd_a);
  }

  long k = 0;
  while (true) {
    if (role == 0) {
      // What the previous iteration left, before anything replaces it; the child checks too
      if (int rc = check_both(a, b, k, d_errors, s, "before the next iteration")) return rc;
      k++;
      store(&sh->begin, k);
      if (wait_for([&] { return load(&sh->verified) >= k; }, "the child's check")) return 2;
      write_pattern<<<64, 256, 0, s>>>(a, N, k);
      if (check(cudaStreamSynchronize(s), "write A")) return 2;
      store(&sh->step[0], k);
      if (wait_for([&] { return load(&sh->step[1]) >= k; }, "the child's iteration")) return 2;
      if (int rc = check_buf(b, k, 3u, 1u, d_errors, s, "B from the child")) return rc;
      if (k == 1 && gate) printf("READY\n");
      if (gate ? access(gate, F_OK) == 0 : k >= 50) {
        store(&sh->stop, 1);
        break;
      }
      if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } else {
      if (wait_for([&] { return load(&sh->begin) > k || load(&sh->stop); }, "the parent's iteration")) return 2;
      if (load(&sh->begin) == k) break; // stopped
      if (int rc = check_both(a, b, k, d_errors, s, "before the next iteration")) return rc;
      k++;
      store(&sh->verified, k);
      if (wait_for([&] { return load(&sh->step[0]) >= k; }, "the parent's write")) return 2;
      if (int rc = check_buf(a, k, 1u, 0u, d_errors, s, "A from the parent")) return rc;
      derive<<<64, 256, 0, s>>>(b, a, N);
      if (check(cudaStreamSynchronize(s), "derive B")) return 2;
      store(&sh->step[1], k);
    }
  }
  if (role == 0) printf("ok   %ld iterations\n", k);
  // After a restore, when gated: both allocations still hold the last iteration
  if (int rc = check_both(a, b, k, d_errors, s, "after the iterations")) return rc;

  // A fresh export, after a restore when gated; and B imported again from the fd kept open
  CUmemGenericAllocationHandle hc, hb2 = 0;
  unsigned *c = nullptr, *b2 = nullptr;
  if (role == 0) {
    int fd_c;
    if (create(&hc, &c, &fd_c, "export C")) return 4;
    write_pattern<<<64, 256, 0, s>>>(c, N, 1000 + k);
    if (check(cudaStreamSynchronize(s), "write C")) return 4;
    if (send_fd(fd_c)) return 4;
    close(fd_c);
    store(&sh->fresh_sent, 1);
    if (wait_for([&] { return load(&sh->fresh_checked) != 0; }, "the child's check of C")) return 4;
  } else {
    if (wait_for([&] { return load(&sh->fresh_sent) != 0; }, "the parent's export of C")) return 4;
    int fd_c;
    if (recv_fd(&fd_c) || import(fd_c, &hc, &c, "import C")) return 4;
    close(fd_c);
    if (int rc = check_buf(c, 1000 + k, 1u, 0u, d_errors, s, "C after the iterations")) return rc;
    if (import(fd_b, &hb2, &b2, "import B again, from the fd kept open")) return 4;
    if (int rc = check_buf(b2, k, 3u, 1u, d_errors, s, "B imported again")) return rc;
    close(fd_b);
    store(&sh->fresh_checked, 1);
  }
  // The old mappings still show the last iteration
  if (int rc = check_both(a, b, k, d_errors, s, "after the fresh export")) return rc;
  if (role == 0) printf("ok   fresh export\n");

  if (b2 && (unmap(b2) || check(d_cuMemRelease(hb2), "cuMemRelease B again"))) return 5;
  if (unmap(c) || check(d_cuMemRelease(hc), "cuMemRelease C")) return 5;
  if (unmap(a) || check(d_cuMemRelease(ha), "cuMemRelease A")) return 5;
  if (unmap(b) || check(d_cuMemRelease(hb), "cuMemRelease B")) return 5;
  if (check(cudaFree(d_errors), "cudaFree errors")) return 5;
  if (check(cudaStreamDestroy(s), "cudaStreamDestroy")) return 5;
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  const char *gate = (argc > 1) ? argv[1] : nullptr;

  sh = (Shared *)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (sh == MAP_FAILED) {
    printf("FAIL mmap\n");
    return 1;
  }
  memset(sh, 0, sizeof(Shared));
  int fds[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    printf("FAIL socketpair: %s\n", strerror(errno));
    return 1;
  }

  // Fork before any CUDA call: CUDA state does not survive fork
  pid_t parent = getpid();
  pid_t child = fork();
  if (child < 0) {
    printf("FAIL fork\n");
    return 1;
  }
  if (child == 0) {
    role = 1;
    peer = parent;
    sock = fds[1];
    close(fds[0]);
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    if (getppid() != parent) _exit(1);
    _exit(run(nullptr));
  }
  role = 0;
  peer = child;
  sock = fds[0];
  close(fds[1]);
  int rc = run(gate);
  if (rc) {
    // When the child failed first, its exit code says how; give it a moment to exit
    int status = 0;
    bool child_first = __atomic_load_n(&sh->failed, __ATOMIC_SEQ_CST) == 2;
    for (int i = 0; peer && i < 50; i++) {
      if (waitpid(peer, &status, WNOHANG) == peer) {
        peer = 0;
        if (child_first && WIFEXITED(status) && WEXITSTATUS(status)) return WEXITSTATUS(status);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (peer) {
      kill(peer, SIGKILL);
      waitpid(peer, nullptr, 0);
    }
    return rc;
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child) {
    printf("FAIL waitpid\n");
    return 6;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    printf("FAIL child exited with status %d\n", status);
    return WIFEXITED(status) && WEXITSTATUS(status) ? WEXITSTATUS(status) : 6;
  }
  printf("PASS\n");
  return 0;
}
