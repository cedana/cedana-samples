/**
  @file cuda_ipc.cu
  @brief Two processes on one GPU sharing device memory and an event through CUDA IPC.

  The process forks before any CUDA call; parent and child each allocate a device buffer and an
  interprocess event, export both (cudaIpcGetMemHandle, cudaIpcGetEventHandle) through a shared
  page, and open the other's (cudaIpcOpenMemHandle, cudaIpcOpenEventHandle), as NCCL and vLLM
  workers do. Every iteration both sides first check that both buffers, each through its own
  allocation and the other through the IPC mapping, still hold what the previous iteration left
  (so a checkpoint between iterations that loses either buffer is caught before it is
  overwritten). Then the parent writes a new pattern into its buffer and records its event; the
  child waits on that event on the device, checks the pattern through its IPC mapping, and writes
  a value derived from it into its own buffer, which the parent checks through its mapping after
  waiting on the child's event. After the iterations each side exports a fresh
  buffer and opens the other's: its handle must differ from every handle exported before (a
  restore that reuses an IPC id makes a later open alias the wrong memory, or spin). An
  interceptor that does not restore the exported allocations, the opened mappings or the events,
  in both processes, reads stale or wrong memory, hangs, or faults.

  With no argument the work is 50 iterations. With a gate-file path as argv[1] the parent prints
  READY after the first iteration and both iterate until that file exists, letting a test
  checkpoint and restore both processes mid-work, then create the gate. Every iteration checks
  every element on both sides, both buffers are checked once more after the last iteration, and
  the fresh exports are checked after that: a difference prints
  MISMATCH. The parent prints PASS and exits 0 once the child has exited 0; non-zero on failure.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda_runtime.h>

static const size_t N = 1 << 20; // 4 MiB of words per buffer

// What the shared page holds; ints are accessed with __atomic builtins
struct Shared {
  cudaIpcMemHandle_t mem[2]; // [0] the parent's buffer, [1] the child's
  cudaIpcEventHandle_t ev[2];
  cudaIpcMemHandle_t fresh[2]; // exported after the iterations
  int exported[2];
  long begin;    // the last iteration the parent has started
  long verified; // the last iteration the child has checked the previous state for
  long step[2];  // the last iteration each side has finished its part of
  int stop;     // the parent has finished its last iteration
  int fresh_exported[2], fresh_closed[2];
  int failed; // 1 + the role of the side that failed first
};

static Shared *sh;
static int role; // 0 the parent, 1 the child
static const char *who[2] = {"parent", "child"};
static pid_t peer;

__host__ __device__ static inline unsigned pattern(long k, int r, size_t i) {
  return (unsigned)i * 2654435761u + (unsigned)k * 97u + (unsigned)r * 0x10000000u;
}

__global__ void write_pattern(unsigned *p, size_t n, long k, int r) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = pattern(k, r, i);
}

// Counts the elements of p that are not pattern(k, r, i) * mul + add
__global__ void check_pattern(const unsigned *p, size_t n, long k, int r, unsigned mul, unsigned add,
                              unsigned *errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    if (p[i] != pattern(k, r, i) * mul + add) atomicAdd(errors, 1u);
}

// dst[i] = src[i] * 3 + 1
__global__ void derive(unsigned *dst, const unsigned *src, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i] * 3u + 1u;
}

static void set_failed() {
  int none = 0;
  __atomic_compare_exchange_n(&sh->failed, &none, 1 + role, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s (%s): %d %s\n", what, who[role], (int)e, cudaGetErrorString(e));
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

// What a buffer holds after iteration k (all zero before the first): the parent's (r 0) the
// pattern, the child's (r 1) the pattern * 3 + 1
static int check_buf(const unsigned *p, long k, int r, unsigned *d_errors, cudaStream_t s, const char *what) {
  unsigned mul = k == 0 ? 0u : r == 0 ? 1u : 3u, add = k == 0 || r == 0 ? 0u : 1u;
  if (check(cudaMemsetAsync(d_errors, 0, sizeof(unsigned), s), "cudaMemsetAsync")) return 2;
  check_pattern<<<64, 256, 0, s>>>(p, N, k, 0, mul, add, d_errors);
  return read_errors(d_errors, s, what, k);
}

// Both buffers, through this side's allocation and its mapping of the other's, after iteration k
static int check_both(unsigned *own, unsigned *peer_buf, long k, unsigned *d_errors, cudaStream_t s,
                      const char *when) {
  char what[96];
  snprintf(what, sizeof(what), "own buffer %s", when);
  if (int rc = check_buf(own, k, role, d_errors, s, what)) return rc;
  snprintf(what, sizeof(what), "other's buffer %s", when);
  return check_buf(peer_buf, k, 1 - role, d_errors, s, what);
}

static int run(const char *gate) {
  const int other = 1 - role;
  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  cudaStream_t s;
  if (check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate")) return 1;
  unsigned *own = nullptr, *d_errors = nullptr;
  if (check(cudaMalloc(&own, N * sizeof(unsigned)), "cudaMalloc")) return 1;
  if (check(cudaMalloc(&d_errors, sizeof(unsigned)), "cudaMalloc errors")) return 1;
  if (check(cudaMemset(own, 0, N * sizeof(unsigned)), "cudaMemset")) return 1;
  cudaEvent_t own_ev, peer_ev;
  if (check(cudaEventCreateWithFlags(&own_ev, cudaEventDisableTiming | cudaEventInterprocess), "cudaEventCreate"))
    return 1;
  if (check(cudaIpcGetMemHandle(&sh->mem[role], own), "cudaIpcGetMemHandle")) return 1;
  if (check(cudaIpcGetEventHandle(&sh->ev[role], own_ev), "cudaIpcGetEventHandle")) return 1;
  if (check(cudaDeviceSynchronize(), "setup")) return 1;
  store(&sh->exported[role], 1);
  if (wait_for([&] { return load(&sh->exported[other]) != 0; }, "the other's exports")) return 1;
  unsigned *peer_buf = nullptr;
  if (check(cudaIpcOpenMemHandle((void **)&peer_buf, sh->mem[other], cudaIpcMemLazyEnablePeerAccess),
            "cudaIpcOpenMemHandle"))
    return 1;
  if (check(cudaIpcOpenEventHandle(&peer_ev, sh->ev[other]), "cudaIpcOpenEventHandle")) return 1;

  long k = 0;
  while (true) {
    if (role == 0) {
      // What the previous iteration left, before anything replaces it; the child checks too
      if (int rc = check_both(own, peer_buf, k, d_errors, s, "before the next iteration")) return rc;
      k++;
      store(&sh->begin, k);
      if (wait_for([&] { return load(&sh->verified) >= k; }, "the child's check")) return 2;
      write_pattern<<<64, 256, 0, s>>>(own, N, k, 0);
      if (check(cudaEventRecord(own_ev, s), "cudaEventRecord")) return 2;
      store(&sh->step[0], k);
      if (wait_for([&] { return load(&sh->step[1]) >= k; }, "the child's iteration")) return 2;
      // The child's buffer, through the mapping, once its work is done on the device
      if (check(cudaStreamWaitEvent(s, peer_ev, 0), "cudaStreamWaitEvent")) return 2;
      if (check(cudaMemsetAsync(d_errors, 0, sizeof(unsigned), s), "cudaMemsetAsync")) return 2;
      check_pattern<<<64, 256, 0, s>>>(peer_buf, N, k, 0, 3u, 1u, d_errors);
      if (int rc = read_errors(d_errors, s, "child's buffer", k)) return rc;
      if (k == 1 && gate) printf("READY\n");
      if (gate ? access(gate, F_OK) == 0 : k >= 50) {
        store(&sh->stop, 1);
        break;
      }
      if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } else {
      if (wait_for([&] { return load(&sh->begin) > k || load(&sh->stop); }, "the parent's iteration")) return 2;
      if (load(&sh->begin) == k) break; // stopped
      // What the previous iteration left, before the parent's write replaces it
      if (int rc = check_both(own, peer_buf, k, d_errors, s, "before the next iteration")) return rc;
      k++;
      store(&sh->verified, k);
      if (wait_for([&] { return load(&sh->step[0]) >= k; }, "the parent's write")) return 2;
      // The parent's buffer, through the mapping, once its write is done on the device
      if (check(cudaStreamWaitEvent(s, peer_ev, 0), "cudaStreamWaitEvent")) return 2;
      if (check(cudaMemsetAsync(d_errors, 0, sizeof(unsigned), s), "cudaMemsetAsync")) return 2;
      check_pattern<<<64, 256, 0, s>>>(peer_buf, N, k, 0, 1u, 0u, d_errors);
      derive<<<64, 256, 0, s>>>(own, peer_buf, N);
      if (check(cudaEventRecord(own_ev, s), "cudaEventRecord")) return 2;
      store(&sh->step[1], k);
      if (int rc = read_errors(d_errors, s, "parent's buffer", k)) return rc;
    }
  }
  if (role == 0) printf("ok   %ld iterations\n", k);
  // After a restore, when gated: both buffers still hold the last iteration
  if (int rc = check_both(own, peer_buf, k, d_errors, s, "after the iterations")) return rc;

  // Fresh exports, after a restore when gated: new handles, opened next to the old ones
  unsigned *fresh = nullptr, *peer_fresh = nullptr;
  if (check(cudaMalloc(&fresh, N * sizeof(unsigned)), "cudaMalloc fresh")) return 4;
  write_pattern<<<64, 256, 0, s>>>(fresh, N, 1000 + k, role);
  if (check(cudaStreamSynchronize(s), "write fresh")) return 4;
  if (check(cudaIpcGetMemHandle(&sh->fresh[role], fresh), "cudaIpcGetMemHandle fresh")) return 4;
  store(&sh->fresh_exported[role], 1);
  if (wait_for([&] { return load(&sh->fresh_exported[other]) != 0; }, "the other's fresh export")) return 4;
  const cudaIpcMemHandle_t *before[] = {&sh->mem[0], &sh->mem[1], &sh->fresh[other]};
  for (const cudaIpcMemHandle_t *h : before) {
    if (memcmp(h, &sh->fresh[role], sizeof(*h)) == 0) {
      printf("MISMATCH (%s) fresh IPC handle equals one exported before\n", who[role]);
      set_failed();
      return 3;
    }
  }
  if (check(cudaIpcOpenMemHandle((void **)&peer_fresh, sh->fresh[other], cudaIpcMemLazyEnablePeerAccess),
            "cudaIpcOpenMemHandle fresh"))
    return 4;
  if (peer_fresh == peer_buf) {
    printf("MISMATCH (%s) fresh IPC mapping at the old mapping's address %p\n", who[role], peer_buf);
    set_failed();
    return 3;
  }
  if (check(cudaMemsetAsync(d_errors, 0, sizeof(unsigned), s), "cudaMemsetAsync")) return 4;
  check_pattern<<<64, 256, 0, s>>>(peer_fresh, N, 1000 + k, other, 1u, 0u, d_errors);
  if (int rc = read_errors(d_errors, s, "other's fresh buffer", k)) return rc;
  // The old allocations and mappings still show the last iteration
  if (int rc = check_both(own, peer_buf, k, d_errors, s, "after the fresh export")) return rc;
  if (role == 0) printf("ok   fresh exports\n");

  if (check(cudaIpcCloseMemHandle(peer_fresh), "cudaIpcCloseMemHandle fresh")) return 5;
  if (check(cudaIpcCloseMemHandle(peer_buf), "cudaIpcCloseMemHandle")) return 5;
  store(&sh->fresh_closed[role], 1);
  // The exporter frees only once the importer has closed its mappings
  if (wait_for([&] { return load(&sh->fresh_closed[other]) != 0; }, "the other to close")) return 5;
  if (check(cudaEventDestroy(peer_ev), "cudaEventDestroy")) return 5;
  if (check(cudaEventDestroy(own_ev), "cudaEventDestroy")) return 5;
  if (check(cudaFree(fresh), "cudaFree fresh")) return 5;
  if (check(cudaFree(own), "cudaFree")) return 5;
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

  // Fork before any CUDA call: CUDA IPC is between processes, and CUDA state does not survive fork
  pid_t parent = getpid();
  pid_t child = fork();
  if (child < 0) {
    printf("FAIL fork\n");
    return 1;
  }
  if (child == 0) {
    role = 1;
    peer = parent;
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    if (getppid() != parent) _exit(1);
    _exit(run(nullptr));
  }
  role = 0;
  peer = child;
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
