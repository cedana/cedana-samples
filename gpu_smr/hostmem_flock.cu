/**
  @file hostmem_flock.cu
  @brief A cudaHostRegister'd host pool filled by GPU work while an flock is held.

  Mirrors an SGLang server with HiCache: an anonymous MAP_SHARED pool, pre-faulted and pinned
  with cudaHostRegister (the KV host pool), and a small /dev/shm file whose slot the scheduler
  rewrites under flock(LOCK_EX) (managers/load_snapshot.py). SGLang holds that lock for ~40 us
  thousands of times a second; here each iteration holds it for hold_ms (default 1000) while a
  vector add runs and its result is copied into the next slot of the pool, then drops it for
  1 ms. A checkpoint taken mid-work almost always finds the lock held.

  While holding the lock, each iteration checks that a second open of the file can't take it
  (LOCK_NB must fail), so a restore that drops the lock prints LOCK LOST. Every slot written is
  checked right away and again at the end, along with the counter in the lock file.

  Arguments are the pool size in MiB (default 512) and hold_ms, then the gate path, last. With
  no gate the work is 10 iterations. With a gate it prints READY after the first iteration and
  iterates until that file exists. Prints PASS and exits 0; non-zero on failure.
 */

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>

static const size_t N = 1 << 20; // 4 MiB of ints per pool slot

__global__ void vectorAdd(const int *a, const int *b, int *c, int k, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    c[i] = a[i] + b[i] + k;
  }
}

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

// a[i] = i and b[i] = 2i, so the add with k gives 3i + k
static int verify(const char *name, const int *buf, int k) {
  for (size_t i = 0; i < N; i++) {
    if (buf[i] != (int)(3 * i) + k) {
      printf("MISMATCH %s[%zu] = %d, expected %d\n", name, i, buf[i], (int)(3 * i) + k);
      return 1;
    }
  }
  return 0;
}

static bool numeric(const char *s) { return *s && strspn(s, "0123456789") == strlen(s); }

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  size_t pool_mib = 512;
  long hold_ms = 1000;
  const char *gate = nullptr;
  int nums = 0;
  for (int i = 1; i < argc; i++) {
    if (!numeric(argv[i])) gate = argv[i];
    else if (nums++ == 0) pool_mib = strtoul(argv[i], nullptr, 10);
    else hold_ms = strtol(argv[i], nullptr, 10);
  }
  const size_t slot_size = N * sizeof(int);
  const size_t slots = pool_mib * 1024 * 1024 / slot_size;
  if (slots == 0) {
    printf("FAIL pool must be at least %zu MiB\n", slot_size >> 20);
    return 1;
  }
  const size_t pool_size = slots * slot_size;

  // The host pool, as SGLang's alloc_mmap allocates it and _cuda_host_register pins it
  int *pool = (int *)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (pool == MAP_FAILED) {
    printf("FAIL mmap %zu MiB: %s\n", pool_size >> 20, strerror(errno));
    return 1;
  }

  // The lock file, as load_snapshot.py's ShmLoadSnapshotWriter opens it. A second open is a
  // separate open file description, so its flock conflicts with ours.
  char path[64];
  snprintf(path, sizeof(path), "/dev/shm/hostmem_flock_%d.shm", (int)getpid());
  int fd = open(path, O_CREAT | O_RDWR, 0600);
  int probe = open(path, O_RDWR);
  if (fd < 0 || probe < 0 || ftruncate(fd, 4096) != 0) {
    printf("FAIL open %s: %s\n", path, strerror(errno));
    return 1;
  }
  long *counter = (long *)mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (counter == MAP_FAILED) {
    printf("FAIL mmap %s: %s\n", path, strerror(errno));
    return 1;
  }
  *counter = 0;

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaHostRegister(pool, pool_size, cudaHostRegisterDefault), "cudaHostRegister")) return 1;
  int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  if (check(cudaMalloc(&d_a, slot_size), "cudaMalloc a")) return 1;
  if (check(cudaMalloc(&d_b, slot_size), "cudaMalloc b")) return 1;
  if (check(cudaMalloc(&d_c, slot_size), "cudaMalloc c")) return 1;
  std::vector<int> h(N);
  for (size_t i = 0; i < N; i++) h[i] = (int)i;
  if (check(cudaMemcpy(d_a, h.data(), slot_size, cudaMemcpyHostToDevice), "cudaMemcpy a")) return 1;
  for (size_t i = 0; i < N; i++) h[i] = (int)(2 * i);
  if (check(cudaMemcpy(d_b, h.data(), slot_size, cudaMemcpyHostToDevice), "cudaMemcpy b")) return 1;
  cudaStream_t stream;
  if (check(cudaStreamCreate(&stream), "cudaStreamCreate")) return 1;
  printf("pool %zu MiB (%zu slots) registered, lock %s held %ld ms per iteration\n",
         pool_size >> 20, slots, path, hold_ms);

  std::vector<int> last(slots, -1); // the k last written to each slot
  long iterations = 0;
  int k = 0;
  while (true) {
    if (flock(fd, LOCK_EX) != 0) {
      printf("FAIL flock: %s\n", strerror(errno));
      return 4;
    }
    if (flock(probe, LOCK_EX | LOCK_NB) == 0 || errno != EWOULDBLOCK) {
      printf("LOCK LOST iteration %ld: a second open took the lock\n", iterations);
      return 4;
    }
    if (*counter != iterations) {
      printf("MISMATCH lock file counter = %ld, expected %ld\n", *counter, iterations);
      return 3;
    }

    // GPU work into the pool while the lock is held, as HiCache backs KV up to host memory
    auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(hold_ms);
    do {
      size_t s = (size_t)k % slots;
      int *slot = pool + s * N;
      vectorAdd<<<256, 256, 0, stream>>>(d_a, d_b, d_c, k, N);
      if (check(cudaMemcpyAsync(slot, d_c, slot_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync")) return 2;
      if (check(cudaStreamSynchronize(stream), "kernel")) return 2;
      if (verify("slot", slot, k)) return 3;
      last[s] = k++;
    } while (std::chrono::steady_clock::now() < until);

    *counter = ++iterations;
    flock(fd, LOCK_UN);
    printf("STEP %ld k=%d\n", iterations, k);

    if (iterations == 1 && gate) printf("READY\n");
    if (gate ? access(gate, F_OK) == 0 : iterations >= 10) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  for (size_t s = 0; s < slots; s++) {
    if (last[s] >= 0 && verify("pool", pool + s * N, last[s])) return 3;
  }
  printf("ok   %ld iterations, %d vector adds, pool intact\n", iterations, k);

  if (check(cudaHostUnregister(pool), "cudaHostUnregister")) return 4;
  close(probe);
  close(fd);
  unlink(path);
  printf("PASS\n");
  return 0;
}
