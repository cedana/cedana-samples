/**
  @file hostmem_ipc.cu
  @brief Multi-GPU registered host memory and CUDA IPC, from a customer's pinned-memory probe.

  One process per GPU (exec'd like vLLM TP workers), each holding GPU state that must survive
  restore, with neighbours reading each other's device memory through CUDA IPC every step.

  Rank 0 first sweeps host memory registration across sizes that broke before (the 8 GiB
  boundary, vLLM's ~12 GB CPU offload buffer): cudaHostRegister (page-aligned and not) and
  cuMemHostRegister on mmap'd memory, cudaHostAlloc, and pinned allocations after a register
  attempt. It then keeps a 9 GiB registered buffer live, so a checkpoint has to carry
  registered host memory. Needs about 48 GiB of RAM (16 GiB sweep peak, 9 GiB live).

  Every second rank 0 prints `STEP <n> ...`. Each step checks that the pinned and registered host
  buffers still hold what the previous step wrote before writing new values. Any lost device or
  host state, or a broken IPC mapping, prints MISMATCH; a failed CUDA call prints FAIL. Either way every rank is killed and
  it exits non-zero. With a gate-file path it prints READY after the first step and steps until
  that file exists, letting a test checkpoint and restore it mid-work, then create the gate:
  every rank checks its state once more, tears down, and it prints PASS and exits 0.

  Usage: hostmem_ipc [world] [gate]   (world 0 or absent: all visible GPUs)
 */

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

constexpr size_t MiB = 1ull << 20;
constexpr size_t GiB = 1ull << 30;
constexpr size_t SWEEP[] = {4096,          MiB,     64 * MiB,     GiB,      4 * GiB,
                            7 * GiB,       8 * GiB - 2 * MiB,     8 * GiB,  8 * GiB + 2 * MiB,
                            9 * GiB,       12001886208ull,        16 * GiB};
constexpr size_t LIVE_REGISTERED = 9 * GiB;  // past the 8 GiB boundary
constexpr size_t STATE_WORDS = 16 * MiB;     // 64 MiB of device state per rank
constexpr size_t CHECK = 64 * MiB;           // bytes round-tripped per host buffer check
constexpr int MAX_RANKS = 16;

static int rank = -1;

#define DIE(word, ...)                                 \
    do {                                               \
        printf(word " rank=%d ", rank);                \
        printf(__VA_ARGS__);                           \
        printf("\n");                                  \
        fflush(stdout);                                \
        exit(1);                                       \
    } while (0)
#define FAIL(...) DIE("FAIL", __VA_ARGS__)          // a CUDA call or the setup failed
#define MISMATCH(...) DIE("MISMATCH", __VA_ARGS__)  // state was lost or corrupted

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) FAIL("%s: %s", #call, cudaGetErrorString(e));    \
    } while (0)

// Shared between ranks through a MAP_SHARED file: a sense-reversing barrier, the IPC handles,
// the gate path and rank 0's decision to stop.
struct Shared {
    int world;
    volatile int arrived;
    volatile int generation;
    volatile int stop;
    char gate[4096];
    cudaIpcMemHandle_t handles[MAX_RANKS];
};
static Shared* shared;

static void barrier() {
    int gen = shared->generation;
    if (__atomic_add_fetch(&shared->arrived, 1, __ATOMIC_ACQ_REL) == shared->world) {
        shared->arrived = 0;
        __atomic_add_fetch(&shared->generation, 1, __ATOMIC_ACQ_REL);
    } else {
        while (__atomic_load_n(&shared->generation, __ATOMIC_ACQUIRE) == gen) usleep(100);
    }
}

__global__ void increment(uint32_t* state, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) state[i]++;
}

__global__ void mismatches(const uint8_t* buf, size_t n, uint8_t want, unsigned long long* count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (buf[i] != want) atomicAdd(count, 1ull);
}

// The first CHECK bytes of host memory hold value, as the GPU reads them over pinned DMA.
static bool holds(const void* host, size_t size, uint8_t value) {
    static uint8_t* scratch;
    static unsigned long long* count;
    if (!scratch) {
        CHECK_CUDA(cudaMalloc(&scratch, CHECK));
        CHECK_CUDA(cudaMalloc(&count, sizeof(*count)));
    }
    size_t n = std::min(size, CHECK);
    CHECK_CUDA(cudaMemset(count, 0, sizeof(*count)));
    CHECK_CUDA(cudaMemcpyAsync(scratch, host, n, cudaMemcpyHostToDevice));
    mismatches<<<256, 256>>>(scratch, n, value, count);
    unsigned long long bad = 0;
    CHECK_CUDA(cudaMemcpy(&bad, count, sizeof(bad), cudaMemcpyDeviceToHost));
    return bad == 0;
}

// Pinned DMA actually works: fill on the host, copy to the GPU asynchronously, compare there.
static bool copy_ok(void* host, size_t size, uint8_t value) {
    memset(host, value, std::min(size, CHECK));
    return holds(host, size, value);
}

static void* alloc_touched(size_t size) {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) FAIL("mmap %zu B failed", size);
    memset(p, 0, size);
    return p;
}

static void sweep() {
    using cuMemHostRegister_t = int (*)(void*, size_t, unsigned);
    using cuMemHostUnregister_t = int (*)(void*);
    void* libcuda = dlopen("libcuda.so.1", RTLD_NOW);
    auto cuRegister = reinterpret_cast<cuMemHostRegister_t>(dlsym(libcuda, "cuMemHostRegister_v2"));
    auto cuUnregister = reinterpret_cast<cuMemHostUnregister_t>(dlsym(libcuda, "cuMemHostUnregister"));
    if (!cuRegister || !cuUnregister) FAIL("cuMemHostRegister_v2/cuMemHostUnregister not found");

    for (size_t size : SWEEP) {
        void* p = alloc_touched(size);  // runtime API, as torch's cudaHostRegister does
        CHECK_CUDA(cudaHostRegister(p, size, cudaHostRegisterDefault));
        if (!copy_ok(p, size, 7)) MISMATCH("cudaHostRegister %zu B copy mismatch", size);
        CHECK_CUDA(cudaHostUnregister(p));
        munmap(p, size);

        // Not page-aligned at either end, like torch/malloc memory past a 64-byte chunk header
        constexpr size_t OFFSET = 64;
        void* base = alloc_touched(size + 4096);
        p = static_cast<char*>(base) + OFFSET;
        CHECK_CUDA(cudaHostRegister(p, size, cudaHostRegisterDefault));
        if (!copy_ok(p, size, 10)) MISMATCH("cudaHostRegister unaligned %zu B copy mismatch", size);
        CHECK_CUDA(cudaHostUnregister(p));
        munmap(base, size + 4096);

        p = alloc_touched(size);  // driver API
        if (int rc = cuRegister(p, size, 0)) FAIL("cuMemHostRegister %zu B rc=%d", size, rc);
        if (!copy_ok(p, size, 8)) MISMATCH("cuMemHostRegister %zu B copy mismatch", size);
        cuUnregister(p);
        munmap(p, size);

        CHECK_CUDA(cudaHostAlloc(&p, size, cudaHostAllocDefault));
        if (!copy_ok(p, size, 9)) MISMATCH("cudaHostAlloc %zu B copy mismatch", size);
        CHECK_CUDA(cudaFreeHost(p));
        printf("SWEEP %zu B ok\n", size);
        fflush(stdout);
    }

    // Pinned allocations must keep working after a register attempt, whatever it returned
    void* p = alloc_touched(MiB);
    if (cudaHostRegister(p, MiB, cudaHostRegisterDefault) == cudaSuccess) cudaHostUnregister(p);
    cudaGetLastError();
    munmap(p, MiB);
    for (int attempt = 1; attempt <= 2; attempt++) {
        CHECK_CUDA(cudaHostAlloc(&p, 2 * MiB, cudaHostAllocDefault));
        if (!copy_ok(p, 2 * MiB, 11)) MISMATCH("cudaHostAlloc after register attempt %d: copy mismatch", attempt);
        CHECK_CUDA(cudaFreeHost(p));
    }
}

static int run_rank(int world) {
    CHECK_CUDA(cudaSetDevice(rank));
    if (rank == 0) sweep();
    barrier();

    uint32_t* state;
    CHECK_CUDA(cudaMalloc(&state, STATE_WORDS * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(state, 0, STATE_WORDS * sizeof(uint32_t)));
    CHECK_CUDA(cudaIpcGetMemHandle(&shared->handles[rank], state));
    barrier();
    uint32_t* peer = nullptr;  // the next rank's state, through CUDA IPC
    if (world > 1) {
        CHECK_CUDA(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&peer), shared->handles[(rank + 1) % world],
                                        cudaIpcMemLazyEnablePeerAccess));
    }

    void* pinned;
    CHECK_CUDA(cudaHostAlloc(&pinned, 4 * MiB, cudaHostAllocDefault));
    memset(pinned, 0, 4 * MiB);  // what step 1 finds there, as in the registered buffer
    size_t live = rank == 0 ? LIVE_REGISTERED : 4 * MiB;
    void* registered = alloc_touched(live);  // as vLLM CPU offload and SGLang HiCache do
    CHECK_CUDA(cudaHostRegister(registered, live, cudaHostRegisterDefault));

    for (uint32_t step = 1;; step++) {
        increment<<<256, 256>>>(state, STATE_WORDS);
        CHECK_CUDA(cudaDeviceSynchronize());
        barrier();  // every rank has finished this step's increment

        uint32_t first, last, theirs = step;
        CHECK_CUDA(cudaMemcpy(&first, state, sizeof(first), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&last, state + STATE_WORDS - 1, sizeof(last), cudaMemcpyDeviceToHost));
        if (peer) CHECK_CUDA(cudaMemcpy(&theirs, peer, sizeof(theirs), cudaMemcpyDeviceToHost));
        if (first != step || last != step) MISMATCH("step=%u device state=%u/%u", step, first, last);
        if (theirs != step) MISMATCH("step=%u peer state via IPC=%u", step, theirs);
        // The host buffers still hold what the previous step wrote, before this step replaces it
        uint8_t prev = step == 1 ? 0 : (step - 1) % 256;
        if (!holds(pinned, 4 * MiB, prev)) MISMATCH("step=%u pinned buffer lost the previous step's contents", step);
        if (!holds(registered, live, prev))
            MISMATCH("step=%u registered buffer lost the previous step's contents (%zu B)", step, live);
        if (!copy_ok(pinned, 4 * MiB, step % 256)) MISMATCH("step=%u pinned copy mismatch", step);
        if (!copy_ok(registered, live, step % 256)) MISMATCH("step=%u registered copy mismatch (%zu B)", step, live);
        if (rank == 0) {
            printf("STEP %u state=%u world=%d\n", step, first, world);
            if (step == 1 && shared->gate[0]) printf("READY\n");
            fflush(stdout);
            shared->stop = shared->gate[0] && access(shared->gate, F_OK) == 0;
        }
        barrier();  // nobody starts the next increment until every peer read is done
        if (shared->stop) break;  // this step, checked after the gate appeared, was the last
        sleep(1);
    }

    if (peer) CHECK_CUDA(cudaIpcCloseMemHandle(peer));
    barrier();  // nobody frees memory a peer still has open
    CHECK_CUDA(cudaHostUnregister(registered));
    munmap(registered, live);
    CHECK_CUDA(cudaFreeHost(pinned));
    CHECK_CUDA(cudaFree(state));
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && strcmp(argv[1], "--rank") == 0) {  // exec'd worker: --rank <r> <shared file>
        rank = atoi(argv[2]);
        int fd = open(argv[3], O_RDWR);
        shared = static_cast<Shared*>(mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (shared == MAP_FAILED) FAIL("mmap %s", argv[3]);
        return run_rank(shared->world);
    }

    setbuf(stdout, NULL);
    int world = argc > 1 ? atoi(argv[1]) : 0;
    const char* gate = argc > 2 ? argv[2] : "";
    if (world <= 0) CHECK_CUDA(cudaGetDeviceCount(&world));
    if (world > MAX_RANKS) FAIL("world %d > %d", world, MAX_RANKS);
    if (strlen(gate) >= sizeof(Shared::gate)) FAIL("gate path too long");

    char path[] = "/tmp/hostmem_ipc.XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, sizeof(Shared)) != 0) FAIL("shared file %s", path);
    shared = static_cast<Shared*>(mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    memset(shared, 0, sizeof(Shared));
    shared->world = world;
    strcpy(shared->gate, gate);

    // Workers exec this binary afresh (no CUDA state inherited across fork), like vLLM's spawn.
    // pids[r] is 0 once rank r is reaped (or was never started), so only live workers are signalled.
    pid_t pids[MAX_RANKS] = {};
    int rc = 0;
    auto stop_workers = [&] {
        for (int r = 0; r < world; r++)
            if (pids[r] > 0) kill(pids[r], SIGKILL);
    };
    for (int r = 0; r < world; r++) {
        pid_t pid = fork();
        if (pid == 0) {
            std::string rs = std::to_string(r);
            execl("/proc/self/exe", argv[0], "--rank", rs.c_str(), path, nullptr);
            _exit(127);
        }
        if (pid < 0) {
            printf("FAIL fork rank %d: %s\n", r, strerror(errno));
            rc = 1;
            stop_workers();
            break;
        }
        pids[r] = pid;
    }
    // A failed rank leaves its peers waiting at a barrier forever, so take them down with it
    int status;
    pid_t pid;
    while ((pid = wait(&status)) > 0 || (pid < 0 && errno == EINTR)) {
        if (pid < 0) continue;
        for (int r = 0; r < world; r++)
            if (pids[r] == pid) pids[r] = 0;
        if (rc == 0 && (!WIFEXITED(status) || WEXITSTATUS(status) != 0)) {
            printf("FAIL a worker (pid %d) exited with status %d; stopping the others\n", pid, status);
            rc = 1;
            stop_workers();
        }
    }
    unlink(path);
    if (rc == 0) printf("PASS\n");
    return rc;
}
