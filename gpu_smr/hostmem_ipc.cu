// Multi-GPU C/R check for host memory mappings and CUDA IPC: one process per GPU (exec'd like vLLM
// TP workers), each holding GPU state that must survive restore, with neighbours reading each
// other's device memory through CUDA IPC every step.
//
// Rank 0 first sweeps host memory registration across sizes that broke before (the 8 GiB boundary,
// vLLM's ~12 GB CPU offload buffer): cudaHostRegister and cuMemHostRegister on malloc'd and mmap'd
// memory, cudaHostAlloc, and pinned allocations after a register attempt. It then keeps a 9 GiB
// registered buffer live, so a checkpoint has to carry registered host memory.
//
// Every second rank 0 prints `STEP <n> ...`. Any lost device/host state or broken IPC mapping
// prints FAIL and exits non-zero. Usage: hostmem_ipc [world]  (default: all visible GPUs)

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
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

#define FAIL(...)                                      \
    do {                                               \
        printf("FAIL rank=%d ", rank);                 \
        printf(__VA_ARGS__);                           \
        printf("\n");                                  \
        fflush(stdout);                                \
        exit(1);                                       \
    } while (0)

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) FAIL("%s: %s", #call, cudaGetErrorString(e));    \
    } while (0)

// Shared between ranks through a MAP_SHARED file: a sense-reversing barrier and the IPC handles.
struct Shared {
    int world;
    volatile int arrived;
    volatile int generation;
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

// Pinned DMA actually works: fill on the host, copy to the GPU asynchronously, compare there.
static bool copy_ok(void* host, size_t size, uint8_t value) {
    static uint8_t* scratch;
    static unsigned long long* count;
    if (!scratch) {
        CHECK_CUDA(cudaMalloc(&scratch, CHECK));
        CHECK_CUDA(cudaMalloc(&count, sizeof(*count)));
    }
    size_t n = std::min(size, CHECK);
    memset(host, value, n);
    CHECK_CUDA(cudaMemset(count, 0, sizeof(*count)));
    CHECK_CUDA(cudaMemcpyAsync(scratch, host, n, cudaMemcpyHostToDevice));
    mismatches<<<256, 256>>>(scratch, n, value, count);
    unsigned long long bad = 0;
    CHECK_CUDA(cudaMemcpy(&bad, count, sizeof(bad), cudaMemcpyDeviceToHost));
    return bad == 0;
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
        if (!copy_ok(p, size, 7)) FAIL("cudaHostRegister %zu B copy mismatch", size);
        CHECK_CUDA(cudaHostUnregister(p));
        munmap(p, size);

        p = alloc_touched(size);  // driver API
        if (int rc = cuRegister(p, size, 0)) FAIL("cuMemHostRegister %zu B rc=%d", size, rc);
        if (!copy_ok(p, size, 8)) FAIL("cuMemHostRegister %zu B copy mismatch", size);
        cuUnregister(p);
        munmap(p, size);

        CHECK_CUDA(cudaHostAlloc(&p, size, cudaHostAllocDefault));
        if (!copy_ok(p, size, 9)) FAIL("cudaHostAlloc %zu B copy mismatch", size);
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
        if (!copy_ok(p, 2 * MiB, 11)) FAIL("cudaHostAlloc after register attempt %d: copy mismatch", attempt);
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
        if (first != step || last != step) FAIL("step=%u device state=%u/%u", step, first, last);
        if (theirs != step) FAIL("step=%u peer state via IPC=%u", step, theirs);
        if (!copy_ok(pinned, 4 * MiB, step % 256)) FAIL("step=%u pinned copy mismatch", step);
        if (!copy_ok(registered, live, step % 256)) FAIL("step=%u registered copy mismatch (%zu B)", step, live);
        if (rank == 0) {
            printf("STEP %u state=%u world=%d\n", step, first, world);
            fflush(stdout);
        }
        barrier();  // nobody starts the next increment until every peer read is done
        sleep(1);
    }
}

int main(int argc, char** argv) {
    if (argc == 4 && strcmp(argv[1], "--rank") == 0) {  // exec'd worker: --rank <r> <shared file>
        rank = atoi(argv[2]);
        int fd = open(argv[3], O_RDWR);
        shared = static_cast<Shared*>(mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (shared == MAP_FAILED) FAIL("mmap %s", argv[3]);
        return run_rank(shared->world);
    }

    int world = argc > 1 ? atoi(argv[1]) : 0;
    if (world <= 0) CHECK_CUDA(cudaGetDeviceCount(&world));
    if (world > MAX_RANKS) FAIL("world %d > %d", world, MAX_RANKS);

    char path[] = "/tmp/hostmem_ipc.XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, sizeof(Shared)) != 0) FAIL("shared file %s", path);
    shared = static_cast<Shared*>(mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    memset(shared, 0, sizeof(Shared));
    shared->world = world;

    // Workers exec this binary afresh (no CUDA state inherited across fork), like vLLM's spawn
    pid_t pids[MAX_RANKS];
    for (int r = 0; r < world; r++) {
        if ((pids[r] = fork()) == 0) {
            std::string rs = std::to_string(r);
            execl("/proc/self/exe", argv[0], "--rank", rs.c_str(), path, nullptr);
            _exit(127);
        }
    }
    // A failed rank leaves its peers waiting at a barrier forever, so take them down with it
    int status, rc = 0;
    while (wait(&status) > 0) {
        if (rc == 0 && (!WIFEXITED(status) || WEXITSTATUS(status) != 0)) {
            rc = 1;
            for (int r = 0; r < world; r++) kill(pids[r], SIGKILL);
        }
    }
    unlink(path);
    return rc;
}
