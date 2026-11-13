// Incremental (delta) checkpoint correctness workload, CPU + GPU.
//
// Holds large buffers in device memory, pageable host memory, pinned host memory and mapped
// (zero-copy) host memory, and each tick changes a small, deterministic part of each, through
// the paths a checkpoint could miss:
//   - a kernel writes a device chunk                         (GPU delta hashing)
//   - the CPU writes a pageable host page                    (CPU soft-dirty tracking)
//   - D2H copies into pinned and pageable host memory        (writes by the GPU/controller, not the CPU)
//   - a kernel writes mapped host memory                     (GPU DMA into host pages)
//   - an H2D copy of a host page into device memory
// When it stops it replays every tick from scratch into fresh buffers and compares each buffer,
// so a checkpoint/restore (or a chain of incremental ones) anywhere in between must not change
// the result.
//
// Usage: delta_verify [dev_mib] [host_mib] [/tmp/gate]
//   Defaults 512 MiB device and 256 MiB pageable host memory (plus 64 MiB pinned, 64 MiB D2H
//   target and 16 MiB mapped). Gate convention: logs READY once set up and "TICK <n>" every 100
//   ticks; with a gate it runs until the gate file exists, without one for 1000 ticks; then logs
//   "PASS <ticks>", or "MISMATCH <buffer> <offset>" (exit 1), or "FAIL ..." on a CUDA error.
//   env DELTA_TICK_MS (default 10) sets the tick interval; DELTA_NO_MAPPED=1 skips the mapped
//   buffer; DELTA_SELFTEST=1 corrupts a byte of each buffer before checking (expects MISMATCH x5).

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CK(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            printf("FAIL %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));   \
            exit(2);                                                                           \
        }                                                                                      \
    } while (0)

static const size_t MIB = 1 << 20;
static const size_t CHUNK = 64 << 10; // device-side unit (GPU delta hashes 64 KiB chunks)
static const size_t PAGE = 4096;
static const size_t PIN_BYTES = 64 * MIB;
static const size_t DST_BYTES = 64 * MIB;
static const size_t MAP_BYTES = 16 * MIB;

__host__ __device__ static inline uint32_t mix(uint64_t a, uint64_t b) {
    uint64_t z = a * 0x9E3779B97F4A7C15ull + b * 0xBF58476D1CE4E5B9ull + 0x94D049BB133111EBull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)(z ^ (z >> 31));
}

__global__ void fill_k(uint32_t *p, size_t n, uint64_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = mix(seed, i);
}

// Overwrite one chunk, and touch one word in another (a partial-chunk write).
__global__ void tick_k(uint32_t *p, size_t chunk, size_t other_word, uint64_t t) {
    size_t words = CHUNK / 4, base = chunk * words;
    for (size_t i = threadIdx.x; i < words; i += blockDim.x)
        p[base + i] = mix(t, base + i) ^ p[base + i];
    if (threadIdx.x == 0)
        p[other_word] += (uint32_t)t;
}

__global__ void mapped_k(uint32_t *m, size_t word, uint64_t t) {
    size_t i = word + threadIdx.x;
    m[i] = mix(t ^ 0x55, i) + m[i];
}

struct State {
    uint32_t *dev;
    uint8_t *host;   // pageable, CPU-written
    uint8_t *pin;    // pinned, D2H target
    uint8_t *dst;    // pageable, D2H target
    uint32_t *map;   // mapped host memory, kernel-written
    uint32_t *map_d; // device pointer for map
};

static size_t dev_bytes, host_bytes;
static bool use_mapped = true;

static void alloc_state(State &s) {
    CK(cudaMalloc(&s.dev, dev_bytes));
    s.host = (uint8_t *)malloc(host_bytes);
    s.dst = (uint8_t *)malloc(DST_BYTES);
    CK(cudaMallocHost(&s.pin, PIN_BYTES));
    s.map = nullptr;
    s.map_d = nullptr;
    if (use_mapped) {
        CK(cudaHostAlloc(&s.map, MAP_BYTES, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer(&s.map_d, s.map, 0));
    }
    if (!s.host || !s.dst) {
        printf("FAIL malloc\n");
        exit(2);
    }
}

static void init_state(State &s) {
    fill_k<<<1024, 256>>>(s.dev, dev_bytes / 4, 1);
    CK(cudaGetLastError());
    uint32_t *h = (uint32_t *)s.host;
    for (size_t i = 0; i < host_bytes / 4; i++)
        h[i] = mix(2, i);
    memset(s.pin, 0, PIN_BYTES);
    memset(s.dst, 0, DST_BYTES);
    if (use_mapped)
        memset(s.map, 0, MAP_BYTES);
    CK(cudaDeviceSynchronize());
}

static void do_tick(State &s, uint64_t t) {
    size_t dev_chunks = dev_bytes / CHUNK, host_pages = host_bytes / PAGE;
    size_t c = (t * 7919) % dev_chunks;
    size_t other = ((t * 104729) % (dev_bytes / 4));

    // GPU writes a device chunk
    tick_k<<<1, 256>>>(s.dev, c, other, t);
    CK(cudaGetLastError());

    // CPU writes a pageable page
    size_t pg = (t * 1299709) % host_pages;
    uint32_t *hp = (uint32_t *)(s.host + pg * PAGE);
    for (size_t i = 0; i < PAGE / 4; i++)
        hp[i] = mix(t ^ 0x77, pg * PAGE / 4 + i) ^ hp[i];

    // D2H of the chunk just written, into pinned and pageable memory
    size_t pin_off = (t % (PIN_BYTES / CHUNK)) * CHUNK;
    size_t dst_off = ((t * 13) % (DST_BYTES / CHUNK)) * CHUNK;
    CK(cudaMemcpy(s.pin + pin_off, (uint8_t *)s.dev + c * CHUNK, CHUNK, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.dst + dst_off, (uint8_t *)s.dev + c * CHUNK, CHUNK, cudaMemcpyDeviceToHost));

    // H2D of the CPU page into another device chunk
    size_t c2 = (t * 15485863) % dev_chunks;
    CK(cudaMemcpy((uint8_t *)s.dev + c2 * CHUNK + (t % (CHUNK / PAGE)) * PAGE, hp, PAGE,
                  cudaMemcpyHostToDevice));

    // GPU writes mapped host memory directly
    if (use_mapped) {
        size_t w = ((t * 31337) % (MAP_BYTES / 4 / 64)) * 64;
        mapped_k<<<1, 64>>>(s.map_d, w, t);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
}

// Compare a device buffer of each state in 64 MiB slices; returns first differing offset or -1.
static long long diff_dev(const uint32_t *a, const uint32_t *b, size_t n) {
    const size_t slice = 64 * MIB;
    uint8_t *x = (uint8_t *)malloc(slice), *y = (uint8_t *)malloc(slice);
    long long res = -1;
    for (size_t off = 0; off < n && res < 0; off += slice) {
        size_t len = n - off < slice ? n - off : slice;
        CK(cudaMemcpy(x, (const uint8_t *)a + off, len, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(y, (const uint8_t *)b + off, len, cudaMemcpyDeviceToHost));
        if (memcmp(x, y, len))
            for (size_t i = 0; i < len; i++)
                if (x[i] != y[i]) {
                    res = (long long)(off + i);
                    break;
                }
    }
    free(x);
    free(y);
    return res;
}

static long long diff_host(const uint8_t *a, const uint8_t *b, size_t n) {
    if (!memcmp(a, b, n))
        return -1;
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i])
            return (long long)i;
    return -1;
}

// Kept in a global so it lives in the checkpointed memory like everything else.
static volatile uint64_t g_tick = 0;

int main(int argc, char **argv) {
    dev_bytes = (argc > 1 ? strtoull(argv[1], 0, 10) : 512) * MIB;
    host_bytes = (argc > 2 ? strtoull(argv[2], 0, 10) : 256) * MIB;
    const char *gate = argc > 3 ? argv[3] : nullptr;
    long tick_ms = getenv("DELTA_TICK_MS") ? strtol(getenv("DELTA_TICK_MS"), 0, 10) : 10;
    use_mapped = !getenv("DELTA_NO_MAPPED");
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (dev_bytes < CHUNK || host_bytes < PAGE) {
        printf("FAIL sizes too small\n");
        return 2;
    }

    State s;
    alloc_state(s);
    init_state(s);
    printf("READY dev=%zuMiB host=%zuMiB pinned=%zuMiB mapped=%d pid=%d\n", dev_bytes / MIB,
           host_bytes / MIB, PIN_BYTES / MIB, use_mapped, getpid());

    for (;;) {
        if (gate ? access(gate, F_OK) == 0 : g_tick >= 1000)
            break;
        uint64_t t = g_tick + 1;
        do_tick(s, t);
        g_tick = t;
        if (t % 100 == 0)
            printf("TICK %llu\n", (unsigned long long)t);
        usleep(tick_ms * 1000);
    }
    uint64_t ticks = g_tick;

    // DELTA_SELFTEST=1 corrupts one byte of each buffer, to check the comparison catches it
    if (getenv("DELTA_SELFTEST")) {
        CK(cudaMemset((uint8_t *)s.dev + 12345, 0xAB, 1));
        s.host[12345] ^= 1;
        s.pin[12345] ^= 1;
        s.dst[12345] ^= 1;
        if (use_mapped)
            ((uint8_t *)s.map)[12345] ^= 1;
        CK(cudaDeviceSynchronize());
    }

    // Replay from scratch and compare
    State r;
    alloc_state(r);
    init_state(r);
    for (uint64_t t = 1; t <= ticks; t++)
        do_tick(r, t);

    struct {
        const char *name;
        long long off;
    } res[] = {
        {"device", diff_dev(s.dev, r.dev, dev_bytes)},
        {"host-pageable", diff_host(s.host, r.host, host_bytes)},
        {"host-pinned-d2h", diff_host(s.pin, r.pin, PIN_BYTES)},
        {"host-pageable-d2h", diff_host(s.dst, r.dst, DST_BYTES)},
        {"host-mapped", use_mapped ? diff_host((uint8_t *)s.map, (uint8_t *)r.map, MAP_BYTES) : -1},
    };
    int bad = 0;
    for (auto &x : res)
        if (x.off >= 0) {
            printf("MISMATCH %s %lld\n", x.name, x.off);
            bad = 1;
        }
    if (!bad)
        printf("PASS %llu\n", (unsigned long long)ticks);
    return bad;
}
