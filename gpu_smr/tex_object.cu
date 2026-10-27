/**
  @file tex_object.cu
  @brief CUDA texture objects created and destroyed before a checkpoint, and read by kernels with
  the handle as a plain parameter and inside a struct parameter, as GROMACS does.

  GROMACS keeps its nonbonded parameter tables in device buffers and makes a texture object on
  each with cudaCreateTextureObject. Its kernels take the handles inside the NBParamGpu struct.
  At each step of the PME tuning it destroys the Coulomb table's texture object and creates a new
  one. The handle is a value in the application's own memory, so an interceptor must give each
  live texture object its old handle after a restore, and must forward the destroy.

  The sample first does 5 rounds of create, read and destroy of a linear texture, as the PME
  tuning does. Then it makes the live objects: a linear texture, read by a kernel that takes the
  handle as a plain parameter, and a pitched 2D texture, read by a kernel that takes the handle
  inside a struct. One more linear texture is created before the live objects and destroyed after
  them, so a checkpoint holds a destroyed object too, and the live handles do not start at the
  lowest slot. The getters cudaGetTextureObjectResourceDesc and
  cudaGetTextureObjectTextureDesc must give back what the creates passed. With --control the
  sample skips the 5 rounds and the extra object, and destroys nothing before the end.

  The pitched buffer comes from cudaMalloc with a pitch rounded up to the device's texture pitch
  alignment, and its rows are laid out on the host and copied with cudaMemcpy. cudaMallocPitch
  and cudaMemcpy2D are left out on purpose: the cedana GPU interceptor answers both with
  cudaErrorStubLibrary (cedana-gpu v0.8.2), and this sample tests texture objects only.

  Each read must give what the CPU computes from the buffer contents. Each iteration changes the
  contents with a kernel, so a read through a texture on the wrong buffer is caught.

  With no gate argument the sample does 100 iterations. With a gate-file path it prints READY
  after the first iteration and iterates until that file exists, so a test can checkpoint and
  restore it mid-work, then create the gate. Prints one ok/FAIL line per step and PASS at the
  end. Exits 0 on PASS, 1 on a failure before the live objects exist, 2 on a failure after, and 3
  on a wrong value.

  Usage: tex_object [--control] [gate-file]
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

static const int N = 4096;  // linear texture elements
static const int W = 256;   // pitched texture width, elements
static const int H = 64;    // pitched texture height

// The handles inside a struct parameter, as GROMACS passes them in NBParamGpu.
struct Params {
  cudaTextureObject_t pitched;
  float scale;
};

static int check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("FAIL %s: %d %s\n", what, (int)e, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

// out[i] = 2 * lin[i], read through the handle passed as a plain parameter
__global__ void readLinear(cudaTextureObject_t tex, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
    out[i] = 2.0f * tex1Dfetch<float>(tex, i);
}

// out[y * W + x] = scale * pitched[y][x], read through the handle inside the struct
__global__ void readPitched(Params p, float *out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < W * H; i += gridDim.x * blockDim.x) {
    int x = i % W, y = i / W;
    out[i] = p.scale * tex2D<float>(p.pitched, x + 0.5f, y + 0.5f);
  }
}

// lin[i] += k + i % 7; the next read must see the new contents
__global__ void bumpLinear(float *lin, float k) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
    lin[i] += k + (float)(i % 7);
}

static cudaResourceDesc linearDesc(float *p) {
  cudaResourceDesc r;
  memset(&r, 0, sizeof(r));
  r.resType = cudaResourceTypeLinear;
  r.res.linear.devPtr = p;
  r.res.linear.desc = cudaCreateChannelDesc<float>();
  r.res.linear.sizeInBytes = N * sizeof(float);
  return r;
}

static cudaResourceDesc pitchedDesc(float *p, size_t pitch) {
  cudaResourceDesc r;
  memset(&r, 0, sizeof(r));
  r.resType = cudaResourceTypePitch2D;
  r.res.pitch2D.devPtr = p;
  r.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  r.res.pitch2D.width = W;
  r.res.pitch2D.height = H;
  r.res.pitch2D.pitchInBytes = pitch;
  return r;
}

static cudaTextureDesc pointDesc() {
  cudaTextureDesc t;
  memset(&t, 0, sizeof(t));
  t.readMode = cudaReadModeElementType;
  t.filterMode = cudaFilterModePoint;
  t.addressMode[0] = cudaAddressModeClamp;
  t.addressMode[1] = cudaAddressModeClamp;
  return t;
}

// Reads the linear texture and compares with 2 * the host copy of its buffer.
static int checkLinear(cudaTextureObject_t tex, float *out, const std::vector<float> &lin) {
  std::vector<float> got(N);
  readLinear<<<16, 256>>>(tex, out);
  if (check(cudaGetLastError(), "readLinear launch")) return 1;
  if (check(cudaMemcpy(got.data(), out, N * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy linear output"))
    return 1;
  for (int i = 0; i < N; i++) {
    if (got[i] != 2.0f * lin[i]) {
      printf("MISMATCH linear element %d: %f, want %f\n", i, got[i], 2.0f * lin[i]);
      return 3;
    }
  }
  return 0;
}

// Reads the pitched texture and compares with scale * the host copy of its buffer.
static int checkPitched(const Params &p, float *out, const std::vector<float> &pit) {
  std::vector<float> got(W * H);
  readPitched<<<16, 256>>>(p, out);
  if (check(cudaGetLastError(), "readPitched launch")) return 1;
  if (check(cudaMemcpy(got.data(), out, W * H * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy pitched output"))
    return 1;
  for (int i = 0; i < W * H; i++) {
    if (got[i] != p.scale * pit[i]) {
      printf("MISMATCH pitched element %d: %f, want %f\n", i, got[i], p.scale * pit[i]);
      return 3;
    }
  }
  return 0;
}

// The getters must give back what the create passed.
static int checkGetters(cudaTextureObject_t tex, float *devPtr) {
  cudaResourceDesc r;
  cudaTextureDesc t;
  if (check(cudaGetTextureObjectResourceDesc(&r, tex), "cudaGetTextureObjectResourceDesc"))
    return 1;
  if (check(cudaGetTextureObjectTextureDesc(&t, tex), "cudaGetTextureObjectTextureDesc"))
    return 1;
  if (r.resType != cudaResourceTypeLinear || r.res.linear.devPtr != devPtr ||
      r.res.linear.sizeInBytes != N * sizeof(float)) {
    printf("MISMATCH resource descriptor: type %d, devPtr %p, size %zu\n", (int)r.resType,
           r.res.linear.devPtr, r.res.linear.sizeInBytes);
    return 3;
  }
  if (t.readMode != cudaReadModeElementType || t.filterMode != cudaFilterModePoint) {
    printf("MISMATCH texture descriptor: readMode %d, filterMode %d\n", (int)t.readMode,
           (int)t.filterMode);
    return 3;
  }
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  bool control = false;
  const char *gate = nullptr;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--control") == 0) {
      control = true;
    } else {
      gate = argv[i];
    }
  }

  float *lin, *pit, *out;
  size_t pitch;
  std::vector<float> hostLin(N), hostPit(W * H);
  for (int i = 0; i < N; i++) hostLin[i] = (float)(i % 101) * 0.5f;
  for (int i = 0; i < W * H; i++) hostPit[i] = (float)(i % 37) - 18.0f;

  if (check(cudaSetDevice(0), "cudaSetDevice")) return 1;
  if (check(cudaMalloc(&lin, N * sizeof(float)), "cudaMalloc linear")) return 1;
  int align;
  if (check(cudaDeviceGetAttribute(&align, cudaDevAttrTexturePitchAlignment, 0),
            "cudaDeviceGetAttribute texture pitch alignment"))
    return 1;
  pitch = (W * sizeof(float) + align - 1) / align * align;
  if (check(cudaMalloc(&pit, pitch * H), "cudaMalloc pitched")) return 1;
  if (check(cudaMalloc(&out, W * H * sizeof(float)), "cudaMalloc output")) return 1;
  if (check(cudaMemcpy(lin, hostLin.data(), N * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy linear input"))
    return 1;
  std::vector<char> rows(pitch * H, 0);
  for (int y = 0; y < H; y++) memcpy(&rows[y * pitch], &hostPit[y * W], W * sizeof(float));
  if (check(cudaMemcpy(pit, rows.data(), pitch * H, cudaMemcpyHostToDevice),
            "cudaMemcpy pitched input"))
    return 1;
  printf("ok   mode %s\n", control ? "control" : "recreate");

  cudaResourceDesc linRes = linearDesc(lin);
  cudaResourceDesc pitRes = pitchedDesc(pit, pitch);
  cudaTextureDesc texDesc = pointDesc();
  int r;

  if (!control) {
    // The PME tuning of GROMACS: create, read and destroy, 5 times.
    for (int round = 0; round < 5; round++) {
      cudaTextureObject_t t;
      if (check(cudaCreateTextureObject(&t, &linRes, &texDesc, nullptr), "create (tuning)"))
        return 1;
      if ((r = checkLinear(t, out, hostLin))) return r == 3 ? 3 : 1;
      if (check(cudaDestroyTextureObject(t), "destroy (tuning)")) return 1;
      printf("ok   round %d: created %llu, read, destroyed\n", round, (unsigned long long)t);
    }
  }

  // A destroyed object in the checkpoint, next to the live ones. It takes the lowest free slot
  // before them, so the live handles do not start at the slot a new context gives first, and a
  // restore must get them back on purpose.
  cudaTextureObject_t extra = 0;
  if (!control &&
      check(cudaCreateTextureObject(&extra, &linRes, &texDesc, nullptr), "create extra"))
    return 2;
  cudaTextureObject_t linTex;
  Params params;
  params.scale = 3.0f;
  if (check(cudaCreateTextureObject(&linTex, &linRes, &texDesc, nullptr), "create linear"))
    return 2;
  if (check(cudaCreateTextureObject(&params.pitched, &pitRes, &texDesc, nullptr),
            "create pitched"))
    return 2;
  if (!control && check(cudaDestroyTextureObject(extra), "destroy extra")) return 2;
  printf("ok   live texture objects: linear %llu, pitched %llu, destroyed extra %llu\n",
         (unsigned long long)linTex, (unsigned long long)params.pitched,
         (unsigned long long)extra);
  if ((r = checkGetters(linTex, lin))) return r == 3 ? 3 : 2;
  printf("ok   getters\n");

  long iterations = 0;
  while (true) {
    if ((r = checkLinear(linTex, out, hostLin))) return r == 3 ? 3 : 2;
    if ((r = checkPitched(params, out, hostPit))) return r == 3 ? 3 : 2;
    float k = (float)(iterations % 5);
    bumpLinear<<<16, 256>>>(lin, k);
    if (check(cudaGetLastError(), "bumpLinear launch")) return 2;
    if (check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return 2;
    for (int i = 0; i < N; i++) hostLin[i] += k + (float)(i % 7);
    iterations++;
    if (iterations == 1) {
      printf("ok   read through both texture objects\n");
      if (gate) printf("READY\n");
    }
    if (gate ? access(gate, F_OK) == 0 : iterations >= 100) break;
    if (gate) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  // One more read after the last change, and the getters, on the restored objects too.
  if ((r = checkLinear(linTex, out, hostLin))) return r == 3 ? 3 : 2;
  if ((r = checkGetters(linTex, lin))) return r == 3 ? 3 : 2;
  printf("ok   %ld iterations\n", iterations);
  if (check(cudaDestroyTextureObject(linTex), "destroy linear")) return 2;
  if (check(cudaDestroyTextureObject(params.pitched), "destroy pitched")) return 2;
  printf("ok   destroyed the live texture objects\n");
  printf("PASS\n");
  return 0;
}
