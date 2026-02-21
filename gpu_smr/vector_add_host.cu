#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;

void handleSignal(int signal) { exit(signal); }

int main() {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  int N = 1 << 20; // 1M
  size_t size = N * sizeof(float);

  // Allocate pinned host memory
  cudaMallocHost((void **)&h_A, size);
  cudaMallocHost((void **)&h_B, size);
  cudaMallocHost((void **)&h_C, size);

  std::srand(std::time(0));
  for (int i = 0; i < N; i++) {
    h_A[i] = i * 0.5f;
    h_B[i] = i * 2.0f;
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  while (true) {
    // Compute directly on host memory
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(h_A, h_B, h_C, N);
    cudaDeviceSynchronize();

    int i = std::rand() % N;
    std::cout << "Sample result: h_C[" << i << "] = " << h_C[i] << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Cleanup (unreachable in this infinite loop, but good practice)
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);

  return 0;
}
