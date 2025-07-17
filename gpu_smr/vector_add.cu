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

float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;

void handleSignal(int signal) {
  std::cout << "\nSignal received (" << signal
            << "). Cleaning up resources...\n";

  if (d_A)
    cudaFree(d_A);
  if (d_B)
    cudaFree(d_B);
  if (d_C)
    cudaFree(d_C);

  if (h_A)
    free(h_A);
  if (h_B)
    free(h_B);
  if (h_C)
    free(h_C);

  std::cout << "Resources cleaned up. Exiting program.\n";
  exit(signal);
}

int main() {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  int N = 1 << 20; // 1M
  size_t size = N * sizeof(float);

  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  std::srand(std::time(0));
  for (int i = 0; i < N; i++) {
    h_A[i] = i * 0.5f;
    h_B[i] = i * 2.0f;
  }

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  while (true) {
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    int i = std::rand() % N;
    std::cout << "Sample result: h_C[" << i << "] = " << h_C[i] << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  return 0;
}
