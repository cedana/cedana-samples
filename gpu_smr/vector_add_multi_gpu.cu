#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <vector>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

std::vector<float *> d_A_list;
std::vector<float *> d_B_list;
std::vector<float *> d_C_list;
float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
int num_gpus = 0;

void handleSignal(int signal) { exit(signal); }

int main() {
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  cudaError_t err = cudaGetDeviceCount(&num_gpus);
  if (err != cudaSuccess) {
    std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  
  if (num_gpus == 0) {
    std::cerr << "No CUDA devices found." << std::endl;
    return 1;
  }
  std::cout << "Found " << num_gpus << " CUDA devices. Running on all devices." << std::endl;

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

  d_A_list.resize(num_gpus);
  d_B_list.resize(num_gpus);
  d_C_list.resize(num_gpus);

  for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(i);
    cudaMalloc((void **)&d_A_list[i], size);
    cudaMalloc((void **)&d_B_list[i], size);
    cudaMalloc((void **)&d_C_list[i], size);

    cudaMemcpy(d_A_list[i], h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_list[i], h_B, size, cudaMemcpyHostToDevice);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  while (true) {
    for (int i = 0; i < num_gpus; ++i) {
      cudaSetDevice(i);
      vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A_list[i], d_B_list[i], d_C_list[i], N);
    }

    for (int i = 0; i < num_gpus; ++i) {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
      // Copy back the result. Since all GPUs do the same calculation on the same data,
      // we can overwrite h_C with any of them.
      cudaMemcpy(h_C, d_C_list[i], size, cudaMemcpyDeviceToHost);
    }

    int i = std::rand() % N;
    std::cout << "Sample result: h_C[" << i << "] = " << h_C[i] << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  return 0;
}
