/**
  @brief Prints CUDA device properties
 */

#include <stdio.h>

#include "cuda_error.cuh"

int main() {
    int nDevices;

    cudaErrChk(cudaGetDeviceCount(&nDevices));
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaErrChk(cudaGetDeviceProperties(&prop, i));
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total Global Memory (GB): %f\n", prop.totalGlobalMem / 1.0e9);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    }
}
