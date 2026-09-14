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
        // CUDA 13 removed cudaDeviceProp::memoryClockRate; the device
        // attribute works across CUDA versions.
        int memoryClockRate;
        cudaErrChk(cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, i));
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0 * memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total Global Memory (GB): %f\n", prop.totalGlobalMem / 1.0e9);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    }
}
