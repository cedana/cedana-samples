/**
 * @file cuda_error.cuh
 * @brief CUDA error handling helpers
 *
 * This code is courtesy of @talonmies
 * (https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590)
 */

#ifndef CUDA_ERROR_CUH
#define CUDA_ERROR_CUH

#define cudaErrChk(ans) \
    { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif  // CUDA_ERROR_CUH
