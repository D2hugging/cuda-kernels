#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: at %s:%d: %s (err_num=%d)\n", __FILE__,     \
              __LINE__, cudaGetErrorString(err), err);                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// print CUDA device info
void printDeviceInfo();

// get optimal block size for a kernel
int getOptimalBlockSize(const void *kernel);

#endif // CUDA_UTILS_H