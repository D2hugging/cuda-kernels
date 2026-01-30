#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "cuda_config.h"
#include <cuda_runtime.h>

#define TILE_SIZE 32 // tile size for shared memory
#define PADDING 1    // for shared memory padding to avoid bank conflicts

// naive
__global__ void transposeNaiveKernel(const float *input, float *output,
                                     int rows, int cols);

// shared memory with bank conflicts
__global__ void transposeSharedMemKernel(const float *input, float *output,
                                         int rows, int cols);

// shared memory without bank conflicts
__global__ void transposeSharedMemPaddedKernel(const float *input,
                                               float *output, int rows,
                                               int cols);

#endif