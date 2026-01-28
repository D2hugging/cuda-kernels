#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include <cuda_runtime.h>
#include "cuda_config.h"

// Use centralized block size constant
#define BLOCK_SIZE DEFAULT_BLOCK_SIZE

// one staged
// stage 1: grid-stride loop + atomic add -> result
__global__ void dotProductStage1NaiveKernel(const float *a, const float *b,
                                            float *result, size_t n);
// two staged
// stage 1: grid-stride loop + atomic add -> block partial sum
__global__ void dotProductStage2NaiveKernel(const float *a, const float *b,
                                            float *partialSum, size_t n);

// stage 1: grid-stride loop + shared memory + reduction -> block
__global__ void dotProductSharedMemKernel(const float *a, const float *b,
                                          float *partialSum, size_t n);

// TODO
__global__ void dotProductWarpShuffleKernel(const float *a, const float *b,
                                            float *partialSum, size_t n);

// stage 2: block partial sum + reduction -> result
__global__ void partialSumReductionKernel(float *partialSum, float *result,
                                          int numBlocks);

#endif