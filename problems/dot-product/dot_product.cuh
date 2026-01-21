#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// two staged
// stage 1: grid-stride loop + atomic add -> block partial sum
__global__ void dotProductAtomicKernel(const float *a, const float *b,
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