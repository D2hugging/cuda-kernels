#include "cuda_utils.h"
#include "dot_product.cuh"
#include <cstddef>
#include <cuda_runtime.h>

// atomic add
__global__ void dotProductAtomicKernel(const float *a, const float *b,
                                       float *partialSum, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  float sum = 0.0f;
  // grid-stride loop
  for (int i = idx; i < n; i += stride) {
    sum += a[i] * b[i];
  }

  atomicAdd(&partialSum[blockIdx.x], sum);
}

__global__ void dotProductSharedMemKernel(const float *a, const float *b,
                                          float *partialSum, size_t n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  float sum = 0.0f;
  // grid-stride loop
  for (int i = idx; i < n; i += stride) {
    sum += a[i] * b[i];
  }
  // write to shared memory
  __shared__ float smem[BLOCK_SIZE];
  smem[tid] = sum;
  __syncthreads();

  // reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  // write block result
  if (tid == 0) {
    partialSum[blockIdx.x] = smem[0];
  }
}

__global__ void partialSumReductionKernel(float *partialSum, float *result,
                                          int numBlocks) {
  __shared__ float smem[BLOCK_SIZE];
  int tid = threadIdx.x;

  // load data into shared memory
  float sum = 0.0f;
  for (int i = tid; i < numBlocks; i += blockDim.x) {
    sum += partialSum[i];
  }
  smem[tid] = sum;
  __syncthreads();

  // reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  // write final result
  if (tid == 0) {
    *result = smem[0];
  }
}
