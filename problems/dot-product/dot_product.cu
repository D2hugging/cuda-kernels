#include "cuda_utils.h"
#include "device_utils.cuh"
#include "dot_product.cuh"
#include <cstddef>
#include <cuda_runtime.h>

__global__ void dotProductStage1NaiveKernel(const float *a, const float *b,
                                            float *result, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  float sum = 0.0f;
  // grid-stride loop
  for (int i = idx; i < n; i += stride) {
    sum += a[i] * b[i];
  }

  atomicAdd(result, sum);
}

// atomic add
__global__ void dotProductStage2NaiveKernel(const float *a, const float *b,
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
  // write to shared memory and perform block-level reduction
  __shared__ float smem[BLOCK_SIZE];
  smem[tid] = sum;

  blockReduceSum<float, BLOCK_SIZE>(smem, tid);

  // write block result
  if (tid == 0) {
    partialSum[blockIdx.x] = smem[0];
  }
}

__global__ void dotProductWarpShuffleKernel(const float *a, const float *b,
                                            float *partialSum, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float sum = 0.0f;

  for (int i = idx; i < n; i += stride) {
    sum += a[i] * b[i];
  }

  // warp-level reduction using shuffle
  sum = warpReduceSum<float>(sum);

  static_assert(BLOCK_SIZE % 32 == 0,
                "BLOCK_SIZE must be multiple of warp size");
  __shared__ float smem[BLOCK_SIZE / 32]; // one per warp
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  if (lane == 0) { // lane 0 of each warp
    smem[wid] = sum;
  }

  __syncthreads();

  if (wid == 0) {
    float finalVal = (threadIdx.x < (blockDim.x / 32)) ? smem[lane] : 0.0f;
    finalVal = warpReduceSum<float>(finalVal);
    if (lane == 0) {
      partialSum[blockIdx.x] = finalVal;
    }
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

  // perform block-level reduction
  blockReduceSum<float, BLOCK_SIZE>(smem, tid);

  // write final result
  if (tid == 0) {
    *result = smem[0];
  }
}
