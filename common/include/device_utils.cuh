#ifndef DEVICE_UTILS_CUH
#define DEVICE_UTILS_CUH

#include <cuda_runtime.h>

// Reusable device-side utility functions for CUDA kernels

/**
 * Performs a block-level reduction in shared memory.
 *
 * Requirements:
 * - smem must have at least blockDim.x elements
 * - tid must be threadIdx.x
 * - All threads in the block must call this function
 * - blockDim.x must be a power of 2
 *
 * After this function returns, smem[0] contains the sum of all elements.
 *
 * @tparam T The data type (typically float or double)
 * @tparam BlockSize The block size (must be known at compile time for
 * optimization)
 * @param smem Pointer to shared memory array
 * @param tid Thread index within the block (threadIdx.x)
 */
template <typename T, int BlockSize>
__device__ __forceinline__ void blockReduceSum(T *smem, int tid) {
  __syncthreads();

#pragma unroll
  for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
}

/**
 * Runtime block size version of blockReduceSum.
 * Use the templated version when block size is known at compile time.
 */
template <typename T>
__device__ __forceinline__ void blockReduceSumDynamic(T *smem, int tid,
                                                      int blockSize) {
  __syncthreads();

  for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
}

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

#endif // DEVICE_UTILS_CUH
