#include "cuda_utils.h"
#include "device_utils.cuh"
#include "transpose.cuh"
#include <cstddef>
#include <cuda_runtime.h>

// naive
__global__ void transposeNaiveKernel(const float *input, float *output,
                                     int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    output[col * rows + row] = input[row * cols + col];
  }
}

// shared memory with bank conflicts
__global__ void transposeSharedMemKernel(const float *input, float *output,
                                         int rows, int cols) {
  __shared__ float smem[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int ty = threadIdx.y; // tile row
  int tx = threadIdx.x; // tile col

  // load data into shared memory (coalesced)
  if (row < rows && col < cols) {
    smem[ty][tx] = input[row * cols + col];
  }

  __syncthreads(); // wait for all threads in the block to finish loading

  // output position (blockIdx swapped)
  int outRow = blockIdx.x * TILE_SIZE + ty;
  int outCol = blockIdx.y * TILE_SIZE + tx;

  // store transposed data from shared memory to global memory (coalesced)
  if (outRow < cols && outCol < rows) {
    output[outRow * rows + outCol] = smem[tx][ty];
  }
}

// shared memory without bank conflicts
__global__ void transposeSharedMemPaddedKernel(const float *input,
                                               float *output, int rows,
                                               int cols) {
  // add padding to avoid bank conflicts
  __shared__ float smem[TILE_SIZE][TILE_SIZE + PADDING];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int ty = threadIdx.y; // tile row
  int tx = threadIdx.x; // tile col

  // load data into shared memory (coalesced)
  if (row < rows && col < cols) {
    smem[ty][tx] = input[row * cols + col];
  }

  __syncthreads(); // wait for all threads in the block to finish loading

  // output position (blockIdx swapped)
  int outRow = blockIdx.x * TILE_SIZE + ty;
  int outCol = blockIdx.y * TILE_SIZE + tx;

  // store transposed data from shared memory to global memory (coalesced)
  if (outRow < cols && outCol < rows) {
    output[outRow * rows + outCol] = smem[tx][ty];
  }
}