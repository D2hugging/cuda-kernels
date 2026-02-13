#include "cuda_utils.h"
#include "device_utils.cuh"
#include "gemm.cuh"
#include <cstddef>
#include <cuda_runtime.h>

// naive kernel
__global__ void gemmNaiveKernel(const float *A, const float *B, float *C, int M, int N, int K) {
  // which row of C this thread handles
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // which column of C this thread handles
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // boundary check
  if (row < M && col < N) {
    float sum = 0.0f;
    // dot product of row `row` of A and column `col` of B
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    // write result to MxN matrix C
    C[row * N + col] = sum;
  }
}

__global__ void gemmTiledKernel(const float *A, const float *B, float *C, int M, int N, int K) {
  // shared memory for tiles of A and B
  __shared__ float smemA[TILE_SIZE][TILE_SIZE];
  __shared__ float smemB[TILE_SIZE][TILE_SIZE];

  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  // which row and column of C this thread handles
  const int64_t row = by * TILE_SIZE + ty;
  const int64_t col = bx * TILE_SIZE + tx;

  float sum = 0.0f;
  // loop over tiles
  const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
  for (int ti = 0; ti < numTiles; ++ti) {
    const int kBase = ti * TILE_SIZE;

    // load tile of A into shared memory
    const int aCol = kBase + tx;
    smemA[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

    // load tile of B into shared memory
    const int bRow = kBase + ty;
    smemB[ty][tx] = (col < N && bRow < K) ? B[bRow * N + col] : 0.0f;

    __syncthreads(); // wait for all threads to load data

// compute partial sum for this tile
#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += smemA[ty][k] * smemB[k][tx];
    }

    __syncthreads(); // wait for all threads to finish computing
  }

  // write result to C
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

__global__ void gemmRegisterBlockingKernel(const float *A, const float *B, float *C, int M, int N, int K) {
  __shared__ float smemA[BM][BK];
  __shared__ float smemB[BK][BN];

  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int tid = ty * blockDim.x + tx;

  const int numThreads = blockDim.y * blockDim.x;

  // starting position in C for this thread's TMxTN tile
  const int cRowStart = by * BM + ty * TM;
  const int cColStart = bx * BN + tx * TN;

  float regC[TM][TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  const int loadACount = (BM * BK) / numThreads;
  const int loadBCount = (BK * BN) / numThreads;

  const int numTiles = (K + BK - 1) / BK;

  for (int ti = 0; ti < numTiles; ++ti) {
    const int kBase = ti * BK;
// Load smemA[BM][BK] - each thread loads multiple elements
#pragma unroll
    for (int i = 0; i < loadACount; ++i) {
      const int idx = tid + i * numThreads;
      const int smemRow = idx / BK;
      const int smemCol = idx % BK;
      const int globalRow = by * BM + smemRow;
      const int globalCol = kBase + smemCol;

      smemA[smemRow][smemCol] = (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
    }

// Load smemB[BK][BN] - each thread loads multiple elements
#pragma unroll
    for (int i = 0; i < loadBCount; ++i) {
      const int idx = tid + i * numThreads;
      const int smemRow = idx / BN;
      const int smemCol = idx % BN;
      const int globalRow = kBase + smemRow;
      const int globalCol = bx * BN + smemCol;

      smemB[smemRow][smemCol] = (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
    }
    // wait
    __syncthreads();

// For each k in the BK tile
#pragma unroll
    for (int k = 0; k < BK; ++k) {
// Load TM elements from smemA into registers
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        regA[m] = smemA[ty * TM + m][k];
      }

// Load TN elements from smemB into registers
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        regB[n] = smemB[k][tx * TN + n];
      }

// Outer product: TM × TN multiply-adds
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
          regC[m][n] += regA[m] * regB[n];
        }
      }
    }
    // wait
    __syncthreads();
  }

// write TM×TN results to global memory
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int globalRow = cRowStart + m;
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int globalCol = cColStart + n;
      if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = regC[m][n];
      }
    }
  }
}

// helper function for vectorized float4 loads with boundary checks
__device__ __forceinline__ float4 loadFloat4Aligned(const float *ptr, int row, int colBase, int rowLimit,
                                                    int colLimit) {
  float4 result;
  const float *addr = &ptr[row * colLimit + colBase];

  // Check if we can safely load 4 consecutive elements
  if (row < rowLimit && colBase + 3 < colLimit) {
    // Vectorized load (4× faster)
    result = *reinterpret_cast<const float4 *>(addr);
  } else {
    // Boundary case: scalar loads with zero padding
    result.x = (row < rowLimit && colBase + 0 < colLimit) ? ptr[row * colLimit + colBase + 0] : 0.0f;
    result.y = (row < rowLimit && colBase + 1 < colLimit) ? ptr[row * colLimit + colBase + 1] : 0.0f;
    result.z = (row < rowLimit && colBase + 2 < colLimit) ? ptr[row * colLimit + colBase + 2] : 0.0f;
    result.w = (row < rowLimit && colBase + 3 < colLimit) ? ptr[row * colLimit + colBase + 3] : 0.0f;
  }

  return result;
}

// vectorized register blocking kernel with float4 loads
__global__ void gemmRegisterBlockingVectorizedKernel(const float *A, const float *B, float *C, int M, int N, int K) {
  // Shared memory for tiles of A and B
  __shared__ float smemA[BM][BK];
  __shared__ float smemB[BK][BN];

  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int tid = ty * blockDim.x + tx;
  const int numThreads = blockDim.y * blockDim.x;

  // Starting position in C for this thread's TM×TN tile
  const int cRowStart = by * BM + ty * TM;
  const int cColStart = bx * BN + tx * TN;

  // Register arrays for accumulation and temporary storage
  float regC[TM][TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  // Calculate how many elements each thread needs to load
  // load in float4 units, so divide by 4
  const int elemsA = BM * BK;                     // Total elements in tile A
  const int elemsB = BK * BN;                     // Total elements in tile B
  const int loadAVec = elemsA / (numThreads * 4); // float4 vectors per thread for A
  const int loadBVec = elemsB / (numThreads * 4); // float4 vectors per thread for B

  const int numTiles = (K + BK - 1) / BK;

  for (int ti = 0; ti < numTiles; ++ti) {
    const int kBase = ti * BK;

    // load tile of A using float4 vectorized loads
#pragma unroll
    for (int i = 0; i < loadAVec; ++i) {
      // each thread loads one float4 (4 consecutive floats)
      const int vecIdx = tid + i * numThreads; // index in float4 units
      const int flatIdx = vecIdx * 4;          // convert to flat float index

      const int smemRow = flatIdx / BK;     // which row in shared memory
      const int smemColBase = flatIdx % BK; // base column (multiple of 4)

      const int globalRow = by * BM + smemRow;       // global row in matrix A
      const int globalColBase = kBase + smemColBase; // global column base in matrix A

      // load 4 consecutive elements using float4
      float4 data = loadFloat4Aligned(A, globalRow, globalColBase, M, K);

      // store to shared memory
      if (smemRow < BM) { // Boundary check for shared memory
        smemA[smemRow][smemColBase + 0] = data.x;
        smemA[smemRow][smemColBase + 1] = data.y;
        smemA[smemRow][smemColBase + 2] = data.z;
        smemA[smemRow][smemColBase + 3] = data.w;
      }
    }

    // load tile of B using float4 vectorized loads
#pragma unroll
    for (int i = 0; i < loadBVec; ++i) {
      const int vecIdx = tid + i * numThreads;
      const int flatIdx = vecIdx * 4;

      const int smemRow = flatIdx / BN;
      const int smemColBase = flatIdx % BN;

      const int globalRow = kBase + smemRow;
      const int globalColBase = bx * BN + smemColBase;

      // load 4 consecutive elements using float4
      float4 data = loadFloat4Aligned(B, globalRow, globalColBase, K, N);

      // store to shared memory
      if (smemRow < BK) {
        smemB[smemRow][smemColBase + 0] = data.x;
        smemB[smemRow][smemColBase + 1] = data.y;
        smemB[smemRow][smemColBase + 2] = data.z;
        smemB[smemRow][smemColBase + 3] = data.w;
      }
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // load TM elements from smemA into registers
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        regA[m] = smemA[ty * TM + m][k];
      }

      // load TN elements from smemB into registers
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        regB[n] = smemB[k][tx * TN + n];
      }

      // outer product: TM × TN multiply-adds
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          regC[m][n] += regA[m] * regB[n];
        }
      }
    }

    __syncthreads();
  }

  // write TM×TN results to global memory
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int globalRow = cRowStart + m;
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int globalCol = cColStart + n;
      if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = regC[m][n];
      }
    }
  }
}