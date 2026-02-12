#ifndef GEMM_H
#define GEMM_H

#include "cuda_config.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE DEFAULT_BLOCK_SIZE // block size for kernels
#define TILE_SIZE 16                  // tile size for shared memory
#define BM 64                         // Block tile size in M
#define BN 64                         // Block tile size in N
#define BK 8                          // Block tile size in K
#define TM 4                          // each Thread computes TM rows
#define TN 4                          // each Thread computes TN cols

// naive kernel
__global__ void gemmNaiveKernel(const float *A, const float *B, float *C, int M,
                                int N, int K);

// tiling kernel
__global__ void gemmTiledKernel(const float *A, const float *B, float *C, int M,
                                int N, int K);

// register blocking kernel
__global__ void gemmRegisterBlockingKernel(const float *A, const float *B,
                                           float *C, int M, int N, int K);

// warp shuffle kernel
__global__ void gemmWarpShuffleKernel(const float *A, const float *B, float *C,
                                      int M, int N, int K);

#endif