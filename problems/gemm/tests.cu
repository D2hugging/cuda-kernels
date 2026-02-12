#include "cuda_config.h"
#include "cuda_utils.h"
#include "gemm.cuh"
#include "test_harness.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

// CPU reference implementation
static void cpuGemm(const float *h_A, const float *h_B, float *h_C, int M,
                    int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += h_A[i * K + k] * h_B[k * N + j];
      }
      h_C[i * N + j] = sum;
    }
  }
}

// Helper to run a GEMM kernel and get the result
static void runGEMMKernel(const float *h_A, const float *h_B, float *h_C_gpu,
                          int M, int N, int K, const char *kernelName) {
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // Launch appropriate kernel based on name
  if (strcmp(kernelName, "naive") == 0) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemmNaiveKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  } else if (strcmp(kernelName, "tiled") == 0) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemmTiledKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  } else if (strcmp(kernelName, "register") == 0) {
    constexpr int threadsM = BM / TM;
    constexpr int threadsN = BN / TN;
    dim3 block(threadsN, threadsM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemmRegisterBlockingKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  }

  CUDA_CHECK(
      cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// Test: 1x1 matrix
static bool testSingleElement() {
  std::vector<float> A = {3.0f};
  std::vector<float> B = {4.0f};
  std::vector<float> C_gpu(1);
  std::vector<float> C_cpu(1);

  runGEMMKernel(A.data(), B.data(), C_gpu.data(), 1, 1, 1, "naive");
  cpuGemm(A.data(), B.data(), C_cpu.data(), 1, 1, 1);

  return almostEqual(C_gpu[0], C_cpu[0], 1e-5f);
}

// Test: identity matrix multiplication
static bool testIdentityMatrix() {
  const int N = 4;
  std::vector<float> A = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<float> B = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> C_gpu(N * N);

  runGEMMKernel(A.data(), B.data(), C_gpu.data(), N, N, N, "tiled");

  // I * B = B, so C should equal B (use strict tolerance for identity)
  return arrayEqualTol(C_gpu.data(), B.data(), N * N, 1e-5f, 1e-6f);
}

// Test: square matrix
static bool testSquareMatrix() {
  const int N = 64;
  std::vector<float> A(N * N);
  std::vector<float> B(N * N);
  std::vector<float> C_gpu(N * N);
  std::vector<float> C_cpu(N * N);

  fillRandom(A.data(), N * N, -1.0f, 1.0f);
  fillRandom(B.data(), N * N, -1.0f, 1.0f);

  runGEMMKernel(A.data(), B.data(), C_gpu.data(), N, N, N, "register");
  cpuGemm(A.data(), B.data(), C_cpu.data(), N, N, N);

  // Use adaptive tolerance: atol scales with sqrt(K)
  float rtol = 1e-4f;
  float atol = 1e-5f * std::sqrt(static_cast<float>(N));
  return arrayEqualTol(C_gpu.data(), C_cpu.data(), N * N, rtol, atol);
}

// Test: non-square matrix (M ≠ N ≠ K)
static bool testNonSquareMatrix() {
  const int M = 128, N = 64, K = 96;
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_gpu(M * N);
  std::vector<float> C_cpu(M * N);

  fillRandom(A.data(), M * K, -1.0f, 1.0f);
  fillRandom(B.data(), K * N, -1.0f, 1.0f);

  runGEMMKernel(A.data(), B.data(), C_gpu.data(), M, N, K, "tiled");
  cpuGemm(A.data(), B.data(), C_cpu.data(), M, N, K);

  // Use adaptive tolerance: atol scales with sqrt(K)
  float rtol = 1e-4f;
  float atol = 1e-5f * std::sqrt(static_cast<float>(K));
  return arrayEqualTol(C_gpu.data(), C_cpu.data(), M * N, rtol, atol);
}

// Test: non-tile-aligned dimensions
static bool testNonTileAlignedDims() {
  const int M = 63, N = 65, K = 67; // Prime-ish numbers
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_gpu(M * N);
  std::vector<float> C_cpu(M * N);

  fillRandom(A.data(), M * K, -1.0f, 1.0f);
  fillRandom(B.data(), K * N, -1.0f, 1.0f);

  runGEMMKernel(A.data(), B.data(), C_gpu.data(), M, N, K, "register");
  cpuGemm(A.data(), B.data(), C_cpu.data(), M, N, K);

  // Use adaptive tolerance: atol scales with sqrt(K)
  float rtol = 1e-4f;
  float atol = 1e-5f * std::sqrt(static_cast<float>(K));
  return arrayEqualTol(C_gpu.data(), C_cpu.data(), M * N, rtol, atol);
}

// Test: zero matrices
static bool testZeroMatrices() {
  const int N = 32;
  std::vector<float> A(N * N, 0.0f);
  std::vector<float> B(N * N, 0.0f);
  std::vector<float> C_gpu(N * N);

  runGEMMKernel(A.data(), B.data(), C_gpu.data(), N, N, N, "tiled");

  // All zeros expected
  for (int i = 0; i < N * N; ++i) {
    if (std::abs(C_gpu[i]) > 1e-10f)
      return false;
  }
  return true;
}

// Test: large matrix
static bool testLargeMatrix() {
  const int M = 512, N = 512, K = 512;
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_gpu(M * N);
  std::vector<float> C_cpu(M * N);

  fillRandom(A.data(), M * K, -1.0f, 1.0f);
  fillRandom(B.data(), K * N, -1.0f, 1.0f);

  runGEMMKernel(A.data(), B.data(), C_gpu.data(), M, N, K, "register");
  cpuGemm(A.data(), B.data(), C_cpu.data(), M, N, K);

  // Use adaptive tolerance: atol scales with sqrt(K)
  float rtol = 1e-4f;
  float atol = 1e-5f * std::sqrt(static_cast<float>(K));
  return arrayEqualTol(C_gpu.data(), C_cpu.data(), M * N, rtol, atol);
}

int main() {
  printDeviceInfo();

  TestHarness harness("GEMM Edge Case Tests");

  harness.run("single element (1x1)", testSingleElement);
  harness.run("identity matrix (4x4)", testIdentityMatrix);
  harness.run("zero matrices (32x32)", testZeroMatrices);
  harness.run("square matrix (64x64)", testSquareMatrix);
  harness.run("non-square (128x64x96)", testNonSquareMatrix);
  harness.run("non-tile-aligned (63x65x67)", testNonTileAlignedDims);
  harness.run("large matrix (512x512x512)", testLargeMatrix);

  return harness.summarize();
}
