#include "benchmark.h"
#include "cuda_utils.h"
#include "gemm.cuh"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

void cpuGemm(const float *h_A, const float *h_B, float *h_C, int M, int N,
             int K) {
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

// Track verification failures for exit code
static int g_verificationFailures = 0;

bool verifyGemm(const float *gpu_out, const float *cpu_out, int M, int N,
                float epsilon = 1e-5f) {
  if (arrayEqual(gpu_out, cpu_out, M * N, epsilon)) {
    std::cout << "Verification PASSED\n";
    return true;
  } else {
    std::cerr << "Verification FAILED\n";
    g_verificationFailures++;
    return false;
  }
}

struct MatrixSize {
  int M; // rows of A , rows of C
  int N; // cols of B , cols of C
  int K; // cols of A , rows of B
  std::string label;
};

int main() {
  printDeviceInfo();

  // test data dims
  std::vector<MatrixSize> test_cases = {
      {512, 512, 512, "512x512x512"},
      {512, 512, 1024, "512x512x1024"},
      {1024, 1024, 512, "1024x1024x512"},
      {1024, 1024, 1024, "1024x1024x1024"},
      {2048, 2048, 2048, "2048x2048x2048"},
      {3072, 3072, 3072, "3072x3072x3072"},
      {4096, 4096, 4096, "4096x4096x4096"},
      {8192, 8192, 8192, "8192x8192x8192"},
  };

  // Configure benchmarker: warmup 10, trials 5
  Benchmarker::Config config;
  config.warmup = 10;
  config.trials = 5;
  Benchmarker bench(config);

  for (const auto &tc : test_cases) {
    std::cout << "\n>>> Testing matrix size: " << tc.M << "x" << tc.N << "x"
              << tc.K << " ("
              << ((tc.M * tc.K + tc.K * tc.N + tc.M * tc.N) * sizeof(float) /
                  (1024 * 1024))
              << " MB data)" << '\n';

    // cpu
    std::vector<float> h_A(tc.M * tc.K);
    std::vector<float> h_B(tc.K * tc.N);
    std::vector<float> h_C(tc.M * tc.N);
    std::vector<float> h_C_gpu(tc.M * tc.N);

    fillRandom(h_A.data(), tc.M * tc.K, -1.0f, 1.0f);
    fillRandom(h_B.data(), tc.K * tc.N, -1.0f, 1.0f);

    cpuGemm(h_A.data(), h_B.data(), h_C.data(), tc.M, tc.N, tc.K);

    // gpu
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, tc.M * tc.K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, tc.K * tc.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, tc.M * tc.N * sizeof(float)));

    // copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), tc.M * tc.K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), tc.K * tc.N * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid((tc.N + TILE_SIZE - 1) / TILE_SIZE,
              (tc.M + TILE_SIZE - 1) / TILE_SIZE); // (x,y,z) x:N, y:M

    // GEMM FLOPs = 2 * M * N * K (multiply + add for each of M*N*K operations)
    const size_t flops = 2ULL * tc.M * tc.N * tc.K;

    // naive kernel
    bench.run<float>(
        "Naive_" + tc.label, (size_t)tc.M * tc.N, // n = output elements
        [&]() {
          gemmNaiveKernel<<<grid, block>>>(d_A, d_B, d_C, tc.M, tc.N, tc.K);
        },
        [&]() -> bool {
          cudaMemcpy(h_C_gpu.data(), d_C, tc.M * tc.N * sizeof(float),
                     cudaMemcpyDeviceToHost);
          return verifyGemm(h_C_gpu.data(), h_C.data(), tc.M, tc.N);
        },
        2,      // memAccessFactor (for bandwidth calculation)
        flops); // totalFlops = 2 * M * N * K

    // tiled kernel with shared memory
    bench.run<float>(
        "Tiled_" + tc.label, (size_t)tc.M * tc.N, // n = output elements
        [&]() {
          gemmTiledKernel<<<grid, block>>>(d_A, d_B, d_C, tc.M, tc.N, tc.K);
        },
        [&]() -> bool {
          cudaMemcpy(h_C_gpu.data(), d_C, tc.M * tc.N * sizeof(float),
                     cudaMemcpyDeviceToHost);
          return verifyGemm(h_C_gpu.data(), h_C.data(), tc.M, tc.N);
        },
        2,      // memAccessFactor
        flops); // totalFlops = 2 * M * N * K

    // register blocking kernel with shared memory

    // Register blocking kernel
    constexpr int threadsM = BM / TM;  // 32/4 = 8
    constexpr int threadsN = BN / TN;  // 32/4 = 8
    dim3 blockReg(threadsN, threadsM); // 8×8 = 64 threads
    dim3 gridReg((tc.N + BN - 1) / BN, (tc.M + BM - 1) / BM);
    bench.run<float>(
        "RegisterBlocking_" + tc.label,
        (size_t)tc.M * tc.N, // n = output elements
        [&]() {
          gemmRegisterBlockingKernel<<<gridReg, blockReg>>>(d_A, d_B, d_C, tc.M,
                                                            tc.N, tc.K);
        },
        [&]() -> bool {
          cudaMemcpy(h_C_gpu.data(), d_C, tc.M * tc.N * sizeof(float),
                     cudaMemcpyDeviceToHost);
          return verifyGemm(h_C_gpu.data(), h_C.data(), tc.M, tc.N);
        },
        2,      // memAccessFactor
        flops); // totalFlops = 2 * M * N * K

    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }

  // Print summary table
  bench.printSummary();

  // Save the results to CSV
  bench.exportToCsv("gemm_benchmark.csv");

  // Return non-zero exit code if any verification failed
  if (g_verificationFailures > 0) {
    std::cerr << "\nTotal verification failures: " << g_verificationFailures
              << '\n';
    return 1;
  }

  std::cout << "\nAll verifications passed!\n";
  return 0;
}
