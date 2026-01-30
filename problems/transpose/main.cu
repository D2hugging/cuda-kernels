#include "benchmark.h"
#include "cuda_utils.h"
#include "test_utils.h"
#include "transpose.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

void cpuTranspose(const float *h_A, float *h_B, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      h_B[j * rows + i] = h_A[i * cols + j];
    }
  }
}

// Track verification failures for exit code
static int g_verificationFailures = 0;

bool verifyTranspose(const float *gpu_out, const float *cpu_out, int rows,
                     int cols, float epsilon = 1e-5f) {
  if (arrayEqual(gpu_out, cpu_out, rows * cols, epsilon)) {
    std::cout << "Verification PASSED\n";
    return true;
  } else {
    std::cerr << "Verification FAILED\n";
    g_verificationFailures++;
    return false;
  }
}

struct MatrixSize {
  size_t rows;
  size_t cols;
  std::string label;
};

int main() {
  printDeviceInfo();

  // test data dims
  std::vector<MatrixSize> test_cases = {
      {256, 256, "256x256"},
      {512, 1024, "512x1024"},
      {1024, 512, "1024x512"},
      {4096, 4096, "4096x4096"},
  };

  // Configure benchmarker: warmup 10, trials 5
  Benchmarker::Config config;
  config.warmup = 10;
  config.trials = 5;
  Benchmarker bench(config);

  for (const auto &test_case : test_cases) {
    std::cout << "\n>>> Testing matrix size: " << test_case.rows << "x"
              << test_case.cols << " ("
              << (test_case.rows * test_case.cols * sizeof(float) /
                  (1024 * 1024))
              << " MB data)" << '\n';

    // cpu
    std::vector<float> h_A(test_case.rows * test_case.cols);
    std::vector<float> h_B(test_case.rows * test_case.cols);
    std::vector<float> h_C(test_case.rows * test_case.cols);

    fillRandom(h_A.data(), test_case.rows * test_case.cols, -1.0f, 1.0f);
    cpuTranspose(h_A.data(), h_B.data(), test_case.rows, test_case.cols);

    // gpu
    float *d_a, *d_b;

    CUDA_CHECK(
        cudaMalloc(&d_a, test_case.rows * test_case.cols * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&d_b, test_case.rows * test_case.cols * sizeof(float)));

    // copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_A.data(),
                          test_case.rows * test_case.cols * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    int gridRows = (test_case.rows + TILE_SIZE - 1) / TILE_SIZE;
    int gridCols = (test_case.cols + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(gridRows, gridCols);

    // naive, coalesced reads
    bench.run<float>(
        "Naive_" + test_case.label, test_case.rows * test_case.cols,
        [&]() {
          transposeNaiveKernel<<<grid, block>>>(d_a, d_b, test_case.rows,
                                                test_case.cols);
        },
        [&]() -> bool {
          cudaMemcpy(h_C.data(), d_b,
                     test_case.rows * test_case.cols * sizeof(float),
                     cudaMemcpyDeviceToHost);
          return verifyTranspose(h_C.data(), h_B.data(), test_case.rows,
                                 test_case.cols);
        },
        2);
    // coalesced writes
    bench.run<float>(
        "CoalescedWrite_" + test_case.label, test_case.rows * test_case.cols,
        [&]() {
          transposeCoalescedWriteKernel<<<grid, block>>>(
              d_a, d_b, test_case.rows, test_case.cols);
        },
        [&]() -> bool {
          cudaMemcpy(h_C.data(), d_b,
                     test_case.rows * test_case.cols * sizeof(float),
                     cudaMemcpyDeviceToHost);
          return verifyTranspose(h_C.data(), h_B.data(), test_case.rows,
                                 test_case.cols);
        },
        2);

    // shared memory with bank conflicts
    bench.run<float>(
        "SharedMem_" + test_case.label, test_case.rows * test_case.cols,
        [&]() {
          transposeSharedMemKernel<<<grid, block>>>(d_a, d_b, test_case.rows,
                                                    test_case.cols);
        },
        [&]() -> bool {
          cudaMemcpy(h_C.data(), d_b,
                     test_case.rows * test_case.cols * sizeof(float),
                     cudaMemcpyDeviceToHost);
          return verifyTranspose(h_C.data(), h_B.data(), test_case.rows,
                                 test_case.cols);
        },
        2);

    // shared memory without bank conflicts
    bench.run<float>(
        "SharedMemPadded_" + test_case.label, test_case.rows * test_case.cols,
        [&]() {
          transposeSharedMemPaddedKernel<<<grid, block>>>(
              d_a, d_b, test_case.rows, test_case.cols);
        },
        [&]() -> bool {
          cudaMemcpy(h_C.data(), d_b,
                     test_case.rows * test_case.cols * sizeof(float),
                     cudaMemcpyDeviceToHost);
          return verifyTranspose(h_C.data(), h_B.data(), test_case.rows,
                                 test_case.cols);
        },
        2);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
  }

  // Print summary table
  bench.printSummary();

  // Save the results to CSV
  bench.exportToCsv("transpose_benchmark.csv");

  // Return non-zero exit code if any verification failed
  if (g_verificationFailures > 0) {
    std::cerr << "\nTotal verification failures: " << g_verificationFailures
              << '\n';
    return 1;
  }

  std::cout << "\nAll verifications passed!\n";
  return 0;
}
