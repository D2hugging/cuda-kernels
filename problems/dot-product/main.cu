#include "benchmark.h"
#include "cuda_utils.h"
#include "dot_product.cuh"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

float cpuDotProduct(const float *a, const float *b, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(a[i]) * b[i];
  }
  return static_cast<float>(sum);
}

void verifyDotProduct(float gpu_res, float cpu_res, float epsilon = 1e-5f) {
  if (almostEqual(gpu_res, cpu_res, epsilon)) {
    std::cout << "Verification PASSED: GPU result = " << gpu_res
              << ", CPU result = " << cpu_res << '\n';
  } else {
    std::cerr << "Verification FAILED: GPU result = " << gpu_res
              << ", CPU result = " << cpu_res << '\n';
  }
}

int main() {
  printDeviceInfo();

  // test data sizes
  std::vector<size_t> n_sizes = {256 * 1024,        512 * 1024,
                                 1024 * 1024,       128 * 1024 * 1024,
                                 256 * 1024 * 1024, 512 * 1024 * 1024,
                                 768 * 1024 * 1024, 1024 * 1024 * 1024};

  // 10 warmups，100 samples
  Benchmarker bench(10, 100);

  for (size_t n : n_sizes) {
    std::cout << "\n>>> Testing N = " << n << " ("
              << (n * sizeof(float) * 2 / (1024 * 1024)) << " MB data)" << '\n';

    // cpu
    std::vector<float> a(n);
    std::vector<float> b(n);
    fillRandom(a.data(), n, -1.0f, 1.0f);
    fillRandom(b.data(), n, -1.0f, 1.0f);
    float cpu_res = cpuDotProduct(a.data(), b.data(), n);

    // gpu
    float *d_a, *d_b, *d_partialSum, *d_res;
    int numBlocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 1024);

    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partialSum, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));

    // copy data to device
    CUDA_CHECK(
        cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // label
    std::string label = "N_" + std::to_string(n / 1024) + "K";

    bench.run(
        "Atomic_" + label, n,
        [&]() {
          cudaMemsetAsync(d_partialSum, 0, numBlocks * sizeof(float), 0);
          dotProductAtomicKernel<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b,
                                                            d_partialSum, n);
          partialSumReductionKernel<<<1, BLOCK_SIZE>>>(d_partialSum, d_res,
                                                       numBlocks);
        },
        [&]() {
          // verify
          float h_res;
          cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
          verifyDotProduct(h_res, cpu_res);
        });

    // shared memory implementation
    bench.run(
        "SharedMem_" + label, n,
        [&]() {
          dotProductSharedMemKernel<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b,
                                                               d_partialSum, n);
          partialSumReductionKernel<<<1, BLOCK_SIZE>>>(d_partialSum, d_res,
                                                       numBlocks);
        },
        [&]() {
          // verify
          float h_res;
          cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
          verifyDotProduct(h_res, cpu_res);
        });

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partialSum);
    cudaFree(d_res);
  }

  // 2. save the results to CSV
  bench.exportToCsv("dot_product_benchmark.csv");

  // save the results to JSON
  // std::ofstream jsonFile("report.json");
  // jsonFile << bench.exportToJson();

  return 0;
}
