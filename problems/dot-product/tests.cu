#include "cuda_config.h"
#include "cuda_utils.h"
#include "dot_product.cuh"
#include "test_harness.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

// CPU reference implementation
static float cpuDotProduct(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(sum);
}

// Helper to run a dot product kernel and get the result
static float runDotProductSharedMem(const float* h_a, const float* h_b, size_t n) {
    float *d_a, *d_b, *d_partialSum, *d_res;
    int numBlocks = std::min(static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE), MAX_GRID_BLOCKS);

    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partialSum, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    dotProductSharedMemKernel<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_partialSum, n);
    partialSumReductionKernel<<<1, BLOCK_SIZE>>>(d_partialSum, d_res, numBlocks);

    float h_res;
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partialSum);
    cudaFree(d_res);

    return h_res;
}

// Test: single element vectors
static bool testSingleElement() {
    std::vector<float> a = {3.0f};
    std::vector<float> b = {4.0f};

    float gpu_res = runDotProductSharedMem(a.data(), b.data(), 1);
    float cpu_res = cpuDotProduct(a.data(), b.data(), 1);

    return almostEqual(gpu_res, cpu_res, 1e-5f);
}

// Test: all zeros
static bool testAllZeros() {
    const size_t n = 1024;
    std::vector<float> a(n, 0.0f);
    std::vector<float> b(n, 0.0f);

    float gpu_res = runDotProductSharedMem(a.data(), b.data(), n);

    return almostEqual(gpu_res, 0.0f, 1e-10f);
}

// Test: known result - [1,1,1,...] dot [1,1,1,...] = n
static bool testOnesKnownResult() {
    const size_t n = 10000;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 1.0f);

    float gpu_res = runDotProductSharedMem(a.data(), b.data(), n);
    float expected = static_cast<float>(n);

    return almostEqual(gpu_res, expected, 1e-3f);
}

// Test: orthogonal vectors
static bool testOrthogonalVectors() {
    const size_t n = 1024;
    std::vector<float> a(n, 0.0f);
    std::vector<float> b(n, 0.0f);

    // a = [1, 0, 0, ...], b = [0, 1, 0, ...]
    a[0] = 1.0f;
    b[1] = 1.0f;

    float gpu_res = runDotProductSharedMem(a.data(), b.data(), n);

    return almostEqual(gpu_res, 0.0f, 1e-10f);
}

// Test: large n verification
static bool testLargeN() {
    const size_t n = 1024 * 1024; // 1M elements
    std::vector<float> a(n);
    std::vector<float> b(n);

    fillRandom(a.data(), n, -1.0f, 1.0f);
    fillRandom(b.data(), n, -1.0f, 1.0f);

    float gpu_res = runDotProductSharedMem(a.data(), b.data(), n);
    float cpu_res = cpuDotProduct(a.data(), b.data(), n);

    // Allow larger epsilon for large accumulations
    float epsilon = std::abs(cpu_res) * 1e-4f + 1e-3f;
    return almostEqual(gpu_res, cpu_res, epsilon);
}

// Test: negative values
static bool testNegativeValues() {
    const size_t n = 1024;
    std::vector<float> a(n, -1.0f);
    std::vector<float> b(n, 1.0f);

    float gpu_res = runDotProductSharedMem(a.data(), b.data(), n);
    float expected = -static_cast<float>(n);

    return almostEqual(gpu_res, expected, 1e-3f);
}

// Test: non-power-of-2 size
static bool testNonPowerOf2Size() {
    const size_t n = 1000; // Not a power of 2
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);

    float gpu_res = runDotProductSharedMem(a.data(), b.data(), n);
    float expected = static_cast<float>(n * 2);

    return almostEqual(gpu_res, expected, 1e-3f);
}

int main() {
    printDeviceInfo();

    TestHarness harness("Dot Product Edge Case Tests");

    harness.run("single element", testSingleElement);
    harness.run("all zeros", testAllZeros);
    harness.run("ones known result (n=10000)", testOnesKnownResult);
    harness.run("orthogonal vectors", testOrthogonalVectors);
    harness.run("large n (1M elements)", testLargeN);
    harness.run("negative values", testNegativeValues);
    harness.run("non-power-of-2 size", testNonPowerOf2Size);

    return harness.summarize();
}
