#include "cuda_config.h"
#include "cuda_utils.h"
#include "test_harness.h"
#include "test_utils.h"
#include "transpose.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

// CPU reference implementation
static void cpuTranspose(const float *h_A, float *h_B, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      h_B[j * rows + i] = h_A[i * cols + j];
    }
  }
}

// Helper to run a transpose kernel and get the result
static void runTransposeSharedMemPadded(const float *h_a, float *h_b,
                                        size_t rows, size_t cols) {

  // gpu
  float *d_a, *d_b;
  CUDA_CHECK(cudaMalloc(&d_a, rows * cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, cols * rows * sizeof(float)));

  // copy data to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, rows * cols * sizeof(float),
                        cudaMemcpyHostToDevice));

  // launch kernel
  dim3 block(TILE_SIZE, TILE_SIZE);
  int yDim = (rows + TILE_SIZE - 1) / TILE_SIZE;
  int xDim = (cols + TILE_SIZE - 1) / TILE_SIZE;
  dim3 grid(xDim, yDim); // (x,y,z) x:cols, y:rows

  transposeSharedMemPaddedKernel<<<grid, block>>>(d_a, d_b, rows, cols);

  // copy result back to host
  CUDA_CHECK(cudaMemcpy(h_b, d_b, cols * rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // free gpu memory
  cudaFree(d_a);
  cudaFree(d_b);
}

// TEST: single element
static bool testSingleElement() {
  std::vector<float> a = {3.0f};
  std::vector<float> b(1);

  runTransposeSharedMemPadded(a.data(), b.data(), 1, 1);

  std::vector<float> c(1);
  cpuTranspose(a.data(), c.data(), 1, 1);

  return arrayEqual(b.data(), c.data(), 1, 1e-5f);
}

// Test: single row (1xN)
static bool testSingleRow() {
  const size_t n = 1024;
  std::vector<float> a(1 * n, 0.0f);
  std::vector<float> b(1 * n);

  runTransposeSharedMemPadded(a.data(), b.data(), 1, n);

  std::vector<float> c(1 * n);
  cpuTranspose(a.data(), c.data(), 1, n);
  return arrayEqual(b.data(), c.data(), 1 * n, 1e-10f);
}

// Test: single column (Nx1)
static bool testSingleColumn() {
  const size_t n = 1024;
  std::vector<float> a(n * 1, 1.0f);
  std::vector<float> b(n * 1);

  runTransposeSharedMemPadded(a.data(), b.data(), n, 1);

  std::vector<float> c(n * 1);
  cpuTranspose(a.data(), c.data(), n, 1);
  return arrayEqual(b.data(), c.data(), n * 1, 1e-10f);
}

// Test: Square power of 2 (1024x1024)
static bool testPowerOf2() {
  const size_t n = 1024;
  std::vector<float> a(n * n);
  std::vector<float> b(n * n);

  fillRandom(a.data(), n * n, -1.0f, 1.0f);

  runTransposeSharedMemPadded(a.data(), b.data(), n, n);

  std::vector<float> c(n * n);
  cpuTranspose(a.data(), c.data(), n, n);

  return arrayEqual(b.data(), c.data(), n * n, 1e-5f);
}

// Test: Non-square (512x1024)
static bool testNonSquare() {
  const size_t rows = 512;
  const size_t cols = 1024;
  std::vector<float> a(rows * cols);
  std::vector<float> b(cols * rows);

  fillRandom(a.data(), rows * cols, -1.0f, 1.0f);

  runTransposeSharedMemPadded(a.data(), b.data(), rows, cols);

  std::vector<float> c(cols * rows);
  cpuTranspose(a.data(), c.data(), rows, cols);

  return arrayEqual(b.data(), c.data(), cols * rows, 1e-5f);
}

// Test: non-power-of-2 size
static bool testNonPowerOf2() {
  const size_t n = 1000; // Not a power of 2
  std::vector<float> a(n * n);
  std::vector<float> b(n * n);

  fillRandom(a.data(), n * n, -1.0f, 1.0f);

  runTransposeSharedMemPadded(a.data(), b.data(), n, n);

  std::vector<float> c(n * n);
  cpuTranspose(a.data(), c.data(), n, n);

  return arrayEqual(b.data(), c.data(), n * n, 1e-5f);
}

// Test: transpose(transpose(A)) == A
static bool testDoubleTranspose() {
  const size_t rows = 512;
  const size_t cols = 256;
  std::vector<float> a(rows * cols);
  std::vector<float> b(cols * rows);
  std::vector<float> c(rows * cols);

  fillRandom(a.data(), rows * cols, -1.0f, 1.0f);

  // First transpose
  runTransposeSharedMemPadded(a.data(), b.data(), rows, cols);
  // Second transpose
  runTransposeSharedMemPadded(b.data(), c.data(), cols, rows);

  return arrayEqual(a.data(), c.data(), rows * cols, 1e-5f);
}

int main() {
  printDeviceInfo();

  TestHarness harness("Transpose Edge Case Tests");

  harness.run("single element", testSingleElement);
  harness.run("single row", testSingleRow);
  harness.run("single column", testSingleColumn);
  harness.run("power of 2 size", testPowerOf2);
  harness.run("non-square shape", testNonSquare);
  harness.run("non-power-of-2 size", testNonPowerOf2);
  harness.run("double transpose", testDoubleTranspose);

  return harness.summarize();
}
