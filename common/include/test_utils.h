#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

template <typename T>
bool almostEqual(T a, T b, T epsilon) {
  return std::fabs(a - b) <= epsilon * std::fmax(std::fabs(a), std::fabs(b));
}

template <typename T>
bool almostEqualTol(T a, T b, T rtol, T atol) {
  // NumPy-style: |a - b| <= atol + rtol * max(|a|, |b|)
  T diff = std::fabs(a - b);
  T threshold = atol + rtol * std::fmax(std::fabs(a), std::fabs(b));
  return diff <= threshold;
}

template <typename T>
bool arrayEqual(const T *a, const T *b, size_t n, T epsilon) {
  for (size_t i = 0; i < n; ++i) {
    if (!almostEqual(a[i], b[i], epsilon)) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i]
                << " (diff: " << std::fabs(a[i] - b[i]) << ")\n";
      return false;
    }
  }
  return true;
}

template <typename T>
bool arrayEqualTol(const T *a, const T *b, size_t n, T rtol, T atol) {
  for (size_t i = 0; i < n; ++i) {
    if (!almostEqualTol(a[i], b[i], rtol, atol)) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i]
                << " (diff: " << std::fabs(a[i] - b[i]) << ")\n";
      return false;
    }
  }
  return true;
}

// data generation
void fillRandom(float *arr, size_t n, float minVal = 0.0f, float maxVal = 1.0f);
void fillSequential(float *arr, size_t n, float startVal = 0.0f,
                    float step = 1.0f);
void fillConstant(float *arr, size_t n, float value);

// print data
void printArray(const float *arr, size_t n, const std::string &label = "Array",
                size_t maxPrint = 10);
void printMatrix(const float *mat, size_t rows, size_t cols,
                 const std::string &label = "Matrix", size_t maxPrintRows = 10,
                 size_t maxPrintCols = 10);

#endif // TEST_UTILS_H