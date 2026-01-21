#include "test_utils.h"
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <random>

bool almostEqual(float a, float b, float epsilon) {
  return std::fabs(a - b) <= epsilon * std::fmax(std::fabs(a), std::fabs(b));
}

bool almostEqual(double a, double b, double epsilon) {
  return std::fabs(a - b) <= epsilon * std::fmax(std::fabs(a), std::fabs(b));
}

bool arrayEqual(const float *a, const float *b, size_t n, float epsilon) {
  for (size_t i = 0; i < n; ++i) {
    if (!almostEqual(a[i], b[i], epsilon)) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i]
                << " (diff: " << std::fabs(a[i] - b[i]) << ")\n";
      return false;
    }
  }
  return true;
}

bool arrayEqual(const double *a, const double *b, size_t n, double epsilon) {
  for (size_t i = 0; i < n; ++i) {
    if (!almostEqual(a[i], b[i], epsilon)) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i]
                << " (diff: " << std::fabs(a[i] - b[i]) << ")\n";
      return false;
    }
  }
  return true;
}

void fillRandom(float *data, size_t n, float minVal, float maxVal) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(minVal, maxVal);
  for (size_t i = 0; i < n; ++i) {
    data[i] = dis(gen);
  }
}

void fillSequential(float *arr, size_t n, float start, float step) {
  for (size_t i = 0; i < n; ++i) {
    arr[i] = start + i * step;
  }
}

void fillConstant(float *arr, size_t n, float value) {
  for (size_t i = 0; i < n; ++i) {
    arr[i] = value;
  }
}

void printArray(const float *arr, size_t n, const std::string &label,
                size_t maxPrint) {
  std::cout << label << " [size=" << n << "]: ";
  maxPrint = std::min(n, maxPrint);
  for (size_t i = 0; i < maxPrint; ++i) {
    std::cout << std::fixed << std::setprecision(4) << arr[i] << ' ';
  }
  if (n > maxPrint) {
    std::cout << "...";
  }
  std::cout << '\n';
}

void printMatrix(const float *mat, size_t rows, size_t cols,
                 const std::string &label, size_t maxPrintRows,
                 size_t maxPrintCols) {
  maxPrintRows = std::min(rows, maxPrintRows);
  maxPrintCols = std::min(cols, maxPrintCols);
  std::cout << label << " [" << rows << 'x' << cols << "]:\n";
  for (size_t i = 0; i < maxPrintRows; ++i) {
    std::cout << ' ';
    for (size_t j = 0; j < maxPrintCols; ++j) {
      std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                << mat[i * cols + j] << ' ';
    }
    if (cols > maxPrintCols) {
      std::cout << "...";
    }
    std::cout << '\n';
  }
  if (rows > maxPrintRows) {
    std::cout << " ...\n";
  }
}
