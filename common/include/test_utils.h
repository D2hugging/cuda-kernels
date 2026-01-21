#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <cstddef>
#include <string>

// comparison of two value
bool almostEqual(float a, float b, float epsilon = 1e-5f);
bool almostEqual(double a, double b, double epsilon = 1e-9);

// array comparison
bool arrayEqual(const float *a, const float *b, size_t n,
                float epsilon = 1e-5f);
bool arrayEqual(const double *a, const double *b, size_t n,
                double epsilon = 1e-9);

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