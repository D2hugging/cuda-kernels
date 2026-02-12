#ifndef BENCHMARKER_H
#define BENCHMARKER_H

#include "cuda_utils.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// Structured benchmark result with full statistics
struct BenchmarkResult {
  std::string tag;
  size_t n;
  size_t elementSize; // sizeof(T) for bandwidth calculation
  int memAccessFactor;
  size_t totalFlops = 0; // Total floating point operations (0 = bandwidth-bound)

  // Full statistics
  float minMs;
  float maxMs;
  float meanMs;
  float medianMs;
  float stddevMs;
  std::vector<float> trials; // Raw trial data (optional)

  // GPU bandwidth in GB/s using actual element size
  double getBandwidthGBs() const {
    if (meanMs <= 0) {
      return 0;
    }
    double totalBytes = static_cast<double>(n) * memAccessFactor * elementSize;
    double seconds = static_cast<double>(meanMs) / 1000.0;
    return (totalBytes / 1e9) / seconds;
  }

  // Bandwidth using min time (peak bandwidth)
  double getPeakBandwidthGBs() const {
    if (minMs <= 0) {
      return 0;
    }
    double totalBytes = static_cast<double>(n) * memAccessFactor * elementSize;
    double seconds = static_cast<double>(minMs) / 1000.0;
    return (totalBytes / 1e9) / seconds;
  }

  // GFLOPS for compute-bound kernels (mean time)
  double getGflops() const {
    if (meanMs <= 0 || totalFlops == 0) {
      return 0;
    }
    double seconds = static_cast<double>(meanMs) / 1000.0;
    return static_cast<double>(totalFlops) / seconds / 1e9;
  }

  // GFLOPS using min time (peak performance)
  double getPeakGflops() const {
    if (minMs <= 0 || totalFlops == 0) {
      return 0;
    }
    double seconds = static_cast<double>(minMs) / 1000.0;
    return static_cast<double>(totalFlops) / seconds / 1e9;
  }
};

// RAII wrapper for CUDA events (prevents leaks)
class CudaEventGuard {
public:
  cudaEvent_t event;

  CudaEventGuard() { CUDA_CHECK(cudaEventCreate(&event)); }
  ~CudaEventGuard() { cudaEventDestroy(event); }

  // Non-copyable
  CudaEventGuard(const CudaEventGuard &) = delete;
  CudaEventGuard &operator=(const CudaEventGuard &) = delete;
};

class Benchmarker {
public:
  struct Config {
    int warmup = 10;
    int trials = 100;
    bool storeRawTrials = false; // Save all trial times (memory cost)
    cudaStream_t stream = 0;     // Optional stream
  };

  // Default constructor with default config
  Benchmarker() : config_() {}

  // Constructor with custom config
  explicit Benchmarker(const Config &config) : config_(config) {}

  // Legacy constructor for backward compatibility
  Benchmarker(int warmup, int trials) : config_{warmup, trials, false, 0} {}

  // Template for element size (float, double, etc.)
  // verifyFunc returns bool: true = passed, false = failed
  // totalFlops: total floating point operations (0 = bandwidth-bound, use for GEMM: 2*M*N*K)
  template <typename T = float>
  void run(const std::string &tag, size_t n, std::function<void()> kernelFunc,
           std::function<bool()> verifyFunc = nullptr, int memAccessFactor = 2,
           size_t totalFlops = 0);

  const std::vector<BenchmarkResult> &getResults() const { return results_; }

  void exportToCsv(const std::string &filename) const;
  std::string exportToJson() const;
  void printSummary() const;

private:
  Config config_;
  std::vector<BenchmarkResult> results_;

  BenchmarkResult computeStats(const std::string &tag, size_t n,
                               size_t elemSize, int memFactor,
                               size_t totalFlops,
                               std::vector<float> &times) const;
};

// Template implementation must be in header
template <typename T>
void Benchmarker::run(const std::string &tag, size_t n,
                      std::function<void()> kernelFunc,
                      std::function<bool()> verifyFunc, int memAccessFactor,
                      size_t totalFlops) {
  // 1. Warmup with sync
  for (int i = 0; i < config_.warmup; ++i) {
    kernelFunc();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 2. Per-trial timing
  std::vector<float> trialTimes(config_.trials);

  for (int i = 0; i < config_.trials; ++i) {
    CudaEventGuard start, stop;

    CUDA_CHECK(cudaEventRecord(start.event, config_.stream));
    kernelFunc();
    CUDA_CHECK(cudaEventRecord(stop.event, config_.stream));
    CUDA_CHECK(cudaEventSynchronize(stop.event));
    CUDA_CHECK(cudaEventElapsedTime(&trialTimes[i], start.event, stop.event));
  }

  // 3. Verify BEFORE storing result
  if (verifyFunc && !verifyFunc()) {
    std::cerr << "VERIFICATION FAILED: " << tag << " - result not recorded\n";
    return;
  }

  // 4. Compute statistics and store
  BenchmarkResult result =
      computeStats(tag, n, sizeof(T), memAccessFactor, totalFlops, trialTimes);
  results_.push_back(std::move(result));
}

#endif
