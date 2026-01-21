#ifndef BENCHMARKER_H
#define BENCHMARKER_H

#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <vector>

// 结构化存储性能结果
struct BenchmarkResult {
  std::string tag;     // tag for the kernel
  size_t n;            // data size
  float avgMs;         // time of kernel running (ms)
  int memAccessFactor; // the memory access factor for bandwidth calculation

  BenchmarkResult(const std::string &t, size_t size, float ms, int memFactor)
      : tag(t), n(size), avgMs(ms), memAccessFactor(memFactor) {}

  // GPU bandwidth in GB/s
  // (data size * memAccessFactor) / (avgMs / 1000.0) / (1e9)
  double getBandwidthGBs() const;
};

class Benchmarker {
public:
  Benchmarker(int warmup = 10, int trials = 100)
      : warmup_(warmup), trials_(trials) {}

  // run the kernel function with given tag and data size
  // (memAccessFactor for bandwidth calculation)
  void run(const std::string &tag, size_t n, std::function<void()> kernelFunc,
           std::function<void()> verifyFunc = nullptr, int memAccessFactor = 2);

  // get the results
  const std::vector<BenchmarkResult> &getResults() const;

  // export to CSV file
  void exportToCsv(const std::string &filename) const;

  // export to json string
  std::string exportToJson() const;

private:
  int warmup_;
  int trials_;
  std::vector<BenchmarkResult> results_;
};

#endif
