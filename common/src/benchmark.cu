#include "benchmark.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

double BenchmarkResult::getBandwidthGBs() const {
  if (avgMs <= 0) {
    return 0;
  }

  double totalBytes = static_cast<double>(n) * memAccessFactor * sizeof(float);
  double seconds = static_cast<double>(avgMs) / 1000.0;
  return (totalBytes / 1e9) / seconds;
}

void Benchmarker::run(const std::string &tag, size_t n,
                      std::function<void()> kernelFunc,
                      std::function<void()> verifyFunc, int memAccessFactor) {
  // 1. warmup
  for (int i = 0; i < warmup_; ++i) {
    kernelFunc();
  }
  cudaDeviceSynchronize();

  // 2. setup timing events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 3. measure
  cudaEventRecord(start, 0);
  for (int i = 0; i < trials_; ++i) {
    kernelFunc();
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float totalMs = 0;
  cudaEventElapsedTime(&totalMs, start, stop);

  // 4. save result
  results_.push_back({tag, n, totalMs / trials_, memAccessFactor});

  // verify
  if (verifyFunc) {
    verifyFunc();
  }

  // 5. cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

const std::vector<BenchmarkResult> &Benchmarker::getResults() const {
  return results_;
}

// export to csv file
void Benchmarker::exportToCsv(const std::string &filename) const {
  std::ofstream file(filename);
  file << "Algorithm,N,AvgTime_ms,Bandwidth_GBs\n";
  for (const auto &res : results_) {
    file << res.tag << "," << res.n << "," << std::fixed << std::setprecision(6)
         << res.avgMs << "," << std::setprecision(2) << res.getBandwidthGBs()
         << "\n";
  }
  std::cout << ">> Results exported to CSV: " << filename << std::endl;
}

// export to json string
std::string Benchmarker::exportToJson() const {
  std::stringstream ss;
  ss << "{\n  \"trials\": " << trials_ << ",\n  \"results\": [\n";
  for (size_t i = 0; i < results_.size(); ++i) {
    const auto &r = results_[i];
    ss << "    {\"tag\": \"" << r.tag << "\", \"n\": " << r.n
       << ", \"avgMs\": " << r.avgMs << ", \"gbps\": " << r.getBandwidthGBs()
       << "}";
    if (i < results_.size() - 1)
      ss << ",";
    ss << "\n";
  }
  ss << "  ]\n}";
  return ss.str();
}