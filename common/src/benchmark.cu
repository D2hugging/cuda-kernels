#include "benchmark.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

BenchmarkResult Benchmarker::computeStats(const std::string &tag, size_t n,
                                          size_t elemSize, int memFactor,
                                          std::vector<float> &times) const {
  // Sort for min/max/median
  std::sort(times.begin(), times.end());

  float sum = std::accumulate(times.begin(), times.end(), 0.0f);
  float mean = sum / static_cast<float>(times.size());

  // Standard deviation
  float sqSum = 0;
  for (float t : times) {
    sqSum += (t - mean) * (t - mean);
  }
  float stddev = std::sqrt(sqSum / static_cast<float>(times.size()));

  // Median
  size_t mid = times.size() / 2;
  float median = (times.size() % 2 == 0)
                     ? (times[mid - 1] + times[mid]) / 2.0f
                     : times[mid];

  BenchmarkResult result;
  result.tag = tag;
  result.n = n;
  result.elementSize = elemSize;
  result.memAccessFactor = memFactor;
  result.minMs = times.front();
  result.maxMs = times.back();
  result.meanMs = mean;
  result.medianMs = median;
  result.stddevMs = stddev;

  if (config_.storeRawTrials) {
    result.trials = std::move(times);
  }

  return result;
}

void Benchmarker::runLegacy(const std::string &tag, size_t n,
                            std::function<void()> kernelFunc,
                            std::function<void()> verifyFunc,
                            int memAccessFactor) {
  // Wrap void verifyFunc as bool-returning (always passes after running)
  std::function<bool()> boolVerify = nullptr;
  if (verifyFunc) {
    boolVerify = [verifyFunc]() {
      verifyFunc();
      return true;  // Legacy behavior: always record result
    };
  }
  run<float>(tag, n, kernelFunc, boolVerify, memAccessFactor);
}

void Benchmarker::exportToCsv(const std::string &filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "ERROR: Failed to open CSV file: " << filename << std::endl;
    return;
  }

  file << "Algorithm,N,MinTime_ms,MaxTime_ms,MeanTime_ms,MedianTime_ms,"
       << "StdDev_ms,Bandwidth_GBs,PeakBandwidth_GBs\n";

  for (const auto &res : results_) {
    file << res.tag << "," << res.n << "," << std::fixed << std::setprecision(6)
         << res.minMs << "," << res.maxMs << "," << res.meanMs << ","
         << res.medianMs << "," << res.stddevMs << "," << std::setprecision(2)
         << res.getBandwidthGBs() << "," << res.getPeakBandwidthGBs() << "\n";
  }

  std::cout << ">> Results exported to CSV: " << filename << std::endl;
}

std::string Benchmarker::exportToJson() const {
  std::stringstream ss;
  ss << "{\n";
  ss << "  \"config\": {\n";
  ss << "    \"warmup\": " << config_.warmup << ",\n";
  ss << "    \"trials\": " << config_.trials << "\n";
  ss << "  },\n";
  ss << "  \"results\": [\n";

  for (size_t i = 0; i < results_.size(); ++i) {
    const auto &r = results_[i];
    ss << "    {\n";
    ss << "      \"tag\": \"" << r.tag << "\",\n";
    ss << "      \"n\": " << r.n << ",\n";
    ss << "      \"elementSize\": " << r.elementSize << ",\n";
    ss << "      \"memAccessFactor\": " << r.memAccessFactor << ",\n";
    ss << "      \"minMs\": " << std::fixed << std::setprecision(6) << r.minMs << ",\n";
    ss << "      \"maxMs\": " << r.maxMs << ",\n";
    ss << "      \"meanMs\": " << r.meanMs << ",\n";
    ss << "      \"medianMs\": " << r.medianMs << ",\n";
    ss << "      \"stddevMs\": " << r.stddevMs << ",\n";
    ss << "      \"bandwidthGBs\": " << std::setprecision(2) << r.getBandwidthGBs() << ",\n";
    ss << "      \"peakBandwidthGBs\": " << r.getPeakBandwidthGBs() << "\n";
    ss << "    }";
    if (i < results_.size() - 1)
      ss << ",";
    ss << "\n";
  }

  ss << "  ]\n}";
  return ss.str();
}

void Benchmarker::printSummary() const {
  std::cout << "\n";
  std::cout << std::left << std::setw(30) << "Algorithm"
            << std::right << std::setw(12) << "N"
            << std::setw(12) << "Mean(ms)"
            << std::setw(12) << "Min(ms)"
            << std::setw(12) << "StdDev(ms)"
            << std::setw(12) << "BW(GB/s)"
            << "\n";
  std::cout << std::string(90, '-') << "\n";

  for (const auto &r : results_) {
    std::cout << std::left << std::setw(30) << r.tag
              << std::right << std::setw(12) << r.n
              << std::fixed << std::setprecision(4)
              << std::setw(12) << r.meanMs
              << std::setw(12) << r.minMs
              << std::setw(12) << r.stddevMs
              << std::setprecision(2)
              << std::setw(12) << r.getBandwidthGBs()
              << "\n";
  }
  std::cout << "\n";
}
