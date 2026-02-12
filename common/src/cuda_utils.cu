#include "cuda_utils.h"
#include <iostream>

void printDeviceInfo() {
  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    std::cerr << "No CUDA devices found\n";
    return;
  }
  std::cout << "Found " << deviceCount << " CUDA device(s):\n";
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

    std::cout << "\nDevice " << i << ": " << prop.name << '\n';
    std::cout << " Compute capability: " << prop.major << "." << prop.minor
              << '\n';
    std::cout << " Total global memory: " << prop.totalGlobalMem / (1024 * 1024)
              << " MB\n";
    std::cout << " Shared memory per block: " << prop.sharedMemPerBlock / 1024
              << " KB\n";
    std::cout << " Warp size: " << prop.warpSize << '\n';
    std::cout << " Max threads per block: " << prop.maxThreadsPerBlock << '\n';

    std::cout << " Max threads per SM: " << prop.maxThreadsPerMultiProcessor
              << '\n';
    std::cout << " Number of SMs: " << prop.multiProcessorCount << '\n';
    std::cout << " Max grid size: (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
    std::cout << " Clock rate: " << prop.clockRate << " MHz\n";
  }
  std::cout << '\n';
}

int getOptimalBlockSize(const void *kernel) {
  int minGridSize = 0;
  int blockSize = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                                kernel, 0, 0));
  return blockSize;
}