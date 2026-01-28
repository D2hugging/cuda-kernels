#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

// Centralized CUDA configuration constants

// Default block size for kernel launches
// 256 threads is a good default for most compute capabilities
constexpr int DEFAULT_BLOCK_SIZE = 256;

// Maximum number of blocks to use for grid-stride loops
constexpr int MAX_GRID_BLOCKS = 1024;

#endif // CUDA_CONFIG_H
