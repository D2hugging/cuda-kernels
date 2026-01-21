cmake_minimum_required(VERSION 3.18)
project(cuda-kernels LANGUAGES CXX CUDA)

# set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA architecture (my card is RTX3090)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 75 (Turing), 80 (Ampere A100), 86 (Ampere RTX 30xx), 89 (Ada RTX 40xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86)
endif()

# compile options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# include custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# enable testing
enable_testing()

# add subdirectories
add_subdirectory(common)
add_subdirectory(problems)

# optional subdirectories
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# print configuration summary
message(STATUS "")
message(STATUS "CUDA C++ Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "")