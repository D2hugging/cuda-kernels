# Common utilities library

set(COMMON_SOURCES
    src/benchmark.cu
    src/cuda_utils.cu
    src/test_utils.cu
)

set(COMMON_HEADERS
    include/benchmark.h
    include/cuda_config.h
    include/cuda_utils.h
    include/device_utils.cuh
    include/test_harness.h
    include/test_utils.h
)

# Create a static library for common utilities
add_library(cuda_common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})

target_include_directories(cuda_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set CUDA properties
set_target_properties(cuda_common PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Apply standard CUDA compile options
target_cuda_compile_options(cuda_common)
