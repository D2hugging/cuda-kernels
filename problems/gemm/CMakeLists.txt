set(TARGET_NAME gemm)

set(SOURCES
    gemm.cu
    main.cu
)

set(HEADERS
    gemm.cuh
)

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${TARGET_NAME} PRIVATE cuda_common)

# Apply standard CUDA compile options
target_cuda_compile_options(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add main benchmark as test
add_test(NAME ${TARGET_NAME}_test COMMAND ${TARGET_NAME})

# Edge case unit tests
set(TEST_TARGET_NAME ${TARGET_NAME}-tests)
add_executable(${TEST_TARGET_NAME}
    tests.cu
    gemm.cu
)

target_link_libraries(${TEST_TARGET_NAME} PRIVATE cuda_common)
target_cuda_compile_options(${TEST_TARGET_NAME})
set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})