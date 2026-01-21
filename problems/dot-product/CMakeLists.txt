set(TARGET_NAME dot-product)

set(SOURCES
    dot_product.cu
    main.cu
)

set(HEADERS
    dot_product.cuh
)

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${TARGET_NAME} PRIVATE cuda_common)

target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
        --use_fast_math;
        -Xcompiler=-Wall
    >
)

set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add test
add_test(NAME ${TARGET_NAME}_test COMMAND ${TARGET_NAME})