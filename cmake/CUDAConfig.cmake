# Helper functions for CUDA configuration

# Function to apply standard CUDA compile options to a target
# This consolidates compile flags in one place to avoid duplication
function(target_cuda_compile_options target_name)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
            --use_fast_math
            -Xcompiler=-Wall
        >
    )
endfunction()

# Function to add a CUDA executable with common settings
function(add_cuda_executable target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;HEADERS;LINK_LIBRARIES" ${ARGN})
    
    add_executable(${target_name} ${ARG_SOURCES} ${ARG_HEADERS})
    
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
            --use_fast_math
            -Xcompiler=-Wall
            -Xcompiler=-Wextra
        >
    )
    
    if(ARG_LINK_LIBRARIES)
        target_link_libraries(${target_name} PRIVATE ${ARG_LINK_LIBRARIES})
    endif()
    
    # Set CUDA separable compilation if needed
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# Function to check CUDA compute capability
function(check_cuda_capability min_capability)
    # Add capability checking logic here
endfunction()