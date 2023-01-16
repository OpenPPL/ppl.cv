set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
hpcc_populate_dep(benchmark)

if(PPLCV_USE_OPENCL)
    add_executable(pplcv_benchmark src/ppl/cv/ocl/utility/infrastructure.cpp
                   ${PPLCV_BENCHMARK_SRC}
                   ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/cv/benchmark_main.cc)
else()
    add_executable(pplcv_benchmark ${PPLCV_BENCHMARK_SRC}
                   ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/cv/benchmark_main.cc)
endif()

if(PPLCV_USE_X86)
    target_compile_definitions(pplcv_benchmark PRIVATE PPLCV_BENCHMARK_OPENCV)
endif()
if(PPLCV_USE_CUDA)
    target_compile_definitions(pplcv_benchmark PRIVATE PPLCV_BENCHMARK_OPENCV_CUDA)
endif()
if(PPLCV_USE_AARCH64)
    target_compile_definitions(pplcv_benchmark PRIVATE PPLCV_BENCHMARK_OPENCV)
endif()
if(PPLCV_USE_RISCV)
    target_compile_definitions(pplcv_benchmark PRIVATE PPLCV_BENCHMARK_OPENCV)
endif()

target_include_directories(pplcv_benchmark PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${opencv_INCLUDE_DIRECTORIES}
    ${benchmark_SOURCE_DIR}/include)

target_link_libraries(pplcv_benchmark PRIVATE pplcv_static benchmark
                      ${opencv_LIBRARIES})
