hpcc_populate_dep(googletest)

if(PPLCV_USE_OPENCL)
    add_executable(pplcv_unittest ${PPLCV_UNITTEST_SRC}
                   src/ppl/cv/ocl/utility/infrastructure.cpp)
else()
    add_executable(pplcv_unittest ${PPLCV_UNITTEST_SRC})
endif()

if(PPLCV_USE_X86)
    target_compile_definitions(pplcv_unittest PRIVATE PPLCV_UNITTEST_OPENCV)
endif()
if(PPLCV_USE_CUDA)
    if(NOT PPLCV_USE_X86)
        message(FATAL_ERROR "cuda unittests require x86 support. please enable compiling x86 libs.")
    endif()
    target_compile_definitions(pplcv_unittest PRIVATE PPLCV_UNITTEST_OPENCV_CUDA)
endif()
target_compile_features(pplcv_unittest PRIVATE cxx_std_11)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()
if(PPLCV_USE_AARCH64)
    target_compile_definitions(pplcv_unittest PRIVATE PPLCV_UNITTEST_OPENCV)
endif()
if(PPLCV_USE_RISCV)
    target_compile_definitions(pplcv_unittest PRIVATE PPLCV_UNITTEST_OPENCV)
endif()

target_include_directories(pplcv_unittest PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${opencv_INCLUDE_DIRECTORIES}
    ${googletest_SOURCE_DIR}/include)

target_link_libraries(pplcv_unittest PRIVATE
    pplcv_static gtest_main ${opencv_LIBRARIES})
