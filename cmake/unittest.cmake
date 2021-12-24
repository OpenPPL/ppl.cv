hpcc_populate_dep(googletest)

add_executable(pplcv_unittest ${PPLCV_UNITTEST_SRC})

if(PPLCV_USE_X86)
    target_compile_definitions(pplcv_unittest PRIVATE PPLCV_UNITTEST_OPENCV)
endif()
if(PPLCV_USE_CUDA)
    target_compile_definitions(pplcv_unittest PRIVATE PPLCV_UNITTEST_OPENCV_CUDA)
endif()
if(PPLCV_USE_AARCH64)
    target_compile_definitions(pplcv_unittest PRIVATE PPLCV_UNITTEST_OPENCV)
endif()

target_include_directories(pplcv_unittest PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${opencv_INCLUDE_DIRECTORIES}
    ${googletest_SOURCE_DIR}/include)

target_link_libraries(pplcv_unittest PRIVATE
    pplcv_static gtest_main ${opencv_LIBRARIES})
