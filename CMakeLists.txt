cmake_minimum_required(VERSION 3.11)
project(ppl.cv)

include(cmake/deps.cmake)

# --------------------------------------------------------------------------- #

option(PPLCV_BUILD_TESTS "build tests" OFF)
option(PPLCV_BUILD_BENCHMARK "build benchmark" OFF)

# --------------------------------------------------------------------------- #

set(PPLCV_SRC )
set(PPLCV_BENCHMARK_SRC )
set(PPLCV_UNITTEST_SRC )
set(PPLCV_INCLUDE_DIRECTORIES )
set(PPLCV_LINK_LIBRARIES )
set(PPLCV_COMPILE_DEFINITIONS )

# --------------------------------------------------------------------------- #

include(cmake/x86.cmake)
include(cmake/cuda.cmake)

hpcc_populate_dep(ppl.common)

list(FILTER PPLCV_SRC EXCLUDE REGEX "(.*)_unittest\\.cpp$")
list(FILTER PPLCV_SRC EXCLUDE REGEX "(.*)_benchmark\\.cpp$")

add_library(pplcv_static STATIC ${PPLCV_SRC})
target_include_directories(pplcv_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
    ${PPLCV_INCLUDE_DIRECTORIES})
target_link_libraries(pplcv_static PUBLIC pplcommon_static ${PPLCV_LINK_LIBRARIES})
target_compile_definitions(pplcv_static PUBLIC ${PPLCV_COMPILE_DEFINITIONS})
target_compile_options(pplcv_static PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-fno-exceptions -Wno-strict-aliasing>)

install(FILES src/ppl/cv/types.h
        DESTINATION include/ppl/cv)
install(TARGETS pplcv_static ARCHIVE DESTINATION lib)

if(PPLCV_BUILD_TESTS OR PPLCV_BUILD_BENCHMARK)
    include(cmake/opencv.cmake)
    if (PPLCV_BUILD_TESTS)
        include(cmake/unittest.cmake)
    endif()
    if(PPLCV_BUILD_BENCHMARK)
        include(cmake/benchmark.cmake)
    endif()
endif()
