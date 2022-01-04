if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${HPCC_DEPS_DIR})
set(FETCHCONTENT_QUIET OFF)

if(PPLCV_HOLD_DEPS)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

FetchContent_Declare(hpcc
    GIT_REPOSITORY https://github.com/openppl-public/hpcc.git
    GIT_TAG 3d5a94e4f8197d009ddbb0be8451ff8e6bbf8445
    SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
    SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

hpcc_declare_pkg_dep(googletest
    https://github.com/google/googletest/archive/refs/tags/release-1.8.1.zip
    ad6868782b5952b7476a7c1c72d5a714)

# --------------------------------------------------------------------------- #

set(PPLCOMMON_BUILD_TESTS OFF CACHE BOOL "disable pplcommon tests")
set(PPLCOMMON_BUILD_BENCHMARK OFF CACHE BOOL "disable pplcommon benchmark")
set(PPLCOMMON_HOLD_DEPS ${PPLCV_HOLD_DEPS})

hpcc_declare_git_dep(pplcommon
    https://github.com/openppl-public/ppl.common.git
    b4170a465164977f19cd53bae8f00d9a6edbd804)

# --------------------------------------------------------------------------- #

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "disable benchmark tests")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")

hpcc_declare_pkg_dep(benchmark
    https://github.com/google/benchmark/archive/refs/tags/v1.5.6.zip
    2abe04dc31fc7f18b5f0647775b16249)

# --------------------------------------------------------------------------- #

set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_PERF_TESTS OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_opencv_apps OFF CACHE BOOL "")

hpcc_declare_pkg_dep(opencv
    https://github.com/opencv/opencv/archive/refs/tags/4.2.0.zip
    b02b54115f1f99cb9e885d1e5988ff70)

hpcc_declare_pkg_dep(opencv_contrib
    https://github.com/opencv/opencv_contrib/archive/refs/tags/4.2.0.zip
    4776354662667c85a91bcd19f6a13da7)
