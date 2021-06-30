if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${HPCC_DEPS_DIR})
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(hpcc
    GIT_REPOSITORY https://github.com/openppl-public/hpcc.git
    GIT_TAG v0.1.0
    SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
    BINARY_DIR ${HPCC_DEPS_DIR}/hpcc-build
    SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild
    UPDATE_COMMAND "")

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

hpcc_declare_git_dep(googletest
    https://github.com/google/googletest.git
    release-1.8.1)

set(PPLCOMMON_BUILD_TESTS OFF CACHE BOOL "disable ppl.common tests")
set(PPLCOMMON_BUILD_BENCHMARK OFF CACHE BOOL "disable ppl.common benchmark")
hpcc_declare_git_dep(ppl.common
    https://github.com/openppl-public/ppl.common.git
    v0.1.0)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "disable benchmark tests")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")
hpcc_declare_git_dep(benchmark
    https://github.com/google/benchmark.git
    v1.5.5)

set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_PERF_TESTS OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_opencv_apps OFF CACHE BOOL "")
hpcc_declare_git_dep(opencv
    https://github.com/opencv/opencv.git
    4.2.0)

hpcc_declare_git_dep(opencv_contrib
    https://github.com/opencv/opencv_contrib.git
    4.2.0)
