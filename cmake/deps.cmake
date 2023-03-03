if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${HPCC_DEPS_DIR})
set(FETCHCONTENT_QUIET OFF)

if(PPLCV_HOLD_DEPS)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

# --------------------------------------------------------------------------- #

set(__HPCC_COMMIT__ af7dcc6c1b1eaf622b3d01472b89ce62d881f66c)

if(PPLCV_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPLCV_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    if(NOT PPLCV_DEP_HPCC_GIT)
        set(PPLCV_DEP_HPCC_GIT "https://github.com/openppl-public/hpcc.git")
    endif()
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPLCV_DEP_HPCC_GIT}
        GIT_TAG ${__HPCC_COMMIT__}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
endif()

unset(__HPCC_COMMIT__)

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

set(PPLCOMMON_BUILD_TESTS OFF CACHE BOOL "disable pplcommon tests")
set(PPLCOMMON_BUILD_BENCHMARK OFF CACHE BOOL "disable pplcommon benchmark")
set(PPLCOMMON_HOLD_DEPS ${PPLCV_HOLD_DEPS})
set(PPLCOMMON_USE_X86_64 ${PPLCV_USE_X86_64})
set(PPLCOMMON_USE_AARCH64 ${PPLCV_USE_AARCH64})
set(PPLCOMMON_USE_OPENCL ${PPLCV_USE_OPENCL})

if(PPLCV_USE_OPENCL)
    set(PPLCOMMON_OPENCL_INCLUDE_DIRS ${PPLCV_OPENCL_INCLUDE_DIRS})
    set(PPLCOMMON_OPENCL_LIBRARIES ${PPLCV_OPENCL_LIBRARIES})
endif()

set(__PPLCOMMON_COMMIT__ aaed621dd09b023602691071050658d0e4f673e2)

if(PPLCV_DEP_PPLCOMMON_PKG)
    hpcc_declare_pkg_dep(pplcommon
        ${PPLCV_DEP_PPLCOMMON_PKG})
else()
    if(NOT PPLCV_DEP_PPLCOMMON_GIT)
        set(PPLCV_DEP_PPLCOMMON_GIT "https://github.com/openppl-public/ppl.common.git")
    endif()
    hpcc_declare_git_dep(pplcommon
        ${PPLCV_DEP_PPLCOMMON_GIT}
        ${__PPLCOMMON_COMMIT__})
endif()

unset(__PPLCOMMON_COMMIT__)

# --------------------------------------------------------------------------- #

set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(__GOOGLETEST_TAG__ release-1.10.0)

if(PPLCV_DEP_GOOGLETEST_PKG)
    hpcc_declare_pkg_dep(googletest
        ${PPLCV_DEP_GOOGLETEST_PKG})
else()
    if(NOT PPLCV_DEP_GOOGLETEST_GIT)
        set(PPLCV_DEP_GOOGLETEST_GIT "https://github.com/google/googletest.git")
    endif()
    hpcc_declare_git_dep(googletest
        ${PPLCV_DEP_GOOGLETEST_GIT}
        ${__GOOGLETEST_TAG__})
endif()

unset(__GOOGLETEST_TAG__)

# --------------------------------------------------------------------------- #

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "disable benchmark tests")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")

set(__BENCHMARK__TAG__ v1.5.6)

if(PPLCV_DEP_BENCHMARK_PKG)
    hpcc_declare_pkg_dep(benchmark
        ${PPLCV_DEP_BENCHMARK_PKG})
else()
    if(NOT PPLCV_DEP_BENCHMARK_GIT)
        set(PPLCV_DEP_BENCHMARK_GIT "https://github.com/google/benchmark.git")
    endif()
    hpcc_declare_git_dep(benchmark
        ${PPLCV_DEP_BENCHMARK_GIT}
        ${__BENCHMARK__TAG__})
endif()

unset(__BENCHMARK__TAG__)

# --------------------------------------------------------------------------- #

set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_PERF_TESTS OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_opencv_apps OFF CACHE BOOL "")
set(OPENCV_EXTRA_MODULES_PATH "${HPCC_DEPS_DIR}/opencv_contrib/modules" CACHE INTERNAL "")

set(__OPENCV_TAG__ 4.4.0)

if(PPLCV_DEP_OPENCV_PKG)
    hpcc_declare_pkg_dep(opencv
        ${PPLCV_DEP_OPENCV_PKG})
else()
    if(NOT PPLCV_DEP_OPENCV_GIT)
        set(PPLCV_DEP_OPENCV_GIT "https://github.com/opencv/opencv.git")
    endif()
    hpcc_declare_git_dep(opencv
        ${PPLCV_DEP_OPENCV_GIT}
        ${__OPENCV_TAG__})
endif()

if(PPLCV_DEP_OPENCV_CONTRIB_PKG)
    hpcc_declare_pkg_dep(opencv_contrib
        ${PPLCV_DEP_OPENCV_CONTRIB_PKG})
else()
    if(NOT PPLCV_DEP_OPENCV_CONTRIB_GIT)
        set(PPLCV_DEP_OPENCV_CONTRIB_GIT "https://github.com/opencv/opencv_contrib.git")
    endif()
    hpcc_declare_git_dep(opencv_contrib
        ${PPLCV_DEP_OPENCV_CONTRIB_GIT}
        ${__OPENCV_TAG__})
endif()

unset(__OPENCV_TAG__)
