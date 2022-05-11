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

set(__HPCC_COMMIT__ 266579679761dd2c37440c700fb5602187056fce)

if(PPLCV_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPLCV_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
elseif(PPLCV_DEP_HPCC_GIT)
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPLCV_DEP_HPCC_GIT}
        GIT_TAG ${__HPCC_COMMIT__}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    FetchContent_Declare(hpcc
        URL https://github.com/openppl-public/hpcc/archive/${__HPCC_COMMIT__}.zip
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

set(__PPLCOMMON_COMMIT__ 868c14594f2d317923b6ab45f9182cb7be394afc)

if(PPLCV_DEP_PPLCOMMON_PKG)
    hpcc_declare_pkg_dep(pplcommon
        ${PPLCV_DEP_PPLCOMMON_PKG})
elseif(PPLCV_DEP_PPLCOMMON_GIT)
    hpcc_declare_git_dep(pplcommon
        ${PPLCV_DEP_PPLCOMMON_GIT}
        ${__PPLCOMMON_COMMIT__})
else()
    hpcc_declare_pkg_dep(pplcommon
        https://github.com/openppl-public/ppl.common/archive/${__PPLCOMMON_COMMIT__}.zip)
endif()

unset(__PPLCOMMON_COMMIT__)

# --------------------------------------------------------------------------- #

set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

set(__GOOGLETEST_TAG__ release-1.8.1)

if(PPLCV_DEP_GOOGLETEST_PKG)
    hpcc_declare_pkg_dep(googletest
        ${PPLCV_DEP_GOOGLETEST_PKG})
elseif(PPLCV_DEP_GOOGLETEST_GIT)
    hpcc_declare_git_dep(googletest
        ${PPLCV_DEP_GOOGLETEST_GIT}
        ${__GOOGLETEST_TAG__})
else()
    hpcc_declare_pkg_dep(googletest
        https://github.com/google/googletest/archive/refs/tags/${__GOOGLETEST_TAG__}.zip)
endif()

unset(__GOOGLETEST_TAG__)

# --------------------------------------------------------------------------- #

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "disable benchmark tests")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")

set(__BENCHMARK__TAG__ v1.5.6)

if(PPLCV_DEP_BENCHMARK_PKG)
    hpcc_declare_pkg_dep(benchmark
        ${PPLCV_DEP_BENCHMARK_PKG})
elseif(PPLCV_DEP_BENCHMARK_GIT)
    hpcc_declare_git_dep(benchmark
        ${PPLCV_DEP_BENCHMARK_GIT}
        ${__BENCHMARK__TAG__})
else()
    hpcc_declare_pkg_dep(benchmark
        https://github.com/google/benchmark/archive/refs/tags/${__BENCHMARK__TAG__}.zip)
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
elseif(PPLCV_DEP_OPENCV_GIT)
    hpcc_declare_git_dep(opencv
        ${PPLCV_DEP_OPENCV_GIT}
        ${__OPENCV_TAG__})
else()
    hpcc_declare_pkg_dep(opencv
        https://github.com/opencv/opencv/archive/refs/tags/${__OPENCV_TAG__}.zip)
endif()

if(PPLCV_DEP_OPENCV_CONTRIB_PKG)
    hpcc_declare_pkg_dep(opencv_contrib
        ${PPLCV_DEP_OPENCV_CONTRIB_PKG})
elseif(PPLCV_DEP_OPENCV_CONTRIB_GIT)
    hpcc_declare_git_dep(opencv_contrib
        ${PPLCV_DEP_OPENCV_CONTRIB_GIT}
        ${__OPENCV_TAG__})
else()
    hpcc_declare_pkg_dep(opencv_contrib
        https://github.com/opencv/opencv_contrib/archive/refs/tags/${__OPENCV_TAG__}.zip)
endif()

unset(__OPENCV_TAG__)
