cmake_minimum_required(VERSION 3.14)
project(ppl.cv)

include(cmake/deps.cmake)

# --------------------------------------------------------------------------- #

option(PPLCV_INSTALL "install ppl.cv libs and headers" ON)
option(PPLCV_BUILD_TESTS "build tests" OFF)
option(PPLCV_BUILD_BENCHMARK "build benchmark" OFF)
option(PPLCV_HOLD_DEPS "don't update dependencies" OFF)

# --------------------------------------------------------------------------- #

hpcc_get_git_tag2version(
    VER_MAJOR PPLCV_VERSION_MAJOR
    VER_MINOR PPLCV_VERSION_MINOR
    VER_PATCH PPLCV_VERSION_PATCH)

# --------------------------------------------------------------------------- #

set(PPLCV_SRC )
set(PPLCV_BENCHMARK_SRC )
set(PPLCV_UNITTEST_SRC )
set(PPLCV_INCLUDE_DIRECTORIES )
set(PPLCV_LINK_LIBRARIES )
set(PPLCV_COMPILE_DEFINITIONS )

# --------------------------------------------------------------------------- #

if(HPCC_USE_X86_64)
    include(cmake/x86.cmake)
endif()

if(HPCC_USE_CUDA)
    include(cmake/cuda.cmake)
endif()

if(HPCC_USE_AARCH64)
    include(cmake/aarch64.cmake)
endif()

hpcc_populate_dep(pplcommon)

list(FILTER PPLCV_SRC EXCLUDE REGEX "(.*)_unittest\\.cpp$")
list(FILTER PPLCV_SRC EXCLUDE REGEX "(.*)_benchmark\\.cpp$")

add_library(pplcv_static STATIC ${PPLCV_SRC})
target_include_directories(pplcv_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
    ${PPLCV_INCLUDE_DIRECTORIES})
target_link_libraries(pplcv_static PUBLIC pplcommon_static ${PPLCV_LINK_LIBRARIES})
target_compile_definitions(pplcv_static PUBLIC
    ${PPLCV_COMPILE_DEFINITIONS}
    PPLCV_VERSION_MAJOR=${PPLCV_VERSION_MAJOR}
    PPLCV_VERSION_MINOR=${PPLCV_VERSION_MINOR}
    PPLCV_VERSION_PATCH=${PPLCV_VERSION_PATCH})

target_compile_options(pplcv_static PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-fno-exceptions -Wno-strict-aliasing>)

if(PPLCV_INSTALL)
    install(DIRECTORY include DESTINATION .)
    install(TARGETS pplcv_static DESTINATION lib)

    set(PPLCV_CMAKE_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/pplcv-config.cmake)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/pplcv-config.cmake.in
        ${PPLCV_CMAKE_CONFIG_FILE}
        @ONLY)
    install(FILES ${PPLCV_CMAKE_CONFIG_FILE} DESTINATION lib/cmake/ppl)
    unset(PPLCV_CMAKE_CONFIG_FILE)
endif()

if(PPLCV_BUILD_TESTS OR PPLCV_BUILD_BENCHMARK)
    include(cmake/opencv.cmake)
    if (PPLCV_BUILD_TESTS)
        include(cmake/unittest.cmake)
    endif()
    if(PPLCV_BUILD_BENCHMARK)
        include(cmake/benchmark.cmake)
    endif()
endif()
