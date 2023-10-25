# PPL CV AARCH64 source cmake script

set(PPLCV_USE_ARM ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

file(GLOB PPLCV_AARCH64_PUBLIC_HEADERS src/ppl/cv/arm/*.h)
install(FILES ${PPLCV_AARCH64_PUBLIC_HEADERS}
        DESTINATION include/ppl/cv/arm)

list(APPEND PPLCV_COMPILE_DEFINITIONS PPLCV_USE_AARCH64)

file(GLOB PPLCV_AARCH64_SRC
     src/ppl/cv/arm/*.cpp)
list(APPEND PPLCV_SRC ${PPLCV_AARCH64_SRC})

# glob benchmark and unittest sources
file(GLOB PPLCV_AARCH64_BENCHMARK_SRC "src/ppl/cv/arm/*_benchmark.cpp")
file(GLOB PPLCV_AARCH64_UNITTEST_SRC "src/ppl/cv/arm/*_unittest.cpp")
list(APPEND PPLCV_BENCHMARK_SRC ${PPLCV_AARCH64_BENCHMARK_SRC})
list(APPEND PPLCV_UNITTEST_SRC ${PPLCV_AARCH64_UNITTEST_SRC})
