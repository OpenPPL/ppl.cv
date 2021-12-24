# PPL CV AARCH64 source cmake script
file(GLOB PPLCV_AARCH64_PUBLIC_HEADERS src/ppl/cv/aarch64/*.h)
install(FILES ${PPLCV_AARCH64_PUBLIC_HEADERS}
        DESTINATION include/ppl/cv/aarch64)

option(WITH_AARCH64 "Build pplcv with aarch64 support" ON)
option(PPLCV_USE_AARCH64 "Build unittest & benchmark with aarch64 support" ON)

list(APPEND PPLCV_COMPILE_DEFINITIONS PPLCV_USE_AARCH64)

file(GLOB PPLCV_AARCH64_SRC
     src/ppl/cv/aarch64/*.cpp)
list(APPEND PPLCV_SRC ${PPLCV_AARCH64_SRC})

# glob benchmark and unittest sources
file(GLOB PPLCV_AARCH64_BENCHMARK_SRC "src/ppl/cv/aarch64/*_benchmark.cpp")
file(GLOB PPLCV_AARCH64_UNITTEST_SRC "src/ppl/cv/aarch64/*_unittest.cpp")
list(APPEND PPLCV_BENCHMARK_SRC ${PPLCV_AARCH64_BENCHMARK_SRC})
list(APPEND PPLCV_UNITTEST_SRC ${PPLCV_AARCH64_UNITTEST_SRC})
