# PPL CV RISCV source cmake script
file(GLOB PPLCV_RISCV_PUBLIC_HEADERS src/ppl/cv/riscv/*.h)
install(FILES ${PPLCV_RISCV_PUBLIC_HEADERS}
        DESTINATION include/ppl/cv/riscv)

set(PPLCV_USE_RISCV ON)
list(APPEND PPLCV_COMPILE_DEFINITIONS PPLCV_USE_RISCV)

file(GLOB PPLCV_RISCV_SRC
     src/ppl/cv/riscv/*.cpp)
list(APPEND PPLCV_SRC ${PPLCV_RISCV_SRC})

# glob benchmark and unittest sources
file(GLOB PPLCV_RISCV_BENCHMARK_SRC "src/ppl/cv/riscv/*_benchmark.cpp")
file(GLOB PPLCV_RISCV_UNITTEST_SRC "src/ppl/cv/riscv/*_unittest.cpp")
list(APPEND PPLCV_BENCHMARK_SRC ${PPLCV_RISCV_BENCHMARK_SRC})
list(APPEND PPLCV_UNITTEST_SRC ${PPLCV_RISCV_UNITTEST_SRC})
