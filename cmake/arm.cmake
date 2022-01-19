# PPL CV AARCH64 source cmake script
file(GLOB PPLCV_AARCH64_PUBLIC_HEADERS src/ppl/cv/arm/*.h)
install(FILES ${PPLCV_AARCH64_PUBLIC_HEADERS}
        DESTINATION include/ppl/cv/arm)

set(PPLCV_USE_AARCH64 ON)
list(APPEND PPLCV_COMPILE_DEFINITIONS PPLCV_USE_AARCH64)

file(GLOB PPLCV_AARCH64_SRC
     src/ppl/cv/arm/*.cpp)

if(ANDROID)
    list(REMOVE_ITEM PPLCV_AARCH64_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/morph_f32.cpp")
    list(REMOVE_ITEM PPLCV_AARCH64_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/morph_u8.cpp")
    list(REMOVE_ITEM PPLCV_AARCH64_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/dilate.cpp")
    list(REMOVE_ITEM PPLCV_AARCH64_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/erode.cpp")
endif()

list(APPEND PPLCV_SRC ${PPLCV_AARCH64_SRC})

# glob benchmark and unittest sources
file(GLOB PPLCV_AARCH64_BENCHMARK_SRC "src/ppl/cv/arm/*_benchmark.cpp")
file(GLOB PPLCV_AARCH64_UNITTEST_SRC "src/ppl/cv/arm/*_unittest.cpp")

if(ANDROID)
    list(REMOVE_ITEM PPLCV_AARCH64_UNITTEST_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/morph_unittest.cpp")
    list(REMOVE_ITEM PPLCV_AARCH64_UNITTEST_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/dilate_unittest.cpp")
    list(REMOVE_ITEM PPLCV_AARCH64_UNITTEST_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/erode_unittest.cpp")

    list(REMOVE_ITEM PPLCV_AARCH64_BENCHMARK_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/dilate_benchmark.cpp")
    list(REMOVE_ITEM PPLCV_AARCH64_BENCHMARK_SRC "${CMAKE_SOURCE_DIR}/src/ppl/cv/arm/erode_benchmark.cpp")
endif()

list(APPEND PPLCV_BENCHMARK_SRC ${PPLCV_AARCH64_BENCHMARK_SRC})
list(APPEND PPLCV_UNITTEST_SRC ${PPLCV_AARCH64_UNITTEST_SRC})
