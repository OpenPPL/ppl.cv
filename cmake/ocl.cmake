list(APPEND PPLCV_COMPILE_DEFINITIONS PPLCV_USE_OPENCL)

file(GLOB_RECURSE __PPLCV_OCL_SRC__ src/ppl/cv/ocl/*.cpp)
list(APPEND PPLCV_SRC ${__PPLCV_OCL_SRC__})
unset(__PPLCV_OCL_SRC__)

list(APPEND PPLCV_LINK_LIBRARIES OpenCL)

file(STRINGS src/ppl/cv/ocl/kerneltypes.h HEADER_STRING NEWLINE_CONSUME)
file(GLOB KERNEL_FILES src/ppl/cv/ocl/*.cl)
# message("kernel files: ${KERNEL_FILES}")
foreach(KERNEL_FILE IN ITEMS ${KERNEL_FILES})
    # message("kernel file: ${KERNEL_FILE}")
    file(STRINGS ${KERNEL_FILE} KERNEL_CONTENT0 NEWLINE_CONSUME)
    string(REPLACE "#include \"kerneltypes.h\"\n" ${HEADER_STRING}
           KERNEL_CONTENT1 ${KERNEL_CONTENT0})
    string(CONCAT KERNEL_CONTENT2 &{KERNEL_CONTENT1})
    # file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/kernels/test.ocl KERNEL_CONTENT2)
    # list(LENGTH KERNEL_CONTENT2 length2)
    # message(STATUS "KERNEL_CONTENT2 length: ${length2}")
    # message("KERNEL_CONTENT2 file: " KERNEL_CONTENT2)
    string(HEX ${KERNEL_CONTENT2} CONTENT_HEX0)
    string(REGEX REPLACE "(.)(.)" "0x\\1\\2, " CONTENT_HEX1 ${CONTENT_HEX0})
    # message("CONTENT_HEX1 file: ${CONTENT_HEX1}")
    set(KERNEL_STRING
        "static const char source_string[] = {${CONTENT_HEX1}0x00}\;")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/kernels/${KERNEL_FILE}
         ${KERNEL_STRING})
endforeach()

file(GLOB __OCL_UNITTEST_SRC__ "src/ppl/cv/ocl/*_unittest.cpp")
list(APPEND PPLCV_UNITTEST_SRC ${__OCL_UNITTEST_SRC__})
unset(__OCL_UNITTEST_SRC__)

file(GLOB __OCL_BENCHMARK_SRC__ "src/ppl/cv/ocl/*_benchmark.cpp")
list(APPEND PPLCV_BENCHMARK_SRC ${__OCL_BENCHMARK_SRC__})
unset(__OCL_BENCHMARK_SRC__)

if(PPLCV_INSTALL)
    file(GLOB __OCL_HEADERS__ src/ppl/cv/ocl/*.h)
    install(FILES ${__OCL_HEADERS__} DESTINATION include/ppl/cv/ocl)
    unset(__OCL_HEADERS__)
endif()
