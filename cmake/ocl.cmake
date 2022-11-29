list(APPEND PPLCV_COMPILE_DEFINITIONS PPLCV_USE_OPENCL)

file(GLOB __PPLCV_OCL_SRC__ src/ppl/cv/ocl/*.cpp)
# file(GLOB_RECURSE __PPLCV_OCL_SRC__ src/ppl/cv/ocl/*.cpp)
list(APPEND PPLCV_SRC ${__PPLCV_OCL_SRC__})
unset(__PPLCV_OCL_SRC__)

<<<<<<< HEAD
list(APPEND PPLCV_LINK_LIBRARIES OpenCL)

=======
>>>>>>> 46a7183 ([opt][ocl]optimize abs with vload/vstore.)
file(STRINGS src/ppl/cv/ocl/kerneltypes.h HEADER_STRING NEWLINE_CONSUME)
file(GLOB KERNEL_FILES src/ppl/cv/ocl/*.cl)
# message("all kernel files: ${KERNEL_FILES}")
foreach(KERNEL_FILE0 IN ITEMS ${KERNEL_FILES})
    # message("a kernel file: ${KERNEL_FILE0}")
    file(STRINGS ${KERNEL_FILE0} KERNEL_STRING NEWLINE_CONSUME)
    # message("source of kernel file: " ${KERNEL_STRING})  #debug
    # message("header of header file: " ${HEADER_STRING})  #debug
    string(HEX ${KERNEL_STRING} KERNEL_HEX)
    string(HEX ${HEADER_STRING} HEADER_HEX)
    string(HEX "#include \"kerneltypes.h\"\n" INCLUDE_HEX)
    string(REPLACE ${INCLUDE_HEX} ${HEADER_HEX} CONTENT_HEX0 ${KERNEL_HEX})
    # list(LENGTH CONTENT_HEX0 length2)  #debug
    # message(STATUS "CONTENT_HEX0 length: ${length2}")  #debug
    # message("HEX source + include of kernel file: " ${CONTENT_HEX0})  #debug

    string(REGEX REPLACE "(.)(.)" "0x\\1\\2, " CONTENT_HEX1 ${CONTENT_HEX0})
    # message("CONTENT_HEX1 file: ${CONTENT_HEX1}")
    set(KERNEL_CONTENT
        "static const char source_string[] = {${CONTENT_HEX1}0x00}\;")
    # message("KERNEL_CONTENT file: ${KERNEL_CONTENT}")
    # message("kernel file: ${KERNEL_FILE0}")
    string(REGEX MATCH "[0-9A-Za-z_]+\.cl$" KERNEL_FILE1 ${KERNEL_FILE0})
    # message("kernel file1: ${KERNEL_FILE1}")  #debug
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/kernels/${KERNEL_FILE1}
         ${KERNEL_CONTENT})
endforeach()

file(GLOB __OCL_UNITTEST_SRC__ "src/ppl/cv/ocl/*_unittest.cpp")
list(APPEND PPLCV_UNITTEST_SRC ${__OCL_UNITTEST_SRC__})
unset(__OCL_UNITTEST_SRC__)
# message("ocl unittest file: ${PPLCV_UNITTEST_SRC}")  #debug

file(GLOB __OCL_BENCHMARK_SRC__ "src/ppl/cv/ocl/*_benchmark.cpp")
list(APPEND PPLCV_BENCHMARK_SRC ${__OCL_BENCHMARK_SRC__})
unset(__OCL_BENCHMARK_SRC__)

if(PPLCV_INSTALL)
    file(GLOB __OCL_HEADERS__ src/ppl/cv/ocl/*.h)
    file(GLOB __KERNEL_HEADERS__ src/ppl/cv/ocl/kerneltypes.h)
    list(REMOVE_ITEM __OCL_HEADERS__ ${__KERNEL_HEADERS__})
    install(FILES ${__OCL_HEADERS__} DESTINATION include/ppl/cv/ocl)
    unset(__OCL_HEADERS__)
    unset(__KERNEL_HEADERS__)
endif()
