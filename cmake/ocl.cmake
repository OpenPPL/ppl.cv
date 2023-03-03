if(NOT PPLCV_OPENCL_INCLUDE_DIRS)
    message(FATAL_ERROR "`PPLCV_OPENCL_INCLUDE_DIRS` is not specified.")
endif()
if(NOT PPLCV_OPENCL_LIBRARIES)
    message(FATAL_ERROR "`PPLCV_OPENCL_LIBRARIES` is not specified.")
endif()
if(NOT CL_TARGET_OPENCL_VERSION)
    message(FATAL_ERROR "`CL_TARGET_OPENCL_VERSION` is not specified.")
endif()

list(APPEND PPLCV_COMPILE_DEFINITIONS
    PPLCV_USE_OPENCL
    CL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})
list(APPEND PPLCV_INCLUDE_DIRECTORIES ${PPLCV_OPENCL_INCLUDE_DIRS})
list(APPEND PPLCV_LINK_LIBRARIES ${PPLCV_OPENCL_LIBRARIES})

file(GLOB __PPLCV_OCL_SRC__ src/ppl/cv/ocl/*.cpp)
list(APPEND PPLCV_SRC ${__PPLCV_OCL_SRC__})
unset(__PPLCV_OCL_SRC__)

file(STRINGS src/ppl/cv/ocl/kerneltypes.h HEADER_STRING NEWLINE_CONSUME)
file(GLOB KERNEL_FILES src/ppl/cv/ocl/*.cl)
foreach(KERNEL_FILE0 IN ITEMS ${KERNEL_FILES})
    file(STRINGS ${KERNEL_FILE0} KERNEL_STRING NEWLINE_CONSUME)
    string(HEX ${KERNEL_STRING} KERNEL_HEX)
    string(HEX ${HEADER_STRING} HEADER_HEX)
    string(HEX "#include \"kerneltypes.h\"\n" INCLUDE_HEX)
    string(REPLACE ${INCLUDE_HEX} ${HEADER_HEX} CONTENT_HEX0 ${KERNEL_HEX})

    string(REGEX REPLACE "(.)(.)" "0x\\1\\2, " CONTENT_HEX1 ${CONTENT_HEX0})
    string(REGEX MATCH "[0-9A-Za-z_]+\.cl$" KERNEL_FILE1 ${KERNEL_FILE0})
    string(REGEX MATCH "^[0-9A-Za-z_]+" KERNEL_FILE2 ${KERNEL_FILE1})
    set(KERNEL_CONTENT
        "static const char ${KERNEL_FILE2}_string[] = {${CONTENT_HEX1}0x00}\;")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/kernels/${KERNEL_FILE1}
         ${KERNEL_CONTENT})
endforeach()

add_executable(compile_kernels src/ppl/cv/ocl/utility/binarycompilation.cpp)
target_compile_features(compile_kernels PRIVATE cxx_std_11)
target_include_directories(compile_kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(compile_kernels PRIVATE pplcommon_static)

file(GLOB __OCL_UNITTEST_SRC__ "src/ppl/cv/ocl/*_unittest.cpp")
list(APPEND PPLCV_UNITTEST_SRC ${__OCL_UNITTEST_SRC__})
unset(__OCL_UNITTEST_SRC__)

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
