list(APPEND PPLCV_COMPILE_DEFINITIONS PPLCV_USE_OCL)

file(GLOB_RECURSE __PPLCV_OCL_SRC__ src/ppl/cv/ocl/*.cpp)
list(APPEND PPLCV_SRC ${__PPLCV_OCL_SRC__})
unset(__PPLCV_OCL_SRC__)

list(APPEND PPLCV_LINK_LIBRARIES OpenCL)

# file(GLOB KERNEL_ORIG src/ppl/cv/ocl/*.cl)
#     file(STRINGS test.cl KERNEL NEWLINE_CONSUME NO_HEX_CONVERSION)
#     set(KERNEL_STRING "static const char* source = \"${KERNEL}\"")
#     file(WRITE test.ocl ${KERNEL_STRING})
file(STRINGS src/ppl/cv/ocl/abs.cl KERNEL)
set(KERNEL_STRING "static char* source_string = \"${KERNEL}\"\;")
file(WRITE src/ppl/cv/ocl/abs.ocl ${KERNEL_STRING})


# ----- installation ----- #

file(GLOB PPLCV_OCL_HEADERS src/ppl/cv/ocl/*.h)
install(FILES ${PPLCV_OCL_HEADERS} DESTINATION include/ppl/cv/ocl)
