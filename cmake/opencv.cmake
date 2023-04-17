if(PPLCV_USE_CUDA)
    set(WITH_CUDA ON)
    set(BUILD_LIST "cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping")
    if(CMAKE_COMPILER_IS_GNUCC)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    endif()
endif()

if(PPLCV_USE_RISCV)
    if(PPLCV_RISCV_RVV_0P71)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__riscv_vector_071")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__riscv_vector_071")
    endif()
endif()

set(BUILD_PNG ON CACHE BOOL "")
set(BUILD_LIST "ximgproc,core,imgproc,imgcodecs,features2d,flann,calib3d,${BUILD_LIST}" CACHE INTERNAL "")

hpcc_populate_dep(opencv_contrib)
hpcc_populate_dep(opencv)

# --------------------------------------------------------------------------- #

set(opencv_INCLUDE_DIRECTORIES )
set(opencv_LIBRARIES )

list(APPEND opencv_INCLUDE_DIRECTORIES
    ${opencv_contrib_SOURCE_DIR}/modules/ximgproc/include
    ${opencv_SOURCE_DIR}/modules/features2d/include
    ${opencv_SOURCE_DIR}/modules/flann/include
    ${opencv_SOURCE_DIR}/modules/calib3d/include
    ${opencv_SOURCE_DIR}/modules/core/include
    ${opencv_SOURCE_DIR}/modules/imgproc/include
    ${opencv_SOURCE_DIR}/modules/imgcodecs/include)
list(APPEND opencv_LIBRARIES opencv_ximgproc opencv_imgproc opencv_core opencv_imgcodecs)

if(PPLCV_USE_CUDA)
    list(APPEND opencv_INCLUDE_DIRECTORIES
        ${opencv_contrib_SOURCE_DIR}/modules/cudev/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudaarithm/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudafilters/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudaimgproc/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudawarping/include)
    list(APPEND opencv_LIBRARIES opencv_cudev opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc opencv_cudawarping)
endif()
