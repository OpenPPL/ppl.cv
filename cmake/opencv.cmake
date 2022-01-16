if(PPLCV_USE_CUDA)
    set(WITH_CUDA ON)
    set(BUILD_LIST "cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping")
endif()

set(BUILD_LIST "ximgproc,core,imgproc,features2d,flann,calib3d,${BUILD_LIST}" CACHE INTERNAL "")

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
    ${opencv_SOURCE_DIR}/modules/imgproc/include)
list(APPEND opencv_LIBRARIES opencv_ximgproc opencv_imgproc opencv_core)

if(PPLCV_USE_CUDA)
    list(APPEND opencv_INCLUDE_DIRECTORIES
        ${opencv_contrib_SOURCE_DIR}/modules/cudev/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudaarithm/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudafilters/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudaimgproc/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudawarping/include)
    list(APPEND opencv_LIBRARIES opencv_cudev opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc opencv_cudawarping)
endif()
