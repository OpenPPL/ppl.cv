set(opencv_INCLUDE_DIRECTORIES )
set(opencv_LIBRARIES )

set(BUILD_LIST "ximgproc,core,imgproc,features2d,flann,imgcodecs,video,calib3d" CACHE INTERNAL "")

# --------------------------------------------------------------------------- #

if(PPLCV_USE_CUDA)
    set(WITH_CUDA ON)
    FetchContent_GetProperties(opencv_contrib)
    if(NOT opencv_contrib_POPULATED)
        FetchContent_Populate(opencv_contrib)
    endif()

    set(BUILD_LIST "${BUILD_LIST},ximgproc,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping" CACHE INTERNAL "")

    list(APPEND opencv_INCLUDE_DIRECTORIES
        ${opencv_contrib_SOURCE_DIR}/modules/ximgproc/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudev/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudaarithm/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudafilters/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudaimgproc/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudawarping/include)
    list(APPEND opencv_LIBRARIES opencv_cudev opencv_ximgproc opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc opencv_cudawarping)
    set(OPENCV_EXTRA_MODULES_PATH "${HPCC_DEPS_DIR}/opencv_contrib/modules" CACHE INTERNAL "")
endif()

# --------------------------------------------------------------------------- #

hpcc_populate_dep(opencv)

FetchContent_GetProperties(opencv_contrib)
if(NOT opencv_contrib_POPULATED)
    FetchContent_Populate(opencv_contrib)
endif()
list(APPEND opencv_INCLUDE_DIRECTORIES
    ${opencv_contrib_SOURCE_DIR}/modules/ximgproc/include
    ${opencv_SOURCE_DIR}/modules/features2d/include
    ${opencv_SOURCE_DIR}/modules/flann/include
    ${opencv_SOURCE_DIR}/modules/imgcodecs/include
    ${opencv_SOURCE_DIR}/modules/video/include
    ${opencv_SOURCE_DIR}/modules/calib3d/include
    ${opencv_SOURCE_DIR}/include
    ${opencv_SOURCE_DIR}/modules/core/include
    ${opencv_SOURCE_DIR}/modules/imgproc/include)
list(APPEND opencv_LIBRARIES opencv_ximgproc opencv_imgproc opencv_core )
set(OPENCV_EXTRA_MODULES_PATH "${HPCC_DEPS_DIR}/opencv_contrib/modules" CACHE INTERNAL "")
