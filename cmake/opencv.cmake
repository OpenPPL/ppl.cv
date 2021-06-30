set(opencv_INCLUDE_DIRECTORIES )
set(opencv_LIBRARIES )

set(BUILD_LIST "core,imgproc" CACHE INTERNAL "")

# --------------------------------------------------------------------------- #

if(PPLCV_USE_CUDA)
    FetchContent_GetProperties(opencv_contrib)
    if(NOT opencv_contrib_POPULATED)
        FetchContent_Populate(opencv_contrib)
    endif()

    set(BUILD_LIST "${BUILD_LIST},cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping" CACHE INTERNAL "")

    list(APPEND opencv_INCLUDE_DIRECTORIES
        ${opencv_contrib_SOURCE_DIR}/modules/cudev/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudaarithm/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudafilters/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudaimgproc/include
        ${opencv_contrib_SOURCE_DIR}/modules/cudawarping/include)
    list(APPEND opencv_LIBRARIES opencv_cudev opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc opencv_cudawarping)
    set(OPENCV_EXTRA_MODULES_PATH "${HPCC_DEPS_DIR}/opencv_contrib/modules" CACHE INTERNAL "")
endif()

# --------------------------------------------------------------------------- #

hpcc_populate_dep(opencv)

list(APPEND opencv_INCLUDE_DIRECTORIES
    ${opencv_SOURCE_DIR}/include
    ${opencv_SOURCE_DIR}/modules/core/include
    ${opencv_SOURCE_DIR}/modules/imgproc/include)
list(APPEND opencv_LIBRARIES opencv_imgproc opencv_core)
