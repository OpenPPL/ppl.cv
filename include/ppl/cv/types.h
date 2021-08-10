// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_HPC_PPL3_CV_TYPES_H_
#define __ST_HPC_PPL3_CV_TYPES_H_
#include "ppl/common/types.h"
#include <stdlib.h>

namespace ppl {
namespace cv {

/**
 * \mainpage
 * The ppl.cv project is high-tuned image processing library with support
 * of \a arm, \a x86,\a CUDA, \a OpenCL, etc.
 * ### How to link ppl.cv's libraries in cmake ?
 * PPLCV provides a cmake configuration file \a PPL3Config.cmake to
 * help introducing ppl.cv's libraries to your cmake project.
 * @code{.cmake}
 * set(PPLCV_ROOT_DIR ${path-to-your-ppl.cv-location})
 * include(${PPLCV_ROOT_DIR}/cmake/PPL3Config.cmake)
 * add_executable(my_executable main.cpp)
 * target_link_libraries(my_executable PRIVATE PPLCV_static)
 * @endcode
 * If you are using PPLCV cuda version, you should also link against
 * cuda libraries:
 * @code{.cmake}
 * find_package(CUDA REQUIRED)
 * target_include_directories(my_executable PRIVATE ${CUDA_INCLUDE_DIRS})
 * target_link_libraries(my_executable PRIVATE PPLCV_static ${CUDA_LIBRARIES})
 * @endcode
 *********************************************************************/

/**
 * \page tutorial
 *********************************************************************/
/** Distance types for Distance Transform */
enum
{
    PPLCV_DIST_USER    =-1,  /**< User defined distance */
    PPLCV_DIST_L1      =1,   /**< distance = |x1-x2| + |y1-y2| */
    PPLCV_DIST_L2      =2,   /**< the simple euclidean distance */
    PPLCV_DIST_C       =3,   /**< distance = max(|x1-x2|,|y1-y2|) */
};

enum DistTypes{
    DIST_USER    =-1,  /**< User defined distance */
    DIST_L1      =1,   /**< distance = |x1-x2| + |y1-y2| */
    DIST_L2      =2,   /**< the simple euclidean distance */
    DIST_C       =3,   /**< distance = max(|x1-x2|,|y1-y2|) */
    DIST_L12     =4,   /**< L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1)) */
    DIST_FAIR    =5,   /**< distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998 */
    DIST_WELSCH  =6,   /**< distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846 */
    DIST_HUBER   =7    /**< distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345 */
};

//! Mask size for distance transform
enum DistanceTransformMasks {
    DIST_MASK_3       = 3, //!< mask=3
    DIST_MASK_5       = 5, //!< mask=5
    DIST_MASK_PRECISE = 0  //!<
};

/**
 * \brief
 * Interpolation algorithm type.
 **********************************/
enum InterpolationType {
    INTERPOLATION_TYPE_LINEAR,       //!< Linear interpolation
    INTERPOLATION_TYPE_NEAREST_POINT, //!< Nearest point interpolation
    INTERPOLATION_TYPE_AREA //!< Area interpolation
};

enum BorderType {
    BORDER_TYPE_CONSTANT       = 0, //!< `iiiiii|abcdefgh|iiiiii` with some specified `i`
    BORDER_TYPE_REPLICATE      = 1, //!< `aaaaaa|abcdefgh|hhhhhh`
    BORDER_TYPE_REFLECT        = 2, //!< `fedcba|abcdefgh|hgfedc`
    BORDER_TYPE_WRAP           = 3, //!< `cdefgh|abcdefgh|abcdef`
    BORDER_TYPE_REFLECT_101    = 4, //!< `gfedcb|abcdefgh|gfedcb`
    BORDER_TYPE_TRANSPARENT    = 5, //!< `uvwxyz|abcdefgh|ijklmn`
    BORDER_TYPE_REFLECT101     = BORDER_TYPE_REFLECT_101,
    BORDER_TYPE_DEFAULT        = BORDER_TYPE_REFLECT_101,
    BORDER_TYPE_ISOLATED       = 16
};

// copy liheng's code
// copied from opencv-4.1.0/modules/core/include/opencv2/core/base.hpp
enum InvertMethod {
    DECOMP_LU       = 0,
    DECOMP_SVD      = 1,
    DECOMP_EIG      = 2,
    DECOMP_CHOLESKY = 3,
    DECOMP_QR       = 4,
    DECOMP_NORMAL   = 16
};

// copied from opencv-4.1.0/modules/core/include/opencv2/core/base.hpp
enum NormTypes {
  NORM_INF       = 1,
  NORM_L1        = 2,
  NORM_L2        = 4,
  NORM_L2SQR     = 5,
  NORM_HAMMING   = 6,
  NORM_HAMMING2  = 7,
  NORM_TYPE_MASK = 7,
  NORM_RELATIVE  = 8,
  NORM_MINMAX    = 32,
};

enum ThresholdTypes {
    THRESH_BINARY     = 0,
    THRESH_BINARY_INV = 1
};

/* Sub-pixel interpolation methods */
enum
{
    INTER_NN        =0,
    INTER_LINEAR    =1,
    INTER_CUBIC     =2,
    INTER_AREA      =3,
    INTER_LANCZOS4  =4
};

/* ... and other image warping flags */
enum
{
    WARP_FILL_OUTLIERS =8,
    WARP_INVERSE_MAP  =16
};

// wavelet kernels
enum WaveletFamily {
    WAVELET_HAAR = 0,
    WAVELET_DB1,
    WAVELET_DB2,
};

//! type of morphological operation
enum MorphTypes{
    MORPH_ERODE    = 0, //!< see #erode
    MORPH_DILATE   = 1, //!< see #dilate
    MORPH_OPEN     = 2, //!< an opening operation
                        //!< \f[\texttt{dst} = \mathrm{open} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \mathrm{erode} ( \texttt{src} , \texttt{element} ))\f]
    MORPH_CLOSE    = 3, //!< a closing operation
                        //!< \f[\texttt{dst} = \mathrm{close} ( \texttt{src} , \texttt{element} )= \mathrm{erode} ( \mathrm{dilate} ( \texttt{src} , \texttt{element} ))\f]
    MORPH_GRADIENT = 4, //!< a morphological gradient
                        //!< \f[\texttt{dst} = \mathrm{morph\_grad} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \texttt{src} , \texttt{element} )- \mathrm{erode} ( \texttt{src} , \texttt{element} )\f]
    MORPH_TOPHAT   = 5, //!< "top hat"
                        //!< \f[\texttt{dst} = \mathrm{tophat} ( \texttt{src} , \texttt{element} )= \texttt{src} - \mathrm{open} ( \texttt{src} , \texttt{element} )\f]
    MORPH_BLACKHAT = 6, //!< "black hat"
                        //!< \f[\texttt{dst} = \mathrm{blackhat} ( \texttt{src} , \texttt{element} )= \mathrm{close} ( \texttt{src} , \texttt{element} )- \texttt{src}\f]
    MORPH_HITMISS  = 7  //!< "hit or miss"
                        //!<   .- Only supported for CV_8UC1 binary images. A tutorial can be found in the documentation
};

typedef unsigned char uchar;   //!< Type alias. For some code backward compatibility.
typedef unsigned short ushort; //!< Type alias. For some code backward compatibility.

enum AdaptiveThresholdTypes {
    /** the threshold value \f$T(x,y)\f$ is a mean of the \f$\texttt{blockSize} \times
    \texttt{blockSize}\f$ neighborhood of \f$(x, y)\f$ minus C */
    ADAPTIVE_THRESH_MEAN_C     = 0,
    /** the threshold value \f$T(x, y)\f$ is a weighted sum (cross-correlation with a Gaussian
    window) of the \f$\texttt{blockSize} \times \texttt{blockSize}\f$ neighborhood of \f$(x, y)\f$
    minus C . The default sigma (standard deviation) is used for the specified blockSize . See
    #getGaussianKernel*/
    ADAPTIVE_THRESH_GAUSSIAN_C = 1
};

enum
{
    CV_THRESH_BINARY      =0,  /**< value = value > threshold ? max_value : 0       */
    CV_THRESH_BINARY_INV  =1,  /**< value = value > threshold ? 0 : max_value       */
    CV_THRESH_TRUNC       =2,  /**< value = value > threshold ? threshold : value   */
    CV_THRESH_TOZERO      =3,  /**< value = value > threshold ? value : 0           */
    CV_THRESH_TOZERO_INV  =4,  /**< value = value > threshold ? 0 : value           */
    CV_THRESH_MASK        =7,
    CV_THRESH_OTSU        =8, /**< use Otsu algorithm to choose the optimal threshold value;
                                 combine the flag with one of the above CV_THRESH_* values */
    CV_THRESH_TRIANGLE    =16  /**< use Triangle algorithm to choose the optimal threshold value;
                                 combine the flag with one of the above CV_THRESH_* values, but not
                                 with CV_THRESH_OTSU */
};

enum
{
    INPAINT_NS    = 0, //!< Use Navier-Stokes based method
    INPAINT_TELEA = 1 //!< Use the algorithm proposed by Alexandru Telea @cite Telea04
};

} // namespace cv
} // namespace ppl

#ifndef __CUDACC__
#define CUDAC inline
#else
#define CUDAC inline __host__ __device__
#endif

//<<<<<<< HEAD
//#ifdef __cplusplus
// extern "C" {
//#endif
// typedef unsigned char   uchar;
// typedef char            int8;
// typedef unsigned char   uint8;
// typedef short           int16;
// typedef unsigned short  uint16;
// typedef unsigned short  ushort;
// typedef int             int32;
// typedef unsigned int    uint32;
// typedef unsigned int    uint;
// typedef float           float32;
// typedef double          float64;
//=======
//#ifdef __cplusplus
//extern "C" {
//#endif
//typedef unsigned char   uchar;
////typedef char            int8;
////typedef unsigned char   uint8;
////typedef short           int16;
////typedef unsigned short  uint16;
//typedef unsigned short  ushort;
////typedef int             int32;
////typedef unsigned int    uint32;
////typedef unsigned int    uint;
////typedef float           float32;
////typedef double          float64;
//>>>>>>> origin/ckl/arm
////
//#ifdef __cplusplus
//}
//#endif
//
// template<typename T>
// struct Complex {
//    T real;
//    T image;
//};
// typedef Complex<float32> complex64;
// typedef Complex<float64> complex128;
//
// template<typename T>
// struct V2 {
//    T x;
//    T y;
//};
// typedef V2<int8>        int8x2;
// typedef V2<uint8>       uint8x2;
// typedef V2<int16>       int16x2;
// typedef V2<uint16>      uint16x2;
// typedef V2<int32>       int32x2;
// typedef V2<uint32>      uint32x2;
// typedef V2<float32>     float32x2;
// typedef V2<float64>     float64x2;
// typedef V2<complex64>   complex64x2;
//
// template<typename T>
// struct V3 {
//    T x;
//    T y;
//    T z;
//};
// typedef V3<int8>        int8x3;
// typedef V3<uint8>       uint8x3;
// typedef V3<int16>       int16x3;
// typedef V3<uint16>      uint16x3;
// typedef V3<int32>       int32x3;
// typedef V3<uint32>      uint32x3;
// typedef V3<float32>     float32x3;
// typedef V3<float64>     float64x3;
//
// template<typename T>
// struct V4 {
//    T x;
//    T y;
//    T z;
//    T w;
//};
// typedef V4<int8>        int8x4;
// typedef V4<uint8>       uint8x4;
// typedef V4<int16>       int16x4;
// typedef V4<uint16>      uint16x4;
// typedef V4<int32>       int32x4;
// typedef V4<uint32>      uint32x4;
// typedef V4<float32>     float32x4;
// typedef V4<float64>     float64x4;


#endif //! __ST_HPC_PPL3_CV_TYPES_H_
