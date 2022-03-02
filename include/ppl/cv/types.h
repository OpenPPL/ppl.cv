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

#ifndef __ST_HPC_PPL_CV_TYPES_H_
#define __ST_HPC_PPL_CV_TYPES_H_
#include "ppl/common/types.h"
#include <stdlib.h>

namespace ppl {
namespace cv {

/** Distance types for Distance Transform */
enum DistTypes {
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
    INTERPOLATION_LINEAR,       //!< Linear interpolation
    INTERPOLATION_NEAREST_POINT, //!< Nearest point interpolation
    INTERPOLATION_AREA //!< Area interpolation
};

enum BorderType {
    BORDER_CONSTANT       = 0, //!< `iiiiii|abcdefgh|iiiiii` with some specified `i`
    BORDER_REPLICATE      = 1, //!< `aaaaaa|abcdefgh|hhhhhh`
    BORDER_REFLECT        = 2, //!< `fedcba|abcdefgh|hgfedc`
    BORDER_WRAP           = 3, //!< `cdefgh|abcdefgh|abcdef`
    BORDER_REFLECT_101    = 4, //!< `gfedcb|abcdefgh|gfedcb`
    BORDER_TRANSPARENT    = 5, //!< `uvwxyz|abcdefgh|ijklmn`
    BORDER_REFLECT101     = BORDER_REFLECT_101,
    BORDER_DEFAULT        = BORDER_REFLECT_101,
    BORDER_ISOLATED       = 16
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
enum {
    INTER_NN        =0,
    INTER_LINEAR    =1,
    INTER_CUBIC     =2,
    INTER_AREA      =3,
    INTER_LANCZOS4  =4
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

enum {
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

} // namespace cv
} // namespace ppl

#ifndef __CUDACC__
#define CUDAC inline
#else
#define CUDAC inline __host__ __device__
#endif

#endif //! __ST_HPC_PPL_CV_TYPES_H_
