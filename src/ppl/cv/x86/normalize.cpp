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

#include "ppl/cv/x86/minMaxLoc.h"
#include "ppl/cv/x86/normalize.h"
#include "ppl/cv/x86/norm.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>

#include <vector>
#include <limits.h>
#include <immintrin.h>
#include <cassert>
#include <float.h>
#ifdef _WIN32
#include <algorithm>
#endif
namespace ppl {
namespace cv {
namespace x86 {

template <typename T, int nc>
::ppl::common::RetCode Normalize(int height,
                                 int width,
                                 int inWidthStride,
                                 const T *inData,
                                 int outWidthStride,
                                 float *outData,
                                 double alpha,
                                 double beta,
                                 ppl::cv::NormTypes normType,
                                 int maskSteps,
                                 const uchar *mask)
{
    assert(height != 0 && width != 0 && inWidthStride != 0);
    if (mask != NULL) {
        assert(maskSteps != 0);
    }
    double scale = 0.0;
    if (mask == NULL) {
        if (normType == ppl::cv::NORM_L1 || normType == ppl::cv::NORM_L2 || normType == ppl::cv::NORM_INF) {
            scale = Norm<T, nc>(height, width, inWidthStride, inData, normType, maskSteps, mask);
            scale = scale > DBL_EPSILON ? alpha / scale : 0.;
            for (int i = 0; i < height; i++) {
                float *dst_ptr = outData + i * outWidthStride;
                for (int j = 0; j < width; j++) {
                    for (int k = 0; k < nc; k++) {
                        float value         = inData[i * inWidthStride + j * nc + k];
                        dst_ptr[j * nc + k] = value * scale;
                    }
                }
            }
        } else if (normType == ppl::cv::NORM_MINMAX && mask == nullptr) {
            T minval = 0, maxval = 0;
            int mincol, minrow, maxcol, maxrow;
            MinMaxLoc<T>(height, width * nc, inWidthStride, (T *)inData, &minval, &maxval, &mincol, &minrow, &maxcol, &maxrow);
            double scale = (std::max(alpha, beta) - std::min(alpha, beta)) * (maxval - minval > DBL_EPSILON ? 1. / (maxval - minval) : 0);
            double shift = std::min(alpha, beta) - minval * scale;
            for (int i = 0; i < height; i++) {
                float *dst_ptr = outData + i * outWidthStride;
                for (int j = 0; j < width; j++) {
                    for (int k = 0; k < nc; k++) {
                        float value         = inData[i * inWidthStride + j * nc + k];
                        dst_ptr[j * nc + k] = value * scale + shift;
                    }
                }
            }
        }
    } else if (mask != NULL) {
        if (normType == ppl::cv::NORM_L1 || normType == ppl::cv::NORM_L2 || normType == ppl::cv::NORM_INF) {
            scale = Norm<T, nc>(height, width, inWidthStride, inData, normType, maskSteps, mask);
            scale = scale > DBL_EPSILON ? alpha / scale : 0.;
            for (int i = 0; i < height; i++) {
                float *dst_ptr   = outData + i * outWidthStride;
                const T *src_ptr = inData + i * inWidthStride;
                for (int j = 0; j < width; j++) {
                    float mask_value = mask[i * maskSteps + j];
                    for (int k = 0; k < nc; k++) {
                        float value         = src_ptr[j * nc + k];
                        value               = mask_value > 0 ? value : 0;
                        dst_ptr[j * nc + k] = value * (float)scale;
                    }
                }
            }
        } else if (normType == ppl::cv::NORM_MINMAX && nc == 1) {
            T minval, maxval;
            int mincol, minrow, maxcol, maxrow;
            MinMaxLoc<T>(height, width, width, inData, &minval, &maxval, &mincol, &minrow, &maxcol, &maxrow, maskSteps, mask);
            float scale = ((maxval - minval) > DBL_EPSILON ? (std::max(alpha, beta) - std::min(alpha, beta)) / (maxval - minval) : 0.0);
            float shift = (std::min(alpha, beta)) - minval * scale;
            for (int i = 0; i < height; i++) {
                float *dst_ptr   = outData + i * outWidthStride;
                const T *src_ptr = inData + i * inWidthStride;
                for (int j = 0; j < width; j++) {
                    float mask_value = mask[i * maskSteps + j];
                    float value      = src_ptr[j];
                    value            = mask_value > 0 ? value : 0;
                    dst_ptr[j]       = value * scale + shift;
                }
            }
        } else if (normType == ppl::cv::NORM_MINMAX && nc > 1) {
            return ppl::common::RC_UNSUPPORTED; // when normType == NORM_MINMAX, only nc == 1 is supported.
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Normalize<uchar, 1>(int height, int width, int inWidthStride, const uchar *inData, int outWidthStride, float *outData, double alpha, double beta, ppl::cv::NormTypes normType, int maskSteps, const uchar *mask);
template ::ppl::common::RetCode Normalize<uchar, 3>(int height, int width, int inWidthStride, const uchar *inData, int outWidthStride, float *outData, double alpha, double beta, ppl::cv::NormTypes normType, int maskSteps, const uchar *mask);
template ::ppl::common::RetCode Normalize<uchar, 4>(int height, int width, int inWidthStride, const uchar *inData, int outWidthStride, float *outData, double alpha, double beta, ppl::cv::NormTypes normType, int maskSteps, const uchar *mask);
template ::ppl::common::RetCode Normalize<float, 1>(int height, int width, int inWidthStride, const float *inData, int outWidthStride, float *outData, double alpha, double beta, ppl::cv::NormTypes normType, int maskSteps, const uchar *mask);
template ::ppl::common::RetCode Normalize<float, 3>(int height, int width, int inWidthStride, const float *inData, int outWidthStride, float *outData, double alpha, double beta, ppl::cv::NormTypes normType, int maskSteps, const uchar *mask);
template ::ppl::common::RetCode Normalize<float, 4>(int height, int width, int inWidthStride, const float *inData, int outWidthStride, float *outData, double alpha, double beta, ppl::cv::NormTypes normType, int maskSteps, const uchar *mask);

}
}
} // namespace ppl::cv::x86
