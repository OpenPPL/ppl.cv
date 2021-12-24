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
#ifdef _WIN32
#include <algorithm>
#endif
namespace ppl {
namespace cv {
namespace x86 {

inline float norm_l2(const float &v)
{
    return v * v;
}

inline float norm_l1(const float &v)
{
    return std::abs(v);
}

inline float norm_inf(const float &v, const float &maximum)
{
    return std::max(std::abs(v), maximum);
}

template <typename T, int nc, bool use_mask, ppl::cv::NormTypes norm_type>
double Norm(int inHeight,
            int inWidth,
            int inWidthStride,
            const T *inData,
            int maskWidthStride,
            const uchar *mask)
{
    double result    = 0.0;
    float result_inf = 0.0;
    for (int i = 0; i < inHeight; ++i) {
        for (int j = 0; j < inWidth; ++j) {
            float mask_value;
            if (use_mask) {
                mask_value = static_cast<float>(mask[i * maskWidthStride + j] != 0);
            }
            for (int k = 0; k < nc; ++k) {
                float value = inData[i * inWidthStride + j * nc + k];
                if (use_mask) {
                    value *= mask_value;
                }
                if (norm_type == ppl::cv::NORM_L1) {
                    result += norm_l1(value);
                } else if (norm_type == ppl::cv::NORM_L2) {
                    result += norm_l2(value);
                } else if (norm_type == ppl::cv::NORM_INF) {
                    result_inf = norm_inf(value, result_inf);
                }
            }
        }
    }
    if (norm_type == ppl::cv::NORM_INF) {
        return static_cast<double>(result_inf);
    } else if (norm_type == ppl::cv::NORM_L2) {
        return std::sqrt(result);
    } else {
        return result;
    }
}

template <typename T, int numChannels>
double Norm(int inHeight,
            int inWidth,
            int inWidthStride,
            const T *inData,
            ppl::cv::NormTypes norm_type,
            int maskWidthStride,
            const uchar *mask)
{
    assert(inHeight != 0 && inWidth != 0 && inWidthStride != 0);
    assert(norm_type == ppl::cv::NORM_L1 || norm_type == ppl::cv::NORM_L2 || norm_type == ppl::cv::NORM_INF);
    if (mask != NULL) {
        assert(maskWidthStride != 0);
    }
    if (norm_type == ppl::cv::NORM_L1) {
        if (mask == NULL) {
            return Norm<T, numChannels, false, ppl::cv::NORM_L1>(inHeight, inWidth, inWidthStride, inData, maskWidthStride, nullptr);
        } else {
            return Norm<T, numChannels, true, ppl::cv::NORM_L1>(inHeight, inWidth, inWidthStride, inData, maskWidthStride, mask);
        }
    } else if (norm_type == ppl::cv::NORM_L2) {
        if (mask == NULL) {
            return Norm<T, numChannels, false, ppl::cv::NORM_L2>(inHeight, inWidth, inWidthStride, inData, maskWidthStride, nullptr);
        } else {
            return Norm<T, numChannels, true, ppl::cv::NORM_L2>(inHeight, inWidth, inWidthStride, inData, maskWidthStride, mask);
        }
    } else if (norm_type == ppl::cv::NORM_INF) {
        if (mask == NULL) {
            return Norm<T, numChannels, false, ppl::cv::NORM_INF>(inHeight, inWidth, inWidthStride, inData, maskWidthStride, nullptr);
        } else {
            return Norm<T, numChannels, true, ppl::cv::NORM_INF>(inHeight, inWidth, inWidthStride, inData, maskWidthStride, mask);
        }
    }
    return 0.0;
}

template double Norm<uchar, 1>(int inHeight, int inWidth, int inWidthStride, const uchar *inData, ppl::cv::NormTypes norm_type, int maskWidthStride, const uchar *mask);

template double Norm<uchar, 3>(int inHeight, int inWidth, int inWidthStride, const uchar *inData, ppl::cv::NormTypes norm_type, int maskWidthStride, const uchar *mask);

template double Norm<uchar, 4>(int inHeight, int inWidth, int inWidthStride, const uchar *inData, ppl::cv::NormTypes norm_type, int maskWidthStride, const uchar *mask);

template double Norm<float, 1>(int inHeight, int inWidth, int inWidthStride, const float *inData, ppl::cv::NormTypes norm_type, int maskWidthStride, const uchar *mask);

template double Norm<float, 3>(int inHeight, int inWidth, int inWidthStride, const float *inData, ppl::cv::NormTypes norm_type, int maskWidthStride, const uchar *mask);

template double Norm<float, 4>(int inHeight, int inWidth, int inWidthStride, const float *inData, ppl::cv::NormTypes norm_type, int maskWidthStride, const uchar *mask);

}
}
} // namespace ppl::cv::x86
