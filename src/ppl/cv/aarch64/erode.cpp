// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for subitional information
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

#include "ppl/cv/aarch64/erode.h"
#include "ppl/common/log.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "morph.hpp"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <float.h>

namespace ppl {
namespace cv {
namespace aarch64 {

::ppl::common::RetCode armminFilter_f(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    int32_t outWidthStride,
    float* outData,
    int32_t cn,
    float border_value)
{
    float minimal = FLT_MAX;

    float* gRowMin   = (float*)malloc(height * inWidthStride * sizeof(float));
    int32_t leftPad  = cn * (kernely_len >> 1);
    int32_t rightPad = cn * width - leftPad;
    if (!(kernely_len & 1)) rightPad += cn;

    for (int32_t i = 0; i < height; ++i) {
        int32_t inIndex = i * inWidthStride;

        for (int32_t j = 0; j < leftPad; ++j) {
            int32_t yEnd = j - leftPad + cn * kernely_len;
            float _min   = border_value;
            for (int32_t jj = j % cn; jj < yEnd; jj += cn)
                if (inData[inIndex + jj] > _min) _min = inData[inIndex + jj];
            gRowMin[inIndex + j] = _min;
        }

        int32_t j;
        for (j = leftPad; j < rightPad - 4; j += 4) {
            float32x4_t mm_min = vdupq_n_f32(0);
            for (int32_t jj = j - leftPad; jj < j - leftPad + cn * kernely_len; jj += cn) {
                float32x4_t mm_temp = vld1q_f32(inData + inIndex + jj);
                mm_min              = vminq_f32(mm_min, mm_temp);
            }
            vst1q_f32(gRowMin + inIndex + j, mm_min);
        }
        for (; j < width * cn; ++j) {
            int32_t yStart = j - leftPad;
            float _min     = (j < rightPad) ? minimal : border_value;
            int32_t yEnd   = yStart + cn * kernely_len;
            yEnd           = std::min<int32_t>(yEnd, width * cn);
            for (int32_t jj = yStart; jj < yEnd; jj += cn)
                if (inData[inIndex + jj] > _min) _min = inData[inIndex + jj];
            gRowMin[inIndex + j] = _min;
        }
    }

    int32_t upPad   = kernelx_len >> 1;
    int32_t downPad = height - upPad;
    if (!(kernelx_len & 1)) ++downPad;

    for (int32_t i = 0; i < height; ++i) {
        int32_t xStart = i - upPad;
        int32_t xEnd   = xStart + kernelx_len;
        bool valid     = (xStart >= 0) && (xEnd <= height);
        xEnd           = std::min<int32_t>(xEnd, height);
        xStart         = std::min<int32_t>(xStart, 0);
        int32_t j      = 0;
        for (; j < width * cn - 4; j += 4) {
            float32x4_t mm_min = vdupq_n_f32(valid ? minimal : border_value);
            for (int32_t ii = xStart; ii < xEnd; ++ii) {
                float32x4_t mm_temp = vld1q_f32(gRowMin + ii * inWidthStride + j);
                mm_min              = vminq_f32(mm_temp, mm_min);
            }
            vst1q_f32(outData + i * outWidthStride + j, mm_min);
        }
        for (; j < width * cn; ++j) {
            float _min = valid ? minimal : border_value;
            for (int32_t ii = xStart; ii < xEnd; ++ii) {
                if (gRowMin[ii * inWidthStride + j] > _min) _min = gRowMin[ii * inWidthStride + j];
            }
            outData[i * outWidthStride + j] = _min;
        }
    }

    free(gRowMin);
    return ppl::common::RC_SUCCESS;
}

template <typename T>
::ppl::common::RetCode armminFilter_normal(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t* kernel,
    int32_t outWidthStride,
    T* outData,
    int32_t cn,
    T border_value)
{
    T minimal;
    if (std::is_same<T, float>::value) {
        minimal = FLT_MAX;
    } else if (std::is_same<T, uint8_t>::value) {
        minimal = 255;
    }
    for (int32_t i = 0; i < height; ++i) {
        for (int32_t j = 0; j < width; ++j) {
            for (int32_t c = 0; c < cn; ++c) {
                T _min = minimal;
                for (int32_t ky = 0; ky < kernely_len; ++ky) {
                    int32_t src_y = i + ky - (kernely_len >> 1);
                    bool valid_y  = ((src_y >= 0) && (src_y < height));
                    for (int32_t kx = 0; kx < kernelx_len; ++kx) {
                        int32_t src_x = j + kx - (kernelx_len >> 1);
                        bool valid_x  = ((src_x >= 0) && (src_x < width));
                        if (kernel[ky * kernelx_len + kx]) {
                            T value = (valid_x && valid_y) ? inData[src_y * inWidthStride + src_x * cn + c] : border_value;
                            _min    = std::min(_min, value);
                        }
                    }
                }
                outData[i * outWidthStride + j * cn + c] = _min;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}
#define Erode(dt, nc, name)                                                                                                                                                                                                                                     \
    template <>                                                                                                                                                                                                                                                 \
    ::ppl::common::RetCode Erode<dt, nc>(int32_t height, int32_t width, int32_t inWidthStride, const dt* inData, int32_t kernelx_len, int32_t kernely_len, const uint8_t* kernel, int32_t outWidthStride, dt* outData, BorderType border_type, dt border_value) \
    {                                                                                                                                                                                                                                                           \
        if (inData == nullptr || outData == nullptr || kernel == nullptr) {                                                                                                                                                                                     \
            return ppl::common::RC_INVALID_VALUE;                                                                                                                                                                                                               \
        }                                                                                                                                                                                                                                                       \
        if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride == 0) {                                                                                                                                                                        \
            return ppl::common::RC_INVALID_VALUE;                                                                                                                                                                                                               \
        }                                                                                                                                                                                                                                                       \
        if (border_type != BORDER_TYPE_CONSTANT) {                                                                                                                                                                                                              \
            border_value = std::numeric_limits<dt>::max();                                                                                                                                                                                                      \
        }                                                                                                                                                                                                                                                       \
        bool flag = true, flag3 = false, flag5 = false;                                                                                                                                                                                                         \
        if (kernelx_len == 3 && kernely_len == 3)                                                                                                                                                                                                               \
            flag3 = true;                                                                                                                                                                                                                                       \
        if (kernelx_len == 5 && kernely_len == 5)                                                                                                                                                                                                               \
            flag5 = true;                                                                                                                                                                                                                                       \
        for (int32_t i = 0; i < kernelx_len * kernely_len; ++i) {                                                                                                                                                                                               \
            if (kernel[i] != 1) {                                                                                                                                                                                                                               \
                flag = false;                                                                                                                                                                                                                                   \
                break;                                                                                                                                                                                                                                          \
            }                                                                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                                                       \
        if (flag && flag3)                                                                                                                                                                                                                                      \
            ppl::cv::aarch64::morph_##name<ErodeVecOp, nc, 3>(height, width, inWidthStride, inData, outWidthStride, outData, ppl::cv::BORDER_TYPE_CONSTANT, border_value);                                                                                      \
        else if (flag && flag5)                                                                                                                                                                                                                                 \
            ppl::cv::aarch64::morph_##name<ErodeVecOp, nc, 5>(height, width, inWidthStride, inData, outWidthStride, outData, ppl::cv::BORDER_TYPE_CONSTANT, border_value);                                                                                      \
        else                                                                                                                                                                                                                                                    \
            return armminFilter_normal(height, width, inWidthStride, inData, kernelx_len, kernely_len, kernel, outWidthStride, outData, nc, border_value);                                                                                                      \
        return ppl::common::RC_SUCCESS;                                                                                                                                                                                                                         \
    }

Erode(float, 1, f32)
Erode(float, 3, f32)
Erode(float, 4, f32)

Erode(uint8_t, 1, u8)
Erode(uint8_t, 3, u8)
Erode(uint8_t, 4, u8)

}
}
} // namespace ppl::cv::aarch64
