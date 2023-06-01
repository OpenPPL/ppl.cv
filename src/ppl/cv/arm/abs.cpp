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

#include "ppl/cv/arm/abs.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>

namespace ppl {
namespace cv {
namespace arm {

::ppl::common::RetCode abs_s8(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride,
    const int8_t *inData,
    int32_t outWidthStride,
    int8_t *outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    width *= channels;
    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        const int8_t *in0 = inData + i * inWidthStride;
        int8_t *out = outData + i * outWidthStride;
        for (; j <= width - 32; j += 32) {
            prefetch(in0 + j);
            int8x16_t vdata0 = vld1q_s8(in0 + j);
            int8x16_t vdata1 = vld1q_s8(in0 + j + 16);
            int8x16_t voutData0 = vqabsq_s8(vdata0);
            int8x16_t voutData1 = vqabsq_s8(vdata1);
            vst1q_s8(out + j, voutData0);
            vst1q_s8(out + j + 16, voutData1);
        }
        for (; j <= width - 16; j += 16) {
            int8x16_t vdata0 = vld1q_s8(in0 + j);
            int8x16_t voutData = vqabsq_s8(vdata0);
            vst1q_s8(out + j, voutData);
        }
        for (; j < width; ++j) {
            int8_t srcVal = inData[i * inWidthStride + j];
            outData[i * outWidthStride + j] = (srcVal == -128) ? 127 : std::abs(srcVal);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template<>
::ppl::common::RetCode Abs<int8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const int8_t *inData,
    int32_t outWidthStride,
    int8_t *outData)
{
    return abs_s8(height, width, 1, inWidthStride, inData, outWidthStride, outData);
}

template<>
::ppl::common::RetCode Abs<int8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const int8_t *inData,
    int32_t outWidthStride,
    int8_t *outData)
{
    return abs_s8(height, width, 3, inWidthStride, inData, outWidthStride, outData);
}

template<>
::ppl::common::RetCode Abs<int8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const int8_t *inData,
    int32_t outWidthStride,
    int8_t *outData)
{
    return abs_s8(height, width, 4, inWidthStride, inData, outWidthStride, outData);
}

::ppl::common::RetCode abs_f32(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride,
    const float32_t *inData,
    int32_t outWidthStride,
    float32_t *outData)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    width *= channels;
    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        const float *in0 = inData + i * inWidthStride;
        float *out = outData + i * outWidthStride;
        for (; j <= width - 8; j += 8) {
            prefetch(in0 + j);
            float32x4_t vdata0 = vld1q_f32(in0 + j);
            float32x4_t vdata1 = vld1q_f32(in0 + j + 4);
            float32x4_t voutData0 = vabsq_f32(vdata0);
            float32x4_t voutData1 = vabsq_f32(vdata1);
            vst1q_f32(out + j, voutData0);
            vst1q_f32(out + j + 4, voutData1);
        }
        for (; j <= width - 4; j += 4) {
            float32x4_t vdata0 = vld1q_f32(in0 + j);
            float32x4_t voutData = vabsq_f32(vdata0);
            vst1q_f32(out + j, voutData);
        }
        for (; j < width; ++j) {
            outData[i * outWidthStride + j] = std::abs(inData[i * inWidthStride + j]);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template<>
::ppl::common::RetCode Abs<float32_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float32_t *inData,
    int32_t outWidthStride,
    float32_t *outData)
{
    return abs_f32(height, width, 1, inWidthStride, inData, outWidthStride, outData);
}

template<>
::ppl::common::RetCode Abs<float32_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float32_t *inData,
    int32_t outWidthStride,
    float32_t *outData)
{
    return abs_f32(height, width, 3, inWidthStride, inData, outWidthStride, outData);
}

template<>
::ppl::common::RetCode Abs<float32_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float32_t *inData,
    int32_t outWidthStride,
    float32_t *outData)
{
    return abs_f32(height, width, 4, inWidthStride, inData, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm
