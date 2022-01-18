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

#include "ppl/cv/arm/arithmetic.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>

namespace ppl {
namespace cv {
namespace arm {

::ppl::common::RetCode add_f32(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData0 && nullptr == inData1 && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride0 == 0 || inWidthStride1 == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    width *= channels;
    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        const float *in0 = inData0 + i * inWidthStride0;
        const float *in1 = inData1 + i * inWidthStride1;
        float *out = outData + i * outWidthStride;
        for (; j <= width - 8; j += 8) {
            prefetch(in0 + j);
            prefetch(in1 + j);
            float32x4_t vdata0 = vld1q_f32(in0 + j);
            float32x4_t vdata2 = vld1q_f32(in0 + j + 4);
            float32x4_t vdata1 = vld1q_f32(in1 + j);
            float32x4_t vdata3 = vld1q_f32(in1 + j + 4);
            float32x4_t voutData0 = vaddq_f32(vdata0, vdata1);
            float32x4_t voutData1 = vaddq_f32(vdata2, vdata3);
            vst1q_f32(out + j, voutData0);
            vst1q_f32(out + j + 4, voutData1);
        }
        for (; j <= width - 4; j += 4) {
            float32x4_t vdata0 = vld1q_f32(inData0 + i * inWidthStride0 + j);
            float32x4_t vdata1 = vld1q_f32(inData1 + i * inWidthStride1 + j);
            float32x4_t voutData = vaddq_f32(vdata0, vdata1);
            vst1q_f32(outData + i * outWidthStride + j, voutData);
        }
        for (; j < width; ++j) {
            outData[i * outWidthStride + j] =
                inData0[i * inWidthStride0 + j] + inData1[i * inWidthStride1 + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode add_u8(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData0 && nullptr == inData1 && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride0 == 0 || inWidthStride1 == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    width *= channels;
    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        const uint8_t *in0 = inData0 + i * inWidthStride0;
        const uint8_t *in1 = inData1 + i * inWidthStride1;
        uint8_t *out = outData + i * outWidthStride;
        for (; j <= width - 32; j += 32) {
            prefetch(in0 + j);
            prefetch(in1 + j);
            uint8x16_t vdata0 = vld1q_u8(in0 + j);
            uint8x16_t vdata2 = vld1q_u8(in0 + j + 16);
            uint8x16_t vdata1 = vld1q_u8(in1 + j);
            uint8x16_t vdata3 = vld1q_u8(in1 + j + 16);
            uint8x16_t voutData0 = vqaddq_u8(vdata0, vdata1);
            uint8x16_t voutData1 = vqaddq_u8(vdata2, vdata3);
            vst1q_u8(out + j, voutData0);
            vst1q_u8(out + j + 16, voutData1);
        }
        for (; j <= width - 16; j += 16) {
            uint8x16_t vdata0 = vld1q_u8(inData0 + i * inWidthStride0 + j);
            uint8x16_t vdata1 = vld1q_u8(inData1 + i * inWidthStride1 + j);
            uint8x16_t voutData = vqaddq_u8(vdata0, vdata1);
            vst1q_u8(outData + i * outWidthStride + j, voutData);
        }
        for (; j < width; ++j) {
            outData[i * outWidthStride + j] =
                inData0[i * inWidthStride0 + j] + inData1[i * inWidthStride1 + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Add<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return add_f32(height, width, 1, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return add_f32(height, width, 3, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return add_f32(height, width, 4, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return add_u8(height, width, 1, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return add_u8(height, width, 3, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return add_u8(height, width, 4, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm
