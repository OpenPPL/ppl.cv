// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for divitional information
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

inline float32x4_t vroundq(const float32x4_t &v)
{
    const int32x4_t signMask = vdupq_n_s32(1 << 31), half = vreinterpretq_s32_f32(vdupq_n_f32(0.5f));
    float32x4_t v_addition = vreinterpretq_f32_s32(vorrq_s32(half, vandq_s32(signMask, vreinterpretq_s32_f32(v))));
    return vaddq_f32(v, v_addition);
}

inline uint32x4_t divSaturate(uint32x4_t v0, uint32x4_t v1)
{
    return vcvtq_u32_f32(vroundq(vmulq_f32(vcvtq_f32_u32(v0), vrecpeq_f32(vcvtq_f32_u32(v1)))));
}

inline uint16x8_t divSaturate(uint16x8_t v0, uint16x8_t v1)
{
    return vcombine_u16(vqmovn_u32(divSaturate(vmovl_u16(vget_low_u16(v0)),
                                               vmovl_u16(vget_low_u16(v1)))),
                        vqmovn_u32(divSaturate(vmovl_u16(vget_high_u16(v0)),
                                               vmovl_u16(vget_high_u16(v1)))));
}

inline uint8x16_t divSaturate(uint8x16_t v0, uint8x16_t v1)
{
    return vcombine_u8(vqmovn_u16(divSaturate(vmovl_u8(vget_low_u8(v0)),
                                              vmovl_u8(vget_low_u8(v1)))),
                       vqmovn_u16(divSaturate(vmovl_u8(vget_high_u8(v0)),
                                              vmovl_u8(vget_high_u8(v1)))));
}

::ppl::common::RetCode div_f32(
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
            float32x4_t voutData0 = vdivq_f32(vdata0, vdata1);
            float32x4_t voutData1 = vdivq_f32(vdata2, vdata3);
            vst1q_f32(out + j, voutData0);
            vst1q_f32(out + j + 4, voutData1);
        }
        for (; j <= width - 4; j += 4) {
            float32x4_t vdata0 = vld1q_f32(inData0 + i * inWidthStride0 + j);
            float32x4_t vdata1 = vld1q_f32(inData1 + i * inWidthStride1 + j);
            float32x4_t voutData = vdivq_f32(vdata0, vdata1);
            vst1q_f32(outData + i * outWidthStride + j, voutData);
        }
        for (; j < width; ++j) {
            outData[i * outWidthStride + j] =
                inData0[i * inWidthStride0 + j] / inData1[i * inWidthStride1 + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode div_u8(
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
        for (; j <= width - 16; j += 16) {
            prefetch(in0 + j);
            prefetch(in1 + j);
            uint8x16_t vdata0 = vld1q_u8(in0 + j);
            uint8x16_t vdata1 = vld1q_u8(in1 + j);
            uint8x16_t vMask = vtstq_u8(vdata1, vdata1);
            uint8x16_t voutData0 = divSaturate(vdata0, vdata1);
            vst1q_u8(out + j, vandq_u8(vMask, voutData0));
        }
        for (; j < width; ++j) {
            outData[i * outWidthStride + j] =
                inData0[i * inWidthStride0 + j] / inData1[i * inWidthStride1 + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode Div<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return div_f32(height, width, 1, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Div<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return div_f32(height, width, 3, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Div<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return div_f32(height, width, 4, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Div<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return div_u8(height, width, 1, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Div<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return div_u8(height, width, 3, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Div<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return div_u8(height, width, 4, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm
