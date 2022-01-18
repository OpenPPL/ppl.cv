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

#include "ppl/cv/arm/addweighted.h"
#include "operation_utils.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
namespace ppl {
namespace cv {
namespace arm {

uint8_t truncate(float v)
{
    return (std::min(std::max(std::round(v), 0.0f), 255.0f));
}

::ppl::common::RetCode addWeighted_f32(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride0,
    const float *inData0,
    float alpha,
    int32_t inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
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
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta = vdupq_n_f32(beta);
    float32x4_t vgamma = vdupq_n_f32(gamma);

    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        for (; j <= width - 4; j += 4) {
            float32x4_t vdata0 = vld1q_f32(inData0 + i * inWidthStride0 + j);
            float32x4_t vdata1 = vld1q_f32(inData1 + i * inWidthStride1 + j);
            float32x4_t vdst = vgamma;
            vdst = vmlaq_f32(vdst, vdata0, valpha);
            vdst = vmlaq_f32(vdst, vdata1, vbeta);
            vst1q_f32(outData + i * outWidthStride + j, vdst);
        }
        for (; j < width; ++j) {
            outData[i * outWidthStride + j] =
                inData0[i * inWidthStride0 + j] * alpha +
                inData1[i * inWidthStride1 + j] * beta +
                gamma;
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode addWeighted_u8(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
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
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta = vdupq_n_f32(beta);
    float32x4_t vgamma = vdupq_n_f32(gamma);

    for (int32_t i = 0; i < height; ++i) {
        const uint8_t *row_indata0 = inData0 + i * inWidthStride0;
        const uint8_t *row_indata1 = inData1 + i * inWidthStride0;
        int32_t j = 0;
        for (; j <= width - 8; j += 8) {
            prefetch(row_indata0 + j);
            prefetch(row_indata1 + j);
            uint16x8_t vdata0 = vmovl_u8(vld1_u8(row_indata0 + j));
            float32x4_t v_dst0_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vdata0)));
            float32x4_t v_dst0_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vdata0)));
            v_dst0_0 = vmlaq_f32(vgamma, v_dst0_0, valpha);
            v_dst0_1 = vmlaq_f32(vgamma, v_dst0_1, valpha);
            uint16x8_t vdata1 = vmovl_u8(vld1_u8(row_indata1 + j));
            float32x4_t v_dst1_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vdata1)));
            float32x4_t v_dst1_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vdata1)));
            v_dst1_0 = vmlaq_f32(v_dst0_0, v_dst1_0, vbeta);
            v_dst1_1 = vmlaq_f32(v_dst0_1, v_dst1_1, vbeta);
            int16x8_t v_dst_tmp = vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_dst1_0)), vmovn_s32(vcvtq_s32_f32(v_dst1_1)));
            uint8x8_t v_dst = vqmovun_s16(v_dst_tmp);
            vst1_u8(outData + i * outWidthStride + j, v_dst);
        }
        for (; j < width; ++j) {
            outData[i * outWidthStride + j] = truncate(
                row_indata0[i * inWidthStride0 + j] * alpha +
                row_indata1[i * inWidthStride1 + j] * beta +
                gamma);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode AddWeighted<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    float alpha,
    int32_t inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    float *outData)
{
    return addWeighted_f32(height, width, 1, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    float alpha,
    int32_t inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    float *outData)
{
    return addWeighted_f32(height, width, 3, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    float alpha,
    int32_t inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    float *outData)
{
    return addWeighted_f32(height, width, 4, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return addWeighted_u8(height, width, 1, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return addWeighted_u8(height, width, 3, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return addWeighted_u8(height, width, 4, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm
