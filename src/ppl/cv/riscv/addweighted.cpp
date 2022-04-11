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

#include "ppl/cv/riscv/addweighted.h"
#include "ppl/cv/riscv/util.h"
#include "ppl/cv/riscv/typetraits.h"
#include "ppl/cv/types.h"
#include <algorithm>
#include <cmath>
namespace ppl {
namespace cv {
namespace riscv {

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
    constexpr int32_t RVV_LMUL = 4;
    int32_t vl = vsetvlmax_exmy<float, RVV_LMUL>();
    width *= channels;

    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        const float *in0 = inData0 + i * inWidthStride0;
        const float *in1 = inData1 + i * inWidthStride1;
        float *out = outData + i * outWidthStride;
        for (; j < width; j += vl) {
            const int32_t vl = vsetvl_exmy<float, RVV_LMUL>(width - j);
            vfloat32m4_t in0_v = vle32_v_f32m4(in0 + j, vl);
            vfloat32m4_t in1_v = vle32_v_f32m4(in1 + j, vl);
            vfloat32m4_t out_v = vfmv_v_f_f32m4(gamma, vl);
            out_v = vfmacc_vf_f32m4(out_v, alpha, in0_v, vl);
            out_v = vfmacc_vf_f32m4(out_v, beta, in1_v, vl);
            vse32_v_f32m4(out + j, out_v, vl);
        }
        // for (; j < width; ++j) {
        //     outData[i * outWidthStride + j] =
        //         inData0[i * inWidthStride0 + j] * alpha +
        //         inData1[i * inWidthStride1 + j] * beta +
        //         gamma;
        // }
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
    constexpr int32_t RVV_LMUL = 1;
    int32_t vl = vsetvlmax_exmy<uint8_t, RVV_LMUL>();
    width *= channels;

    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        const uint8_t *in0 = inData0 + i * inWidthStride0;
        const uint8_t *in1 = inData1 + i * inWidthStride1;
        uint8_t *out = outData + i * outWidthStride;
        for (; j <= width - vl; j += vl) {
            vuint8m1_t in0_v = vle8_v_u8m1(in0 + j, vl);
            vuint8m1_t in1_v = vle8_v_u8m1(in1 + j, vl);
            vfloat32m4_t in0_f32_v = vfwcvt_f_xu_v_f32m4(vwcvtu_x_x_v_u16m2(in0_v, vl), vl);
            vfloat32m4_t in1_f32_v = vfwcvt_f_xu_v_f32m4(vwcvtu_x_x_v_u16m2(in1_v, vl), vl);
            vfloat32m4_t out_f32_v = vfmv_v_f_f32m4(gamma, vl);
            out_f32_v = vfmacc_vf_f32m4(out_f32_v, alpha, in0_f32_v, vl);
            out_f32_v = vfmacc_vf_f32m4(out_f32_v, beta, in1_f32_v, vl);
            vuint8m1_t out_v = vnclipu_wx_u8m1(vfncvt_xu_f_w_u16m2(out_f32_v, vl), 0, vl);
            vse8_v_u8m1(out + j, out_v, vl);
        }
        for (; j < width; ++j) {
            out[j] = sat_cast_u8(in0[j] * alpha + in1[j] * beta + gamma);
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
} // namespace ppl::cv::riscv
