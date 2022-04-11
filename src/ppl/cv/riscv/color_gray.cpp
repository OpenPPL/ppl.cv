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

#include "ppl/cv/riscv/cvtcolor.h"
#include "ppl/cv/riscv/typetraits.h"
#include "ppl/cv/types.h"
#include <float.h>

namespace ppl {
namespace cv {
namespace riscv {

template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode cvt_color_bgr2gray_uint8_t(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst,
    bool isBGR)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const size_t vl = vsetvlmax_e8m4();
    const int32_t num_unroll = vl;

    uint8_t k_r = 77; //0.299;
    uint8_t k_g = 150; //0.587;
    uint8_t k_b = 29; //0.114;
    if (!isBGR) {
        uint8_t temp = k_r;
        k_r = k_b;
        k_b = temp;
    }
    const uint8_t SHIFT_LEFT = 1 << 7;
    vuint16m8_t shift_v = vmv_v_x_u16m8(SHIFT_LEFT, vl);

    auto in = src;
    auto out = dst;
    for (int32_t k = 0; k < height; k++, in += srcStride, out += dstStride) {
        int32_t i = 0;
        for (; i < width; i += num_unroll) {
            const size_t vl = vsetvl_e8m4(width - i);
            vuint8m4_t in_b_v = vlse8_v_u8m4(in + ncSrc * i + 0, ncSrc * sizeof(uint8_t), vl);
            vuint8m4_t in_g_v = vlse8_v_u8m4(in + ncSrc * i + 1, ncSrc * sizeof(uint8_t), vl);
            vuint8m4_t in_r_v = vlse8_v_u8m4(in + ncSrc * i + 2, ncSrc * sizeof(uint8_t), vl);

            vuint16m8_t out_v = vsaddu_vv_u16m8(
                vwmaccu_vx_u16m8(shift_v, k_b, in_b_v, vl),
                vwmaccu_vx_u16m8(vwmulu_vx_u16m8(in_g_v, k_g, vl), k_r, in_r_v, vl),
                vl);
            vse8_v_u8m4(out + ncDst * i, vnclipu_wx_u8m4(out_v, 8, vl), vl);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode cvt_color_bgr2gray_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float* src,
    const int32_t dstStride,
    float* dst,
    bool isBGR)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const size_t vl = vsetvlmax_e32m4();
    const int32_t num_unroll = vl;

    float k_r = 0.299;
    float k_g = 0.587;
    float k_b = 0.114;
    if (!isBGR) {
        float temp = k_r;
        k_r = k_b;
        k_b = temp;
    }

    auto in = src;
    auto out = dst;
    for (int32_t k = 0; k < height; k++, in += srcStride, out += dstStride) {
        int32_t i = 0;
        for (; i < width; i += num_unroll) {
            const size_t vl = vsetvl_e32m4(width - i);
            vfloat32m4_t in_b_v = vlse32_v_f32m4(in + ncSrc * i + 0, ncSrc * sizeof(float), vl);
            vfloat32m4_t in_g_v = vlse32_v_f32m4(in + ncSrc * i + 1, ncSrc * sizeof(float), vl);
            vfloat32m4_t in_r_v = vlse32_v_f32m4(in + ncSrc * i + 2, ncSrc * sizeof(float), vl);

            vfloat32m4_t out_v = vfmacc_vf_f32m4(
                vfmacc_vf_f32m4(
                    vfmul_vf_f32m4(in_b_v, k_b, vl), k_g, in_g_v, vl),
                k_r,
                in_r_v,
                vl);
            vse32_v_f32m4(out + ncDst * i, out_v, vl);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode cvt_color_gray2bgr_uint8_t(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const int32_t num_unroll = 8;

    for (int32_t k = 0; k < height; k++, src += srcStride, dst += dstStride) {
        auto in = src;
        auto out = dst;
        int32_t i = 0;
        for (; i <= width - num_unroll; i += num_unroll, in += ncSrc * num_unroll, out += ncDst * num_unroll) {
            out[ncDst * 0 + 0] = in[0];
            out[ncDst * 0 + 1] = in[0];
            out[ncDst * 0 + 2] = in[0];
            if (ncDst == 4) out[ncDst * 0 + 3] = 255;
            out[ncDst * 1 + 0] = in[1];
            out[ncDst * 1 + 1] = in[1];
            out[ncDst * 1 + 2] = in[1];
            if (ncDst == 4) out[ncDst * 1 + 3] = 255;
            out[ncDst * 2 + 0] = in[2];
            out[ncDst * 2 + 1] = in[2];
            out[ncDst * 2 + 2] = in[2];
            if (ncDst == 4) out[ncDst * 2 + 3] = 255;
            out[ncDst * 3 + 0] = in[3];
            out[ncDst * 3 + 1] = in[3];
            out[ncDst * 3 + 2] = in[3];
            if (ncDst == 4) out[ncDst * 3 + 3] = 255;
            out[ncDst * 4 + 0] = in[4];
            out[ncDst * 4 + 1] = in[4];
            out[ncDst * 4 + 2] = in[4];
            if (ncDst == 4) out[ncDst * 4 + 3] = 255;
            out[ncDst * 5 + 0] = in[5];
            out[ncDst * 5 + 1] = in[5];
            out[ncDst * 5 + 2] = in[5];
            if (ncDst == 4) out[ncDst * 5 + 3] = 255;
            out[ncDst * 6 + 0] = in[6];
            out[ncDst * 6 + 1] = in[6];
            out[ncDst * 6 + 2] = in[6];
            if (ncDst == 4) out[ncDst * 6 + 3] = 255;
            out[ncDst * 7 + 0] = in[7];
            out[ncDst * 7 + 1] = in[7];
            out[ncDst * 7 + 2] = in[7];
            if (ncDst == 4) out[ncDst * 7 + 3] = 255;
        }
        for (; i < width; ++i, in += ncSrc, out += ncDst) {
            out[0] = in[0];
            out[1] = in[0];
            out[2] = in[0];
            if (ncDst == 4) out[3] = 255;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode cvt_color_gray2bgr_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float* src,
    const int32_t dstStride,
    float* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int32_t num_unroll = 2;
    const size_t vl = vsetvlmax_e32m4();
    vfloat32m4_t out_a_v = vfmv_v_f_f32m4(1.0, vl);

    auto in = src;
    auto out = dst;
    for (int32_t k = 0; k < height; k++, in += srcStride, out += dstStride) {
        {
            int32_t i = 0;
            for (; i <= width - num_unroll; i += num_unroll) {
                out[i * ncDst + ncDst * 0 + 0] = in[i * ncSrc + 0];
                out[i * ncDst + ncDst * 0 + 1] = in[i * ncSrc + 0];
                out[i * ncDst + ncDst * 0 + 2] = in[i * ncSrc + 0];
                out[i * ncDst + ncDst * 1 + 0] = in[i * ncSrc + 1];
                out[i * ncDst + ncDst * 1 + 1] = in[i * ncSrc + 1];
                out[i * ncDst + ncDst * 1 + 2] = in[i * ncSrc + 1];
            }
            for (; i < width; ++i) {
                float in0 = in[i * ncSrc];
                out[i * ncDst + 0] = in0;
                out[i * ncDst + 1] = in0;
                out[i * ncDst + 2] = in0;
            }
        }
        if (ncDst == 4) {
            int32_t i = 0;
            for (; i < width; i += vl) {
                const size_t avl = vsetvl_e32m4(width - i);
                vsse32_v_f32m4(out + i * ncDst + 3, ncDst * sizeof(float), out_a_v, avl);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_bgr2gray_uint8_t<3, 1>(height, width, inWidthStride, inData, outWidthStride, outData, true);
}
template <>
::ppl::common::RetCode BGRA2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_bgr2gray_uint8_t<4, 1>(height, width, inWidthStride, inData, outWidthStride, outData, true);
}
template <>
::ppl::common::RetCode BGR2GRAY<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_bgr2gray_f32<3, 1>(height, width, inWidthStride, inData, outWidthStride, outData, true);
}
template <>
::ppl::common::RetCode BGRA2GRAY<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_bgr2gray_f32<4, 1>(height, width, inWidthStride, inData, outWidthStride, outData, true);
}
template <>
::ppl::common::RetCode GRAY2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_gray2bgr_uint8_t<1, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode GRAY2BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_gray2bgr_uint8_t<1, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode GRAY2BGR<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_gray2bgr_f32<1, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode GRAY2BGRA<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_gray2bgr_f32<1, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode RGB2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_bgr2gray_uint8_t<3, 1>(height, width, inWidthStride, inData, outWidthStride, outData, false);
}
template <>
::ppl::common::RetCode RGBA2GRAY<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_bgr2gray_uint8_t<4, 1>(height, width, inWidthStride, inData, outWidthStride, outData, false);
}
template <>
::ppl::common::RetCode RGB2GRAY<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_bgr2gray_f32<3, 1>(height, width, inWidthStride, inData, outWidthStride, outData, false);
}
template <>
::ppl::common::RetCode RGBA2GRAY<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_bgr2gray_f32<4, 1>(height, width, inWidthStride, inData, outWidthStride, outData, false);
}
template <>
::ppl::common::RetCode GRAY2RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_gray2bgr_uint8_t<1, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode GRAY2RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_gray2bgr_uint8_t<1, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}

template <>
::ppl::common::RetCode GRAY2RGB<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_gray2bgr_f32<1, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode GRAY2RGBA<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_gray2bgr_f32<1, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::riscv
