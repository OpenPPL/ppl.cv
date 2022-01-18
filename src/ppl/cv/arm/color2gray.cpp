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

#include "ppl/cv/arm/cvtcolor.h"
#include "ppl/cv/arm/typetraits.hpp"
#include "ppl/cv/types.h"
#include <arm_neon.h>
#include <float.h>

namespace ppl {
namespace cv {
namespace arm {

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
    const uint8_t* srcPtr = src;
    uint8_t* dstPtr = dst;

    typedef typename DT<ncSrc, uint8_t>::vec_DT srcType;
    typedef typename DT<ncDst, uint8_t>::vec_DT dstType;

    uint8_t k_r = 77; //0.299;
    uint8_t k_g = 150; //0.587;
    uint8_t k_b = 29; //0.114;
    if (!isBGR) {
        uint8_t temp = k_r;
        k_r = k_b;
        k_b = temp;
    }
    uint8_t SHIFT_LEFT = 1 << 7;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        uint8x8_t v_kr = vdup_n_u8(k_r);
        uint8x8_t v_kg = vdup_n_u8(k_g);
        uint8x8_t v_kb = vdup_n_u8(k_b);
        uint8x8_t v_SHIFT_LEFT = vdup_n_u8(SHIFT_LEFT);

        int32_t i;
        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src0 = vldx_u8_f32<ncSrc, uint8_t, srcType>(srcPtr + ncSrc * (i));

            uint16x8_t v_tmp = vmull_u8(v_src0.val[0], v_kb);
            v_tmp = vmlal_u8(v_tmp, v_src0.val[1], v_kg);
            v_tmp = vmlal_u8(v_tmp, v_src0.val[2], v_kr);
            v_tmp = vaddw_u8(v_tmp, v_SHIFT_LEFT);

            dstType v_dst0 = vshrn_n_u16(v_tmp, 8);

            vstx_u8_f32<ncDst, uint8_t, dstType>(dstPtr + ncDst * i, v_dst0);
        }

        for (; i < width; i++) {
            int32_t b = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], r = srcPtr[ncSrc * i + 2];

            int32_t gray = (k_r * r + k_b * b + k_g * g + SHIFT_LEFT) >> 8;
            gray = gray > 255 ? 255 : gray;

            dstPtr[ncDst * i] = (uint8_t)gray;
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
    const float* srcPtr = src;
    float* dstPtr = dst;

    typedef typename DT<ncSrc, float>::vec_DT srcType;
    typedef typename DT<ncDst, float>::vec_DT dstType;

    float k_r = 0.299;
    float k_g = 0.587;
    float k_b = 0.114;
    if (!isBGR) {
        float temp = k_r;
        k_r = k_b;
        k_b = temp;
    }
    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        int32_t i = 0;

        float32x4_t v_kr = vdupq_n_f32(k_r);
        float32x4_t v_kg = vdupq_n_f32(k_g);
        float32x4_t v_kb = vdupq_n_f32(k_b);

        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src0 = vldx_u8_f32<ncSrc, float, srcType>(srcPtr + ncSrc * i);
            srcType v_src1 = vldx_u8_f32<ncSrc, float, srcType>(srcPtr + ncSrc * (i + 4));

            dstType v_dst0 = vmulq_f32(v_src0.val[0], v_kb);
            v_dst0 = vmlaq_f32(v_dst0, v_src0.val[1], v_kg);
            v_dst0 = vmlaq_f32(v_dst0, v_src0.val[2], v_kr);

            dstType v_dst1 = vmulq_f32(v_src1.val[0], v_kb);
            v_dst1 = vmlaq_f32(v_dst1, v_src1.val[1], v_kg);
            v_dst1 = vmlaq_f32(v_dst1, v_src1.val[2], v_kr);

            vstx_u8_f32<ncDst, float, dstType>(dstPtr + ncDst * i, v_dst0);
            vstx_u8_f32<ncDst, float, dstType>(dstPtr + ncDst * (i + 4), v_dst1);
        }

        for (; i < width; i++) {
            float b = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], r = srcPtr[ncSrc * i + 2];
            dstPtr[ncDst * i] = (k_r * r + k_b * b + k_g * g);
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
    const uint8_t* srcPtr = src;
    uint8_t* dstPtr = dst;

    typedef typename DT<ncSrc, uint8_t>::vec_DT srcType;
    typedef typename DT<ncDst, uint8_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;

    dstType v_dst0;
    if (ncDst == 4) { v_dst0.val[3] = vdup_n_u8(255); }

    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        int32_t i;
        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src0 = vldx_u8_f32<ncSrc, uint8_t, srcType>(srcPtr + ncSrc * (i));

            v_dst0.val[0] = v_src0;
            v_dst0.val[1] = v_src0;
            v_dst0.val[2] = v_src0;

            vstx_u8_f32<ncDst, uint8_t, dstType>(dstPtr + ncDst * i, v_dst0);
        }

        for (; i < width; i++) {
            uint8_t gray = srcPtr[i];

            dstPtr[ncDst * i] = gray;
            dstPtr[ncDst * i + 1] = gray;
            dstPtr[ncDst * i + 2] = gray;
            if (ncDst == 4)
                dstPtr[4 * i + 3] = (uint8_t)255;
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
    const float* srcPtr = src;
    float* dstPtr = dst;

    typedef typename DT<ncSrc, float>::vec_DT srcType;
    typedef typename DT<ncDst, float>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;

    dstType v_dst0, v_dst1;
    if (ncDst == 4) {
        v_dst0.val[3] = vdupq_n_f32(1.0);
        v_dst1.val[3] = vdupq_n_f32(1.0);
    }

    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        int32_t i;
        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src0 = vldx_u8_f32<ncSrc, float, srcType>(srcPtr + ncSrc * (i));
            srcType v_src1 = vldx_u8_f32<ncSrc, float, srcType>(srcPtr + ncSrc * (i + 4));
            v_dst0.val[0] = v_src0;
            v_dst0.val[1] = v_src0;
            v_dst0.val[2] = v_src0;
            v_dst1.val[0] = v_src1;
            v_dst1.val[1] = v_src1;
            v_dst1.val[2] = v_src1;

            vstx_u8_f32<ncDst, float, dstType>(dstPtr + ncDst * i, v_dst0);
            vstx_u8_f32<ncDst, float, dstType>(dstPtr + ncDst * (i + 4), v_dst1);
        }

        for (; i < width; i++) {
            float gray = srcPtr[i];

            dstPtr[ncDst * i] = gray;
            dstPtr[ncDst * i + 1] = gray;
            dstPtr[ncDst * i + 2] = gray;
            if (ncDst == 4)
                dstPtr[4 * i + 3] = (float)1.0;
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
} // namespace ppl::cv::arm
