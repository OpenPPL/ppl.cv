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
#include <algorithm>
#include <complex>
#include <string.h>
#include <arm_neon.h>

namespace ppl {
namespace cv {
namespace arm {

::ppl::common::RetCode cvt_color_bgra2bgr_uint8_t(
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

    const int32_t ncSrc = 4;
    const int32_t ncDst = 3;

    const uint8_t* srcPtr = src;
    uint8_t* dstPtr       = dst;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        int32_t i = 0;
        for (; i <= width - 8; i += 8) {
            asm volatile(
                "ld4 {v0.8b, v1.8b, v2.8b, v3.8b}, [%0]\n\t"
                "st3 {v0.8b, v1.8b, v2.8b}, [%1]\n\t"
                :
                : "r"(srcPtr + ncSrc * i), "r"(dstPtr + ncDst * i)
                : "cc", "memory", "v0", "v1", "v2", "v3");
        }
        for (; i < width; i++) {
            uint8_t b = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], r = srcPtr[ncSrc * i + 2];

            dstPtr[ncDst * i]     = b;
            dstPtr[ncDst * i + 1] = g;
            dstPtr[ncDst * i + 2] = r;
            if (ncDst == 4) {
                dstPtr[4 * i + 3] = 255;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode cvt_color_bgr2bgr_uint8_t(
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
    uint8_t* dstPtr       = dst;

    typedef typename DT<3, uint8_t>::vec_DT srcType;
    typedef typename DT<4, uint8_t>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst0;
        v_dst0.val[3] = vdup_n_u8(255);

        int32_t i;
        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src0 = vldx_u8_f32<3, uint8_t, srcType>(srcPtr + 3 * i);

            v_dst0.val[0] = v_src0.val[0];
            v_dst0.val[1] = v_src0.val[1];
            v_dst0.val[2] = v_src0.val[2];

            vstx_u8_f32<4, uint8_t, dstType>(dstPtr + 4 * i, v_dst0);
        }
        for (; i < width; i++) {
            uint8_t b = srcPtr[3 * i], g = srcPtr[3 * i + 1], r = srcPtr[3 * i + 2];

            dstPtr[4 * i]     = b;
            dstPtr[4 * i + 1] = g;
            dstPtr[4 * i + 2] = r;
            dstPtr[4 * i + 3] = 255;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t ncSrc, int32_t ncDst>
::ppl::common::RetCode cvt_color_bgr2bgr_f32(
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
    float* dstPtr       = dst;

    typedef typename DT<ncSrc, float>::vec_DT srcType;
    typedef typename DT<ncDst, float>::vec_DT dstType;

    const int32_t src_step = srcStride;
    const int32_t dst_step = dstStride;
    for (int32_t k = 0; k < height; k++, srcPtr += src_step, dstPtr += dst_step) {
        dstType v_dst0, v_dst1;
        if (ncDst == 4) {
            v_dst0.val[3] = vdupq_n_f32(1.0);
            v_dst1.val[3] = vdupq_n_f32(1.0);
        }

        int32_t i = 0;

        for (i = 0; i <= width - 8; i += 8) {
            srcType v_src0 = vldx_u8_f32<ncSrc, float, srcType>(srcPtr + ncSrc * i);
            srcType v_src1 = vldx_u8_f32<ncSrc, float, srcType>(srcPtr + ncSrc * (i + 4));

            v_dst0.val[0] = v_src0.val[0];
            v_dst0.val[1] = v_src0.val[1];
            v_dst0.val[2] = v_src0.val[2];
            v_dst1.val[0] = v_src1.val[0];
            v_dst1.val[1] = v_src1.val[1];
            v_dst1.val[2] = v_src1.val[2];

            vstx_u8_f32<ncDst, float, dstType>(dstPtr + ncDst * i, v_dst0);
            vstx_u8_f32<ncDst, float, dstType>(dstPtr + ncDst * (i + 4), v_dst1);
        }

        for (; i < width; i++) {
            float b = srcPtr[ncSrc * i], g = srcPtr[ncSrc * i + 1], r = srcPtr[ncSrc * i + 2];

            dstPtr[ncDst * i]     = b;
            dstPtr[ncDst * i + 1] = g;
            dstPtr[ncDst * i + 2] = r;
            if (ncDst == 4) {
                dstPtr[4 * i + 3] = 1.0;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_bgr2bgr_uint8_t(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGR2BGRA<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_bgr2bgr_f32<3, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGRA2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_bgra2bgr_uint8_t(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGRA2BGR<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_bgr2bgr_f32<4, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::arm
