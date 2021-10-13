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

#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/cv/x86/util.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>

#include <limits.h>
#include <immintrin.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

#define CY_coeff  1220542
#define CUB_coeff 2116026
#define CUG_coeff -409993
#define CVG_coeff -852492
#define CVR_coeff 1673527
#define SHIFT     20

// Coefficients for RGB to YUV420p conversion
#define CRY_coeff 269484
#define CGY_coeff 528482
#define CBY_coeff 102760
#define CRU_coeff -155188
#define CGU_coeff -305135
#define CBU_coeff 460324
#define CGV_coeff -385875
#define CBV_coeff -74448

#define DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))
template <int32_t srccn, int32_t bIdx, bool isUV>
void rgb_2_nv(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    for (int32_t i = 0; i < height; i += 2) {
        const uint8_t *src0 = inData + i * inWidthStride;
        const uint8_t *src1 = inData + (i + 1) * inWidthStride;
        uint8_t *dst0       = outY + i * outYStride;
        uint8_t *dst1       = outY + (i + 1) * outYStride;
        uint8_t *dst2       = outUV + (i / 2) * outUVStride;
        for (int32_t j = 0; j < width / 2; ++j, src0 += 2 * srccn, src1 += 2 * srccn) {
            int32_t r00 = src0[2 - bIdx];
            int32_t g00 = src0[1];
            int32_t b00 = src0[bIdx];
            int32_t r01 = src0[2 - bIdx + srccn];
            int32_t g01 = src0[1 + srccn];
            int32_t b01 = src0[bIdx + srccn];
            int32_t r10 = src1[2 - bIdx];
            int32_t g10 = src1[1];
            int32_t b10 = src1[bIdx];
            int32_t r11 = src1[2 - bIdx + srccn];
            int32_t g11 = src1[1 + srccn];
            int32_t b11 = src1[bIdx + srccn];

            const int32_t shifted16 = (16 << SHIFT);
            const int32_t halfShift = (1 << (SHIFT - 1));

            int32_t y00 = CRY_coeff * r00 + CGY_coeff * g00 + CBY_coeff * b00 + halfShift + shifted16;
            int32_t y01 = CRY_coeff * r01 + CGY_coeff * g01 + CBY_coeff * b01 + halfShift + shifted16;
            int32_t y10 = CRY_coeff * r10 + CGY_coeff * g10 + CBY_coeff * b10 + halfShift + shifted16;
            int32_t y11 = CRY_coeff * r11 + CGY_coeff * g11 + CBY_coeff * b11 + halfShift + shifted16;

            dst0[2 * j + 0] = sat_cast_u8(y00 >> SHIFT);
            dst0[2 * j + 1] = sat_cast_u8(y01 >> SHIFT);
            dst1[2 * j + 0] = sat_cast_u8(y10 >> SHIFT);
            dst1[2 * j + 1] = sat_cast_u8(y11 >> SHIFT);

            const int32_t shifted128 = (128 << SHIFT);
            int32_t u00              = CRU_coeff * r00 + CGU_coeff * g00 + CBU_coeff * b00 + halfShift + shifted128;
            int32_t v00              = CBU_coeff * r00 + CGV_coeff * g00 + CBV_coeff * b00 + halfShift + shifted128;

            if (isUV) {
                dst2[2 * j]     = sat_cast_u8(u00 >> SHIFT);
                dst2[2 * j + 1] = sat_cast_u8(v00 >> SHIFT);
            } else {
                dst2[2 * j]     = sat_cast_u8(v00 >> SHIFT);
                dst2[2 * j + 1] = sat_cast_u8(u00 >> SHIFT);
            }
        }
    }
}

template <int32_t dstcn, int32_t blueIdx, bool isUV>
void nv_2_rgb(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    const uint8_t delta_uv = 128, alpha = 255;
    for (int32_t i = 0; i < height; i += 2) {
        const uint8_t *src0 = inY + i * inYStride;
        const uint8_t *src1 = inY + (i + 1) * inYStride;
        const uint8_t *src2 = inUV + (i / 2) * inUVStride;
        uint8_t *dst0       = outData + i * outWidthStride;
        uint8_t *dst1       = outData + (i + 1) * outWidthStride;
        for (int32_t j = 0; j < width; j += 2, dst0 += 2 * dstcn, dst1 += 2 * dstcn) {
            int32_t y00 = std::max(0, int32_t(src0[j]) - 16) * CY_coeff;
            int32_t y01 = std::max(0, int32_t(src0[j + 1]) - 16) * CY_coeff;
            int32_t y10 = std::max(0, int32_t(src1[j]) - 16) * CY_coeff;
            int32_t y11 = std::max(0, int32_t(src1[j + 1]) - 16) * CY_coeff;
            int32_t u;
            int32_t v;
            if (isUV) {
                u = int32_t(src2[j]) - delta_uv;
                v = int32_t(src2[j + 1]) - delta_uv;
            } else {
                v = int32_t(src2[j]) - delta_uv;
                u = int32_t(src2[j + 1]) - delta_uv;
            }
            int32_t ruv = (1 << (SHIFT - 1)) + CVR_coeff * v;
            int32_t guv = (1 << (SHIFT - 1)) + CVG_coeff * v + CUG_coeff * u;
            int32_t buv = (1 << (SHIFT - 1)) + CUB_coeff * u;

            dst0[blueIdx]     = sat_cast_u8((y00 + buv) >> SHIFT);
            dst0[1]           = sat_cast_u8((y00 + guv) >> SHIFT);
            dst0[blueIdx ^ 2] = sat_cast_u8((y00 + ruv) >> SHIFT);

            dst1[blueIdx]     = sat_cast_u8((y10 + buv) >> SHIFT);
            dst1[1]           = sat_cast_u8((y10 + guv) >> SHIFT);
            dst1[blueIdx ^ 2] = sat_cast_u8((y10 + ruv) >> SHIFT);

            dst0[blueIdx + dstcn]       = sat_cast_u8((y01 + buv) >> SHIFT);
            dst0[1 + dstcn]             = sat_cast_u8((y01 + guv) >> SHIFT);
            dst0[(blueIdx ^ 2) + dstcn] = sat_cast_u8((y01 + ruv) >> SHIFT);

            dst1[blueIdx + dstcn]       = sat_cast_u8((y11 + buv) >> SHIFT);
            dst1[1 + dstcn]             = sat_cast_u8((y11 + guv) >> SHIFT);
            dst1[(blueIdx ^ 2) + dstcn] = sat_cast_u8((y11 + ruv) >> SHIFT);

            if (dstcn == 4) {
                dst1[3]         = alpha;
                dst0[3]         = alpha;
                dst1[3 + dstcn] = alpha;
                dst0[3 + dstcn] = alpha;
            }
        }
    }
}

template <>
::ppl::common::RetCode BGR2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::nv_2_rgb<3, 0, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    } else {
        nv_2_rgb<3, 0, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 0, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::nv_2_rgb<3, 2, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    } else {
        nv_2_rgb<3, 2, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 2, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::nv_2_rgb<3, 0, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    } else {
        nv_2_rgb<3, 0, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 0, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::nv_2_rgb<3, 2, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    } else {
        nv_2_rgb<3, 2, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 2, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::nv_2_rgb<3, 0, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    } else {
        nv_2_rgb<3, 0, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 0, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::nv_2_rgb<3, 2, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    } else {
        nv_2_rgb<3, 2, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 2, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::nv_2_rgb<3, 0, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    } else {
        nv_2_rgb<3, 0, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 0, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::nv_2_rgb<3, 2, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    } else {
        nv_2_rgb<3, 2, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 2, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}
}
}
} // namespace ppl::cv::x86
