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
#include "ppl/cv/types.h"
#include "ppl/cv/x86/util.hpp"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"

#include <string.h>
#include <cmath>

#include <limits.h>
#include <immintrin.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////

const int32_t CY_coeff  = 1220542;
const int32_t CUB_coeff = 2116026;
const int32_t CUG_coeff = -409993;
const int32_t CVG_coeff = -852492;
const int32_t CVR_coeff = 1673527;
const int32_t SHIFT     = 20;

// Coefficients for RGB to YUV420p conversion
const int32_t CRY_coeff = 269484;
const int32_t CGY_coeff = 528482;
const int32_t CBY_coeff = 102760;
const int32_t CRU_coeff = -155188;
const int32_t CGU_coeff = -305135;
const int32_t CBU_coeff = 460324;
const int32_t CGV_coeff = -385875;
const int32_t CBV_coeff = -74448;
struct YUV420p2RGB_u8 {
    YUV420p2RGB_u8(int32_t _bIdx)
        : bIdx(_bIdx) {}

    ::ppl::common::RetCode operator()(
        int32_t height,
        int32_t width,
        int32_t yStride,
        const uint8_t *y1,
        int32_t uStride,
        const uint8_t *u1,
        int32_t vStride,
        const uint8_t *v1,
        int32_t outWidthStride,
        uint8_t *dst) const
    {
        if (nullptr == y1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == u1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == v1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst) {
            return ppl::common::RC_INVALID_VALUE;
        }

        for (int32_t j = 0; j < height; j += 2, y1 += yStride * 2, u1 += uStride, v1 += vStride) {
            uint8_t *row1     = dst + j * outWidthStride;
            uint8_t *row2     = dst + (j + 1) * outWidthStride;
            const uint8_t *y2 = y1 + yStride;

            for (int32_t i = 0; i < width / 2; i += 1, row1 += 6, row2 += 6) {
                int32_t u = int32_t(u1[i]) - 128;
                int32_t v = int32_t(v1[i]) - 128;

                int32_t ruv = (1 << (SHIFT - 1)) + CVR_coeff * v;
                int32_t guv = (1 << (SHIFT - 1)) + CVG_coeff * v + CUG_coeff * u;
                int32_t buv = (1 << (SHIFT - 1)) + CUB_coeff * u;

                int32_t y00    = std::max(0, int32_t(y1[2 * i]) - 16) * CY_coeff;
                row1[2 - bIdx] = sat_cast_u8((y00 + ruv) >> SHIFT);
                row1[1]        = sat_cast_u8((y00 + guv) >> SHIFT);
                row1[bIdx]     = sat_cast_u8((y00 + buv) >> SHIFT);

                int32_t y01    = std::max(0, int32_t(y1[2 * i + 1]) - 16) * CY_coeff;
                row1[5 - bIdx] = sat_cast_u8((y01 + ruv) >> SHIFT);
                row1[4]        = sat_cast_u8((y01 + guv) >> SHIFT);
                row1[3 + bIdx] = sat_cast_u8((y01 + buv) >> SHIFT);

                int32_t y10    = std::max(0, int32_t(y2[2 * i]) - 16) * CY_coeff;
                row2[2 - bIdx] = sat_cast_u8((y10 + ruv) >> SHIFT);
                row2[1]        = sat_cast_u8((y10 + guv) >> SHIFT);
                row2[bIdx]     = sat_cast_u8((y10 + buv) >> SHIFT);

                int32_t y11    = std::max(0, int32_t(y2[2 * i + 1]) - 16) * CY_coeff;
                row2[5 - bIdx] = sat_cast_u8((y11 + ruv) >> SHIFT);
                row2[4]        = sat_cast_u8((y11 + guv) >> SHIFT);
                row2[3 + bIdx] = sat_cast_u8((y11 + buv) >> SHIFT);
            }
        }
        return ppl::common::RC_SUCCESS;
    }
    int32_t bIdx;
};

struct YUV420p2RGBA_u8 {
    YUV420p2RGBA_u8(int32_t _bIdx)
        : bIdx(_bIdx) {}

    ::ppl::common::RetCode operator()(
        int32_t height,
        int32_t width,
        int32_t yStride,
        const uint8_t *y1,
        int32_t uStride,
        const uint8_t *u1,
        int32_t vStride,
        const uint8_t *v1,
        int32_t outWidthStride,
        uint8_t *dst) const
    {
        if (nullptr == y1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == u1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == v1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst) {
            return ppl::common::RC_INVALID_VALUE;
        }

        for (int32_t j = 0; j < height; j += 2, y1 += yStride * 2, u1 += uStride, v1 += vStride) {
            uint8_t *row1     = dst + j * outWidthStride;
            uint8_t *row2     = dst + (j + 1) * outWidthStride;
            const uint8_t *y2 = y1 + yStride;

            for (int32_t i = 0; i < width / 2; i += 1, row1 += 8, row2 += 8) {
                int32_t u = int32_t(u1[i]) - 128;
                int32_t v = int32_t(v1[i]) - 128;

                int32_t ruv = (1 << (SHIFT - 1)) + CVR_coeff * v;
                int32_t guv = (1 << (SHIFT - 1)) + CVG_coeff * v + CUG_coeff * u;
                int32_t buv = (1 << (SHIFT - 1)) + CUB_coeff * u;

                int32_t y00    = std::max(0, int32_t(y1[2 * i]) - 16) * CY_coeff;
                row1[2 - bIdx] = sat_cast_u8((y00 + ruv) >> SHIFT);
                row1[1]        = sat_cast_u8((y00 + guv) >> SHIFT);
                row1[bIdx]     = sat_cast_u8((y00 + buv) >> SHIFT);
                row1[3]        = uint8_t(0xff);

                int32_t y01    = std::max(0, int32_t(y1[2 * i + 1]) - 16) * CY_coeff;
                row1[6 - bIdx] = sat_cast_u8((y01 + ruv) >> SHIFT);
                row1[5]        = sat_cast_u8((y01 + guv) >> SHIFT);
                row1[4 + bIdx] = sat_cast_u8((y01 + buv) >> SHIFT);
                row1[7]        = uint8_t(0xff);

                int32_t y10    = std::max(0, int32_t(y2[2 * i]) - 16) * CY_coeff;
                row2[2 - bIdx] = sat_cast_u8((y10 + ruv) >> SHIFT);
                row2[1]        = sat_cast_u8((y10 + guv) >> SHIFT);
                row2[bIdx]     = sat_cast_u8((y10 + buv) >> SHIFT);
                row2[3]        = uint8_t(0xff);

                int32_t y11    = std::max(0, int32_t(y2[2 * i + 1]) - 16) * CY_coeff;
                row2[6 - bIdx] = sat_cast_u8((y11 + ruv) >> SHIFT);
                row2[5]        = sat_cast_u8((y11 + guv) >> SHIFT);
                row2[4 + bIdx] = sat_cast_u8((y11 + buv) >> SHIFT);
                row2[7]        = uint8_t(0xff);
            }
        }
        return ppl::common::RC_SUCCESS;
    }
    int32_t bIdx;
};

struct RGBtoYUV420p_u8 {
    RGBtoYUV420p_u8(int32_t _bIdx)
        : bIdx(_bIdx) {}

    ::ppl::common::RetCode operator()(
        int32_t height,
        int32_t width,
        int32_t cn,
        int32_t inWidthStride,
        const uint8_t *src,
        int32_t yStride,
        uint8_t *dst_y,
        int32_t uStride,
        uint8_t *dst_u,
        int32_t vStride,
        uint8_t *dst_v) const
    {
        if (nullptr == src) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst_y) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst_u) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst_v) {
            return ppl::common::RC_INVALID_VALUE;
        }
        int32_t w = width;
        int32_t h = height;

        for (int32_t i = 0; i < h / 2; i++) {
            const uint8_t *row0 = src + i * 2 * inWidthStride;
            const uint8_t *row1 = src + (i * 2 + 1) * inWidthStride;

            uint8_t *y = dst_y + i * 2 * yStride;
            uint8_t *u = dst_u + i * uStride;
            uint8_t *v = dst_v + i * vStride;

            for (int32_t j = 0, k = 0; j < w * cn; j += 2 * cn, k++) {
                int32_t r00 = row0[2 - bIdx + j];
                int32_t g00 = row0[1 + j];
                int32_t b00 = row0[bIdx + j];
                int32_t r01 = row0[2 - bIdx + cn + j];
                int32_t g01 = row0[1 + cn + j];
                int32_t b01 = row0[bIdx + cn + j];
                int32_t r10 = row1[2 - bIdx + j];
                int32_t g10 = row1[1 + j];
                int32_t b10 = row1[bIdx + j];
                int32_t r11 = row1[2 - bIdx + cn + j];
                int32_t g11 = row1[1 + cn + j];
                int32_t b11 = row1[bIdx + cn + j];

                const int32_t shifted16 = (16 << SHIFT);
                const int32_t halfShift = (1 << (SHIFT - 1));
                int32_t y00             = CRY_coeff * r00 + CGY_coeff * g00 + CBY_coeff * b00 + halfShift + shifted16;
                int32_t y01             = CRY_coeff * r01 + CGY_coeff * g01 + CBY_coeff * b01 + halfShift + shifted16;
                int32_t y10             = CRY_coeff * r10 + CGY_coeff * g10 + CBY_coeff * b10 + halfShift + shifted16;
                int32_t y11             = CRY_coeff * r11 + CGY_coeff * g11 + CBY_coeff * b11 + halfShift + shifted16;

                y[2 * k + 0]           = sat_cast_u8(y00 >> SHIFT);
                y[2 * k + 1]           = sat_cast_u8(y01 >> SHIFT);
                y[2 * k + yStride + 0] = sat_cast_u8(y10 >> SHIFT);
                y[2 * k + yStride + 1] = sat_cast_u8(y11 >> SHIFT);

                const int32_t shifted128 = (128 << SHIFT);
                int32_t u00              = CRU_coeff * r00 + CGU_coeff * g00 + CBU_coeff * b00 + halfShift + shifted128;
                int32_t v00              = CBU_coeff * r00 + CGV_coeff * g00 + CBV_coeff * b00 + halfShift + shifted128;

                u[k] = sat_cast_u8(u00 >> SHIFT);
                v[k] = sat_cast_u8(v00 >> SHIFT);
            }
        }
        return ppl::common::RC_SUCCESS;
    }

    int32_t bIdx;
};

::ppl::common::RetCode BGR2I420SSE(
    const uint8_t *src,
    uint8_t *dstY,
    uint8_t *dstU,
    uint8_t *dstV,
    int32_t width,
    int32_t height,
    int32_t inWidthStride,
    int32_t yStride,
    int32_t uStride,
    int32_t vStride,
    bool flag_rgb = false)
{
    const int32_t shift      = 13;
    const int32_t half_shift = 1 << (shift - 1);

    int32_t coeff_YB = 0.098f * (1 << shift), coeff_YG = 0.504f * (1 << shift), coeff_YR = 0.257f * (1 << shift), bias_YC = 32;
    int32_t coeff_UB = 0.439f * (1 << shift), coeff_UG = -0.291f * (1 << shift), coeff_UR = -0.148f * (1 << shift), bias_UC = 257;
    int32_t coeff_VB = -0.071 * (1 << shift), coeff_VG = -0.368f * (1 << shift), coeff_VR = 0.439f * (1 << shift), bias_VC = 257;
    if (flag_rgb) {
        std::swap(coeff_YB, coeff_YR);
        std::swap(coeff_UB, coeff_UR);
        std::swap(coeff_VB, coeff_VR);
    }
    __m128i coeff_YBG = _mm_setr_epi16(coeff_YB, coeff_YG, coeff_YB, coeff_YG, coeff_YB, coeff_YG, coeff_YB, coeff_YG);
    __m128i coeff_YRC = _mm_setr_epi16(coeff_YR, bias_YC, coeff_YR, bias_YC, coeff_YR, bias_YC, coeff_YR, bias_YC);
    __m128i coeff_UBG = _mm_setr_epi16(coeff_UB, coeff_UG, coeff_UB, coeff_UG, coeff_UB, coeff_UG, coeff_UB, coeff_UG);
    __m128i coeff_URC = _mm_setr_epi16(coeff_UR, bias_UC, coeff_UR, bias_UC, coeff_UR, bias_UC, coeff_UR, bias_UC);
    __m128i coeff_VBG = _mm_setr_epi16(coeff_VB, coeff_VG, coeff_VB, coeff_VG, coeff_VB, coeff_VG, coeff_VB, coeff_VG);
    __m128i coeff_VRC = _mm_setr_epi16(coeff_VR, bias_VC, coeff_VR, bias_VC, coeff_VR, bias_VC, coeff_VR, bias_VC);
    __m128i half      = _mm_setr_epi16(0, half_shift, 0, half_shift, 0, half_shift, 0, half_shift);
    __m128i vzero     = _mm_setzero_si128();

    int32_t vsize = 16;
    for (int32_t h = 0; h < height; h++) {
        const uint8_t *src_ptr = src + h * inWidthStride;
        uint8_t *dstY_ptr      = dstY + h * yStride;
        uint8_t *dstU_ptr      = dstU + (h / 2) * uStride;
        uint8_t *dstV_ptr      = dstV + (h / 2) * vStride;
        bool evenh             = (h % 2) == 0;
        int32_t w              = 0;
        for (; w <= width / 2 - 16; w += vsize, src_ptr += vsize * 6) {
            __m128i data0_0 = _mm_loadu_si128((__m128i *)(src_ptr + 0));
            __m128i data0_1 = _mm_loadu_si128((__m128i *)(src_ptr + 16));
            __m128i data0_2 = _mm_loadu_si128((__m128i *)(src_ptr + 32));

            __m128i data1_0 = _mm_loadu_si128((__m128i *)(src_ptr + 48));
            __m128i data1_1 = _mm_loadu_si128((__m128i *)(src_ptr + 64));
            __m128i data1_2 = _mm_loadu_si128((__m128i *)(src_ptr + 80));

            __m128i v0_bgl = _mm_shuffle_epi8(data0_0, _mm_setr_epi8(0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, -1, -1, -1, -1, -1));
            __m128i v1_bgl = _mm_shuffle_epi8(data1_0, _mm_setr_epi8(0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, -1, -1, -1, -1, -1));

            v0_bgl = _mm_or_si128(v0_bgl, _mm_shuffle_epi8(data0_1, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 3, 5, 6)));
            v1_bgl = _mm_or_si128(v1_bgl, _mm_shuffle_epi8(data1_1, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 3, 5, 6)));

            __m128i v0_bgh  = _mm_shuffle_epi8(data0_1, _mm_setr_epi8(8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
            __m128i v1_bgh  = _mm_shuffle_epi8(data1_1, _mm_setr_epi8(8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
            v0_bgh          = _mm_or_si128(v0_bgh, _mm_shuffle_epi8(data0_2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14)));
            v1_bgh          = _mm_or_si128(v1_bgh, _mm_shuffle_epi8(data1_2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14)));
            __m128i v0_rcl  = _mm_shuffle_epi8(data0_0, _mm_setr_epi8(2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1));
            __m128i v1_rcl  = _mm_shuffle_epi8(data1_0, _mm_setr_epi8(2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1));
            v0_rcl          = _mm_or_si128(v0_rcl, _mm_shuffle_epi8(data0_1, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 4, -1, 7, -1)));
            v1_rcl          = _mm_or_si128(v1_rcl, _mm_shuffle_epi8(data1_1, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 4, -1, 7, -1)));
            __m128i v0_rch  = _mm_shuffle_epi8(data0_1, _mm_setr_epi8(10, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
            __m128i v1_rch  = _mm_shuffle_epi8(data1_1, _mm_setr_epi8(10, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
            v0_rch          = _mm_or_si128(v0_rch, _mm_shuffle_epi8(data0_2, _mm_setr_epi8(-1, -1, -1, -1, 0, -1, 3, -1, 6, -1, 9, -1, 12, -1, 15, -1)));
            v1_rch          = _mm_or_si128(v1_rch, _mm_shuffle_epi8(data1_2, _mm_setr_epi8(-1, -1, -1, -1, 0, -1, 3, -1, 6, -1, 9, -1, 12, -1, 15, -1)));
            __m128i v0_bgll = _mm_unpacklo_epi8(v0_bgl, vzero);
            __m128i v1_bgll = _mm_unpacklo_epi8(v1_bgl, vzero);
            __m128i v0_bglh = _mm_unpackhi_epi8(v0_bgl, vzero);
            __m128i v1_bglh = _mm_unpackhi_epi8(v1_bgl, vzero);
            __m128i v0_rcll = _mm_or_si128(_mm_unpacklo_epi8(v0_rcl, vzero), half);
            __m128i v1_rcll = _mm_or_si128(_mm_unpacklo_epi8(v1_rcl, vzero), half);
            __m128i v0_rclh = _mm_or_si128(_mm_unpackhi_epi8(v0_rcl, vzero), half);
            __m128i v1_rclh = _mm_or_si128(_mm_unpackhi_epi8(v1_rcl, vzero), half);
            __m128i v0_bghl = _mm_unpacklo_epi8(v0_bgh, vzero);
            __m128i v1_bghl = _mm_unpacklo_epi8(v1_bgh, vzero);
            __m128i v0_bghh = _mm_unpackhi_epi8(v0_bgh, vzero);
            __m128i v1_bghh = _mm_unpackhi_epi8(v1_bgh, vzero);
            __m128i v0_rchl = _mm_or_si128(_mm_unpacklo_epi8(v0_rch, vzero), half);
            __m128i v1_rchl = _mm_or_si128(_mm_unpacklo_epi8(v1_rch, vzero), half);
            __m128i v0_rchh = _mm_or_si128(_mm_unpackhi_epi8(v0_rch, vzero), half);
            __m128i v1_rchh = _mm_or_si128(_mm_unpackhi_epi8(v1_rch, vzero), half);

            __m128i Y_ll0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bgll, coeff_YBG), _mm_madd_epi16(v0_rcll, coeff_YRC)), shift);
            __m128i Y_lh0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bglh, coeff_YBG), _mm_madd_epi16(v0_rclh, coeff_YRC)), shift);
            __m128i Y_hl0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bghl, coeff_YBG), _mm_madd_epi16(v0_rchl, coeff_YRC)), shift);
            __m128i Y_hh0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bghh, coeff_YBG), _mm_madd_epi16(v0_rchh, coeff_YRC)), shift);
            _mm_storeu_si128((__m128i *)(dstY_ptr + 2 * w + 0), _mm_packus_epi16(_mm_packus_epi32(Y_ll0, Y_lh0), _mm_packus_epi32(Y_hl0, Y_hh0)));

            __m128i Y_ll1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bgll, coeff_YBG), _mm_madd_epi16(v1_rcll, coeff_YRC)), shift);
            __m128i Y_lh1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bglh, coeff_YBG), _mm_madd_epi16(v1_rclh, coeff_YRC)), shift);
            __m128i Y_hl1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bghl, coeff_YBG), _mm_madd_epi16(v1_rchl, coeff_YRC)), shift);
            __m128i Y_hh1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bghh, coeff_YBG), _mm_madd_epi16(v1_rchh, coeff_YRC)), shift);
            _mm_storeu_si128((__m128i *)(dstY_ptr + 2 * w + vsize), _mm_packus_epi16(_mm_packus_epi32(Y_ll1, Y_lh1), _mm_packus_epi32(Y_hl1, Y_hh1)));

            if (evenh) {
                __m128i U_ll0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bgll, coeff_UBG), _mm_madd_epi16(v0_rcll, coeff_URC)), shift);
                __m128i U_lh0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bglh, coeff_UBG), _mm_madd_epi16(v0_rclh, coeff_URC)), shift);
                __m128i U_hl0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bghl, coeff_UBG), _mm_madd_epi16(v0_rchl, coeff_URC)), shift);
                __m128i U_hh0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bghh, coeff_UBG), _mm_madd_epi16(v0_rchh, coeff_URC)), shift);
                __m128i U0    = _mm_packus_epi16(_mm_packus_epi32(U_ll0, U_lh0), _mm_packus_epi32(U_hl0, U_hh0));
                __m128i U_ll1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bgll, coeff_UBG), _mm_madd_epi16(v1_rcll, coeff_URC)), shift);
                __m128i U_lh1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bglh, coeff_UBG), _mm_madd_epi16(v1_rclh, coeff_URC)), shift);
                __m128i U_hl1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bghl, coeff_UBG), _mm_madd_epi16(v1_rchl, coeff_URC)), shift);
                __m128i U_hh1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bghh, coeff_UBG), _mm_madd_epi16(v1_rchh, coeff_URC)), shift);
                __m128i U1    = _mm_packus_epi16(_mm_packus_epi32(U_ll1, U_lh1), _mm_packus_epi32(U_hl1, U_hh1));
                __m128i mask  = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
                _mm_storeu_si128((__m128i *)(dstU_ptr + w), _mm_unpacklo_epi64(_mm_shuffle_epi8(U0, mask), _mm_shuffle_epi8(U1, mask)));

                __m128i V_ll0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bgll, coeff_VBG), _mm_madd_epi16(v0_rcll, coeff_VRC)), shift);
                __m128i V_lh0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bglh, coeff_VBG), _mm_madd_epi16(v0_rclh, coeff_VRC)), shift);
                __m128i V_hl0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bghl, coeff_VBG), _mm_madd_epi16(v0_rchl, coeff_VRC)), shift);
                __m128i V_hh0 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v0_bghh, coeff_VBG), _mm_madd_epi16(v0_rchh, coeff_VRC)), shift);
                __m128i V0    = _mm_packus_epi16(_mm_packus_epi32(V_ll0, V_lh0), _mm_packus_epi32(V_hl0, V_hh0));
                __m128i V_ll1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bgll, coeff_VBG), _mm_madd_epi16(v1_rcll, coeff_VRC)), shift);
                __m128i V_lh1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bglh, coeff_VBG), _mm_madd_epi16(v1_rclh, coeff_VRC)), shift);
                __m128i V_hl1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bghl, coeff_VBG), _mm_madd_epi16(v1_rchl, coeff_VRC)), shift);
                __m128i V_hh1 = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(v1_bghh, coeff_VBG), _mm_madd_epi16(v1_rchh, coeff_VRC)), shift);
                __m128i V1    = _mm_packus_epi16(_mm_packus_epi32(V_ll1, V_lh1), _mm_packus_epi32(V_hl1, V_hh1));
                _mm_storeu_si128((__m128i *)(dstV_ptr + w), _mm_unpacklo_epi64(_mm_shuffle_epi8(V0, mask), _mm_shuffle_epi8(V1, mask)));
            }
        }
        for (; w < width / 2; w++, src_ptr += 6) {
            const int32_t shifted16 = (16 << SHIFT);
            const int32_t halfShift = (1 << (SHIFT - 1));
            int32_t Blue = src_ptr[0], Green = src_ptr[1], Red = src_ptr[2];
            int32_t Blue_1 = src_ptr[3], Green_1 = src_ptr[4], Red_1 = src_ptr[5];
            if (flag_rgb) {
                std::swap(Blue, Red);
                std::swap(Blue_1, Red_1);
            }
            int32_t y0          = CBY_coeff * Blue + CGY_coeff * Green + CRY_coeff * Red + halfShift + shifted16;
            int32_t y1          = CBY_coeff * Blue_1 + CGY_coeff * Green_1 + CRY_coeff * Red_1 + halfShift + shifted16;
            dstY_ptr[2 * w]     = sat_cast_u8(y0 >> SHIFT);
            dstY_ptr[2 * w + 1] = sat_cast_u8(y1 >> SHIFT);
            if (evenh) {
                const int32_t halfShift  = (1 << (SHIFT - 1));
                const int32_t shifted128 = (128 << SHIFT);
                dstU_ptr[w]              = (CBU_coeff * Blue + CGU_coeff * Green + CRU_coeff * Red + halfShift + shifted128) >> SHIFT;
                dstV_ptr[w]              = (CBV_coeff * Blue + CGV_coeff * Green + CBU_coeff * Red + halfShift + shifted128) >> SHIFT;
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t dcn, int32_t bIdx>
::ppl::common::RetCode YUV420ptoRGB(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (width % 2 != 0 || height % 2 != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (dcn == 3) {
        YUV420p2RGB_u8 s = YUV420p2RGB_u8(bIdx);
        return s.operator()(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    } else if (dcn == 4) {
        YUV420p2RGBA_u8 s = YUV420p2RGBA_u8(bIdx);
        return s.operator()(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    }
}

template <int32_t scn, int32_t bIdx>
::ppl::common::RetCode RGBtoYUV420p(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (width % 2 != 0 || height % 2 != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    RGBtoYUV420p_u8 s = RGBtoYUV420p_u8(bIdx);
    return s.operator()(height, width, scn, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

template <>
::ppl::common::RetCode I4202BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataU = inData + height * inWidthStride;
    const uint8_t *inDataV = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::i420_2_rgb<3, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<3, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<3, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode YV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::i420_2_rgb<3, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<3, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<3, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode I4202BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataU = inData + height * inWidthStride;
    const uint8_t *inDataV = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<4, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<4, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode YV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<4, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<4, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode BGR2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataU = outData + height * outWidthStride;
    uint8_t *outDataV = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_SSE41)) {
        return BGR2I420SSE(inData, outDataY, outDataU, outDataV, width, height, inWidthStride, outWidthStride, outWidthStride / 2, outWidthStride / 2);
    }
    return RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}
template <>
::ppl::common::RetCode BGR2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_SSE41)) {
        return BGR2I420SSE(inData, outDataY, outDataU, outDataV, width, height, inWidthStride, outWidthStride, outWidthStride / 2, outWidthStride / 2);
    }
    return RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode BGRA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataU = outData + height * outWidthStride;
    uint8_t *outDataV = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<4, 0>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}
template <>
::ppl::common::RetCode BGRA2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<4, 0>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode I4202RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataU = inData + height * inWidthStride;
    const uint8_t *inDataV = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::i420_2_rgb<3, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<3, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<3, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode YV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::i420_2_rgb<3, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<3, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<3, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode I4202RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataU = inData + height * inWidthStride;
    const uint8_t *inDataV = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<4, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<4, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode YV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<4, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<4, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode RGB2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataU = outData + height * outWidthStride;
    uint8_t *outDataV = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_SSE41)) {
        return BGR2I420SSE(inData, outDataY, outDataU, outDataV, width, height, inWidthStride, outWidthStride, outWidthStride / 2, outWidthStride / 2, true);
    }
    return RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode RGB2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_SSE41)) {
        return BGR2I420SSE(inData, outDataY, outDataU, outDataV, width, height, inWidthStride, outWidthStride, outWidthStride / 2, outWidthStride / 2, true);
    }
    return RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode RGBA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataU = outData + height * outWidthStride;
    uint8_t *outDataV = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<4, 2>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode RGBA2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<4, 2>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

// multiple plane implement
template <>
::ppl::common::RetCode I4202BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inDataY && nullptr == inDataU && nullptr == inDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUStride == 0 || inVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::i420_2_rgb<3, 0>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    } else if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<3, 0>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<3, 0>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode I4202BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inDataY && nullptr == inDataU && nullptr == inDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUStride == 0 || inVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<4, 0>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<4, 0>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode BGR2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataY && nullptr == outDataU && nullptr == outDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUStride == 0 || outVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_SSE41)) {
        return BGR2I420SSE(inData, outDataY, outDataU, outDataV, width, height, inWidthStride, outYStride, outUStride, outVStride);
    }
    return RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

template <>
::ppl::common::RetCode BGRA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataY && nullptr == outDataU && nullptr == outDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUStride == 0 || outVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return RGBtoYUV420p<4, 0>(height, width, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

template <>
::ppl::common::RetCode I4202RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inDataY && nullptr == inDataU && nullptr == inDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUStride == 0 || inVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::i420_2_rgb<3, 2>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    } else if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<3, 2>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<3, 2>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode I4202RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inDataY && nullptr == inDataU && nullptr == inDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUStride == 0 || inVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        return YUV420ptoRGB_avx<4, 2>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    } else {
        return YUV420ptoRGB<4, 2>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    }
}
template <>
::ppl::common::RetCode RGB2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataY && nullptr == outDataU && nullptr == outDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUStride == 0 || outVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_SSE41)) {
        return BGR2I420SSE(inData, outDataY, outDataU, outDataV, width, height, inWidthStride, outYStride, outUStride, outVStride, true);
    }
    return RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

template <>
::ppl::common::RetCode RGBA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataY && nullptr == outDataU && nullptr == outDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUStride == 0 || outVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return RGBtoYUV420p<4, 2>(height, width, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

}
}
} // namespace ppl::cv::x86
