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

#include "ppl/cv/x86/intrinutils.hpp"
#include "ppl/cv/x86/util.hpp"
#include <stdint.h>
#include <immintrin.h>
#include "ppl/common/retcode.h"

#include <vector>
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

struct YUV420p2RGB_u8_avx128 {
    YUV420p2RGB_u8_avx128(int32_t _bIdx)
        : bIdx(_bIdx)
    {
        v_c0   = _mm_set1_epi32(CVR_coeff);
        v_c1   = _mm_set1_epi32(CVG_coeff);
        v_c2   = _mm_set1_epi32(CUG_coeff);
        v_c3   = _mm_set1_epi32(CUB_coeff);
        v_c4   = _mm_set1_epi32(CY_coeff);
        v_zero = _mm_setzero_si128();
        v128   = _mm_set1_epi32(128);
        vshift = _mm_set1_epi32(1 << (SHIFT - 1));
    }

    ::ppl::common::RetCode process(__m128i v_y, __m128i v_u, __m128i v_v, __m128i &v_u1, __m128i &v_v1, __m128i &v_r, __m128i &v_g, __m128i &v_b) const
    {
        __m128i ruv = _mm_add_epi32(_mm_mullo_epi32(v_c0, v_v), vshift);
        __m128i guv = _mm_add_epi32(_mm_mullo_epi32(v_c1, v_v), vshift);
        guv         = _mm_add_epi32(guv, _mm_mullo_epi32(v_c2, v_u));
        __m128i buv = _mm_add_epi32(_mm_mullo_epi32(v_c3, v_u), vshift);

        __m128i v_y_p = _mm_unpacklo_epi16(v_y, v_zero);
        __m128i y00   = _mm_mullo_epi32(_mm_max_epi32(_mm_sub_epi32(v_y_p, _mm_set1_epi32(16)), v_zero), v_c4);
        __m128i v_b0  = _mm_srai_epi32(_mm_add_epi32(y00, ruv), SHIFT);
        __m128i v_g0  = _mm_srai_epi32(_mm_add_epi32(y00, guv), SHIFT);
        __m128i v_r0  = _mm_srai_epi32(_mm_add_epi32(y00, buv), SHIFT);

        __m128i ruv1 = _mm_add_epi32(_mm_mullo_epi32(v_c0, v_v1), vshift);
        __m128i guv1 = _mm_add_epi32(_mm_mullo_epi32(v_c1, v_v1), vshift);
        guv1         = _mm_add_epi32(guv1, _mm_mullo_epi32(v_c2, v_u1));
        __m128i buv1 = _mm_add_epi32(_mm_mullo_epi32(v_c3, v_u1), vshift);

        __m128i v_y_p1 = _mm_unpackhi_epi16(v_y, v_zero);
        __m128i y01    = _mm_mullo_epi32(_mm_max_epi32(_mm_sub_epi32(v_y_p1, _mm_set1_epi32(16)), v_zero), v_c4);
        __m128i v_b1   = _mm_srai_epi32(_mm_add_epi32(y01, ruv1), SHIFT);
        __m128i v_g1   = _mm_srai_epi32(_mm_add_epi32(y01, guv1), SHIFT);
        __m128i v_r1   = _mm_srai_epi32(_mm_add_epi32(y01, buv1), SHIFT);

        if (bIdx == 0) {
            v_r = _mm_packs_epi32(v_r0, v_r1);
            v_g = _mm_packs_epi32(v_g0, v_g1);
            v_b = _mm_packs_epi32(v_b0, v_b1);
        } else {
            v_b = _mm_packs_epi32(v_r0, v_r1);
            v_g = _mm_packs_epi32(v_g0, v_g1);
            v_r = _mm_packs_epi32(v_b0, v_b1);
        }
        return ppl::common::RC_SUCCESS;
    }

    ::ppl::common::RetCode operator()(int32_t height, int32_t width, int32_t yStride, const uint8_t *y1, int32_t uStride, const uint8_t *u1, int32_t vStride, const uint8_t *v1, int32_t outWidthStride, uint8_t *dst) const
    {
        for (int32_t j = 0; j < height; j += 2, y1 += 2 * yStride, u1 += uStride, v1 += vStride) {
            uint8_t *row1     = dst + j * outWidthStride;
            uint8_t *row2     = dst + (j + 1) * outWidthStride;
            const uint8_t *y2 = y1 + yStride;
            int32_t i         = 0;
            for (; i < width / 2 - 8; i += 8, row1 += 16 * 3, row2 += 16 * 3) {
                __m128i v_u = _mm_loadl_epi64((__m128i const *)(u1 + i));
                __m128i v_v = _mm_loadl_epi64((__m128i const *)(v1 + i));

                __m128i v_y0 = _mm_loadu_si128((__m128i const *)(y1 + 2 * i));
                //__m128i v_y1 = _mm_loadu_si128((__m128i const *)(y1 + 2 * i + 8));

                __m128i v_r_0 = v_zero, v_g_0 = v_zero, v_b_0 = v_zero;
                __m128i v_ulo   = _mm_unpacklo_epi8(v_u, v_zero);
                __m128i v_u1olo = _mm_unpacklo_epi16(v_ulo, v_zero);
                __m128i v_u1    = _mm_unpacklo_epi32(v_u1olo, v_u1olo);
                __m128i v_u2    = _mm_unpackhi_epi32(v_u1olo, v_u1olo);
                v_u1            = _mm_sub_epi32(v_u1, v128);
                v_u2            = _mm_sub_epi32(v_u2, v128);
                __m128i v_vlo   = _mm_unpacklo_epi8(v_v, v_zero);
                __m128i v_v1olo = _mm_unpacklo_epi16(v_vlo, v_zero);
                __m128i v_v1    = _mm_unpacklo_epi32(v_v1olo, v_v1olo);
                __m128i v_v2    = _mm_unpackhi_epi32(v_v1olo, v_v1olo);
                v_v1            = _mm_sub_epi32(v_v1, v128);
                v_v2            = _mm_sub_epi32(v_v2, v128);
                process(_mm_unpacklo_epi8(v_y0, v_zero),
                        v_u1,
                        v_v1,
                        v_u2,
                        v_v2,
                        v_r_0,
                        v_g_0,
                        v_b_0);

                __m128i v_r_1 = v_zero, v_g_1 = v_zero, v_b_1 = v_zero;
                __m128i v_u1ohi = _mm_unpackhi_epi16(v_ulo, v_zero);
                __m128i v_u3    = _mm_unpacklo_epi32(v_u1ohi, v_u1ohi);
                __m128i v_u4    = _mm_unpackhi_epi32(v_u1ohi, v_u1ohi);
                v_u3            = _mm_sub_epi32(v_u3, v128);
                v_u4            = _mm_sub_epi32(v_u4, v128);
                __m128i v_v1ohi = _mm_unpackhi_epi16(v_vlo, v_zero);
                __m128i v_v3    = _mm_unpacklo_epi32(v_v1ohi, v_v1ohi);
                __m128i v_v4    = _mm_unpackhi_epi32(v_v1ohi, v_v1ohi);
                v_v3            = _mm_sub_epi32(v_v3, v128);
                v_v4            = _mm_sub_epi32(v_v4, v128);

                process(_mm_unpackhi_epi8(v_y0, v_zero),
                        v_u3,
                        v_v3,
                        v_u4,
                        v_v4,
                        v_r_1,
                        v_g_1,
                        v_b_1);
                __m128i v_r0 = _mm_packus_epi16(v_r_0, v_r_1);
                __m128i v_g0 = _mm_packus_epi16(v_g_0, v_g_1);
                __m128i v_b0 = _mm_packus_epi16(v_b_0, v_b_1);

                __m128i v_y1 = _mm_loadu_si128((__m128i const *)(y2 + 2 * i));
                //__m128i v_y1 = _mm_loadu_si128((__m128i const *)(y1 + 2 * i + 8));

                __m128i v_r_2 = v_zero, v_g_2 = v_zero, v_b_2 = v_zero;
                process(_mm_unpacklo_epi8(v_y1, v_zero),
                        v_u1,
                        v_v1,
                        v_u2,
                        v_v2,
                        v_r_2,
                        v_g_2,
                        v_b_2);

                __m128i v_r_3 = v_zero, v_g_3 = v_zero, v_b_3 = v_zero;
                process(_mm_unpackhi_epi8(v_y1, v_zero),
                        v_u3,
                        v_v3,
                        v_u4,
                        v_v4,
                        v_r_3,
                        v_g_3,
                        v_b_3);
                __m128i v_r1 = _mm_packus_epi16(v_r_2, v_r_3);
                __m128i v_g1 = _mm_packus_epi16(v_g_2, v_g_3);
                __m128i v_b1 = _mm_packus_epi16(v_b_2, v_b_3);

                _mm_interleave_epi8(v_r0, v_r1, v_g0, v_g1, v_b0, v_b1);
                _mm_storeu_si128((__m128i *)(row1), v_r0);
                _mm_storeu_si128((__m128i *)(row1 + 16), v_r1);
                _mm_storeu_si128((__m128i *)(row1 + 32), v_g0);
                _mm_storeu_si128((__m128i *)(row2), v_g1);
                _mm_storeu_si128((__m128i *)(row2 + 16), v_b0);
                _mm_storeu_si128((__m128i *)(row2 + 32), v_b1);
            }
            for (; i < width / 2; i += 1, row1 += 6, row2 += 6) {
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
    __m128i v_c0, v_c1, v_c2, v_c3, v_c4, v_zero, vshift, v128;
};

struct YUV420p2RGBA_u8_avx128 {
    YUV420p2RGBA_u8_avx128(int32_t _bIdx)
        : bIdx(_bIdx)
    {
        v_c0          = _mm_set1_epi32(CVR_coeff);
        v_c1          = _mm_set1_epi32(CVG_coeff);
        v_c2          = _mm_set1_epi32(CUG_coeff);
        v_c3          = _mm_set1_epi32(CUB_coeff);
        v_c4          = _mm_set1_epi32(CY_coeff);
        v_zero        = _mm_setzero_si128();
        v128          = _mm_set1_epi32(128);
        vshift        = _mm_set1_epi32(1 << (SHIFT - 1));
        uint8_t alpha = 255;
        v_alpha       = _mm_set1_epi8(*(char *)&alpha);
    }

    ::ppl::common::RetCode process(__m128i v_y, __m128i v_u, __m128i v_v, __m128i &v_u1, __m128i &v_v1, __m128i &v_r, __m128i &v_g, __m128i &v_b) const
    {
        __m128i ruv = _mm_add_epi32(_mm_mullo_epi32(v_c0, v_v), vshift);
        __m128i guv = _mm_add_epi32(_mm_mullo_epi32(v_c1, v_v), vshift);
        guv         = _mm_add_epi32(guv, _mm_mullo_epi32(v_c2, v_u));
        __m128i buv = _mm_add_epi32(_mm_mullo_epi32(v_c3, v_u), vshift);

        __m128i v_y_p = _mm_unpacklo_epi16(v_y, v_zero);
        __m128i y00   = _mm_mullo_epi32(_mm_max_epi32(_mm_sub_epi32(v_y_p, _mm_set1_epi32(16)), v_zero), v_c4);
        __m128i v_b0  = _mm_srai_epi32(_mm_add_epi32(y00, ruv), SHIFT);
        __m128i v_g0  = _mm_srai_epi32(_mm_add_epi32(y00, guv), SHIFT);
        __m128i v_r0  = _mm_srai_epi32(_mm_add_epi32(y00, buv), SHIFT);

        __m128i ruv1 = _mm_add_epi32(_mm_mullo_epi32(v_c0, v_v1), vshift);
        __m128i guv1 = _mm_add_epi32(_mm_mullo_epi32(v_c1, v_v1), vshift);
        guv1         = _mm_add_epi32(guv1, _mm_mullo_epi32(v_c2, v_u1));
        __m128i buv1 = _mm_add_epi32(_mm_mullo_epi32(v_c3, v_u1), vshift);

        __m128i v_y_p1 = _mm_unpackhi_epi16(v_y, v_zero);
        __m128i y01    = _mm_mullo_epi32(_mm_max_epi32(_mm_sub_epi32(v_y_p1, _mm_set1_epi32(16)), v_zero), v_c4);
        __m128i v_b1   = _mm_srai_epi32(_mm_add_epi32(y01, ruv1), SHIFT);
        __m128i v_g1   = _mm_srai_epi32(_mm_add_epi32(y01, guv1), SHIFT);
        __m128i v_r1   = _mm_srai_epi32(_mm_add_epi32(y01, buv1), SHIFT);

        if (bIdx == 0) {
            v_r = _mm_packs_epi32(v_r0, v_r1);
            v_g = _mm_packs_epi32(v_g0, v_g1);
            v_b = _mm_packs_epi32(v_b0, v_b1);
        } else {
            v_b = _mm_packs_epi32(v_r0, v_r1);
            v_g = _mm_packs_epi32(v_g0, v_g1);
            v_r = _mm_packs_epi32(v_b0, v_b1);
        }
        return ppl::common::RC_SUCCESS;
    }

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
        for (int32_t j = 0; j < height; j += 2, y1 += 2 * yStride, u1 += uStride, v1 += vStride) {
            uint8_t *row1     = dst + j * outWidthStride;
            uint8_t *row2     = dst + (j + 1) * outWidthStride;
            const uint8_t *y2 = y1 + yStride;
            int32_t i         = 0;
            for (; i < width / 2 - 8; i += 8, row1 += 16 * 4, row2 += 16 * 4) {
                __m128i v_u = _mm_loadl_epi64((__m128i const *)(u1 + i));
                __m128i v_v = _mm_loadl_epi64((__m128i const *)(v1 + i));

                __m128i v_y0 = _mm_loadu_si128((__m128i const *)(y1 + 2 * i));
                //__m128i v_y1 = _mm_loadu_si128((__m128i const *)(y1 + 2 * i + 8));

                __m128i v_r_0 = v_zero, v_g_0 = v_zero, v_b_0 = v_zero;
                __m128i v_ulo   = _mm_unpacklo_epi8(v_u, v_zero);
                __m128i v_u1olo = _mm_unpacklo_epi16(v_ulo, v_zero);
                __m128i v_u1    = _mm_unpacklo_epi32(v_u1olo, v_u1olo);
                __m128i v_u2    = _mm_unpackhi_epi32(v_u1olo, v_u1olo);
                v_u1            = _mm_sub_epi32(v_u1, v128);
                v_u2            = _mm_sub_epi32(v_u2, v128);
                __m128i v_vlo   = _mm_unpacklo_epi8(v_v, v_zero);
                __m128i v_v1olo = _mm_unpacklo_epi16(v_vlo, v_zero);
                __m128i v_v1    = _mm_unpacklo_epi32(v_v1olo, v_v1olo);
                __m128i v_v2    = _mm_unpackhi_epi32(v_v1olo, v_v1olo);
                v_v1            = _mm_sub_epi32(v_v1, v128);
                v_v2            = _mm_sub_epi32(v_v2, v128);
                process(_mm_unpacklo_epi8(v_y0, v_zero),
                        v_u1,
                        v_v1,
                        v_u2,
                        v_v2,
                        v_r_0,
                        v_g_0,
                        v_b_0);

                __m128i v_r_1 = v_zero, v_g_1 = v_zero, v_b_1 = v_zero;
                __m128i v_u1ohi = _mm_unpackhi_epi16(v_ulo, v_zero);
                __m128i v_u3    = _mm_unpacklo_epi32(v_u1ohi, v_u1ohi);
                __m128i v_u4    = _mm_unpackhi_epi32(v_u1ohi, v_u1ohi);
                v_u3            = _mm_sub_epi32(v_u3, v128);
                v_u4            = _mm_sub_epi32(v_u4, v128);
                __m128i v_v1ohi = _mm_unpackhi_epi16(v_vlo, v_zero);
                __m128i v_v3    = _mm_unpacklo_epi32(v_v1ohi, v_v1ohi);
                __m128i v_v4    = _mm_unpackhi_epi32(v_v1ohi, v_v1ohi);
                v_v3            = _mm_sub_epi32(v_v3, v128);
                v_v4            = _mm_sub_epi32(v_v4, v128);

                process(_mm_unpackhi_epi8(v_y0, v_zero),
                        v_u3,
                        v_v3,
                        v_u4,
                        v_v4,
                        v_r_1,
                        v_g_1,
                        v_b_1);
                __m128i v_r0 = _mm_packus_epi16(v_r_0, v_r_1);
                __m128i v_g0 = _mm_packus_epi16(v_g_0, v_g_1);
                __m128i v_b0 = _mm_packus_epi16(v_b_0, v_b_1);

                __m128i v_y1 = _mm_loadu_si128((__m128i const *)(y2 + 2 * i));
                //__m128i v_y1 = _mm_loadu_si128((__m128i const *)(y1 + 2 * i + 8));

                __m128i v_r_2 = v_zero, v_g_2 = v_zero, v_b_2 = v_zero;
                process(_mm_unpacklo_epi8(v_y1, v_zero),
                        v_u1,
                        v_v1,
                        v_u2,
                        v_v2,
                        v_r_2,
                        v_g_2,
                        v_b_2);

                __m128i v_r_3 = v_zero, v_g_3 = v_zero, v_b_3 = v_zero;
                process(_mm_unpackhi_epi8(v_y1, v_zero),
                        v_u3,
                        v_v3,
                        v_u4,
                        v_v4,
                        v_r_3,
                        v_g_3,
                        v_b_3);
                __m128i v_r1 = _mm_packus_epi16(v_r_2, v_r_3);
                __m128i v_g1 = _mm_packus_epi16(v_g_2, v_g_3);
                __m128i v_b1 = _mm_packus_epi16(v_b_2, v_b_3);

                __m128i v_a0 = v_alpha, v_a1 = v_alpha;
                _mm_interleave_epi8(v_r0, v_r1, v_g0, v_g1, v_b0, v_b1, v_a0, v_a1);
                _mm_storeu_si128((__m128i *)(row1), v_r0);
                _mm_storeu_si128((__m128i *)(row1 + 16), v_r1);
                _mm_storeu_si128((__m128i *)(row1 + 32), v_g0);
                _mm_storeu_si128((__m128i *)(row1 + 48), v_g1);
                _mm_storeu_si128((__m128i *)(row2), v_b0);
                _mm_storeu_si128((__m128i *)(row2 + 16), v_b1);
                _mm_storeu_si128((__m128i *)(row2 + 32), v_a0);
                _mm_storeu_si128((__m128i *)(row2 + 48), v_a1);
            }
            for (; i < width / 2; i += 1, row1 += 8, row2 += 8) {
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
    __m128i v_c0, v_c1, v_c2, v_c3, v_c4, v_zero, vshift, v128, v_alpha;
};

// Simplify interface
template <int32_t dcn, int32_t bIdx>
::ppl::common::RetCode YUV420ptoRGB_avx(
    int32_t height,
    int32_t width,
    int32_t yStride,
    const uint8_t *inDataY,
    int32_t uStride,
    const uint8_t *inDataU,
    int32_t vStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if ((width % 2 != 0) || (height % 2 != 0)) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (dcn == 3) {
        YUV420p2RGB_u8_avx128 s = YUV420p2RGB_u8_avx128(bIdx);
        return s.operator()(height, width, yStride, inDataY, uStride, inDataU, vStride, inDataV, outWidthStride, outData);
    } else if (dcn == 4) {
        YUV420p2RGBA_u8_avx128 s = YUV420p2RGBA_u8_avx128(bIdx);
        return s.operator()(height, width, yStride, inDataY, uStride, inDataU, vStride, inDataV, outWidthStride, outData);
    }
}

template ::ppl::common::RetCode YUV420ptoRGB_avx<3, 0>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData);
template ::ppl::common::RetCode YUV420ptoRGB_avx<4, 0>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData);
template ::ppl::common::RetCode YUV420ptoRGB_avx<3, 2>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData);
template ::ppl::common::RetCode YUV420ptoRGB_avx<4, 2>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData);

}
}
} // namespace ppl::cv::x86
