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

#include "ppl/cv/types.h"
#include "ppl/cv/x86/util.hpp"
#include "ppl/common/retcode.h"
#include <string.h>
#include <cmath>

#include <limits.h>
#include <immintrin.h>
#include <algorithm>

#define CY_coeff  1220542
#define CUB_coeff 2116026
#define CUG_coeff -409993
#define CVG_coeff -852492
#define CVR_coeff 1673527
#define SHIFT     20

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

#define DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

template <int32_t dstcn, int32_t blueIdx>
::ppl::common::RetCode i420_2_rgb(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUStride,
    const uint8_t *inU,
    int32_t inVStride,
    const uint8_t *inV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    const uint8_t delta_uv = 128, alpha = 255;
    __m256i CY_coeff_VEC  = _mm256_set1_epi32(CY_coeff);
    __m256i CUB_coeff_VEC = _mm256_set1_epi32(CUB_coeff);
    __m256i CUG_coeff_VEC = _mm256_set1_epi32(CUG_coeff);
    __m256i CVG_coeff_VEC = _mm256_set1_epi32(CVG_coeff);
    __m256i CVR_coeff_VEC = _mm256_set1_epi32(CVR_coeff);

    __m256i delta_y_vec  = _mm256_set1_epi32(16);
    __m256i delta_uv_vec = _mm256_set1_epi32(delta_uv);
    __m256i zero_vec     = _mm256_set1_epi32(0);
    __m256i bias_vec     = _mm256_set1_epi32(1 << (SHIFT - 1));

    __m256i permute_idx = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);

    for (int32_t i = 0; i < height; i += 2) {
        const uint8_t *src0 = inY + i * inYStride;
        const uint8_t *src1 = inY + (i + 1) * inYStride;
        const uint8_t *src2 = inU + (i / 2) * inUStride;
        const uint8_t *src3 = inV + (i / 2) * inVStride;
        uint8_t *dst0       = outData + i * outWidthStride;
        uint8_t *dst1       = outData + (i + 1) * outWidthStride;

        for (int32_t j = 0; j < width / 8 * 8; j += 8, dst0 += 8 * dstcn, dst1 += 8 * dstcn) {
            __m256i y0_vec  = _mm256_mullo_epi32(_mm256_max_epi32(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(src0 + j))), delta_y_vec), zero_vec), CY_coeff_VEC);
            __m256i y1_vec  = _mm256_mullo_epi32(_mm256_max_epi32(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(src1 + j))), delta_y_vec), zero_vec), CY_coeff_VEC);
            __m256i u_vec   = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_load_ss(reinterpret_cast<const float *>(src2 + j / 2)))), delta_uv_vec);
            __m256i v_vec   = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_load_ss(reinterpret_cast<const float *>(src3 + j / 2)))), delta_uv_vec);
            u_vec           = _mm256_permutevar8x32_epi32(u_vec, permute_idx);
            v_vec           = _mm256_permutevar8x32_epi32(v_vec, permute_idx);
            __m256i ruv_vec = _mm256_add_epi32(_mm256_mullo_epi32(v_vec, CVR_coeff_VEC), bias_vec);
            __m256i guv_vec = _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(v_vec, CVG_coeff_VEC), bias_vec), _mm256_mullo_epi32(CUG_coeff_VEC, u_vec));
            __m256i buv_vec = _mm256_add_epi32(_mm256_mullo_epi32(u_vec, CUB_coeff_VEC), bias_vec);

            __m256i b0_vec = _mm256_srai_epi32(_mm256_add_epi32(y0_vec, buv_vec), SHIFT);
            __m256i b1_vec = _mm256_srai_epi32(_mm256_add_epi32(y1_vec, buv_vec), SHIFT);

            __m256i g0_vec = _mm256_srai_epi32(_mm256_add_epi32(y0_vec, guv_vec), SHIFT);
            __m256i g1_vec = _mm256_srai_epi32(_mm256_add_epi32(y1_vec, guv_vec), SHIFT);

            __m256i r0_vec = _mm256_srai_epi32(_mm256_add_epi32(y0_vec, ruv_vec), SHIFT);
            __m256i r1_vec = _mm256_srai_epi32(_mm256_add_epi32(y1_vec, ruv_vec), SHIFT);

            if (dstcn == 3) {
                __m256i shuffle_epi8_idx_vec  = _mm256_set_epi8(0, 0, 0, 0, 11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0, 0, 0, 0, 0, 11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0);
                __m256i shuffle_epi32_idx_vec = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);

                // row 0
                __m256i first_vec  = (blueIdx == 0) ? b0_vec : r0_vec;
                __m256i second_vec = g0_vec;
                __m256i third_vec  = (blueIdx == 0) ? r0_vec : b0_vec;

                __m256i out_vec = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_packus_epi16(_mm256_packus_epi32(first_vec, second_vec), _mm256_packus_epi32(third_vec, zero_vec)), shuffle_epi8_idx_vec), shuffle_epi32_idx_vec);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst0), _mm256_extractf128_si256(out_vec, 0));
                _mm_storel_epi64(reinterpret_cast<__m128i *>(dst0 + 16), _mm256_extractf128_si256(out_vec, 1));

                // row 1
                first_vec  = (blueIdx == 0) ? b1_vec : r1_vec;
                second_vec = g1_vec;
                third_vec  = (blueIdx == 0) ? r1_vec : b1_vec;

                out_vec = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_packus_epi16(_mm256_packus_epi32(first_vec, second_vec), _mm256_packus_epi32(third_vec, zero_vec)), shuffle_epi8_idx_vec), shuffle_epi32_idx_vec);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst1), _mm256_extractf128_si256(out_vec, 0));
                _mm_storel_epi64(reinterpret_cast<__m128i *>(dst1 + 16), _mm256_extractf128_si256(out_vec, 1));
            }
        }
        for (int32_t j = width / 8 * 8; j < width; j += 2, dst0 += 2 * dstcn, dst1 += 2 * dstcn) {
            int32_t y00 = std::max(0, int32_t(src0[j]) - 16) * CY_coeff;
            int32_t y01 = std::max(0, int32_t(src0[j + 1]) - 16) * CY_coeff;
            int32_t y10 = std::max(0, int32_t(src1[j]) - 16) * CY_coeff;
            int32_t y11 = std::max(0, int32_t(src1[j + 1]) - 16) * CY_coeff;
            int32_t u   = int32_t(src2[j / 2]) - delta_uv;
            int32_t v   = int32_t(src3[j / 2]) - delta_uv;

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
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode i420_2_rgb<3, 0>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUStride,
    const uint8_t *inU,
    int32_t inVStride,
    const uint8_t *inV,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode i420_2_rgb<3, 2>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUStride,
    const uint8_t *inU,
    int32_t inVStride,
    const uint8_t *inV,
    int32_t outWidthStride,
    uint8_t *outData);

}
}
}
} // namespace ppl::cv::x86::fma
