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

#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/cv/types.h"
#include "ppl/cv/x86/util.hpp"
#include <string.h>
#include <cmath>

#include <limits.h>
#include <immintrin.h>
#include <algorithm>
#define QUANTIZED_BITS       16
#define QUANTIZED_MULTIPLIER (1 << QUANTIZED_BITS)
#define QUANTIZED_BIAS       (1 << (QUANTIZED_BITS - 1))

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

template <typename T>
static inline T clip(T x, T a, T b)
{
    return std::max(a, std::min(x, b));
}

static inline int32_t saturate_cast(double value)
{
    int32_t round2zero = (int32_t)value;
    if (value >= 0) {
        return (value - round2zero != 0.5) ? (int32_t)(value + 0.5) : round2zero % 2 == 0 ? round2zero
                                                                                          : round2zero + 1;
    } else {
        return (round2zero - value != 0.5) ? (int32_t)(value - 0.5) : round2zero % 2 == 0 ? round2zero
                                                                                          : round2zero - 1;
    }
}

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_nearest(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T *dst,
    const T *src,
    const double *M,
    T delta)
{
    int32_t *_abdelta = new int32_t[outWidth * 2];
    int32_t *adelta = &_abdelta[0], *bdelta = adelta + outWidth;
    for (int32_t x = 0; x < outWidth; x++) {
        adelta[x] = saturate_cast(M[0] * x * 1024);
        bdelta[x] = saturate_cast(M[3] * x * 1024);
    }
    for (int32_t i = 0; i < outHeight; i++) {
        int32_t base_x    = saturate_cast((M[1] * i + M[2]) * 1024) + 512;
        int32_t base_y    = saturate_cast((M[4] * i + M[5]) * 1024) + 512;
        __m256i baseX_vec = _mm256_set1_epi32(base_x);
        __m256i baseY_vec = _mm256_set1_epi32(base_y);
        for (int32_t block_j = 0; block_j < round_up(outWidth, 8); block_j += 8) {
            int32_t sx_array[8];
            int32_t sy_array[8];
            __m256i sx_vec = _mm256_srai_epi32(_mm256_add_epi32(baseX_vec, _mm256_loadu_si256((__m256i const *)(adelta + block_j))), 10);
            __m256i sy_vec = _mm256_srai_epi32(_mm256_add_epi32(baseY_vec, _mm256_loadu_si256((__m256i const *)(bdelta + block_j))), 10);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sx_array), sx_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sy_array), sy_vec);
            for (int32_t j = block_j; j < std::min(block_j + 8, outWidth); j++) {
                int32_t sy = sy_array[j - block_j];
                int32_t sx = sx_array[j - block_j];
                if (borderMode == ppl::cv::BORDER_CONSTANT) {
                    int32_t idxSrc = sy * inWidthStride + sx * nc;
                    int32_t idxDst = i * outWidthStride + j * nc;
                    if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                        for (int32_t i = 0; i < nc; i++)
                            dst[idxDst + i] = src[idxSrc + i];
                    } else {
                        for (int32_t i = 0; i < nc; i++) {
                            dst[idxDst + i] = delta;
                        }
                    }
                } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                    sx             = clip(sx, 0, inWidth - 1);
                    sy             = clip(sy, 0, inHeight - 1);
                    int32_t idxSrc = sy * inWidthStride + sx * nc;
                    int32_t idxDst = i * outWidthStride + j * nc;
                    for (int32_t i = 0; i < nc; i++) {
                        dst[idxDst + i] = src[idxSrc + i];
                    }
                } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                    if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                        int32_t idxSrc = sy * inWidthStride + sx * nc;
                        int32_t idxDst = i * outWidthStride + j * nc;
                        for (int32_t i = 0; i < nc; i++)
                            dst[idxDst + i] = src[idxSrc + i];
                    } else {
                        continue;
                    }
                }
            }
        }
    }
    delete[] _abdelta;
    return ppl::common::RC_SUCCESS;
}

template <int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_linear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *dst,
    const float *src,
    const double *M,
    float delta)
{
    uint32_t cur_mode = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    __m256 base_seq_vec = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
    __m256 one_vec      = _mm256_set1_ps(1.0f);
    __m256 m3_vec       = _mm256_set1_ps(M[3]);
    __m256 m0_vec       = _mm256_set1_ps(M[0]);
    for (int32_t i = 0; i < outHeight; i++) {
        float base_x     = M[1] * i + M[2];
        float base_y     = M[4] * i + M[5];
        __m256 baseX_vec = _mm256_set1_ps(base_x);
        __m256 baseY_vec = _mm256_set1_ps(base_y);
        for (int32_t block_j = 0; block_j < round_up(outWidth, 8); block_j += 8) {
            int32_t sx0_array[8];
            int32_t sy0_array[8];
            float tab0_array[8];
            float tab1_array[8];
            float tab2_array[8];
            float tab3_array[8];
            __m256 seq_vec  = _mm256_add_ps(base_seq_vec, _mm256_set1_ps(block_j));
            __m256 x_vec    = _mm256_fmadd_ps(m0_vec, seq_vec, baseX_vec);
            __m256 y_vec    = _mm256_fmadd_ps(m3_vec, seq_vec, baseY_vec);
            __m256i sx0_vec = _mm256_cvttps_epi32(x_vec);
            __m256i sy0_vec = _mm256_cvttps_epi32(y_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sx0_array), sx0_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sy0_array), sy0_vec);
            __m256 u_vec     = _mm256_sub_ps(x_vec, _mm256_cvtepi32_ps(sx0_vec));
            __m256 v_vec     = _mm256_sub_ps(y_vec, _mm256_cvtepi32_ps(sy0_vec));
            __m256 taby0_vec = _mm256_sub_ps(one_vec, v_vec);
            __m256 taby1_vec = v_vec;
            __m256 tabx0_vec = _mm256_sub_ps(one_vec, u_vec);
            __m256 tabx1_vec = u_vec;
            _mm256_storeu_ps(tab0_array, _mm256_mul_ps(taby0_vec, tabx0_vec));
            _mm256_storeu_ps(tab1_array, _mm256_mul_ps(taby0_vec, tabx1_vec));
            _mm256_storeu_ps(tab2_array, _mm256_mul_ps(taby1_vec, tabx0_vec));
            _mm256_storeu_ps(tab3_array, _mm256_mul_ps(taby1_vec, tabx1_vec));
            for (int32_t j = block_j; j < std::min(block_j + 8, outWidth); ++j) {
                int32_t idx  = j - block_j;
                int32_t sx0  = sx0_array[idx];
                int32_t sy0  = sy0_array[idx];
                float tab[4] = {tab0_array[idx], tab1_array[idx], tab2_array[idx], tab3_array[idx]};
                float v0, v1, v2, v3;
                int32_t idxDst = (i * outWidthStride + j * nc);
                if (borderMode == ppl::cv::BORDER_CONSTANT) {
                    bool flag = (sx0 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 + 1 < inHeight);
                    if (flag) {
                        int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                        int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                        if (nc == 1) {
                            v0          = src[position1];
                            v1          = src[position1 + 1];
                            v2          = src[position2];
                            v3          = src[position2 + 1];
                            dst[idxDst] = static_cast<float>(tab[0] * v0 + tab[1] * v1 + tab[2] * v2 + tab[3] * v3);
                        } else if (nc == 3) {
                            dst[idxDst]     = static_cast<float>(tab[0] * src[position1] + tab[1] * src[position1 + 3] +
                                                             tab[2] * src[position2] + tab[3] * src[position2 + 3]);
                            dst[idxDst + 1] = static_cast<float>(tab[0] * src[position1 + 1] + tab[1] * src[position1 + 3 + 1] +
                                                                 tab[2] * src[position2 + 1] + tab[3] * src[position2 + 3 + 1]);
                            dst[idxDst + 2] = static_cast<float>(tab[0] * src[position1 + 2] + tab[1] * src[position1 + 3 + 2] +
                                                                 tab[2] * src[position2 + 2] + tab[3] * src[position2 + 3 + 2]);
                        } else {
                            for (int32_t k = 0; k < nc; k++) {
                                v0              = src[position1 + k];
                                v1              = src[position1 + nc + k];
                                v2              = src[position2 + k];
                                v3              = src[position2 + nc + k];
                                float sum       = tab[0] * v0 + tab[1] * v1 + tab[2] * v2 + tab[3] * v3;
                                dst[idxDst + k] = static_cast<float>(sum);
                            }
                        }
                    } else if (sx0 >= inWidth || sx0 + 1 < 0 || sy0 >= inHeight || sy0 + 1 < 0) {
                        for (int32_t k = 0; k < nc; k++) {
                            dst[idxDst + k] = delta;
                        }
                    } else {
                        bool flag0        = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                        bool flag1        = (flag0 && (sx0 + 1 < inWidth));
                        bool flag2        = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                        bool flag3        = (flag2 && (sx0 + 1 < inWidth));
                        int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                        int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                        if (nc == 1) {
                            v0          = flag0 ? src[position1] : delta;
                            v1          = flag1 ? src[position1 + 1] : delta;
                            v2          = flag2 ? src[position2] : delta;
                            v3          = flag3 ? src[position2 + 1] : delta;
                            float sum   = tab[0] * v0 + tab[1] * v1 + tab[2] * v2 + tab[3] * v3;
                            dst[idxDst] = static_cast<float>(sum);
                        } else {
                            for (int32_t k = 0; k < nc; k++) {
                                v0              = flag0 ? src[position1 + k] : delta;
                                v1              = flag1 ? src[position1 + nc + k] : delta;
                                v2              = flag2 ? src[position2 + k] : delta;
                                v3              = flag3 ? src[position2 + nc + k] : delta;
                                float sum       = tab[0] * v0 + tab[1] * v1 + tab[2] * v2 + tab[3] * v3;
                                dst[idxDst + k] = static_cast<float>(sum);
                            }
                        }
                    }
                } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                    int32_t sx1     = sx0 + 1;
                    int32_t sy1     = sy0 + 1;
                    sx0             = clip(sx0, 0, inWidth - 1);
                    sx1             = clip(sx1, 0, inWidth - 1);
                    sy0             = clip(sy0, 0, inHeight - 1);
                    sy1             = clip(sy1, 0, inHeight - 1);
                    const float *t0 = src + sy0 * inWidthStride + sx0 * nc;
                    const float *t1 = src + sy0 * inWidthStride + sx1 * nc;
                    const float *t2 = src + sy1 * inWidthStride + sx0 * nc;
                    const float *t3 = src + sy1 * inWidthStride + sx1 * nc;
                    for (int32_t k = 0; k < nc; ++k) {
                        float sum       = tab[0] * t0[k] + tab[1] * t1[k] + tab[2] * t2[k] + tab[3] * t3[k];
                        dst[idxDst + k] = static_cast<float>(sum);
                    }
                } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                    bool flag = (sx0 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 + 1 < inHeight);
                    if (flag) {
                        for (int32_t k = 0; k < nc; k++) {
                            int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                            int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                            v0                = src[position1 + k];
                            v1                = src[position1 + nc + k];
                            v2                = src[position2 + k];
                            v3                = src[position2 + nc + k];
                            float sum         = tab[0] * v0 + tab[1] * v1 + tab[2] * v2 + tab[3] * v3;
                            dst[idxDst + k]   = static_cast<float>(sum);
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
    }
    _MM_SET_ROUNDING_MODE(cur_mode);
    return ppl::common::RC_SUCCESS;
}

template <int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_linear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t *dst,
    const uint8_t *src,
    const double *M,
    uint8_t delta)
{
    uint32_t cur_mode = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    __m256 base_seq_vec             = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
    __m256 quantized_multiplier_vec = _mm256_set1_ps(QUANTIZED_MULTIPLIER);
    __m128i quantized_bias_vec      = _mm_set1_epi32(QUANTIZED_BIAS);
    __m256 one_vec                  = _mm256_set1_ps(1.0f);
    __m256 m3_vec                   = _mm256_set1_ps(M[3]);
    __m256 m0_vec                   = _mm256_set1_ps(M[0]);
    for (int32_t i = 0; i < outHeight; i++) {
        float base_x     = M[1] * i + M[2];
        float base_y     = M[4] * i + M[5];
        __m256 baseX_vec = _mm256_set1_ps(base_x);
        __m256 baseY_vec = _mm256_set1_ps(base_y);
        for (int32_t block_j = 0; block_j < round_up(outWidth, 8); block_j += 8) {
            int32_t sx0_array[8];
            int32_t sy0_array[8];
            int32_t tab0_array[8];
            int32_t tab1_array[8];
            int32_t tab2_array[8];
            int32_t tab3_array[8];
            __m256 seq_vec   = _mm256_add_ps(base_seq_vec, _mm256_set1_ps(block_j));
            __m256 x_vec     = _mm256_fmadd_ps(m0_vec, seq_vec, baseX_vec);
            __m256 y_vec     = _mm256_fmadd_ps(m3_vec, seq_vec, baseY_vec);
            __m256i sx0_vec  = _mm256_cvtps_epi32(x_vec);
            __m256i sy0_vec  = _mm256_cvtps_epi32(y_vec);
            __m256 u_vec     = _mm256_sub_ps(x_vec, _mm256_cvtepi32_ps(sx0_vec));
            __m256 v_vec     = _mm256_sub_ps(y_vec, _mm256_cvtepi32_ps(sy0_vec));
            __m256 taby0_vec = _mm256_sub_ps(one_vec, v_vec);
            __m256 taby1_vec = v_vec;
            __m256 tabx0_vec = _mm256_sub_ps(one_vec, u_vec);
            __m256 tabx1_vec = u_vec;
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(tab0_array), _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_mul_ps(taby0_vec, tabx0_vec), quantized_multiplier_vec)));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(tab1_array), _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_mul_ps(taby0_vec, tabx1_vec), quantized_multiplier_vec)));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(tab2_array), _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_mul_ps(taby1_vec, tabx0_vec), quantized_multiplier_vec)));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(tab3_array), _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_mul_ps(taby1_vec, tabx1_vec), quantized_multiplier_vec)));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sx0_array), sx0_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(sy0_array), sy0_vec);
            for (int32_t j = block_j; j < std::min(block_j + 8, outWidth); ++j) {
                int32_t idx = j - block_j;
                int32_t sx0 = sx0_array[idx];
                int32_t sy0 = sy0_array[idx];
                uint8_t v0, v1, v2, v3;
                int32_t idxDst = (i * outWidthStride + j * nc);
                if (borderMode == ppl::cv::BORDER_CONSTANT) {
                    bool all_valid = (sx0 >= 0 && sx0 < (inWidth - 1) && sy0 >= 0 && sy0 < (inHeight - 1));
                    if (all_valid) {
                        int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                        int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                        for (int32_t k = 0; k < nc; k++) {
                            v0              = src[position1 + k];
                            v1              = src[position1 + nc + k];
                            v2              = src[position2 + k];
                            v3              = src[position2 + nc + k];
                            int32_t sum     = (tab0_array[idx] * v0 + tab1_array[idx] * v1 + tab2_array[idx] * v2 + tab3_array[idx] * v3 + QUANTIZED_BIAS) >> QUANTIZED_BITS;
                            dst[idxDst + k] = static_cast<uint8_t>(sum);
                        }
                    } else {
                        bool all_invalid = (sx0 < -1 || sx0 >= inWidth || sy0 < -1 || sy0 >= inHeight);
                        if (all_invalid) {
                            v0 = delta;
                            for (int32_t k = 0; k < nc; k++) {
                                int32_t sum     = (tab0_array[idx] * v0 + tab1_array[idx] * v0 + tab2_array[idx] * v0 + tab3_array[idx] * v0 + QUANTIZED_BIAS) >> QUANTIZED_BITS;
                                dst[idxDst + k] = static_cast<uint8_t>(sum);
                            }
                        } else {
                            bool flag0        = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                            bool flag1        = (flag0 && (sx0 + 1 < inWidth));
                            bool flag2        = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                            bool flag3        = (flag2 && (sx0 + 1 < inWidth));
                            int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                            int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                            for (int32_t k = 0; k < nc; k++) {
                                v0              = flag0 ? src[position1 + k] : delta;
                                v1              = flag1 ? src[position1 + nc + k] : delta;
                                v2              = flag2 ? src[position2 + k] : delta;
                                v3              = flag3 ? src[position2 + nc + k] : delta;
                                int32_t sum     = (tab0_array[idx] * v0 + tab1_array[idx] * v1 + tab2_array[idx] * v2 + tab3_array[idx] * v3 + QUANTIZED_BIAS) >> QUANTIZED_BITS;
                                dst[idxDst + k] = static_cast<uint8_t>(sum);
                            }
                        }
                    }
                } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                    int32_t sx1        = sx0 + 1;
                    int32_t sy1        = sy0 + 1;
                    bool valid_for_all = (sx0 >= 0 && sx0 < (inWidth - 1) && sy0 >= 0 && sy0 < (inHeight - 1));
                    if (valid_for_all) {
                        sx1 = sx0 + 1;
                        sy1 = sy0 + 1;
                    } else {
                        sx0 = clip(sx0, 0, inWidth - 1);
                        sx1 = clip(sx1, 0, inWidth - 1);
                        sy0 = clip(sy0, 0, inHeight - 1);
                        sy1 = clip(sy1, 0, inHeight - 1);
                    }
                    const uint8_t *t0 = src + sy0 * inWidthStride + sx0 * nc;
                    const uint8_t *t1 = src + sy0 * inWidthStride + sx1 * nc;
                    const uint8_t *t2 = src + sy1 * inWidthStride + sx0 * nc;
                    const uint8_t *t3 = src + sy1 * inWidthStride + sx1 * nc;
                    if (nc == 4) {
                        __m128i v0_vec             = _mm_cvtepu8_epi32(_mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(t0))));
                        __m128i v1_vec             = _mm_cvtepu8_epi32(_mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(t1))));
                        __m128i v2_vec             = _mm_cvtepu8_epi32(_mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(t2))));
                        __m128i v3_vec             = _mm_cvtepu8_epi32(_mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(t3))));
                        __m128i quantized_tab0_vec = _mm_set1_epi32(tab0_array[idx]);
                        __m128i quantized_tab1_vec = _mm_set1_epi32(tab1_array[idx]);
                        __m128i quantized_tab2_vec = _mm_set1_epi32(tab2_array[idx]);
                        __m128i quantized_tab3_vec = _mm_set1_epi32(tab3_array[idx]);
                        __m128i result             = _mm_srai_epi32(_mm_add_epi32(quantized_bias_vec,
                                                                      _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(quantized_tab0_vec, v0_vec), _mm_mullo_epi32(quantized_tab1_vec, v1_vec)),
                                                                                    _mm_add_epi32(_mm_mullo_epi32(quantized_tab2_vec, v2_vec), _mm_mullo_epi32(quantized_tab3_vec, v3_vec)))),
                                                        QUANTIZED_BITS);
                        _mm_store_ss(reinterpret_cast<float *>(dst + idxDst), _mm_castsi128_ps(_mm_packus_epi16(_mm_packus_epi32(result, result), result)));
                    } else {
                        for (int32_t k = 0; k < nc; ++k) {
                            uint8_t v0      = t0[k];
                            uint8_t v1      = t1[k];
                            uint8_t v2      = t2[k];
                            uint8_t v3      = t3[k];
                            int32_t sum     = (tab0_array[idx] * v0 + tab1_array[idx] * v1 + tab2_array[idx] * v2 + tab3_array[idx] * v3 + QUANTIZED_BIAS) >> QUANTIZED_BITS;
                            dst[idxDst + k] = static_cast<uint8_t>(sum);
                        }
                    }
                } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                    bool flag0 = (sx0 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 + 1 < inHeight);
                    if (flag0) {
                        int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                        int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                        for (int32_t k = 0; k < nc; k++) {
                            v0              = src[position1 + k];
                            v1              = src[position1 + nc + k];
                            v2              = src[position2 + k];
                            v3              = src[position2 + nc + k];
                            int32_t sum     = (tab0_array[idx] * v0 + tab1_array[idx] * v1 + tab2_array[idx] * v2 + tab3_array[idx] * v3 + QUANTIZED_BIAS) >> QUANTIZED_BITS;
                            dst[idxDst + k] = static_cast<uint8_t>(sum);
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
    }
    _MM_SET_ROUNDING_MODE(cur_mode);
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_linear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T *dst,
    const T *src,
    const double *M,
    T delta)
{
    return warpaffine_linear<nc, borderMode>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, dst, src, M, delta);
}

template ::ppl::common::RetCode warpaffine_linear<float, 1, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 2, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 3, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 4, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 1, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 2, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 3, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 4, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<float, 1, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 2, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 3, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 4, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 1, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 2, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 3, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 4, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<float, 1, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 2, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 3, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<float, 4, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 1, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 2, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 3, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_linear<uint8_t, 4, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);

template ::ppl::common::RetCode warpaffine_nearest<float, 1, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 2, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 3, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 4, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 1, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 2, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 3, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 4, BORDER_CONSTANT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 1, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 2, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 3, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 4, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 1, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 2, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 3, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 4, BORDER_TRANSPARENT>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 1, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 2, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 3, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<float, 4, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, float *dst, const float *src, const double *M, float delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 1, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 2, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 3, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
template ::ppl::common::RetCode warpaffine_nearest<uint8_t, 4, BORDER_REPLICATE>(int32_t inHeight, int32_t inWidth, int32_t inWidthStride, int32_t outHeight, int32_t outWidth, int32_t outWidthStride, uint8_t *dst, const uint8_t *src, const double *M, uint8_t delta);
}
}
}
} // namespace ppl::cv::x86::fma
