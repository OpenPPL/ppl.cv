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
#include "ppl/cv/x86/util.hpp"
#include "ppl/cv/types.h"
#include <assert.h>
#include <string.h>
#include <cmath>
#include <stdio.h>
#include <vector>
#include <limits.h>
#include <immintrin.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

#define FILTER_B(start, end)                                                                                                                                                              \
    for (y = (start); y < (end); y++) {                                                                                                                                                   \
        for (x = 0; x <= imageOutSizeX * cn - 16; x += 16) {                                                                                                                              \
            __m256 accumulator0_vec = _mm256_setzero_ps();                                                                                                                                \
            __m256 accumulator1_vec = _mm256_setzero_ps();                                                                                                                                \
            for (int fx = 0; fx < filterSize; fx++) {                                                                                                                                     \
                for (int fy = 0; fy < filterSize; fy++) {                                                                                                                                 \
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);                                                                                           \
                    __m128i data_u8_vec   = _mm_loadu_si128((const __m128i *)(imageIn + x + fx * cn + (fy + y) * inWidthStride));                                                         \
                    __m256 data0_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));                                                                                        \
                    __m256 data1_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));      \
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);                                                                             \
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);                                                                             \
                }                                                                                                                                                                         \
            }                                                                                                                                                                             \
            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_vec), _mm256_cvtps_epi32(accumulator1_vec));                                                    \
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);                                                                                           \
            _mm_storeu_si128((__m128i *)(imageOut + x + y * outWidthStride), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1))); \
        }                                                                                                                                                                                 \
        for (; x < imageOutSizeX * cn; x++) {                                                                                                                                             \
            float sum = 0;                                                                                                                                                                \
            for (int fx = 0; fx < filterSize; fx++) {                                                                                                                                     \
                for (int fy = 0; fy < filterSize; fy++) {                                                                                                                                 \
                    float f = filter[fx + fy * filterSize];                                                                                                                               \
                    sum += f * imageIn[(fy + y) * inWidthStride + x + fx * cn];                                                                                                           \
                }                                                                                                                                                                         \
            }                                                                                                                                                                             \
            imageOut[x + y * outWidthStride] = sat_cast(senseRound_f(sum));                                                                                                               \
        }                                                                                                                                                                                 \
    }
#define FILTER_F(start, end)                                                                                       \
    for (y = (start); y < (end); y++) {                                                                            \
        for (x = 0; x <= imageOutSizeX * cn - 16; x += 16) {                                                       \
            __m256 accumulator0_vec = _mm256_setzero_ps();                                                         \
            __m256 accumulator1_vec = _mm256_setzero_ps();                                                         \
            for (int fx = 0; fx < filterSize; fx++) {                                                              \
                for (int fy = 0; fy < filterSize; fy++) {                                                          \
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);                    \
                    __m256 data0_f32_vec  = _mm256_loadu_ps(imageIn + x + fx * cn + (fy + y) * inWidthStride);     \
                    __m256 data1_f32_vec  = _mm256_loadu_ps(imageIn + x + 8 + fx * cn + (fy + y) * inWidthStride); \
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);      \
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);      \
                }                                                                                                  \
            }                                                                                                      \
            _mm256_storeu_ps(imageOut + x + y * outWidthStride, accumulator0_vec);                                 \
            _mm256_storeu_ps(imageOut + x + 8 + y * outWidthStride, accumulator1_vec);                             \
        }                                                                                                          \
        for (; x < imageOutSizeX * cn; x++) {                                                                      \
            float sum = 0;                                                                                         \
            for (int fx = 0; fx < filterSize; fx++) {                                                              \
                for (int fy = 0; fy < filterSize; fy++) {                                                          \
                    float f = filter[fx + fy * filterSize];                                                        \
                    sum += f * imageIn[(fy + y) * inWidthStride + x + fx * cn];                                    \
                }                                                                                                  \
            }                                                                                                      \
            imageOut[x + y * outWidthStride] = ((sum));                                                            \
        }                                                                                                          \
    }

static int senseRound_f(float value)
{
    __m128 t = _mm_set_ss(value);
    return _mm_cvtss_si32(t);
    //return static_cast<int>(value);
}
static uint8_t sat_cast(int data)
{
    int val;
    val = data > 255 ? 255 : data;
    val = val < 0 ? 0 : val;
    return (uint8_t)val;
}

template <BorderType borderType>
inline int BorderInterpolate(int p, int len)
{
    if (borderType == ppl::cv::BORDER_REFLECT_101) {
        p = p < 0 ? (-p) : 2 * len - p - 2;
    } else if (borderType == ppl::cv::BORDER_REFLECT) {
        p = p < 0 ? (-p - 1) : 2 * len - p - 1;
    } else if (borderType == ppl::cv::BORDER_REPLICATE) {
        p = (p < 0) ? 0 : len - 1;
    } else if (borderType == ppl::cv::BORDER_CONSTANT) {
        p = -1;
    }
    return p;
}
inline int interpolate(int p, int len, BorderType border_type)
{
    if (border_type == ppl::cv::BORDER_REFLECT)
        return BorderInterpolate<ppl::cv::BORDER_REFLECT>(p, len);
    if (border_type == ppl::cv::BORDER_REFLECT101)
        return BorderInterpolate<ppl::cv::BORDER_REFLECT101>(p, len);
    if (border_type == ppl::cv::BORDER_REPLICATE)
        return BorderInterpolate<ppl::cv::BORDER_REPLICATE>(p, len);

    return 0;
}

static void maketable(
    std::vector<int> &tab,
    std::vector<int> &table,
    int left,
    int right,
    int imageInSizeX,
    int srcWidth,
    int cn,
    BorderType border_type)
{
    int i, j, k;
    for (i = 0; i < left; i++) {
        j = interpolate(i - left, srcWidth, border_type);
        j *= cn;
        for (k = 0; k < cn; k++) {
            tab[i * cn + k]   = j + k;
            table[i * cn + k] = j + k;
        }
    }
    for (i = 0; i < right; i++) {
        j = interpolate(srcWidth + i, srcWidth, border_type);
        j *= cn;
        for (k = 0; k < cn; k++) {
            tab[(i + left) * cn + k]                     = j + k;
            table[((3 * left) + 2 * right + i) * cn + k] = j + k;
        }
    }

    for (int i = left; i < 3 * left; i++) {
        for (k = 0; k < cn; k++) {
            table[i * cn + k] = (i - left) * cn + k;
        }
    }
    for (int i = 0; i < 2 * right; i++) {
        for (k = 0; k < cn; k++) {
            table[(i + 3 * left) * cn + k] = (srcWidth - 2 * left + i) * cn + k;
        }
    }
}

template <typename T>
inline static void copytopborder(
    T *dst,
    int imageInSizeX,
    const T *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    int inWidthStride,
    int cn,
    int top,
    int left,
    int right,
    const std::vector<int> &tab,
    BorderType border_type)
{
    const int elemSize = sizeof(T);
    T *dstInner        = dst + inWidthStride * top + left;
    // T* dstInner_tmp = dstInner;
    T *src_tmp         = const_cast<T *>(src);
    int i, j;
    for (i = 0; i < 2 * top; i++, dstInner += inWidthStride, src_tmp += srcWidthStride) {
        if (dstInner != src)
            memcpy(dstInner, src_tmp, srcWidth * elemSize);
        for (j = 0; j < left; j++)
            dstInner[j - left] = src_tmp[tab[j]];
        for (j = 0; j < right; j++)
            dstInner[j + srcWidth] = src_tmp[tab[j + left]];
    }

    T *imageIn_tmp = dst + inWidthStride * top;
    // copy border for top
    for (i = 0; i < top; i++) {
        j = interpolate(i - top, srcHeight, border_type);
        memcpy(imageIn_tmp + (i - top) * inWidthStride, imageIn_tmp + j * inWidthStride, imageInSizeX * cn * elemSize);
    }
}
template <typename T>
inline static void copybottomborder(
    T *dst,
    int imageInSizeX,
    const T *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    int inWidthStride,
    int cn,
    int top,
    int bottom,
    int left,
    int right,
    const std::vector<int> &tab,
    BorderType border_type)
{
    auto dstInner = dst + inWidthStride * top + left;
    // auto dstInner_tmp = dstInner;
    dstInner += inWidthStride * (srcHeight - 1);
    int elemSize     = sizeof(T);
    auto imageIn_tmp = dst + inWidthStride * top;
    auto src_tmp     = const_cast<T *>(src) + srcWidthStride * (srcHeight - 1);
    int i, j;

    for (i = 0; i < bottom * 2; i++, dstInner -= inWidthStride, src_tmp -= srcWidthStride) {
        memcpy(dstInner, src_tmp, srcWidth * elemSize);
        for (j = 0; j < left; j++)
            dstInner[j - left] = src_tmp[tab[j]];
        for (j = 0; j < right; j++)
            dstInner[j + srcWidth] = src_tmp[tab[j + left]];
    }
    for (i = 0; i < bottom; i++) {
        j = interpolate(i + srcHeight, srcHeight, border_type);
        memcpy(imageIn_tmp + (i + srcHeight) * inWidthStride, imageIn_tmp + j * inWidthStride, imageInSizeX * cn * elemSize);
    }
}

void convolution_b_r(
    int imageInSizeX,
    int imageInSizeY,
    int inWidthStride,
    uint8_t *imageIn,
    int filterSize,
    const float *filter,
    int outWidthStride,
    uint8_t *imageOut,
    int cn,
    const uint8_t *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    BorderType border_type)
{
    const int imageOutInnerX = imageInSizeX - 2 * filterSize + 2;
    const int imageOutInnerY = imageInSizeY - 2 * filterSize + 2;
    const int imageOutSizeY  = imageInSizeY - filterSize + 1;
    const int imageOutSizeX  = imageInSizeX - filterSize + 1;

    int left  = filterSize / 2;
    int right = filterSize / 2;

    const int top    = filterSize / 2;
    const int bottom = filterSize / 2;

    std::vector<int> tab((imageInSizeX - srcWidth) * cn);
    std::vector<int> table((left + right) * 3 * cn);

    maketable(tab, table, left, right, imageInSizeX, srcWidth, cn, border_type);
    left *= cn;
    right *= cn;
    srcWidth *= cn;

    int x, y;
    //copy top border
    copytopborder<uint8_t>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, left, right, tab, border_type);
    // filter for top
    FILTER_B(0, top);
    //filter for inner
    for (y = top; y < (imageOutInnerY) / 4 * 4 + top; y += 4) {
        int y0 = y;
        //filter for left
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + table[x + fx * cn]; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }
        //filter for middle
        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_y0_vec = _mm256_setzero_ps();
            __m256 accumulator1_y0_vec = _mm256_setzero_ps();
            __m256 accumulator0_y1_vec = _mm256_setzero_ps();
            __m256 accumulator1_y1_vec = _mm256_setzero_ps();
            __m256 accumulator0_y2_vec = _mm256_setzero_ps();
            __m256 accumulator1_y2_vec = _mm256_setzero_ps();
            __m256 accumulator0_y3_vec = _mm256_setzero_ps();
            __m256 accumulator1_y3_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto src_start  = src + (fy + y0 - top) * srcWidthStride + x - left + fx * cn;
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);
                    {
                        // for row0, y0
                        __m128i data_u8_vec  = _mm_loadu_si128((const __m128i *)(src_start));
                        __m256 data0_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                        __m256 data1_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                        accumulator0_y0_vec  = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_y0_vec);
                        accumulator1_y0_vec  = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_y0_vec);
                    }
                    {
                        // for row1, y1
                        __m128i data_u8_vec  = _mm_loadu_si128((const __m128i *)(src_start + srcWidthStride));
                        __m256 data0_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                        __m256 data1_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                        accumulator0_y1_vec  = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_y1_vec);
                        accumulator1_y1_vec  = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_y1_vec);
                    }
                    {
                        // for row2, y2
                        __m128i data_u8_vec  = _mm_loadu_si128((const __m128i *)(src_start + 2 * srcWidthStride));
                        __m256 data0_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                        __m256 data1_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                        accumulator0_y2_vec  = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_y2_vec);
                        accumulator1_y2_vec  = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_y2_vec);
                    }
                    {
                        // for row3, y3
                        __m128i data_u8_vec  = _mm_loadu_si128((const __m128i *)(src_start + 3 * srcWidthStride));
                        __m256 data0_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                        __m256 data1_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                        accumulator0_y3_vec  = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_y3_vec);
                        accumulator1_y3_vec  = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_y3_vec);
                    }
                }
            }

            const auto imageOut_offset = imageOut + x + y0 * outWidthStride;

            {
                __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_y0_vec), _mm256_cvtps_epi32(accumulator1_y0_vec));
                __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
                _mm_storeu_si128((__m128i *)(imageOut_offset), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            }
            {
                __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_y1_vec), _mm256_cvtps_epi32(accumulator1_y1_vec));
                __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
                _mm_storeu_si128((__m128i *)(imageOut_offset + outWidthStride), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            }
            {
                __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_y2_vec), _mm256_cvtps_epi32(accumulator1_y2_vec));
                __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
                _mm_storeu_si128((__m128i *)(imageOut_offset + 2 * outWidthStride), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            }
            {
                __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_y3_vec), _mm256_cvtps_epi32(accumulator1_y3_vec));
                __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
                _mm_storeu_si128((__m128i *)(imageOut_offset + 3 * outWidthStride), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            }
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + x - left + fx * cn; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }

        //filter for right
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    ; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }
    }
    for (y = (imageOutInnerY) / 4 * 4 + top; y < imageOutInnerY + top; y++) {
        //filter for left
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y - top) * srcWidthStride + table[x + fx * cn]; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f           = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                }
            }
            const auto imageOut_offset = x + y * outWidthStride;
            imageOut[imageOut_offset]  = sat_cast(senseRound_f(sum0));
        }
        //filter for middle
        for (; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_vec = _mm256_setzero_ps();
            __m256 accumulator1_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);
                    __m128i data_u8_vec   = _mm_loadu_si128((const __m128i *)(src + x - left + fx * cn + (fy + y - top) * srcWidthStride));
                    __m256 data0_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                    __m256 data1_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);
                }
            }
            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_vec), _mm256_cvtps_epi32(accumulator1_vec));
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(imageOut + x + y * outWidthStride), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f = filter[fx + fy * filterSize];
                    sum += f * src[(fy + y - top) * srcWidthStride + x - left + fx * cn];
                }
            }
            imageOut[x + y * outWidthStride] = sat_cast(senseRound_f(sum));
        }
        //filter for right
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;

            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    ; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                }
            }
            const auto imageOut_offset = x + y * outWidthStride;
            imageOut[imageOut_offset]  = sat_cast(senseRound_f(sum0));
        }
    }
    //copy bottom border
    copybottomborder<uint8_t>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, bottom, left, right, tab, border_type);
    // filter for bottom
    FILTER_B(imageOutInnerY + top, imageOutSizeY);
}

void convolution_f_r(
    int imageInSizeX,
    int imageInSizeY,
    int inWidthStride,
    float *imageIn,
    int filterSize,
    const float *filter,
    int outWidthStride,
    float *imageOut,
    int cn,
    const float *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    BorderType border_type)
{
    const int imageOutInnerX = imageInSizeX - 2 * filterSize + 2;
    const int imageOutInnerY = imageInSizeY - 2 * filterSize + 2;
    const int imageOutSizeY  = imageInSizeY - filterSize + 1;
    const int imageOutSizeX  = imageInSizeX - filterSize + 1;

    int left  = filterSize / 2;
    int right = filterSize / 2;

    const int top    = filterSize / 2;
    const int bottom = filterSize / 2;

    std::vector<int> tab((imageInSizeX - srcWidth) * cn);
    std::vector<int> table((left + right) * 3 * cn);

    maketable(tab, table, left, right, imageInSizeX, srcWidth, cn, border_type);

    left *= cn;
    right *= cn;
    srcWidth *= cn;
    int x, y;

    copytopborder<float>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, left, right, tab, border_type);
    //filter for top
    FILTER_F(0, top);
    for (y = top; y < (imageOutInnerY) / 4 * 4 + top; y += 4) {
        int y0               = y;
        int y1               = y + 1;
        int y2               = y + 2;
        int y3               = y + 3;
        const auto y0_offset = (y - top) * srcWidthStride;
        for (x = 0; x < left; x++) {
            //four rows
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f        = filter[fx + fy * filterSize];
                    auto src_start = (fy + y0 - top) * srcWidthStride + table[x + fx * cn];
                    sum0 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum1 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum2 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum3 += f * src[src_start];
                }
            }
            imageOut[x + y0 * outWidthStride] = sat_cast(senseRound_f(sum0));
            imageOut[x + y1 * outWidthStride] = sat_cast(senseRound_f(sum1));
            imageOut[x + y2 * outWidthStride] = sat_cast(senseRound_f(sum2));
            imageOut[x + y3 * outWidthStride] = sat_cast(senseRound_f(sum3));
        }

        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            const auto src_offset      = src + x - left + y0_offset;
            __m256 accumulator0_y0_vec = _mm256_setzero_ps();
            __m256 accumulator1_y0_vec = _mm256_setzero_ps();
            __m256 accumulator0_y1_vec = _mm256_setzero_ps();
            __m256 accumulator1_y1_vec = _mm256_setzero_ps();
            __m256 accumulator0_y2_vec = _mm256_setzero_ps();
            __m256 accumulator1_y2_vec = _mm256_setzero_ps();
            __m256 accumulator0_y3_vec = _mm256_setzero_ps();
            __m256 accumulator1_y3_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto filter_offset = fy * srcWidthStride + fx * cn;
                    const auto offset        = src_offset + filter_offset;

                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);
                    {
                        // for row0, y0
                        // __m128i data_u8_vec = _mm_loadu_si128((const __m128i*)(src+x+fx*cn + (fy+y0)*srcWidthStride));
                        __m256 data0_f32_vec = _mm256_loadu_ps(offset);
                        __m256 data1_f32_vec = _mm256_loadu_ps(offset + 8);
                        accumulator0_y0_vec  = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_y0_vec);
                        accumulator1_y0_vec  = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_y0_vec);
                    }
                    {
                        // for row1, y1
                        // __m128i data_u8_vec = _mm_loadu_si128((const __m128i*)(src+x+fx*cn + (fy+y1)*srcWidthStride));
                        __m256 data0_f32_vec = _mm256_loadu_ps(offset + srcWidthStride);
                        __m256 data1_f32_vec = _mm256_loadu_ps(offset + 8 + srcWidthStride);
                        accumulator0_y1_vec  = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_y1_vec);
                        accumulator1_y1_vec  = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_y1_vec);
                    }
                    {
                        // for row2, y2
                        // __m128i data_u8_vec = _mm_loadu_si128((const __m128i*)(src+x+fx*cn + (fy+y2)*srcWidthStride));
                        __m256 data0_f32_vec = _mm256_loadu_ps(offset + 2 * srcWidthStride);
                        __m256 data1_f32_vec = _mm256_loadu_ps(offset + 8 + 2 * srcWidthStride);
                        accumulator0_y2_vec  = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_y2_vec);
                        accumulator1_y2_vec  = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_y2_vec);
                    }
                    {
                        // for row3, y3
                        __m256 data0_f32_vec = _mm256_loadu_ps(offset + 3 * srcWidthStride);
                        __m256 data1_f32_vec = _mm256_loadu_ps(offset + 8 + 3 * srcWidthStride);
                        accumulator0_y3_vec  = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_y3_vec);
                        accumulator1_y3_vec  = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_y3_vec);
                    }
                }
            }
            const auto imageOut_offset = imageOut + x + y0 * outWidthStride;

            {
                _mm256_storeu_ps(imageOut_offset, accumulator0_y0_vec);
                _mm256_storeu_ps(imageOut_offset + 8, accumulator1_y0_vec);
            }
            {
                _mm256_storeu_ps(imageOut_offset + outWidthStride, accumulator0_y1_vec);
                _mm256_storeu_ps(imageOut_offset + 8 + outWidthStride, accumulator1_y1_vec);
            }
            {
                _mm256_storeu_ps(imageOut_offset + 2 * outWidthStride, accumulator0_y2_vec);
                _mm256_storeu_ps(imageOut_offset + 8 + 2 * outWidthStride, accumulator1_y2_vec);
            }
            {
                _mm256_storeu_ps(imageOut_offset + 3 * outWidthStride, accumulator0_y3_vec);
                _mm256_storeu_ps(imageOut_offset + 8 + 3 * outWidthStride, accumulator1_y3_vec);
            }
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            //four rows
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f = filter[fx + fy * filterSize];
                    sum0 += f * src[(fy + y0 - top) * srcWidthStride + x - left + fx * cn];
                    sum1 += f * src[(fy + y1 - top) * srcWidthStride + x - left + fx * cn];
                    sum2 += f * src[(fy + y2 - top) * srcWidthStride + x - left + fx * cn];
                    sum3 += f * src[(fy + y3 - top) * srcWidthStride + x - left + fx * cn];
                }
            }
            imageOut[x + y0 * outWidthStride] = sat_cast(senseRound_f(sum0));
            imageOut[x + y1 * outWidthStride] = sat_cast(senseRound_f(sum1));
            imageOut[x + y2 * outWidthStride] = sat_cast(senseRound_f(sum2));
            imageOut[x + y3 * outWidthStride] = sat_cast(senseRound_f(sum3));
        }
        for (; x < imageOutSizeX * cn; x++) {
            //four rows
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f        = filter[fx + fy * filterSize];
                    auto src_start = (fy + y0 - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    sum0 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum1 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum2 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum3 += f * src[src_start];
                }
            }
            imageOut[x + y0 * outWidthStride] = sat_cast(senseRound_f(sum0));
            imageOut[x + y1 * outWidthStride] = sat_cast(senseRound_f(sum1));
            imageOut[x + y2 * outWidthStride] = sat_cast(senseRound_f(sum2));
            imageOut[x + y3 * outWidthStride] = sat_cast(senseRound_f(sum3));
        }
    }
    for (y = (imageOutInnerY) / 4 * 4 + top; y < imageOutInnerY + top; y++) {
        for (x = 0; x < left; x++) {
            //four rows
            float sum0 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f        = filter[fx + fy * filterSize];
                    auto src_start = (fy + y - top) * srcWidthStride + table[x + fx * cn];
                    sum0 += f * src[src_start];
                }
            }
            imageOut[x + y * outWidthStride] = sum0;
        }
        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_vec = _mm256_setzero_ps();
            __m256 accumulator1_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);
                    __m256 data0_f32_vec  = _mm256_loadu_ps(src + x - left + fx * cn + (fy + y - top) * srcWidthStride);
                    __m256 data1_f32_vec  = _mm256_loadu_ps(src + x + 8 + fx * cn - left + (fy + y - top) * srcWidthStride);
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);
                }
            }
            _mm256_storeu_ps(imageOut + x + y * outWidthStride, accumulator0_vec);
            _mm256_storeu_ps(imageOut + x + 8 + y * outWidthStride, accumulator1_vec);
        }

        for (; x < left + imageOutInnerX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f = filter[fx + fy * filterSize];
                    sum += f * src[(fy + y - top) * srcWidthStride + x + fx * cn - left];
                }
            }
            imageOut[x + y * outWidthStride] = sum;
        }

        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f        = filter[fx + fy * filterSize];
                    auto src_start = (fy + y - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    sum0 += f * src[src_start];
                }
            }
            imageOut[x + y * outWidthStride] = sum0;
        }
    }

    copybottomborder<float>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, bottom, left, right, tab, border_type);
    //filter for bottom border
    FILTER_F(imageOutInnerY + top, imageOutSizeY);
}

template <>
void convolution_b<3>(
    int imageInSizeX,
    int imageInSizeY,
    int inWidthStride,
    uint8_t *imageIn,
    const float *filter,
    int outWidthStride,
    uint8_t *imageOut,
    int cn,
    const uint8_t *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    BorderType border_type)
{
    const int filterSize     = 3;
    const int imageOutInnerX = imageInSizeX - 2 * 2;
    const int imageOutInnerY = imageInSizeY - 2 * 2;

    const int imageOutSizeY = imageInSizeY - 2;
    const int imageOutSizeX = imageInSizeX - 2;

    int left  = filterSize / 2;
    int right = filterSize / 2;

    const int top    = filterSize / 2;
    const int bottom = filterSize / 2;

    std::vector<int> tab((imageInSizeX - srcWidth) * cn);
    std::vector<int> table((left + right) * 3 * cn);

    maketable(tab, table, left, right, imageInSizeX, srcWidth, cn, border_type);
    left *= cn;
    right *= cn;
    srcWidth *= cn;
    int x, y;
    copytopborder<uint8_t>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, left, right, tab, border_type);
    FILTER_B(0, top);
    //filter for middle

    for (y = top; y < (imageOutInnerY) / 4 * 4 + top; y += 4) {
        int y0 = y;
        //filter for left
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + table[x + fx * cn]; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f           = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }

        //filter for inner
        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 result_vec0_0 = _mm256_setzero_ps();
            __m256 result_vec0_1 = _mm256_setzero_ps();
            __m256 result_vec1_0 = _mm256_setzero_ps();
            __m256 result_vec1_1 = _mm256_setzero_ps();
            __m256 result_vec2_0 = _mm256_setzero_ps();
            __m256 result_vec2_1 = _mm256_setzero_ps();
            __m256 result_vec3_0 = _mm256_setzero_ps();
            __m256 result_vec3_1 = _mm256_setzero_ps();
            for (int fx = 0; fx < 3; fx++) {
                auto image_start    = src + srcWidthStride * (y - top) + (x - left) + fx * cn;
                auto filter_start   = filter + fx;
                __m256 kernel_vec0  = _mm256_broadcast_ss(filter_start);
                // row0  computation starts
                __m128i data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                __m256 acc_vec0     = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                __m256 acc_vec1     = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 3;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec0_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec0_1);

                // row1 computation starts
                __m256 kernel_vec1 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 3;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec1_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec1_1);

                // row2 computation starts
                __m256 kernel_vec2 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec2_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec2_1);

                // row3 computation starts

                data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec3_0);

                result_vec1_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec3_1);

                // row4 computation starts
                data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;

                result_vec2_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec3_0);

                result_vec2_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec3_1);

                // row5 computation starts
                data_u8_vec   = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0      = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1      = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec3_0);

                result_vec3_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec3_1);
            }
            auto out_start           = imageOut + x + y * outWidthStride;
            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec0_0), _mm256_cvtps_epi32(result_vec0_1));
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec1_0), _mm256_cvtps_epi32(result_vec1_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec2_0), _mm256_cvtps_epi32(result_vec2_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec3_0), _mm256_cvtps_epi32(result_vec3_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < 3; fx++) {
                for (int fy = 0; fy < 3; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + x - left + fx * cn; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f           = filter[fx + fy * 3];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto out_start                     = x + y0 * outWidthStride;
            imageOut[out_start]                      = sat_cast(senseRound_f(sum0));
            imageOut[out_start + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[out_start + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[out_start + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }

        //filter for right
        for (; x < imageOutSizeX * cn; x++) {
            // printf("hey %d \n",x);

            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    ; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }
    }
    for (y = (imageOutInnerY) / 4 * 4 + top; y < imageOutInnerY + top; y++) {
        //filter for left
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y - top) * srcWidthStride + table[x + fx * cn]; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f           = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                }
            }
            const auto imageOut_offset = x + y * outWidthStride;
            imageOut[imageOut_offset]  = sat_cast(senseRound_f(sum0));
        }

        //filter for inner
        for (; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_vec = _mm256_setzero_ps();
            __m256 accumulator1_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);
                    __m128i data_u8_vec   = _mm_loadu_si128((const __m128i *)(src + x - left + fx * cn + (fy + y - top) * srcWidthStride));
                    __m256 data0_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                    __m256 data1_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);
                }
            }
            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_vec), _mm256_cvtps_epi32(accumulator1_vec));
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(imageOut + x + y * outWidthStride), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f = filter[fx + fy * filterSize];
                    sum += f * src[(fy + y - top) * srcWidthStride + x - left + fx * cn];
                }
            }
            imageOut[x + y * outWidthStride] = sat_cast(senseRound_f(sum));
        }
        //filter for right
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    float f           = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                }
            }
            const auto imageOut_offset = x + y * outWidthStride;
            imageOut[imageOut_offset]  = sat_cast(senseRound_f(sum0));
        }
    }
    //copy bottom border
    copybottomborder<uint8_t>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, bottom, left, right, tab, border_type);
    // filter for bottom
    FILTER_B(imageOutInnerY + top, imageOutSizeY);
}

template <>
void convolution_b<7>(
    int imageInSizeX,
    int imageInSizeY,
    int inWidthStride,
    uint8_t *imageIn,
    const float *filter,
    int outWidthStride,
    uint8_t *imageOut,
    int cn,
    const uint8_t *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    BorderType border_type)
{
    const int filterSize     = 7;
    const int imageOutInnerX = imageInSizeX - 6 * 2;
    const int imageOutInnerY = imageInSizeY - 6 * 2;

    const int imageOutSizeY = imageInSizeY - 6;
    const int imageOutSizeX = imageInSizeX - 6;

    int left  = filterSize / 2;
    int right = filterSize / 2;

    const int top    = filterSize / 2;
    const int bottom = filterSize / 2;

    std::vector<int> tab((imageInSizeX - srcWidth) * cn);
    std::vector<int> table((left + right) * 3 * cn);
    maketable(tab, table, left, right, imageInSizeX, srcWidth, cn, border_type);
    left *= cn;
    right *= cn;
    srcWidth *= cn;
    int x, y;
    //copy top border
    copytopborder<uint8_t>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, left, right, tab, border_type);

    FILTER_B(0, top);

    //filter for middle
    for (y = top; y < (imageOutInnerY) / 4 * 4 + top; y += 4) {
        int y0 = y;
        //filter for left
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + table[x + fx * cn]; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f           = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }
        // filter for inner
        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 result_vec0_0 = _mm256_setzero_ps();
            __m256 result_vec0_1 = _mm256_setzero_ps();
            __m256 result_vec1_0 = _mm256_setzero_ps();
            __m256 result_vec1_1 = _mm256_setzero_ps();
            __m256 result_vec2_0 = _mm256_setzero_ps();
            __m256 result_vec2_1 = _mm256_setzero_ps();
            __m256 result_vec3_0 = _mm256_setzero_ps();
            __m256 result_vec3_1 = _mm256_setzero_ps();
            for (int fx = 0; fx < 7; fx++) {
                auto image_start    = src + srcWidthStride * (y - top) + (x - left) + fx * cn;
                auto filter_start   = filter + fx;
                __m256 kernel_vec0  = _mm256_broadcast_ss(filter_start);
                // row0  computation starts
                __m128i data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                __m256 acc_vec0     = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                __m256 acc_vec1     = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));

                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec0_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec0_1);

                // row1 computation starts
                __m256 kernel_vec1 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec1_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec1_1);

                // row2 computation starts
                __m256 kernel_vec2 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec2_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec2_1);

                // row3 computation starts
                __m256 kernel_vec3 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec3_1);

                // row4 computation starts
                __m256 kernel_vec4 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec3_1);

                // row5 computation starts
                __m256 kernel_vec5 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec5, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec5, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec3_1);

                // row6 computation starts
                __m256 kernel_vec6 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec6, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec5, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec6, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec5, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec3_1);
                // result0 available

                // result7 computation
                data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec6, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec5, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec3_0);

                result_vec1_1 = _mm256_fmadd_ps(kernel_vec6, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec5, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec3_1);
                // result1 available

                data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec6, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec5, acc_vec0, result_vec3_0);

                result_vec2_1 = _mm256_fmadd_ps(kernel_vec6, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec5, acc_vec1, result_vec3_1);
                // result2 available

                data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec6, acc_vec0, result_vec3_0);

                result_vec3_1 = _mm256_fmadd_ps(kernel_vec6, acc_vec1, result_vec3_1);
            }
            auto out_start           = imageOut + x + y * outWidthStride;
            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec0_0), _mm256_cvtps_epi32(result_vec0_1));
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec1_0), _mm256_cvtps_epi32(result_vec1_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec2_0), _mm256_cvtps_epi32(result_vec2_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec3_0), _mm256_cvtps_epi32(result_vec3_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + x - left + fx * cn; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * 7];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto out_start                     = x + y0 * outWidthStride;
            imageOut[out_start]                      = sat_cast(senseRound_f(sum0));
            imageOut[out_start + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[out_start + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[out_start + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }
        // filter for right
        for (x = left + imageOutInnerX * cn; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    ; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * 7];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }
    }

    for (y = (imageOutInnerY) / 4 * 4 + top; y < imageOutInnerY + top; y++) {
        //filter for left
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    const auto offset = (fy + y - top) * srcWidthStride + table[x + fx * cn]; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f           = filter[fx + fy * 7];
                    sum0 += f * src[offset];
                }
            }
            const auto imageOut_offset = x + y * outWidthStride;
            imageOut[imageOut_offset]  = sat_cast(senseRound_f(sum0));
        }

        //filter for inner
        for (; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_vec = _mm256_setzero_ps();
            __m256 accumulator1_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);
                    __m128i data_u8_vec   = _mm_loadu_si128((const __m128i *)(src + x - left + fx * cn + (fy + y - top) * srcWidthStride));
                    __m256 data0_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                    __m256 data1_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);
                }
            }
            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_vec), _mm256_cvtps_epi32(accumulator1_vec));
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(imageOut + x + y * outWidthStride), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    float f = filter[fx + fy * 7];
                    sum += f * src[(fy + y - top) * srcWidthStride + x - left + fx * cn];
                }
            }
            imageOut[x + y * outWidthStride] = sat_cast(senseRound_f(sum));
        }
        //filter for right
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    const auto offset = (fy + y - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    ; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f = filter[fx + fy * 7];
                    sum0 += f * src[offset];
                }
            }
            const auto imageOut_offset = x + y * outWidthStride;
            imageOut[imageOut_offset]  = sat_cast(senseRound_f(sum0));
        }
    }
    //copy bottom border
    copybottomborder<uint8_t>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, bottom, left, right, tab, border_type);
    // filter for bottom
    FILTER_B(imageOutInnerY + top, imageOutSizeY);
}

template <>
void convolution_b<5>(
    int imageInSizeX,
    int imageInSizeY,
    int inWidthStride,
    uint8_t *imageIn,
    const float *filter,
    int outWidthStride,
    uint8_t *imageOut,
    int cn,
    const uint8_t *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    BorderType border_type)
{
    const int filterSize     = 5;
    const int imageOutInnerX = imageInSizeX - 4 * 2;
    const int imageOutInnerY = imageInSizeY - 4 * 2;

    const int imageOutSizeY = imageInSizeY - 4;
    const int imageOutSizeX = imageInSizeX - 4;

    int left  = filterSize / 2;
    int right = filterSize / 2;

    const int top    = filterSize / 2;
    const int bottom = filterSize / 2;

    std::vector<int> tab((imageInSizeX - srcWidth) * cn);
    std::vector<int> table((left + right) * 3 * cn);

    maketable(tab, table, left, right, imageInSizeX, srcWidth, cn, border_type);
    left *= cn;
    right *= cn;
    srcWidth *= cn;
    int x, y;
    copytopborder<uint8_t>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, left, right, tab, border_type);
    FILTER_B(0, top);

    //filter for middle
    for (y = top; y < (imageOutInnerY) / 4 * 4 + top; y += 4) {
        int y0 = y;
        //filter for left
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + table[x + fx * cn]; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f           = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }

        //filter for inner
        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 result_vec0_0 = _mm256_setzero_ps();
            __m256 result_vec0_1 = _mm256_setzero_ps();
            __m256 result_vec1_0 = _mm256_setzero_ps();
            __m256 result_vec1_1 = _mm256_setzero_ps();
            __m256 result_vec2_0 = _mm256_setzero_ps();
            __m256 result_vec2_1 = _mm256_setzero_ps();
            __m256 result_vec3_0 = _mm256_setzero_ps();
            __m256 result_vec3_1 = _mm256_setzero_ps();
            for (int fx = 0; fx < 5; fx++) {
                auto image_start    = src + srcWidthStride * (y - top) + (x - left) + fx * cn;
                auto filter_start   = filter + fx;
                __m256 kernel_vec0  = _mm256_broadcast_ss(filter_start);
                // row0  computation starts
                __m128i data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                __m256 acc_vec0     = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                __m256 acc_vec1     = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 5;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec0_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec0_1);

                // row1 computation starts
                __m256 kernel_vec1 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 5;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec1_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec1_1);

                // row2 computation starts
                __m256 kernel_vec2 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 5;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec2_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec2_1);

                // row3 computation starts
                __m256 kernel_vec3 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 5;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec3_1);

                // row4 computation starts
                __m256 kernel_vec4 = _mm256_broadcast_ss(filter_start);
                data_u8_vec        = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1           = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;
                filter_start += 5;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec3_1);

                // row5 computation starts
                data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;

                result_vec1_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec3_0);

                result_vec1_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec3_1);

                // row6 computation starts
                data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;

                result_vec2_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec3_0);

                result_vec2_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec3_1);
                // result0 available

                // result7 computation
                data_u8_vec = _mm_loadu_si128((const __m128i *)(image_start));
                acc_vec0    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                acc_vec1    = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                image_start += srcWidthStride;

                result_vec3_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec3_0);

                result_vec3_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec3_1);
                // result1 available
            }
            auto out_start           = imageOut + x + y * outWidthStride;
            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec0_0), _mm256_cvtps_epi32(result_vec0_1));
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec1_0), _mm256_cvtps_epi32(result_vec1_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec2_0), _mm256_cvtps_epi32(result_vec2_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
            out_start += outWidthStride;

            result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(result_vec3_0), _mm256_cvtps_epi32(result_vec3_1));
            result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(out_start), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < 5; fx++) {
                for (int fy = 0; fy < 5; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + x - left + fx * cn; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * 5];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto out_start                     = x + y0 * outWidthStride;
            imageOut[out_start]                      = sat_cast(senseRound_f(sum0));
            imageOut[out_start + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[out_start + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[out_start + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }

        //filter for right
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y0 - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    ; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                    sum1 += f * src[offset + srcWidthStride];
                    sum2 += f * src[offset + srcWidthStride * 2];
                    sum3 += f * src[offset + srcWidthStride * 3];
                }
            }
            const auto imageOut_offset                     = x + y0 * outWidthStride;
            imageOut[imageOut_offset]                      = sat_cast(senseRound_f(sum0));
            imageOut[imageOut_offset + outWidthStride]     = sat_cast(senseRound_f(sum1));
            imageOut[imageOut_offset + outWidthStride * 2] = sat_cast(senseRound_f(sum2));
            imageOut[imageOut_offset + outWidthStride * 3] = sat_cast(senseRound_f(sum3));
        }
    }
    for (y = (imageOutInnerY) / 4 * 4 + top; y < imageOutInnerY + top; y++) {
        //filter for left
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y - top) * srcWidthStride + table[x + fx * cn]; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;
                    float f           = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                }
            }
            const auto imageOut_offset = x + y * outWidthStride;
            imageOut[imageOut_offset]  = sat_cast(senseRound_f(sum0));
        }

        //filter for inner
        for (; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_vec = _mm256_setzero_ps();
            __m256 accumulator1_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * filterSize);
                    __m128i data_u8_vec   = _mm_loadu_si128((const __m128i *)(src + x - left + fx * cn + (fy + y - top) * srcWidthStride));
                    __m256 data0_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_u8_vec));
                    __m256 data1_f32_vec  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_u8_vec), _mm_castsi128_ps(data_u8_vec)))));
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);
                }
            }
            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_vec), _mm256_cvtps_epi32(accumulator1_vec));
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(imageOut + x + y * outWidthStride), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    float f = filter[fx + fy * filterSize];
                    sum += f * src[(fy + y - top) * srcWidthStride + x - left + fx * cn];
                }
            }
            imageOut[x + y * outWidthStride] = sat_cast(senseRound_f(sum));
        }
        //filter for right
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;

            for (int fx = 0; fx < filterSize; fx++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    const auto offset = (fy + y - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    ; // x - left + ( y0 - top + fy) * srcWidthStride + fx*cn;

                    float f = filter[fx + fy * filterSize];
                    sum0 += f * src[offset];
                }
            }
            const auto imageOut_offset = x + y * outWidthStride;
            imageOut[imageOut_offset]  = sat_cast(senseRound_f(sum0));
        }
    }
    //copy bottom border
    copybottomborder<uint8_t>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, bottom, left, right, tab, border_type);
    // filter for bottom
    FILTER_B(imageOutInnerY + top, imageOutSizeY);
}

template <>
void convolution_f<3>(
    int imageInSizeX,
    int imageInSizeY,
    int inWidthStride,
    float *imageIn,
    const float *filter,
    int outWidthStride,
    float *imageOut,
    int cn,
    const float *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    BorderType border_type)
{
    //tear imageOut down into two parts: inner part with no respect to border; border part with respect to  border(left and right)
    //we store indices in vector<int> table of border for left and right
    const int imageOutInnerX = imageInSizeX - 2 * 2;
    const int imageOutInnerY = imageInSizeY - 2 * 2;
    const int imageOutSizeY  = imageInSizeY - 2;
    const int imageOutSizeX  = imageInSizeX - 2;
    const int filterSize     = 3;

    int left  = 1;
    int right = 1;

    const int top    = 1;
    const int bottom = 1;
    std::vector<int> tab((imageInSizeX - srcWidth) * cn);
    std::vector<int> table((left + right) * 3 * cn);
    maketable(tab, table, left, right, imageInSizeX, srcWidth, cn, border_type);

    left *= cn;
    right *= cn;
    srcWidth *= cn;

    int y, x;

    copytopborder<float>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, left, right, tab, border_type);
    // filter for top border
    FILTER_F(0, top);
    // filter for inner
    for (y = top; y < (imageOutInnerY) / 8 * 8 + top; y += 8) {
        int y0 = y, y1 = y + 1, y2 = y + 2, y3 = y + 3, y4 = y + 4, y5 = y + 5, y6 = y + 6, y7 = y + 7;
        // perform filter
        // left side
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            for (int fx = 0; fx < 3; fx++) {
                for (int fy = 0; fy < 3; fy++) {
                    float f        = filter[fx + fy * 3];
                    auto src_start = (fy + y0 - 1) * srcWidthStride + table[x + fx * cn];
                    sum0 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum1 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum2 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum3 += f * src[src_start];
                    src_start += srcWidthStride;

                    sum4 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum5 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum6 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum7 += f * src[src_start];
                }
            }

            auto out_start      = x + y0 * outWidthStride;
            imageOut[out_start] = sum0;

            out_start += outWidthStride;

            imageOut[out_start] = sum1;
            out_start += outWidthStride;

            imageOut[out_start] = sum2;
            out_start += outWidthStride;

            imageOut[out_start] = sum3;
            out_start += outWidthStride;

            imageOut[out_start] = sum4;
            out_start += outWidthStride;

            imageOut[out_start] = sum5;
            out_start += outWidthStride;

            imageOut[out_start] = sum6;
            out_start += outWidthStride;

            imageOut[out_start] = sum7;
        }

        for (x = left; x <= left + imageOutInnerX * cn - 8; x += 8) {
            __m256 result_vec0 = _mm256_setzero_ps();
            __m256 result_vec1 = _mm256_setzero_ps();
            __m256 result_vec2 = _mm256_setzero_ps();
            __m256 result_vec3 = _mm256_setzero_ps();
            __m256 result_vec4 = _mm256_setzero_ps();
            __m256 result_vec5 = _mm256_setzero_ps();
            __m256 result_vec6 = _mm256_setzero_ps();
            __m256 result_vec7 = _mm256_setzero_ps();

            for (int fx = 0; fx < 3; fx++) {
                auto image_start   = src + srcWidthStride * (y - 1) + (x - left) + fx * cn;
                __m256 kernel_vec0 = _mm256_broadcast_ss(filter + fx);

                // result0  computation starts
                __m256 acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec0);

                __m256 kernel_vec1 = _mm256_broadcast_ss(filter + 3 + fx);
                // result1 computation starts
                acc_vec0           = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec0);
                result_vec1 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec1);

                __m256 kernel_vec2 = _mm256_broadcast_ss(filter + 6 + fx);
                // result2 computation starts
                acc_vec0           = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec0);
                result_vec1 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec1);
                result_vec2 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec2);

                // result3 computation starts
                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec1 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec1);
                result_vec2 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec2);
                result_vec3 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec3);

                // result4 computation starts
                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec2 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec2);
                result_vec3 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec3);
                result_vec4 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec4);
                // result0 available

                // result5 computation starts
                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec3 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec3);
                result_vec4 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec4);
                result_vec5 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec5);
                // result1 available

                // result6 computation starts
                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec4 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec4);
                result_vec5 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec5);
                result_vec6 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec6);
                // result2 available
                // result7 computation
                acc_vec0    = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec5 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec5);
                result_vec6 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec6);
                result_vec7 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec7);
                // result3 available

                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec6 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec6);
                result_vec7 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec7);
                // result4 available

                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec7 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec7);
                // result5 available
                // result7 available
            }
            const auto out_start = imageOut + x + y * outWidthStride;
            _mm256_storeu_ps(out_start, result_vec0);
            _mm256_storeu_ps(out_start + outWidthStride, result_vec1);
            _mm256_storeu_ps(out_start + 2 * outWidthStride, result_vec2);
            _mm256_storeu_ps(out_start + 3 * outWidthStride, result_vec3);
            _mm256_storeu_ps(out_start + 4 * outWidthStride, result_vec4);
            _mm256_storeu_ps(out_start + 5 * outWidthStride, result_vec5);
            _mm256_storeu_ps(out_start + 6 * outWidthStride, result_vec6);
            _mm256_storeu_ps(out_start + 7 * outWidthStride, result_vec7);
        }

        for (; x < left + imageOutInnerX * cn; x++) {
            //  four rows
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            for (int fx = 0; fx < 3; fx++) {
                for (int fy = 0; fy < 3; fy++) {
                    float f = filter[fx + fy * 3];
                    sum0 += f * src[(fy + y0 - 1) * srcWidthStride + x - left + fx * cn];
                    sum1 += f * src[(fy + y1 - 1) * srcWidthStride + x - left + fx * cn];
                    sum2 += f * src[(fy + y2 - 1) * srcWidthStride + x - left + fx * cn];
                    sum3 += f * src[(fy + y3 - 1) * srcWidthStride + x - left + fx * cn];

                    sum4 += f * src[(fy + y4 - 1) * srcWidthStride + x - left + fx * cn];
                    sum5 += f * src[(fy + y5 - 1) * srcWidthStride + x - left + fx * cn];
                    sum6 += f * src[(fy + y6 - 1) * srcWidthStride + x - left + fx * cn];
                    sum7 += f * src[(fy + y7 - 1) * srcWidthStride + x - left + fx * cn];
                }
            }
            const auto out_start = x + y0 * outWidthStride;

            imageOut[out_start]                      = sum0;
            imageOut[out_start + outWidthStride]     = sum1;
            imageOut[out_start + 2 * outWidthStride] = sum2;
            imageOut[out_start + 3 * outWidthStride] = sum3;

            imageOut[out_start + 4 * outWidthStride] = sum4;
            imageOut[out_start + 5 * outWidthStride] = sum5;
            imageOut[out_start + 6 * outWidthStride] = sum6;
            imageOut[out_start + 7 * outWidthStride] = sum7;
        }

        // right side
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            for (int fx = 0; fx < 3; fx++) {
                for (int fy = 0; fy < 3; fy++) {
                    float f        = filter[fx + fy * 3];
                    auto src_start = (fy + y0 - 1) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    sum0 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum1 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum2 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum3 += f * src[src_start];
                    src_start += srcWidthStride;

                    sum4 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum5 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum6 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum7 += f * src[src_start];
                }
            }
            const auto out_start = x + y0 * outWidthStride;

            imageOut[out_start]                      = sum0;
            imageOut[out_start + outWidthStride]     = sum1;
            imageOut[out_start + 2 * outWidthStride] = sum2;
            imageOut[out_start + 3 * outWidthStride] = sum3;

            imageOut[out_start + 4 * outWidthStride] = sum4;
            imageOut[out_start + 5 * outWidthStride] = sum5;
            imageOut[out_start + 6 * outWidthStride] = sum6;
            imageOut[out_start + 7 * outWidthStride] = sum7;
        }
    }
    for (y = (imageOutInnerY) / 8 * 8 + top; y < imageOutInnerY + top; y++) {
        // left side
        for (x = 0; x < left; x++) {
            float sum = 0;
            for (int fx = 0; fx < 3; fx++) {
                for (int fy = 0; fy < 3; fy++) {
                    float f        = filter[fx + fy * 3];
                    auto src_start = (fy + y - 1) * srcWidthStride + table[x + fx * cn];
                    sum += f * src[src_start];
                }
            }
            const auto out_start = x + y * outWidthStride;
            imageOut[out_start]  = sum;
        }

        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_vec = _mm256_setzero_ps();
            __m256 accumulator1_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < 3; fx++) {
                for (int fy = 0; fy < 3; fy++) {
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * 3);
                    __m256 data0_f32_vec  = _mm256_loadu_ps(src + x - 1 * cn + fx * cn + (fy + y - 1) * srcWidthStride);
                    __m256 data1_f32_vec  = _mm256_loadu_ps(src + x + 8 + fx * cn - 1 * cn + (fy + y - 1) * srcWidthStride);
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);
                }
            }
            _mm256_storeu_ps(imageOut + x + y * outWidthStride, accumulator0_vec);
            _mm256_storeu_ps(imageOut + x + 8 + y * outWidthStride, accumulator1_vec);
        }

        for (; x < left + imageOutInnerX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < 3; fx++) {
                for (int fy = 0; fy < 3; fy++) {
                    float f = filter[fx + fy * 3];
                    sum += f * src[(fy + y - 1) * srcWidthStride + x - 1 * cn + fx * cn];
                }
            }
            imageOut[x + y * outWidthStride] = sum;
        }
        //right side
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;

            for (int fx = 0; fx < 3; fx++) {
                for (int fy = 0; fy < 3; fy++) {
                    float f        = filter[fx + fy * 3];
                    auto src_start = (fy + y - 1) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    sum0 += f * src[src_start];
                }
            }
            const auto out_start = x + y * outWidthStride;
            imageOut[out_start]  = sum0;
        }
    }
    // prepare data for bottom border
    copybottomborder<float>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, bottom, left, right, tab, border_type);
    //filter for bottom border
    FILTER_F(imageOutInnerY + top, imageOutSizeY);
}

template <>
void convolution_f<7>(
    int imageInSizeX,
    int imageInSizeY,
    int inWidthStride,
    float *imageIn,
    const float *filter,
    int outWidthStride,
    float *imageOut,
    int cn,
    const float *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    BorderType border_type)
{
    const int filterSize     = 7;
    const int imageOutInnerX = imageInSizeX - 6 * 2;
    const int imageOutInnerY = imageInSizeY - 6 * 2;

    const int imageOutSizeY = imageInSizeY - 6;
    const int imageOutSizeX = imageInSizeX - 6;

    int left  = filterSize / 2;
    int right = filterSize / 2;

    const int top    = filterSize / 2;
    const int bottom = filterSize / 2;

    std::vector<int> tab((imageInSizeX - srcWidth) * cn);
    std::vector<int> table((left + right) * 3 * cn);

    maketable(tab, table, left, right, imageInSizeX, srcWidth, cn, border_type);
    left *= cn;
    right *= cn;
    srcWidth *= cn;
    int x, y;
    // //make top border
    copytopborder<float>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, left, right, tab, border_type);
    //filter for top border
    FILTER_F(0, top);
    //filter for inner
    for (y = top; y < (imageOutInnerY) / 4 * 4 + top; y += 4) {
        int y0 = y, y1 = y + 1, y2 = y + 2, y3 = y + 3;

        // left
        for (x = 0; x < left; x++) {
            // four rows
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    auto src_start = (fy + y0 - top) * srcWidthStride + table[x + fx * cn];
                    float f        = filter[fx + fy * 7];
                    sum0 += f * src[src_start];
                    src_start += srcWidthStride;

                    sum1 += f * src[src_start];
                    src_start += srcWidthStride;

                    sum2 += f * src[src_start];
                    src_start += srcWidthStride;

                    sum3 += f * src[src_start];
                }
            }
            imageOut[x + y0 * outWidthStride] = (sum0);
            imageOut[x + y1 * outWidthStride] = (sum1);
            imageOut[x + y2 * outWidthStride] = (sum2);
            imageOut[x + y3 * outWidthStride] = (sum3);
        }
        // inner

        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 result_vec0_0 = _mm256_setzero_ps();
            __m256 result_vec0_1 = _mm256_setzero_ps();
            __m256 result_vec1_0 = _mm256_setzero_ps();
            __m256 result_vec1_1 = _mm256_setzero_ps();
            __m256 result_vec2_0 = _mm256_setzero_ps();
            __m256 result_vec2_1 = _mm256_setzero_ps();
            __m256 result_vec3_0 = _mm256_setzero_ps();
            __m256 result_vec3_1 = _mm256_setzero_ps();

            for (int fx = 0; fx < 7; fx++) {
                auto image_start   = src + srcWidthStride * (y - top) + (x - left) + fx * cn;
                auto filter_start  = filter + fx;
                __m256 kernel_vec0 = _mm256_broadcast_ss(filter_start);
                // row0  computation starts
                __m256 acc_vec0    = _mm256_loadu_ps(image_start);
                __m256 acc_vec1    = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec0_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec0_1);

                // row1 computation starts
                __m256 kernel_vec1 = _mm256_broadcast_ss(filter_start);
                acc_vec0           = _mm256_loadu_ps(image_start);
                acc_vec1           = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec1_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec1_1);

                // row2 computation starts
                __m256 kernel_vec2 = _mm256_broadcast_ss(filter_start);
                acc_vec0           = _mm256_loadu_ps(image_start);
                acc_vec1           = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec2_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec2_1);

                // row3 computation starts
                __m256 kernel_vec3 = _mm256_broadcast_ss(filter_start);
                acc_vec0           = _mm256_loadu_ps(image_start);
                acc_vec1           = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec0, acc_vec1, result_vec3_1);

                // row4 computation starts
                __m256 kernel_vec4 = _mm256_broadcast_ss(filter_start);
                acc_vec0           = _mm256_loadu_ps(image_start);
                acc_vec1           = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec1, acc_vec1, result_vec3_1);

                // row5 computation starts
                __m256 kernel_vec5 = _mm256_broadcast_ss(filter_start);
                acc_vec0           = _mm256_loadu_ps(image_start);
                acc_vec1           = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                filter_start += 7;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec5, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec5, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec2, acc_vec1, result_vec3_1);

                // row6 computation starts
                __m256 kernel_vec6 = _mm256_broadcast_ss(filter_start);
                acc_vec0           = _mm256_loadu_ps(image_start);
                acc_vec1           = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                result_vec0_0 = _mm256_fmadd_ps(kernel_vec6, acc_vec0, result_vec0_0);
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec5, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec3_0);

                result_vec0_1 = _mm256_fmadd_ps(kernel_vec6, acc_vec1, result_vec0_1);
                result_vec1_1 = _mm256_fmadd_ps(kernel_vec5, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec3, acc_vec1, result_vec3_1);
                // result0 available

                // result7 computation
                acc_vec0 = _mm256_loadu_ps(image_start);
                acc_vec1 = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                result_vec1_0 = _mm256_fmadd_ps(kernel_vec6, acc_vec0, result_vec1_0);
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec5, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec3_0);

                result_vec1_1 = _mm256_fmadd_ps(kernel_vec6, acc_vec1, result_vec1_1);
                result_vec2_1 = _mm256_fmadd_ps(kernel_vec5, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec4, acc_vec1, result_vec3_1);
                // result1 available

                acc_vec0 = _mm256_loadu_ps(image_start);
                acc_vec1 = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                result_vec2_0 = _mm256_fmadd_ps(kernel_vec6, acc_vec0, result_vec2_0);
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec5, acc_vec0, result_vec3_0);

                result_vec2_1 = _mm256_fmadd_ps(kernel_vec6, acc_vec1, result_vec2_1);
                result_vec3_1 = _mm256_fmadd_ps(kernel_vec5, acc_vec1, result_vec3_1);
                // result2 available

                acc_vec0 = _mm256_loadu_ps(image_start);
                acc_vec1 = _mm256_loadu_ps(image_start + 8);
                image_start += srcWidthStride;
                result_vec3_0 = _mm256_fmadd_ps(kernel_vec6, acc_vec0, result_vec3_0);

                result_vec3_1 = _mm256_fmadd_ps(kernel_vec6, acc_vec1, result_vec3_1);
            }
            const auto out_start = imageOut + x + y0 * outWidthStride;
            _mm256_storeu_ps(out_start, result_vec0_0);
            _mm256_storeu_ps(out_start + 8, result_vec0_1);
            _mm256_storeu_ps(out_start + outWidthStride, result_vec1_0);
            _mm256_storeu_ps(out_start + 8 + outWidthStride, result_vec1_1);
            _mm256_storeu_ps(out_start + 2 * outWidthStride, result_vec2_0);
            _mm256_storeu_ps(out_start + 8 + 2 * outWidthStride, result_vec2_1);
            _mm256_storeu_ps(out_start + 3 * outWidthStride, result_vec3_0);
            _mm256_storeu_ps(out_start + 8 + 3 * outWidthStride, result_vec3_1);
        }
        for (x = std::max(0, left + imageOutInnerX * cn - 16); x < left + imageOutInnerX * cn; x++) {
            // four rows
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    float f = filter[fx + fy * 7];

                    sum0 += f * src[(fy + y0 - top) * srcWidthStride + x - left + fx * cn];
                    sum1 += f * src[(fy + y1 - top) * srcWidthStride + x - left + fx * cn];
                    sum2 += f * src[(fy + y2 - top) * srcWidthStride + x - left + fx * cn];
                    sum3 += f * src[(fy + y3 - top) * srcWidthStride + x - left + fx * cn];
                }
            }
            imageOut[x + y0 * outWidthStride] = (sum0);
            imageOut[x + y1 * outWidthStride] = (sum1);
            imageOut[x + y2 * outWidthStride] = (sum2);
            imageOut[x + y3 * outWidthStride] = (sum3);
        }
        //right

        for (; x < imageOutSizeX * cn; x++) {
            // four rows
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    auto src_start = (fy + y0 - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    float f        = filter[fx + fy * 7];
                    sum0 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum1 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum2 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum3 += f * src[src_start];
                }
            }
            imageOut[x + y0 * outWidthStride] = (sum0);
            imageOut[x + y1 * outWidthStride] = (sum1);
            imageOut[x + y2 * outWidthStride] = (sum2);
            imageOut[x + y3 * outWidthStride] = (sum3);
        }
    }
    for (y = (imageOutInnerY) / 4 * 4 + top; y < imageOutInnerY + top; y++) {
        //for left
        for (x = 0; x < left; x++) {
            float sum = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    float f        = filter[fx + fy * 7];
                    auto src_start = (fy + y - top) * srcWidthStride + table[x + fx * cn];
                    sum += f * src[src_start];
                }
            }
            const auto out_start = x + y * outWidthStride;
            imageOut[out_start]  = ((sum));
        }
        //inner
        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_vec = _mm256_setzero_ps();
            __m256 accumulator1_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * 7);
                    __m256 data0_f32_vec  = _mm256_loadu_ps(src + x - left + fx * cn + (fy + y - top) * srcWidthStride);
                    __m256 data1_f32_vec  = _mm256_loadu_ps(src + x + 8 + fx * cn - left + (fy + y - top) * srcWidthStride);
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);
                }
            }
            _mm256_storeu_ps(imageOut + x + y * outWidthStride, accumulator0_vec);
            _mm256_storeu_ps(imageOut + x + 8 + y * outWidthStride, accumulator1_vec);
        }
        for (; x < left + imageOutInnerX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    float f = filter[fx + fy * 7];
                    sum += f * src[(fy + y - top) * srcWidthStride + x - left + fx * cn];
                }
            }
            imageOut[x + y * outWidthStride] = (sum);
        }

        //for right
        for (; x < imageOutSizeX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < 7; fx++) {
                for (int fy = 0; fy < 7; fy++) {
                    float f        = filter[fx + fy * 7];
                    auto src_start = (fy + y - top) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    sum += f * src[src_start];
                }
            }
            const auto out_start = x + y * outWidthStride;
            imageOut[out_start]  = (sum);
        }
    }
    copybottomborder<float>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, bottom, left, right, tab, border_type);
    //filter for bottom border
    FILTER_F(imageOutInnerY + top, imageOutSizeY);
}

template <>
void convolution_f<5>(
    int imageInSizeX,
    int imageInSizeY,
    int inWidthStride,
    float *imageIn,
    const float *filter,
    int outWidthStride,
    float *imageOut,
    int cn,
    const float *src,
    int srcHeight,
    int srcWidth,
    int srcWidthStride,
    BorderType border_type)
{
    //tear imageOut down into two parts: inner part with no respect to border; border part with respect to  border(left and right)
    //we store indices in vector<int> table of border for left and right
    const int imageOutInnerX = imageInSizeX - 4 * 2;
    const int imageOutInnerY = imageInSizeY - 4 * 2;
    const int imageOutSizeY  = imageInSizeY - 4;
    const int imageOutSizeX  = imageInSizeX - 4;
    const int filterSize     = 5;

    int left  = 2;
    int right = 2;

    const int top    = 2;
    const int bottom = 2;

    std::vector<int> tab((imageInSizeX - srcWidth) * cn);
    std::vector<int> table((left + right) * 3 * cn);
    maketable(tab, table, left, right, imageInSizeX, srcWidth, cn, border_type);

    left *= cn;
    right *= cn;
    srcWidth *= cn;

    int y, x;
    copytopborder<float>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, left, right, tab, border_type);
    // filter for top border
    FILTER_F(0, top);
    // filter for inner
    for (y = top; y < (imageOutInnerY) / 8 * 8 + top; y += 8) {
        int y0 = y, y1 = y + 1, y2 = y + 2, y3 = y + 3, y4 = y + 4, y5 = y + 5, y6 = y + 6, y7 = y + 7;
        // perform filter
        // left side
        for (x = 0; x < left; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            for (int fx = 0; fx < 5; fx++) {
                for (int fy = 0; fy < 5; fy++) {
                    float f        = filter[fx + fy * 5];
                    auto src_start = (fy + y0 - 2) * srcWidthStride + table[x + fx * cn];
                    sum0 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum1 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum2 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum3 += f * src[src_start];
                    src_start += srcWidthStride;

                    sum4 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum5 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum6 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum7 += f * src[src_start];
                }
            }

            auto out_start      = x + y0 * outWidthStride;
            imageOut[out_start] = ((sum0));

            out_start += outWidthStride;

            imageOut[out_start] = ((sum1));
            out_start += outWidthStride;

            imageOut[out_start] = ((sum2));
            out_start += outWidthStride;

            imageOut[out_start] = ((sum3));
            out_start += outWidthStride;

            imageOut[out_start] = ((sum4));
            out_start += outWidthStride;

            imageOut[out_start] = ((sum5));
            out_start += outWidthStride;

            imageOut[out_start] = ((sum6));
            out_start += outWidthStride;

            imageOut[out_start] = ((sum7));
        }

        for (x = left; x <= left + imageOutInnerX * cn - 8; x += 8) {
            __m256 result_vec0 = _mm256_setzero_ps();
            __m256 result_vec1 = _mm256_setzero_ps();
            __m256 result_vec2 = _mm256_setzero_ps();
            __m256 result_vec3 = _mm256_setzero_ps();
            __m256 result_vec4 = _mm256_setzero_ps();
            __m256 result_vec5 = _mm256_setzero_ps();
            __m256 result_vec6 = _mm256_setzero_ps();
            __m256 result_vec7 = _mm256_setzero_ps();

            for (int fx = 0; fx < 5; fx++) {
                auto image_start   = src + srcWidthStride * (y - 2) + (x - left) + fx * cn;
                __m256 kernel_vec0 = _mm256_broadcast_ss(filter + fx);

                // result0  computation starts
                __m256 acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec0 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec0);

                __m256 kernel_vec1 = _mm256_broadcast_ss(filter + 5 + fx);
                // result1 computation starts
                acc_vec0           = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec0 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec0);
                result_vec1 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec1);

                __m256 kernel_vec2 = _mm256_broadcast_ss(filter + 10 + fx);
                // result2 computation starts
                acc_vec0           = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec0 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec0);
                result_vec1 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec1);
                result_vec2 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec2);

                __m256 kernel_vec3 = _mm256_broadcast_ss(filter + 15 + fx);
                // result3 computation starts
                acc_vec0           = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec0 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec0);
                result_vec1 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec1);
                result_vec2 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec2);
                result_vec3 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec3);

                __m256 kernel_vec4 = _mm256_broadcast_ss(filter + 20 + fx);
                // result4 computation starts
                acc_vec0           = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec0 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec0);
                result_vec1 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec1);
                result_vec2 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec2);
                result_vec3 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec3);
                result_vec4 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec4);
                // result0 available

                // result5 computation starts
                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec1 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec1);
                result_vec2 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec2);
                result_vec3 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec3);
                result_vec4 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec4);
                result_vec5 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec5);
                // result1 available

                // result6 computation starts
                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec2 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec2);
                result_vec3 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec3);
                result_vec4 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec4);
                result_vec5 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec5);
                result_vec6 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec6);
                // result2 available
                // result7 computation
                acc_vec0    = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec3 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec3);
                result_vec4 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec4);
                result_vec5 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec5);
                result_vec6 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec6);
                result_vec7 = _mm256_fmadd_ps(kernel_vec0, acc_vec0, result_vec7);
                // result3 available

                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec4 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec4);
                result_vec5 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec5);
                result_vec6 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec6);
                result_vec7 = _mm256_fmadd_ps(kernel_vec1, acc_vec0, result_vec7);
                // result4 available

                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec5 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec5);
                result_vec6 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec6);
                result_vec7 = _mm256_fmadd_ps(kernel_vec2, acc_vec0, result_vec7);
                // result5 available

                acc_vec0 = _mm256_loadu_ps(image_start);
                image_start += srcWidthStride;
                result_vec6 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec6);
                result_vec7 = _mm256_fmadd_ps(kernel_vec3, acc_vec0, result_vec7);
                // result6 available

                acc_vec0    = _mm256_loadu_ps(image_start);
                result_vec7 = _mm256_fmadd_ps(kernel_vec4, acc_vec0, result_vec7);
                // result7 available
            }
            const auto out_start = imageOut + x + y * outWidthStride;
            _mm256_storeu_ps(out_start, result_vec0);
            _mm256_storeu_ps(out_start + outWidthStride, result_vec1);
            _mm256_storeu_ps(out_start + 2 * outWidthStride, result_vec2);
            _mm256_storeu_ps(out_start + 3 * outWidthStride, result_vec3);
            _mm256_storeu_ps(out_start + 4 * outWidthStride, result_vec4);
            _mm256_storeu_ps(out_start + 5 * outWidthStride, result_vec5);
            _mm256_storeu_ps(out_start + 6 * outWidthStride, result_vec6);
            _mm256_storeu_ps(out_start + 7 * outWidthStride, result_vec7);
        }

        for (; x < left + imageOutInnerX * cn; x++) {
            //  four rows
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            for (int fx = 0; fx < 5; fx++) {
                for (int fy = 0; fy < 5; fy++) {
                    float f = filter[fx + fy * 5];
                    sum0 += f * src[(fy + y0 - 2) * srcWidthStride + x - left + fx * cn];
                    sum1 += f * src[(fy + y1 - 2) * srcWidthStride + x - left + fx * cn];
                    sum2 += f * src[(fy + y2 - 2) * srcWidthStride + x - left + fx * cn];
                    sum3 += f * src[(fy + y3 - 2) * srcWidthStride + x - left + fx * cn];

                    sum4 += f * src[(fy + y4 - 2) * srcWidthStride + x - left + fx * cn];
                    sum5 += f * src[(fy + y5 - 2) * srcWidthStride + x - left + fx * cn];
                    sum6 += f * src[(fy + y6 - 2) * srcWidthStride + x - left + fx * cn];
                    sum7 += f * src[(fy + y7 - 2) * srcWidthStride + x - left + fx * cn];
                }
            }
            const auto out_start = x + y0 * outWidthStride;

            imageOut[out_start]                      = ((sum0));
            imageOut[out_start + outWidthStride]     = ((sum1));
            imageOut[out_start + 2 * outWidthStride] = ((sum2));
            imageOut[out_start + 3 * outWidthStride] = ((sum3));

            imageOut[out_start + 4 * outWidthStride] = ((sum4));
            imageOut[out_start + 5 * outWidthStride] = ((sum5));
            imageOut[out_start + 6 * outWidthStride] = ((sum6));
            imageOut[out_start + 7 * outWidthStride] = ((sum7));
        }

        // right side
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            for (int fx = 0; fx < 5; fx++) {
                for (int fy = 0; fy < 5; fy++) {
                    float f        = filter[fx + fy * 5];
                    auto src_start = (fy + y0 - 2) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    sum0 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum1 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum2 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum3 += f * src[src_start];
                    src_start += srcWidthStride;

                    sum4 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum5 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum6 += f * src[src_start];
                    src_start += srcWidthStride;
                    sum7 += f * src[src_start];
                }
            }
            const auto out_start = x + y0 * outWidthStride;

            imageOut[out_start]                      = ((sum0));
            imageOut[out_start + outWidthStride]     = ((sum1));
            imageOut[out_start + 2 * outWidthStride] = ((sum2));
            imageOut[out_start + 3 * outWidthStride] = ((sum3));

            imageOut[out_start + 4 * outWidthStride] = ((sum4));
            imageOut[out_start + 5 * outWidthStride] = ((sum5));
            imageOut[out_start + 6 * outWidthStride] = ((sum6));
            imageOut[out_start + 7 * outWidthStride] = ((sum7));
        }
    }
    for (y = (imageOutInnerY) / 8 * 8 + top; y < imageOutInnerY + top; y++) {
        // left side
        for (x = 0; x < left; x++) {
            float sum = 0;
            for (int fx = 0; fx < 5; fx++) {
                for (int fy = 0; fy < 5; fy++) {
                    float f        = filter[fx + fy * 5];
                    auto src_start = (fy + y - 2) * srcWidthStride + table[x + fx * cn];
                    sum += f * src[src_start];
                }
            }
            const auto out_start = x + y * outWidthStride;
            imageOut[out_start]  = ((sum));
        }

        for (x = left; x <= left + imageOutInnerX * cn - 16; x += 16) {
            __m256 accumulator0_vec = _mm256_setzero_ps();
            __m256 accumulator1_vec = _mm256_setzero_ps();
            for (int fx = 0; fx < 5; fx++) {
                for (int fy = 0; fy < 5; fy++) {
                    __m256 filter_f32_vec = _mm256_broadcast_ss(filter + fx + fy * 5);
                    __m256 data0_f32_vec  = _mm256_loadu_ps(src + x - 2 * cn + fx * cn + (fy + y - 2) * srcWidthStride);
                    __m256 data1_f32_vec  = _mm256_loadu_ps(src + x + 8 + fx * cn - 2 * cn + (fy + y - 2) * srcWidthStride);
                    accumulator0_vec      = _mm256_fmadd_ps(filter_f32_vec, data0_f32_vec, accumulator0_vec);
                    accumulator1_vec      = _mm256_fmadd_ps(filter_f32_vec, data1_f32_vec, accumulator1_vec);
                }
            }
            _mm256_storeu_ps(imageOut + x + y * outWidthStride, accumulator0_vec);
            _mm256_storeu_ps(imageOut + x + 8 + y * outWidthStride, accumulator1_vec);
        }

        for (; x < left + imageOutInnerX * cn; x++) {
            float sum = 0;
            for (int fx = 0; fx < 5; fx++) {
                for (int fy = 0; fy < 5; fy++) {
                    float f = filter[fx + fy * 5];
                    sum += f * src[(fy + y - 2) * srcWidthStride + x - 2 * cn + fx * cn];
                }
            }
            imageOut[x + y * outWidthStride] = ((sum));
        }
        //right side
        for (; x < imageOutSizeX * cn; x++) {
            float sum0 = 0;

            for (int fx = 0; fx < 5; fx++) {
                for (int fy = 0; fy < 5; fy++) {
                    float f        = filter[fx + fy * 5];
                    auto src_start = (fy + y - 2) * srcWidthStride + table[x - imageOutInnerX * cn + 2 * left + fx * cn];
                    sum0 += f * src[src_start];
                }
            }
            const auto out_start = x + y * outWidthStride;
            imageOut[out_start]  = ((sum0));
        }
    }
    // prepare data for bottom border
    copybottomborder<float>(imageIn, imageInSizeX, src, srcHeight, srcWidth, srcWidthStride, inWidthStride, cn, top, bottom, left, right, tab, border_type);
    //filter for bottom border
    FILTER_F(imageOutInnerY + top, imageOutSizeY);
}

}
}
}
} // namespace ppl::cv::x86::fma
