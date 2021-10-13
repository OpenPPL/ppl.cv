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

#include "ppl/cv/x86/resize.h"

#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"

#include <string.h>
#include <limits.h>
#include <immintrin.h>
#include <float.h>
#include <stdint.h>
#include <cmath>
#include <stdlib.h>
#include <algorithm>

#include "ppl/cv/x86/fma/internal_fma.hpp"

namespace ppl {
namespace cv {
namespace x86 {

#define INTER_RESIZE_COEF_BITS  (11)
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)

static inline int32_t resize_img_floor(float a)
{
    return (((a) >= 0) ? ((int32_t)a) : ((int32_t)a - 1));
}

static inline int32_t resize_img_round(float value)
{
    double intpart, fractpart;
    fractpart = modf(value, &intpart);
    if (fabs(fractpart) != 0.5f || ((((int32_t)intpart) % 2) != 0)) {
        return (int32_t)(value + (value >= 0 ? 0.5f : -0.5f));
    } else {
        return (int32_t)intpart;
    }
}

static inline int16_t resize_img_saturate_cast_short(float x)
{
    int32_t iv = resize_img_round(x);

    return (iv > SHRT_MIN ? (iv < SHRT_MAX ? iv : SHRT_MAX) : SHRT_MIN);
}

static void resize_linear_calc_offset_u8(
    int32_t inHeight,
    int32_t inWidth,
    int32_t channels,
    int32_t outHeight,
    int32_t outWidth,
    int32_t &w_max,
    int32_t *h_offset,
    int32_t *w_offset,
    int16_t *h_coeff,
    int16_t *w_coeff)
{
    double inv_scale_h = (double)outHeight / inHeight;
    double scale_h     = 1.0 / inv_scale_h;

    for (int32_t h = 0; h < outHeight; ++h) {
        float float_h = (h + 0.5) * scale_h - 0.5;
        int32_t int_h = resize_img_floor(float_h);
        float_h -= int_h;

        h_offset[h] = int_h;
        h_coeff[h]  = resize_img_saturate_cast_short((1.0f - float_h) * INTER_RESIZE_COEF_SCALE);
    }

    double inv_scale_w = (double)outWidth / inWidth;
    double scale_w     = 1.0 / inv_scale_w;

    w_max = 0;
    for (int32_t w = 0; w < outWidth; ++w) {
        float float_w = (w + 0.5) * scale_w - 0.5;
        int32_t int_w = resize_img_floor(float_w);
        float_w -= int_w;

        if (int_w < 0) {
            int_w   = 0;
            float_w = 0;
        }
        if (int_w + 1 >= inWidth) {
            float_w = 0;
            int_w   = inWidth - 1;
        }
        if (int_w <= inWidth - 2) {
            w_max = w;
        }

        w_offset[w] = int_w * channels;
        for (int32_t c = 0; c < channels; ++c) {
            w_coeff[(w * channels + c) * 2 + 0] = resize_img_saturate_cast_short((1.0f - float_w) * INTER_RESIZE_COEF_SCALE);
            w_coeff[(w * channels + c) * 2 + 1] = resize_img_saturate_cast_short(float_w * INTER_RESIZE_COEF_SCALE);
        }
    }
}

static void resize_linear_w_oneline_u8(
    int32_t inWidth,
    int32_t outWidth,
    int32_t channels,
    const uint8_t *inData,
    int32_t w_max,
    const int32_t *w_offset,
    const int16_t *w_coeff,
    int32_t *row)
{
    int32_t i = 0;

    if (1 == channels &&
        ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        i = fma::resize_linear_w_oneline_c1_u8_fma(inWidth, inData, outWidth, w_offset, w_coeff, INTER_RESIZE_COEF_SCALE, row);
    }
    if (3 == channels &&
        ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        i = fma::resize_linear_w_oneline_c3_u8_fma(inWidth, inData, outWidth, w_offset, w_coeff, INTER_RESIZE_COEF_SCALE, row);
    }
    if (4 == channels &&
        ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        i = fma::resize_linear_w_oneline_c4_u8_fma(inWidth, inData, outWidth, w_offset, w_coeff, INTER_RESIZE_COEF_SCALE, row);
    }

    for (; i < w_max; ++i) {
        for (int32_t j = 0; j < channels; ++j) {
            int32_t w_idx_0   = w_offset[i] + j;
            int32_t w_idx_1   = w_idx_0 + channels;
            int16_t coeff_0   = w_coeff[(i * channels + j) * 2 + 0];
            int16_t coeff_1   = w_coeff[(i * channels + j) * 2 + 1];
            int32_t row_value = inData[w_idx_0] * coeff_0 +
                                inData[w_idx_1] * coeff_1;
            row[i * channels + j] = row_value >> 4;
        }
    }

    for (; i < outWidth; ++i) {
        for (int32_t j = 0; j < channels; ++j) {
            int32_t w_idx_0   = w_offset[i] + j;
            int32_t w_idx_1   = w_idx_0 >= (inWidth - 1) * channels ? w_idx_0 : w_idx_0 + channels;
            int16_t coeff_0   = w_coeff[(i * channels + j) * 2 + 0];
            int16_t coeff_1   = w_coeff[(i * channels + j) * 2 + 1];
            int32_t row_value = inData[w_idx_0] * coeff_0 +
                                inData[w_idx_1] * coeff_1;
            row[i * channels + j] = row_value >> 4;
        }
    }
}

static void resize_linear_h_u8(
    int32_t outWidth,
    int32_t channels,
    const int32_t *row_0,
    const int32_t *row_1,
    int32_t h_idx,
    int16_t h_coeff,
    uint8_t *outData)
{
    int32_t i         = 0;
    int16_t h_coeff_1 = INTER_RESIZE_COEF_SCALE - h_coeff;

    __m128i m_h_coeff_0 = _mm_set1_epi16(h_coeff);
    __m128i m_h_coeff_1 = _mm_set1_epi16(h_coeff_1);
    __m128i m_epi16_two = _mm_set1_epi16(2);

    for (; i <= outWidth * channels - 16; i += 16) {
        __m128i m_data_row_0_0 = _mm_load_si128((const __m128i *)(row_0 + i + 0));
        __m128i m_data_row_0_1 = _mm_load_si128((const __m128i *)(row_0 + i + 4));
        __m128i m_data_row_0_2 = _mm_load_si128((const __m128i *)(row_0 + i + 8));
        __m128i m_data_row_0_3 = _mm_load_si128((const __m128i *)(row_0 + i + 12));
        __m128i m_data_row_1_0 = _mm_load_si128((const __m128i *)(row_1 + i + 0));
        __m128i m_data_row_1_1 = _mm_load_si128((const __m128i *)(row_1 + i + 4));
        __m128i m_data_row_1_2 = _mm_load_si128((const __m128i *)(row_1 + i + 8));
        __m128i m_data_row_1_3 = _mm_load_si128((const __m128i *)(row_1 + i + 12));

        __m128i m_data_row_0_01 = _mm_packs_epi32(m_data_row_0_0, m_data_row_0_1);
        __m128i m_data_row_0_23 = _mm_packs_epi32(m_data_row_0_2, m_data_row_0_3);
        __m128i m_data_row_1_01 = _mm_packs_epi32(m_data_row_1_0, m_data_row_1_1);
        __m128i m_data_row_1_23 = _mm_packs_epi32(m_data_row_1_2, m_data_row_1_3);

        __m128i m_rst_01 = _mm_adds_epi16(_mm_mulhi_epi16(m_data_row_0_01, m_h_coeff_0),
                                          _mm_mulhi_epi16(m_data_row_1_01, m_h_coeff_1));
        __m128i m_rst_23 = _mm_adds_epi16(_mm_mulhi_epi16(m_data_row_0_23, m_h_coeff_0),
                                          _mm_mulhi_epi16(m_data_row_1_23, m_h_coeff_1));
        m_rst_01         = _mm_srai_epi16(_mm_adds_epi16(m_rst_01, m_epi16_two), 2);
        m_rst_23         = _mm_srai_epi16(_mm_adds_epi16(m_rst_23, m_epi16_two), 2);
        _mm_storeu_si128((__m128i *)(outData + i), _mm_packus_epi16(m_rst_01, m_rst_23));
    }

    for (; i < outWidth * channels; ++i) {
        int32_t rst_value = (((h_coeff * row_0[i]) >> 16) +
                             ((h_coeff_1 * row_1[i]) >> 16) + 2) >>
                            2;
        outData[i] = rst_value;
    }
}

static void resize_linear_kernel_u8(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t channels,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t *outData)
{
    int32_t cn_width           = channels * outWidth;
    uint64_t size_for_h_offset = (outHeight * sizeof(int32_t) + 128 - 1) / 128 * 128;
    uint64_t size_for_w_offset = (cn_width * sizeof(int32_t) + 128 - 1) / 128 * 128;
    uint64_t size_for_h_coeff  = (outHeight * sizeof(int16_t) * 2 + 128 - 1) / 128 * 128;
    uint64_t size_for_w_coeff  = (cn_width * sizeof(int16_t) * 2 + 128 - 1) / 128 * 128;
    uint64_t size_for_row_0    = (cn_width * sizeof(int32_t) + 128 - 1) / 128 * 128;
    uint64_t size_for_row_1    = (cn_width * sizeof(int32_t) + 128 - 1) / 128 * 128;

    uint64_t total_size = size_for_h_offset + size_for_w_offset + size_for_h_coeff + size_for_w_coeff + size_for_row_0 + size_for_row_1;

    void *temp_buffer = ppl::common::AlignedAlloc(total_size, 128);
    int32_t *h_offset = (int32_t *)temp_buffer;
    int32_t *w_offset = (int32_t *)((unsigned char *)h_offset + size_for_h_offset);
    int16_t *h_coeff  = (int16_t *)((unsigned char *)w_offset + size_for_w_offset);
    int16_t *w_coeff  = (int16_t *)((unsigned char *)h_coeff + size_for_h_coeff);
    int32_t *row_0    = (int32_t *)((unsigned char *)w_coeff + size_for_w_coeff);
    int32_t *row_1    = (int32_t *)((unsigned char *)row_0 + size_for_row_0);

    int32_t w_max = 0;
    resize_linear_calc_offset_u8(inHeight, inWidth, channels, outHeight, outWidth, w_max, h_offset, w_offset, h_coeff, w_coeff);

    if (1 == channels &&
        inHeight > outHeight &&
        ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        fma::resize_linear_kernel_c1_shrink_u8_fma(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, h_offset, w_offset, h_coeff, w_coeff, INTER_RESIZE_COEF_SCALE, outData);

        ppl::common::AlignedFree(temp_buffer);
        return;
    }

    int32_t h = 0;

    int32_t prev_h[2]    = {-1, -1};
    int32_t *prev_ptr[2] = {nullptr, nullptr};

    int32_t reuse_count;
    int32_t *row_ptr[2];

    for (; h < outHeight; ++h) {
        reuse_count = 0;
        row_ptr[0]  = nullptr;
        row_ptr[1]  = nullptr;

        int32_t src_h_idx_0 = h_offset[h];
        int32_t src_h_idx_1 = src_h_idx_0 == inHeight - 1 ? inHeight - 1 : src_h_idx_0 + 1;
        if (src_h_idx_0 < 0) {
            src_h_idx_0 = 0;
        }

        if (src_h_idx_0 == prev_h[0]) {
            reuse_count++;
            row_ptr[0] = prev_ptr[0];

            if (src_h_idx_1 == prev_h[0]) {
                reuse_count++;
                row_ptr[1] = prev_ptr[0];
            } else if (src_h_idx_1 == prev_h[1]) {
                reuse_count++;
                row_ptr[1] = prev_ptr[1];
            }
        } else if (src_h_idx_0 == prev_h[1]) {
            reuse_count++;
            row_ptr[0] = prev_ptr[1];

            if (src_h_idx_1 == prev_h[1]) {
                reuse_count++;
                row_ptr[1] = prev_ptr[1];
            }
        }

        if (reuse_count == 0) {
            row_ptr[0] = row_0;
            row_ptr[1] = row_1;

            resize_linear_w_oneline_u8(inWidth, outWidth, channels, inData + src_h_idx_0 * inWidthStride, w_max, w_offset, w_coeff, row_ptr[0]);
            resize_linear_w_oneline_u8(inWidth, outWidth, channels, inData + src_h_idx_1 * inWidthStride, w_max, w_offset, w_coeff, row_ptr[1]);
        } else {
            if (reuse_count == 1) {
                if (row_ptr[0] == row_0) {
                    row_ptr[1] = row_1;
                } else {
                    row_ptr[1] = row_0;
                }
                resize_linear_w_oneline_u8(inWidth, outWidth, channels, inData + src_h_idx_1 * inWidthStride, w_max, w_offset, w_coeff, row_ptr[1]);
            }
        }
        resize_linear_h_u8(outWidth, channels, row_ptr[0], row_ptr[1], h_offset[h], h_coeff[h], outData + h * outWidthStride);

        prev_h[0]   = src_h_idx_0;
        prev_h[1]   = src_h_idx_1;
        prev_ptr[0] = row_ptr[0];
        prev_ptr[1] = row_ptr[1];
    }
    ppl::common::AlignedFree(temp_buffer);
}

static void resize_linear_shrink2_c1_kernel_u8(
    const uint8_t *inData,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t *outData)
{
    __m128i m_zero      = _mm_set1_epi8(0);
    __m128i m_epi16_two = _mm_set1_epi16(2);
    for (int32_t h = 0; h < outHeight; ++h) {
        int32_t w = 0;

        if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
            w = fma::resize_linear_shrink2_oneline_c1_kernel_u8_fma(inData + h * 2 * inWidthStride, inWidthStride, outWidth, outData + h * outWidthStride);
        }
        for (; w <= outWidth - 16; w += 16) {
            __m128i m_data[4];
            m_data[0] = _mm_loadu_si128((const __m128i *)(inData + (h * 2 + 0) * inWidthStride + w * 2 + 0));
            m_data[1] = _mm_loadu_si128((const __m128i *)(inData + (h * 2 + 0) * inWidthStride + w * 2 + 16));
            m_data[2] = _mm_loadu_si128((const __m128i *)(inData + (h * 2 + 1) * inWidthStride + w * 2 + 0));
            m_data[3] = _mm_loadu_si128((const __m128i *)(inData + (h * 2 + 1) * inWidthStride + w * 2 + 16));

            __m128i m_data_s16[8];
            m_data_s16[0] = _mm_unpacklo_epi8(m_data[0], m_zero);
            m_data_s16[1] = _mm_unpackhi_epi8(m_data[0], m_zero);
            m_data_s16[2] = _mm_unpacklo_epi8(m_data[1], m_zero);
            m_data_s16[3] = _mm_unpackhi_epi8(m_data[1], m_zero);
            m_data_s16[4] = _mm_unpacklo_epi8(m_data[2], m_zero);
            m_data_s16[5] = _mm_unpackhi_epi8(m_data[2], m_zero);
            m_data_s16[6] = _mm_unpacklo_epi8(m_data[3], m_zero);
            m_data_s16[7] = _mm_unpackhi_epi8(m_data[3], m_zero);

            __m128i m_dst[4];
            m_dst[0] = _mm_hadd_epi16(m_data_s16[0], m_data_s16[1]);
            m_dst[1] = _mm_hadd_epi16(m_data_s16[2], m_data_s16[3]);
            m_dst[2] = _mm_hadd_epi16(m_data_s16[4], m_data_s16[5]);
            m_dst[3] = _mm_hadd_epi16(m_data_s16[6], m_data_s16[7]);

            m_dst[0] = _mm_adds_epi16(m_dst[0], m_dst[2]);
            m_dst[1] = _mm_adds_epi16(m_dst[1], m_dst[3]);

            m_dst[0] = _mm_adds_epi16(m_dst[0], m_epi16_two);
            m_dst[1] = _mm_adds_epi16(m_dst[1], m_epi16_two);

            m_dst[0] = _mm_srai_epi16(m_dst[0], 2);
            m_dst[1] = _mm_srai_epi16(m_dst[1], 2);

            m_dst[0] = _mm_packus_epi16(m_dst[0], m_dst[1]);

            _mm_storeu_si128((__m128i *)(outData + h * outWidthStride + w), m_dst[0]);
        }
        for (; w < outWidth; ++w) {
            int32_t val = inData[(h * 2 + 0) * inWidthStride + w * 2 + 0] +
                          inData[(h * 2 + 0) * inWidthStride + w * 2 + 1] +
                          inData[(h * 2 + 1) * inWidthStride + w * 2 + 0] +
                          inData[(h * 2 + 1) * inWidthStride + w * 2 + 1];
            val += 2;
            val >>= 2;
            outData[h * outWidthStride + w] = val;
        }
    }
}

static void resize_linear_shrink2_c4_kernel_u8(
    const uint8_t *inData,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t *outData)
{
    const int32_t channels = 4;

    __m128i m_zero      = _mm_set1_epi8(0);
    __m128i m_epi16_two = _mm_set1_epi16(2);
    for (int32_t h = 0; h < outHeight; ++h) {
        int32_t w = 0;

        if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
            w = fma::resize_linear_shrink2_oneline_c4_kernel_u8_fma(inData + h * 2 * inWidthStride, inWidthStride, outWidth, outData + h * outWidthStride);
        }
        for (; w <= outWidth - 4; w += 4) {
            __m128i m_data[4];
            m_data[0] = _mm_loadu_si128((const __m128i *)(inData + (h * 2 + 0) * inWidthStride + w * channels * 2 + 0));
            m_data[1] = _mm_loadu_si128((const __m128i *)(inData + (h * 2 + 0) * inWidthStride + w * channels * 2 + 16));
            m_data[2] = _mm_loadu_si128((const __m128i *)(inData + (h * 2 + 1) * inWidthStride + w * channels * 2 + 0));
            m_data[3] = _mm_loadu_si128((const __m128i *)(inData + (h * 2 + 1) * inWidthStride + w * channels * 2 + 16));

            __m128i m_data_u8_temp[4];
            m_data_u8_temp[0] = _mm_unpacklo_epi8(m_data[0], m_data[2]);
            m_data_u8_temp[1] = _mm_unpackhi_epi8(m_data[0], m_data[2]);
            m_data_u8_temp[2] = _mm_unpacklo_epi8(m_data[1], m_data[3]);
            m_data_u8_temp[3] = _mm_unpackhi_epi8(m_data[1], m_data[3]);

            __m128i m_data_s16[8];
            m_data_s16[0] = _mm_unpacklo_epi8(m_data_u8_temp[0], m_zero);
            m_data_s16[1] = _mm_unpackhi_epi8(m_data_u8_temp[0], m_zero);
            m_data_s16[2] = _mm_unpacklo_epi8(m_data_u8_temp[1], m_zero);
            m_data_s16[3] = _mm_unpackhi_epi8(m_data_u8_temp[1], m_zero);
            m_data_s16[4] = _mm_unpacklo_epi8(m_data_u8_temp[2], m_zero);
            m_data_s16[5] = _mm_unpackhi_epi8(m_data_u8_temp[2], m_zero);
            m_data_s16[6] = _mm_unpacklo_epi8(m_data_u8_temp[3], m_zero);
            m_data_s16[7] = _mm_unpackhi_epi8(m_data_u8_temp[3], m_zero);

            __m128i m_dst[2];
            m_dst[0] = _mm_adds_epi16(_mm_hadd_epi16(m_data_s16[0], m_data_s16[2]),
                                      _mm_hadd_epi16(m_data_s16[1], m_data_s16[3]));
            m_dst[1] = _mm_adds_epi16(_mm_hadd_epi16(m_data_s16[4], m_data_s16[6]),
                                      _mm_hadd_epi16(m_data_s16[5], m_data_s16[7]));

            m_dst[0] = _mm_adds_epi16(m_dst[0], m_epi16_two);
            m_dst[1] = _mm_adds_epi16(m_dst[1], m_epi16_two);

            m_dst[0] = _mm_srai_epi16(m_dst[0], 2);
            m_dst[1] = _mm_srai_epi16(m_dst[1], 2);

            m_dst[0] = _mm_packus_epi16(m_dst[0], m_dst[1]);

            _mm_storeu_si128((__m128i *)(outData + h * outWidthStride + w * channels), m_dst[0]);
        }

        for (; w < outWidth; ++w) {
            int32_t val_0 = inData[(h * 2 + 0) * inWidthStride + w * channels * 2 + 0] +
                            inData[(h * 2 + 0) * inWidthStride + w * channels * 2 + 0 + channels] +
                            inData[(h * 2 + 1) * inWidthStride + w * channels * 2 + 0] +
                            inData[(h * 2 + 1) * inWidthStride + w * channels * 2 + 0 + channels];
            int32_t val_1 = inData[(h * 2 + 0) * inWidthStride + w * channels * 2 + 1] +
                            inData[(h * 2 + 0) * inWidthStride + w * channels * 2 + 1 + channels] +
                            inData[(h * 2 + 1) * inWidthStride + w * channels * 2 + 1] +
                            inData[(h * 2 + 1) * inWidthStride + w * channels * 2 + 1 + channels];
            int32_t val_2 = inData[(h * 2 + 0) * inWidthStride + w * channels * 2 + 2] +
                            inData[(h * 2 + 0) * inWidthStride + w * channels * 2 + 2 + channels] +
                            inData[(h * 2 + 1) * inWidthStride + w * channels * 2 + 2] +
                            inData[(h * 2 + 1) * inWidthStride + w * channels * 2 + 2 + channels];
            int32_t val_3 = inData[(h * 2 + 0) * inWidthStride + w * channels * 2 + 3] +
                            inData[(h * 2 + 0) * inWidthStride + w * channels * 2 + 3 + channels] +
                            inData[(h * 2 + 1) * inWidthStride + w * channels * 2 + 3] +
                            inData[(h * 2 + 1) * inWidthStride + w * channels * 2 + 3 + channels];

            val_0 += 2;
            val_1 += 2;
            val_2 += 2;
            val_3 += 2;
            val_0 >>= 2;
            val_1 >>= 2;
            val_2 >>= 2;
            val_3 >>= 2;
            outData[h * outWidthStride + w * channels + 0] = val_0;
            outData[h * outWidthStride + w * channels + 1] = val_1;
            outData[h * outWidthStride + w * channels + 2] = val_2;
            outData[h * outWidthStride + w * channels + 3] = val_3;
        }
    }
}

template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (outHeight * 2 == inHeight && outWidth * 2 == inWidth) {
        resize_linear_shrink2_c1_kernel_u8(inData, inWidthStride, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }

    resize_linear_kernel_u8(
        inHeight, inWidth, inWidthStride, inData, 1, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    resize_linear_kernel_u8(
        inHeight, inWidth, inWidthStride, inData, 3, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (outHeight * 2 == inHeight && outWidth * 2 == inWidth) {
        resize_linear_shrink2_c4_kernel_u8(inData, inWidthStride, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }

    resize_linear_kernel_u8(
        inHeight, inWidth, inWidthStride, inData, 4, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
