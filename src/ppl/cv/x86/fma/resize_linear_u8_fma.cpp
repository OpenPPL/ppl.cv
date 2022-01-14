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

#include <immintrin.h>
#include <math.h>
#include "internal_fma.hpp"
#include "ppl/common/sys.h"
#include <stdint.h>

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

#define _mm256_extract_epi32(x, n)	\
    _mm_extract_epi32 (_mm256_extractf128_si256 ((x), (n) >> 2), (n) % 4);

int32_t resize_linear_w_oneline_c1_u8_fma(
    int32_t in_width,
    const uint8_t *in_data,
    int32_t out_width,
    const int32_t *w_offset,
    const int16_t *w_coeff,
    int16_t COEFF_SUM,
    int32_t *row)
{
    int32_t last_w = out_width;
    while (last_w > 0 && w_offset[last_w - 1] >= in_width - 3) {
        last_w--;
    }

    __m256i b0 = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13);
    const int32_t b1 = (2 << 2) + 0;

    int32_t w = 0;
    for (; w <= out_width - 32; w += 32) {
        __m256i m_data[4];
        m_data[0] = _mm256_i32gather_epi32((const int32_t *)in_data, _mm256_load_si256((const __m256i *)(w_offset + w + 0)), 1);
        m_data[1] = _mm256_i32gather_epi32((const int32_t *)in_data, _mm256_load_si256((const __m256i *)(w_offset + w + 8)), 1);
        m_data[2] = _mm256_i32gather_epi32((const int32_t *)in_data, _mm256_load_si256((const __m256i *)(w_offset + w + 16)), 1);
        m_data[3] = _mm256_i32gather_epi32((const int32_t *)in_data, _mm256_load_si256((const __m256i *)(w_offset + w + 24)), 1);

        m_data[0] = _mm256_shuffle_epi8(m_data[0], b0);
        m_data[1] = _mm256_shuffle_epi8(m_data[1], b0);
        m_data[2] = _mm256_shuffle_epi8(m_data[2], b0);
        m_data[3] = _mm256_shuffle_epi8(m_data[3], b0);

        m_data[0] = _mm256_permute4x64_epi64(m_data[0], b1);
        m_data[1] = _mm256_permute4x64_epi64(m_data[1], b1);
        m_data[2] = _mm256_permute4x64_epi64(m_data[2], b1);
        m_data[3] = _mm256_permute4x64_epi64(m_data[3], b1);

        m_data[0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[0]));
        m_data[1] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[1]));
        m_data[2] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[2]));
        m_data[3] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[3]));

        m_data[0] = _mm256_madd_epi16(m_data[0], _mm256_load_si256((const __m256i *)(w_coeff + (w + 0) * 2)));
        m_data[1] = _mm256_madd_epi16(m_data[1], _mm256_load_si256((const __m256i *)(w_coeff + (w + 8) * 2)));
        m_data[2] = _mm256_madd_epi16(m_data[2], _mm256_load_si256((const __m256i *)(w_coeff + (w + 16) * 2)));
        m_data[3] = _mm256_madd_epi16(m_data[3], _mm256_load_si256((const __m256i *)(w_coeff + (w + 24) * 2)));

        m_data[0] = _mm256_srai_epi32(m_data[0], 4);
        m_data[1] = _mm256_srai_epi32(m_data[1], 4);
        m_data[2] = _mm256_srai_epi32(m_data[2], 4);
        m_data[3] = _mm256_srai_epi32(m_data[3], 4);

        _mm256_store_si256((__m256i *)(row + w + 0), m_data[0]);
        _mm256_store_si256((__m256i *)(row + w + 8), m_data[1]);
        _mm256_store_si256((__m256i *)(row + w + 16), m_data[2]);
        _mm256_store_si256((__m256i *)(row + w + 24), m_data[3]);
    }
    return w;
}

void resize_linear_kernel_c1_shrink_u8_fma(
    int32_t in_height,
    int32_t in_width,
    int32_t in_stride,
    const uint8_t *in_data,
    int32_t out_height,
    int32_t out_width,
    int32_t out_stride,
    const int32_t *h_offset,
    const int32_t *w_offset,
    int16_t *h_coeff,
    int16_t *w_coeff,
    int16_t INTER_RESIZE_COEF_SCALE,
    uint8_t *out_data)
{
    int32_t last_w = out_width;
    while (last_w >= 0 && w_offset[last_w - 1] >= in_width - 3) {
        last_w--;
    }

    __m256i b0 = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13);
    const int32_t b1 = (2 << 2) + 0;

    __m256i m_epi16_two = _mm256_set1_epi16(2);

    int32_t h = 0;
    for (; h <= out_height - 4; h += 4) {
        int32_t h_idx[8];
        int16_t h_coeff_cur[8];
        const uint8_t *h_ptr[8];
        for (int32_t i = 0; i < 4; ++i) {
            h_idx[2 * i + 0]       = h_offset[h + i];
            h_idx[2 * i + 1]       = h_offset[h + i] == in_height - 1 ? in_height - 1 : h_idx[2 * i + 0] + 1;
            h_coeff_cur[2 * i + 0] = h_coeff[h + i];
            h_coeff_cur[2 * i + 1] = INTER_RESIZE_COEF_SCALE - h_coeff_cur[2 * i + 0];
            h_ptr[2 * i + 0]       = in_data + h_idx[2 * i + 0] * in_stride;
            h_ptr[2 * i + 1]       = in_data + h_idx[2 * i + 1] * in_stride;
        }

        __m256i m_h_coeff[4];
        for (int32_t i = 0; i < 4; ++i) {
            if (i % 2) { // 1 & 3
                m_h_coeff[i] = _mm256_setr_epi16(
                    h_coeff_cur[i * 2 - 1], h_coeff_cur[i * 2 - 1], h_coeff_cur[i * 2 - 1], h_coeff_cur[i * 2 - 1], h_coeff_cur[i * 2 + 1], h_coeff_cur[i * 2 + 1], h_coeff_cur[i * 2 + 1], h_coeff_cur[i * 2 + 1], h_coeff_cur[i * 2 - 1], h_coeff_cur[i * 2 - 1], h_coeff_cur[i * 2 - 1], h_coeff_cur[i * 2 - 1], h_coeff_cur[i * 2 + 1], h_coeff_cur[i * 2 + 1], h_coeff_cur[i * 2 + 1], h_coeff_cur[i * 2 + 1]);
            } else { // 0 & 2
                m_h_coeff[i] = _mm256_setr_epi16(
                    h_coeff_cur[i * 2 + 0], h_coeff_cur[i * 2 + 0], h_coeff_cur[i * 2 + 0], h_coeff_cur[i * 2 + 0], h_coeff_cur[i * 2 + 2], h_coeff_cur[i * 2 + 2], h_coeff_cur[i * 2 + 2], h_coeff_cur[i * 2 + 2], h_coeff_cur[i * 2 + 0], h_coeff_cur[i * 2 + 0], h_coeff_cur[i * 2 + 0], h_coeff_cur[i * 2 + 0], h_coeff_cur[i * 2 + 2], h_coeff_cur[i * 2 + 2], h_coeff_cur[i * 2 + 2], h_coeff_cur[i * 2 + 2]);
            }
        }

        int32_t w = 0;
        for (; w <= last_w - 8; w += 8) {
            __m256i m_data[8];
            __m256i m_w_offset = _mm256_load_si256((const __m256i *)(w_offset + w));
            __m256i m_w_coeff  = _mm256_load_si256((const __m256i *)(w_coeff + w * 2));
            m_data[0]          = _mm256_i32gather_epi32((const int32_t *)h_ptr[0], m_w_offset, 1);
            m_data[1]          = _mm256_i32gather_epi32((const int32_t *)h_ptr[1], m_w_offset, 1);
            m_data[2]          = _mm256_i32gather_epi32((const int32_t *)h_ptr[2], m_w_offset, 1);
            m_data[3]          = _mm256_i32gather_epi32((const int32_t *)h_ptr[3], m_w_offset, 1);
            m_data[4]          = _mm256_i32gather_epi32((const int32_t *)h_ptr[4], m_w_offset, 1);
            m_data[5]          = _mm256_i32gather_epi32((const int32_t *)h_ptr[5], m_w_offset, 1);
            m_data[6]          = _mm256_i32gather_epi32((const int32_t *)h_ptr[6], m_w_offset, 1);
            m_data[7]          = _mm256_i32gather_epi32((const int32_t *)h_ptr[7], m_w_offset, 1);

            m_data[0] = _mm256_shuffle_epi8(m_data[0], b0);
            m_data[1] = _mm256_shuffle_epi8(m_data[1], b0);
            m_data[2] = _mm256_shuffle_epi8(m_data[2], b0);
            m_data[3] = _mm256_shuffle_epi8(m_data[3], b0);
            m_data[4] = _mm256_shuffle_epi8(m_data[4], b0);
            m_data[5] = _mm256_shuffle_epi8(m_data[5], b0);
            m_data[6] = _mm256_shuffle_epi8(m_data[6], b0);
            m_data[7] = _mm256_shuffle_epi8(m_data[7], b0);

            m_data[0] = _mm256_permute4x64_epi64(m_data[0], b1);
            m_data[1] = _mm256_permute4x64_epi64(m_data[1], b1);
            m_data[2] = _mm256_permute4x64_epi64(m_data[2], b1);
            m_data[3] = _mm256_permute4x64_epi64(m_data[3], b1);
            m_data[4] = _mm256_permute4x64_epi64(m_data[4], b1);
            m_data[5] = _mm256_permute4x64_epi64(m_data[5], b1);
            m_data[6] = _mm256_permute4x64_epi64(m_data[6], b1);
            m_data[7] = _mm256_permute4x64_epi64(m_data[7], b1);

            m_data[0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[0]));
            m_data[1] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[1]));
            m_data[2] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[2]));
            m_data[3] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[3]));
            m_data[4] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[4]));
            m_data[5] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[5]));
            m_data[6] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[6]));
            m_data[7] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[7]));

            m_data[0] = _mm256_madd_epi16(m_data[0], m_w_coeff);
            m_data[1] = _mm256_madd_epi16(m_data[1], m_w_coeff);
            m_data[2] = _mm256_madd_epi16(m_data[2], m_w_coeff);
            m_data[3] = _mm256_madd_epi16(m_data[3], m_w_coeff);
            m_data[4] = _mm256_madd_epi16(m_data[4], m_w_coeff);
            m_data[5] = _mm256_madd_epi16(m_data[5], m_w_coeff);
            m_data[6] = _mm256_madd_epi16(m_data[6], m_w_coeff);
            m_data[7] = _mm256_madd_epi16(m_data[7], m_w_coeff);

            m_data[0] = _mm256_srai_epi32(m_data[0], 4);
            m_data[1] = _mm256_srai_epi32(m_data[1], 4);
            m_data[2] = _mm256_srai_epi32(m_data[2], 4);
            m_data[3] = _mm256_srai_epi32(m_data[3], 4);
            m_data[4] = _mm256_srai_epi32(m_data[4], 4);
            m_data[5] = _mm256_srai_epi32(m_data[5], 4);
            m_data[6] = _mm256_srai_epi32(m_data[6], 4);
            m_data[7] = _mm256_srai_epi32(m_data[7], 4);

            m_data[0] = _mm256_packs_epi32(m_data[0], m_data[2]);
            m_data[1] = _mm256_packs_epi32(m_data[1], m_data[3]);
            m_data[2] = _mm256_packs_epi32(m_data[4], m_data[6]);
            m_data[3] = _mm256_packs_epi32(m_data[5], m_data[7]);

            m_data[0] = _mm256_mulhi_epi16(m_data[0], m_h_coeff[0]);
            m_data[1] = _mm256_mulhi_epi16(m_data[1], m_h_coeff[1]);
            m_data[2] = _mm256_mulhi_epi16(m_data[2], m_h_coeff[2]);
            m_data[3] = _mm256_mulhi_epi16(m_data[3], m_h_coeff[3]);

            m_data[0] = _mm256_adds_epi16(m_data[0], m_data[1]);
            m_data[1] = _mm256_adds_epi16(m_data[2], m_data[3]);

            m_data[0] = _mm256_adds_epi16(m_data[0], m_epi16_two);
            m_data[1] = _mm256_adds_epi16(m_data[1], m_epi16_two);

            m_data[0] = _mm256_srai_epi16(m_data[0], 2);
            m_data[1] = _mm256_srai_epi16(m_data[1], 2);

            m_data[0] = _mm256_packus_epi16(m_data[0], m_data[1]);

            *(int32_t *)(out_data + (h + 0) * out_stride + w + 0) = _mm256_extract_epi32(m_data[0], 0);
            *(int32_t *)(out_data + (h + 0) * out_stride + w + 4) = _mm256_extract_epi32(m_data[0], 4);
            *(int32_t *)(out_data + (h + 1) * out_stride + w + 0) = _mm256_extract_epi32(m_data[0], 1);
            *(int32_t *)(out_data + (h + 1) * out_stride + w + 4) = _mm256_extract_epi32(m_data[0], 5);
            *(int32_t *)(out_data + (h + 2) * out_stride + w + 0) = _mm256_extract_epi32(m_data[0], 2);
            *(int32_t *)(out_data + (h + 2) * out_stride + w + 4) = _mm256_extract_epi32(m_data[0], 6);
            *(int32_t *)(out_data + (h + 3) * out_stride + w + 0) = _mm256_extract_epi32(m_data[0], 3);
            *(int32_t *)(out_data + (h + 3) * out_stride + w + 4) = _mm256_extract_epi32(m_data[0], 7);
        }

        for (; w < out_width; ++w) {
            int32_t w_idx_0 = w_offset[w];
            int32_t w_idx_1 = w_idx_0 >= in_width - 1 ? w_idx_0 : w_idx_0 + 1;
            int16_t coeff_0 = w_coeff[w * 2 + 0];
            int16_t coeff_1 = w_coeff[w * 2 + 1];

            for (int32_t hh = 0; hh < 4; ++hh) {
                int32_t row_0_value = in_data[h_idx[2 * hh + 0] * in_stride + w_idx_0] * coeff_0 +
                                      in_data[h_idx[2 * hh + 0] * in_stride + w_idx_1] * coeff_1;
                int32_t row_1_value = in_data[h_idx[2 * hh + 1] * in_stride + w_idx_0] * coeff_0 +
                                      in_data[h_idx[2 * hh + 1] * in_stride + w_idx_1] * coeff_1;

                row_0_value >>= 4;
                row_1_value >>= 4;
                int32_t rst_value = (((h_coeff_cur[2 * hh + 0] * row_0_value) >> 16) +
                                     ((h_coeff_cur[2 * hh + 1] * row_1_value) >> 16) + 2) >>
                                    2;
                out_data[(h + hh) * out_stride + w] = rst_value;
            }
        }
    }
    for (; h < out_height; ++h) {
        int32_t h_idx_0   = h_offset[h];
        int32_t h_idx_1   = h_idx_0 == in_height - 1 ? in_height - 1 : h_idx_0 + 1;
        int16_t h_coeff_0 = h_coeff[h];
        int16_t h_coeff_1 = INTER_RESIZE_COEF_SCALE - h_coeff_0;

        for (int32_t w = 0; w < out_width; ++w) {
            int32_t w_idx_0 = w_offset[w];
            int32_t w_idx_1 = w_idx_0 >= in_width - 1 ? w_idx_0 : w_idx_0 + 1;
            int16_t coeff_0 = w_coeff[w * 2 + 0];
            int16_t coeff_1 = w_coeff[w * 2 + 1];

            int32_t row_0_value = in_data[h_idx_0 * in_stride + w_idx_0] * coeff_0 +
                                  in_data[h_idx_0 * in_stride + w_idx_1] * coeff_1;
            int32_t row_1_value = in_data[h_idx_1 * in_stride + w_idx_0] * coeff_0 +
                                  in_data[h_idx_1 * in_stride + w_idx_1] * coeff_1;

            row_0_value >>= 4;
            row_1_value >>= 4;
            int32_t rst_value = (((h_coeff_0 * row_0_value) >> 16) +
                                 ((h_coeff_1 * row_1_value) >> 16) + 2) >>
                                2;
            out_data[h * out_stride + w] = rst_value;
        }
    }
}

int32_t resize_linear_shrink2_oneline_c1_kernel_u8_fma(
    const uint8_t *in_ptr,
    int32_t in_stride,
    int32_t out_width,
    uint8_t *out_ptr)
{
    __m256i m_zero            = _mm256_set1_epi8(0);
    __m256i m_epi16_two       = _mm256_set1_epi16(2);
    const int32_t shuffle_int = (0 << 0) +
                                (2 << 2) +
                                (1 << 4) +
                                (3 << 6);

    int32_t w = 0;
    for (; w <= out_width - 32; w += 32) {
        __m256i m_data[4];
        m_data[0] = _mm256_loadu_si256((const __m256i *)(in_ptr + 0 * in_stride + w * 2 + 0));
        m_data[1] = _mm256_loadu_si256((const __m256i *)(in_ptr + 0 * in_stride + w * 2 + 32));
        m_data[2] = _mm256_loadu_si256((const __m256i *)(in_ptr + 1 * in_stride + w * 2 + 0));
        m_data[3] = _mm256_loadu_si256((const __m256i *)(in_ptr + 1 * in_stride + w * 2 + 32));

        __m256i m_data_s16[8];
        m_data_s16[0] = _mm256_unpacklo_epi8(m_data[0], m_zero);
        m_data_s16[1] = _mm256_unpackhi_epi8(m_data[0], m_zero);
        m_data_s16[2] = _mm256_unpacklo_epi8(m_data[1], m_zero);
        m_data_s16[3] = _mm256_unpackhi_epi8(m_data[1], m_zero);
        m_data_s16[4] = _mm256_unpacklo_epi8(m_data[2], m_zero);
        m_data_s16[5] = _mm256_unpackhi_epi8(m_data[2], m_zero);
        m_data_s16[6] = _mm256_unpacklo_epi8(m_data[3], m_zero);
        m_data_s16[7] = _mm256_unpackhi_epi8(m_data[3], m_zero);

        __m256i m_dst[4];
        m_dst[0] = _mm256_hadd_epi16(m_data_s16[0], m_data_s16[1]);
        m_dst[1] = _mm256_hadd_epi16(m_data_s16[2], m_data_s16[3]);
        m_dst[2] = _mm256_hadd_epi16(m_data_s16[4], m_data_s16[5]);
        m_dst[3] = _mm256_hadd_epi16(m_data_s16[6], m_data_s16[7]);

        m_dst[0] = _mm256_adds_epi16(m_dst[0], m_dst[2]);
        m_dst[1] = _mm256_adds_epi16(m_dst[1], m_dst[3]);

        m_dst[0] = _mm256_adds_epi16(m_dst[0], m_epi16_two);
        m_dst[1] = _mm256_adds_epi16(m_dst[1], m_epi16_two);

        m_dst[0] = _mm256_srai_epi16(m_dst[0], 2);
        m_dst[1] = _mm256_srai_epi16(m_dst[1], 2);

        m_dst[0] = _mm256_packus_epi16(m_dst[0], m_dst[1]);

        m_dst[0] = _mm256_permute4x64_epi64(m_dst[0], shuffle_int);

        _mm256_storeu_si256((__m256i *)(out_ptr + w), m_dst[0]);
    }
    return w;
}

int32_t resize_linear_w_oneline_c3_u8_fma(
    int32_t in_width,
    const uint8_t *in_data,
    int32_t out_width,
    const int32_t *w_offset,
    const int16_t *w_coeff,
    int16_t COEFF_SUM,
    int32_t *row)
{
    const int32_t channels = 3;

    int32_t last_w = out_width;
    while (last_w > 0 && w_offset[last_w - 1] >= (in_width - 2) * channels) {
        last_w--;
    }

    __m256i b0 = _mm256_setr_epi8(
        0, 3, 1, 4, 2, 5, 8, 11, 9, 12, 10, 13, 0, 0, 0, 0, 0, 3, 1, 4, 2, 5, 8, 11, 9, 12, 10, 13, 0, 0, 0, 0);

    const int32_t useless = 0;
    __m256i b1_0          = _mm256_setr_epi32(
        0, 1, 2, 4, 5, 6, useless, useless);
    __m256i b1_1 = _mm256_setr_epi32(
        2, 4, 5, 6, useless, useless, 0, 1);
    __m256i b1_2 = _mm256_setr_epi32(
        5, 6, useless, useless, 0, 1, 2, 4);
    __m256i b1_3 = _mm256_setr_epi32(
        useless, useless, 0, 1, 2, 4, 5, 6);

    const int32_t blend_0_and_1 = 0b11000000;
    const int32_t blend_1_and_2 = 0b11110000;
    const int32_t blend_2_and_3 = 0b11111100;

    int32_t w = 0;
    for (; w <= out_width - 16; w += 16) {
        __m256i m_data[4];
        m_data[0] = _mm256_i32gather_epi64((const long long int *)in_data, _mm_load_si128((const __m128i *)(w_offset + w + 0)), 1);
        m_data[1] = _mm256_i32gather_epi64((const long long int *)in_data, _mm_load_si128((const __m128i *)(w_offset + w + 4)), 1);
        m_data[2] = _mm256_i32gather_epi64((const long long int *)in_data, _mm_load_si128((const __m128i *)(w_offset + w + 8)), 1);
        m_data[3] = _mm256_i32gather_epi64((const long long int *)in_data, _mm_load_si128((const __m128i *)(w_offset + w + 12)), 1);

        m_data[0] = _mm256_shuffle_epi8(m_data[0], b0);
        m_data[1] = _mm256_shuffle_epi8(m_data[1], b0);
        m_data[2] = _mm256_shuffle_epi8(m_data[2], b0);
        m_data[3] = _mm256_shuffle_epi8(m_data[3], b0);

        m_data[0] = _mm256_permutevar8x32_epi32(m_data[0], b1_0);
        m_data[1] = _mm256_permutevar8x32_epi32(m_data[1], b1_1);
        m_data[2] = _mm256_permutevar8x32_epi32(m_data[2], b1_2);
        m_data[3] = _mm256_permutevar8x32_epi32(m_data[3], b1_3);

        __m256i m_data_calc[3];
        m_data_calc[0] = _mm256_blend_epi32(m_data[0], m_data[1], blend_0_and_1);
        m_data_calc[1] = _mm256_blend_epi32(m_data[1], m_data[2], blend_1_and_2);
        m_data_calc[2] = _mm256_blend_epi32(m_data[2], m_data[3], blend_2_and_3);

        __m256i m_data_calc_s16[6];
        m_data_calc_s16[0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data_calc[0]));
        m_data_calc_s16[2] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data_calc[1]));
        m_data_calc_s16[4] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data_calc[2]));
        m_data_calc_s16[1] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(m_data_calc[0], 1));
        m_data_calc_s16[3] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(m_data_calc[1], 1));
        m_data_calc_s16[5] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(m_data_calc[2], 1));

        __m256i m_data_calc_s32[6];
        m_data_calc_s32[0] = _mm256_madd_epi16(m_data_calc_s16[0], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 0) * 2)));
        m_data_calc_s32[1] = _mm256_madd_epi16(m_data_calc_s16[1], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 8) * 2)));
        m_data_calc_s32[2] = _mm256_madd_epi16(m_data_calc_s16[2], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 16) * 2)));
        m_data_calc_s32[3] = _mm256_madd_epi16(m_data_calc_s16[3], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 24) * 2)));
        m_data_calc_s32[4] = _mm256_madd_epi16(m_data_calc_s16[4], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 32) * 2)));
        m_data_calc_s32[5] = _mm256_madd_epi16(m_data_calc_s16[5], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 40) * 2)));

        m_data_calc_s32[0] = _mm256_srai_epi32(m_data_calc_s32[0], 4);
        m_data_calc_s32[1] = _mm256_srai_epi32(m_data_calc_s32[1], 4);
        m_data_calc_s32[2] = _mm256_srai_epi32(m_data_calc_s32[2], 4);
        m_data_calc_s32[3] = _mm256_srai_epi32(m_data_calc_s32[3], 4);
        m_data_calc_s32[4] = _mm256_srai_epi32(m_data_calc_s32[4], 4);
        m_data_calc_s32[5] = _mm256_srai_epi32(m_data_calc_s32[5], 4);

        _mm256_store_si256((__m256i *)(row + w * channels + 0), m_data_calc_s32[0]);
        _mm256_store_si256((__m256i *)(row + w * channels + 8), m_data_calc_s32[1]);
        _mm256_store_si256((__m256i *)(row + w * channels + 16), m_data_calc_s32[2]);
        _mm256_store_si256((__m256i *)(row + w * channels + 24), m_data_calc_s32[3]);
        _mm256_store_si256((__m256i *)(row + w * channels + 32), m_data_calc_s32[4]);
        _mm256_store_si256((__m256i *)(row + w * channels + 40), m_data_calc_s32[5]);
    }
    return w;
}

int32_t resize_linear_w_oneline_c4_u8_fma(
    int32_t in_width,
    const uint8_t *in_data,
    int32_t out_width,
    const int32_t *w_offset,
    const int16_t *w_coeff,
    int16_t COEFF_SUM,
    int32_t *row)
{
    const int32_t channels = 4;

    int32_t last_w = out_width;
    while (last_w > 0 && w_offset[last_w - 1] >= (in_width - 1) * channels) {
        last_w--;
    }

    __m256i b0 = _mm256_setr_epi8(
        0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15, 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);

    int32_t w = 0;
    for (; w <= out_width - 16; w += 16) {
        __m256i m_data[4];
        m_data[0] = _mm256_i32gather_epi64((const long long int *)in_data, _mm_load_si128((const __m128i *)(w_offset + w + 0)), 1);
        m_data[1] = _mm256_i32gather_epi64((const long long int *)in_data, _mm_load_si128((const __m128i *)(w_offset + w + 4)), 1);
        m_data[2] = _mm256_i32gather_epi64((const long long int *)in_data, _mm_load_si128((const __m128i *)(w_offset + w + 8)), 1);
        m_data[3] = _mm256_i32gather_epi64((const long long int *)in_data, _mm_load_si128((const __m128i *)(w_offset + w + 12)), 1);

        m_data[0] = _mm256_shuffle_epi8(m_data[0], b0);
        m_data[1] = _mm256_shuffle_epi8(m_data[1], b0);
        m_data[2] = _mm256_shuffle_epi8(m_data[2], b0);
        m_data[3] = _mm256_shuffle_epi8(m_data[3], b0);

        __m256i m_data_s16[8];
        m_data_s16[0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[0]));
        m_data_s16[2] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[1]));
        m_data_s16[4] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[2]));
        m_data_s16[6] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(m_data[3]));
        m_data_s16[1] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(m_data[0], 1));
        m_data_s16[3] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(m_data[1], 1));
        m_data_s16[5] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(m_data[2], 1));
        m_data_s16[7] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(m_data[3], 1));

        __m256i m_data_calc_s32[8];
        m_data_calc_s32[0] = _mm256_madd_epi16(m_data_s16[0], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 0) * 2)));
        m_data_calc_s32[1] = _mm256_madd_epi16(m_data_s16[1], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 8) * 2)));
        m_data_calc_s32[2] = _mm256_madd_epi16(m_data_s16[2], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 16) * 2)));
        m_data_calc_s32[3] = _mm256_madd_epi16(m_data_s16[3], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 24) * 2)));
        m_data_calc_s32[4] = _mm256_madd_epi16(m_data_s16[4], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 32) * 2)));
        m_data_calc_s32[5] = _mm256_madd_epi16(m_data_s16[5], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 40) * 2)));
        m_data_calc_s32[6] = _mm256_madd_epi16(m_data_s16[6], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 48) * 2)));
        m_data_calc_s32[7] = _mm256_madd_epi16(m_data_s16[7], _mm256_load_si256((const __m256i *)(w_coeff + (w * channels + 56) * 2)));

        m_data_calc_s32[0] = _mm256_srai_epi32(m_data_calc_s32[0], 4);
        m_data_calc_s32[1] = _mm256_srai_epi32(m_data_calc_s32[1], 4);
        m_data_calc_s32[2] = _mm256_srai_epi32(m_data_calc_s32[2], 4);
        m_data_calc_s32[3] = _mm256_srai_epi32(m_data_calc_s32[3], 4);
        m_data_calc_s32[4] = _mm256_srai_epi32(m_data_calc_s32[4], 4);
        m_data_calc_s32[5] = _mm256_srai_epi32(m_data_calc_s32[5], 4);
        m_data_calc_s32[6] = _mm256_srai_epi32(m_data_calc_s32[6], 4);
        m_data_calc_s32[7] = _mm256_srai_epi32(m_data_calc_s32[7], 4);

        _mm256_store_si256((__m256i *)(row + w * channels + 0), m_data_calc_s32[0]);
        _mm256_store_si256((__m256i *)(row + w * channels + 8), m_data_calc_s32[1]);
        _mm256_store_si256((__m256i *)(row + w * channels + 16), m_data_calc_s32[2]);
        _mm256_store_si256((__m256i *)(row + w * channels + 24), m_data_calc_s32[3]);
        _mm256_store_si256((__m256i *)(row + w * channels + 32), m_data_calc_s32[4]);
        _mm256_store_si256((__m256i *)(row + w * channels + 40), m_data_calc_s32[5]);
        _mm256_store_si256((__m256i *)(row + w * channels + 48), m_data_calc_s32[6]);
        _mm256_store_si256((__m256i *)(row + w * channels + 56), m_data_calc_s32[7]);
    }
    return w;
}

int32_t resize_linear_shrink2_oneline_c4_kernel_u8_fma(
    const uint8_t *in_ptr,
    int32_t in_stride,
    int32_t out_width,
    uint8_t *out_ptr)
{
    const int32_t channels    = 4;
    __m256i m_zero            = _mm256_set1_epi8(0);
    __m256i m_epi16_two       = _mm256_set1_epi16(2);
    const int32_t shuffle_int = (0 << 0) +
                                (2 << 2) +
                                (1 << 4) +
                                (3 << 6);

    int32_t w = 0;

    for (; w <= out_width - 8; w += 8) {
        __m256i m_data[4];
        m_data[0] = _mm256_loadu_si256((const __m256i *)(in_ptr + 0 * in_stride + w * channels * 2 + 0));
        m_data[1] = _mm256_loadu_si256((const __m256i *)(in_ptr + 0 * in_stride + w * channels * 2 + 32));
        m_data[2] = _mm256_loadu_si256((const __m256i *)(in_ptr + 1 * in_stride + w * channels * 2 + 0));
        m_data[3] = _mm256_loadu_si256((const __m256i *)(in_ptr + 1 * in_stride + w * channels * 2 + 32));

        __m256i m_data_u8_temp[4];
        m_data_u8_temp[0] = _mm256_unpacklo_epi8(m_data[0], m_data[2]);
        m_data_u8_temp[1] = _mm256_unpackhi_epi8(m_data[0], m_data[2]);
        m_data_u8_temp[2] = _mm256_unpacklo_epi8(m_data[1], m_data[3]);
        m_data_u8_temp[3] = _mm256_unpackhi_epi8(m_data[1], m_data[3]);

        __m256i m_data_s16[8];
        m_data_s16[0] = _mm256_unpacklo_epi8(m_data_u8_temp[0], m_zero);
        m_data_s16[1] = _mm256_unpackhi_epi8(m_data_u8_temp[0], m_zero);
        m_data_s16[2] = _mm256_unpacklo_epi8(m_data_u8_temp[1], m_zero);
        m_data_s16[3] = _mm256_unpackhi_epi8(m_data_u8_temp[1], m_zero);
        m_data_s16[4] = _mm256_unpacklo_epi8(m_data_u8_temp[2], m_zero);
        m_data_s16[5] = _mm256_unpackhi_epi8(m_data_u8_temp[2], m_zero);
        m_data_s16[6] = _mm256_unpacklo_epi8(m_data_u8_temp[3], m_zero);
        m_data_s16[7] = _mm256_unpackhi_epi8(m_data_u8_temp[3], m_zero);

        __m256i m_dst[2];
        m_dst[0] = _mm256_adds_epi16(_mm256_hadd_epi16(m_data_s16[0], m_data_s16[2]),
                                     _mm256_hadd_epi16(m_data_s16[1], m_data_s16[3]));
        m_dst[1] = _mm256_adds_epi16(_mm256_hadd_epi16(m_data_s16[4], m_data_s16[6]),
                                     _mm256_hadd_epi16(m_data_s16[5], m_data_s16[7]));

        m_dst[0] = _mm256_adds_epi16(m_dst[0], m_epi16_two);
        m_dst[1] = _mm256_adds_epi16(m_dst[1], m_epi16_two);

        m_dst[0] = _mm256_srai_epi16(m_dst[0], 2);
        m_dst[1] = _mm256_srai_epi16(m_dst[1], 2);

        m_dst[0] = _mm256_packus_epi16(m_dst[0], m_dst[1]);
        m_dst[0] = _mm256_permute4x64_epi64(m_dst[0], shuffle_int);

        _mm256_storeu_si256((__m256i *)(out_ptr + w * channels), m_dst[0]);
    }
    return w;
}

}
}
}
} // namespace ppl::cv::x86::fma
