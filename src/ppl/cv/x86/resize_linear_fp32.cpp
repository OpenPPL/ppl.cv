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
#include <math.h>

#include "ppl/cv/x86/fma/internal_fma.hpp"

namespace ppl {
namespace cv {
namespace x86 {

static inline int32_t resize_img_floor(float a)
{
    return (((a) >= 0) ? ((int32_t)a) : ((int32_t)a - 1));
}

static void resize_linear_calc_offset_fp32(
    int32_t inHeight,
    int32_t inWidth,
    int32_t channels,
    int32_t outHeight,
    int32_t outWidth,
    int32_t &w_max,
    int32_t *h_offset,
    int32_t *w_offset,
    float *h_coeff,
    float *w_coeff)
{
    double inv_scale_h = (double)outHeight / inHeight;
    double scale_h     = 1.0 / inv_scale_h;
    for (int32_t h = 0; h < outHeight; ++h) {
        float float_h = (h + 0.5) * scale_h - 0.5;
        int32_t int_h = resize_img_floor(float_h);
        float_h -= int_h;

        if (int_h < 0) {
            int_h   = 0;
            float_h = 0;
        }
        if (int_h >= inHeight - 1) {
            int_h   = inHeight - 1;
            float_h = 0;
        }

        h_offset[h] = int_h;
        h_coeff[h]  = 1.0f - float_h;
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
        if (int_w >= inWidth - 1) {
            int_w   = inWidth - 1;
            float_w = 0;
        }
        if (int_w <= inWidth - 2) {
            w_max = w;
        }

        for (int32_t c = 0; c < channels; ++c) {
            w_offset[w * channels + c] = int_w * channels + c;
            w_coeff[w * channels + c]  = 1.0f - float_w;
        }
    }
}

static void resize_linear_twoline_fp32(
    int32_t inWidth,
    int32_t outWidth,
    int32_t channels,
    const float *inData_0,
    const float *inData_1,
    int32_t w_max,
    const int32_t *w_offset,
    const float *w_coeff,
    int32_t h_idx,
    float h_coeff,
    float *row_0,
    float *row_1,
    float *outData)
{
    int32_t i = 0;

    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        i = fma::resize_linear_twoline_fp32_fma(w_max * channels, channels, inData_0, inData_1, w_offset, w_coeff, h_coeff, row_0, row_1, outData);
    }

    __m128 m_h_coeff_0 = _mm_set1_ps(h_coeff);
    __m128 m_h_coeff_1 = _mm_set1_ps(1.0f - h_coeff);
    __m128 m_one       = _mm_set1_ps(1.0f);

    if (channels == 4) {
        for (; i < w_max * channels; i += 4) {
            __m128 m_data_0 = _mm_loadu_ps(inData_0 + w_offset[i]);
            __m128 m_data_1 = _mm_loadu_ps(inData_0 + w_offset[i] + 4);
            __m128 m_data_2 = _mm_loadu_ps(inData_1 + w_offset[i]);
            __m128 m_data_3 = _mm_loadu_ps(inData_1 + w_offset[i] + 4);

            __m128 m_w_coeff_0 = _mm_load_ps(w_coeff + i);
            __m128 m_w_coeff_1 = _mm_sub_ps(m_one, m_w_coeff_0);

            __m128 m_rst_0     = _mm_mul_ps(m_data_0, m_w_coeff_0);
            __m128 m_rst_1     = _mm_mul_ps(m_data_1, m_w_coeff_1);
            __m128 m_rst_2     = _mm_mul_ps(m_data_2, m_w_coeff_0);
            __m128 m_rst_3     = _mm_mul_ps(m_data_3, m_w_coeff_1);
            __m128 m_rst_row_0 = _mm_add_ps(m_rst_0, m_rst_1);
            __m128 m_rst_row_1 = _mm_add_ps(m_rst_2, m_rst_3);
            __m128 m_rst       = _mm_add_ps(_mm_mul_ps(m_rst_row_0, m_h_coeff_0),
                                      _mm_mul_ps(m_rst_row_1, m_h_coeff_1));

            _mm_store_ps(row_0 + i, m_rst_row_0);
            _mm_store_ps(row_1 + i, m_rst_row_1);
            _mm_storeu_ps(outData + i, m_rst);
        }
    }

    for (; i < w_max * channels - 4; i += 4) {
        __m128 m_data_0 = _mm_set_ps(inData_0[w_offset[i + 3]],
                                     inData_0[w_offset[i + 2]],
                                     inData_0[w_offset[i + 1]],
                                     inData_0[w_offset[i + 0]]);
        __m128 m_data_1 = _mm_set_ps(inData_0[w_offset[i + 3] + channels],
                                     inData_0[w_offset[i + 2] + channels],
                                     inData_0[w_offset[i + 1] + channels],
                                     inData_0[w_offset[i + 0] + channels]);
        __m128 m_data_2 = _mm_set_ps(inData_1[w_offset[i + 3]],
                                     inData_1[w_offset[i + 2]],
                                     inData_1[w_offset[i + 1]],
                                     inData_1[w_offset[i + 0]]);
        __m128 m_data_3 = _mm_set_ps(inData_1[w_offset[i + 3] + channels],
                                     inData_1[w_offset[i + 2] + channels],
                                     inData_1[w_offset[i + 1] + channels],
                                     inData_1[w_offset[i + 0] + channels]);

        __m128 m_w_coeff_0 = _mm_load_ps(w_coeff + i);
        __m128 m_w_coeff_1 = _mm_sub_ps(m_one, m_w_coeff_0);

        __m128 m_rst_0     = _mm_mul_ps(m_data_0, m_w_coeff_0);
        __m128 m_rst_1     = _mm_mul_ps(m_data_1, m_w_coeff_1);
        __m128 m_rst_2     = _mm_mul_ps(m_data_2, m_w_coeff_0);
        __m128 m_rst_3     = _mm_mul_ps(m_data_3, m_w_coeff_1);
        __m128 m_rst_row_0 = _mm_add_ps(m_rst_0, m_rst_1);
        __m128 m_rst_row_1 = _mm_add_ps(m_rst_2, m_rst_3);
        __m128 m_rst       = _mm_add_ps(_mm_mul_ps(m_rst_row_0, m_h_coeff_0),
                                  _mm_mul_ps(m_rst_row_1, m_h_coeff_1));

        _mm_store_ps(row_0 + i, m_rst_row_0);
        _mm_store_ps(row_1 + i, m_rst_row_1);
        _mm_storeu_ps(outData + i, m_rst);
    }

    for (; i < w_max * channels; ++i) {
        int32_t w_idx_0   = w_offset[i];
        int32_t w_idx_1   = w_idx_0 + channels;
        float coeff       = w_coeff[i];
        float row_0_value = inData_0[w_idx_0] * coeff + inData_0[w_idx_1] * (1.0f - coeff);
        float row_1_value = inData_1[w_idx_0] * coeff + inData_1[w_idx_1] * (1.0f - coeff);

        row_0[i]    = row_0_value;
        row_1[i]    = row_1_value;
        outData[i] = (row_0_value * h_coeff) + (row_1_value * (1.0f - h_coeff));
    }
    for (; i < outWidth * channels; ++i) {
        int32_t w_idx_0   = w_offset[i];
        int32_t w_idx_1   = w_idx_0 >= (inWidth - 1) * channels ? w_idx_0 : w_idx_0 + channels;
        float coeff       = w_coeff[i];
        float row_0_value = inData_0[w_idx_0] * coeff + inData_0[w_idx_1] * (1.0f - coeff);
        float row_1_value = inData_1[w_idx_0] * coeff + inData_1[w_idx_1] * (1.0f - coeff);

        row_0[i]    = row_0_value;
        row_1[i]    = row_1_value;
        outData[i] = (row_0_value * h_coeff) + (row_1_value * (1.0f - h_coeff));
    }
}

static void resize_linear_w_oneline_fp32(
    int32_t inWidth,
    int32_t outWidth,
    int32_t channels,
    const float *inData,
    int32_t w_max,
    const int32_t *w_offset,
    const float *w_coeff,
    float *row)
{
    __m128 m_one = _mm_set1_ps(1.0f);
    int32_t i    = 0;

    if (channels == 4) {
        for (; i < w_max * channels; i += 4) {
            __m128 m_data_0 = _mm_loadu_ps(inData + w_offset[i]);
            __m128 m_data_1 = _mm_loadu_ps(inData + w_offset[i] + 4);

            __m128 m_w_coeff_0 = _mm_load_ps(w_coeff + i);
            __m128 m_w_coeff_1 = _mm_sub_ps(m_one, m_w_coeff_0);
            __m128 m_rst_0     = _mm_mul_ps(m_data_0, m_w_coeff_0);
            __m128 m_rst_1     = _mm_mul_ps(m_data_1, m_w_coeff_1);
            __m128 m_rst_row   = _mm_add_ps(m_rst_0, m_rst_1);

            _mm_store_ps(row + i, m_rst_row);
        }
    }

    for (; i < w_max * channels - 4; i += 4) {
        __m128 m_data_0 = _mm_set_ps(inData[w_offset[i + 3]],
                                     inData[w_offset[i + 2]],
                                     inData[w_offset[i + 1]],
                                     inData[w_offset[i + 0]]);
        __m128 m_data_1 = _mm_set_ps(inData[w_offset[i + 3] + channels],
                                     inData[w_offset[i + 2] + channels],
                                     inData[w_offset[i + 1] + channels],
                                     inData[w_offset[i + 0] + channels]);

        __m128 m_w_coeff_0 = _mm_load_ps(w_coeff + i);
        __m128 m_w_coeff_1 = _mm_sub_ps(m_one, m_w_coeff_0);
        __m128 m_rst_0     = _mm_mul_ps(m_data_0, m_w_coeff_0);
        __m128 m_rst_1     = _mm_mul_ps(m_data_1, m_w_coeff_1);
        __m128 m_rst_row   = _mm_add_ps(m_rst_0, m_rst_1);

        _mm_store_ps(row + i, m_rst_row);
    }

    for (; i < outWidth * channels; ++i) {
        int32_t w_idx_0 = w_offset[i];
        int32_t w_idx_1 = w_idx_0 >= (inWidth - 1) * channels ? w_idx_0 : w_idx_0 + channels;
        float coeff     = w_coeff[i];
        float row_value = inData[w_idx_0] * coeff + inData[w_idx_1] * (1.0f - coeff);
        row[i]          = row_value;
    }
}

static void resize_linear_h_fp32(
    int32_t outWidth,
    int32_t channels,
    const float *row_0,
    const float *row_1,
    int32_t h_idx,
    float h_coeff,
    float *outData)
{
    int32_t i = 0;

    __m128 m_h_coeff_0 = _mm_set1_ps(h_coeff);
    __m128 m_h_coeff_1 = _mm_set1_ps(1.0f - h_coeff);
    for (; i <= outWidth * channels - 4; i += 4) {
        __m128 m_data_0 = _mm_load_ps(row_0 + i);
        __m128 m_data_1 = _mm_load_ps(row_1 + i);
        __m128 m_rst    = _mm_add_ps(_mm_mul_ps(m_data_0, m_h_coeff_0),
                                  _mm_mul_ps(m_data_1, m_h_coeff_1));
        _mm_storeu_ps(outData + i, m_rst);
    }
    for (; i < outWidth * channels; ++i) {
        outData[i] = row_0[i] * h_coeff + row_1[i] * (1.0f - h_coeff);
    }
}

static void resize_linear_kernel_fp32(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t channels,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    int32_t cn_width           = channels * outWidth;
    uint64_t size_for_h_offset = (outHeight * sizeof(int32_t) + 128 - 1) / 128 * 128;
    uint64_t size_for_w_offset = (cn_width * sizeof(int32_t) + 128 - 1) / 128 * 128;
    uint64_t size_for_h_coeff  = (outHeight * sizeof(float) + 128 - 1) / 128 * 128;
    uint64_t size_for_w_coeff  = (cn_width * sizeof(float) + 128 - 1) / 128 * 128;
    uint64_t size_for_row_0    = (cn_width * sizeof(float) + 128 - 1) / 128 * 128;
    uint64_t size_for_row_1    = (cn_width * sizeof(float) + 128 - 1) / 128 * 128;

    uint64_t total_size = size_for_h_offset + size_for_w_offset + size_for_h_coeff + size_for_w_coeff + size_for_row_0 + size_for_row_1;

    void *temp_buffer = ppl::common::AlignedAlloc(total_size, 128);

    int32_t *h_offset = (int32_t *)temp_buffer;
    int32_t *w_offset = (int32_t *)((unsigned char *)h_offset + size_for_h_offset);
    float *h_coeff    = (float *)((unsigned char *)w_offset + size_for_w_offset);
    float *w_coeff    = (float *)((unsigned char *)h_coeff + size_for_h_coeff);
    float *row_0      = (float *)((unsigned char *)w_coeff + size_for_w_coeff);
    float *row_1      = (float *)((unsigned char *)row_0 + size_for_row_0);

    int32_t w_max = 0;
    resize_linear_calc_offset_fp32(inHeight, inWidth, channels, outHeight, outWidth, w_max, h_offset, w_offset, h_coeff, w_coeff);

    int32_t prev_h[2]  = {-1, -1};
    float *prev_ptr[2] = {nullptr, nullptr};

    int32_t reuse_count;
    float *row_ptr[2];

    for (int32_t h = 0; h < outHeight; ++h) {
        reuse_count = 0;
        row_ptr[0]  = nullptr;
        row_ptr[1]  = nullptr;

        int32_t src_h_idx_0 = h_offset[h];
        int32_t src_h_idx_1 = src_h_idx_0 == inHeight - 1 ? src_h_idx_0 : src_h_idx_0 + 1;

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

            resize_linear_twoline_fp32(inWidth, outWidth, channels, inData + src_h_idx_0 * inWidthStride, inData + src_h_idx_1 * inWidthStride, w_max, w_offset, w_coeff, h_offset[h], h_coeff[h], row_ptr[0], row_ptr[1], outData + h * outWidthStride);
        } else {
            if (reuse_count == 1) {
                if (row_ptr[0] == row_0) {
                    row_ptr[1] = row_1;
                } else {
                    row_ptr[1] = row_0;
                }
                resize_linear_w_oneline_fp32(inWidth, outWidth, channels, inData + src_h_idx_1 * inWidthStride, w_max, w_offset, w_coeff, row_ptr[1]);
            }
            resize_linear_h_fp32(outWidth, channels, row_ptr[0], row_ptr[1], h_offset[h], h_coeff[h], outData + h * outWidthStride);
        }

        prev_h[0]   = src_h_idx_0;
        prev_h[1]   = src_h_idx_1;
        prev_ptr[0] = row_ptr[0];
        prev_ptr[1] = row_ptr[1];
    }
    ppl::common::AlignedFree(temp_buffer);
}

static void resize_linear_shrink2_c1_kernel_fp32(
    const float *inData,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    __m128 m_p25 = _mm_set1_ps(0.25f);

    for (int32_t i = 0; i < outHeight; ++i) {
        int32_t j = 0;
        for (; j <= outWidth - 4; j += 4) {
            __m128 m_data_row0_left  = _mm_loadu_ps(inData + (i * 2 + 0) * inWidthStride + j * 2 + 0);
            __m128 m_data_row0_right = _mm_loadu_ps(inData + (i * 2 + 0) * inWidthStride + j * 2 + 4);
            __m128 m_data_row1_left  = _mm_loadu_ps(inData + (i * 2 + 1) * inWidthStride + j * 2 + 0);
            __m128 m_data_row1_right = _mm_loadu_ps(inData + (i * 2 + 1) * inWidthStride + j * 2 + 4);
            __m128 m_data_0          = _mm_shuffle_ps(m_data_row0_left, m_data_row0_right, 0b10001000);
            __m128 m_data_1          = _mm_shuffle_ps(m_data_row0_left, m_data_row0_right, 0b11011101);
            __m128 m_data_2          = _mm_shuffle_ps(m_data_row1_left, m_data_row1_right, 0b10001000);
            __m128 m_data_3          = _mm_shuffle_ps(m_data_row1_left, m_data_row1_right, 0b11011101);
            __m128 m_rst             = _mm_add_ps(_mm_add_ps(m_data_0, m_data_1), _mm_add_ps(m_data_2, m_data_3));
            m_rst                    = _mm_mul_ps(m_rst, m_p25);
            _mm_storeu_ps(outData + i * outWidthStride + j, m_rst);
        }
        for (; j < outWidth; ++j) {
            float rst = inData[(i * 2 + 0) * inWidthStride + j * 2 + 0] +
                        inData[(i * 2 + 0) * inWidthStride + j * 2 + 1] +
                        inData[(i * 2 + 1) * inWidthStride + j * 2 + 0] +
                        inData[(i * 2 + 1) * inWidthStride + j * 2 + 1];
            rst *= 0.25f;
            outData[i * outWidthStride + j] = rst;
        }
    }
}

static void resize_linear_shrink2_c3_kernel_fp32(
    const float *inData,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    __m128 m_p25 = _mm_set1_ps(0.25f);
    for (int32_t i = 0; i < outHeight; ++i) {
        int32_t j = 0;
        for (; j < outWidth - 1; ++j) {
            __m128 m_data_0 = _mm_loadu_ps(inData + (i * 2 + 0) * inWidthStride + j * 3 * 2 + 0);
            __m128 m_data_1 = _mm_loadu_ps(inData + (i * 2 + 0) * inWidthStride + j * 3 * 2 + 3);
            __m128 m_data_2 = _mm_loadu_ps(inData + (i * 2 + 1) * inWidthStride + j * 3 * 2 + 0);
            __m128 m_data_3 = _mm_loadu_ps(inData + (i * 2 + 1) * inWidthStride + j * 3 * 2 + 3);
            __m128 m_rst    = _mm_add_ps(_mm_add_ps(m_data_0, m_data_1), _mm_add_ps(m_data_2, m_data_3));
            m_rst           = _mm_mul_ps(m_rst, m_p25);
            _mm_storeu_ps(outData + i * outWidthStride + j * 3, m_rst);
        }
        for (; j < outWidth; ++j) {
            float rst_0 = inData[(i * 2 + 0) * inWidthStride + j * 3 * 2 + 0] +
                          inData[(i * 2 + 0) * inWidthStride + j * 3 * 2 + 3] +
                          inData[(i * 2 + 1) * inWidthStride + j * 3 * 2 + 0] +
                          inData[(i * 2 + 1) * inWidthStride + j * 3 * 2 + 3];
            float rst_1 = inData[(i * 2 + 0) * inWidthStride + j * 3 * 2 + 1] +
                          inData[(i * 2 + 0) * inWidthStride + j * 3 * 2 + 4] +
                          inData[(i * 2 + 1) * inWidthStride + j * 3 * 2 + 1] +
                          inData[(i * 2 + 1) * inWidthStride + j * 3 * 2 + 4];
            float rst_2 = inData[(i * 2 + 0) * inWidthStride + j * 3 * 2 + 2] +
                          inData[(i * 2 + 0) * inWidthStride + j * 3 * 2 + 5] +
                          inData[(i * 2 + 1) * inWidthStride + j * 3 * 2 + 2] +
                          inData[(i * 2 + 1) * inWidthStride + j * 3 * 2 + 5];

            rst_0 *= 0.25f;
            rst_1 *= 0.25f;
            rst_2 *= 0.25f;
            outData[i * outWidthStride + j * 3 + 0] = rst_0;
            outData[i * outWidthStride + j * 3 + 1] = rst_1;
            outData[i * outWidthStride + j * 3 + 2] = rst_2;
        }
    }
}

static void resize_linear_shrink2_c4_kernel_fp32(
    const float *inData,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    __m128 m_p25 = _mm_set1_ps(0.25f);

    for (int32_t i = 0; i < outHeight; ++i) {
        for (int32_t j = 0; j < outWidth; ++j) {
            __m128 m_data_0 = _mm_loadu_ps(inData + (i * 2 + 0) * inWidthStride + j * 4 * 2 + 0);
            __m128 m_data_1 = _mm_loadu_ps(inData + (i * 2 + 0) * inWidthStride + j * 4 * 2 + 4);
            __m128 m_data_2 = _mm_loadu_ps(inData + (i * 2 + 1) * inWidthStride + j * 4 * 2 + 0);
            __m128 m_data_3 = _mm_loadu_ps(inData + (i * 2 + 1) * inWidthStride + j * 4 * 2 + 4);
            __m128 m_rst    = _mm_add_ps(_mm_add_ps(m_data_0, m_data_1), _mm_add_ps(m_data_2, m_data_3));
            m_rst           = _mm_mul_ps(m_rst, m_p25);
            _mm_storeu_ps(outData + i * outWidthStride + j * 4, m_rst);
        }
    }
}

template <>
::ppl::common::RetCode ResizeLinear<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (outHeight * 2 == inHeight && outWidth * 2 == inWidth) {
        resize_linear_shrink2_c1_kernel_fp32(inData, inWidthStride, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }

    resize_linear_kernel_fp32(
        inHeight, inWidth, inWidthStride, inData, 1, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (outHeight * 2 == inHeight && outWidth * 2 == inWidth) {
        resize_linear_shrink2_c3_kernel_fp32(inData, inWidthStride, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }

    resize_linear_kernel_fp32(
        inHeight, inWidth, inWidthStride, inData, 3, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (outHeight * 2 == inHeight && outWidth * 2 == inWidth) {
        resize_linear_shrink2_c4_kernel_fp32(inData, inWidthStride, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }

    resize_linear_kernel_fp32(
        inHeight, inWidth, inWidthStride, inData, 4, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
