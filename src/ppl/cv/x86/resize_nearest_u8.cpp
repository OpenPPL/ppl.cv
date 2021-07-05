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

#include <string.h>
#include <limits.h>
#include <immintrin.h>
#include <float.h>
#include <stdint.h>
#include <math.h>

namespace ppl {
namespace cv {
namespace x86 {

static inline int32_t resize_img_floor(float a)
{
    return (((a) >= 0) ? ((int32_t)a) : ((int32_t)a - 1));
}

static void resize_nearest_calc_offset_u8(
    int32_t inHeight,
    int32_t inWidth,
    int32_t outHeight,
    int32_t outWidth,
    int32_t *h_offset,
    int32_t *w_offset)
{
    double inv_scale_h = (double)outHeight / inHeight;
    double scale_h     = 1.0 / inv_scale_h;

    for (int32_t h = 0; h < outHeight; ++h) {
        int32_t int_h = resize_img_floor(h * scale_h);
        if (int_h >= inHeight - 1) {
            int_h = inHeight - 1;
        }
        h_offset[h] = int_h;
    }

    double inv_scale_w = (double)outWidth / inWidth;
    double scale_w     = 1.0 / inv_scale_w;

    for (int32_t w = 0; w < outWidth; ++w) {
        int32_t int_w = resize_img_floor(w * scale_w);
        if (int_w >= inWidth - 1) {
            int_w = inWidth - 1;
        }
        w_offset[w] = int_w;
    }
}

static void resize_nearest_c1_w_fourline_kernel_u8(
    const uint8_t *inData_0,
    const uint8_t *inData_1,
    const uint8_t *inData_2,
    const uint8_t *inData_3,
    int32_t outWidth,
    int32_t *w_offset,
    uint8_t *outData_0,
    uint8_t *outData_1,
    uint8_t *outData_2,
    uint8_t *outData_3)
{
    int32_t i = 0;
    for (; i <= outWidth - 16; i += 16) {
        __m128i m_data_0 = _mm_setr_epi8(inData_0[w_offset[i + 0]], inData_0[w_offset[i + 1]], inData_0[w_offset[i + 2]], inData_0[w_offset[i + 3]], inData_0[w_offset[i + 4]], inData_0[w_offset[i + 5]], inData_0[w_offset[i + 6]], inData_0[w_offset[i + 7]], inData_0[w_offset[i + 8]], inData_0[w_offset[i + 9]], inData_0[w_offset[i + 10]], inData_0[w_offset[i + 11]], inData_0[w_offset[i + 12]], inData_0[w_offset[i + 13]], inData_0[w_offset[i + 14]], inData_0[w_offset[i + 15]]);

        __m128i m_data_1 = _mm_setr_epi8(inData_1[w_offset[i + 0]], inData_1[w_offset[i + 1]], inData_1[w_offset[i + 2]], inData_1[w_offset[i + 3]], inData_1[w_offset[i + 4]], inData_1[w_offset[i + 5]], inData_1[w_offset[i + 6]], inData_1[w_offset[i + 7]], inData_1[w_offset[i + 8]], inData_1[w_offset[i + 9]], inData_1[w_offset[i + 10]], inData_1[w_offset[i + 11]], inData_1[w_offset[i + 12]], inData_1[w_offset[i + 13]], inData_1[w_offset[i + 14]], inData_1[w_offset[i + 15]]);

        __m128i m_data_2 = _mm_setr_epi8(inData_2[w_offset[i + 0]], inData_2[w_offset[i + 1]], inData_2[w_offset[i + 2]], inData_2[w_offset[i + 3]], inData_2[w_offset[i + 4]], inData_2[w_offset[i + 5]], inData_2[w_offset[i + 6]], inData_2[w_offset[i + 7]], inData_2[w_offset[i + 8]], inData_2[w_offset[i + 9]], inData_2[w_offset[i + 10]], inData_2[w_offset[i + 11]], inData_2[w_offset[i + 12]], inData_2[w_offset[i + 13]], inData_2[w_offset[i + 14]], inData_2[w_offset[i + 15]]);

        __m128i m_data_3 = _mm_setr_epi8(inData_3[w_offset[i + 0]], inData_3[w_offset[i + 1]], inData_3[w_offset[i + 2]], inData_3[w_offset[i + 3]], inData_3[w_offset[i + 4]], inData_3[w_offset[i + 5]], inData_3[w_offset[i + 6]], inData_3[w_offset[i + 7]], inData_3[w_offset[i + 8]], inData_3[w_offset[i + 9]], inData_3[w_offset[i + 10]], inData_3[w_offset[i + 11]], inData_3[w_offset[i + 12]], inData_3[w_offset[i + 13]], inData_3[w_offset[i + 14]], inData_3[w_offset[i + 15]]);

        _mm_storeu_si128((__m128i *)(outData_0 + i), m_data_0);
        _mm_storeu_si128((__m128i *)(outData_1 + i), m_data_1);
        _mm_storeu_si128((__m128i *)(outData_2 + i), m_data_2);
        _mm_storeu_si128((__m128i *)(outData_3 + i), m_data_3);
    }
    for (; i < outWidth; ++i) {
        outData_0[i] = inData_0[w_offset[i]];
        outData_1[i] = inData_1[w_offset[i]];
        outData_2[i] = inData_2[w_offset[i]];
        outData_3[i] = inData_3[w_offset[i]];
    }
}

static void resize_nearest_c3_w_fourline_kernel_u8(
    const uint8_t *inData_0,
    const uint8_t *inData_1,
    const uint8_t *inData_2,
    const uint8_t *inData_3,
    int32_t outWidth,
    int32_t *w_offset,
    uint8_t *outData_0,
    uint8_t *outData_1,
    uint8_t *outData_2,
    uint8_t *outData_3)
{
    int32_t i = 0;
    for (; i < outWidth; ++i) {
        outData_0[i * 3 + 0] = inData_0[w_offset[i] * 3 + 0];
        outData_0[i * 3 + 1] = inData_0[w_offset[i] * 3 + 1];
        outData_0[i * 3 + 2] = inData_0[w_offset[i] * 3 + 2];
        outData_1[i * 3 + 0] = inData_1[w_offset[i] * 3 + 0];
        outData_1[i * 3 + 1] = inData_1[w_offset[i] * 3 + 1];
        outData_1[i * 3 + 2] = inData_1[w_offset[i] * 3 + 2];
        outData_2[i * 3 + 0] = inData_2[w_offset[i] * 3 + 0];
        outData_2[i * 3 + 1] = inData_2[w_offset[i] * 3 + 1];
        outData_2[i * 3 + 2] = inData_2[w_offset[i] * 3 + 2];
        outData_3[i * 3 + 0] = inData_3[w_offset[i] * 3 + 0];
        outData_3[i * 3 + 1] = inData_3[w_offset[i] * 3 + 1];
        outData_3[i * 3 + 2] = inData_3[w_offset[i] * 3 + 2];
    }
}

static void resize_nearest_c4_w_fourline_kernel_u8(
    const uint8_t *inData_0,
    const uint8_t *inData_1,
    const uint8_t *inData_2,
    const uint8_t *inData_3,
    int32_t outWidth,
    int32_t *w_offset,
    uint8_t *outData_0,
    uint8_t *outData_1,
    uint8_t *outData_2,
    uint8_t *outData_3)
{
    int32_t i = 0;
    for (; i <= outWidth - 4; i += 4) {
        __m128 m_data_0 = _mm_setr_ps(*(const float *)(inData_0 + w_offset[i + 0] * 4),
                                      *(const float *)(inData_0 + w_offset[i + 1] * 4),
                                      *(const float *)(inData_0 + w_offset[i + 2] * 4),
                                      *(const float *)(inData_0 + w_offset[i + 3] * 4));
        __m128 m_data_1 = _mm_setr_ps(*(const float *)(inData_1 + w_offset[i + 0] * 4),
                                      *(const float *)(inData_1 + w_offset[i + 1] * 4),
                                      *(const float *)(inData_1 + w_offset[i + 2] * 4),
                                      *(const float *)(inData_1 + w_offset[i + 3] * 4));
        __m128 m_data_2 = _mm_setr_ps(*(const float *)(inData_2 + w_offset[i + 0] * 4),
                                      *(const float *)(inData_2 + w_offset[i + 1] * 4),
                                      *(const float *)(inData_2 + w_offset[i + 2] * 4),
                                      *(const float *)(inData_2 + w_offset[i + 3] * 4));
        __m128 m_data_3 = _mm_setr_ps(*(const float *)(inData_3 + w_offset[i + 0] * 4),
                                      *(const float *)(inData_3 + w_offset[i + 1] * 4),
                                      *(const float *)(inData_3 + w_offset[i + 2] * 4),
                                      *(const float *)(inData_3 + w_offset[i + 3] * 4));

        _mm_storeu_ps((float *)(outData_0 + i * 4), m_data_0);
        _mm_storeu_ps((float *)(outData_1 + i * 4), m_data_1);
        _mm_storeu_ps((float *)(outData_2 + i * 4), m_data_2);
        _mm_storeu_ps((float *)(outData_3 + i * 4), m_data_3);
    }
    for (; i < outWidth; ++i) {
        *(int32_t *)(outData_0 + i * 4) = *(const int32_t *)(inData_0 + w_offset[i] * 4);
        *(int32_t *)(outData_1 + i * 4) = *(const int32_t *)(inData_1 + w_offset[i] * 4);
        *(int32_t *)(outData_2 + i * 4) = *(const int32_t *)(inData_2 + w_offset[i] * 4);
        *(int32_t *)(outData_3 + i * 4) = *(const int32_t *)(inData_3 + w_offset[i] * 4);
    }
}

static void resize_nearest_c1_w_oneline_kernel_u8(
    const uint8_t *inData,
    int32_t outWidth,
    int32_t *w_offset,
    uint8_t *outData)
{
    int32_t i = 0;
    for (; i <= outWidth - 4; i += 4) {
        outData[i + 0] = inData[w_offset[i + 0]];
        outData[i + 1] = inData[w_offset[i + 1]];
        outData[i + 2] = inData[w_offset[i + 2]];
        outData[i + 3] = inData[w_offset[i + 3]];
    }
    for (; i < outWidth; ++i) {
        outData[i] = inData[w_offset[i]];
    }
}

static void resize_nearest_c3_w_oneline_kernel_u8(
    const uint8_t *inData,
    int32_t outWidth,
    int32_t *w_offset,
    uint8_t *outData)
{
    int32_t i = 0;
    for (; i < outWidth; ++i) {
        outData[i * 3 + 0] = inData[w_offset[i] * 3 + 0];
        outData[i * 3 + 1] = inData[w_offset[i] * 3 + 1];
        outData[i * 3 + 2] = inData[w_offset[i] * 3 + 2];
    }
}

static void resize_nearest_c4_w_oneline_kernel_u8(
    const uint8_t *inData,
    int32_t outWidth,
    int32_t *w_offset,
    uint8_t *outData)
{
    for (int32_t i = 0; i < outWidth; ++i) {
        *(int32_t *)(outData + i * 4) = *(const int32_t *)(inData + w_offset[i] * 4);
    }
}

static void resize_nearest_kernel_u8(
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
    uint64_t size_for_h_offset = (outHeight * sizeof(int32_t) + 128 - 1) / 128 * 128;
    uint64_t size_for_w_offset = (outWidth * sizeof(int32_t) + 128 - 1) / 128 * 128;
    uint64_t total_size        = size_for_h_offset + size_for_w_offset;

    void *temp_buffer = ppl::common::AlignedAlloc(total_size, 128);

    int32_t *h_offset = (int32_t *)temp_buffer;
    int32_t *w_offset = (int32_t *)((unsigned char *)h_offset + size_for_h_offset);

    resize_nearest_calc_offset_u8(inHeight, inWidth, outHeight, outWidth, h_offset, w_offset);

    int32_t i = 0;
    for (; i <= outHeight - 4; i += 4) {
        if (channels == 1) {
            resize_nearest_c1_w_fourline_kernel_u8(
                inData + h_offset[i + 0] * inWidthStride,
                inData + h_offset[i + 1] * inWidthStride,
                inData + h_offset[i + 2] * inWidthStride,
                inData + h_offset[i + 3] * inWidthStride,
                outWidth,
                w_offset,
                outData + (i + 0) * outWidthStride,
                outData + (i + 1) * outWidthStride,
                outData + (i + 2) * outWidthStride,
                outData + (i + 3) * outWidthStride);
        }
        if (channels == 3) {
            resize_nearest_c3_w_fourline_kernel_u8(
                inData + h_offset[i + 0] * inWidthStride,
                inData + h_offset[i + 1] * inWidthStride,
                inData + h_offset[i + 2] * inWidthStride,
                inData + h_offset[i + 3] * inWidthStride,
                outWidth,
                w_offset,
                outData + (i + 0) * outWidthStride,
                outData + (i + 1) * outWidthStride,
                outData + (i + 2) * outWidthStride,
                outData + (i + 3) * outWidthStride);
        }
        if (channels == 4) {
            resize_nearest_c4_w_fourline_kernel_u8(
                inData + h_offset[i + 0] * inWidthStride,
                inData + h_offset[i + 1] * inWidthStride,
                inData + h_offset[i + 2] * inWidthStride,
                inData + h_offset[i + 3] * inWidthStride,
                outWidth,
                w_offset,
                outData + (i + 0) * outWidthStride,
                outData + (i + 1) * outWidthStride,
                outData + (i + 2) * outWidthStride,
                outData + (i + 3) * outWidthStride);
        }
    }
    for (; i < outHeight; ++i) {
        int32_t h_idx = h_offset[i];
        if (channels == 1) {
            resize_nearest_c1_w_oneline_kernel_u8(inData + h_idx * inWidthStride,
                                                  outWidth,
                                                  w_offset,
                                                  outData + i * outWidthStride);
        }
        if (channels == 3) {
            resize_nearest_c3_w_oneline_kernel_u8(inData + h_idx * inWidthStride,
                                                  outWidth,
                                                  w_offset,
                                                  outData + i * outWidthStride);
        }
        if (channels == 4) {
            resize_nearest_c4_w_oneline_kernel_u8(inData + h_idx * inWidthStride,
                                                  outWidth,
                                                  w_offset,
                                                  outData + i * outWidthStride);
        }
    }

    ppl::common::AlignedFree(temp_buffer);
}

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 1>(
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

    resize_nearest_kernel_u8(
        inHeight, inWidth, inWidthStride, inData, 1, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 3>(
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

    resize_nearest_kernel_u8(
        inHeight, inWidth, inWidthStride, inData, 3, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 4>(
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

    resize_nearest_kernel_u8(
        inHeight, inWidth, inWidthStride, inData, 4, outHeight, outWidth, outWidthStride, outData);

    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
