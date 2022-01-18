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

#include <cassert>

#include <vector>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

template <typename T, int32_t channels>
::ppl::common::RetCode Add_fma(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData)
{
    if (nullptr == inData0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == inData1) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride0 <= 0 ||
        inWidthStride1 <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int32_t i = 0;
    for (; i <= height * width * channels - 32; i += 32) {
        __m256i vdata0 = _mm256_lddqu_si256((__m256i *)(inData0 + i));
        __m256i vdata1 = _mm256_lddqu_si256((__m256i *)(inData1 + i));
        __m256i vdst   = _mm256_adds_epu8(vdata0, vdata1);
        _mm256_storeu_si256((__m256i *)(outData + i), vdst);
    }
    for (; i < height * width * channels; i++) {
        outData[i] = sat_cast_u8(inData0[i] + inData1[i]);
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t channels>
::ppl::common::RetCode Mul_fma(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData,
    float alpha)
{
    if (nullptr == inData0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == inData1) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (height <= 0 ||
        width <= 0 ||
        inWidthStride0 <= 0 ||
        inWidthStride1 <= 0 ||
        outWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t i = 0;
    for (; i <= height * width * channels - 32; i += 32) {
        __m256i vdata00 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(inData0 + i + 0)));
        __m256i vdata01 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(inData0 + i + 16)));
        __m256i vdata10 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(inData1 + i + 0)));
        __m256i vdata11 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(inData1 + i + 16)));
        __m256i vdst0   = _mm256_abs_epi16(_mm256_mullo_epi16(vdata00, vdata10));
        __m256i vdst1   = _mm256_abs_epi16(_mm256_mullo_epi16(vdata01, vdata11));
        _mm256_storeu_si256((__m256i *)(outData + i), _mm256_permute4x64_epi64(_mm256_packus_epi16(vdst0, vdst1), 0b11011000));
    }
    for (; i < height * width * channels; i++) {
        outData[i] = sat_cast_u8(inData0[i] * inData1[i]);
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t channels>
::ppl::common::RetCode Subtract(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (channels == 1) {
        int32_t i = 0;
        for (; i <= height * width - 32; i += 32) {
            __m256i vdata   = _mm256_loadu_si256((__m256i *)(inData + i));
            __m256i vscalar = _mm256_set1_epi8(scalar[0]);
            __m256i vdst    = _mm256_subs_epu8(vdata, vscalar);
            _mm256_storeu_si256((__m256i *)(outData + i), vdst);
        }
        for (; i < height * width; i++) {
            outData[i] = sat_cast_u8(inData[i] - scalar[0]);
        }
    } else if (channels == 3) {
        uint8_t scalar_tmp[32] = {0};
        for (int32_t i = 0; i < 30; i += 3) {
            scalar_tmp[i + 0] = scalar[0];
            scalar_tmp[i + 1] = scalar[1];
            scalar_tmp[i + 2] = scalar[2];
        }
        int32_t j = 0;
        for (; j <= height * width * 3 - 30; j += 30) {
            __m256i vdata   = _mm256_lddqu_si256((__m256i *)(inData + j));
            __m256i vscalar = _mm256_lddqu_si256((__m256i *)scalar_tmp);
            __m256i vdst    = _mm256_subs_epu8(vdata, vscalar);
            _mm256_storeu_si256((__m256i *)(outData + j), vdst);
        }
        for (; j < height * width * 3; j += 3) {
            outData[j + 0] = sat_cast_u8(inData[j + 0] - scalar[0]);
            outData[j + 1] = sat_cast_u8(inData[j + 1] - scalar[1]);
            outData[j + 2] = sat_cast_u8(inData[j + 2] - scalar[2]);
        }
    } else if (channels == 4) {
        uint8_t scalar_tmp[32];
        for (int32_t i = 0; i < 32; i += 4) {
            scalar_tmp[i + 0] = scalar[0];
            scalar_tmp[i + 1] = scalar[1];
            scalar_tmp[i + 2] = scalar[2];
            scalar_tmp[i + 3] = scalar[3];
        }
        int32_t j = 0;
        for (; j <= height * width * 4 - 32; j += 32) {
            __m256i vdata   = _mm256_lddqu_si256((__m256i *)(inData + j));
            __m256i vscalar = _mm256_lddqu_si256((__m256i *)scalar_tmp);
            __m256i vdst    = _mm256_subs_epu8(vdata, vscalar);
            _mm256_storeu_si256((__m256i *)(outData + j), vdst);
        }
        for (; j < height * width * 4; j += 4) {
            outData[j + 0] = sat_cast_u8(inData[j + 0] - scalar[0]);
            outData[j + 1] = sat_cast_u8(inData[j + 1] - scalar[1]);
            outData[j + 2] = sat_cast_u8(inData[j + 2] - scalar[2]);
            outData[j + 3] = sat_cast_u8(inData[j + 3] - scalar[3]);
        }
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode Add_fma<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode Add_fma<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode Add_fma<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode Add_fma<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Add_fma<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Add_fma<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData);

template ::ppl::common::RetCode Mul_fma<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData,
    float alpha);

template ::ppl::common::RetCode Mul_fma<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData,
    float alpha);

template ::ppl::common::RetCode Mul_fma<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData,
    float alpha);

template ::ppl::common::RetCode Mul_fma<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Mul_fma<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Mul_fma<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData,
    float alpha);

template ::ppl::common::RetCode Subtract<1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode Subtract<3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData);

template ::ppl::common::RetCode Subtract<4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    const uint8_t *scalar,
    int32_t outWidthStride,
    uint8_t *outData);
}
}
}
} // namespace ppl::cv::x86::fma
