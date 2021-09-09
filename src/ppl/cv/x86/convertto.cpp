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

#include "ppl/cv/x86/convertto.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include <string.h>
#include <immintrin.h>

#include <cmath>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

template <typename TSrc, typename TDst>
::ppl::common::RetCode ChangeDataTypeAndScale(
    int32_t height,
    int32_t width,
    int32_t nc,
    int32_t inWidthStride,
    const TSrc* inData,
    float scale,
    int32_t outWidthStride,
    TDst* outData);

template <>
::ppl::common::RetCode ChangeDataTypeAndScale<uint8_t, float>(
    int32_t height,
    int32_t width,
    int32_t nc,
    int32_t inWidthStride,
    const uint8_t* inData,
    float scale,
    int32_t outWidthStride,
    float* outData)
{
    __m128 scale_vec = _mm_set1_ps(scale);
    for (int32_t h = 0; h < height; ++h) {
        const uint8_t* base_in = inData + h * inWidthStride;
        float* base_out      = outData + h * outWidthStride;
        for (int32_t w = 0; w < (nc * width) / 4 * 4; w += 4) {
            __m128i data_u8x4_vec    = _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(base_in + w)));
            __m128i data_int32x4_vec = _mm_cvtepu8_epi32(data_u8x4_vec);
            __m128 data_fp32x4_vec   = _mm_cvtepi32_ps(data_int32x4_vec);
            data_fp32x4_vec          = _mm_mul_ps(data_fp32x4_vec, scale_vec);
            _mm_storeu_ps(base_out + w, data_fp32x4_vec);
        }
        for (int32_t w = (nc * width) / 4 * 4; w < (nc * width); ++w) {
            base_out[w] = scale * static_cast<float>(base_in[w]);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ChangeDataTypeAndScale<float, uint8_t>(
    int32_t height,
    int32_t width,
    int32_t nc,
    int32_t inWidthStride,
    const float* inData,
    float scale,
    int32_t outWidthStride,
    uint8_t* outData)
{
    __m128 scale_vec = _mm_set1_ps(scale);
    for (int32_t h = 0; h < height; ++h) {
        const float* base_in = inData + h * inWidthStride;
        uint8_t* base_out      = outData + h * outWidthStride;
        for (int32_t w = 0; w < (nc * width) / 16 * 16; w += 16) {
            __m128i data0_int32x4_vec = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(base_in + w), scale_vec));
            __m128i data1_int32x4_vec = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(base_in + w + 4), scale_vec));
            __m128i data2_int32x4_vec = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(base_in + w + 8), scale_vec));
            __m128i data3_int32x4_vec = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(base_in + w + 12), scale_vec));
            __m128i result_vec        = _mm_packus_epi16(_mm_packs_epi32(data0_int32x4_vec, data1_int32x4_vec), _mm_packs_epi32(data2_int32x4_vec, data3_int32x4_vec));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(base_out + w), result_vec);
        }
        for (int32_t w = (nc * width) / 16 * 16; w < (nc * width); ++w) {
            base_out[w] = static_cast<uint8_t>(std::min(std::max(static_cast<int32_t>(scale * base_in[w]), 0), 255));
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ConvertTo<float, 1, uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    float scale,
    int32_t outWidthStride,
    uint8_t* outData)
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
    return ChangeDataTypeAndScale<float, uint8_t>(height, width, 1, inWidthStride, inData, scale, outWidthStride, outData);
}

template <>
::ppl::common::RetCode ConvertTo<float, 3, uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    float scale,
    int32_t outWidthStride,
    uint8_t* outData)
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
    return ChangeDataTypeAndScale<float, uint8_t>(height, width, 3, inWidthStride, inData, scale, outWidthStride, outData);
}

template <>
::ppl::common::RetCode ConvertTo<float, 4, uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    float scale,
    int32_t outWidthStride,
    uint8_t* outData)
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
    return ChangeDataTypeAndScale<float, uint8_t>(height, width, 4, inWidthStride, inData, scale, outWidthStride, outData);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, 1, float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    float scale,
    int32_t outWidthStride,
    float* outData)
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
    return ChangeDataTypeAndScale<uint8_t, float>(height, width, 1, inWidthStride, inData, scale, outWidthStride, outData);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, 3, float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    float scale,
    int32_t outWidthStride,
    float* outData)
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
    return ChangeDataTypeAndScale<uint8_t, float>(height, width, 3, inWidthStride, inData, scale, outWidthStride, outData);
}

template <>
::ppl::common::RetCode ConvertTo<uint8_t, 4, float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    float scale,
    int32_t outWidthStride,
    float* outData)
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
    return ChangeDataTypeAndScale<uint8_t, float>(height, width, 4, inWidthStride, inData, scale, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::x86
