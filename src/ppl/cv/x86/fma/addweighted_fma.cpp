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

#include <algorithm>
#include <cmath>
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "internal_fma.hpp"
#include <immintrin.h>
#include <cmath>
#include <assert.h>

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

namespace {
uint8_t truncate(float v)
{
    return (std::min(std::max(std::round(v), 0.0f), 255.0f));
}
} // namespace

template <int32_t channels>
::ppl::common::RetCode addWighted_fma(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    uint8_t *outData)
{
    __m256 alpha_vec = _mm256_broadcast_ss(&alpha);
    __m256 beta_vec  = _mm256_broadcast_ss(&beta);
    __m256 gamma_vec = _mm256_broadcast_ss(&gamma);

    for (int32_t i = 0; i < height; ++i) {
        const uint8_t *base_in0 = inData0 + i * inWidthStride0;
        const uint8_t *base_in1 = inData1 + i * inWidthStride1;
        uint8_t *base_out       = outData + i * outWidthStride;
        int32_t j;
        for (j = 0; j <= width * channels - 16; j += 16) {
            __m128i data_in0_vec     = _mm_loadu_si128((const __m128i *)(base_in0 + j));
            __m256 data0_in0_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_in0_vec));
            __m256 data1_in0_f32_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_in0_vec), _mm_castsi128_ps(data_in0_vec)))));
            __m256 midval0_y0_vec    = _mm256_fmadd_ps(alpha_vec, data0_in0_f32_vec, gamma_vec);
            __m256 midval1_y0_vec    = _mm256_fmadd_ps(alpha_vec, data1_in0_f32_vec, gamma_vec);

            __m128i data_in1_vec       = _mm_loadu_si128((const __m128i *)(base_in1 + j));
            __m256 data0_in1_f32_vec   = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(data_in1_vec));
            __m256 data1_in1_f32_vec   = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(data_in1_vec), _mm_castsi128_ps(data_in1_vec)))));
            __m256 accumulator0_y0_vec = _mm256_fmadd_ps(beta_vec, data0_in1_f32_vec, midval0_y0_vec);
            __m256 accumulator1_y0_vec = _mm256_fmadd_ps(beta_vec, data1_in1_f32_vec, midval1_y0_vec);

            __m256i result_int16_vec = _mm256_packs_epi32(_mm256_cvtps_epi32(accumulator0_y0_vec), _mm256_cvtps_epi32(accumulator1_y0_vec));
            __m256i result_u8_vec    = _mm256_packus_epi16(result_int16_vec, result_int16_vec);
            _mm_storeu_si128((__m128i *)(base_out + j), _mm_unpacklo_epi32(_mm256_extractf128_si256(result_u8_vec, 0), _mm256_extractf128_si256(result_u8_vec, 1)));
        }
        for (; j < width * channels; ++j) {
            base_out[j] = truncate(base_in0[j] * alpha + base_in1[j] * beta + gamma);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode addWighted_fma<1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    uint8_t *outData);
template ::ppl::common::RetCode addWighted_fma<3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    uint8_t *outData);
template ::ppl::common::RetCode addWighted_fma<4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    float alpha,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    uint8_t *outData);

}
}
}
} // namespace ppl::cv::x86::fma