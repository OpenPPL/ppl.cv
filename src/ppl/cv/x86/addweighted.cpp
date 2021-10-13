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

#include "ppl/cv/x86/addweighted.h"
#include "ppl/cv/types.h"
#include "ppl/cv/x86/util.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>
#include <limits.h>
#include <immintrin.h>
#include <algorithm>
namespace ppl {
namespace cv {
namespace x86 {

::ppl::common::RetCode addWeighted_f32(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride0,
    const float *inData0,
    float alpha,
    int32_t inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    float *outData)
{
    if (nullptr == inData0 && nullptr == inData1 && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride0 == 0 || inWidthStride1 == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    width *= channels;
    __m128 alpha_vec = _mm_set1_ps(alpha);
    __m128 beta_vec  = _mm_set1_ps(beta);
    __m128 gamma_vec = _mm_set1_ps(gamma);

    for (int32_t i = 0; i < height; ++i) {
        const float *base_in0 = inData0 + i * inWidthStride0;
        const float *base_in1 = inData1 + i * inWidthStride1;
        float *base_out       = outData + i * outWidthStride;
        for (int32_t j = 0; j < width / 4 * 4; j += 4) {
            __m128 data0_vec = _mm_loadu_ps(base_in0 + j);
            __m128 data1_vec = _mm_loadu_ps(base_in1 + j);
            __m128 dst_vec   = gamma_vec;
            dst_vec          = _mm_add_ps(_mm_add_ps(_mm_mul_ps(data0_vec, alpha_vec), _mm_mul_ps(data1_vec, beta_vec)), dst_vec);
            _mm_storeu_ps(base_out + j, dst_vec);
        }
        for (int32_t j = width / 4 * 4; j < width; ++j) {
            base_out[j] = inData0[j] * alpha + inData1[j] * beta + gamma;
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t channels>
::ppl::common::RetCode addWeighted_u8(
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
    if (nullptr == inData0 && nullptr == inData1 && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride0 == 0 || inWidthStride1 == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return ppl::cv::x86::fma::addWighted_fma<channels>(height, width, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
    }

    for (int32_t i = 0; i < height; ++i) {
        const uint8_t *base_in0 = inData0 + i * inWidthStride0;
        const uint8_t *base_in1 = inData1 + i * inWidthStride1;
        uint8_t *base_out       = outData + i * outWidthStride;
        for (int32_t j = 0; j < width * channels; ++j) {
            base_out[j] = sat_cast_u8(base_in0[j] * alpha + base_in1[j] * beta + gamma);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode AddWeighted<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    float alpha,
    int32_t inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    float *outData)
{
    return addWeighted_f32(height, width, 1, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    float alpha,
    int32_t inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    float *outData)
{
    return addWeighted_f32(height, width, 3, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    float alpha,
    int32_t inWidthStride1,
    const float *inData1,
    float beta,
    float gamma,
    int32_t outWidthStride,
    float *outData)
{
    return addWeighted_f32(height, width, 4, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<uint8_t, 1>(
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
    return addWeighted_u8<1>(height, width, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<uint8_t, 3>(
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
    return addWeighted_u8<3>(height, width, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

template <>
::ppl::common::RetCode AddWeighted<uint8_t, 4>(
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
    return addWeighted_u8<4>(height, width, inWidthStride0, inData0, alpha, inWidthStride1, inData1, beta, gamma, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::x86
