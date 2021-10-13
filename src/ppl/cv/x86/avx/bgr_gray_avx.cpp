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

#include "ppl/cv/x86/avx/intrinutils_avx.hpp"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"
#include <vector>
#include <stdint.h>
#include <cstring>
#include <immintrin.h>

namespace ppl {
namespace cv {
namespace x86 {

template <typename _Tp>
struct RGB2Gray {
    typedef _Tp channel_type;

    RGB2Gray(int32_t _srccn, int32_t blueIdx, const float *_coeffs)
        : srccn(_srccn)
    {
        static const float coeffs0[] = {0.299f, 0.587f, 0.114f};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3 * sizeof(coeffs[0]));
        if (blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp *src, _Tp *dst, int32_t n) const
    {
        int32_t scn = srccn;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for (int32_t i = 0; i < n; i++, src += scn)
            dst[i] = (src[0] * cb + src[1] * cg + src[2] * cr);
    }
    int32_t srccn;
    float coeffs[3];
};

template <>
struct RGB2Gray<float> {
    typedef float channel_type;

    RGB2Gray(int32_t _srccn, int32_t blueIdx, const float *_coeffs)
        : srccn(_srccn)
    {
        static const float coeffs0[] = {0.299f, 0.587f, 0.114f};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3 * sizeof(coeffs[0]));
        if (blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);

        core        = 1;
        bSupportAVX = ppl::common::CpuSupports(ppl::common::ISA_X86_AVX);
        if (bSupportAVX) {
            v_cb = _mm256_set1_ps(coeffs[0]);
            v_cg = _mm256_set1_ps(coeffs[1]);
            v_cr = _mm256_set1_ps(coeffs[2]);
        }
    }

    void process(__m256 v_b, __m256 v_g, __m256 v_r, __m256 &v_gray) const
    {
        v_gray = _mm256_mul_ps(v_r, v_cr);
        v_gray = _mm256_add_ps(v_gray, _mm256_mul_ps(v_g, v_cg));
        v_gray = _mm256_add_ps(v_gray, _mm256_mul_ps(v_b, v_cb));
    }

    void operator()(const float *src, float *dst, int32_t n) const
    {
        int32_t scn = srccn, i = 0;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];

        if (bSupportAVX) {
            if (scn == 3) {
                for (; i <= n - 8; i += 8, src += scn * 8) {
                    __m256 v_r0, v_g0, v_b0;
                    _mm256_deinterleave_ps(src, v_r0, v_g0, v_b0);

                    __m256 v_gray0;
                    process(v_r0, v_g0, v_b0, v_gray0);

                    _mm256_storeu_ps(dst + i, v_gray0);
                }
            } else if (scn == 4) {
                for (; i <= n - 8; i += 8, src += scn * 8) {
                    __m256 v_r0, v_g0, v_b0, v_a0;

                    _mm256_deinterleave_ps(src, v_r0, v_g0, v_b0, v_a0);

                    __m256 v_gray0;
                    process(v_r0, v_g0, v_b0, v_gray0);

                    _mm256_storeu_ps(dst + i, v_gray0);
                }
            }
        }

        for (; i < n; i++, src += scn)
            dst[i] = src[0] * cb + src[1] * cg + src[2] * cr;
    }

    int32_t srccn;
    float coeffs[3];
    __m256 v_cb, v_cg, v_cr;
    int32_t core;
    bool bSupportAVX;
};

template <>
::ppl::common::RetCode BGR2GRAYImage_avx<float, 3, float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
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
    const float *src  = inData;
    float *dst        = outData;
    RGB2Gray<float> s = RGB2Gray<float>(3, 0, NULL);
    for (int32_t i = 0; i < height; ++i) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2GRAYImage_avx<float, 4, float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
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
    const float *src  = inData;
    float *dst        = outData;
    RGB2Gray<float> s = RGB2Gray<float>(4, 0, NULL);
    for (int32_t i = 0; i < height; ++i) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2GRAYImage_avx<float, 3, float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
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
    const float *src  = inData;
    float *dst        = outData;
    RGB2Gray<float> s = RGB2Gray<float>(3, 2, NULL);
    for (int32_t i = 0; i < height; ++i) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2GRAYImage_avx<float, 4, float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData)
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
    const float *src  = inData;
    float *dst        = outData;
    RGB2Gray<float> s = RGB2Gray<float>(4, 2, NULL);
    for (int32_t i = 0; i < height; ++i) {
        s.operator()(src, dst, width);
        src += inWidthStride;
        dst += outWidthStride;
    }
    return ppl::common::RC_SUCCESS;
}
}
}
} // namespace ppl::cv::x86
