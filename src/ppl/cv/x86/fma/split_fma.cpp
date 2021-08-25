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

#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "intrinutils_fma.hpp"
#include <vector>
#include <stdint.h>
#include <immintrin.h>

#define YMM_FP32_LANE_NUM 8

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

template <typename T, int nc>
::ppl::common::RetCode splitAOS2SOA(
    int height,
    int width,
    int inWidthStride,
    const T* in,
    int outWidthStride,
    T** out);

template <>
::ppl::common::RetCode splitAOS2SOA<float, 3>(
    int height,
    int width,
    int inWidthStride,
    const float* in,
    int outWidthStride,
    float** out)
{
    __m256i r_idx_vec = _mm256_set_epi32(5, 2, 7, 4, 1, 6, 3, 0);
    __m256i g_idx_vec = _mm256_set_epi32(6, 3, 0, 5, 2, 7, 4, 1);
    __m256i b_idx_vec = _mm256_set_epi32(7, 4, 1, 6, 3, 0, 5, 2);
    for (int h = 0; h < height; ++h) {
        const float* base_in = in + h * inWidthStride;
        float* base_r = out[0] + h * outWidthStride;
        float* base_g = out[1] + h * outWidthStride;
        float* base_b = out[2] + h * outWidthStride;
        for (int w = 0; w < width / YMM_FP32_LANE_NUM * YMM_FP32_LANE_NUM; w += YMM_FP32_LANE_NUM) {
            __m256 data0_vec = _mm256_loadu_ps(base_in + 3 * w);
            __m256 data1_vec = _mm256_loadu_ps(base_in + 3 * w + 8);
            __m256 data2_vec = _mm256_loadu_ps(base_in + 3 * w + 16);
            __m256 r_vec = _mm256_blend_ps(data0_vec, data1_vec, 0x92);
            r_vec        = _mm256_blend_ps(r_vec, data2_vec, 0x24);
            __m256 g_vec = _mm256_blend_ps(data2_vec, data0_vec, 0x92);
            g_vec        = _mm256_blend_ps(g_vec, data1_vec, 0x24);
            __m256 b_vec = _mm256_blend_ps(data1_vec, data2_vec, 0x92);
            b_vec        = _mm256_blend_ps(b_vec, data0_vec, 0x24);
            _mm256_storeu_ps(base_r + w, _mm256_permutevar8x32_ps(r_vec, r_idx_vec));
            _mm256_storeu_ps(base_g + w, _mm256_permutevar8x32_ps(g_vec, g_idx_vec));
            _mm256_storeu_ps(base_b + w, _mm256_permutevar8x32_ps(b_vec, b_idx_vec));
        }
        for (int w = width / YMM_FP32_LANE_NUM * YMM_FP32_LANE_NUM; w < width; ++w) {
            base_r[w] = base_in[3 * w];
            base_g[w] = base_in[3 * w + 1];
            base_b[w] = base_in[3 * w + 2];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode splitAOS2SOA<float, 4>(
    int height,
    int width,
    int inWidthStride,
    const float* in,
    int outWidthStride,
    float** out)
{
    for (int32_t h = 0; h < height; h++) {
        int32_t w = 0;
        const float* src_ptr = in + h * inWidthStride;
        float* dst0_ptr = out[0] + h * outWidthStride;
        float* dst1_ptr = out[1] + h * outWidthStride;
        float* dst2_ptr = out[2] + h * outWidthStride;
        float* dst3_ptr = out[3] + h * outWidthStride;
        for (; w <= width - 8; w += 8) {
            __m256 vr, vb, vg, va;
            v_load_deinterleave(src_ptr, vr, vb, vg, va);
            _mm256_storeu_ps(dst0_ptr, vr);
            _mm256_storeu_ps(dst1_ptr, vb);
            _mm256_storeu_ps(dst2_ptr, vg);
            _mm256_storeu_ps(dst3_ptr, va);
            src_ptr += 8 * 4;
            dst0_ptr += 8;
            dst1_ptr += 8;
            dst2_ptr += 8;
            dst3_ptr += 8;
        }
        for (int32_t i = 0; w < width; w++, i++) {
            dst0_ptr[i] = src_ptr[i * 4 + 0];
            dst1_ptr[i] = src_ptr[i * 4 + 1];
            dst2_ptr[i] = src_ptr[i * 4 + 2];
            dst3_ptr[i] = src_ptr[i * 4 + 3];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode splitAOS2SOA<uint8_t, 3>(
    int height,
    int width,
    int inWidthStride,
    const uint8_t* in,
    int outWidthStride,
    uint8_t** out)
{
    for (int32_t h = 0; h < height; h++) {
        int32_t w = 0;
        const uint8_t* src_ptr = in + h * inWidthStride;
        uint8_t* dst0_ptr = out[0] + h * outWidthStride;
        uint8_t* dst1_ptr = out[1] + h * outWidthStride;
        uint8_t* dst2_ptr = out[2] + h * outWidthStride;
        for (; w <= width - 32; w += 32) {
            __m256i vr, vb, vg;
            v_load_deinterleave(src_ptr, vr, vb, vg);
            _mm256_storeu_si256((__m256i*)dst0_ptr, vr);
            _mm256_storeu_si256((__m256i*)dst1_ptr, vb);
            _mm256_storeu_si256((__m256i*)dst2_ptr, vg);
            src_ptr += 32 * 3;
            dst0_ptr += 32;
            dst1_ptr += 32;
            dst2_ptr += 32;
        }
        for (int32_t i = 0; w < width; w++, i++) {
            dst0_ptr[i] = src_ptr[i * 3 + 0];
            dst1_ptr[i] = src_ptr[i * 3 + 1];
            dst2_ptr[i] = src_ptr[i * 3 + 2];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode splitAOS2SOA<uint8_t, 4>(
    int height,
    int width,
    int inWidthStride,
    const uint8_t* in,
    int outWidthStride,
    uint8_t** out)
{
    for (int32_t h = 0; h < height; h++) {
        int32_t w = 0;
        const uint8_t* src_ptr = in + h * inWidthStride;
        uint8_t* dst0_ptr = out[0] + h * outWidthStride;
        uint8_t* dst1_ptr = out[1] + h * outWidthStride;
        uint8_t* dst2_ptr = out[2] + h * outWidthStride;
        uint8_t* dst3_ptr = out[3] + h * outWidthStride;
        for (; w <= width - 32; w += 32) {
            __m256i vr, vb, vg, va;
            v_load_deinterleave(src_ptr, vr, vb, vg, va);
            _mm256_storeu_si256((__m256i*)dst0_ptr, vr);
            _mm256_storeu_si256((__m256i*)dst1_ptr, vb);
            _mm256_storeu_si256((__m256i*)dst2_ptr, vg);
            _mm256_storeu_si256((__m256i*)dst3_ptr, va);
            src_ptr += 32 * 4;
            dst0_ptr += 32;
            dst1_ptr += 32;
            dst2_ptr += 32;
            dst3_ptr += 32;
        }
        for (int32_t i = 0; w < width; w++, i++) {
            dst0_ptr[i] = src_ptr[i * 4 + 0];
            dst1_ptr[i] = src_ptr[i * 4 + 1];
            dst2_ptr[i] = src_ptr[i * 4 + 2];
            dst3_ptr[i] = src_ptr[i * 4 + 3];
        }
    }
    return ppl::common::RC_SUCCESS;
}

}
}
}
} // namespace ppl::cv::x86::fma
