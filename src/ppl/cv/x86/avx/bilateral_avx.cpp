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
#include "intrinutils_avx.hpp"
#include "internal_avx.hpp"
#include "ppl/common/sys.h"
#include "ppl/cv/x86/util.hpp"
#include "ppl/cv/x86/copymakeborder.h"

#include <immintrin.h>
#include <cmath>
#include <cstring>

#include <vector>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

template <int32_t cn>
void bilateralFilter_32f_avx(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* src,
    int32_t outWidthStride,
    float* dst,
    int32_t d,
    float sigma_color,
    float sigma_space)
{
    double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
    int32_t i, j, k, maxk, radius;
    double minValSrc = 100000, maxValSrc = -1;
    const int32_t kExpNumBinsPerChannel = 1 << 12;
    int32_t kExpNumBins                 = 0;
    float lastExpVal                    = 1.f;
    float len, scale_index;

    if (sigma_color <= 0)
        sigma_color = 1;
    if (sigma_space <= 0)
        sigma_space = 1;

    if (d <= 0)
        radius = std::round(sigma_space * 1.5);
    else
        radius = d / 2;
    radius = std::max(radius, 1);
    d      = radius * 2 + 1;
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width * cn; j++) {
            float val = src[i * inWidthStride + j];
            if (val > maxValSrc)
                maxValSrc = val;
            if (val < minValSrc)
                minValSrc = val;
        }
    }

    int32_t tempHeight = height + 2 * radius;
    int32_t tempWidth  = width + 2 * radius;
    int32_t tempstep = (tempWidth)*cn;
    std::vector<float> padded_images(tempHeight * tempWidth * cn);
    float* temp = padded_images.data();
    CopyMakeBorder<float, cn>(height, width, inWidthStride, src, tempHeight, tempWidth, tempstep, temp, ppl::cv::BORDER_REFLECT_101, 0);

    std::vector<float> _space_weight(d * d);
    std::vector<int32_t> _space_ofs(d * d);
    float* space_weight = &_space_weight[0];
    int32_t* space_ofs  = &_space_ofs[0];

    len         = (float)(maxValSrc - minValSrc) * cn;
    kExpNumBins = kExpNumBinsPerChannel * cn;
    std::vector<float> _expLUT(kExpNumBins + 2);
    float* expLUT = &_expLUT[0];
    scale_index = kExpNumBins / len;

    for (i = 0; i < kExpNumBins + 2; i++) {
        if (lastExpVal > 0.f) {
            double val = i / scale_index;
            expLUT[i]  = (float)std::exp(val * val * gauss_color_coeff);
            lastExpVal = expLUT[i];
        } else
            expLUT[i] = 0.f;
    }
    for (i = -radius, maxk = 0; i <= radius; i++) {
        for (j = -radius; j <= radius; j++) {
            double r = std::sqrt((double)i * i + (double)j * j);
            if (r > radius)
                continue;
            space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
            space_ofs[maxk++]  = (int32_t)(i * (tempstep) + j * cn);
        }
    }
#ifdef _MSC_VER
    int __declspec(align(32)) idxBuf[8] ;
    static const unsigned int __declspec(align(32)) bufSignMask[] = {
        0x80000000, 0x80000000, 0x80000000,0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
#else
     int idxBuf[8] __attribute__((aligned(64)));
     static const unsigned int bufSignMask[] __attribute__((aligned(64))) = {
         0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000 };

#endif
    for (i = 0; i < height; i++) {
        const float* sptr = (const float*)(temp + (i + radius) * tempstep) + radius * cn;
        float* dptr       = (float*)(dst + i * outWidthStride);

        __m256 _scale_index, _signMask;
        _scale_index = _mm256_broadcast_ss(&scale_index);
        _signMask    = _mm256_load_ps((const float*)bufSignMask);

        if (cn == 1) {
            for (j = 0; j <= width - 8; j += 8) {
                __m256 sum   = _mm256_setzero_ps();
                __m256 wsum  = sum;
                __m256 _val0 = _mm256_loadu_ps(sptr + j);
                for (k = 0; k < maxk; k++) {
                    __m256 _val   = _mm256_loadu_ps(sptr + j + space_ofs[k]);
                    __m256 _alpha = _mm256_mul_ps(_mm256_andnot_ps(_signMask, _mm256_sub_ps(_val, _val0)), _scale_index);
                    __m256i _idx  = _mm256_cvtps_epi32(_alpha);
                    _mm256_store_si256((__m256i*)idxBuf, _idx);
                    _alpha          = _mm256_sub_ps(_alpha, _mm256_cvtepi32_ps(_idx));
                    __m256 _explut  = _mm256_set_ps(expLUT[idxBuf[7]], expLUT[idxBuf[6]], expLUT[idxBuf[5]], expLUT[idxBuf[4]], expLUT[idxBuf[3]], expLUT[idxBuf[2]], expLUT[idxBuf[1]], expLUT[idxBuf[0]]);
                    __m256 _explut1 = _mm256_set_ps(expLUT[idxBuf[7] + 1], expLUT[idxBuf[6] + 1], expLUT[idxBuf[5] + 1], expLUT[idxBuf[4] + 1], expLUT[idxBuf[3] + 1], expLUT[idxBuf[2] + 1], expLUT[idxBuf[1] + 1], expLUT[idxBuf[0] + 1]);

                    __m256 _sw = _mm256_broadcast_ss(space_weight + k);
                    __m256 _w  = _mm256_mul_ps(_sw, _mm256_add_ps(_explut, _mm256_mul_ps(_alpha, _mm256_sub_ps(_explut1, _explut))));

                    sum  = _mm256_add_ps(sum, _mm256_mul_ps(_val, _w));
                    wsum = _mm256_add_ps(wsum, _w);
                }
                _mm256_storeu_ps(dptr + j, _mm256_div_ps(sum, wsum));
            }

            for (; j < width; j++) {
                float sum = 0, wsum = 0;
                float val0 = sptr[j];
                for (k = 0; k < maxk; k++) {
                    float val   = sptr[j + space_ofs[k]];
                    float alpha = (float)(std::abs(val - val0) * scale_index);
                    int32_t idx = int32_t(alpha);
                    alpha -= idx;
                    float w = space_weight[k] * (expLUT[idx] + alpha * (expLUT[idx + 1] - expLUT[idx]));
                    sum += val * w;
                    wsum += w;
                }
                dptr[j] = (float)(sum / wsum);
            }
        } else if (cn == 3) {
            for (j = 0; j <= width * 3 - 24; j += 24) {
                __m256 sum_b = _mm256_setzero_ps();
                __m256 sum_g = sum_b, sum_r = sum_b, wsum = sum_b;

                __m256 b0;
                __m256 g0;
                __m256 r0;
                _mm256_deinterleave_ps(sptr + j, b0, g0, r0);

                for (k = 0; k < maxk; k++) {
                    __m256 b1, g1, r1;
                    _mm256_deinterleave_ps(sptr + j + space_ofs[k], b1, g1, r1);
                    __m256 sum3nc = _mm256_add_ps(_mm256_andnot_ps(_signMask, _mm256_sub_ps(b1, b0)), _mm256_andnot_ps(_signMask, _mm256_sub_ps(g1, g0)));
                    sum3nc        = _mm256_add_ps(sum3nc, _mm256_andnot_ps(_signMask, _mm256_sub_ps(r1, r0)));
                    __m256 _alpha = _mm256_mul_ps(sum3nc, _scale_index);
                    __m256i _idx  = _mm256_cvtps_epi32(_alpha);
                    _mm256_store_si256((__m256i*)idxBuf, _idx);
                    _alpha          = _mm256_sub_ps(_alpha, _mm256_cvtepi32_ps(_idx));
                    __m256 _explut  = _mm256_set_ps(expLUT[idxBuf[7]], expLUT[idxBuf[6]], expLUT[idxBuf[5]], expLUT[idxBuf[4]], expLUT[idxBuf[3]], expLUT[idxBuf[2]], expLUT[idxBuf[1]], expLUT[idxBuf[0]]);
                    __m256 _explut1 = _mm256_set_ps(expLUT[idxBuf[7] + 1], expLUT[idxBuf[6] + 1], expLUT[idxBuf[5] + 1], expLUT[idxBuf[4] + 1], expLUT[idxBuf[3] + 1], expLUT[idxBuf[2] + 1], expLUT[idxBuf[1] + 1], expLUT[idxBuf[0] + 1]);

                    __m256 _sw = _mm256_broadcast_ss(space_weight + k);
                    __m256 _w  = _mm256_mul_ps(_sw, _mm256_add_ps(_explut, _mm256_mul_ps(_alpha, _mm256_sub_ps(_explut1, _explut))));

                    sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(b1, _w));
                    sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(g1, _w));
                    sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(r1, _w));
                    wsum  = _mm256_add_ps(wsum, _w);
                }
                wsum = _mm256_div_ps(_mm256_set1_ps(1.f), wsum);
                b0   = _mm256_mul_ps(sum_b, wsum);
                g0   = _mm256_mul_ps(sum_g, wsum);
                r0   = _mm256_mul_ps(sum_r, wsum);
                _mm256_interleave1_ps(dptr + j, b0, g0, r0);
            }

            for (; j < width * 3; j += 3) {
                float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                float b0 = sptr[j], g0 = sptr[j + 1], r0 = sptr[j + 2];
                for (k = 0; k < maxk; k++) {
                    const float* sptr_k = sptr + j + space_ofs[k];
                    float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
                    float alpha = (float)((std::abs(b - b0) +
                                           std::abs(g - g0) + std::abs(r - r0)) *
                                          scale_index);
                    int32_t idx = std::floor(alpha);
                    alpha -= idx;
                    float w = space_weight[k] * (expLUT[idx] + alpha * (expLUT[idx + 1] - expLUT[idx]));
                    sum_b += b * w;
                    sum_g += g * w;
                    sum_r += r * w;
                    wsum += w;
                }
                wsum = 1.f / wsum;
                b0   = sum_b * wsum;
                g0   = sum_g * wsum;
                r0   = sum_r * wsum;
                dptr[j]     = b0;
                dptr[j + 1] = g0;
                dptr[j + 2] = r0;
            }
        }
    }
}

template void bilateralFilter_32f_avx<1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* src,
    int32_t outWidthStride,
    float* dst,
    int32_t d,
    float sigma_color,
    float sigma_space);

template void bilateralFilter_32f_avx<3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* src,
    int32_t outWidthStride,
    float* dst,
    int32_t d,
    float sigma_color,
    float sigma_space);

}
}
} // namespace ppl::cv::x86
