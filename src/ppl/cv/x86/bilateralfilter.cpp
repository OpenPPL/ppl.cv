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

#include "ppl/cv/x86/bilateralfilter.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/copymakeborder.h"
#include "ppl/cv/x86/util.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"

#include <string.h>
#include <limits.h>
#include <immintrin.h>
#include <cmath>

#include <algorithm>
#include <vector>

namespace ppl {
namespace cv {
namespace x86 {

template <int32_t cn>
void bilateralFilter_32f(
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

    for (i = 0; i < height; i++) {
        const float* sptr = (const float*)(temp + (i + radius) * tempstep) + radius * cn;
        float* dptr       = (float*)(dst + i * outWidthStride);

        if (cn == 1) {
            int32_t j = 0;
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
            int32_t j = 0;
            for (; j < width * 3; j += 3) {
                float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                float b0 = sptr[j], g0 = sptr[j + 1], r0 = sptr[j + 2];
                for (k = 0; k < maxk; k++) {
                    const float* sptr_k = sptr + j + space_ofs[k];
                    float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
                    float alpha = (float)((std::abs(b - b0) +
                                           std::abs(g - g0) + std::abs(r - r0)) *
                                          scale_index);
                    int32_t idx = floor(alpha);
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

template <int32_t cn>
static void bilateralFilter_b(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* src,
    int32_t outWidthStride,
    uint8_t* dst,
    int32_t d,
    float sigma_color,
    float sigma_space)
{
    double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
    int32_t i, j, k, maxk, radius;

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

    int32_t tempHeight = height + 2 * radius;
    int32_t tempWidth  = width + 2 * radius;

    int32_t tempstep = (tempWidth)*cn;
    std::vector<uint8_t> padded_images(tempHeight * tempWidth * cn);
    uint8_t* temp = padded_images.data();
    CopyMakeBorder<uint8_t, cn>(height, width, inWidthStride, src, tempHeight, tempWidth, tempstep, temp, ppl::cv::BORDER_REFLECT_101, 0);

    std::vector<float> _color_weight(cn * 255);
    std::vector<float> _space_weight(d * d);
    std::vector<int32_t> _space_ofs(d * d);
    float* color_weight = &_color_weight[0];
    float* space_weight = &_space_weight[0];
    int32_t* space_ofs  = &_space_ofs[0];

    for (i = 0; i < 255 * cn; i++)
        color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

    for (i = -radius, maxk = 0; i <= radius; i++) {
        for (j = -radius; j <= radius; j++) {
            double r = std::sqrt((double)i * i + (double)j * j);
            if (r > radius)
                continue;
            space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
            space_ofs[maxk++]  = (int32_t)(i * tempstep + j * cn);
        }
    }

    for (i = 0; i < height; i++) {
        const uint8_t* sptr = temp + (i + radius) * tempstep + radius * cn;
        uint8_t* dptr       = dst + i * outWidthStride;
        if (cn == 1) {
            int32_t j = 0;
            for (; j < width; j++) {
                float sum = 0, wsum = 0;
                int32_t val0 = sptr[j];
                for (k = 0; k < maxk; k++) {
                    int32_t val = sptr[j + space_ofs[k]];
                    float w     = space_weight[k] * color_weight[(size_t)std::abs(val - val0)];
                    sum += val * w;
                    wsum += w;
                }
                dptr[j] = (uint8_t)std::round(sum / wsum);
            }
        } else {
            for (j = 0; j < width * 3; j += 3) {
                float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                int32_t b0 = sptr[j], g0 = sptr[j + 1], r0 = sptr[j + 2];
                k = 0;
                for (; k < maxk; k++) {
                    const uint8_t* sptr_k = sptr + j + space_ofs[k];
                    int32_t b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
                    float w = space_weight[k] * color_weight[(size_t)(std::abs(b - b0) +
                                                                      std::abs(g - g0) + std::abs(r - r0))];
                    sum_b += b * w;
                    sum_g += g * w;
                    sum_r += r * w;
                    wsum += w;
                }
                wsum = 1.f / wsum;
                b0   = std::round(sum_b * wsum);
                g0   = std::round(sum_g * wsum);
                r0   = std::round(sum_r * wsum);
                dptr[j]     = (uint8_t)b0;
                dptr[j + 1] = (uint8_t)g0;
                dptr[j + 2] = (uint8_t)r0;
            }
        }
    }
}

template <>
::ppl::common::RetCode BilateralFilter<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t diameter,
    float color,
    float space,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    bilateralFilter_b<1>(height, width, inWidthStride, inData, outWidthStride, outData, diameter, color, space);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BilateralFilter<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t diameter,
    float color,
    float space,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    bilateralFilter_b<3>(height, width, inWidthStride, inData, outWidthStride, outData, diameter, color, space);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BilateralFilter<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t diameter,
    float color,
    float space,
    int32_t outWidthStride,
    float* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        bilateralFilter_32f_avx<1>(height, width, inWidthStride, inData, outWidthStride, outData, diameter, color, space);
    } else {
        bilateralFilter_32f<1>(height, width, inWidthStride, inData, outWidthStride, outData, diameter, color, space);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BilateralFilter<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t diameter,
    float color,
    float space,
    int32_t outWidthStride,
    float* outData,
    BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT_101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_AVX)) {
        bilateralFilter_32f_avx<3>(height, width, inWidthStride, inData, outWidthStride, outData, diameter, color, space);
    } else {
        bilateralFilter_32f<3>(height, width, inWidthStride, inData, outWidthStride, outData, diameter, color, space);
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
