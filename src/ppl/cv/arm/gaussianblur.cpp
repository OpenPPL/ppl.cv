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

#include "ppl/cv/arm/gaussianblur.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include <string.h>
#include <cmath>
#include "ppl/cv/arm/copymakeborder.h"
#include <limits.h>
#include <algorithm>
#include <vector>

#include "filter_engine.hpp"

namespace ppl {
namespace cv {
namespace arm {

static std::vector<double> getGaussianKernel_double(float sigma, int32_t n)
{
    const int32_t SMALL_GAUSSIAN_SIZE = 7;
    static const double small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] = {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}};
    bool fix = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0;
    const double *fixed_kernel = fix ? small_gaussian_tab[n >> 1] : 0;
    std::vector<double> kernel(n);
    double sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale2X = -0.5 / (sigmaX * sigmaX);
    double sum = 0;

    int32_t i;
    for (i = 0; i < n; i++) {
        double x = i - (n - 1) * 0.5;
        double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X * x * x);
        kernel[i] = (double)t;
        sum += kernel[i];
    }

    sum = 1. / sum;
    for (i = 0; i < n; i++) {
        kernel[i] = static_cast<double>(kernel[i] * sum);
    }
    return kernel;
}

static std::vector<float> getGaussianKernel(float sigma, int32_t n)
{
    std::vector<float> kernel(n);
    std::vector<double> doubleKernel = getGaussianKernel_double(sigma, n);
    for (int i = 0; i < n; i++) {
        kernel[i] = static_cast<float>(doubleKernel[i]);
    }
    return kernel;
}

enum imageDepth { sense8U = 1, sense32F = 2 };

static int32_t senseRound(float value)
{
    return lrintf(value);
}

static void createGaussianKernels(std::vector<float> &k, int32_t ksize, float sigma, imageDepth depth)
{
    // automatic detection of kernel size from sigma
    if (ksize <= 0 && sigma > 0) ksize = senseRound(sigma * (depth == sense8U ? 3 : 4) * 2 + 1) | 1;

    // assert(ksize > 0 && ksize % 2 == 1);

    sigma = std::max(sigma, 0.f);

    k = getGaussianKernel(sigma, ksize);
}

static void createGaussianKernels_double(std::vector<double> &k, int32_t ksize, float sigma, imageDepth depth)
{
    // automatic detection of kernel size from sigma
    if (ksize <= 0 && sigma > 0) ksize = senseRound(sigma * (depth == sense8U ? 3 : 4) * 2 + 1) | 1;

    // assert(ksize > 0 && ksize % 2 == 1);

    sigma = std::max(sigma, 0.f);

    k = getGaussianKernel_double(sigma, ksize);
}

static uint16_t saturate_cast_u32_u16(uint32_t val)
{
    if (val > 65535) {
        return 65535;
    } else {
        return static_cast<uint16_t>(val);
    }
}

static uint8_t saturate_cast_u32_u8(uint32_t val)
{
    if (val > 255) {
        return 255;
    } else {
        return static_cast<uint8_t>(val);
    }
}

template <typename ST, typename DT>
struct SymmRowFilter {
    SymmRowFilter(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
    }

    void operator()(const ST *_src, DT *_dst, int width, int cn) const
    {
        int ksize = kernel.size();
        int ksize2 = ksize / 2;
        const float *kx = kernel.data() + ksize2;
        const ST *src0 = _src + ksize2 * cn;
        DT *dst = _dst;

        width *= cn;

        int i = 0;
        for (; i < width; i++) {
            const ST *src = src0 + i;
            float s = kx[0] * src[0];
            for (int k = 1; k <= ksize2; k++) {
                s += (src[k * cn] + src[-k * cn]) * kx[k];
            }
            // need saturate cast here for uint8_t
            dst[i] = s;
        }
    }
    std::vector<float> kernel;
};

template <>
struct SymmRowFilter<float, float> {
    SymmRowFilter(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
    }

    void operator()(const float *_src, float *_dst, int width, int cn) const
    {
        int ksize = kernel.size();
        int ksize2 = ksize / 2;
        const float *kx = kernel.data() + ksize2;
        const float *src0 = _src + ksize2 * cn;
        float *dst = _dst;

        width *= cn;

        int i = 0;
        for (; i <= width - 16; i += 16) {
            float32x4_t vScale = vdupq_n_f32(kx[0]);

            const float *src = src0 + i;
            prefetch(src);
            float32x4_t vIn0 = vld1q_f32(src + 0);
            float32x4_t vIn1 = vld1q_f32(src + 4);
            float32x4_t vIn2 = vld1q_f32(src + 8);
            float32x4_t vIn3 = vld1q_f32(src + 12);

            float32x4_t vRes0 = vmulq_f32(vIn0, vScale);
            float32x4_t vRes1 = vmulq_f32(vIn1, vScale);
            float32x4_t vRes2 = vmulq_f32(vIn2, vScale);
            float32x4_t vRes3 = vmulq_f32(vIn3, vScale);

            for (int k = 1; k <= ksize2; k++) {
                vScale = vdupq_n_f32(kx[k]);

                const float *srcNegK = src - k * cn;
                const float *srcK = src + k * cn;
                prefetch(srcNegK);
                float32x4_t vInNegK0 = vld1q_f32(srcNegK + 0);
                float32x4_t vInNegK1 = vld1q_f32(srcNegK + 4);
                float32x4_t vInNegK2 = vld1q_f32(srcNegK + 8);
                float32x4_t vInNegK3 = vld1q_f32(srcNegK + 12);
                prefetch(srcK);
                float32x4_t vInK0 = vld1q_f32(srcK + 0);
                float32x4_t vInK1 = vld1q_f32(srcK + 4);
                float32x4_t vInK2 = vld1q_f32(srcK + 8);
                float32x4_t vInK3 = vld1q_f32(srcK + 12);

                vIn0 = vaddq_f32(vInK0, vInNegK0);
                vIn1 = vaddq_f32(vInK1, vInNegK1);
                vIn2 = vaddq_f32(vInK2, vInNegK2);
                vIn3 = vaddq_f32(vInK3, vInNegK3);

                vRes0 = vfmaq_f32(vRes0, vIn0, vScale);
                vRes1 = vfmaq_f32(vRes1, vIn1, vScale);
                vRes2 = vfmaq_f32(vRes2, vIn2, vScale);
                vRes3 = vfmaq_f32(vRes3, vIn3, vScale);
            }

            vst1q_f32(dst + i + 0, vRes0);
            vst1q_f32(dst + i + 4, vRes1);
            vst1q_f32(dst + i + 8, vRes2);
            vst1q_f32(dst + i + 12, vRes3);
        }
        for (; i < width; i++) {
            const float *src = src0 + i;
            float s = kx[0] * src[0];
            for (int k = 1; k <= ksize2; k++) {
                s += (src[k * cn] + src[-k * cn]) * kx[k];
            }
            dst[i] = s;
        }
    }
    std::vector<float> kernel;
};

template <typename ST, typename DT>
struct SymmColumnFilter {
    SymmColumnFilter(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
        ksize = _kernel.size();
    }

    void operator()(const ST *const *src, DT *dst, int dststep, int count, int width)
    {
        int ksize2 = ksize / 2;
        const ST *ky = kernel.data() + ksize2;
        int i, k;
        src += ksize2;

        for (; count--; dst += dststep, src++) {
            DT *D = (DT *)dst;
            i = 0;
            for (; i < width; i++) {
                ST s0 = ky[0] * src[0][i];
                for (k = 1; k <= ksize2; k++) {
                    s0 += ky[k] * (src[k][i] + src[-k][i]);
                }
                // need saturate cast here for uint8_t
                D[i] = s0;
            }
        }
    }

    std::vector<float> kernel;
    int ksize;
};

template <>
struct SymmColumnFilter<float, float> {
    SymmColumnFilter(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
        ksize = _kernel.size();
    }

    void operator()(const float *const *src, float *dst, int dststep, int count, int width)
    {
        int ksize2 = ksize / 2;
        const float *ky = kernel.data() + ksize2;
        int i, k;
        src += ksize2;

        for (; count--; dst += dststep, src++) {
            float *D = (float *)dst;
            i = 0;
            for (; i <= width - 16; i += 16) {
                float32x4_t vScale = vdupq_n_f32(ky[0]);
                prefetch(src[0] + i);
                float32x4_t vIn0 = vld1q_f32(src[0] + i + 0);
                float32x4_t vIn1 = vld1q_f32(src[0] + i + 4);
                float32x4_t vIn2 = vld1q_f32(src[0] + i + 8);
                float32x4_t vIn3 = vld1q_f32(src[0] + i + 12);

                float32x4_t vRes0 = vmulq_f32(vIn0, vScale);
                float32x4_t vRes1 = vmulq_f32(vIn1, vScale);
                float32x4_t vRes2 = vmulq_f32(vIn2, vScale);
                float32x4_t vRes3 = vmulq_f32(vIn3, vScale);

                for (k = 1; k <= ksize2; k++) {
                    vScale = vdupq_n_f32(ky[k]);
                    prefetch(src[k] + i);
                    float32x4_t vInK0 = vld1q_f32(src[k] + i + 0);
                    float32x4_t vInK1 = vld1q_f32(src[k] + i + 4);
                    float32x4_t vInK2 = vld1q_f32(src[k] + i + 8);
                    float32x4_t vInK3 = vld1q_f32(src[k] + i + 12);
                    prefetch(src[-k] + i);
                    float32x4_t vInNegK0 = vld1q_f32(src[-k] + i + 0);
                    float32x4_t vInNegK1 = vld1q_f32(src[-k] + i + 4);
                    float32x4_t vInNegK2 = vld1q_f32(src[-k] + i + 8);
                    float32x4_t vInNegK3 = vld1q_f32(src[-k] + i + 12);

                    vIn0 = vaddq_f32(vInK0, vInNegK0);
                    vIn1 = vaddq_f32(vInK1, vInNegK1);
                    vIn2 = vaddq_f32(vInK2, vInNegK2);
                    vIn3 = vaddq_f32(vInK3, vInNegK3);

                    vRes0 = vfmaq_f32(vRes0, vIn0, vScale);
                    vRes1 = vfmaq_f32(vRes1, vIn1, vScale);
                    vRes2 = vfmaq_f32(vRes2, vIn2, vScale);
                    vRes3 = vfmaq_f32(vRes3, vIn3, vScale);
                }

                vst1q_f32(dst + i + 0, vRes0);
                vst1q_f32(dst + i + 4, vRes1);
                vst1q_f32(dst + i + 8, vRes2);
                vst1q_f32(dst + i + 12, vRes3);
            }
            for (; i < width; i++) {
                float s0 = ky[0] * src[0][i];
                for (k = 1; k <= ksize2; k++)
                    s0 += ky[k] * (src[k][i] + src[-k][i]);
                D[i] = s0;
            }
        }
    }

    std::vector<float> kernel;
    int ksize;
};

using ufixedpoint16 = uint16_t;
using ufixedpoint32 = uint32_t;
static constexpr int fixedShift = 8;
static constexpr int fixedShift_ufix32 = 16;
ufixedpoint16 fixed_convert_u8_to_ufix16(uint8_t v)
{
    return (static_cast<uint16_t>(v) << fixedShift);
}

ufixedpoint32 fixed_convert_ufix16_to_ufix32(ufixedpoint16 v)
{
    return (static_cast<uint32_t>(v) << (fixedShift_ufix32 - fixedShift));
}

ufixedpoint16 fixed_fn_mul(ufixedpoint16 a, uint32_t b)
{
    return saturate_cast_u32_u16(static_cast<uint32_t>(a) * b);
}

ufixedpoint32 fixed_ff_mul(ufixedpoint16 a, uint32_t b)
{
    return static_cast<uint32_t>(a) * b;
}

ufixedpoint16 fixed_ff_sat_add(ufixedpoint16 a, ufixedpoint16 b)
{
    uint16_t res = a + b;
    return a > res ? static_cast<uint16_t>(0xffff) : res;
}

ufixedpoint32 fixed_ff_sat_add_32(ufixedpoint32 a, ufixedpoint32 b)
{
    uint32_t res = a + b;
    return a > res ? static_cast<uint32_t>(0xffffffff) : res;
}

uint8_t fixed_sat_cast_ufix16_to_u8(ufixedpoint16 val)
{
    uint8_t res = saturate_cast_u32_u8((val + ((1 << fixedShift) >> 1)) >> fixedShift);
    return res;
}

uint8_t fixed_sat_cast_ufix32_to_u8(ufixedpoint32 val)
{
    // should saturate add be used here? but opencv simply used add.
    uint32_t res = (val + ((1 << fixedShift_ufix32) >> 1)) >> fixedShift_ufix32;
    return res > 255 ? 255 : res;
}

struct FixedPointSymmRowFilter {
    FixedPointSymmRowFilter(const std::vector<ufixedpoint16> &_kernel)
    {
        kernel = _kernel;
    }

    void operator()(const uint8_t *_src, ufixedpoint16 *_dst, int width, int cn) const
    {
        int ksize = kernel.size();
        int ksize2 = ksize / 2;
        const ufixedpoint16 *kx = kernel.data() + ksize2;
        const uint8_t *src0 = _src + ksize2 * cn;
        uint16_t *dst = _dst;

        width *= cn;

        int i = 0;
        for (; i <= width - 16; i += 16) {
            const uint8_t *src = src0 + i;
            prefetch(src);

            uint8x16_t vInU8 = vld1q_u8(src);
            uint16x8_t vIn0 = vmovl_u8(vget_low_u8(vInU8));
            uint16x8_t vIn1 = vmovl_high_u8(vInU8);
            uint16x8_t vScale = vdupq_n_u16(kx[0]);

            // ufixedpoint16 s = fixed_fn_mul(kx[0], src[0]);
            uint32x4_t vMullRes00 = vmull_u16(vget_low_u16(vScale), vget_low_u16(vIn0));
            uint32x4_t vMullRes01 = vmull_high_u16(vScale, vIn0);
            uint32x4_t vMullRes10 = vmull_u16(vget_low_u16(vScale), vget_low_u16(vIn1));
            uint32x4_t vMullRes11 = vmull_high_u16(vScale, vIn1);

            uint16x4_t vOut0_low = vqmovn_u32(vMullRes00);
            uint16x8_t vOut0 = vqmovn_high_u32(vOut0_low, vMullRes01);
            uint16x4_t vOut1_low = vqmovn_u32(vMullRes10);
            uint16x8_t vOut1 = vqmovn_high_u32(vOut1_low, vMullRes11);

            for (int k = 1; k <= ksize2; k++) {
                // s = fixed_ff_sat_add(s, fixed_fn_mul(kx[k], static_cast<uint16_t>(src[k * cn]) + src[-k * cn]));
                vScale = vdupq_n_u16(kx[k]);

                prefetch(src - k * cn);
                uint8x16_t vInNegKU8 = vld1q_u8(src - k * cn);
                uint16x8_t vInNegK0 = vmovl_u8(vget_low_u8(vInNegKU8));
                uint16x8_t vInNegK1 = vmovl_high_u8(vInNegKU8);
                prefetch(src + k * cn);
                uint8x16_t vInKU8 = vld1q_u8(src + k * cn);

                // surely won't overflow here
                vIn0 = vaddw_u8(vInNegK0, vget_low_u8(vInKU8));
                vIn1 = vaddw_high_u8(vInNegK1, vInKU8);

                vMullRes00 = vmull_u16(vget_low_u16(vScale), vget_low_u16(vIn0));
                vMullRes01 = vmull_high_u16(vScale, vIn0);
                vMullRes10 = vmull_u16(vget_low_u16(vScale), vget_low_u16(vIn1));
                vMullRes11 = vmull_high_u16(vScale, vIn1);

                uint16x4_t vMullResU16_0_low = vqmovn_u32(vMullRes00);
                uint16x8_t vMullResU16_0 = vqmovn_high_u32(vMullResU16_0_low, vMullRes01);
                uint16x4_t vMullResU16_1_low = vqmovn_u32(vMullRes10);
                uint16x8_t vMullResU16_1 = vqmovn_high_u32(vMullResU16_1_low, vMullRes11);

                vOut0 = vqaddq_u16(vOut0, vMullResU16_0);
                vOut1 = vqaddq_u16(vOut1, vMullResU16_1);
            }
            vst1q_u16(dst + i, vOut0);
            vst1q_u16(dst + i + 8, vOut1);
        }

        for (; i < width; i++) {
            const uint8_t *src = src0 + i;
            ufixedpoint16 s = fixed_fn_mul(kx[0], src[0]);
            for (int k = 1; k <= ksize2; k++) {
                s = fixed_ff_sat_add(s, fixed_fn_mul(kx[k], static_cast<uint16_t>(src[k * cn]) + src[-k * cn]));
            }
            dst[i] = s;
        }
    }
    std::vector<ufixedpoint16> kernel;
};

struct FixedPointSymmColumnFilter {
    FixedPointSymmColumnFilter(const std::vector<ufixedpoint16> &_kernel)
    {
        kernel = _kernel;
        ksize = _kernel.size();
    }

    void operator()(const ufixedpoint16 *const *src, uint8_t *dst, int dststep, int count, int width)
    {
        int ksize2 = ksize / 2;
        const ufixedpoint16 *ky = kernel.data() + ksize2;
        int i, k;
        src += ksize2;

        for (; count--; dst += dststep, src++) {
            i = 0;
            for (; i <= width - 16; i += 16) {
                prefetch(src[0] + i);
                uint16x8_t vIn0 = vld1q_u16(src[0] + i);
                uint16x8_t vIn1 = vld1q_u16(src[0] + i + 8);
                uint16x8_t vScale = vdupq_n_u16(ky[0]);

                // ufixedpoint32 s0 = fixed_ff_mul(ky[0], src[0][i]);
                uint32x4_t vResU32_00 = vmull_u16(vget_low_u16(vScale), vget_low_u16(vIn0));
                uint32x4_t vResU32_01 = vmull_high_u16(vScale, vIn0);
                uint32x4_t vResU32_10 = vmull_u16(vget_low_u16(vScale), vget_low_u16(vIn1));
                uint32x4_t vResU32_11 = vmull_high_u16(vScale, vIn1);

                for (k = 1; k <= ksize2; k++) {
                    // s0 = fixed_ff_sat_add_32(s0, fixed_ff_mul(ky[k], static_cast<uint32_t>(src[k][i]) + src[-k][i]));
                    uint32x4_t vScaleU32 = vdupq_n_u32(ky[k]);

                    prefetch(src[-k] + i);
                    uint16x8_t vInNegK0 = vld1q_u16(src[-k] + i);
                    uint16x8_t vInNegK1 = vld1q_u16(src[-k] + i + 8);
                    prefetch(src[k] + i);
                    uint16x8_t vInK0 = vld1q_u16(src[k] + i);
                    uint16x8_t vInK1 = vld1q_u16(src[k] + i + 8);

                    uint32x4_t vInU32_00 = vaddl_u16(vget_low_u16(vInNegK0), vget_low_u16(vInK0));
                    uint32x4_t vInU32_01 = vaddl_high_u16(vInNegK0, vInK0);
                    uint32x4_t vInU32_10 = vaddl_u16(vget_low_u16(vInNegK1), vget_low_u16(vInK1));
                    uint32x4_t vInU32_11 = vaddl_high_u16(vInNegK1, vInK1);

                    // won't overflow here since we are actullay doing (uint16_t + uint16_t) * uint16_t
                    uint32x4_t vMulRes00 = vmulq_u32(vScaleU32, vInU32_00);
                    uint32x4_t vMulRes01 = vmulq_u32(vScaleU32, vInU32_01);
                    uint32x4_t vMulRes10 = vmulq_u32(vScaleU32, vInU32_10);
                    uint32x4_t vMulRes11 = vmulq_u32(vScaleU32, vInU32_11);

                    vResU32_00 = vqaddq_u32(vResU32_00, vMulRes00);
                    vResU32_01 = vqaddq_u32(vResU32_01, vMulRes01);
                    vResU32_10 = vqaddq_u32(vResU32_10, vMulRes10);
                    vResU32_11 = vqaddq_u32(vResU32_11, vMulRes11);
                }

                // use vqrshrn_n_u32 for shift and satruating cast process
                uint16x4_t vOutU16_00 = vqrshrn_n_u32(vResU32_00, fixedShift_ufix32);
                uint16x8_t vOutU16_0 = vqrshrn_high_n_u32(vOutU16_00, vResU32_01, fixedShift_ufix32);
                uint16x4_t vOutU16_10 = vqrshrn_n_u32(vResU32_10, fixedShift_ufix32);
                uint16x8_t vOutU16_1 = vqrshrn_high_n_u32(vOutU16_10, vResU32_11, fixedShift_ufix32);

                uint8x8_t vOutU8_00 = vqmovn_u16(vOutU16_0);
                uint8x16_t vOutU8_0 = vqmovn_high_u16(vOutU8_00, vOutU16_1);

                vst1q_u8(dst + i, vOutU8_0);
            }

            for (; i < width; i++) {
                ufixedpoint32 s0 = fixed_ff_mul(ky[0], src[0][i]);
                for (k = 1; k <= ksize2; k++) {
                    s0 = fixed_ff_sat_add_32(s0, fixed_ff_mul(ky[k], static_cast<uint32_t>(src[k][i]) + src[-k][i]));
                }
                dst[i] = fixed_sat_cast_ufix32_to_u8(s0);
            }
        }
    }

    std::vector<ufixedpoint16> kernel;
    int ksize;
};

struct FixedPointSymmGaussianRowFilter {
    // specilized for Gaussian, since all kernel values sum up to 1, so we can ignore overflow
    FixedPointSymmGaussianRowFilter(const std::vector<ufixedpoint16> &_kernel)
    {
        kernel = _kernel;
    }

    void operator()(const uint8_t *_src, ufixedpoint16 *_dst, int width, int cn) const
    {
        int ksize = kernel.size();
        int ksize2 = ksize / 2;
        const ufixedpoint16 *kx = kernel.data() + ksize2;
        const uint8_t *src0 = _src + ksize2 * cn;
        uint16_t *dst = _dst;

        width *= cn;

        int i = 0;
        for (; i <= width - 16; i += 16) {
            const uint8_t *src = src0 + i;
            prefetch(src);

            uint8x16_t vInU8 = vld1q_u8(src);
            uint16x8_t vIn0 = vmovl_u8(vget_low_u8(vInU8));
            uint16x8_t vIn1 = vmovl_high_u8(vInU8);
            uint16x8_t vScale = vdupq_n_u16(kx[0]);

            // ufixedpoint16 s = fixed_fn_mul(kx[0], src[0]);
            // vIn0 is less than 0xff, vScale is less than 0x10000, so no overflow
            uint16x8_t vOut0 = vmulq_u16(vScale, vIn0);
            uint16x8_t vOut1 = vmulq_u16(vScale, vIn1);

            for (int k = 1; k <= ksize2; k++) {
                // s = fixed_ff_sat_add(s, fixed_fn_mul(kx[k], static_cast<uint16_t>(src[k * cn]) + src[-k * cn]));
                vScale = vdupq_n_u16(kx[k]);

                prefetch(src - k * cn);
                uint8x16_t vInNegKU8 = vld1q_u8(src - k * cn);
                uint16x8_t vInNegK0 = vmovl_u8(vget_low_u8(vInNegKU8));
                uint16x8_t vInNegK1 = vmovl_high_u8(vInNegKU8);
                prefetch(src + k * cn);
                uint8x16_t vInKU8 = vld1q_u8(src + k * cn);

                // surely won't overflow here
                vIn0 = vaddw_u8(vInNegK0, vget_low_u8(vInKU8));
                vIn1 = vaddw_high_u8(vInNegK1, vInKU8);

                uint16x8_t vMulRes0 = vmulq_u16(vScale, vIn0);
                uint16x8_t vMulRes1 = vmulq_u16(vScale, vIn1);

                vOut0 = vqaddq_u16(vOut0, vMulRes0);
                vOut1 = vqaddq_u16(vOut1, vMulRes1);
            }
            vst1q_u16(dst + i, vOut0);
            vst1q_u16(dst + i + 8, vOut1);
        }

        for (; i < width; i++) {
            const uint8_t *src = src0 + i;
            ufixedpoint16 s = fixed_fn_mul(kx[0], src[0]);
            for (int k = 1; k <= ksize2; k++) {
                s = fixed_ff_sat_add(s, fixed_fn_mul(kx[k], static_cast<uint16_t>(src[k * cn]) + src[-k * cn]));
            }
            dst[i] = s;
        }
    }
    std::vector<ufixedpoint16> kernel;
};

struct FixedPointSymmRowFilter3N121 {
    FixedPointSymmRowFilter3N121() {}

    void operator()(const uint8_t *_src, ufixedpoint16 *_dst, int width, int cn) const
    {
        const uint8_t *src0 = _src + cn;
        uint16_t *dst = _dst;

        width *= cn;

        int i = 0;
        for (; i <= width - 16; i += 16) {
            const uint8_t *src = src0 + i;
            prefetch(src);
            uint8x16_t vIn = vld1q_u8(src);
            prefetch(src - cn);
            uint8x16_t vInNegCn = vld1q_u8(src - cn);
            prefetch(src + cn);
            uint8x16_t vInCn = vld1q_u8(src + cn);

            uint16x8_t vIn0 = vshll_n_u8(vget_low_u8(vIn), fixedShift - 1);
            uint16x8_t vIn1 = vshll_high_n_u8(vIn, fixedShift - 1);
            uint16x8_t vInNegCn0 = vshll_n_u8(vget_low_u8(vInNegCn), fixedShift - 2);
            uint16x8_t vInNegCn1 = vshll_high_n_u8(vInNegCn, fixedShift - 2);
            uint16x8_t vInCn0 = vshll_n_u8(vget_low_u8(vInCn), fixedShift - 2);
            uint16x8_t vInCn1 = vshll_high_n_u8(vInCn, fixedShift - 2);

            uint16x8_t vOut0 = vqaddq_u16(vIn0, vInNegCn0);
            uint16x8_t vOut1 = vqaddq_u16(vIn1, vInNegCn1);
            vOut0 = vqaddq_u16(vOut0, vInCn0);
            vOut1 = vqaddq_u16(vOut1, vInCn1);

            vst1q_u16(dst + i, vOut0);
            vst1q_u16(dst + i + 8, vOut1);
        }
        for (; i < width; i++) {
            const uint8_t *src = src0 + i;
            ufixedpoint16 s =
                fixed_ff_sat_add(fixed_convert_u8_to_ufix16(src[0]) >> 1, fixed_convert_u8_to_ufix16(src[cn]) >> 2);
            s = fixed_ff_sat_add(s, fixed_convert_u8_to_ufix16(src[-cn]) >> 2);
            dst[i] = s;
        }
    }
};

struct FixedPointSymmColumnFilter3N121 {
    FixedPointSymmColumnFilter3N121() {}

    void operator()(const ufixedpoint16 *const *src, uint8_t *dst, int dststep, int count, int width)
    {
        src += 1;

        for (; count--; dst += dststep, src++) {
            int i = 0;

            for (; i <= width - 16; i += 16) {
                prefetch(src[0] + i);
                uint16x8_t vIn0 = vld1q_u16(src[0] + i);
                uint16x8_t vIn1 = vld1q_u16(src[0] + i + 8);
                prefetch(src[-1] + i);
                uint16x8_t vInNegOne0 = vld1q_u16(src[-1] + i);
                uint16x8_t vInNegOne1 = vld1q_u16(src[-1] + i + 8);
                prefetch(src[1] + i);
                uint16x8_t vInOne0 = vld1q_u16(src[1] + i);
                uint16x8_t vInOne1 = vld1q_u16(src[1] + i + 8);

                uint32x4_t vOut00 = vaddl_u16(vget_low_u16(vIn0), vget_low_u16(vIn0));
                uint32x4_t vOut01 = vaddl_high_u16(vIn0, vIn0);
                uint32x4_t vOut10 = vaddl_u16(vget_low_u16(vIn1), vget_low_u16(vIn1));
                uint32x4_t vOut11 = vaddl_high_u16(vIn1, vIn1);

                vOut00 = vqaddq_u32(vOut00, vaddl_u16(vget_low_u16(vInNegOne0), vget_low_u16(vInOne0)));
                vOut01 = vqaddq_u32(vOut01, vaddl_high_u16(vInNegOne0, vInOne0));
                vOut10 = vqaddq_u32(vOut10, vaddl_u16(vget_low_u16(vInNegOne1), vget_low_u16(vInOne1)));
                vOut11 = vqaddq_u32(vOut11, vaddl_high_u16(vInNegOne1, vInOne1));

                // use vqrshrn_n_u32 for shift and satruating cast process
                // originally ufix16, but shr-ed 2 bit in add (0.25 * a + 0.5 * b + 0.25 * c = ((a+b+b+c) >> 2))
                constexpr int shiftVal = fixedShift_ufix32 - fixedShift + 2;
                uint16x4_t vOutU16_00 = vqrshrn_n_u32(vOut00, shiftVal);
                uint16x8_t vOutU16_0 = vqrshrn_high_n_u32(vOutU16_00, vOut01, shiftVal);
                uint16x4_t vOutU16_10 = vqrshrn_n_u32(vOut10, shiftVal);
                uint16x8_t vOutU16_1 = vqrshrn_high_n_u32(vOutU16_10, vOut11, shiftVal);

                uint8x8_t vOutU8_00 = vqmovn_u16(vOutU16_0);
                uint8x16_t vOutU8_0 = vqmovn_high_u16(vOutU8_00, vOutU16_1);

                vst1q_u8(dst + i, vOutU8_0);
            }

            for (; i < width; i++) {
                ufixedpoint32 s0 = fixed_convert_ufix16_to_ufix32(src[0][i]) >> 1;
                s0 = fixed_ff_sat_add_32(s0, fixed_convert_ufix16_to_ufix32(src[1][i]) >> 2);
                s0 = fixed_ff_sat_add_32(s0, fixed_convert_ufix16_to_ufix32(src[-1][i]) >> 2);
                dst[i] = fixed_sat_cast_ufix32_to_u8(s0);
            }
        }
    }
};

static void getGaussianKernelFixedPoint_ED(std::vector<int64_t> &result,
                                           const std::vector<double> kernel,
                                           int fractionBits)
{
    const int n = (int)kernel.size();

    int64_t fractionMultiplier = 1 << fractionBits;
    double fractionMultiplier_d = fractionMultiplier;

    result.resize(n);

    int n2_ = n / 2; // n is odd
    double err = 0;
    int64_t sum = 0;
    for (int i = 0; i < n2_; i++) {
        double adj_v = kernel[i] * fractionMultiplier_d + err;
        int64_t v0 = lrintf(adj_v);
        err = adj_v - v0;

        result[i] = v0;
        result[n - 1 - i] = v0;
        sum += v0;
    }
    sum *= 2;
    int64_t v_center = fractionMultiplier - sum;
    result[n2_] = v_center;
    return;
}

static void getFixedpointGaussianKernel(std::vector<ufixedpoint16> &res, int n, double sigma)
{
    std::vector<double> res_d;
    createGaussianKernels_double(res_d, n, sigma, sense8U);

    std::vector<int64_t> fixed_256;
    getGaussianKernelFixedPoint_ED(fixed_256, res_d, 8);

    res.resize(n);
    for (int i = 0; i < n; i++) {
        res[i] = static_cast<ufixedpoint16>(fixed_256[i]);
    }
    return;
}

template <int cn>
void GaussianBlur_f(int32_t height,
                    int32_t width,
                    int32_t inWidthStride,
                    const float *inData,
                    int32_t kernel_len,
                    float sigma,
                    int32_t outWidthStride,
                    float *outData,
                    ppl::cv::BorderType border_type)
{
    std::vector<float> kernel;
    createGaussianKernels(kernel, kernel_len, sigma, sense32F);

    SymmRowFilter<float, float> rowFilter = SymmRowFilter<float, float>(kernel);
    SymmColumnFilter<float, float> colFilter = SymmColumnFilter<float, float>(kernel);
    SeparableFilterEngine<float, float, float, SymmRowFilter<float, float>, SymmColumnFilter<float, float>> engine(
        height, width, cn, kernel_len, kernel_len, border_type, 0, rowFilter, colFilter);

    engine.process(inData, inWidthStride, outData, outWidthStride);
}

template <int cn>
void GaussianBlur_b(int32_t height,
                    int32_t width,
                    int32_t inWidthStride,
                    const uint8_t *inData,
                    int32_t kernel_len,
                    float sigma,
                    int32_t outWidthStride,
                    uint8_t *outData,
                    ppl::cv::BorderType border_type)
{
    if (kernel_len == 3 and sigma == 0.0f) {
        std::vector<ufixedpoint16> kernel;
        getFixedpointGaussianKernel(kernel, kernel_len, sigma);

        FixedPointSymmRowFilter3N121 rowFilter = FixedPointSymmRowFilter3N121();
        FixedPointSymmColumnFilter3N121 colFilter = FixedPointSymmColumnFilter3N121();
        SeparableFilterEngine<uint8_t,
                              ufixedpoint16,
                              uint8_t,
                              FixedPointSymmRowFilter3N121,
                              FixedPointSymmColumnFilter3N121>
            engine(height, width, cn, kernel_len, kernel_len, border_type, 0, rowFilter, colFilter);

        engine.process(inData, inWidthStride, outData, outWidthStride);
    } else {
        std::vector<ufixedpoint16> kernel;
        getFixedpointGaussianKernel(kernel, kernel_len, sigma);

        FixedPointSymmGaussianRowFilter rowFilter = FixedPointSymmGaussianRowFilter(kernel);
        FixedPointSymmColumnFilter colFilter = FixedPointSymmColumnFilter(kernel);
        SeparableFilterEngine<uint8_t,
                              ufixedpoint16,
                              uint8_t,
                              FixedPointSymmGaussianRowFilter,
                              FixedPointSymmColumnFilter>
            engine(height, width, cn, kernel_len, kernel_len, border_type, 0, rowFilter, colFilter);

        engine.process(inData, inWidthStride, outData, outWidthStride);
    }
}

template <>
::ppl::common::RetCode GaussianBlur<float, 1>(int32_t height,
                                              int32_t width,
                                              int32_t inWidthStride,
                                              const float *inData,
                                              int32_t kernel_len,
                                              float sigma,
                                              int32_t outWidthStride,
                                              float *outData,
                                              BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride < width || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    GaussianBlur_f<1>(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GaussianBlur<float, 3>(int32_t height,
                                              int32_t width,
                                              int32_t inWidthStride,
                                              const float *inData,
                                              int32_t kernel_len,
                                              float sigma,
                                              int32_t outWidthStride,
                                              float *outData,
                                              BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride < width || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    GaussianBlur_f<3>(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GaussianBlur<float, 4>(int32_t height,
                                              int32_t width,
                                              int32_t inWidthStride,
                                              const float *inData,
                                              int32_t kernel_len,
                                              float sigma,
                                              int32_t outWidthStride,
                                              float *outData,
                                              BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride < width || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    GaussianBlur_f<4>(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);

    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GaussianBlur<uint8_t, 1>(int32_t height,
                                                int32_t width,
                                                int32_t inWidthStride,
                                                const uint8_t *inData,
                                                int32_t kernel_len,
                                                float sigma,
                                                int32_t outWidthStride,
                                                uint8_t *outData,
                                                BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride < width || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    GaussianBlur_b<1>(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GaussianBlur<uint8_t, 3>(int32_t height,
                                                int32_t width,
                                                int32_t inWidthStride,
                                                const uint8_t *inData,
                                                int32_t kernel_len,
                                                float sigma,
                                                int32_t outWidthStride,
                                                uint8_t *outData,
                                                BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride < width || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    GaussianBlur_b<3>(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GaussianBlur<uint8_t, 4>(int32_t height,
                                                int32_t width,
                                                int32_t inWidthStride,
                                                const uint8_t *inData,
                                                int32_t kernel_len,
                                                float sigma,
                                                int32_t outWidthStride,
                                                uint8_t *outData,
                                                BorderType border_type)
{
    if (nullptr == inData || nullptr == outData) { return ppl::common::RC_INVALID_VALUE; }
    if (width == 0 || height == 0 || inWidthStride < width || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    GaussianBlur_b<4>(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
