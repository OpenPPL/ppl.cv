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

#include "ppl/cv/x86/gaussianblur.h"
#include "ppl/cv/x86/copymakeborder.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/common/x86/sysinfo.h"
#include "ppl/common/sys.h"
#include "ppl/cv/types.h"
#include <string.h>
#include <cmath>

#include <vector>
#include <limits.h>
#include <immintrin.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace x86 {

static int32_t borderInterpolate(int32_t p, int32_t len, BorderType borderType = BORDER_REFLECT_101)
{
    if (borderType == ppl::cv::BORDER_REFLECT_101) {
        do p = p < 0 ? (-p) : 2 * len - p - 2;
        while((unsigned)p >= (unsigned)len);
    } else if (borderType == ppl::cv::BORDER_REFLECT) {
        do p = p < 0 ? (-p - 1) : 2 * len - p - 1;
        while((unsigned)p >= (unsigned)len);
    } else if (borderType == ppl::cv::BORDER_REPLICATE) {
        p = (p < 0) ? 0 : len - 1;
    } else if (borderType == ppl::cv::BORDER_CONSTANT) {
        p = -1;
    }
    return p;
}

template <typename T>
static void makeborder_onfly(
    const T *src,
    int32_t srcHeight,
    int32_t srcWidth,
    int32_t srcstep,
    T *left_right,
    int32_t lrstep,
    T *up_down,
    int32_t udstep,
    int32_t cn,
    int32_t radius)
{
    int32_t i, j, k;
    int32_t srcWidth_cs = srcWidth;
    srcWidth *= cn;
    int32_t elemSize     = sizeof(T);
    int32_t left         = radius;
    int32_t right        = radius;
    float *left_right_cs = left_right;
    float *up_down_cs    = up_down;

    left_right = left_right + radius * lrstep + radius * cn;
    for (i = 0; i < radius; i++) {
        j = borderInterpolate(i - radius, srcHeight);
        memcpy(up_down + (i)*udstep + radius * cn, src + j * srcstep, srcWidth * elemSize);
    }
    for (; i < 3 * radius; i++) {
        memcpy(up_down + (i)*udstep + radius * cn, src + (i - radius) * srcstep, srcWidth * elemSize);
    }

    for (i = 0; i < radius; i++) {
        j = borderInterpolate(i + srcHeight, srcHeight);
        memcpy(up_down + (i + 3 * radius + 2 * radius) * udstep + radius * cn, src + j * srcstep, srcWidth * elemSize);
    }
    for (i = 0; i < 2 * radius; i++) {
        memcpy(up_down + (i + 3 * radius) * udstep + radius * cn, src + (srcHeight - 2 * radius + i) * srcstep, srcWidth * elemSize);
    }

    //make tab
    std::vector<int32_t> tab((2 * radius) * cn);
    for (i = 0; i < left; i++) {
        j = borderInterpolate(i - left, srcWidth_cs) * cn;
        for (k = 0; k < cn; k++)
            tab[i * cn + k] = j + k;
    }

    for (i = 0; i < right; i++) {
        j = borderInterpolate(srcWidth_cs + i, srcWidth_cs) * cn;
        for (k = 0; k < cn; k++)
            tab[(i + left) * cn + k] = j + k;
    }

    left *= cn;
    right *= cn;

    up_down = up_down + radius * cn;
    for (i = 0; i < 6 * radius; i++, up_down += udstep) {
        for (j = 0; j < left; j++)
            up_down[j - left] = up_down[tab[j]];
        for (j = 0; j < right; j++)
            up_down[j + srcWidth] = up_down[tab[j + left]];
    }

    for (i = 0; i < srcHeight; i++, left_right += lrstep, src += srcstep) {
        memcpy(left_right, src, 2 * radius * cn * elemSize);
        memcpy(left_right + 2 * radius * cn, src + srcWidth - 2 * radius * cn, 2 * radius * cn * elemSize);
        for (j = 0; j < left; j++)
            left_right[j - left] = src[tab[j]];
        for (j = 0; j < right; j++)
            left_right[j + 4 * radius * cn] = src[tab[j + left]];
    }

    //up border border
    T *left_right_shift = left_right_cs + radius * cn;
    up_down             = up_down_cs;
    for (i = 0; i < radius; i++) {
        T *left_right_index = left_right_shift + i * lrstep;
        T *up_downtemp      = up_down + (i)*udstep + radius * cn;
        memcpy(left_right_index, up_downtemp, 2 * radius * cn * elemSize);
        memcpy(left_right_index + 2 * radius * cn, up_downtemp + srcWidth - 2 * radius * cn, 2 * radius * cn * elemSize);
        for (j = 0; j < left; j++) {
            left_right_index[j - left] = up_downtemp[tab[j]];
        }
        for (j = 0; j < right; j++)
            left_right_index[j + 4 * radius * cn] = up_downtemp[tab[j + left]];
    }
    //down border border
    left_right_shift = left_right_cs + (srcHeight + radius) * lrstep + radius * cn;
    for (i = 0; i < radius; i++) {
        T *up_downtemp      = up_down + (i + 5 * radius) * udstep + radius * cn;
        T *left_right_index = left_right_shift + i * lrstep;
        memcpy(left_right_index, up_downtemp, 2 * radius * cn * elemSize);
        memcpy(left_right_index + 2 * radius * cn, up_downtemp + srcWidth - 2 * radius * cn, 2 * radius * cn * elemSize);
        for (j = 0; j < left; j++) {
            left_right_index[j - left] = up_downtemp[tab[j]];
        }
        for (j = 0; j < right; j++)
            left_right_index[j + 4 * radius * cn] = up_downtemp[tab[j + left]];
    }
}

static std::vector<float> getGaussianKernel(double sigma, int32_t n)
{
    const int32_t SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
        {
            {1.f},
            {0.25f, 0.5f, 0.25f},
            {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
            {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}};
    bool fix                  = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0;
    const float *fixed_kernel = fix ? small_gaussian_tab[n >> 1] : 0;
    std::vector<float> kernel(n);
    double sigmaX  = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale2X = -0.5 / (sigmaX * sigmaX);
    double sum     = 0;

    int32_t i;
    for (i = 0; i < n; i++) {
        double x  = i - (n - 1) * 0.5;
        double t  = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X * x * x);
        kernel[i] = (float)t;
        sum += kernel[i];
    }

    sum = 1. / sum;
    for (i = 0; i < n; i++) {
        kernel[i] = float(kernel[i] * sum);
    }
    return kernel;
}
enum imageDepth {
    sense8U  = 1,
    sense32F = 2
};
static int32_t senseRound(double value)
{
    __m128d t = _mm_set_sd(value);
    return _mm_cvtsd_si32(t);
}
static void createGaussianKernels(std::vector<float> &k, int32_t ksize, double sigma, imageDepth depth)
{
    // automatic detection of kernel size from sigma
    if (ksize <= 0 && sigma > 0)
        ksize = senseRound(sigma * (depth == sense8U ? 3 : 4) * 2 + 1) | 1;

    // assert(ksize > 0 && ksize % 2 == 1);

    sigma = std::max(sigma, 0.);

    k = getGaussianKernel(sigma, ksize);
}

struct RowVec_32f {
    RowVec_32f(const std::vector<float> &_kernel)
    {
        kernel      = _kernel;
        core        = 1;
        bSupportAVX = true;
    }

    void operator()(const float *_src, float *_dst, int32_t width, int32_t cn) const
    {
        int32_t _ksize    = kernel.size();
        const float *src0 = (const float *)_src;
        float *dst        = (float *)_dst;

        int32_t i = 0, k;
        width *= cn;

        if (bSupportAVX) {
            for (; i <= width - 16; i += 16) {
                const float *src = src0 + i;
                __m256 f, s0 = _mm256_setzero_ps(), s1 = s0, x0, x1;
                for (k = 0; k < _ksize; k++, src += cn) {
                    //f = _mm256_set1_ps(kernel[k]);
                    f  = _mm256_broadcast_ss(&kernel[k]);
                    x0 = _mm256_loadu_ps(src);
                    x1 = _mm256_loadu_ps(src + 8);
                    s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
                    s1 = _mm256_add_ps(s1, _mm256_mul_ps(x1, f));
                }
                _mm256_store_ps(dst + i, s0);
                _mm256_store_ps(dst + i + 8, s1);
            }
        }

        for (; i < width; i++) {
            const float *src = src0 + i;
            float s          = 0;
            for (k = 0; k < _ksize; k++, src += cn) {
                s += src[0] * kernel[k];
            }
            dst[i] = s;
        }
    }
    std::vector<float> kernel;
    int32_t core;
    bool bSupportAVX;
};

struct SymmColumnVec_32f {
    SymmColumnVec_32f(const std::vector<float> &_kernel)
    {
        kernel      = _kernel;
        core        = 1;
        bSupportAVX = true;
    }

    void operator()(float **_src, float *_dst, int32_t width) const
    {
        int32_t ksize2    = (kernel.size()) / 2;
        const float *ky   = &kernel[ksize2];
        int32_t i         = 0, k;
        const float **src = (const float **)_src;
        const float *S, *S2;
        float *dst = (float *)_dst;

        if (bSupportAVX) {
            for (; i <= width - 32; i += 32) {
                __m256 f = _mm256_broadcast_ss(ky);
                __m256 s0, s1, s2, s3;
                __m256 x0, x1;
                S  = src[0] + i;
                s0 = _mm256_load_ps(S);
                s1 = _mm256_load_ps(S + 8);
                s0 = _mm256_mul_ps(s0, f);
                s1 = _mm256_mul_ps(s1, f);
                s2 = _mm256_load_ps(S + 16);
                s3 = _mm256_load_ps(S + 24);
                s2 = _mm256_mul_ps(s2, f);
                s3 = _mm256_mul_ps(s3, f);

                for (k = 1; k <= ksize2; k++) {
                    S  = src[k] + i;
                    S2 = src[-k] + i;
                    f  = _mm256_broadcast_ss(ky + k);
                    x0 = _mm256_add_ps(_mm256_load_ps(S), _mm256_load_ps(S2));
                    x1 = _mm256_add_ps(_mm256_load_ps(S + 8), _mm256_load_ps(S2 + 8));
                    s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
                    s1 = _mm256_add_ps(s1, _mm256_mul_ps(x1, f));
                    x0 = _mm256_add_ps(_mm256_load_ps(S + 16), _mm256_load_ps(S2 + 16));
                    x1 = _mm256_add_ps(_mm256_load_ps(S + 24), _mm256_load_ps(S2 + 24));
                    s2 = _mm256_add_ps(s2, _mm256_mul_ps(x0, f));
                    s3 = _mm256_add_ps(s3, _mm256_mul_ps(x1, f));
                }

                _mm256_storeu_ps(dst + i, s0);
                _mm256_storeu_ps(dst + i + 8, s1);
                _mm256_storeu_ps(dst + i + 16, s2);
                _mm256_storeu_ps(dst + i + 24, s3);
            }

            for (; i <= width - 8; i += 8) {
                __m256 f = _mm256_broadcast_ss(ky);
                __m256 x0, s0 = _mm256_load_ps(src[0] + i);
                s0 = _mm256_mul_ps(s0, f);

                for (k = 1; k <= ksize2; k++) {
                    f  = _mm256_broadcast_ss(ky + k);
                    x0 = _mm256_add_ps(_mm256_load_ps(src[k] + i), _mm256_load_ps(src[-k] + i));
                    s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
                }

                _mm256_storeu_ps(dst + i, s0);
            }
        }

        for (; i < width; i++) {
            float f = ky[0];
            float s = (*(src[0] + i)) * f;
            for (k = 1; k <= ksize2; k++) {
                f        = ky[k];
                float s0 = *(src[k] + i);
                float s1 = *(src[-k] + i);
                s += (s0 + s1) * f;
            }
            dst[i] = s;
        }
    }
    std::vector<float> kernel;
    int32_t core;
    bool bSupportAVX;
};
#define TRANSPOSE_KERNEL_DEF       \
    __m256 ymm0, ymm1, ymm2, ymm3; \
    __m256 ymm4, ymm5, ymm6, ymm7; \
    __m256 ymm8, ymm11;            \
    __m256 ymm12, ymm13;

#define ROW1_Op(src)                          \
    ymm0 = _mm256_loadu_ps(src + i);          \
    ymm1 = _mm256_loadu_ps(src + i + cn);     \
    ymm2 = _mm256_loadu_ps(src + i + 2 * cn); \
    ymm0 = _mm256_add_ps(ymm0, ymm2);         \
    ymm0 = _mm256_mul_ps(ymm0, f0);           \
    ymm1 = _mm256_mul_ps(ymm1, f1);           \
    ymm0 = _mm256_add_ps(ymm0, ymm1);

#define ROW2_Op(src)                          \
    ymm3 = _mm256_loadu_ps(src + i);          \
    ymm4 = _mm256_loadu_ps(src + i + cn);     \
    ymm5 = _mm256_loadu_ps(src + i + 2 * cn); \
    ymm3 = _mm256_add_ps(ymm3, ymm5);         \
    ymm3 = _mm256_mul_ps(ymm3, f0);           \
    ymm4 = _mm256_mul_ps(ymm4, f1);           \
    ymm3 = _mm256_add_ps(ymm3, ymm4);

#define ROW3_Op(src)                          \
    ymm6 = _mm256_loadu_ps(src + i);          \
    ymm7 = _mm256_loadu_ps(src + i + cn);     \
    ymm8 = _mm256_loadu_ps(src + i + 2 * cn); \
    ymm6 = _mm256_add_ps(ymm6, ymm8);         \
    ymm6 = _mm256_mul_ps(ymm6, f0);           \
    ymm7 = _mm256_mul_ps(ymm7, f1);           \
    ymm6 = _mm256_add_ps(ymm6, ymm7);

#define COL1_Op                        \
    ymm13 = _mm256_add_ps(ymm0, ymm6); \
    ymm13 = _mm256_mul_ps(ymm13, f0);  \
    ymm11 = _mm256_mul_ps(ymm3, f1);   \
    ymm12 = _mm256_add_ps(ymm13, ymm11);

#define COL2_Op                        \
    ymm13 = _mm256_add_ps(ymm0, ymm3); \
    ymm13 = _mm256_mul_ps(ymm13, f0);  \
    ymm11 = _mm256_mul_ps(ymm6, f1);   \
    ymm12 = _mm256_add_ps(ymm13, ymm11);

#define COL3_Op                        \
    ymm13 = _mm256_add_ps(ymm3, ymm6); \
    ymm13 = _mm256_mul_ps(ymm13, f0);  \
    ymm11 = _mm256_mul_ps(ymm0, f1);   \
    ymm12 = _mm256_add_ps(ymm13, ymm11);

struct RowVec_32f_k3 {
    RowVec_32f_k3(const std::vector<float> &_kernel)
    {
        kernel      = _kernel;
        core        = 1;
        bSupportAVX = true;
    }

    void operator()(const float **_src, float *_dst, int32_t width, int32_t cn, int32_t dstep) const
    {
        __m256 f0, f1;
        if (bSupportAVX) {
            f0 = _mm256_broadcast_ss(&kernel[0]);
            f1 = _mm256_broadcast_ss(&kernel[1]);
        }

        float k0          = kernel[0];
        float k1          = kernel[1];
        const float *src0 = _src[-1];
        const float *src1 = _src[0];

        width     = width * cn;
        int32_t i = 0;
        if (bSupportAVX) {
            for (; i <= width - 8; i += 8) {
                TRANSPOSE_KERNEL_DEF
                ROW1_Op(src0);
                ROW2_Op(src1);
                for (int32_t j = 0; j < 9; j += 3) {
                    ROW3_Op(_src[j + 1]);
                    COL1_Op
                        _mm256_storeu_ps(_dst + j * dstep + i, ymm12);
                    ROW1_Op(_src[j + 2]);
                    COL2_Op
                        _mm256_storeu_ps(_dst + (j + 1) * dstep + i, ymm12);
                    ROW2_Op(_src[j + 3]);
                    COL3_Op
                        _mm256_storeu_ps(_dst + (j + 2) * dstep + i, ymm12);
                }
            }
        }

        for (; i < width; i++) {
            float row1 = (src0[i] + src0[i + 2 * cn]) * k0 + src0[i + cn] * k1;
            float row2 = (src1[i] + src1[i + 2 * cn]) * k0 + src1[i + cn] * k1;
            float row3;
            for (int32_t j = 0; j < 9; j += 3) {
                row3                      = (_src[j + 1][i] + _src[j + 1][i + 2 * cn]) * k0 + _src[j + 1][i + cn] * k1;
                float valcol              = (row1 + row3) * k0 + row2 * k1;
                _dst[(j)*dstep + i]       = valcol;
                row1                      = (_src[j + 2][i] + _src[j + 2][i + 2 * cn]) * k0 + _src[j + 2][i + cn] * k1;
                valcol                    = (row2 + row1) * k0 + row3 * k1;
                _dst[(j + 1) * dstep + i] = valcol;
                row2                      = (_src[j + 3][i] + _src[j + 3][i + 2 * cn]) * k0 + _src[j + 3][i + cn] * k1;
                valcol                    = (row2 + row3) * k0 + row1 * k1;
                _dst[(j + 2) * dstep + i] = valcol;
            }
        }
    }
    std::vector<float> kernel;
    int32_t core;
    bool bSupportAVX;
};

struct RowVec_32f_k3_raw {
    RowVec_32f_k3_raw(const std::vector<float> &_kernel)
    {
        kernel      = _kernel;
        core        = 1;
        bSupportAVX = true;
    }

    void operator()(const float **_src, float *_dst, int32_t width, int32_t cn, int32_t dstep) const
    {
        __m256 f0, f1;
        if (bSupportAVX) {
            f0 = _mm256_broadcast_ss(&kernel[0]);
            f1 = _mm256_broadcast_ss(&kernel[1]);
        }

        float k0          = kernel[0];
        float k1          = kernel[1];
        const float *src0 = _src[-1];
        const float *src1 = _src[0];
        const float *src2 = _src[1];

        width     = width * cn;
        int32_t i = 0;
        if (bSupportAVX) {
            for (; i <= width - 8; i += 8) {
                TRANSPOSE_KERNEL_DEF
                ROW1_Op(src0);
                ROW2_Op(src1);
                ROW3_Op(src2);
                COL1_Op
                    _mm256_storeu_ps(_dst + i, ymm12);
            }
        }

        for (; i < width; i++) {
            float row1 = (src0[i] + src0[i + 2 * cn]) * k0 + src0[i + cn] * k1;
            float row2 = (src1[i] + src1[i + 2 * cn]) * k0 + src1[i + cn] * k1;
            float row3 = (src2[i] + src2[i + 2 * cn]) * k0 + src2[i + cn] * k1;

            float valcol = (row1 + row3) * k0 + row2 * k1;
            _dst[i]      = valcol;
        }
    }
    std::vector<float> kernel;
    int32_t core;
    bool bSupportAVX;
};

#undef TRANSPOSE_KERNEL_DEF
#undef ROW1_Op
#undef ROW2_Op
#undef ROW3_Op
#undef COL1_op
#undef COL2_op
#undef COL3_op

//kernel optimazation
#define k5_TRANSPOSE_KERNEL_DEF      \
    __m256 ymm0, ymm1, ymm2, ymm3;   \
    __m256 ymm4, ymm5, ymm6, ymm7;   \
    __m256 ymm8, ymm9, ymm10, ymm11; \
    __m256 ymm12;
#define k5_ROW_Op(src)                        \
    ymm0 = _mm256_loadu_ps(src + i);          \
    ymm1 = _mm256_loadu_ps(src + i + cn);     \
    ymm2 = _mm256_loadu_ps(src + i + 2 * cn); \
    ymm3 = _mm256_loadu_ps(src + i + 3 * cn); \
    ymm4 = _mm256_loadu_ps(src + i + 4 * cn); \
    ymm0 = _mm256_add_ps(ymm0, ymm4);         \
    ymm0 = _mm256_mul_ps(ymm0, f0);           \
    ymm1 = _mm256_add_ps(ymm1, ymm3);         \
    ymm1 = _mm256_mul_ps(ymm1, f1);           \
    ymm2 = _mm256_mul_ps(ymm2, f2);

#define k5_COL1_Op                     \
    ymm10 = _mm256_add_ps(ymm5, ymm9); \
    ymm10 = _mm256_mul_ps(ymm10, f0);  \
    ymm11 = _mm256_add_ps(ymm6, ymm8); \
    ymm11 = _mm256_mul_ps(ymm11, f1);  \
    ymm12 = _mm256_mul_ps(ymm7, f2);   \
    ymm12 = _mm256_add_ps(_mm256_add_ps(ymm10, ymm11), ymm12);

#define k5_COL2_Op                     \
    ymm10 = _mm256_add_ps(ymm6, ymm5); \
    ymm10 = _mm256_mul_ps(ymm10, f0);  \
    ymm11 = _mm256_add_ps(ymm7, ymm9); \
    ymm11 = _mm256_mul_ps(ymm11, f1);  \
    ymm12 = _mm256_mul_ps(ymm8, f2);   \
    ymm12 = _mm256_add_ps(_mm256_add_ps(ymm10, ymm11), ymm12);

#define k5_COL3_Op                     \
    ymm10 = _mm256_add_ps(ymm7, ymm6); \
    ymm10 = _mm256_mul_ps(ymm10, f0);  \
    ymm11 = _mm256_add_ps(ymm8, ymm5); \
    ymm11 = _mm256_mul_ps(ymm11, f1);  \
    ymm12 = _mm256_mul_ps(ymm9, f2);   \
    ymm12 = _mm256_add_ps(_mm256_add_ps(ymm10, ymm11), ymm12);

#define k5_COL4_Op                     \
    ymm10 = _mm256_add_ps(ymm8, ymm7); \
    ymm10 = _mm256_mul_ps(ymm10, f0);  \
    ymm11 = _mm256_add_ps(ymm9, ymm6); \
    ymm11 = _mm256_mul_ps(ymm11, f1);  \
    ymm12 = _mm256_mul_ps(ymm5, f2);   \
    ymm12 = _mm256_add_ps(_mm256_add_ps(ymm10, ymm11), ymm12);

#define k5_COL5_Op                     \
    ymm10 = _mm256_add_ps(ymm8, ymm9); \
    ymm10 = _mm256_mul_ps(ymm10, f0);  \
    ymm11 = _mm256_add_ps(ymm5, ymm7); \
    ymm11 = _mm256_mul_ps(ymm11, f1);  \
    ymm12 = _mm256_mul_ps(ymm6, f2);   \
    ymm12 = _mm256_add_ps(_mm256_add_ps(ymm10, ymm11), ymm12);

struct RowVec_32f_k5_raw {
    RowVec_32f_k5_raw(const std::vector<float> &_kernel)
    {
        kernel      = _kernel;
        core        = 1;
        bSupportAVX = true;
    }

    void operator()(const float **_src, float *_dst, int32_t width, int32_t cn, int32_t dstep) const
    {
        __m256 f0, f1, f2;
        if (bSupportAVX) {
            f0 = _mm256_broadcast_ss(&kernel[0]);
            f1 = _mm256_broadcast_ss(&kernel[1]);
            f2 = _mm256_broadcast_ss(&kernel[2]);
        }

        float k0          = kernel[0];
        float k1          = kernel[1];
        float k2          = kernel[2];
        const float *src0 = _src[-2];
        const float *src1 = _src[-1];
        const float *src2 = _src[0];
        const float *src3 = _src[1];
        const float *src4 = _src[2];

        width     = width * cn;
        int32_t i = 0;

        if (bSupportAVX) {
            for (; i <= width - 8; i += 8) {
                k5_TRANSPOSE_KERNEL_DEF
                    k5_ROW_Op(src0)
                        ymm5 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                k5_ROW_Op(src1)
                    ymm6 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                k5_ROW_Op(src2)
                    ymm7 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                k5_ROW_Op(src3)
                    ymm8 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                k5_ROW_Op(src4)
                    ymm9 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);

                k5_COL1_Op
                    _mm256_storeu_ps(_dst + i, ymm12);
            }
        }
        for (; i < width; i++) {
            float row1 = (src0[i] + src0[i + 4 * cn]) * k0 + (src0[i + cn] + src0[i + 3 * cn]) * k1 + src0[i + 2 * cn] * k2;
            float row2 = (src1[i] + src1[i + 4 * cn]) * k0 + (src1[i + cn] + src1[i + 3 * cn]) * k1 + src1[i + 2 * cn] * k2;
            float row3 = (src2[i] + src2[i + 4 * cn]) * k0 + (src2[i + cn] + src2[i + 3 * cn]) * k1 + src2[i + 2 * cn] * k2;
            float row4 = (src3[i] + src3[i + 4 * cn]) * k0 + (src3[i + cn] + src3[i + 3 * cn]) * k1 + src3[i + 2 * cn] * k2;
            float row5 = (src4[i] + src4[i + 4 * cn]) * k0 + (src4[i + cn] + src4[i + 3 * cn]) * k1 + src4[i + 2 * cn] * k2;

            float valcol = (row1 + row5) * k0 + (row2 + row4) * k1 + row3 * k2;
            _dst[i]      = valcol;
        }
    }
    std::vector<float> kernel;
    int32_t core;
    bool bSupportAVX;
};

struct RowVec_32f_k5 {
    RowVec_32f_k5(const std::vector<float> &_kernel)
    {
        kernel      = _kernel;
        core        = 1;
        bSupportAVX = true;
    }

    void operator()(const float **_src, float *_dst, int32_t width, int32_t cn, int32_t dstep) const
    {
        __m256 f0, f1, f2;
        if (bSupportAVX) {
            f0 = _mm256_broadcast_ss(&kernel[0]);
            f1 = _mm256_broadcast_ss(&kernel[1]);
            f2 = _mm256_broadcast_ss(&kernel[2]);
        }

        float k0          = kernel[0];
        float k1          = kernel[1];
        float k2          = kernel[2];
        const float *src0 = _src[-2];
        const float *src1 = _src[-1];
        const float *src2 = _src[0];
        const float *src3 = _src[1];

        width     = width * cn;
        int32_t i = 0;
        if (bSupportAVX) {
            for (; i <= width - 8; i += 8) {
                k5_TRANSPOSE_KERNEL_DEF
                    k5_ROW_Op(src0)
                        ymm5 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                k5_ROW_Op(src1)
                    ymm6 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                k5_ROW_Op(src2)
                    ymm7 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                k5_ROW_Op(src3)
                    ymm8 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);

                for (int32_t j = 0; j < 10; j += 5) {
                    k5_ROW_Op(_src[j + 2]);
                    ymm9 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                    k5_COL1_Op;
                    _mm256_storeu_ps(_dst + j * dstep + i, ymm12);

                    k5_ROW_Op(_src[j + 3]);
                    ymm5 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                    k5_COL2_Op;
                    _mm256_storeu_ps(_dst + (j + 1) * dstep + i, ymm12);

                    k5_ROW_Op(_src[j + 4]);
                    ymm6 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                    k5_COL3_Op;
                    _mm256_storeu_ps(_dst + (j + 2) * dstep + i, ymm12);

                    k5_ROW_Op(_src[j + 5]);
                    ymm7 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                    k5_COL4_Op;
                    _mm256_storeu_ps(_dst + (j + 3) * dstep + i, ymm12);

                    k5_ROW_Op(_src[j + 6]);
                    ymm8 = _mm256_add_ps(_mm256_add_ps(ymm0, ymm1), ymm2);
                    k5_COL5_Op;
                    _mm256_storeu_ps(_dst + (j + 4) * dstep + i, ymm12);
                }
            }
        }

        for (; i < width; i++) {
            float row1 = (src0[i] + src0[i + 4 * cn]) * k0 + (src0[i + cn] + src0[i + 3 * cn]) * k1 + src0[i + 2 * cn] * k2;
            float row2 = (src1[i] + src1[i + 4 * cn]) * k0 + (src1[i + cn] + src1[i + 3 * cn]) * k1 + src1[i + 2 * cn] * k2;
            float row3 = (src2[i] + src2[i + 4 * cn]) * k0 + (src2[i + cn] + src2[i + 3 * cn]) * k1 + src2[i + 2 * cn] * k2;
            float row4 = (src3[i] + src3[i + 4 * cn]) * k0 + (src3[i + cn] + src3[i + 3 * cn]) * k1 + src3[i + 2 * cn] * k2;
            float row5;
            for (int32_t j = 0; j < 10; j += 5) {
                row5                      = (_src[j + 2][i] + _src[j + 2][i + 4 * cn]) * k0 + (_src[j + 2][i + cn] + _src[j + 2][i + 3 * cn]) * k1 + _src[j + 2][i + 2 * cn] * k2;
                float valcol              = (row1 + row5) * k0 + (row2 + row4) * k1 + row3 * k2;
                _dst[(j)*dstep + i]       = valcol;
                row1                      = (_src[j + 3][i] + _src[j + 3][i + 4 * cn]) * k0 + (_src[j + 3][i + cn] + _src[j + 3][i + 3 * cn]) * k1 + _src[j + 3][i + 2 * cn] * k2;
                valcol                    = (row2 + row1) * k0 + (row3 + row5) * k1 + row4 * k2;
                _dst[(j + 1) * dstep + i] = valcol;
                row2                      = (_src[j + 4][i] + _src[j + 4][i + 4 * cn]) * k0 + (_src[j + 4][i + cn] + _src[j + 4][i + 3 * cn]) * k1 + _src[j + 4][i + 2 * cn] * k2;
                valcol                    = (row3 + row2) * k0 + (row4 + row1) * k1 + row5 * k2;
                _dst[(j + 2) * dstep + i] = valcol;
                row3                      = (_src[j + 5][i] + _src[j + 5][i + 4 * cn]) * k0 + (_src[j + 5][i + cn] + _src[j + 5][i + 3 * cn]) * k1 + _src[j + 5][i + 2 * cn] * k2;
                valcol                    = (row4 + row3) * k0 + (row5 + row2) * k1 + row1 * k2;
                _dst[(j + 3) * dstep + i] = valcol;
                row4                      = (_src[j + 6][i] + _src[j + 6][i + 4 * cn]) * k0 + (_src[j + 6][i + cn] + _src[j + 6][i + 3 * cn]) * k1 + _src[j + 6][i + 2 * cn] * k2;
                valcol                    = (row5 + row4) * k0 + (row1 + row3) * k1 + row2 * k2;
                _dst[(j + 4) * dstep + i] = valcol;
            }
        }
    }
    std::vector<float> kernel;
    int32_t core;
    bool bSupportAVX;
};

static void x86GaussianBlur_fs3(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t kernel_len,
    float sigma,
    int32_t outWidthStride,
    float *outData,
    int32_t cn)
{
    std::vector<float> kernel;
    createGaussianKernels(kernel, kernel_len, sigma, sense32F);

    int32_t radius             = kernel_len / 2;
    int32_t innerHeight        = height - 2 * radius;
    int32_t innerWidth         = width - 2 * radius;
    const float **pReRowFilter = (const float **)malloc(height * sizeof(inData));
    for (int32_t i = 0; i < height; i++) {
        pReRowFilter[i] = inData + (i)*inWidthStride;
    }

    int32_t lrheight  = height + radius * 2;
    int32_t lrstep    = 3 * radius * cn * 2;
    int32_t udwidth   = width + radius * 2;
    int32_t udstep    = udwidth * cn;
    float *up_down    = (float *)malloc(udwidth * cn * radius * 3 * 2 * sizeof(float));
    float *left_right = (float *)malloc(lrheight * lrstep * sizeof(float));

    makeborder_onfly<float>(inData, height, width, inWidthStride, left_right, lrstep, up_down, udstep, cn, radius);

    //up and down border
    const float **updownBor = (const float **)malloc(radius * 3 * 2 * sizeof(inData));
    for (int32_t i = 0; i < radius * 3 * 2; i++) {
        updownBor[i] = up_down + (i)*udstep;
    }
    //left and right border
    const float **leftrightBor = (const float **)malloc(lrheight * 2 * sizeof(inData));
    for (int32_t i = 0; i < lrheight; i++) {
        leftrightBor[i]            = left_right + (i)*lrstep;
        leftrightBor[i + lrheight] = left_right + (i)*lrstep + 3 * radius * cn;
    }

    RowVec_32f_k3 rowVecOp = RowVec_32f_k3(kernel);
    int32_t i              = 0;
    for (; i <= innerHeight - 9; i += 9) {
        const float **src = pReRowFilter + i + radius;
        float *dst        = outData + (i + radius) * outWidthStride + radius * cn;
        rowVecOp.operator()((const float **)src, dst, innerWidth, cn, outWidthStride);
    }
    RowVec_32f_k3_raw rowVecOp_raw = RowVec_32f_k3_raw(kernel);
    for (; i < innerHeight; i++) {
        const float **src = pReRowFilter + i + radius;
        float *dst        = outData + (i + radius) * outWidthStride + radius * cn;
        rowVecOp_raw.operator()((const float **)src, dst, innerWidth, cn, outWidthStride);
    }

    for (i = 0; i < radius; i++) {
        const float **src = updownBor + i + radius;
        float *dst        = outData + i * outWidthStride;
        rowVecOp_raw.operator()((const float **)src, dst, width, cn, outWidthStride);

        const float **src1 = updownBor + i + 4 * radius;
        float *dst1        = outData + (i + height - radius) * outWidthStride;
        rowVecOp_raw.operator()((const float **)src1, dst1, width, cn, outWidthStride);
    }

    for (i = 0; i < height; i++) {
        const float **src = leftrightBor + i + radius;
        float *dst        = outData + i * outWidthStride;
        rowVecOp_raw.operator()((const float **)src, dst, radius, cn, outWidthStride);

        const float **src1 = leftrightBor + lrheight + i + radius;
        float *dst1        = outData + (i)*outWidthStride + width * cn - radius * cn;
        rowVecOp_raw.operator()((const float **)src1, dst1, radius, cn, outWidthStride);
    }

    free(up_down);
    free(left_right);
    free(updownBor);
    free(pReRowFilter);
    free(leftrightBor);
}

static void x86GaussianBlur_fs5(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t kernel_len,
    float sigma,
    int32_t outWidthStride,
    float *outData,
    int32_t cn)
{
    std::vector<float> kernel;
    createGaussianKernels(kernel, kernel_len, sigma, sense32F);

    int32_t radius             = kernel_len / 2;
    int32_t innerHeight        = height - 2 * radius;
    int32_t innerWidth         = width - 2 * radius;
    const float **pReRowFilter = (const float **)malloc(height * sizeof(inData));
    for (int32_t i = 0; i < height; i++) {
        pReRowFilter[i] = inData + (i)*inWidthStride;
    }

    int32_t lrheight  = height + radius * 2;
    int32_t lrstep    = 3 * radius * cn * 2;
    int32_t udwidth   = width + radius * 2;
    int32_t udstep    = udwidth * cn;
    float *up_down    = (float *)malloc(udwidth * cn * radius * 3 * 2 * sizeof(float));
    float *left_right = (float *)malloc(lrheight * lrstep * sizeof(float));

    makeborder_onfly<float>(inData, height, width, inWidthStride, left_right, lrstep, up_down, udstep, cn, radius);
    //up and down border
    const float **updownBor = (const float **)malloc(radius * 3 * 2 * sizeof(inData));
    for (int32_t i = 0; i < radius * 3 * 2; i++) {
        updownBor[i] = up_down + (i)*udstep;
    }
    //left and right border
    const float **leftrightBor = (const float **)malloc(lrheight * 2 * sizeof(inData));
    for (int32_t i = 0; i < lrheight; i++) {
        leftrightBor[i]            = left_right + (i)*lrstep;
        leftrightBor[i + lrheight] = left_right + (i)*lrstep + 3 * radius * cn;
    }

    RowVec_32f_k5 rowVecOp = RowVec_32f_k5(kernel);
    int32_t i              = 0;
    for (; i <= innerHeight - 10; i += 10) {
        const float **src = pReRowFilter + i + radius;
        float *dst        = outData + (i + radius) * outWidthStride + radius * cn;
        rowVecOp.operator()((const float **)src, dst, innerWidth, cn, outWidthStride);
    }

    RowVec_32f_k5_raw rowVecOp_raw = RowVec_32f_k5_raw(kernel);
    for (; i < innerHeight; i++) {
        const float **src = pReRowFilter + i + radius;
        float *dst        = outData + (i + radius) * outWidthStride + radius * cn;
        rowVecOp_raw.operator()((const float **)src, dst, innerWidth, cn, outWidthStride);
    }

    for (i = 0; i < radius; i++) {
        const float **src = updownBor + i + radius;
        float *dst        = outData + i * outWidthStride;
        rowVecOp_raw.operator()((const float **)src, dst, width, cn, outWidthStride);

        const float **src1 = updownBor + i + 4 * radius;
        float *dst1        = outData + (i + height - radius) * outWidthStride;
        rowVecOp_raw.operator()((const float **)src1, dst1, width, cn, outWidthStride);
    }

    for (i = 0; i < height; i++) {
        const float **src = leftrightBor + i + radius;
        float *dst        = outData + i * outWidthStride;
        rowVecOp_raw.operator()((const float **)src, dst, radius, cn, outWidthStride);

        const float **src1 = leftrightBor + lrheight + i + radius;
        float *dst1        = outData + (i)*outWidthStride + width * cn - radius * cn;
        rowVecOp_raw.operator()((const float **)src1, dst1, radius, cn, outWidthStride);
    }

    free(up_down);
    free(left_right);
    free(updownBor);
    free(pReRowFilter);
    free(leftrightBor);
}

template <int cn>
static void x86GaussianBlur_flarge(
    int32_t height,
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

    int32_t radius     = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth  = width + 2 * radius;

    int32_t bsrcWidthStep = (bsrcWidth + 7) / 8 * 8 * cn;
    float *bsrc           = (float *)_mm_malloc(bsrcHeight * bsrcWidthStep * sizeof(float), 64);
    CopyMakeBorder<float, cn>(height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);
    float *resultRowFilter = bsrc;
    float **pReRowFilter   = (float **)malloc(bsrcHeight * sizeof(resultRowFilter));
    for (int32_t i = 0; i < bsrcHeight; i++) {
        pReRowFilter[i] = resultRowFilter + (i)*bsrcWidthStep;
    }

    RowVec_32f rowVecOp = RowVec_32f(kernel);
    for (int32_t i = 0; i < bsrcHeight; i++) {
        float *src = bsrc + i * bsrcWidthStep;
        float *dst = resultRowFilter + i * bsrcWidthStep;
        rowVecOp.operator()(src, dst, width, cn);
    }

    SymmColumnVec_32f colVecOp = SymmColumnVec_32f(kernel);

    for (int32_t i = 0; i < height; i++) {
        float **src = pReRowFilter + i + radius;
        float *dst  = outData + i * outWidthStride;
        colVecOp.operator()(src, dst, width *cn);
    }

    _mm_free(bsrc);
    free(pReRowFilter);
    bsrc         = NULL;
    pReRowFilter = NULL;
}

template<int cn>
void x86GaussianBlur_f_vector_avx(
    int32_t height,
    int32_t width,
    const float *inData,
    int32_t kernel_len,
    float sigma,
    float *outData,
    ppl::cv::BorderType border_type)
{   
    std::vector<float> kernel;
    createGaussianKernels(kernel, kernel_len, sigma, sense32F);

    int32_t radius    = kernel_len / 2;
    int32_t radius_cn = radius * cn;
    int32_t pad_size  = 3 * radius_cn * 2;
    float*  pad_data  = (float *) malloc(pad_size * sizeof(inData));

    int32_t i, j, size = sizeof(float);
    int32_t length     = width ^ 1 ? width : height;
    int32_t length_cn  = length * cn;
    // make border
    for (i = 0; i < radius; i++) {
        j = borderInterpolate(i - radius, length, border_type);
        memcpy(pad_data + i * cn, inData + j * cn, cn * size);

        j = borderInterpolate(length + i, length, border_type);
        memcpy(pad_data + 5 * radius_cn + i * cn, inData + j * cn, cn * size);
    }
    memcpy(pad_data + radius_cn, inData, 2 * radius_cn * size);
    memcpy(pad_data + 3 * radius_cn, inData + length_cn - 2 * radius_cn, 2 * radius_cn * size);

    bool bSupportAVX = ppl::common::CpuSupports(ppl::common::ISA_X86_AVX);

    // do gaussianblur on pad data
    i = 0;
    if (bSupportAVX) {
        for (; i <= radius_cn - 8; i += 8) {
            float* src0 = pad_data + i;                              
            float* src1 = pad_data + 3 * radius_cn + i;
            __m256 f, res0 = _mm256_setzero_ps(), res1 = res0, x0, x1;
            for (j = 0; j < kernel_len; j++, src0 += cn, src1 += cn) {
                f    = _mm256_broadcast_ss(&kernel[j]);
                x0   = _mm256_loadu_ps(src0);
                x1   = _mm256_loadu_ps(src1);
                res0 = _mm256_add_ps(res0, _mm256_mul_ps(x0, f));
                res1 = _mm256_add_ps(res1, _mm256_mul_ps(x1, f));
            }

            _mm256_storeu_ps(outData + i, res0);
            _mm256_storeu_ps(outData + length_cn - radius_cn + i, res1);
        }
    }

    for (; i < radius_cn; i++) {
        float* src0 = pad_data + i;
        float* src1 = pad_data + 3 * radius_cn + i;
        float res0 = 0, res1 = 0;
        for (j = 0; j < kernel_len; j++, src0 += cn, src1 += cn) {
            res0 += src0[0] * kernel[j];
            res1 += src1[0] * kernel[j];
        }
        outData[i] = res0;
        outData[length_cn - radius_cn + i] = res1;
    }

    i = 0;
    const int32_t rest = length_cn - 2 * radius_cn;
    if (bSupportAVX) {
        for (; i <= rest - 8; i += 8) {
            const float* src = inData + i;
            __m256 f, res = _mm256_setzero_ps(), x0;
            for (j = 0; j < kernel_len; j++, src += cn) {
                f   = _mm256_broadcast_ss(&kernel[j]);
                x0  = _mm256_loadu_ps(src);
                res = _mm256_add_ps(res, _mm256_mul_ps(x0, f));
            }
            _mm256_storeu_ps(outData + radius_cn + i, res);
        }
    }

    // do gaussianblur on src data
    for (; i < rest; i++) {
        const float* src = inData + i;
        float res = 0;
        for (j = 0; j < kernel_len; j++, src += cn) {
            res += src[0] * kernel[j];
        }
        outData[radius_cn + i] = res;
    }

    free(pad_data);
}

template <int cn>
void x86GaussianBlur_f_avx(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t kernel_len,
    float sigma,
    int32_t outWidthStride,
    float *outData,
    ppl::cv::BorderType border_type)
{
    // only one element
    if (height == 1 && width == 1) {
        outData[0] = inData[0];
        return;
    }
    // vector
    if (height == 1 || width == 1) {
        x86GaussianBlur_f_vector_avx<cn>(height, width, inData, kernel_len, sigma, outData, border_type);
        return;
    }
    int32_t radius = kernel_len / 2;
    if (radius == 1 && border_type == ppl::cv::BORDER_REFLECT_101)
        x86GaussianBlur_fs3(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, cn);
    else if (radius == 2 && border_type == ppl::cv::BORDER_REFLECT_101)
        x86GaussianBlur_fs5(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, cn);
    else
        x86GaussianBlur_flarge<cn>(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
}

template void x86GaussianBlur_f_avx<1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t kernel_len,
    float sigma,
    int32_t outWidthStride,
    float *outData,
    ppl::cv::BorderType border_type);
template void x86GaussianBlur_f_avx<3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t kernel_len,
    float sigma,
    int32_t outWidthStride,
    float *outData,
    ppl::cv::BorderType border_type);
template void x86GaussianBlur_f_avx<4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t kernel_len,
    float sigma,
    int32_t outWidthStride,
    float *outData,
    ppl::cv::BorderType border_type);

}
}
} // namespace ppl::cv::x86
