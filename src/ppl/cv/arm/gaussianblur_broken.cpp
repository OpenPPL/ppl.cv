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

namespace ppl::cv::arm {

int32_t borderInterpolate(int32_t p, int32_t len)
{
    p = p < 0 ? -p : 2 * len - p - 2;
    return p;
}

template <typename T>
static void makeborder_onfly(const T *src,
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
    int32_t elemSize = sizeof(T);
    int32_t left = radius;
    int32_t right = radius;
    float *left_right_cs = left_right;
    float *up_down_cs = up_down;

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
        memcpy(up_down + (i + 3 * radius) * udstep + radius * cn,
               src + (srcHeight - 2 * radius + i) * srcstep,
               srcWidth * elemSize);
    }

    // make tab
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

    // up border border
    T *left_right_shift = left_right_cs + radius * cn;
    up_down = up_down_cs;
    for (i = 0; i < radius; i++) {
        T *left_right_index = left_right_shift + i * lrstep;
        T *up_downtemp = up_down + (i)*udstep + radius * cn;
        memcpy(left_right_index, up_downtemp, 2 * radius * cn * elemSize);
        memcpy(
            left_right_index + 2 * radius * cn, up_downtemp + srcWidth - 2 * radius * cn, 2 * radius * cn * elemSize);
        for (j = 0; j < left; j++) {
            left_right_index[j - left] = up_downtemp[tab[j]];
        }
        for (j = 0; j < right; j++)
            left_right_index[j + 4 * radius * cn] = up_downtemp[tab[j + left]];
    }
    // down border border
    left_right_shift = left_right_cs + (srcHeight + radius) * lrstep + radius * cn;
    for (i = 0; i < radius; i++) {
        T *up_downtemp = up_down + (i + 5 * radius) * udstep + radius * cn;
        T *left_right_index = left_right_shift + i * lrstep;
        memcpy(left_right_index, up_downtemp, 2 * radius * cn * elemSize);
        memcpy(
            left_right_index + 2 * radius * cn, up_downtemp + srcWidth - 2 * radius * cn, 2 * radius * cn * elemSize);
        for (j = 0; j < left; j++) {
            left_right_index[j - left] = up_downtemp[tab[j]];
        }
        for (j = 0; j < right; j++)
            left_right_index[j + 4 * radius * cn] = up_downtemp[tab[j + left]];
    }
}

static uint8_t sat_cast(int32_t data)
{
    int32_t val;
    val = data > 255 ? 255 : data;
    val = data < 0 ? 0 : val;
    return val;
}

std::vector<float> getGaussianKernel(float sigma, int32_t n)
{
    const int32_t SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] = {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}};
    bool fix = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0;
    const float *fixed_kernel = fix ? small_gaussian_tab[n >> 1] : 0;
    std::vector<float> kernel(n);
    float sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
    float scale2X = -0.5 / (sigmaX * sigmaX);
    float sum = 0;

    int32_t i;
    for (i = 0; i < n; i++) {
        float x = i - (n - 1) * 0.5;
        float t = fixed_kernel ? (float)fixed_kernel[i] : std::exp(scale2X * x * x);
        kernel[i] = (float)t;
        sum += kernel[i];
    }

    sum = 1. / sum;
    for (i = 0; i < n; i++) {
        kernel[i] = float(kernel[i] * sum);
    }
    return kernel;
}

enum imageDepth { sense8U = 1, sense32F = 2 };

static int32_t senseRound(float value)
{
    return std::roundf(value);
}
static void createGaussianKernels(std::vector<float> &k, int32_t ksize, float sigma, imageDepth depth)
{
    // automatic detection of kernel size from sigma
    if (ksize <= 0 && sigma > 0) ksize = senseRound(sigma * (depth == sense8U ? 3 : 4) * 2 + 1) | 1;

    // assert(ksize > 0 && ksize % 2 == 1);

    sigma = std::max(sigma, 0.f);

    k = getGaussianKernel(sigma, ksize);
}

struct RowVec_32f {
    RowVec_32f(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
        core = 1;
    }

    void operator()(const float *_src, float *_dst, int32_t width, int32_t cn) const
    {
        int32_t _ksize = kernel.size();
        const float *src0 = (const float *)_src;
        float *dst = (float *)_dst;

        int32_t i = 0, k;
        width *= cn;

        for (; i < width; i++) {
            const float *src = src0 + i;
            float s = 0;
            for (k = 0; k < _ksize; k++, src += cn) {
                s += src[0] * kernel[k];
            }
            dst[i] = s;
        }
    }
    std::vector<float> kernel;
    int32_t core;
};

struct SymmColumnVec_32f {
    SymmColumnVec_32f(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
        core = 1;
    }

    void operator()(float **_src, float *_dst, int32_t width) const
    {
        int32_t ksize2 = (kernel.size()) / 2;
        const float *ky = &kernel[ksize2];
        int32_t i = 0, k;
        const float **src = (const float **)_src;
        float *dst = (float *)_dst;

        for (; i < width; i++) {
            float f = ky[0];
            float s = (*(src[0] + i)) * f;
            for (k = 1; k <= ksize2; k++) {
                f = ky[k];
                float s0 = *(src[k] + i);
                float s1 = *(src[-k] + i);
                s += (s0 + s1) * f;
            }
            dst[i] = s;
        }
    }
    std::vector<float> kernel;
    int32_t core;
};

struct RowVec_32f_k3 {
    RowVec_32f_k3(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
        core = 1;
    }

    void operator()(const float **_src, float *_dst, int32_t width, int32_t cn, int32_t dstep) const
    {
        float k0 = kernel[0];
        float k1 = kernel[1];
        const float *src0 = _src[-1];
        const float *src1 = _src[0];

        width = width * cn;
        int32_t i = 0;

        for (; i < width; i++) {
            float row1 = (src0[i] + src0[i + 2 * cn]) * k0 + src0[i + cn] * k1;
            float row2 = (src1[i] + src1[i + 2 * cn]) * k0 + src1[i + cn] * k1;
            float row3;
            for (int32_t j = 0; j < 9; j += 3) {
                row3 = (_src[j + 1][i] + _src[j + 1][i + 2 * cn]) * k0 + _src[j + 1][i + cn] * k1;
                float valcol = (row1 + row3) * k0 + row2 * k1;
                _dst[(j)*dstep + i] = valcol;
                row1 = (_src[j + 2][i] + _src[j + 2][i + 2 * cn]) * k0 + _src[j + 2][i + cn] * k1;
                valcol = (row2 + row1) * k0 + row3 * k1;
                _dst[(j + 1) * dstep + i] = valcol;
                row2 = (_src[j + 3][i] + _src[j + 3][i + 2 * cn]) * k0 + _src[j + 3][i + cn] * k1;
                valcol = (row2 + row3) * k0 + row1 * k1;
                _dst[(j + 2) * dstep + i] = valcol;
            }
        }
    }
    std::vector<float> kernel;
    int32_t core;
};

struct RowVec_32f_k3_raw {
    RowVec_32f_k3_raw(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
        core = 1;
    }

    void operator()(const float **_src, float *_dst, int32_t width, int32_t cn, int32_t dstep) const
    {
        float k0 = kernel[0];
        float k1 = kernel[1];
        const float *src0 = _src[-1];
        const float *src1 = _src[0];
        const float *src2 = _src[1];

        width = width * cn;
        int32_t i = 0;

        for (; i < width; i++) {
            float row1 = (src0[i] + src0[i + 2 * cn]) * k0 + src0[i + cn] * k1;
            float row2 = (src1[i] + src1[i + 2 * cn]) * k0 + src1[i + cn] * k1;
            float row3 = (src2[i] + src2[i + 2 * cn]) * k0 + src2[i + cn] * k1;

            float valcol = (row1 + row3) * k0 + row2 * k1;
            _dst[i] = valcol;
        }
    }
    std::vector<float> kernel;
    int32_t core;
};

struct RowVec_32f_k5_raw {
    RowVec_32f_k5_raw(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
        core = 1;
    }

    void operator()(const float **_src, float *_dst, int32_t width, int32_t cn, int32_t dstep) const
    {
        float k0 = kernel[0];
        float k1 = kernel[1];
        float k2 = kernel[2];
        const float *src0 = _src[-2];
        const float *src1 = _src[-1];
        const float *src2 = _src[0];
        const float *src3 = _src[1];
        const float *src4 = _src[2];

        width = width * cn;
        int32_t i = 0;

        for (; i < width; i++) {
            float row1 =
                (src0[i] + src0[i + 4 * cn]) * k0 + (src0[i + cn] + src0[i + 3 * cn]) * k1 + src0[i + 2 * cn] * k2;
            float row2 =
                (src1[i] + src1[i + 4 * cn]) * k0 + (src1[i + cn] + src1[i + 3 * cn]) * k1 + src1[i + 2 * cn] * k2;
            float row3 =
                (src2[i] + src2[i + 4 * cn]) * k0 + (src2[i + cn] + src2[i + 3 * cn]) * k1 + src2[i + 2 * cn] * k2;
            float row4 =
                (src3[i] + src3[i + 4 * cn]) * k0 + (src3[i + cn] + src3[i + 3 * cn]) * k1 + src3[i + 2 * cn] * k2;
            float row5 =
                (src4[i] + src4[i + 4 * cn]) * k0 + (src4[i + cn] + src4[i + 3 * cn]) * k1 + src4[i + 2 * cn] * k2;

            float valcol = (row1 + row5) * k0 + (row2 + row4) * k1 + row3 * k2;
            _dst[i] = valcol;
        }
    }
    std::vector<float> kernel;
    int32_t core;
};

struct RowVec_32f_k5 {
    RowVec_32f_k5(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
        core = 1;
    }

    void operator()(const float **_src, float *_dst, int32_t width, int32_t cn, int32_t dstep) const
    {
        float k0 = kernel[0];
        float k1 = kernel[1];
        float k2 = kernel[2];
        const float *src0 = _src[-2];
        const float *src1 = _src[-1];
        const float *src2 = _src[0];
        const float *src3 = _src[1];

        width = width * cn;
        int32_t i = 0;

        for (; i < width; i++) {
            float row1 =
                (src0[i] + src0[i + 4 * cn]) * k0 + (src0[i + cn] + src0[i + 3 * cn]) * k1 + src0[i + 2 * cn] * k2;
            float row2 =
                (src1[i] + src1[i + 4 * cn]) * k0 + (src1[i + cn] + src1[i + 3 * cn]) * k1 + src1[i + 2 * cn] * k2;
            float row3 =
                (src2[i] + src2[i + 4 * cn]) * k0 + (src2[i + cn] + src2[i + 3 * cn]) * k1 + src2[i + 2 * cn] * k2;
            float row4 =
                (src3[i] + src3[i + 4 * cn]) * k0 + (src3[i + cn] + src3[i + 3 * cn]) * k1 + src3[i + 2 * cn] * k2;
            float row5;
            for (int32_t j = 0; j < 10; j += 5) {
                row5 = (_src[j + 2][i] + _src[j + 2][i + 4 * cn]) * k0 +
                       (_src[j + 2][i + cn] + _src[j + 2][i + 3 * cn]) * k1 + _src[j + 2][i + 2 * cn] * k2;
                float valcol = (row1 + row5) * k0 + (row2 + row4) * k1 + row3 * k2;
                _dst[(j)*dstep + i] = valcol;
                row1 = (_src[j + 3][i] + _src[j + 3][i + 4 * cn]) * k0 +
                       (_src[j + 3][i + cn] + _src[j + 3][i + 3 * cn]) * k1 + _src[j + 3][i + 2 * cn] * k2;
                valcol = (row2 + row1) * k0 + (row3 + row5) * k1 + row4 * k2;
                _dst[(j + 1) * dstep + i] = valcol;
                row2 = (_src[j + 4][i] + _src[j + 4][i + 4 * cn]) * k0 +
                       (_src[j + 4][i + cn] + _src[j + 4][i + 3 * cn]) * k1 + _src[j + 4][i + 2 * cn] * k2;
                valcol = (row3 + row2) * k0 + (row4 + row1) * k1 + row5 * k2;
                _dst[(j + 2) * dstep + i] = valcol;
                row3 = (_src[j + 5][i] + _src[j + 5][i + 4 * cn]) * k0 +
                       (_src[j + 5][i + cn] + _src[j + 5][i + 3 * cn]) * k1 + _src[j + 5][i + 2 * cn] * k2;
                valcol = (row4 + row3) * k0 + (row5 + row2) * k1 + row1 * k2;
                _dst[(j + 3) * dstep + i] = valcol;
                row4 = (_src[j + 6][i] + _src[j + 6][i + 4 * cn]) * k0 +
                       (_src[j + 6][i + cn] + _src[j + 6][i + 3 * cn]) * k1 + _src[j + 6][i + 2 * cn] * k2;
                valcol = (row5 + row4) * k0 + (row1 + row3) * k1 + row2 * k2;
                _dst[(j + 4) * dstep + i] = valcol;
            }
        }
    }
    std::vector<float> kernel;
    int32_t core;
};

struct RowVec_8u32s {
    RowVec_8u32s(const std::vector<float> &_kernel)
    {
        kernel = _kernel;
    }

    void operator()(const uint8_t *_src, float *_dst, int32_t width, int32_t cn) const
    {
        int32_t i = 0, k, _ksize = kernel.size();
        float *dst = (float *)_dst;
        const float *_kx = &kernel[0];
        width *= cn;
        for (; i < width; i++) {
            const uint8_t *src = _src + i;
            float s = 0;
            for (k = 0; k < _ksize; k++, src += cn) {
                float f = _kx[k];
                uint8_t x = src[0];
                // s += f * x;
                s = std::fma(f, x, s);
            }
            dst[i] = senseRound(s);
        }
    }
    std::vector<float> kernel;
};
struct SymmColumnVec_32s8u {
    SymmColumnVec_32s8u(const std::vector<float> &_kernel, float _delta)
    {
        //_kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        kernel = _kernel;
        delta = (float)(_delta);
    }

    void operator()(float **_src, uint8_t *dst, int32_t width) const
    {
        int32_t ksize2 = kernel.size() / 2;
        const float *ky = &kernel[ksize2];
        int32_t i = 0, k;
        const float **src = (const float **)_src;
        for (; i < width; i++) {
            float f = ky[0];
            float s = (*(src[0] + i)) * f;
            for (k = 1; k <= ksize2; k++) {
                f = ky[k];
                float s0 = *(src[k] + i);
                float s1 = *(src[-k] + i);
                // s += (s0 + s1) * f;
                s = std::fmaf((s0 + s1), f, s);
            }
            dst[i] = sat_cast(senseRound(s));
            //	dst[i] = uint8_t(s);
        }
    }

    float delta;
    std::vector<float> kernel;
};

void GaussianBlur_fs3(int32_t height,
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

    int32_t radius = kernel_len / 2;
    int32_t innerHeight = height - 2 * radius;
    int32_t innerWidth = width - 2 * radius;
    const float **pReRowFilter = (const float **)malloc(height * sizeof(inData));
    for (int32_t i = 0; i < height; i++) {
        pReRowFilter[i] = inData + (i)*inWidthStride;
    }

    int32_t lrheight = height + radius * 2;
    int32_t lrstep = 3 * radius * cn * 2;
    int32_t udwidth = width + radius * 2;
    int32_t udstep = udwidth * cn;
    float *up_down = (float *)malloc(udwidth * cn * radius * 3 * 2 * sizeof(float));
    float *left_right = (float *)malloc(lrheight * lrstep * sizeof(float));

    makeborder_onfly<float>(inData, height, width, inWidthStride, left_right, lrstep, up_down, udstep, cn, radius);

    // up and down border
    const float **updownBor = (const float **)malloc(radius * 3 * 2 * sizeof(inData));
    for (int32_t i = 0; i < radius * 3 * 2; i++) {
        updownBor[i] = up_down + (i)*udstep;
    }
    // left and right border
    const float **leftrightBor = (const float **)malloc(lrheight * 2 * sizeof(inData));
    for (int32_t i = 0; i < lrheight; i++) {
        leftrightBor[i] = left_right + (i)*lrstep;
        leftrightBor[i + lrheight] = left_right + (i)*lrstep + 3 * radius * cn;
    }

    RowVec_32f_k3 rowVecOp = RowVec_32f_k3(kernel);
    int32_t i = 0;
    for (; i <= innerHeight - 9; i += 9) {
        const float **src = pReRowFilter + i + radius;
        float *dst = outData + (i + radius) * outWidthStride + radius * cn;
        rowVecOp.operator()((const float **)src, dst, innerWidth, cn, outWidthStride);
    }
    RowVec_32f_k3_raw rowVecOp_raw = RowVec_32f_k3_raw(kernel);
    for (; i < innerHeight; i++) {
        const float **src = pReRowFilter + i + radius;
        float *dst = outData + (i + radius) * outWidthStride + radius * cn;
        rowVecOp_raw.operator()((const float **)src, dst, innerWidth, cn, outWidthStride);
    }

    for (i = 0; i < radius; i++) {
        const float **src = updownBor + i + radius;
        float *dst = outData + i * outWidthStride;
        rowVecOp_raw.operator()((const float **)src, dst, width, cn, outWidthStride);

        const float **src1 = updownBor + i + 4 * radius;
        float *dst1 = outData + (i + height - radius) * outWidthStride;
        rowVecOp_raw.operator()((const float **)src1, dst1, width, cn, outWidthStride);
    }

    for (i = 0; i < height; i++) {
        const float **src = leftrightBor + i + radius;
        float *dst = outData + i * outWidthStride;
        rowVecOp_raw.operator()((const float **)src, dst, radius, cn, outWidthStride);

        const float **src1 = leftrightBor + lrheight + i + radius;
        float *dst1 = outData + (i)*outWidthStride + width * cn - radius * cn;
        rowVecOp_raw.operator()((const float **)src1, dst1, radius, cn, outWidthStride);
    }

    free(up_down);
    free(left_right);
    free(updownBor);
    free(pReRowFilter);
    free(leftrightBor);
}
void GaussianBlur_fs5(int32_t height,
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

    int32_t radius = kernel_len / 2;
    int32_t innerHeight = height - 2 * radius;
    int32_t innerWidth = width - 2 * radius;
    const float **pReRowFilter = (const float **)malloc(height * sizeof(inData));
    for (int32_t i = 0; i < height; i++) {
        pReRowFilter[i] = inData + (i)*inWidthStride;
    }

    int32_t lrheight = height + radius * 2;
    int32_t lrstep = 3 * radius * cn * 2;
    int32_t udwidth = width + radius * 2;
    int32_t udstep = udwidth * cn;
    float *up_down = (float *)malloc(udwidth * cn * radius * 3 * 2 * sizeof(float));
    float *left_right = (float *)malloc(lrheight * lrstep * sizeof(float));

    makeborder_onfly<float>(inData, height, width, inWidthStride, left_right, lrstep, up_down, udstep, cn, radius);
    // up and down border
    const float **updownBor = (const float **)malloc(radius * 3 * 2 * sizeof(inData));
    for (int32_t i = 0; i < radius * 3 * 2; i++) {
        updownBor[i] = up_down + (i)*udstep;
    }
    // left and right border
    const float **leftrightBor = (const float **)malloc(lrheight * 2 * sizeof(inData));
    for (int32_t i = 0; i < lrheight; i++) {
        leftrightBor[i] = left_right + (i)*lrstep;
        leftrightBor[i + lrheight] = left_right + (i)*lrstep + 3 * radius * cn;
    }

    RowVec_32f_k5 rowVecOp = RowVec_32f_k5(kernel);
    int32_t i = 0;
    for (; i <= innerHeight - 10; i += 10) {
        const float **src = pReRowFilter + i + radius;
        float *dst = outData + (i + radius) * outWidthStride + radius * cn;
        rowVecOp.operator()((const float **)src, dst, innerWidth, cn, outWidthStride);
    }

    RowVec_32f_k5_raw rowVecOp_raw = RowVec_32f_k5_raw(kernel);
    for (; i < innerHeight; i++) {
        const float **src = pReRowFilter + i + radius;
        float *dst = outData + (i + radius) * outWidthStride + radius * cn;
        rowVecOp_raw.operator()((const float **)src, dst, innerWidth, cn, outWidthStride);
    }

    for (i = 0; i < radius; i++) {
        const float **src = updownBor + i + radius;
        float *dst = outData + i * outWidthStride;
        rowVecOp_raw.operator()((const float **)src, dst, width, cn, outWidthStride);

        const float **src1 = updownBor + i + 4 * radius;
        float *dst1 = outData + (i + height - radius) * outWidthStride;
        rowVecOp_raw.operator()((const float **)src1, dst1, width, cn, outWidthStride);
    }

    for (i = 0; i < height; i++) {
        const float **src = leftrightBor + i + radius;
        float *dst = outData + i * outWidthStride;
        rowVecOp_raw.operator()((const float **)src, dst, radius, cn, outWidthStride);

        const float **src1 = leftrightBor + lrheight + i + radius;
        float *dst1 = outData + (i)*outWidthStride + width * cn - radius * cn;
        rowVecOp_raw.operator()((const float **)src1, dst1, radius, cn, outWidthStride);
    }

    free(up_down);
    free(left_right);
    free(updownBor);
    free(pReRowFilter);
    free(leftrightBor);
}

template <int cn>
void GaussianBlur_flarge(int32_t height,
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

    int32_t radius = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth = width + 2 * radius;

    int32_t bsrcWidthStep = (bsrcWidth + 7) / 8 * 8 * cn;
    float *bsrc = (float *)malloc(bsrcHeight * bsrcWidthStep * sizeof(float));
    // makeborder<float>(inData,height,width, inWidthStride,
    //         bsrc, bsrcHeight, bsrcWidth, bsrcWidthStep, cn);
    CopyMakeBorder<float, cn>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);
    float *resultRowFilter = bsrc;
    float **pReRowFilter = (float **)malloc(bsrcHeight * sizeof(resultRowFilter));
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
        float *dst = outData + i * outWidthStride;
        colVecOp.operator()(src, dst, width * cn);
    }

    free(bsrc);
    free(pReRowFilter);
    bsrc = NULL;
    pReRowFilter = NULL;
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

    int32_t radius = kernel_len / 2;
    if (radius == 1 && border_type == ppl::cv::BORDER_REFLECT_101)
        GaussianBlur_fs3(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, cn);
    else if (radius == 2 && border_type == ppl::cv::BORDER_REFLECT_101)
        GaussianBlur_fs5(height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, cn);
    else
        GaussianBlur_flarge<cn>(
            height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
}

template <typename T, int32_t cn>
void GaussianBlur_b(int32_t height,
                    int32_t width,
                    int32_t inWidthStride,
                    const uint8_t *inData,
                    int32_t kernel_len,
                    float sigma,
                    int32_t outWidthStride,
                    uint8_t *outData,
                    BorderType border_type)
{
    std::vector<float> kernel;
    createGaussianKernels(kernel, kernel_len, sigma, sense8U);

    int32_t radius = kernel_len / 2;
    int32_t bsrcHeight = height + 2 * radius;
    int32_t bsrcWidth = width + 2 * radius;

    int32_t bsrcWidthStep = (bsrcWidth)*cn;
    uint8_t *bsrc = (uint8_t *)malloc(bsrcHeight * bsrcWidth * cn * sizeof(uint8_t));
    CopyMakeBorder<uint8_t, cn>(
        height, width, inWidthStride, inData, bsrcHeight, bsrcWidth, bsrcWidthStep, bsrc, border_type);

    int32_t alignWidth = (width + 3) / 4 * 4;
    int32_t rfstep = alignWidth * cn;
    float *resultRowFilter = (float *)malloc(bsrcHeight * alignWidth * cn * sizeof(float));

    RowVec_8u32s rowVecOp = RowVec_8u32s(kernel);

    for (int32_t i = 0; i < bsrcHeight; i++) {
        uint8_t *src = bsrc + i * bsrcWidthStep;
        float *dst = resultRowFilter + i * rfstep;
        rowVecOp.operator()(src, dst, width, cn);
    }
    SymmColumnVec_32s8u colVecOp = SymmColumnVec_32s8u(kernel, 0);
    float **pReRowFilter = (float **)malloc(bsrcHeight * sizeof(resultRowFilter));
    for (int32_t i = 0; i < bsrcHeight; i++) {
        pReRowFilter[i] = resultRowFilter + (i)*rfstep;
    }
    for (int32_t i = 0; i < height; i++) {
        float **src = pReRowFilter + i + radius;
        uint8_t *dst = outData + i * outWidthStride;
        colVecOp.operator()(src, dst, width * cn);
    }
    free(bsrc);
    free(resultRowFilter);
    free(pReRowFilter);
    bsrc = NULL;
    resultRowFilter = NULL;
    pReRowFilter = NULL;
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
    GaussianBlur_b<uint8_t, 3>(
        height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
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
    GaussianBlur_b<uint8_t, 1>(
        height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
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
    GaussianBlur_b<uint8_t, 4>(
        height, width, inWidthStride, inData, kernel_len, sigma, outWidthStride, outData, border_type);
    return ppl::common::RC_SUCCESS;
}

} // namespace ppl::cv::arm
