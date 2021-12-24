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

#include "ppl/cv/x86/warpperspective.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>

#include <vector>
#include <limits.h>
#include <immintrin.h>
#ifdef _WIN32
#include <algorithm>
#endif
namespace ppl {
namespace cv {
namespace x86 {

template <typename T>
static inline T clip(T x, T a, T b)
{
    return std::max(a, std::min(x, b));
}
template <typename Tsrc, typename Tdst>
static inline float cvtScale()
{
    return 1.f;
}
template <>
inline float cvtScale<unsigned char, float>()
{
    return 1. / 255;
}
template <>
inline float cvtScale<float, unsigned char>()
{
    return 255.f;
}
template <>
inline float cvtScale<unsigned short, float>()
{
    return 1. / 255;
}
template <>
inline float cvtScale<float, unsigned short>()
{
    return 255.f;
}

template <typename Tsrc, typename Tdst>
inline Tdst bilinear_sample(Tsrc t[][2], float u, float v);

template <>
inline float bilinear_sample<uchar, float>(uchar t[][2], float u, float v)
{
    float a0 = (1.0f - u) * (1.0f - v);
    float a1 = u * (1.0f - v);
    float a2 = (1.0f - u) * v;
    float a3 = u * v;
    return (t[0][0] * a0 + t[1][0] * a1 + t[0][1] * a2 + t[1][1] * a3) * cvtScale<uchar, float>();
}

template <>
inline float bilinear_sample<float, float>(float t[][2], float u, float v)
{
    float a0 = (1.0f - u) * (1.0f - v);
    float a1 = u * (1.0f - v);
    float a2 = (1.0f - u) * v;
    float a3 = u * v;
    return (t[0][0] * a0 + t[1][0] * a1 + t[0][1] * a2 + t[1][1] * a3) * cvtScale<float, float>();
}

template <>
inline uchar bilinear_sample<uchar, uchar>(uchar t[][2], float u, float v)
{
    float a0 = (1.0f - u) * (1.0f - v);
    float a1 = u * (1.0f - v);
    float a2 = (1.0f - u) * v;
    float a3 = u * v;
    return std::min(std::max(static_cast<int>((t[0][0] * a0 + t[1][0] * a1 + t[0][1] * a2 + t[1][1] * a3) * cvtScale<uchar, uchar>()), 0), 255);
}

template <typename T, int nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpperspective_nearest(int inHeight, int inWidth, int inWidthStride, int outHeight, int outWidth, int outWidthStride, T* dst, const T* src, const double M[][3], T delta = 0)
{
    for (int i = 0; i < outHeight; i++) {
        float baseW = M[2][1] * i + M[2][2];
        float baseX = M[0][1] * i + M[0][2];
        float baseY = M[1][1] * i + M[1][2];
        for (int j = 0; j < outWidth; j++) {
            float w = M[2][0] * j + baseW;
            float x = M[0][0] * j + baseX;
            float y = M[1][0] * j + baseY;
            int sy  = static_cast<int>(std::round(y / w));
            int sx  = static_cast<int>(std::round(x / w));
            if (borderMode == ppl::cv::BORDER_TYPE_CONSTANT) {
                int idxSrc = sy * inWidthStride + sx * nc;
                int idxDst = i * outWidthStride + j * nc;
                if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                    for (int i = 0; i < nc; i++)
                        dst[idxDst + i] = src[idxSrc + i];
                } else {
                    for (int i = 0; i < nc; i++) {
                        dst[idxDst + i] = delta;
                    }
                }
            } else if (borderMode == ppl::cv::BORDER_TYPE_REPLICATE) {
                sx         = clip(sx, 0, inWidth - 1);
                sy         = clip(sy, 0, inHeight - 1);
                int idxSrc = sy * inWidthStride + sx * nc;
                int idxDst = i * outWidthStride + j * nc;
                for (int i = 0; i < nc; i++) {
                    dst[idxDst + i] = src[idxSrc + i];
                }
            } else if (borderMode == ppl::cv::BORDER_TYPE_TRANSPARENT) {
                if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                    int idxSrc = sy * inWidthStride + sx * nc;
                    int idxDst = i * outWidthStride + j * nc;
                    for (int i = 0; i < nc; i++)
                        dst[idxDst + i] = src[idxSrc + i];
                } else {
                    continue;
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpperspective_linear(int inHeight, int inWidth, int inWidthStride, int outHeight, int outWidth, int outWidthStride, T* dst, const T* src, const double M[][3], T delta = 0)
{
    for (int i = 0; i < outHeight; i++) {
        float baseW = M[2][1] * i + M[2][2];
        float baseX = M[0][1] * i + M[0][2];
        float baseY = M[1][1] * i + M[1][2];
        for (int j = 0; j < outWidth; j++) {
            float w = (M[2][0] * j + baseW);
            float x = M[0][0] * j + baseX;
            float y = M[1][0] * j + baseY;
            y       = y / w;
            x       = x / w;
            int sx0 = (int)x;
            int sy0 = (int)y;
            float u = x - sx0;
            float v = y - sy0;

            float tab[4];
            float taby[2], tabx[2];
            float v0, v1, v2, v3;
            taby[0] = 1.0f - 1.0f * v;
            taby[1] = v;
            tabx[0] = 1.0f - u;
            tabx[1] = u;

            tab[0] = taby[0] * tabx[0];
            tab[1] = taby[0] * tabx[1];
            tab[2] = taby[1] * tabx[0];
            tab[3] = taby[1] * tabx[1];

            int idxDst = (i * outWidthStride + j * nc);

            if (borderMode == ppl::cv::BORDER_TYPE_CONSTANT) {
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                for (int k = 0; k < nc; k++) {
                    int position1 = (sy0 * inWidthStride + sx0 * nc);
                    int position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                    v0            = flag0 ? src[position1 + k] : delta;
                    v1            = flag1 ? src[position1 + nc + k] : delta;
                    v2            = flag2 ? src[position2 + k] : delta;
                    v3            = flag3 ? src[position2 + nc + k] : delta;
                    float sum     = 0;
                    sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                    dst[idxDst + k] = static_cast<T>(sum);
                }
            } else if (borderMode == ppl::cv::BORDER_TYPE_REPLICATE) {
                int sx1     = sx0 + 1;
                int sy1     = sy0 + 1;
                sx0         = clip(sx0, 0, inWidth - 1);
                sx1         = clip(sx1, 0, inWidth - 1);
                sy0         = clip(sy0, 0, inHeight - 1);
                sy1         = clip(sy1, 0, inHeight - 1);
                const T* t0 = src + sy0 * inWidthStride + sx0 * nc;
                const T* t1 = src + sy0 * inWidthStride + sx1 * nc;
                const T* t2 = src + sy1 * inWidthStride + sx0 * nc;
                const T* t3 = src + sy1 * inWidthStride + sx1 * nc;
                for (int k = 0; k < nc; ++k) {
                    float sum = 0;
                    sum += t0[k] * tab[0] + t1[k] * tab[1] + t2[k] * tab[2] + t3[k] * tab[3];
                    dst[idxDst + k] = static_cast<T>(sum);
                }
            } else if (borderMode == ppl::cv::BORDER_TYPE_TRANSPARENT) {
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                if (flag0 && flag1 && flag2 && flag3) {
                    for (int k = 0; k < nc; k++) {
                        int position1 = (sy0 * inWidthStride + sx0 * nc);
                        int position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                        v0            = src[position1 + k];
                        v1            = src[position1 + nc + k];
                        v2            = src[position2 + k];
                        v3            = src[position2 + nc + k];
                        float sum     = 0;
                        sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                        dst[idxDst + k] = static_cast<T>(sum);
                    }
                } else {
                    continue;
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int nc>
::ppl::common::RetCode WarpPerspectiveNearestPoint(int inHeight, int inWidth, int inWidthStride, const T* inData, int outHeight, int outWidth, int outWidthStride, T* outData, const double* affineMatrix, BorderType border_type, T border_value)
{
    double M[3][3];
    M[0][0] = affineMatrix[0];
    M[0][1] = affineMatrix[1];
    M[0][2] = affineMatrix[2];
    M[1][0] = affineMatrix[3];
    M[1][1] = affineMatrix[4];
    M[1][2] = affineMatrix[5];
    M[2][0] = affineMatrix[6];
    M[2][1] = affineMatrix[7];
    M[2][2] = affineMatrix[8];
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        if (border_type == ppl::cv::BORDER_TYPE_CONSTANT) {
            fma::warpperspective_nearest<T, nc, ppl::cv::BORDER_TYPE_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        } else if (border_type == ppl::cv::BORDER_TYPE_REPLICATE) {
            fma::warpperspective_nearest<T, nc, ppl::cv::BORDER_TYPE_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        } else if (border_type == ppl::cv::BORDER_TYPE_TRANSPARENT) {
            fma::warpperspective_nearest<T, nc, ppl::cv::BORDER_TYPE_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        }
    } else {
        if (border_type == ppl::cv::BORDER_TYPE_CONSTANT) {
            warpperspective_nearest<T, nc, ppl::cv::BORDER_TYPE_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        } else if (border_type == ppl::cv::BORDER_TYPE_REPLICATE) {
            warpperspective_nearest<T, nc, ppl::cv::BORDER_TYPE_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        } else if (border_type == ppl::cv::BORDER_TYPE_TRANSPARENT) {
            warpperspective_nearest<T, nc, ppl::cv::BORDER_TYPE_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int nc>
::ppl::common::RetCode WarpPerspectiveLinear(int inHeight, int inWidth, int inWidthStride, const T* inData, int outHeight, int outWidth, int outWidthStride, T* outData, const double* affineMatrix, BorderType border_type, T border_value)
{
    double M[3][3];
    M[0][0] = affineMatrix[0];
    M[0][1] = affineMatrix[1];
    M[0][2] = affineMatrix[2];
    M[1][0] = affineMatrix[3];
    M[1][1] = affineMatrix[4];
    M[1][2] = affineMatrix[5];
    M[2][0] = affineMatrix[6];
    M[2][1] = affineMatrix[7];
    M[2][2] = affineMatrix[8];
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        if (border_type == ppl::cv::BORDER_TYPE_CONSTANT) {
            fma::warpperspective_linear<T, nc, ppl::cv::BORDER_TYPE_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        } else if (border_type == ppl::cv::BORDER_TYPE_REPLICATE) {
            fma::warpperspective_linear<T, nc, ppl::cv::BORDER_TYPE_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        } else if (border_type == ppl::cv::BORDER_TYPE_TRANSPARENT) {
            fma::warpperspective_linear<T, nc, ppl::cv::BORDER_TYPE_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        }
    } else {
        if (border_type == ppl::cv::BORDER_TYPE_CONSTANT) {
            warpperspective_linear<T, nc, ppl::cv::BORDER_TYPE_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        } else if (border_type == ppl::cv::BORDER_TYPE_REPLICATE) {
            warpperspective_linear<T, nc, ppl::cv::BORDER_TYPE_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        } else if (border_type == ppl::cv::BORDER_TYPE_TRANSPARENT) {
            warpperspective_linear<T, nc, ppl::cv::BORDER_TYPE_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode WarpPerspectiveNearestPoint<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpPerspectiveNearestPoint<float, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpPerspectiveNearestPoint<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpPerspectiveNearestPoint<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpPerspectiveNearestPoint<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpPerspectiveNearestPoint<uint8_t, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpPerspectiveNearestPoint<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpPerspectiveNearestPoint<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpPerspectiveLinear<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpPerspectiveLinear<float, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpPerspectiveLinear<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpPerspectiveLinear<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpPerspectiveLinear<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpPerspectiveLinear<uint8_t, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpPerspectiveLinear<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpPerspectiveLinear<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

}
}
} // namespace ppl::cv::x86
