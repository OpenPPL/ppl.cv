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

#include "ppl/cv/x86/warpaffine.h"
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
inline T clip(T value, T min_value, T max_value)
{
    return std::min(std::max(value, min_value), max_value);
}

static inline int32_t saturate_cast(double value)
{
    int32_t round2zero = (int32_t)value;
    if (value >= 0) {
        return (value - round2zero != 0.5) ? (int32_t)(value + 0.5) : round2zero % 2 == 0 ? round2zero
                                                                                          : round2zero + 1;
    } else {
        return (round2zero - value != 0.5) ? (int32_t)(value - 0.5) : round2zero % 2 == 0 ? round2zero
                                                                                          : round2zero - 1;
    }
}
static inline int32_t floor(float value)
{
    int32_t i = (int32_t)value;
    return i - (i > value);
}

static inline short saturate_cast(int32_t v)
{
    return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX
                                                                               : SHRT_MIN);
}

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_nearest(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* dst,
    const T* src,
    const double* M,
    T delta)
{
    for (int32_t i = 0; i < outHeight; i++) {
        int32_t base_x = saturate_cast((M[1] * i + M[2]) * 1024) + 512;
        int32_t base_y = saturate_cast((M[4] * i + M[5]) * 1024) + 512;
        for (int32_t j = 0; j < outWidth; j++) {
            int32_t sx = (base_x + saturate_cast(M[0] * j * 1024)) >> 10;
            int32_t sy = (base_y + saturate_cast(M[3] * j * 1024)) >> 10;
            if (borderMode == ppl::cv::BORDER_CONSTANT) {
                int32_t idxSrc = sy * inWidthStride + sx * nc;
                int32_t idxDst = i * outWidthStride + j * nc;
                if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                    for (int32_t i = 0; i < nc; i++)
                        dst[idxDst + i] = src[idxSrc + i];
                } else {
                    for (int32_t i = 0; i < nc; i++) {
                        dst[idxDst + i] = delta;
                    }
                }
            } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                sx             = clip(sx, 0, inWidth - 1);
                sy             = clip(sy, 0, inHeight - 1);
                int32_t idxSrc = sy * inWidthStride + sx * nc;
                int32_t idxDst = i * outWidthStride + j * nc;
                for (int32_t i = 0; i < nc; i++) {
                    dst[idxDst + i] = src[idxSrc + i];
                }
            } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                    int32_t idxSrc = sy * inWidthStride + sx * nc;
                    int32_t idxDst = i * outWidthStride + j * nc;
                    for (int32_t i = 0; i < nc; i++)
                        dst[idxDst + i] = src[idxSrc + i];
                } else {
                    continue;
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_linear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* dst,
    const T* src,
    const double* M,
    T delta)
{
    for (int32_t i = 0; i < outHeight; i++) {
        float base_x = M[1] * i + M[2];
        float base_y = M[4] * i + M[5];
        for (int32_t j = 0; j < outWidth; j++) {
            float x     = base_x + M[0] * j;
            float y     = base_y + M[3] * j;
            int32_t sx0 = (int32_t)x;
            int32_t sy0 = (int32_t)y;

            float u = x - sx0;
            float v = y - sy0;

            float tab[4];
            float taby[2], tabx[2];
            float v0, v1, v2, v3;
            taby[0] = 1.0f - v;
            taby[1] = v;
            tabx[0] = 1.0f - u;
            tabx[1] = u;

            tab[0] = taby[0] * tabx[0];
            tab[1] = taby[0] * tabx[1];
            tab[2] = taby[1] * tabx[0];
            tab[3] = taby[1] * tabx[1];

            int32_t idxDst = (i * outWidthStride + j * nc);

            if (borderMode == ppl::cv::BORDER_CONSTANT) {
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                for (int32_t k = 0; k < nc; k++) {
                    int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                    int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                    v0                = flag0 ? src[position1 + k] : delta;
                    v1                = flag1 ? src[position1 + nc + k] : delta;
                    v2                = flag2 ? src[position2 + k] : delta;
                    v3                = flag3 ? src[position2 + nc + k] : delta;
                    float sum         = 0;
                    sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                    dst[idxDst + k] = static_cast<T>(sum);
                }
            } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                int32_t sx1 = sx0 + 1;
                int32_t sy1 = sy0 + 1;
                sx0         = clip(sx0, 0, inWidth - 1);
                sx1         = clip(sx1, 0, inWidth - 1);
                sy0         = clip(sy0, 0, inHeight - 1);
                sy1         = clip(sy1, 0, inHeight - 1);
                const T* t0 = src + sy0 * inWidthStride + sx0 * nc;
                const T* t1 = src + sy0 * inWidthStride + sx1 * nc;
                const T* t2 = src + sy1 * inWidthStride + sx0 * nc;
                const T* t3 = src + sy1 * inWidthStride + sx1 * nc;
                for (int32_t k = 0; k < nc; ++k) {
                    float sum = 0;
                    sum += t0[k] * tab[0] + t1[k] * tab[1] + t2[k] * tab[2] + t3[k] * tab[3];
                    dst[idxDst + k] = static_cast<T>(sum);
                }
            } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                if (flag0 && flag1 && flag2 && flag3) {
                    for (int32_t k = 0; k < nc; k++) {
                        int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                        int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                        v0                = src[position1 + k];
                        v1                = src[position1 + nc + k];
                        v2                = src[position2 + k];
                        v3                = src[position2 + nc + k];
                        float sum         = 0;
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

template <typename T, int32_t nc>
::ppl::common::RetCode WarpAffineNearestPoint(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData,
    const double* affineMatrix,
    BorderType border_type,
    T border_value)
{
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        if (border_type == ppl::cv::BORDER_CONSTANT) {
            return fma::warpaffine_nearest<T, nc, ppl::cv::BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        } else if (border_type == ppl::cv::BORDER_REPLICATE) {
            return fma::warpaffine_nearest<T, nc, ppl::cv::BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
            return fma::warpaffine_nearest<T, nc, ppl::cv::BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        }
    } else {
        if (border_type == ppl::cv::BORDER_CONSTANT) {
            return warpaffine_nearest<T, nc, ppl::cv::BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        } else if (border_type == ppl::cv::BORDER_REPLICATE) {
            return warpaffine_nearest<T, nc, ppl::cv::BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
            return warpaffine_nearest<T, nc, ppl::cv::BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode WarpAffineNearestPoint<float, 1>(
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

template ::ppl::common::RetCode WarpAffineNearestPoint<float, 2>(
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

template ::ppl::common::RetCode WarpAffineNearestPoint<float, 3>(
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

template ::ppl::common::RetCode WarpAffineNearestPoint<float, 4>(
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

template ::ppl::common::RetCode WarpAffineNearestPoint<uint8_t, 1>(
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

template ::ppl::common::RetCode WarpAffineNearestPoint<uint8_t, 2>(
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

template ::ppl::common::RetCode WarpAffineNearestPoint<uint8_t, 3>(
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

template ::ppl::common::RetCode WarpAffineNearestPoint<uint8_t, 4>(
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

template <typename T, int32_t nc>
::ppl::common::RetCode WarpAffineLinear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData,
    const double* affineMatrix,
    BorderType border_type,
    T border_value)
{
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        if (border_type == ppl::cv::BORDER_CONSTANT) {
            return fma::warpaffine_linear<T, nc, ppl::cv::BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        } else if (border_type == ppl::cv::BORDER_REPLICATE) {
            return fma::warpaffine_linear<T, nc, ppl::cv::BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
            return fma::warpaffine_linear<T, nc, ppl::cv::BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        }
    } else {
        if (border_type == ppl::cv::BORDER_CONSTANT) {
            return warpaffine_linear<T, nc, ppl::cv::BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        } else if (border_type == ppl::cv::BORDER_REPLICATE) {
            return warpaffine_linear<T, nc, ppl::cv::BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
            return warpaffine_linear<T, nc, ppl::cv::BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode WarpAffineLinear<float, 1>(
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

template ::ppl::common::RetCode WarpAffineLinear<float, 2>(
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

template ::ppl::common::RetCode WarpAffineLinear<float, 3>(
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

template ::ppl::common::RetCode WarpAffineLinear<float, 4>(
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

template ::ppl::common::RetCode WarpAffineLinear<uint8_t, 1>(
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

template ::ppl::common::RetCode WarpAffineLinear<uint8_t, 2>(
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

template ::ppl::common::RetCode WarpAffineLinear<uint8_t, 3>(
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

template ::ppl::common::RetCode WarpAffineLinear<uint8_t, 4>(
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
