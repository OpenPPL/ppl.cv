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

#include "ppl/cv/x86/remap.h"
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
#include <cassert>
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

template <typename T, int nc, ppl::cv::BorderType borderMode>
void remap_nearest(int inHeight, int inWidth, int inWidthStride, const T* src, int outHeight, int outWidth, int outWidthStride, T* dst, const float* map_x, const float* map_y, T delta)
{
    for (int i = 0; i < outHeight; i++) {
        for (int j = 0; j < outWidth; j++) {
            int idxDst = i * outWidthStride + j * nc;
            int idxMap = i * outWidth + j;
            float x    = map_x[idxMap];
            float y    = map_y[idxMap];
            int sy     = static_cast<int>(std::round(y));
            int sx     = static_cast<int>(std::round(x));
            if (borderMode == ppl::cv::BORDER_TYPE_CONSTANT) {
                int idxSrc = sy * inWidthStride + sx * nc;
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
                for (int i = 0; i < nc; i++) {
                    dst[idxDst + i] = src[idxSrc + i];
                }
            } else if (borderMode == ppl::cv::BORDER_TYPE_TRANSPARENT) {
                if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                    int idxSrc = sy * inWidthStride + sx * nc;
                    for (int i = 0; i < nc; i++) {
                        dst[idxDst + i] = src[idxSrc + i];
                    }
                } else {
                    continue;
                }
            }
        }
    }
}

template <typename T, int nc, ppl::cv::BorderType borderMode>
void remap_linear(int inHeight, int inWidth, int inWidthStride, const T* src, int outHeight, int outWidth, int outWidthStride, T* dst, const float* map_x, const float* map_y, T delta)
{
    for (int i = 0; i < outHeight; i++) {
        for (int j = 0; j < outWidth; j++) {
            int idxMap = i * outWidth + j;
            float x    = map_x[idxMap];
            float y    = map_y[idxMap];
            int sx0    = (int)x;
            int sy0    = (int)y;
            float u    = x - sx0;
            float v    = y - sy0;

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
}

template <typename T, ppl::cv::BorderType borderMode>
void remap_linear(int nc, int inHeight, int inWidth, int inWidthStride, const T* src, int outHeight, int outWidth, int outWidthStride, T* dst, const float* map_x, const float* map_y, T delta)
{
    for (int i = 0; i < outHeight; i++) {
        for (int j = 0; j < outWidth; j++) {
            int idxMap = i * outWidth + j;
            float x    = map_x[idxMap];
            float y    = map_y[idxMap];
            int sx0    = (int)x;
            int sy0    = (int)y;
            float u    = x - sx0;
            float v    = y - sy0;

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
}

template <typename T, int nc>
::ppl::common::RetCode RemapLinear(int inHeight, int inWidth, int inWidthStride, const T* inData, int outHeight, int outWidth, int outWidthStride, T* outData, const float* mapx, const float* mapy, BorderType border_type, T border_value)
{
    assert(border_type == ppl::cv::BORDER_TYPE_CONSTANT || border_type == ppl::cv::BORDER_TYPE_REPLICATE || border_type == ppl::cv::BORDER_TYPE_TRANSPARENT);
    if (border_type == ppl::cv::BORDER_TYPE_CONSTANT) {
        remap_linear<T, nc, BORDER_TYPE_CONSTANT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    } else if (border_type == ppl::cv::BORDER_TYPE_REPLICATE) {
        remap_linear<T, nc, BORDER_TYPE_REPLICATE>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    } else if (border_type == ppl::cv::BORDER_TYPE_TRANSPARENT) {
        remap_linear<T, nc, BORDER_TYPE_TRANSPARENT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T>
::ppl::common::RetCode RemapLinear(int nc, int inHeight, int inWidth, int inWidthStride, const T* inData, int outHeight, int outWidth, int outWidthStride, T* outData, const float* mapx, const float* mapy, BorderType border_type, T border_value)
{
    assert(border_type == ppl::cv::BORDER_TYPE_CONSTANT || border_type == ppl::cv::BORDER_TYPE_REPLICATE || border_type == ppl::cv::BORDER_TYPE_TRANSPARENT);
    if (border_type == ppl::cv::BORDER_TYPE_CONSTANT) {
        remap_linear<T, BORDER_TYPE_CONSTANT>(nc, inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    } else if (border_type == ppl::cv::BORDER_TYPE_REPLICATE) {
        remap_linear<T, BORDER_TYPE_REPLICATE>(nc, inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    } else if (border_type == ppl::cv::BORDER_TYPE_TRANSPARENT) {
        remap_linear<T, BORDER_TYPE_TRANSPARENT>(nc, inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int nc>
::ppl::common::RetCode RemapNearestPoint(int inHeight, int inWidth, int inWidthStride, const T* inData, int outHeight, int outWidth, int outWidthStride, T* outData, const float* mapx, const float* mapy, BorderType border_type, T border_value)
{
    assert(border_type == ppl::cv::BORDER_TYPE_CONSTANT || border_type == ppl::cv::BORDER_TYPE_REPLICATE || border_type == ppl::cv::BORDER_TYPE_TRANSPARENT);
    if (border_type == ppl::cv::BORDER_TYPE_CONSTANT) {
        remap_nearest<T, nc, BORDER_TYPE_CONSTANT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    } else if (border_type == ppl::cv::BORDER_TYPE_REPLICATE) {
        remap_nearest<T, nc, BORDER_TYPE_REPLICATE>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    } else if (border_type == ppl::cv::BORDER_TYPE_TRANSPARENT) {
        remap_nearest<T, nc, BORDER_TYPE_TRANSPARENT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, border_value);
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode RemapLinear<float, 1>(int inHeight, int inWidth, int inWidthStride, const float* inData, int outHeight, int outWidth, int outWidthStride, float* outData, const float* mapx, const float* mapy, BorderType border_type, float border_value);

template ::ppl::common::RetCode RemapLinear<float, 3>(int inHeight, int inWidth, int inWidthStride, const float* inData, int outHeight, int outWidth, int outWidthStride, float* outData, const float* mapx, const float* mapy, BorderType border_type, float border_value);

template ::ppl::common::RetCode RemapLinear<float, 4>(int inHeight, int inWidth, int inWidthStride, const float* inData, int outHeight, int outWidth, int outWidthStride, float* outData, const float* mapx, const float* mapy, BorderType border_type, float border_value);

template ::ppl::common::RetCode RemapLinear<float>(int nc, int inHeight, int inWidth, int inWidthStride, const float* inData, int outHeight, int outWidth, int outWidthStride, float* outData, const float* mapx, const float* mapy, BorderType border_type, float border_value);

template ::ppl::common::RetCode RemapLinear<uchar, 1>(int inHeight, int inWidth, int inWidthStride, const uchar* inData, int outHeight, int outWidth, int outWidthStride, uchar* outData, const float* mapx, const float* mapy, BorderType border_type, uchar border_value);

template ::ppl::common::RetCode RemapLinear<uchar, 3>(int inHeight, int inWidth, int inWidthStride, const uchar* inData, int outHeight, int outWidth, int outWidthStride, uchar* outData, const float* mapx, const float* mapy, BorderType border_type, uchar border_value);

template ::ppl::common::RetCode RemapLinear<uchar, 4>(int inHeight, int inWidth, int inWidthStride, const uchar* inData, int outHeight, int outWidth, int outWidthStride, uchar* outData, const float* mapx, const float* mapy, BorderType border_type, uchar border_value);

template ::ppl::common::RetCode RemapLinear<uchar>(int nc, int inHeight, int inWidth, int inWidthStride, const uchar* inData, int outHeight, int outWidth, int outWidthStride, uchar* outData, const float* mapx, const float* mapy, BorderType border_type, uchar border_value);

template ::ppl::common::RetCode RemapNearestPoint<float, 1>(int inHeight, int inWidth, int inWidthStride, const float* inData, int outHeight, int outWidth, int outWidthStride, float* outData, const float* mapx, const float* mapy, BorderType border_type, float border_value);

template ::ppl::common::RetCode RemapNearestPoint<float, 3>(int inHeight, int inWidth, int inWidthStride, const float* inData, int outHeight, int outWidth, int outWidthStride, float* outData, const float* mapx, const float* mapy, BorderType border_type, float border_value);

template ::ppl::common::RetCode RemapNearestPoint<float, 4>(int inHeight, int inWidth, int inWidthStride, const float* inData, int outHeight, int outWidth, int outWidthStride, float* outData, const float* mapx, const float* mapy, BorderType border_type, float border_value);

template ::ppl::common::RetCode RemapNearestPoint<uchar, 1>(int inHeight, int inWidth, int inWidthStride, const uchar* inData, int outHeight, int outWidth, int outWidthStride, uchar* outData, const float* mapx, const float* mapy, BorderType border_type, uchar border_value);

template ::ppl::common::RetCode RemapNearestPoint<uchar, 3>(int inHeight, int inWidth, int inWidthStride, const uchar* inData, int outHeight, int outWidth, int outWidthStride, uchar* outData, const float* mapx, const float* mapy, BorderType border_type, uchar border_value);

template ::ppl::common::RetCode RemapNearestPoint<uchar, 4>(int inHeight, int inWidth, int inWidthStride, const uchar* inData, int outHeight, int outWidth, int outWidthStride, uchar* outData, const float* mapx, const float* mapy, BorderType border_type, uchar border_value);

}
}
} // namespace ppl::cv::x86
