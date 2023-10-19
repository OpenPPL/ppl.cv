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

#include "ppl/cv/arm/warpperspective.h"
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"
#include <string.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits.h>
#include <arm_neon.h>

#define BLOCK_SIZE       (32)
#define INTER_TABLE_BITS (5)
#define INTER_TABLE_SIZE (32)

namespace ppl {
namespace cv {
namespace arm {

static float BilinearTab_f[INTER_TABLE_SIZE * INTER_TABLE_SIZE][2][2];
static short BilinearTab_i[INTER_TABLE_SIZE * INTER_TABLE_SIZE][2][2];
const int32_t INTER_REMAP_COEF_BITS = 15;
const int32_t INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

template <typename T>
static inline T clip(T x, T a, T b)
{
    return std::max(a, std::min(x, b));
}

inline static uint8_t saturate_cast_u8(int32_t x)
{
    return static_cast<uint8_t>(clip(x, 0, 255));
}

template <typename T>
inline static short saturate_cast_short(T x)
{
    return x > SHRT_MIN ? x < SHRT_MAX ? x : SHRT_MAX : SHRT_MIN;
}

static inline void interpolateLinear(float x, float* coeffs)
{
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

static void initInterTab1D(float* tab, int32_t tabsz)
{
    float scale = 1.f / tabsz;
    for (int32_t i = 0; i < tabsz; i++, tab += 2)
        interpolateLinear(i * scale, tab);
}

static void initInterTab2D()
{
    static bool is_inited = false;
    if (is_inited) { return; }
    float* tab = BilinearTab_f[0][0];
    short* itab = BilinearTab_i[0][0];
    int32_t ksize = 2;

    float* _tab = new float[8 * INTER_TABLE_SIZE];
    int32_t i, j, k1, k2;
    initInterTab1D(_tab, INTER_TABLE_SIZE);
    for (i = 0; i < INTER_TABLE_SIZE; i++) {
        for (j = 0; j < INTER_TABLE_SIZE; j++, tab += ksize * ksize, itab += ksize * ksize) {
            int32_t isum = 0;

            for (k1 = 0; k1 < ksize; k1++) {
                float vy = _tab[i * ksize + k1];
                for (k2 = 0; k2 < ksize; k2++) {
                    float v = vy * _tab[j * ksize + k2];
                    tab[k1 * ksize + k2] = v;
                    isum += itab[k1 * ksize + k2] = saturate_cast_short(v * INTER_REMAP_COEF_SCALE);
                }
            }

            if (isum != INTER_REMAP_COEF_SCALE) {
                int32_t diff = isum - INTER_REMAP_COEF_SCALE;
                int32_t ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
                for (k1 = ksize2; k1 < ksize2 + 2; k1++)
                    for (k2 = ksize2; k2 < ksize2 + 2; k2++) {
                        if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2])
                            mk1 = k1, mk2 = k2;
                        else if (itab[k1 * ksize + k2] > itab[Mk1 * ksize + Mk2])
                            Mk1 = k1, Mk2 = k2;
                    }
                if (diff < 0)
                    itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
                else
                    itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
            }
        }
    }
    is_inited = true;
}

template <typename T, typename WT>
T remapLinearCast(WT val)
{
    return (T)val;
}

template <>
uint8_t remapLinearCast<uint8_t, int32_t>(int32_t val)
{
    constexpr uint32_t SHIFT = INTER_REMAP_COEF_BITS;
    constexpr uint32_t DELTA = 1 << (INTER_REMAP_COEF_BITS - 1);
    return saturate_cast_u8((val + DELTA) >> SHIFT);
}

template <typename T, typename AT, typename WT, int32_t nc, ppl::cv::BorderType borderMode>
void remapBilinear(const T* src,
                   int32_t inHeight,
                   int32_t inWidth,
                   int32_t inWidthStride,
                   T* dst,
                   int32_t outHeight,
                   int32_t outWidth,
                   int32_t outWidthStride,
                   int16_t* coord_map,
                   int16_t* alpha_map,
                   AT* interTable,
                   T delta = 0)
{
    T deltaArr[4] = {delta, delta, delta, delta};
    initInterTab2D();

    // cval: borderVal: delta
    // safe area to proceed without considering border
    uint32_t uwidth_m = std::max(inWidth - 1, 0);
    uint32_t uheight_m = std::max(inHeight - 1, 0);
    // if( _src.type() == CV_8UC3 && SIMD ) width1 = std::max(ssize.width-2, 0);

    for (int i = 0; i < outHeight; i++) {
        T* dstLine = dst + outWidthStride * i;
        const int16_t* coordMapLine = coord_map + BLOCK_SIZE * 2 * i;
        const int16_t* alphaMapLine = alpha_map + BLOCK_SIZE * i;

        int prevJ = 0;
        bool prevInSafeArea = false;

        for (int j = 0; j <= outWidth; j++) {
            // last time: must process previous items
            // non-last time: process previous items if from safe to unsafe area or vice versa
            // unsigned comparation: filter out less than 0 and more than val in the same time
            bool curInSafeArea = (j == outWidth) ? (!prevInSafeArea)
                                                 : (((unsigned)coordMapLine[j * 2] < uwidth_m) &&
                                                    ((unsigned)coordMapLine[j * 2 + 1] < uheight_m));
            if (curInSafeArea == prevInSafeArea) { continue; }

            int jSegmentEnd = j;
            j = prevJ;

            prevJ = jSegmentEnd;
            prevInSafeArea = curInSafeArea;

            if (!curInSafeArea) {
                for (; j < jSegmentEnd; j++) {
                    int sx = coordMapLine[j * 2];
                    int sy = coordMapLine[j * 2 + 1];
                    const AT* w = interTable + 4 * alphaMapLine[j];
                    const T* s = src + sy * inWidthStride + sx * nc;

                    for (int k = 0; k < nc; k++) {
                        T s00 = s[k];
                        T s01 = s[k + nc];
                        T s10 = s[k + inWidthStride];
                        T s11 = s[k + inWidthStride + nc];
                        WT data = s00 * w[0] + s01 * w[1] + s10 * w[2] + s11 * w[3];
                        dstLine[j * nc + k] = remapLinearCast<T, WT>(data);
                    }
                }
            } else {
                for (; j < jSegmentEnd; j++) {
                    int sx = coordMapLine[j * 2];
                    int sy = coordMapLine[j * 2 + 1];

                    // fastpath 1: use delta if transparent and fully outside border
                    bool allInOutArea = (sx >= inWidth || (sx + 1) < 0 || sy >= inHeight || (sx + 1) < 0);
                    if (borderMode == ppl::cv::BORDER_CONSTANT && allInOutArea) {
                        for (int k = 0; k < nc; k++) {
                            dstLine[j * nc + k] = deltaArr[k];
                        }
                        continue;
                    }

                    // fastpath 2: skip if transparent and partially outside
                    bool partiallyInOutArea =
                        ((unsigned)sx >= (unsigned)(inWidth - 1) || (unsigned)sy >= (unsigned)(inHeight - 1));
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT && partiallyInOutArea) { continue; }

                    int sx0, sx1, sy0, sy1;
                    const T *v0, *v1, *v2, *v3;
                    const AT* w = interTable + 4 * alphaMapLine[j];
                    sx0 = sx;
                    sx1 = sx + 1;
                    sy0 = sy;
                    sy1 = sy + 1;
                    if (borderMode == ppl::cv::BORDER_REPLICATE) {
                        sx0 = clip(sx0, 0, inWidth - 1);
                        sx1 = clip(sx1, 0, inWidth - 1);
                        sy0 = clip(sy0, 0, inHeight - 1);
                        sy1 = clip(sy1, 0, inHeight - 1);
                        v0 = src + sy0 * inWidthStride + sx0 * nc;
                        v1 = src + sy0 * inWidthStride + sx1 * nc;
                        v2 = src + sy1 * inWidthStride + sx0 * nc;
                        v3 = src + sy1 * inWidthStride + sx1 * nc;
                    } else if (borderMode == ppl::cv::BORDER_CONSTANT) {
                        bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                        bool flag1 = (sx1 >= 0 && sx1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                        bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy1 >= 0 && sy1 < inHeight);
                        bool flag3 = (sx1 >= 0 && sx1 < inWidth && sy1 >= 0 && sy1 < inHeight);
                        v0 = flag0 ? src + sy0 * inWidthStride + sx0 * nc : deltaArr;
                        v1 = flag1 ? src + sy0 * inWidthStride + sx1 * nc : deltaArr;
                        v2 = flag2 ? src + sy1 * inWidthStride + sx0 * nc : deltaArr;
                        v3 = flag3 ? src + sy1 * inWidthStride + sx1 * nc : deltaArr;
                    } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        v0 = src + sy0 * inWidthStride + sx0 * nc;
                        v1 = src + sy0 * inWidthStride + sx1 * nc;
                        v2 = src + sy1 * inWidthStride + sx0 * nc;
                        v3 = src + sy1 * inWidthStride + sx1 * nc;
                    }

                    for (int k = 0; k < nc; k++) {
                        WT data = v0[k] * w[0] + v1[k] * w[1] + v2[k] * w[2] + v3[k] * w[3];
                        dstLine[j * nc + k] = remapLinearCast<T, WT>(data);
                    }
                }
            }
        }
    }
}

void WarpPerspective_CoordCompute_Nearest_Line(const double* M,
                                               int16_t* coord_map,
                                               int bw,
                                               double X0,
                                               double Y0,
                                               double W0)
{
    float64x2_t v_baseXd = {0.0, 1.0};
    const float64x2_t v_2d = vdupq_n_f64(2.0);
    const float64x2_t v_W0d_c = vdupq_n_f64(W0);
    const float64x2_t v_X0d_c = vdupq_n_f64(X0);
    const float64x2_t v_Y0d_c = vdupq_n_f64(Y0);
    const float64x2_t v_M0d = vdupq_n_f64(M[0]);
    const float64x2_t v_M3d = vdupq_n_f64(M[3]);
    const float64x2_t v_M6d = vdupq_n_f64(M[6]);
    const float32x4_t v_shrtmaxf = vdupq_n_f32((float)SHRT_MAX);
    const float32x4_t v_shrtminf = vdupq_n_f32((float)SHRT_MIN);

    int jj = 0;
    for (; jj <= bw - 16; jj += 16) {
        // 0 1 2 3
        int32x4_t vCoordX0123i, vCoordY0123i;
        {
            float64x2_t v_W0d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X0d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y0d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            float64x2_t v_W1d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X1d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y1d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            // for NEON, fdiv produces 0 when divisor is 0
            v_X0d = vdivq_f64(v_X0d, v_W0d);
            v_Y0d = vdivq_f64(v_Y0d, v_W0d);
            v_X1d = vdivq_f64(v_X1d, v_W1d);
            v_Y1d = vdivq_f64(v_Y1d, v_W1d);

            // assume that the number won't overflow float
            // this almost always holds and permits less clip
            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_shrtminf), v_shrtmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_shrtminf), v_shrtmaxf);

            vCoordX0123i = vcvtnq_s32_f32(v_X01f);
            vCoordY0123i = vcvtnq_s32_f32(v_Y01f);
        }

        // 4 5 6 7
        int32x4_t vCoordX4567i, vCoordY4567i;
        {
            float64x2_t v_W0d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X0d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y0d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            float64x2_t v_W1d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X1d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y1d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            v_X0d = vdivq_f64(v_X0d, v_W0d);
            v_Y0d = vdivq_f64(v_Y0d, v_W0d);
            v_X1d = vdivq_f64(v_X1d, v_W1d);
            v_Y1d = vdivq_f64(v_Y1d, v_W1d);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_shrtminf), v_shrtmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_shrtminf), v_shrtmaxf);

            vCoordX4567i = vcvtnq_s32_f32(v_X01f);
            vCoordY4567i = vcvtnq_s32_f32(v_Y01f);
        }

        // 8 9 10 11
        int32x4_t vCoordX89abi, vCoordY89abi;
        {
            float64x2_t v_W0d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X0d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y0d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            float64x2_t v_W1d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X1d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y1d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            v_X0d = vdivq_f64(v_X0d, v_W0d);
            v_Y0d = vdivq_f64(v_Y0d, v_W0d);
            v_X1d = vdivq_f64(v_X1d, v_W1d);
            v_Y1d = vdivq_f64(v_Y1d, v_W1d);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_shrtminf), v_shrtmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_shrtminf), v_shrtmaxf);

            vCoordX89abi = vcvtnq_s32_f32(v_X01f);
            vCoordY89abi = vcvtnq_s32_f32(v_Y01f);
        }

        // 12 13 14 15
        int32x4_t vCoordXcdefi, vCoordYcdefi;
        {
            float64x2_t v_W0d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X0d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y0d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            float64x2_t v_W1d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X1d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y1d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            v_X0d = vdivq_f64(v_X0d, v_W0d);
            v_Y0d = vdivq_f64(v_Y0d, v_W0d);
            v_X1d = vdivq_f64(v_X1d, v_W1d);
            v_Y1d = vdivq_f64(v_Y1d, v_W1d);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_shrtminf), v_shrtmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_shrtminf), v_shrtmaxf);

            vCoordXcdefi = vcvtnq_s32_f32(v_X01f);
            vCoordYcdefi = vcvtnq_s32_f32(v_Y01f);
        }

        int16x8x2_t vCoord0, vCoord1;
        int16x4_t vCoord0XHalf = vmovn_s32(vCoordX0123i);
        vCoord0.val[0] = vmovn_high_s32(vCoord0XHalf, vCoordX4567i);
        int16x4_t vCoord0YHalf = vmovn_s32(vCoordY0123i);
        vCoord0.val[1] = vmovn_high_s32(vCoord0YHalf, vCoordY4567i);
        vst2q_s16(coord_map + jj * 2, vCoord0);

        int16x4_t vCoord1XHalf = vmovn_s32(vCoordX89abi);
        vCoord1.val[0] = vmovn_high_s32(vCoord1XHalf, vCoordXcdefi);
        int16x4_t vCoord1YHalf = vmovn_s32(vCoordY89abi);
        vCoord1.val[1] = vmovn_high_s32(vCoord1YHalf, vCoordYcdefi);
        vst2q_s16(coord_map + jj * 2 + 16, vCoord1);
    }

    for (; jj < bw; jj++) {
        double W = W0 + M[6] * jj;
        W = W ? 1. / W : 0;
        double X = std::max((double)SHRT_MIN, std::min((double)SHRT_MAX, (X0 + M[0] * jj) * W));
        double Y = std::max((double)SHRT_MIN, std::min((double)SHRT_MAX, (Y0 + M[3] * jj) * W));
        int16_t Xh = lrint(X);
        int16_t Yh = lrint(Y);

        coord_map[jj * 2] = Xh;
        coord_map[jj * 2 + 1] = Yh;
    }
}

void WarpPerspective_CoordCompute_Linear_Line(const double* M,
                                              int16_t* coord_map,
                                              int16_t* alpha_map,
                                              int bw,
                                              double X0,
                                              double Y0,
                                              double W0)
{
    float64x2_t v_baseXd = {0.0, 1.0};
    const float64x2_t v_2d = vdupq_n_f64(2.0);
    const float64x2_t v_W0d_c = vdupq_n_f64(W0);
    const float64x2_t v_X0d_c = vdupq_n_f64(X0);
    const float64x2_t v_Y0d_c = vdupq_n_f64(Y0);
    const float64x2_t v_M0d = vdupq_n_f64(M[0]);
    const float64x2_t v_M3d = vdupq_n_f64(M[3]);
    const float64x2_t v_M6d = vdupq_n_f64(M[6]);
    const float64x2_t v_interTblSzd = vdupq_n_f64((double)INTER_TABLE_SIZE);
    const int16x8_t v_interTblMaski = vdupq_n_s16(INTER_TABLE_SIZE - 1);
    const float32x4_t v_intmaxf = vdupq_n_f32((float)INT_MAX);
    const float32x4_t v_intminf = vdupq_n_f32((float)INT_MIN);

    int jj = 0;
    for (; jj <= bw - 16; jj += 16) {
        // 0 1 2 3
        int32x4_t vCoordX0123i, vCoordY0123i;
        {
            float64x2_t v_W0d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X0d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y0d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            float64x2_t v_W1d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X1d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y1d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            // for NEON, fdiv produces 0 when divisor is 0
            v_X0d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W0d), v_X0d);
            v_Y0d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W0d), v_Y0d);
            v_X1d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W1d), v_X1d);
            v_Y1d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W1d), v_Y1d);

            // assume that the number won't overflow float
            // this almost always holds and permits less clip
            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_intminf), v_intmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_intminf), v_intmaxf);

            vCoordX0123i = vcvtnq_s32_f32(v_X01f);
            vCoordY0123i = vcvtnq_s32_f32(v_Y01f);
        }

        // 4 5 6 7
        int32x4_t vCoordX4567i, vCoordY4567i;
        {
            float64x2_t v_W0d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X0d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y0d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            float64x2_t v_W1d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X1d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y1d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            v_X0d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W0d), v_X0d);
            v_Y0d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W0d), v_Y0d);
            v_X1d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W1d), v_X1d);
            v_Y1d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W1d), v_Y1d);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_intminf), v_intmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_intminf), v_intmaxf);

            vCoordX4567i = vcvtnq_s32_f32(v_X01f);
            vCoordY4567i = vcvtnq_s32_f32(v_Y01f);
        }

        // 8 9 10 11
        int32x4_t vCoordX89abi, vCoordY89abi;
        {
            float64x2_t v_W0d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X0d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y0d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            float64x2_t v_W1d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X1d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y1d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            v_X0d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W0d), v_X0d);
            v_Y0d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W0d), v_Y0d);
            v_X1d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W1d), v_X1d);
            v_Y1d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W1d), v_Y1d);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_intminf), v_intmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_intminf), v_intmaxf);

            vCoordX89abi = vcvtnq_s32_f32(v_X01f);
            vCoordY89abi = vcvtnq_s32_f32(v_Y01f);
        }

        // 12 13 14 15
        int32x4_t vCoordXcdefi, vCoordYcdefi;
        {
            float64x2_t v_W0d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X0d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y0d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            float64x2_t v_W1d = vmlaq_f64(v_W0d_c, v_baseXd, v_M6d);
            float64x2_t v_X1d = vmlaq_f64(v_X0d_c, v_baseXd, v_M0d);
            float64x2_t v_Y1d = vmlaq_f64(v_Y0d_c, v_baseXd, v_M3d);
            v_baseXd = vaddq_f64(v_baseXd, v_2d);

            v_X0d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W0d), v_X0d);
            v_Y0d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W0d), v_Y0d);
            v_X1d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W1d), v_X1d);
            v_Y1d = vmulq_f64(vdivq_f64(v_interTblSzd, v_W1d), v_Y1d);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_intminf), v_intmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_intminf), v_intmaxf);

            vCoordXcdefi = vcvtnq_s32_f32(v_X01f);
            vCoordYcdefi = vcvtnq_s32_f32(v_Y01f);
        }

        // construct alpha vector
        // truncate as needed value is less than 16
        int16x4_t v_AlphaX0Halfh = vmovn_s32(vCoordX0123i);
        int16x8_t v_AlphaX0h = vmovn_high_s32(v_AlphaX0Halfh, vCoordX4567i);
        int16x4_t v_AlphaY0Halfh = vmovn_s32(vCoordY0123i);
        int16x8_t v_AlphaY0h = vmovn_high_s32(v_AlphaY0Halfh, vCoordY4567i);
        int16x4_t v_AlphaX1Halfh = vmovn_s32(vCoordX89abi);
        int16x8_t v_AlphaX1h = vmovn_high_s32(v_AlphaX1Halfh, vCoordXcdefi);
        int16x4_t v_AlphaY1Halfh = vmovn_s32(vCoordY89abi);
        int16x8_t v_AlphaY1h = vmovn_high_s32(v_AlphaY1Halfh, vCoordYcdefi);
        // Use SHLI to do shift and combine in single instruction
        v_AlphaY0h = vandq_s16(v_AlphaY0h, v_interTblMaski);
        v_AlphaY1h = vandq_s16(v_AlphaY1h, v_interTblMaski);
        int16x8_t v_Alpha0h = vsliq_n_s16(v_AlphaX0h, v_AlphaY0h, INTER_TABLE_BITS);
        int16x8_t v_Alpha1h = vsliq_n_s16(v_AlphaX1h, v_AlphaY1h, INTER_TABLE_BITS);
        // store to alpha map
        vst1q_s16(alpha_map + jj, v_Alpha0h);
        vst1q_s16(alpha_map + jj + 8, v_Alpha1h);

        // use RSHRN to do shift and narrow in single instruction
        int16x8x2_t vCoord0, vCoord1;
        int16x4_t vCoord0XHalf = vqshrn_n_s32(vCoordX0123i, INTER_TABLE_BITS);
        vCoord0.val[0] = vqshrn_high_n_s32(vCoord0XHalf, vCoordX4567i, INTER_TABLE_BITS);
        int16x4_t vCoord0YHalf = vqshrn_n_s32(vCoordY0123i, INTER_TABLE_BITS);
        vCoord0.val[1] = vqshrn_high_n_s32(vCoord0YHalf, vCoordY4567i, INTER_TABLE_BITS);
        vst2q_s16(coord_map + jj * 2, vCoord0);

        int16x4_t vCoord1XHalf = vqshrn_n_s32(vCoordX89abi, INTER_TABLE_BITS);
        vCoord1.val[0] = vqshrn_high_n_s32(vCoord1XHalf, vCoordXcdefi, INTER_TABLE_BITS);
        int16x4_t vCoord1YHalf = vqshrn_n_s32(vCoordY89abi, INTER_TABLE_BITS);
        vCoord1.val[1] = vqshrn_high_n_s32(vCoord1YHalf, vCoordYcdefi, INTER_TABLE_BITS);
        vst2q_s16(coord_map + jj * 2 + 16, vCoord1);
    }

    for (; jj < bw; jj++) {
        double W = W0 + M[6] * jj;
        W = W ? INTER_TABLE_SIZE / W : 0;
        double Xd = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * jj) * W));
        double Yd = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * jj) * W));
        int X = lrint(Xd);
        int Y = lrint(Yd);

        coord_map[jj * 2] = clip((X >> INTER_TABLE_BITS), SHRT_MIN, SHRT_MAX);
        coord_map[jj * 2 + 1] = clip((Y >> INTER_TABLE_BITS), SHRT_MIN, SHRT_MAX);
        alpha_map[jj] = (short)(((Y & (INTER_TABLE_SIZE - 1)) << INTER_TABLE_BITS) + (X & (INTER_TABLE_SIZE - 1)));
    }
}

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpperspective_nearest(int32_t inHeight,
                                               int32_t inWidth,
                                               int32_t inWidthStride,
                                               int32_t outHeight,
                                               int32_t outWidth,
                                               int32_t outWidthStride,
                                               T* dst,
                                               const T* src,
                                               const double M[][3],
                                               T delta = 0)
{
    int16_t coord_data[BLOCK_SIZE * BLOCK_SIZE * 2];
    int16_t* coord_map = (int16_t*)&coord_data[0];

    for (int32_t i = 0; i < outHeight; i += BLOCK_SIZE) {
        for (int32_t j = 0; j < outWidth; j += BLOCK_SIZE) {
            int bh = std::min(BLOCK_SIZE, outHeight - i);
            int bw = std::min(BLOCK_SIZE, outWidth - j);
            double baseWb = M[2][0] * j + M[2][1] * i + M[2][2];
            double baseXb = M[0][0] * j + M[0][1] * i + M[0][2];
            double baseYb = M[1][0] * j + M[1][1] * i + M[1][2];
            for (int32_t bi = 0; bi < bh; bi++) {
                double baseW = M[2][1] * bi + baseWb;
                double baseX = M[0][1] * bi + baseXb;
                double baseY = M[1][1] * bi + baseYb;
                WarpPerspective_CoordCompute_Nearest_Line(
                    &M[0][0], coord_map + BLOCK_SIZE * bi * 2, bw, baseX, baseY, baseW);
            }

            for (int32_t bi = 0; bi < bh; bi++) {
                for (int32_t bj = 0; bj < bw; bj++) {
                    int ii = i + bi;
                    int jj = j + bj;
                    int32_t sx = coord_map[bi * BLOCK_SIZE * 2 + bj * 2];
                    int32_t sy = coord_map[bi * BLOCK_SIZE * 2 + bj * 2 + 1];
                    if (borderMode == ppl::cv::BORDER_CONSTANT) {
                        int32_t idxSrc = sy * inWidthStride + sx * nc;
                        int32_t idxDst = ii * outWidthStride + jj * nc;
                        bool cond = (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight);
                        for (int32_t k = 0; k < nc; k++) {
                            dst[idxDst + k] = cond ? src[idxSrc + k] : delta;
                        }
                    } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                        sx = clip(sx, 0, inWidth - 1);
                        sy = clip(sy, 0, inHeight - 1);
                        int32_t idxSrc = sy * inWidthStride + sx * nc;
                        int32_t idxDst = ii * outWidthStride + jj * nc;
                        for (int32_t k = 0; k < nc; k++) {
                            dst[idxDst + k] = src[idxSrc + k];
                        }
                    } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                            int32_t idxSrc = sy * inWidthStride + sx * nc;
                            int32_t idxDst = ii * outWidthStride + jj * nc;
                            for (int32_t k = 0; k < nc; k++) {
                                dst[idxDst + k] = src[idxSrc + k];
                            }
                        }
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpperspective_linear(int32_t inHeight,
                                              int32_t inWidth,
                                              int32_t inWidthStride,
                                              int32_t outHeight,
                                              int32_t outWidth,
                                              int32_t outWidthStride,
                                              T* dst,
                                              const T* src,
                                              const double M[][3],
                                              T delta = 0)
{
    int16_t coord_data[BLOCK_SIZE * BLOCK_SIZE * 2];
    int16_t alpha_data[BLOCK_SIZE * BLOCK_SIZE];
    int16_t* coord_map = (int16_t*)&coord_data[0];
    int16_t* alpha_map = (int16_t*)&alpha_data[0];

    for (int32_t i = 0; i < outHeight; i += BLOCK_SIZE) {
        for (int32_t j = 0; j < outWidth; j += BLOCK_SIZE) {
            int bh = std::min(BLOCK_SIZE, outHeight - i);
            int bw = std::min(BLOCK_SIZE, outWidth - j);
            double baseWb = M[2][0] * j + M[2][1] * i + M[2][2];
            double baseXb = M[0][0] * j + M[0][1] * i + M[0][2];
            double baseYb = M[1][0] * j + M[1][1] * i + M[1][2];
            for (int32_t bi = 0; bi < bh; bi++) {
                double baseW = M[2][1] * bi + baseWb;
                double baseX = M[0][1] * bi + baseXb;
                double baseY = M[1][1] * bi + baseYb;
                WarpPerspective_CoordCompute_Linear_Line(
                    &M[0][0], coord_map + BLOCK_SIZE * bi * 2, alpha_map + BLOCK_SIZE * bi, bw, baseX, baseY, baseW);
            }

            if (std::is_same<T, uint8_t>::value) {
                remapBilinear<uint8_t, int16_t, int32_t, nc, borderMode>(
                    reinterpret_cast<const uint8_t*>(src),
                    inHeight,
                    inWidth,
                    inWidthStride,
                    reinterpret_cast<uint8_t*>(dst + i * outWidthStride + j * nc),
                    bh,
                    bw,
                    outWidthStride,
                    coord_map,
                    alpha_map,
                    &BilinearTab_i[0][0][0],
                    delta);
            } else if (std::is_same<T, float>::value) {
                remapBilinear<float, float, float, nc, borderMode>(
                    reinterpret_cast<const float*>(src),
                    inHeight,
                    inWidth,
                    inWidthStride,
                    reinterpret_cast<float*>(dst + i * outWidthStride + j * nc),
                    bh,
                    bw,
                    outWidthStride,
                    coord_map,
                    alpha_map,
                    &BilinearTab_f[0][0][0],
                    delta);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t nc>
::ppl::common::RetCode WarpPerspectiveNearestPoint(int32_t inHeight,
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

    if (border_type == ppl::cv::BORDER_CONSTANT) {
        warpperspective_nearest<T, nc, ppl::cv::BORDER_CONSTANT>(
            inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        warpperspective_nearest<T, nc, ppl::cv::BORDER_REPLICATE>(
            inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
    } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
        warpperspective_nearest<T, nc, ppl::cv::BORDER_TRANSPARENT>(
            inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
    }

    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t nc>
::ppl::common::RetCode WarpPerspectiveLinear(int32_t inHeight,
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

    if (border_type == ppl::cv::BORDER_CONSTANT) {
        warpperspective_linear<T, nc, ppl::cv::BORDER_CONSTANT>(
            inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        warpperspective_linear<T, nc, ppl::cv::BORDER_REPLICATE>(
            inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
    } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
        warpperspective_linear<T, nc, ppl::cv::BORDER_TRANSPARENT>(
            inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, M, border_value);
    }

    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t nc>
::ppl::common::RetCode WarpPerspective(int32_t inHeight,
                                       int32_t inWidth,
                                       int32_t inWidthStride,
                                       const T* inData,
                                       int32_t outHeight,
                                       int32_t outWidth,
                                       int32_t outWidthStride,
                                       T* outData,
                                       const double* affineMatrix,
                                       InterpolationType interpolation,
                                       BorderType border_type,
                                       T border_value)
{
    if (interpolation == INTERPOLATION_LINEAR) {
        return WarpPerspectiveLinear<T, nc>(inHeight,
                                            inWidth,
                                            inWidthStride,
                                            inData,
                                            outHeight,
                                            outWidth,
                                            outWidthStride,
                                            outData,
                                            affineMatrix,
                                            border_type,
                                            border_value);
    } else if (interpolation == INTERPOLATION_NEAREST_POINT) {
        return WarpPerspectiveNearestPoint<T, nc>(inHeight,
                                                  inWidth,
                                                  inWidthStride,
                                                  inData,
                                                  outHeight,
                                                  outWidth,
                                                  outWidthStride,
                                                  outData,
                                                  affineMatrix,
                                                  border_type,
                                                  border_value);
    } else {
        return ppl::common::RC_INVALID_VALUE;
    }
}

template ::ppl::common::RetCode WarpPerspective<float, 1>(int32_t inHeight,
                                                          int32_t inWidth,
                                                          int32_t inWidthStride,
                                                          const float* inData,
                                                          int32_t outHeight,
                                                          int32_t outWidth,
                                                          int32_t outWidthStride,
                                                          float* outData,
                                                          const double* affineMatrix,
                                                          InterpolationType interpolation,
                                                          BorderType border_type,
                                                          float border_value);

template ::ppl::common::RetCode WarpPerspective<float, 3>(int32_t inHeight,
                                                          int32_t inWidth,
                                                          int32_t inWidthStride,
                                                          const float* inData,
                                                          int32_t outHeight,
                                                          int32_t outWidth,
                                                          int32_t outWidthStride,
                                                          float* outData,
                                                          const double* affineMatrix,
                                                          InterpolationType interpolation,
                                                          BorderType border_type,
                                                          float border_value);

template ::ppl::common::RetCode WarpPerspective<float, 4>(int32_t inHeight,
                                                          int32_t inWidth,
                                                          int32_t inWidthStride,
                                                          const float* inData,
                                                          int32_t outHeight,
                                                          int32_t outWidth,
                                                          int32_t outWidthStride,
                                                          float* outData,
                                                          const double* affineMatrix,
                                                          InterpolationType interpolation,
                                                          BorderType border_type,
                                                          float border_value);

template ::ppl::common::RetCode WarpPerspective<uint8_t, 1>(int32_t inHeight,
                                                            int32_t inWidth,
                                                            int32_t inWidthStride,
                                                            const uint8_t* inData,
                                                            int32_t outHeight,
                                                            int32_t outWidth,
                                                            int32_t outWidthStride,
                                                            uint8_t* outData,
                                                            const double* affineMatrix,
                                                            InterpolationType interpolation,
                                                            BorderType border_type,
                                                            uint8_t border_value);

template ::ppl::common::RetCode WarpPerspective<uint8_t, 3>(int32_t inHeight,
                                                            int32_t inWidth,
                                                            int32_t inWidthStride,
                                                            const uint8_t* inData,
                                                            int32_t outHeight,
                                                            int32_t outWidth,
                                                            int32_t outWidthStride,
                                                            uint8_t* outData,
                                                            const double* affineMatrix,
                                                            InterpolationType interpolation,
                                                            BorderType border_type,
                                                            uint8_t border_value);

template ::ppl::common::RetCode WarpPerspective<uint8_t, 4>(int32_t inHeight,
                                                            int32_t inWidth,
                                                            int32_t inWidthStride,
                                                            const uint8_t* inData,
                                                            int32_t outHeight,
                                                            int32_t outWidth,
                                                            int32_t outWidthStride,
                                                            uint8_t* outData,
                                                            const double* affineMatrix,
                                                            InterpolationType interpolation,
                                                            BorderType border_type,
                                                            uint8_t border_value);

}
}
} // namespace ppl::cv::arm
