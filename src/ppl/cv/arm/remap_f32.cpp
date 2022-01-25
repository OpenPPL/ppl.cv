// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for mulitional information
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

#include "ppl/cv/arm/remap.h"
#include "ppl/cv/types.h"
#include <arm_neon.h>
#include <limits.h>
#include <math.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace arm {

template <typename T>
static inline T clip(T value, T min_value, T max_value)
{
    return std::min(std::max(value, min_value), max_value);
}

static inline void vector_mlaq_lane0(float32x4_t& sum, float32x4_t& a, float32x4_t& b)
{
    asm volatile("fmla %0.4s, %2.4s, %3.s[0]  \r\n"
                 : "=w"(sum)
                 : "0"(sum), "w"(a), "w"(b)
                 :);
}

static inline void vector_mlaq_lane1(float32x4_t& sum, float32x4_t& a, float32x4_t& b)
{
    asm volatile("fmla %0.4s, %2.4s, %3.s[1]  \r\n"
                 : "=w"(sum)
                 : "0"(sum), "w"(a), "w"(b)
                 :);
}

static inline void vector_mlaq_lane2(float32x4_t& sum, float32x4_t& a, float32x4_t& b)
{
    asm volatile("fmla %0.4s, %2.4s, %3.s[2]  \r\n"
                 : "=w"(sum)
                 : "0"(sum), "w"(a), "w"(b)
                 :);
}

static inline void vector_mlaq_lane3(float32x4_t& sum, float32x4_t& a, float32x4_t& b)
{
    asm volatile("fmla %0.4s, %2.4s, %3.s[3]  \r\n"
                 : "=w"(sum)
                 : "0"(sum), "w"(a), "w"(b)
                 :);
}

static inline void vector_mulq_lane0(float32x4_t& d, float32x4_t& a, float32x4_t& b)
{
    asm volatile("fmul %0.4s, %1.4s, %2.s[0]  \r\n"
                 : "=w"(d)
                 : "w"(a), "w"(b)
                 :);
}

static inline void vector_mulq_lane1(float32x4_t& d, float32x4_t& a, float32x4_t& b)
{
    asm volatile("fmul %0.4s, %1.4s, %2.s[1]  \r\n"
                 : "=w"(d)
                 : "w"(a), "w"(b)
                 :);
}

static inline void vector_mulq_lane2(float32x4_t& d, float32x4_t& a, float32x4_t& b)
{
    asm volatile("fmul %0.4s, %1.4s, %2.s[2]  \r\n"
                 : "=w"(d)
                 : "w"(a), "w"(b)
                 :);
}

static inline void vector_mulq_lane3(float32x4_t& d, float32x4_t& a, float32x4_t& b)
{
    asm volatile("fmul %0.4s, %1.4s, %2.s[3]  \r\n"
                 : "=w"(d)
                 : "w"(a), "w"(b)
                 :);
}

const int32_t INTER_BITS = 5;
const int32_t INTER_TAB_SIZE = 1 << INTER_BITS;
static float BilinearTab_f[INTER_TAB_SIZE * INTER_TAB_SIZE][2][2];

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
    static bool is_inited = false; // may cause trouble in multi-thread program
    if (is_inited) {
        return;
    }
    float* tab = 0;
    int32_t ksize = 0;
    tab = BilinearTab_f[0][0], ksize = 2;

    float* _tab = new float[8 * INTER_TAB_SIZE];
    int32_t i, j, k1, k2;
    initInterTab1D(_tab, INTER_TAB_SIZE);
    for (i = 0; i < INTER_TAB_SIZE; i++) {
        for (j = 0; j < INTER_TAB_SIZE; j++, tab += ksize * ksize) {
            for (k1 = 0; k1 < ksize; k1++) {
                float vy = _tab[i * ksize + k1];
                for (k2 = 0; k2 < ksize; k2++) {
                    float v = vy * _tab[j * ksize + k2];
                    tab[k1 * ksize + k2] = v;
                }
            }
        }
    }
    is_inited = true;
}

static inline int32_t round2even(float value)
{
    return vcvtns_s32_f32(value);
}

static inline int32x4_t vector_round2even_s32_f32(float32x4_t v)
{
    return vcvtnq_s32_f32(v);
}

template <int32_t nc, cv::BorderType borderMode>
void remap_linear_float(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* map_x,
    const float* map_y,
    float borderValue)
{
    initInterTab2D();
    const int32_t edge_thresh = (outWidth / 4) * 4;
    for (int32_t row = 0; row < outHeight; row++) {
        for (int32_t col = 0; col < outWidth; col++) {
            int32_t idxXY = row * outWidth + col;
            int32_t idxDst = row * outWidthStride + col * nc;
            int32_t sx, sy;
            if (col < edge_thresh) { // opencv armv7 use different round in edge area
                sx = round2even(map_x[idxXY] * INTER_TAB_SIZE);
                sy = round2even(map_y[idxXY] * INTER_TAB_SIZE);
            } else {
                sx = round2even(map_x[idxXY] * INTER_TAB_SIZE);
                sy = round2even(map_y[idxXY] * INTER_TAB_SIZE);
            }
            int32_t v = (sy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
            int32_t sx0 = sx >> INTER_BITS;
            int32_t sy0 = sy >> INTER_BITS;
            float* tab = BilinearTab_f[v][0];
            float v0, v1, v2, v3;
            if (borderMode == ppl::cv::BORDER_CONSTANT) {
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                for (int32_t k = 0; k < nc; k++) {
                    v0 = flag0 ? inData[position1 + k] : borderValue;
                    v1 = flag1 ? inData[position1 + nc + k] : borderValue;
                    v2 = flag2 ? inData[position2 + k] : borderValue;
                    v3 = flag3 ? inData[position2 + nc + k] : borderValue;
                    float sum = (v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3]);
                    outData[idxDst + k] = sum;
                }
            } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                int32_t sx1 = sx0 + 1;
                int32_t sy1 = sy0 + 1;
                sx0 = clip(sx0, 0, inWidth - 1);
                sx1 = clip(sx1, 0, inWidth - 1);
                sy0 = clip(sy0, 0, inHeight - 1);
                sy1 = clip(sy1, 0, inHeight - 1);
                const float* t0 = inData + sy0 * inWidthStride + sx0 * nc;
                const float* t1 = inData + sy0 * inWidthStride + sx1 * nc;
                const float* t2 = inData + sy1 * inWidthStride + sx0 * nc;
                const float* t3 = inData + sy1 * inWidthStride + sx1 * nc;
                for (int32_t k = 0; k < nc; ++k) {
                    float sum = (t0[k] * tab[0] + t1[k] * tab[1] + t2[k] * tab[2] + t3[k] * tab[3]);
                    outData[idxDst + k] = sum;
                }
            } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                bool all_in = flag0 && flag1 && flag2 && flag3;
                if (!all_in) {
                    continue;
                }
                for (int32_t k = 0; k < nc; k++) {
                    v0 = inData[position1 + k];
                    v1 = inData[position1 + nc + k];
                    v2 = inData[position2 + k];
                    v3 = inData[position2 + nc + k];
                    float sum = (v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3]);
                    outData[idxDst + k] = sum;
                }
            }
        }
    }
} // remap_linear_float

template <int32_t nc>
void remap_linear_float_constant(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* map_x,
    const float* map_y,
    float borderValue)
{
    initInterTab2D();
    if (nc == 1) {
        const int32x4_t C_bitmask = vdupq_n_s32(INTER_TAB_SIZE - 1);
        for (int32_t row = 0; row < outHeight; row++) {
            int32_t col = 0;
            int32_t idxXY = row * outWidth;
            int32_t idxDst = row * outWidthStride;
            for (; col + 4 <= outWidth; col += 4, idxXY += 4) {
                float32x4_t v_mapx = vmulq_n_f32(vld1q_f32(map_x + idxXY), (float)INTER_TAB_SIZE);
                float32x4_t v_mapy = vmulq_n_f32(vld1q_f32(map_y + idxXY), (float)INTER_TAB_SIZE);
                int32x4_t v_sx0 = vector_round2even_s32_f32(v_mapx);
                int32x4_t v_sy0 = vector_round2even_s32_f32(v_mapy);
                int32x4_t v_tabx = vandq_s32(v_sx0, C_bitmask);
                int32x4_t v_taby = vandq_s32(v_sy0, C_bitmask);
                int32x4_t v_tab = vmlaq_n_s32(v_tabx, v_taby, INTER_TAB_SIZE);
                v_sx0 = vshrq_n_s32(v_sx0, INTER_BITS);
                v_sy0 = vshrq_n_s32(v_sy0, INTER_BITS);

                int32_t sx0 = vgetq_lane_s32(v_sx0, 0); // loop unroll 4
                int32_t sy0 = vgetq_lane_s32(v_sy0, 0);
                float* tab = BilinearTab_f[vgetq_lane_s32(v_tab, 0)][0];
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                float v0 = flag0 ? inData[position1] : borderValue;
                float v1 = flag1 ? inData[position1 + 1] : borderValue;
                float v2 = flag2 ? inData[position2] : borderValue;
                float v3 = flag3 ? inData[position2 + 1] : borderValue;
                float sum = (v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3]);
                outData[idxDst++] = sum;

                sx0 = vgetq_lane_s32(v_sx0, 1);
                sy0 = vgetq_lane_s32(v_sy0, 1);
                tab = BilinearTab_f[vgetq_lane_s32(v_tab, 1)][0];
                flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                position1 = (sy0 * inWidthStride + sx0 * nc);
                position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                v0 = flag0 ? inData[position1] : borderValue;
                v1 = flag1 ? inData[position1 + 1] : borderValue;
                v2 = flag2 ? inData[position2] : borderValue;
                v3 = flag3 ? inData[position2 + 1] : borderValue;
                sum = (v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3]);
                outData[idxDst++] = sum;

                sx0 = vgetq_lane_s32(v_sx0, 2);
                sy0 = vgetq_lane_s32(v_sy0, 2);
                tab = BilinearTab_f[vgetq_lane_s32(v_tab, 2)][0];
                flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                position1 = (sy0 * inWidthStride + sx0 * nc);
                position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                v0 = flag0 ? inData[position1] : borderValue;
                v1 = flag1 ? inData[position1 + 1] : borderValue;
                v2 = flag2 ? inData[position2] : borderValue;
                v3 = flag3 ? inData[position2 + 1] : borderValue;
                sum = (v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3]);
                outData[idxDst++] = sum;

                sx0 = vgetq_lane_s32(v_sx0, 3);
                sy0 = vgetq_lane_s32(v_sy0, 3);
                tab = BilinearTab_f[vgetq_lane_s32(v_tab, 3)][0];
                flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                position1 = (sy0 * inWidthStride + sx0 * nc);
                position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                v0 = flag0 ? inData[position1] : borderValue;
                v1 = flag1 ? inData[position1 + 1] : borderValue;
                v2 = flag2 ? inData[position2] : borderValue;
                v3 = flag3 ? inData[position2 + 1] : borderValue;
                sum = (v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3]);
                outData[idxDst++] = sum;
            }
            for (; col < outWidth; col++, idxXY++) {
                float v0, v1, v2, v3;
                int32_t sx = round2even(map_x[idxXY] * INTER_TAB_SIZE);
                int32_t sy = round2even(map_y[idxXY] * INTER_TAB_SIZE);
                int32_t v = (sy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
                int32_t sx0 = sx >> INTER_BITS;
                int32_t sy0 = sy >> INTER_BITS;
                float* tab = BilinearTab_f[v][0];
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                v0 = flag0 ? inData[position1] : borderValue;
                v1 = flag1 ? inData[position1 + 1] : borderValue;
                v2 = flag2 ? inData[position2] : borderValue;
                v3 = flag3 ? inData[position2 + 1] : borderValue;
                float sum = v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                outData[idxDst++] = sum;
            }
        }
    } else if (nc == 3 || nc == 4) {
        const int32x4_t C_bitmask = vdupq_n_s32(INTER_TAB_SIZE - 1);
        for (int32_t row = 0; row < outHeight; row++) {
            int32_t col = 0;
            int32_t idxXY = row * outWidth;
            int32_t idxDst = row * outWidthStride;
            for (; col + 4 <= outWidth; col += 4, idxXY += 4) {
                float32x4_t v_mapx = vmulq_n_f32(vld1q_f32(map_x + idxXY), (float)INTER_TAB_SIZE);
                float32x4_t v_mapy = vmulq_n_f32(vld1q_f32(map_y + idxXY), (float)INTER_TAB_SIZE);
                int32x4_t v_sx0 = vector_round2even_s32_f32(v_mapx);
                int32x4_t v_sy0 = vector_round2even_s32_f32(v_mapy);
                int32x4_t v_tabx = vandq_s32(v_sx0, C_bitmask);
                int32x4_t v_taby = vandq_s32(v_sy0, C_bitmask);
                int32x4_t v_tab = vmlaq_n_s32(v_tabx, v_taby, INTER_TAB_SIZE);
                v_sx0 = vshrq_n_s32(v_sx0, INTER_BITS);
                v_sy0 = vshrq_n_s32(v_sy0, INTER_BITS);

                int32_t sx0 = vgetq_lane_s32(v_sx0, 0); // loop unroll 4
                int32_t sy0 = vgetq_lane_s32(v_sy0, 0);
                float* tab_ptr = BilinearTab_f[vgetq_lane_s32(v_tab, 0)][0];
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                float32x4_t tab = vld1q_f32(tab_ptr);
                float32x4_t v0 = flag0 ? vld1q_f32(inData + position1) : vdupq_n_f32(borderValue);
                float32x4_t v1 = flag1 ? vld1q_f32(inData + position1 + nc) : vdupq_n_f32(borderValue);
                float32x4_t v2 = flag2 ? vld1q_f32(inData + position2) : vdupq_n_f32(borderValue);
                float32x4_t v3 = flag3 ? vld1q_f32(inData + position2 + nc) : vdupq_n_f32(borderValue);
                float32x4_t sum;
                vector_mulq_lane0(sum, v0, tab);
                vector_mlaq_lane1(sum, v1, tab);
                vector_mlaq_lane2(sum, v2, tab);
                vector_mlaq_lane3(sum, v3, tab);
                vst1q_f32(outData + idxDst, sum);
                idxDst += nc;

                sx0 = vgetq_lane_s32(v_sx0, 1);
                sy0 = vgetq_lane_s32(v_sy0, 1);
                tab_ptr = BilinearTab_f[vgetq_lane_s32(v_tab, 1)][0];
                flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                position1 = (sy0 * inWidthStride + sx0 * nc);
                position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                tab = vld1q_f32(tab_ptr);
                v0 = flag0 ? vld1q_f32(inData + position1) : vdupq_n_f32(borderValue);
                v1 = flag1 ? vld1q_f32(inData + position1 + nc) : vdupq_n_f32(borderValue);
                v2 = flag2 ? vld1q_f32(inData + position2) : vdupq_n_f32(borderValue);
                v3 = flag3 ? vld1q_f32(inData + position2 + nc) : vdupq_n_f32(borderValue);
                vector_mulq_lane0(sum, v0, tab);
                vector_mlaq_lane1(sum, v1, tab);
                vector_mlaq_lane2(sum, v2, tab);
                vector_mlaq_lane3(sum, v3, tab);
                vst1q_f32(outData + idxDst, sum);
                idxDst += nc;

                sx0 = vgetq_lane_s32(v_sx0, 2);
                sy0 = vgetq_lane_s32(v_sy0, 2);
                tab_ptr = BilinearTab_f[vgetq_lane_s32(v_tab, 2)][0];
                flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                position1 = (sy0 * inWidthStride + sx0 * nc);
                position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                tab = vld1q_f32(tab_ptr);
                v0 = flag0 ? vld1q_f32(inData + position1) : vdupq_n_f32(borderValue);
                v1 = flag1 ? vld1q_f32(inData + position1 + nc) : vdupq_n_f32(borderValue);
                v2 = flag2 ? vld1q_f32(inData + position2) : vdupq_n_f32(borderValue);
                v3 = flag3 ? vld1q_f32(inData + position2 + nc) : vdupq_n_f32(borderValue);
                vector_mulq_lane0(sum, v0, tab);
                vector_mlaq_lane1(sum, v1, tab);
                vector_mlaq_lane2(sum, v2, tab);
                vector_mlaq_lane3(sum, v3, tab);
                vst1q_f32(outData + idxDst, sum);
                idxDst += nc;

                sx0 = vgetq_lane_s32(v_sx0, 3);
                sy0 = vgetq_lane_s32(v_sy0, 3);
                tab_ptr = BilinearTab_f[vgetq_lane_s32(v_tab, 3)][0];
                flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                position1 = (sy0 * inWidthStride + sx0 * nc);
                position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                tab = vld1q_f32(tab_ptr);
                v0 = flag0 ? vld1q_f32(inData + position1) : vdupq_n_f32(borderValue);
                v1 = flag1 ? vld1q_f32(inData + position1 + nc) : vdupq_n_f32(borderValue);
                v2 = flag2 ? vld1q_f32(inData + position2) : vdupq_n_f32(borderValue);
                v3 = flag3 ? vld1q_f32(inData + position2 + nc) : vdupq_n_f32(borderValue);
                vector_mulq_lane0(sum, v0, tab);
                vector_mlaq_lane1(sum, v1, tab);
                vector_mlaq_lane2(sum, v2, tab);
                vector_mlaq_lane3(sum, v3, tab);
                vst1q_f32(outData + idxDst, sum);
                idxDst += nc;
            }
            for (; col < outWidth; col++, idxXY++) {
                float v0, v1, v2, v3;
                int32_t sx = round2even(map_x[idxXY] * INTER_TAB_SIZE);
                int32_t sy = round2even(map_y[idxXY] * INTER_TAB_SIZE);
                int32_t v = (sy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1));
                int32_t sx0 = sx >> INTER_BITS;
                int32_t sy0 = sy >> INTER_BITS;
                float* tab = BilinearTab_f[v][0];
                bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                for (int32_t k = 0; k < nc; k++) {
                    v0 = flag0 ? inData[position1 + k] : borderValue;
                    v1 = flag1 ? inData[position1 + nc + k] : borderValue;
                    v2 = flag2 ? inData[position2 + k] : borderValue;
                    v3 = flag3 ? inData[position2 + nc + k] : borderValue;
                    float sum = v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                    outData[idxDst + k] = sum;
                }
                idxDst += nc;
            }
        }
    }
} //remap_linear_float_constant

template <int32_t nc, cv::BorderType borderMode>
void remap_nearest_float(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* map_x,
    const float* map_y,
    float borderValue)
{
    const int32_t edge_thresh = (outWidth / 4) * 4;
    for (int32_t row = 0; row < outHeight; row++) {
        for (int32_t col = 0; col < outWidth; col++) {
            int32_t idxDst = row * outWidthStride + col * nc;
            int32_t idxMap = row * outWidth + col;
            float x = map_x[idxMap];
            float y = map_y[idxMap];
            int32_t sx, sy;
            if (col < edge_thresh) { // opencv armv7 use different round in edge area
                sx = round2even(x);
                sy = round2even(y);
            } else {
                sx = round2even(x);
                sy = round2even(y);
            }
            if (borderMode == ppl::cv::BORDER_CONSTANT) {
                int32_t idxSrc = sy * inWidthStride + sx * nc;
                if (sx >= 0 && sx < inWidth && sy >= 0 && sy < inHeight) {
                    for (int32_t i = 0; i < nc; i++)
                        outData[idxDst + i] = inData[idxSrc + i];
                } else {
                    for (int32_t i = 0; i < nc; i++) {
                        outData[idxDst + i] = borderValue;
                    }
                }
            } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                sx = clip(sx, 0, inWidth - 1);
                sy = clip(sy, 0, inHeight - 1);
                int32_t idxSrc = sy * inWidthStride + sx * nc;
                for (int32_t i = 0; i < nc; i++) {
                    outData[idxDst + i] = inData[idxSrc + i];
                }
            } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                bool in = (sx >= 0) && (sx < inWidth) && (sy >= 0) && (sy < inHeight);
                if (!in) {
                    continue;
                }
                int32_t idxSrc = sy * inWidthStride + sx * nc;
                for (int32_t i = 0; i < nc; i++) {
                    outData[idxDst + i] = inData[idxSrc + i];
                }
            }
        }
    }
} // remap_nearest_float

template <>
::ppl::common::RetCode RemapLinear<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type,
    float borderValue)
{
    if (nullptr == outData || nullptr == inData || nullptr == mapx || nullptr == mapy) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || outHeight <= 0 || outWidth <= 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (border_type == BORDER_CONSTANT) {
        remap_linear_float_constant<1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_REPLICATE) {
        remap_linear_float<1, BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_TRANSPARENT) {
        remap_linear_float<1, BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RemapLinear<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type,
    float borderValue)
{
    if (nullptr == outData || nullptr == inData || nullptr == mapx || nullptr == mapy) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || outHeight <= 0 || outWidth <= 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (border_type == BORDER_CONSTANT) {
        remap_linear_float_constant<3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_REPLICATE) {
        remap_linear_float<3, BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_TRANSPARENT) {
        remap_linear_float<3, BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RemapLinear<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type,
    float borderValue)
{
    if (nullptr == outData || nullptr == inData || nullptr == mapx || nullptr == mapy) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || outHeight <= 0 || outWidth <= 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (border_type == BORDER_CONSTANT) {
        remap_linear_float_constant<4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_REPLICATE) {
        remap_linear_float<4, BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_TRANSPARENT) {
        remap_linear_float<4, BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RemapNearestPoint<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type,
    float borderValue)
{
    if (nullptr == outData || nullptr == inData || nullptr == mapx || nullptr == mapy) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || outHeight <= 0 || outWidth <= 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (border_type == BORDER_CONSTANT) {
        remap_nearest_float<1, BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_REPLICATE) {
        remap_nearest_float<1, BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_TRANSPARENT) {
        remap_nearest_float<1, BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RemapNearestPoint<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type,
    float borderValue)
{
    if (nullptr == outData || nullptr == inData || nullptr == mapx || nullptr == mapy) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || outHeight <= 0 || outWidth <= 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (border_type == BORDER_CONSTANT) {
        remap_nearest_float<3, BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_REPLICATE) {
        remap_nearest_float<3, BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_TRANSPARENT) {
        remap_nearest_float<3, BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RemapNearestPoint<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type,
    float borderValue)
{
    if (nullptr == outData || nullptr == inData || nullptr == mapx || nullptr == mapy) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || outHeight <= 0 || outWidth <= 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (border_type == BORDER_CONSTANT) {
        remap_nearest_float<4, BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_REPLICATE) {
        remap_nearest_float<4, BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    } else if (border_type == BORDER_TRANSPARENT) {
        remap_nearest_float<4, BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData, mapx, mapy, borderValue);
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
