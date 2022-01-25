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

#include "ppl/cv/arm/warpaffine.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/log.h"
#include "common.hpp"
#include "operation_utils.hpp"
#include <arm_neon.h>
#include <limits.h>
#include <algorithm>
#include <cmath>

namespace ppl {
namespace cv {
namespace arm {

template <typename _Tp>
static inline _Tp *alignPtr(_Tp *ptr, int32_t n = (int32_t)sizeof(_Tp))
{
    return (_Tp *)(((size_t)ptr + n - 1) & -n);
}

template <typename T>
inline T *getRowPtr(T *base, int32_t stride, int32_t row)
{
    T *baseRaw = const_cast<T *>(reinterpret_cast<const T *>(base));
    return reinterpret_cast<T *>(baseRaw + row * stride);
}

template <typename T>
inline const T round(const T &a, const T &b)
{
    return a / b * b;
}

template <typename T>
inline const T round_up(const T &a, const T &b)
{
    return (a + b - static_cast<T>(1)) / b * b;
}

const int32_t AB_BITS = 10;
const int32_t AB_SCALE = 1 << AB_BITS;
const int32_t INTER_BITS = 5;
const int32_t INTER_TAB_SIZE = 1 << INTER_BITS;
const int32_t INTER_REMAP_COEF_BITS = 15;
const int32_t INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

template <typename T>
inline T clip(T value, T min_value, T max_value)
{
    return HPC::utils::min(HPC::utils::max(value, min_value), max_value);
}

template <typename T, int32_t cn>
::ppl::common::RetCode warpAffine_nearest(
    T *dst,
    const T *src,
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    const float *M,
    int32_t borderMode,
    float borderValue = 0.0f)
{
    const int32_t BLOCK_SIZE = 32;
    int32_t _map[BLOCK_SIZE * BLOCK_SIZE + 32];
    int32_t *map = alignPtr(_map, 16);

    int32_t round_delta = AB_SCALE >> 1;

    float32x4_t v_m1 = vdupq_n_f32(M[1]);
    float32x4_t v_m2 = vdupq_n_f32(M[2]);
    float32x4_t v_m4 = vdupq_n_f32(M[4]);
    float32x4_t v_m5 = vdupq_n_f32(M[5]);

    int32_t *adelta = (int32_t *)malloc(outWidth * sizeof(int32_t));
    int32_t *bdelta = (int32_t *)malloc(outWidth * sizeof(int32_t));
    for (int32_t x = 0; x < outWidth; ++x) {
        adelta[x] = rint(M[0] * x * AB_SCALE);
        bdelta[x] = rint(M[3] * x * AB_SCALE);
    }

    if (borderMode == BORDER_REPLICATE) {
        int32x4_t v_zero4 = vdupq_n_s32(0);
        int32x4_t max_width = vdupq_n_s32(inWidth - 1);
        int32x4_t max_height = vdupq_n_s32(inHeight - 1);
        int32x4_t v_cn = vdupq_n_s32(cn);
        int32x4_t v_inWidthStride = vdupq_n_s32(inWidthStride);
        int32x4_t v_round_delta = vdupq_n_s32(round_delta);
        float32x4_t v_AB_SCALE = vdupq_n_f32(AB_SCALE);
        for (int32_t i = 0; i < outHeight; i += BLOCK_SIZE) {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, inHeight - i);
            for (int32_t j = 0; j < outWidth; j += BLOCK_SIZE) {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, inWidth - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y) {
                    int32_t *map_row = getRowPtr(&map[0], blockWidth, y);
                    size_t x = 0, dsty = y + i;
                    float32x4_t v_y = vdupq_n_f32(dsty);
                    float32x4_t v_yx = vmlaq_f32(v_m2, v_m1, v_y), v_yy = vmlaq_f32(v_m5, v_m4, v_y);

                    for (; x + 4 <= blockWidth; x += 4) {
                        int32_t dstx = x + j;
                        int32x4_t X0 = vaddq_s32(vcvtq_s32_f32(vmulq_f32(v_yx, v_AB_SCALE)), v_round_delta);
                        int32x4_t Y0 = vaddq_s32(vcvtq_s32_f32(vmulq_f32(v_yy, v_AB_SCALE)), v_round_delta);
                        int32x4_t srcX = vshrq_n_s32(vaddq_s32(X0, vld1q_s32(adelta + dstx)), AB_BITS);
                        int32x4_t srcY = vshrq_n_s32(vaddq_s32(Y0, vld1q_s32(bdelta + dstx)), AB_BITS);
                        srcX = vminq_s32(vmaxq_s32(srcX, v_zero4), max_width);
                        srcY = vminq_s32(vmaxq_s32(srcY, v_zero4), max_height);
                        int32x4_t v_src_index = vmlaq_s32(vmulq_s32(srcY, v_inWidthStride), srcX, v_cn);
                        vst1q_s32(map_row + x, v_src_index);
                    }

                    for (; x < blockWidth; ++x) {
                        int32_t dstx = x + j;
                        int32_t X0 = rint((M[1] * dsty + M[2]) * AB_SCALE) + round_delta;
                        int32_t Y0 = rint((M[4] * dsty + M[5]) * AB_SCALE) + round_delta;
                        int32_t srcX = (X0 + adelta[dstx]) >> AB_BITS;
                        int32_t srcY = (Y0 + bdelta[dstx]) >> AB_BITS;
                        srcX = clip(srcX, 0, inWidth - 1);
                        srcY = clip(srcY, 0, inHeight - 1);
                        map_row[x] = srcY * inWidthStride + srcX * cn;
                    }
                }
                for (size_t y = 0; y < blockHeight; ++y) {
                    const int32_t *map_row = getRowPtr(map, blockWidth, y);
                    float *dst_row = getRowPtr(dst, outWidthStride, i + y) + j * cn;

                    for (size_t x = 0; x < blockWidth; x++) {
                        if (cn == 1)
                            dst_row[x] = src[map_row[x]];
                        else {
                            float32x4_t res = {src[map_row[x]], src[map_row[x] + 1], src[map_row[x] + 2], cn == 4 ? src[map_row[x] + 3] : dst_row[x * cn + 3]};
                            vst1q_f32(dst_row + x * cn, res);
                        }
                    }
                }
            }
        }
    } else if (borderMode == BORDER_CONSTANT) {
        int32x4_t v_nega = vdupq_n_s32(-1);
        int32x4_t max_width = vdupq_n_s32(inWidth - 1);
        int32x4_t max_height = vdupq_n_s32(inHeight - 1);
        int32x4_t v_cn = vdupq_n_s32(cn);
        int32x4_t v_inWidthStride = vdupq_n_s32(inWidthStride);
        int32x4_t v_round_delta = vdupq_n_s32(round_delta);
        float32x4_t v_AB_SCALE = vdupq_n_f32(AB_SCALE);
        for (int32_t i = 0; i < outHeight; i += BLOCK_SIZE) {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, inHeight - i);
            for (int32_t j = 0; j < outWidth; j += BLOCK_SIZE) {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, inWidth - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y) {
                    int32_t *map_row = getRowPtr(&map[0], blockWidth, y);
                    size_t x = 0, dsty = y + i;
                    float32x4_t v_y = vdupq_n_f32(dsty);
                    float32x4_t v_yx = vmlaq_f32(v_m2, v_m1, v_y), v_yy = vmlaq_f32(v_m5, v_m4, v_y);

                    for (; x + 4 <= blockWidth; x += 4) {
                        int32_t dstx = x + j;
                        int32x4_t X0 = vaddq_s32(vcvtq_s32_f32(vmulq_f32(v_yx, v_AB_SCALE)), v_round_delta);
                        int32x4_t Y0 = vaddq_s32(vcvtq_s32_f32(vmulq_f32(v_yy, v_AB_SCALE)), v_round_delta);
                        int32x4_t srcX = vshrq_n_s32(vaddq_s32(X0, vld1q_s32(adelta + dstx)), AB_BITS);
                        int32x4_t srcY = vshrq_n_s32(vaddq_s32(Y0, vld1q_s32(bdelta + dstx)), AB_BITS);
                        uint32x4_t flg0 = vcleq_u32(vreinterpretq_u32_s32(srcX), vreinterpretq_u32_s32(max_width));
                        uint32x4_t flg1 = vcleq_u32(vreinterpretq_u32_s32(srcY), vreinterpretq_u32_s32(max_height));
                        flg0 = vandq_u32(flg0, flg1);
                        int32x4_t v_src_index = vmlaq_s32(vmulq_s32(srcY, v_inWidthStride), srcX, v_cn);
                        v_src_index = vbslq_s32(flg0, v_src_index, v_nega);
                        vst1q_s32(map_row + x, v_src_index);
                    }

                    for (; x < blockWidth; ++x) {
                        int32_t dstx = x + j;
                        int32_t X0 = rint((M[1] * dsty + M[2]) * AB_SCALE) + round_delta;
                        int32_t Y0 = rint((M[4] * dsty + M[5]) * AB_SCALE) + round_delta;
                        int32_t srcX = (X0 + adelta[dstx]) >> AB_BITS;
                        int32_t srcY = (Y0 + bdelta[dstx]) >> AB_BITS;
                        if ((unsigned)(srcX - 0) <= (unsigned)(inWidth - 1 - 0) && (unsigned)(srcY - 0) <= (unsigned)(inHeight - 1 - 0))
                            map_row[x] = srcY * inWidthStride + srcX * cn;
                        else
                            map_row[x] = -1;
                    }
                }
                for (size_t y = 0; y < blockHeight; ++y) {
                    const int32_t *map_row = getRowPtr(map, blockWidth, y);
                    float *dst_row = getRowPtr(dst, outWidthStride, i + y) + j * cn;

                    for (size_t x = 0; x < blockWidth; x++) {
                        if (cn == 1)
                            dst_row[x] = map_row[x] >= 0 ? src[map_row[x]] : borderValue;
                        else {
                            float32x4_t res = vdupq_n_f32(borderValue);
                            if (map_row[x] >= 0) {
                                res = vld1q_f32(src + map_row[x]);
                            }
                            if (cn == 3) {
                                res = vsetq_lane_f32(dst_row[x * cn + 3], res, 3);
                            }
                            vst1q_f32(dst_row + x * cn, res);
                        }
                    }
                }
            }
        }
    } else if (borderMode == BORDER_TRANSPARENT) {
        int32x4_t v_nega = vdupq_n_s32(-1);
        int32x4_t max_width = vdupq_n_s32(inWidth - 1);
        int32x4_t max_height = vdupq_n_s32(inHeight - 1);
        int32x4_t v_cn = vdupq_n_s32(cn);
        int32x4_t v_inWidthStride = vdupq_n_s32(inWidthStride);
        int32x4_t v_round_delta = vdupq_n_s32(round_delta);
        float32x4_t v_AB_SCALE = vdupq_n_f32(AB_SCALE);
        for (int32_t i = 0; i < outHeight; i += BLOCK_SIZE) {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, inHeight - i);
            for (int32_t j = 0; j < outWidth; j += BLOCK_SIZE) {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, inWidth - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y) {
                    int32_t *map_row = getRowPtr(&map[0], blockWidth, y);
                    size_t x = 0, dsty = y + i;
                    float32x4_t v_y = vdupq_n_f32(dsty);
                    float32x4_t v_yx = vmlaq_f32(v_m2, v_m1, v_y), v_yy = vmlaq_f32(v_m5, v_m4, v_y);

                    for (; x + 4 <= blockWidth; x += 4) {
                        int32_t dstx = x + j;
                        int32x4_t X0 = vaddq_s32(vcvtq_s32_f32(vmulq_f32(v_yx, v_AB_SCALE)), v_round_delta);
                        int32x4_t Y0 = vaddq_s32(vcvtq_s32_f32(vmulq_f32(v_yy, v_AB_SCALE)), v_round_delta);
                        int32x4_t srcX = vshrq_n_s32(vaddq_s32(X0, vld1q_s32(adelta + dstx)), AB_BITS);
                        int32x4_t srcY = vshrq_n_s32(vaddq_s32(Y0, vld1q_s32(bdelta + dstx)), AB_BITS);
                        uint32x4_t flg0 = vcleq_u32(vreinterpretq_u32_s32(srcX), vreinterpretq_u32_s32(max_width));
                        uint32x4_t flg1 = vcleq_u32(vreinterpretq_u32_s32(srcY), vreinterpretq_u32_s32(max_height));
                        flg0 = vandq_u32(flg0, flg1);
                        int32x4_t v_src_index = vmlaq_s32(vmulq_s32(srcY, v_inWidthStride), srcX, v_cn);
                        v_src_index = vbslq_s32(flg0, v_src_index, v_nega);
                        vst1q_s32(map_row + x, v_src_index);
                    }

                    for (; x < blockWidth; ++x) {
                        int32_t dstx = x + j;
                        int32_t X0 = rint((M[1] * dsty + M[2]) * AB_SCALE) + round_delta;
                        int32_t Y0 = rint((M[4] * dsty + M[5]) * AB_SCALE) + round_delta;
                        int32_t srcX = (X0 + adelta[dstx]) >> AB_BITS;
                        int32_t srcY = (Y0 + bdelta[dstx]) >> AB_BITS;
                        if ((unsigned)(srcX - 0) <= (unsigned)(inWidth - 1 - 0) && (unsigned)(srcY - 0) <= (unsigned)(inHeight - 1 - 0))
                            map_row[x] = srcY * inWidthStride + srcX * cn;
                        else
                            map_row[x] = -1;
                    }
                }
                for (size_t y = 0; y < blockHeight; ++y) {
                    const int32_t *map_row = getRowPtr(map, blockWidth, y);
                    float *dst_row = getRowPtr(dst, outWidthStride, i + y) + j * cn;

                    for (size_t x = 0; x < blockWidth; x++) {
                        if (map_row[x] < 0)
                            continue;
                        if (cn == 1)
                            dst_row[x] = src[map_row[x]];
                        else {
                            float32x4_t res = vld1q_f32(src + map_row[x]);
                            if (cn == 3) {
                                res = vsetq_lane_f32(dst_row[x * cn + 3], res, 3);
                            }
                            vst1q_f32(dst_row + x * cn, res);
                        }
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

static void initTab_linear_short(float *short_tab)
{
    float scale = 1.f / INTER_TAB_SIZE;
    for (int32_t i = 0; i < INTER_TAB_SIZE; ++i) {
        float vy = i * scale;
        for (int32_t j = 0; j < INTER_TAB_SIZE; ++j, short_tab += 4) {
            float vx = j * scale;
            short_tab[0] = static_cast<float>(HPC::utils::saturate_cast<uint16_t>((1 - vy) * (1 - vx) * INTER_REMAP_COEF_SCALE));
            short_tab[1] = static_cast<float>(HPC::utils::saturate_cast<uint16_t>((1 - vy) * vx * INTER_REMAP_COEF_SCALE));
            short_tab[2] = static_cast<float>(HPC::utils::saturate_cast<uint16_t>(vy * (1 - vx) * INTER_REMAP_COEF_SCALE));
            short_tab[3] = static_cast<float>(HPC::utils::saturate_cast<uint16_t>(vy * vx * INTER_REMAP_COEF_SCALE));
        }
    }
}

::ppl::common::RetCode warpAffine_linear_float(
    float *dst,
    const float *src,
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    const float *M,
    int32_t cn,
    int32_t borderMode,
    float borderValue = 0.0f)
{
    float *_short_tab = (float *)malloc(INTER_TAB_SIZE * INTER_TAB_SIZE * 8 * sizeof(float) + 16);
    float *short_tab = alignPtr(_short_tab, 16);
    initTab_linear_short(short_tab);

    int32_t round_delta = AB_SCALE / INTER_TAB_SIZE / 2;

    int32_t *adelta = (int32_t *)malloc(outWidth * sizeof(int32_t));
    int32_t *bdelta = (int32_t *)malloc(outWidth * sizeof(int32_t));
    for (int32_t i = 0; i < outWidth; ++i) {
        adelta[i] = rint(M[0] * i * AB_SCALE);
        bdelta[i] = rint(M[3] * i * AB_SCALE);
    }
    const int32_t BLOCK_SZ = 64;
    int32_t _XY_INT[BLOCK_SZ * BLOCK_SZ * 4 + 16];
    int32_t *XY_INT = alignPtr(_XY_INT, 16);
    int16_t _XY_DEC[BLOCK_SZ * BLOCK_SZ + 16];
    int16_t *XY_DEC = alignPtr(_XY_DEC, 16);
    int32_t bh0 = std::min<int32_t>(BLOCK_SZ / 2, outHeight);
    int32_t bw0 = std::min<int32_t>(BLOCK_SZ * BLOCK_SZ / bh0, outWidth);
    bh0 = std::min<int32_t>(BLOCK_SZ * BLOCK_SZ / bw0, outHeight);
    int32x4_t v_inWidthStride = vdupq_n_s32(inWidthStride);
    int32x4_t v_cn = vdupq_n_s32(cn);

    int32x4_t dec_mask = vdupq_n_s32((INTER_TAB_SIZE - 1) << (AB_BITS - INTER_BITS));
    int32x4_t v_nega = vdupq_n_s32(-1);
    int32x4_t max_width = vdupq_n_s32(inWidth - 1);
    int32x4_t max_height = vdupq_n_s32(inHeight - 1);
    if (borderMode != BORDER_REPLICATE) {
        for (int32_t y = 0; y < outHeight; y += bh0) {
            int32_t bh = std::min<int32_t>(bh0, outHeight - y);
            for (int32_t x = 0; x < outWidth; x += bw0) {
                int32_t bw = std::min<int32_t>(bw0, outWidth - x);
                for (int32_t y1 = 0; y1 < bh; ++y1) {
                    int32_t *xy_int_p = XY_INT + y1 * bw * 4;
                    int16_t *xy_dec_p = XY_DEC + y1 * bw;
                    int32_t x_int = (int32_t)((M[1] * (y + y1) + M[2]) * AB_SCALE) + round_delta;
                    int32_t y_int = (int32_t)((M[4] * (y + y1) + M[5]) * AB_SCALE) + round_delta;

                    int32x4_t m_X_int = vdupq_n_s32(x_int);
                    int32x4_t m_Y_int = vdupq_n_s32(y_int);
                    int32_t x1 = 0;
                    for (; x1 <= bw - 8; x1 += 8) {
                        int32x4_t tx0, tx1, ty0, ty1;
                        tx0 = vaddq_s32(m_X_int, vld1q_s32(adelta + x + x1));
                        tx1 = vaddq_s32(m_X_int, vld1q_s32(adelta + x + x1 + 4));
                        ty0 = vaddq_s32(m_Y_int, vld1q_s32(bdelta + x + x1));
                        ty1 = vaddq_s32(m_Y_int, vld1q_s32(bdelta + x + x1 + 4));

                        int16x8_t fx, fy;
                        fx = vcombine_s16(
                            vqmovn_s32(vandq_s32(tx0, dec_mask)),
                            vqmovn_s32(vandq_s32(tx1, dec_mask)));
                        fy = vcombine_s16(
                            vqmovn_s32(vandq_s32(ty0, dec_mask)),
                            vqmovn_s32(vandq_s32(ty1, dec_mask)));
                        int16x8_t final_f = vshlq_n_s16(vaddq_s16(vshrq_n_s16(fx, AB_BITS - INTER_BITS), fy), 2);
                        vst1q_s16(xy_dec_p + x1, final_f);

                        tx0 = vshrq_n_s32(tx0, AB_BITS);
                        tx1 = vshrq_n_s32(tx1, AB_BITS);
                        ty0 = vshrq_n_s32(ty0, AB_BITS);
                        ty1 = vshrq_n_s32(ty1, AB_BITS);

                        uint32x4_t vx0_0 = vandq_u32(vcgezq_s32(tx0), vcleq_s32(tx0, max_width));
                        uint32x4_t vx0_1 = vandq_u32(vcgeq_s32(tx0, v_nega), vcltq_s32(tx0, max_width));
                        uint32x4_t vy0_0 = vandq_u32(vcgezq_s32(ty0), vcleq_s32(ty0, max_height));
                        uint32x4_t vy0_1 = vandq_u32(vcgeq_s32(ty0, v_nega), vcltq_s32(ty0, max_height));

                        int32x4x4_t dst0;
                        dst0.val[0] = vmlaq_s32(vmulq_s32(ty0, v_inWidthStride), tx0, v_cn);
                        dst0.val[1] = vaddq_s32(dst0.val[0], v_cn);
                        dst0.val[2] = vaddq_s32(dst0.val[0], v_inWidthStride);
                        dst0.val[3] = vaddq_s32(dst0.val[2], v_cn);

                        dst0.val[0] = vbslq_s32(vandq_u32(vx0_0, vy0_0), dst0.val[0], v_nega);
                        dst0.val[1] = vbslq_s32(vandq_u32(vx0_1, vy0_0), dst0.val[1], v_nega);
                        dst0.val[2] = vbslq_s32(vandq_u32(vx0_0, vy0_1), dst0.val[2], v_nega);
                        dst0.val[3] = vbslq_s32(vandq_u32(vx0_1, vy0_1), dst0.val[3], v_nega);

                        vst4q_s32(xy_int_p + x1 * 4, dst0);

                        int32x4x4_t dst1;
                        uint32x4_t vx1_0 = vandq_u32(vcgezq_s32(tx1), vcleq_s32(tx1, max_width));
                        uint32x4_t vx1_1 = vandq_u32(vcgeq_s32(tx1, v_nega), vcltq_s32(tx1, max_width));
                        uint32x4_t vy1_0 = vandq_u32(vcgezq_s32(ty1), vcleq_s32(ty1, max_height));
                        uint32x4_t vy1_1 = vandq_u32(vcgeq_s32(ty1, v_nega), vcltq_s32(ty1, max_height));

                        dst1.val[0] = vmlaq_s32(vmulq_s32(ty1, v_inWidthStride), tx1, v_cn);
                        dst1.val[1] = vaddq_s32(dst1.val[0], v_cn);
                        dst1.val[2] = vaddq_s32(dst1.val[0], v_inWidthStride);
                        dst1.val[3] = vaddq_s32(dst1.val[2], v_cn);

                        dst1.val[0] = vbslq_s32(vandq_u32(vx1_0, vy1_0), dst1.val[0], v_nega);
                        dst1.val[1] = vbslq_s32(vandq_u32(vx1_1, vy1_0), dst1.val[1], v_nega);
                        dst1.val[2] = vbslq_s32(vandq_u32(vx1_0, vy1_1), dst1.val[2], v_nega);
                        dst1.val[3] = vbslq_s32(vandq_u32(vx1_1, vy1_1), dst1.val[3], v_nega);
                        vst4q_s32(xy_int_p + (x1 + 4) * 4, dst1);
                    }
                    for (; x1 < bw; ++x1) {
                        int32_t x_value = (x_int + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
                        int32_t y_value = (y_int + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
                        int16_t src_x = HPC::utils::saturate_cast<uint16_t>(x_value >> INTER_BITS);
                        int16_t src_y = HPC::utils::saturate_cast<uint16_t>(y_value >> INTER_BITS);
                        bool flag[4];
                        flag[0] = (src_x >= 0 && src_x < inWidth && src_y >= 0 && src_y < inHeight);
                        flag[1] = (src_x + 1 >= 0 && src_x + 1 < inWidth && src_y >= 0 && src_y < inHeight);
                        flag[2] = (src_x >= 0 && src_x < inWidth && src_y + 1 >= 0 && src_y + 1 < inHeight);
                        flag[3] = (src_x + 1 >= 0 && src_x + 1 < inWidth && src_y + 1 >= 0 && src_y + 1 < inHeight);
                        xy_int_p[x1 * 4] = flag[0] ? src_y * inWidthStride + src_x * cn : -1;
                        xy_int_p[x1 * 4 + 1] = flag[1] ? src_y * inWidthStride + (src_x + 1) * cn : -1;
                        xy_int_p[x1 * 4 + 2] = flag[2] ? (src_y + 1) * inWidthStride + src_x * cn : -1;
                        xy_int_p[x1 * 4 + 3] = flag[3] ? (src_y + 1) * inWidthStride + (src_x + 1) * cn : -1;
                        xy_dec_p[x1] = ((y_value & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE +
                                        (x_value & (INTER_TAB_SIZE - 1)));
                    }
                }
                float32x4_t v_border = vdupq_n_f32(borderValue);
                for (int32_t y1 = 0; y1 < bh; ++y1) {
                    float32x4_t v_scale = vdupq_n_f32(INTER_REMAP_COEF_SCALE);
                    int32_t dstY = y1 + y;
                    int32_t *xy_int_p = XY_INT + y1 * bw * 4;
                    int16_t *xy_dec_p = XY_DEC + y1 * bw;
                    for (int32_t x1 = 0; x1 < bw; x1 += 4) {
                        int32_t dstX = x1 + x;
                        int32_t dstIndex = dstY * outWidthStride + dstX * cn;

                        if (borderMode == BORDER_CONSTANT) {
                            float32x4_t p_s_t0 = vld1q_f32(short_tab + xy_dec_p[x1]);
                            float32x4_t p_s_t1 = vld1q_f32(short_tab + xy_dec_p[x1 + 1]);
                            float32x4_t p_s_t2 = vld1q_f32(short_tab + xy_dec_p[x1 + 2]);
                            float32x4_t p_s_t3 = vld1q_f32(short_tab + xy_dec_p[x1 + 3]);
                            if (cn == 1) {
                                float32x4_t src_value0 = vcombine_f32(vld1_f32(src + xy_int_p[x1 << 2]), vld1_f32(src + xy_int_p[(x1 << 2) + 2]));
                                src_value0 = vbslq_f32(vcgezq_s32(vld1q_s32(xy_int_p + (x1 << 2))), src_value0, v_border);
                                float32x4_t src_value1 = vcombine_f32(vld1_f32(src + xy_int_p[(x1 + 1) << 2]), vld1_f32(src + xy_int_p[((x1 + 1) << 2) + 2]));
                                src_value1 = vbslq_f32(vcgezq_s32(vld1q_s32(xy_int_p + ((x1 + 1) << 2))), src_value1, v_border);
                                float32x4_t src_value2 = vcombine_f32(vld1_f32(src + xy_int_p[(x1 + 2) << 2]), vld1_f32(src + xy_int_p[((x1 + 2) << 2) + 2]));
                                src_value2 = vbslq_f32(vcgezq_s32(vld1q_s32(xy_int_p + ((x1 + 2) << 2))), src_value2, v_border);
                                float32x4_t src_value3 = vcombine_f32(vld1_f32(src + xy_int_p[(x1 + 3) << 2]), vld1_f32(src + xy_int_p[((x1 + 3) << 2) + 2]));
                                src_value3 = vbslq_f32(vcgezq_s32(vld1q_s32(xy_int_p + ((x1 + 3) << 2))), src_value3, v_border);

                                src_value0 = vmulq_f32(p_s_t0, src_value0);
                                src_value1 = vmulq_f32(p_s_t1, src_value1);
                                src_value2 = vmulq_f32(p_s_t2, src_value2);
                                src_value3 = vmulq_f32(p_s_t3, src_value3);
                                float32x4_t res = {vaddvq_f32(src_value0), vaddvq_f32(src_value1), vaddvq_f32(src_value2), vaddvq_f32(src_value3)};
                                vst1q_f32(dst + dstIndex, vdivq_f32(res, v_scale));
                            } else if (cn == 4) {
                                float32x4x4_t src_value0 = {
                                    xy_int_p[x1 * 4] >= 0 ? vld1q_f32(src + xy_int_p[x1 << 2]) : v_border,
                                    xy_int_p[x1 * 4 + 1] >= 0 ? vld1q_f32(src + xy_int_p[(x1 << 2) + 1]) : v_border,
                                    xy_int_p[x1 * 4 + 2] >= 0 ? vld1q_f32(src + xy_int_p[(x1 << 2) + 2]) : v_border,
                                    xy_int_p[x1 * 4 + 3] >= 0 ? vld1q_f32(src + xy_int_p[(x1 << 2) + 3]) : v_border,
                                };
                                float32x4x4_t src_value1 = {
                                    xy_int_p[(x1 + 1) * 4] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 1) << 2)]) : v_border,
                                    xy_int_p[(x1 + 1) * 4 + 1] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 1]) : v_border,
                                    xy_int_p[(x1 + 1) * 4 + 2] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 2]) : v_border,
                                    xy_int_p[(x1 + 1) * 4 + 3] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 3]) : v_border,
                                };
                                float32x4x4_t src_value2 = {
                                    xy_int_p[(x1 + 2) * 4] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 2) << 2)]) : v_border,
                                    xy_int_p[(x1 + 2) * 4 + 1] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 1]) : v_border,
                                    xy_int_p[(x1 + 2) * 4 + 2] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 2]) : v_border,
                                    xy_int_p[(x1 + 2) * 4 + 3] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 3]) : v_border,
                                };
                                float32x4x4_t src_value3 = {
                                    xy_int_p[(x1 + 3) * 4] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 3) << 2)]) : v_border,
                                    xy_int_p[(x1 + 3) * 4 + 1] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 1]) : v_border,
                                    xy_int_p[(x1 + 3) * 4 + 2] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 2]) : v_border,
                                    xy_int_p[(x1 + 3) * 4 + 3] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 3]) : v_border,
                                };
                                src_value0.val[0] = vmulq_laneq_f32(src_value0.val[0], p_s_t0, 0);
                                src_value0.val[1] = vmulq_laneq_f32(src_value0.val[1], p_s_t0, 1);
                                src_value0.val[2] = vmulq_laneq_f32(src_value0.val[2], p_s_t0, 2);
                                src_value0.val[3] = vmulq_laneq_f32(src_value0.val[3], p_s_t0, 3);

                                src_value1.val[0] = vmulq_laneq_f32(src_value1.val[0], p_s_t1, 0);
                                src_value1.val[1] = vmulq_laneq_f32(src_value1.val[1], p_s_t1, 1);
                                src_value1.val[2] = vmulq_laneq_f32(src_value1.val[2], p_s_t1, 2);
                                src_value1.val[3] = vmulq_laneq_f32(src_value1.val[3], p_s_t1, 3);

                                src_value2.val[0] = vmulq_laneq_f32(src_value2.val[0], p_s_t2, 0);
                                src_value2.val[1] = vmulq_laneq_f32(src_value2.val[1], p_s_t2, 1);
                                src_value2.val[2] = vmulq_laneq_f32(src_value2.val[2], p_s_t2, 2);
                                src_value2.val[3] = vmulq_laneq_f32(src_value2.val[3], p_s_t2, 3);

                                src_value3.val[0] = vmulq_laneq_f32(src_value3.val[0], p_s_t3, 0);
                                src_value3.val[1] = vmulq_laneq_f32(src_value3.val[1], p_s_t3, 1);
                                src_value3.val[2] = vmulq_laneq_f32(src_value3.val[2], p_s_t3, 2);
                                src_value3.val[3] = vmulq_laneq_f32(src_value3.val[3], p_s_t3, 3);

                                vst1q_f32(dst + dstIndex, vdivq_f32(vaddq_f32(vaddq_f32(src_value0.val[0], src_value0.val[1]), vaddq_f32(src_value0.val[2], src_value0.val[3])), v_scale));
                                vst1q_f32(dst + dstIndex + cn, vdivq_f32(vaddq_f32(vaddq_f32(src_value1.val[0], src_value1.val[1]), vaddq_f32(src_value1.val[2], src_value1.val[3])), v_scale));
                                vst1q_f32(dst + dstIndex + cn * 2, vdivq_f32(vaddq_f32(vaddq_f32(src_value2.val[0], src_value2.val[1]), vaddq_f32(src_value2.val[2], src_value2.val[3])), v_scale));
                                vst1q_f32(dst + dstIndex + cn * 3, vdivq_f32(vaddq_f32(vaddq_f32(src_value3.val[0], src_value3.val[1]), vaddq_f32(src_value3.val[2], src_value3.val[3])), v_scale));
                            } else {
                                float32x4x4_t src_value0 = {
                                    xy_int_p[x1 * 4] >= 0 ? vld1q_f32(src + xy_int_p[x1 << 2]) : v_border,
                                    xy_int_p[x1 * 4 + 1] >= 0 ? vld1q_f32(src + xy_int_p[(x1 << 2) + 1]) : v_border,
                                    xy_int_p[x1 * 4 + 2] >= 0 ? vld1q_f32(src + xy_int_p[(x1 << 2) + 2]) : v_border,
                                    xy_int_p[x1 * 4 + 3] >= 0 ? vld1q_f32(src + xy_int_p[(x1 << 2) + 3]) : v_border,
                                };
                                float32x4x4_t src_value1 = {
                                    xy_int_p[(x1 + 1) * 4] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 1) << 2)]) : v_border,
                                    xy_int_p[(x1 + 1) * 4 + 1] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 1]) : v_border,
                                    xy_int_p[(x1 + 1) * 4 + 2] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 2]) : v_border,
                                    xy_int_p[(x1 + 1) * 4 + 3] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 3]) : v_border,
                                };
                                float32x4x4_t src_value2 = {
                                    xy_int_p[(x1 + 2) * 4] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 2) << 2)]) : v_border,
                                    xy_int_p[(x1 + 2) * 4 + 1] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 1]) : v_border,
                                    xy_int_p[(x1 + 2) * 4 + 2] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 2]) : v_border,
                                    xy_int_p[(x1 + 2) * 4 + 3] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 3]) : v_border,
                                };
                                float32x4x4_t src_value3 = {
                                    xy_int_p[(x1 + 3) * 4] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 3) << 2)]) : v_border,
                                    xy_int_p[(x1 + 3) * 4 + 1] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 1]) : v_border,
                                    xy_int_p[(x1 + 3) * 4 + 2] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 2]) : v_border,
                                    xy_int_p[(x1 + 3) * 4 + 3] >= 0 ? vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 3]) : v_border,
                                };
                                src_value0.val[0] = vmulq_laneq_f32(src_value0.val[0], p_s_t0, 0);
                                src_value0.val[1] = vmulq_laneq_f32(src_value0.val[1], p_s_t0, 1);
                                src_value0.val[2] = vmulq_laneq_f32(src_value0.val[2], p_s_t0, 2);
                                src_value0.val[3] = vmulq_laneq_f32(src_value0.val[3], p_s_t0, 3);

                                src_value1.val[0] = vmulq_laneq_f32(src_value1.val[0], p_s_t1, 0);
                                src_value1.val[1] = vmulq_laneq_f32(src_value1.val[1], p_s_t1, 1);
                                src_value1.val[2] = vmulq_laneq_f32(src_value1.val[2], p_s_t1, 2);
                                src_value1.val[3] = vmulq_laneq_f32(src_value1.val[3], p_s_t1, 3);

                                src_value2.val[0] = vmulq_laneq_f32(src_value2.val[0], p_s_t2, 0);
                                src_value2.val[1] = vmulq_laneq_f32(src_value2.val[1], p_s_t2, 1);
                                src_value2.val[2] = vmulq_laneq_f32(src_value2.val[2], p_s_t2, 2);
                                src_value2.val[3] = vmulq_laneq_f32(src_value2.val[3], p_s_t2, 3);

                                src_value3.val[0] = vmulq_laneq_f32(src_value3.val[0], p_s_t3, 0);
                                src_value3.val[1] = vmulq_laneq_f32(src_value3.val[1], p_s_t3, 1);
                                src_value3.val[2] = vmulq_laneq_f32(src_value3.val[2], p_s_t3, 2);
                                src_value3.val[3] = vmulq_laneq_f32(src_value3.val[3], p_s_t3, 3);

                                float32x4_t res0 = vdivq_f32(vaddq_f32(vaddq_f32(src_value0.val[0], src_value0.val[1]), vaddq_f32(src_value0.val[2], src_value0.val[3])), v_scale);
                                float32x4_t res1 = vdivq_f32(vaddq_f32(vaddq_f32(src_value1.val[0], src_value1.val[1]), vaddq_f32(src_value1.val[2], src_value1.val[3])), v_scale);
                                float32x4_t res2 = vdivq_f32(vaddq_f32(vaddq_f32(src_value2.val[0], src_value2.val[1]), vaddq_f32(src_value2.val[2], src_value2.val[3])), v_scale);
                                float32x4_t res3 = vdivq_f32(vaddq_f32(vaddq_f32(src_value3.val[0], src_value3.val[1]), vaddq_f32(src_value3.val[2], src_value3.val[3])), v_scale);

                                res0 = vcopyq_laneq_f32(res0, 3, res1, 0);
                                res1 = vcombine_f32(vget_low_f32(vextq_f32(res1, res2, 1)), vget_low_f32(res2));
                                res2 = vextq_f32(vdupq_n_f32(vgetq_lane_f32(res2, 2)), res3, 3);
                                vst1q_f32(dst + dstIndex, res0);
                                vst1q_f32(dst + dstIndex + 4, res1);
                                vst1q_f32(dst + dstIndex + 8, res2);
                            }
                        } else if (borderMode == BORDER_TRANSPARENT) {
                            float32x4_t p_s_t0 = vld1q_f32(short_tab + xy_dec_p[x1]);
                            float32x4_t p_s_t1 = vld1q_f32(short_tab + xy_dec_p[x1 + 1]);
                            float32x4_t p_s_t2 = vld1q_f32(short_tab + xy_dec_p[x1 + 2]);
                            float32x4_t p_s_t3 = vld1q_f32(short_tab + xy_dec_p[x1 + 3]);
                            if (cn == 1) {
                                uint32x4_t flag = {vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + (x1 << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 1) << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 2) << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 3) << 2))))};
                                if (vgetq_lane_u32(flag, 0) == 0) {
                                    float32x4_t src_value0 = vcombine_f32(vld1_f32(src + xy_int_p[x1 << 2]), vld1_f32(src + xy_int_p[(x1 << 2) + 2]));
                                    src_value0 = vmulq_f32(p_s_t0, src_value0);
                                    dst[dstIndex] = vaddvq_f32(src_value0) / INTER_REMAP_COEF_SCALE;
                                }
                                if (vgetq_lane_u32(flag, 1) == 0) {
                                    float32x4_t src_value1 = vcombine_f32(vld1_f32(src + xy_int_p[(x1 + 1) << 2]), vld1_f32(src + xy_int_p[((x1 + 1) << 2) + 2]));
                                    src_value1 = vmulq_f32(p_s_t1, src_value1);
                                    dst[dstIndex + 1] = vaddvq_f32(src_value1) / INTER_REMAP_COEF_SCALE;
                                }
                                if (vgetq_lane_u32(flag, 2) == 0) {
                                    float32x4_t src_value2 = vcombine_f32(vld1_f32(src + xy_int_p[(x1 + 2) << 2]), vld1_f32(src + xy_int_p[((x1 + 2) << 2) + 2]));
                                    src_value2 = vmulq_f32(p_s_t2, src_value2);
                                    dst[dstIndex + 2] = vaddvq_f32(src_value2) / INTER_REMAP_COEF_SCALE;
                                }
                                if (vgetq_lane_u32(flag, 3) == 0) {
                                    float32x4_t src_value3 = vcombine_f32(vld1_f32(src + xy_int_p[(x1 + 3) << 2]), vld1_f32(src + xy_int_p[((x1 + 3) << 2) + 2]));
                                    src_value3 = vmulq_f32(p_s_t3, src_value3);
                                    dst[dstIndex + 3] = vaddvq_f32(src_value3) / INTER_REMAP_COEF_SCALE;
                                }
                            } else if (cn == 4) {
                                uint32x4_t flag = {vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + (x1 << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 1) << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 2) << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 3) << 2))))};
                                if (vgetq_lane_u32(flag, 0) == 0) {
                                    float32x4x4_t src_value0 = {
                                        vld1q_f32(src + xy_int_p[x1 << 2]),
                                        vld1q_f32(src + xy_int_p[(x1 << 2) + 1]),
                                        vld1q_f32(src + xy_int_p[(x1 << 2) + 2]),
                                        vld1q_f32(src + xy_int_p[(x1 << 2) + 3]),
                                    };
                                    src_value0.val[0] = vmulq_laneq_f32(src_value0.val[0], p_s_t0, 0);
                                    src_value0.val[1] = vmulq_laneq_f32(src_value0.val[1], p_s_t0, 1);
                                    src_value0.val[2] = vmulq_laneq_f32(src_value0.val[2], p_s_t0, 2);
                                    src_value0.val[3] = vmulq_laneq_f32(src_value0.val[3], p_s_t0, 3);
                                    vst1q_f32(dst + dstIndex, vdivq_f32(vaddq_f32(vaddq_f32(src_value0.val[0], src_value0.val[1]), vaddq_f32(src_value0.val[2], src_value0.val[3])), v_scale));
                                }
                                if (vgetq_lane_u32(flag, 1) == 0) {
                                    float32x4x4_t src_value1 = {
                                        vld1q_f32(src + xy_int_p[((x1 + 1) << 2)]),
                                        vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 1]),
                                        vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 2]),
                                        vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 3]),
                                    };

                                    src_value1.val[0] = vmulq_laneq_f32(src_value1.val[0], p_s_t1, 0);
                                    src_value1.val[1] = vmulq_laneq_f32(src_value1.val[1], p_s_t1, 1);
                                    src_value1.val[2] = vmulq_laneq_f32(src_value1.val[2], p_s_t1, 2);
                                    src_value1.val[3] = vmulq_laneq_f32(src_value1.val[3], p_s_t1, 3);
                                    vst1q_f32(dst + dstIndex + cn, vdivq_f32(vaddq_f32(vaddq_f32(src_value1.val[0], src_value1.val[1]), vaddq_f32(src_value1.val[2], src_value1.val[3])), v_scale));
                                }
                                if (vgetq_lane_u32(flag, 2) == 0) {
                                    float32x4x4_t src_value2 = {
                                        vld1q_f32(src + xy_int_p[((x1 + 2) << 2)]),
                                        vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 1]),
                                        vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 2]),
                                        vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 3]),
                                    };

                                    src_value2.val[0] = vmulq_laneq_f32(src_value2.val[0], p_s_t2, 0);
                                    src_value2.val[1] = vmulq_laneq_f32(src_value2.val[1], p_s_t2, 1);
                                    src_value2.val[2] = vmulq_laneq_f32(src_value2.val[2], p_s_t2, 2);
                                    src_value2.val[3] = vmulq_laneq_f32(src_value2.val[3], p_s_t2, 3);
                                    vst1q_f32(dst + dstIndex + cn * 2, vdivq_f32(vaddq_f32(vaddq_f32(src_value2.val[0], src_value2.val[1]), vaddq_f32(src_value2.val[2], src_value2.val[3])), v_scale));
                                }
                                if (vgetq_lane_u32(flag, 3) == 0) {
                                    float32x4x4_t src_value3 = {
                                        vld1q_f32(src + xy_int_p[((x1 + 3) << 2)]),
                                        vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 1]),
                                        vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 2]),
                                        vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 3]),
                                    };
                                    src_value3.val[0] = vmulq_laneq_f32(src_value3.val[0], p_s_t3, 0);
                                    src_value3.val[1] = vmulq_laneq_f32(src_value3.val[1], p_s_t3, 1);
                                    src_value3.val[2] = vmulq_laneq_f32(src_value3.val[2], p_s_t3, 2);
                                    src_value3.val[3] = vmulq_laneq_f32(src_value3.val[3], p_s_t3, 3);
                                    vst1q_f32(dst + dstIndex + cn * 3, vdivq_f32(vaddq_f32(vaddq_f32(src_value3.val[0], src_value3.val[1]), vaddq_f32(src_value3.val[2], src_value3.val[3])), v_scale));
                                }

                            } else {
                                uint32x4_t flag = {vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + (x1 << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 1) << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 2) << 2)))),
                                                   vaddvq_u32(vcltzq_s32(vld1q_s32(xy_int_p + ((x1 + 3) << 2))))};
                                if (vgetq_lane_u32(flag, 0) == 0) {
                                    float32x4x4_t src_value0 = {
                                        vld1q_f32(src + xy_int_p[x1 << 2]),
                                        vld1q_f32(src + xy_int_p[(x1 << 2) + 1]),
                                        vld1q_f32(src + xy_int_p[(x1 << 2) + 2]),
                                        vld1q_f32(src + xy_int_p[(x1 << 2) + 3])};
                                    src_value0.val[0] = vmulq_laneq_f32(src_value0.val[0], p_s_t0, 0);
                                    src_value0.val[1] = vmulq_laneq_f32(src_value0.val[1], p_s_t0, 1);
                                    src_value0.val[2] = vmulq_laneq_f32(src_value0.val[2], p_s_t0, 2);
                                    src_value0.val[3] = vmulq_laneq_f32(src_value0.val[3], p_s_t0, 3);
                                    float32x4_t res0 = vdivq_f32(vaddq_f32(vaddq_f32(src_value0.val[0], src_value0.val[1]), vaddq_f32(src_value0.val[2], src_value0.val[3])), v_scale);
                                    vst1_f32(dst + dstIndex, vget_low_f32(res0));
                                    dst[dstIndex + 2] = vgetq_lane_f32(res0, 2);
                                }
                                if (vgetq_lane_u32(flag, 1) == 0) {
                                    float32x4x4_t src_value1 = {
                                        vld1q_f32(src + xy_int_p[((x1 + 1) << 2)]),
                                        vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 1]),
                                        vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 2]),
                                        vld1q_f32(src + xy_int_p[((x1 + 1) << 2) + 3])};
                                    src_value1.val[0] = vmulq_laneq_f32(src_value1.val[0], p_s_t1, 0);
                                    src_value1.val[1] = vmulq_laneq_f32(src_value1.val[1], p_s_t1, 1);
                                    src_value1.val[2] = vmulq_laneq_f32(src_value1.val[2], p_s_t1, 2);
                                    src_value1.val[3] = vmulq_laneq_f32(src_value1.val[3], p_s_t1, 3);
                                    float32x4_t res1 = vdivq_f32(vaddq_f32(vaddq_f32(src_value1.val[0], src_value1.val[1]), vaddq_f32(src_value1.val[2], src_value1.val[3])), v_scale);
                                    vst1_f32(dst + dstIndex + cn, vget_low_f32(res1));
                                    dst[dstIndex + cn + 2] = vgetq_lane_f32(res1, 2);
                                }
                                if (vgetq_lane_u32(flag, 2) == 0) {
                                    float32x4x4_t src_value2 = {
                                        vld1q_f32(src + xy_int_p[((x1 + 2) << 2)]),
                                        vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 1]),
                                        vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 2]),
                                        vld1q_f32(src + xy_int_p[((x1 + 2) << 2) + 3])};

                                    src_value2.val[0] = vmulq_laneq_f32(src_value2.val[0], p_s_t2, 0);
                                    src_value2.val[1] = vmulq_laneq_f32(src_value2.val[1], p_s_t2, 1);
                                    src_value2.val[2] = vmulq_laneq_f32(src_value2.val[2], p_s_t2, 2);
                                    src_value2.val[3] = vmulq_laneq_f32(src_value2.val[3], p_s_t2, 3);
                                    float32x4_t res2 = vdivq_f32(vaddq_f32(vaddq_f32(src_value2.val[0], src_value2.val[1]), vaddq_f32(src_value2.val[2], src_value2.val[3])), v_scale);
                                    vst1_f32(dst + dstIndex + cn * 2, vget_low_f32(res2));
                                    dst[dstIndex + cn * 2 + 2] = vgetq_lane_f32(res2, 2);
                                }
                                if (vgetq_lane_u32(flag, 3) == 0) {
                                    float32x4x4_t src_value3 = {
                                        vld1q_f32(src + xy_int_p[((x1 + 3) << 2)]),
                                        vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 1]),
                                        vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 2]),
                                        vld1q_f32(src + xy_int_p[((x1 + 3) << 2) + 3])};

                                    src_value3.val[0] = vmulq_laneq_f32(src_value3.val[0], p_s_t3, 0);
                                    src_value3.val[1] = vmulq_laneq_f32(src_value3.val[1], p_s_t3, 1);
                                    src_value3.val[2] = vmulq_laneq_f32(src_value3.val[2], p_s_t3, 2);
                                    src_value3.val[3] = vmulq_laneq_f32(src_value3.val[3], p_s_t3, 3);
                                    float32x4_t res3 = vdivq_f32(vaddq_f32(vaddq_f32(src_value3.val[0], src_value3.val[1]), vaddq_f32(src_value3.val[2], src_value3.val[3])), v_scale);
                                    vst1_f32(dst + dstIndex + cn * 3, vget_low_f32(res3));
                                    dst[dstIndex + cn * 3 + 2] = vgetq_lane_f32(res3, 2);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        int32x4_t v_zero4 = vdupq_n_s32(0);
        for (int32_t y = 0; y < outHeight; y += bh0) {
            int32_t bh = std::min<int32_t>(bh0, outHeight - y);
            for (int32_t x = 0; x < outWidth; x += bw0) {
                int32_t bw = std::min<int32_t>(bw0, outWidth - x);
                for (int32_t y1 = 0; y1 < bh; ++y1) {
                    int32_t *xy_int_p = XY_INT + y1 * bw * 4;
                    int16_t *xy_dec_p = XY_DEC + y1 * bw;
                    int32_t x_int = (int32_t)((M[1] * (y + y1) + M[2]) * AB_SCALE) + round_delta;
                    int32_t y_int = (int32_t)((M[4] * (y + y1) + M[5]) * AB_SCALE) + round_delta;
                    int32x4_t m_X_int = vdupq_n_s32(x_int);
                    int32x4_t m_Y_int = vdupq_n_s32(y_int);
                    int32_t x1 = 0;
                    for (; x1 <= bw - 8; x1 += 8) {
                        int32x4_t tx0, tx1, ty0, ty1;
                        tx0 = vaddq_s32(m_X_int, vld1q_s32(adelta + x + x1));
                        tx1 = vaddq_s32(m_X_int, vld1q_s32(adelta + x + x1 + 4));
                        ty0 = vaddq_s32(m_Y_int, vld1q_s32(bdelta + x + x1));
                        ty1 = vaddq_s32(m_Y_int, vld1q_s32(bdelta + x + x1 + 4));

                        int16x8_t fx, fy;
                        fx = vcombine_s16(
                            vqmovn_s32(vandq_s32(tx0, dec_mask)),
                            vqmovn_s32(vandq_s32(tx1, dec_mask)));
                        fy = vcombine_s16(
                            vqmovn_s32(vandq_s32(ty0, dec_mask)),
                            vqmovn_s32(vandq_s32(ty1, dec_mask)));
                        int16x8_t final_f = vshlq_n_s16(vaddq_s16(vshrq_n_s16(fx, AB_BITS - INTER_BITS), fy), 2);
                        vst1q_s16(xy_dec_p + x1, final_f);

                        tx0 = vshrq_n_s32(tx0, AB_BITS);
                        tx1 = vshrq_n_s32(tx1, AB_BITS);
                        ty0 = vshrq_n_s32(ty0, AB_BITS);
                        ty1 = vshrq_n_s32(ty1, AB_BITS);

                        int32x4_t vx0 = vminq_s32(vmaxq_s32(tx0, v_zero4), max_width);
                        int32x4_t vx1 = vminq_s32(vmaxq_s32(vsubq_s32(tx0, v_nega), v_zero4), max_width);
                        int32x4_t vy0 = vminq_s32(vmaxq_s32(ty0, v_zero4), max_height);
                        int32x4_t vy1 = vminq_s32(vmaxq_s32(vsubq_s32(ty0, v_nega), v_zero4), max_height);

                        int32x4x4_t dst0;
                        dst0.val[0] = vmlaq_s32(vmulq_s32(vy0, v_inWidthStride), vx0, v_cn);
                        dst0.val[1] = vmlaq_s32(vmulq_s32(vy0, v_inWidthStride), vx1, v_cn);
                        dst0.val[2] = vmlaq_s32(vmulq_s32(vy1, v_inWidthStride), vx0, v_cn);
                        dst0.val[3] = vmlaq_s32(vmulq_s32(vy1, v_inWidthStride), vx1, v_cn);

                        vst4q_s32(xy_int_p + x1 * 4, dst0);

                        int32x4x4_t dst1;

                        vx0 = vminq_s32(vmaxq_s32(tx1, v_zero4), max_width);
                        vx1 = vminq_s32(vmaxq_s32(vsubq_s32(tx1, v_nega), v_zero4), max_width);
                        vy0 = vminq_s32(vmaxq_s32(ty1, v_zero4), max_height);
                        vy1 = vminq_s32(vmaxq_s32(vsubq_s32(ty1, v_nega), v_zero4), max_height);

                        dst1.val[0] = vmlaq_s32(vmulq_s32(vy0, v_inWidthStride), vx0, v_cn);
                        dst1.val[1] = vmlaq_s32(vmulq_s32(vy0, v_inWidthStride), vx1, v_cn);
                        dst1.val[2] = vmlaq_s32(vmulq_s32(vy1, v_inWidthStride), vx0, v_cn);
                        dst1.val[3] = vmlaq_s32(vmulq_s32(vy1, v_inWidthStride), vx1, v_cn);
                        vst4q_s32(xy_int_p + (x1 + 4) * 4, dst1);
                    }
                    for (; x1 < bw; ++x1) {
                        int32_t x_value = (x_int + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
                        int32_t y_value = (y_int + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
                        xy_dec_p[x1] = ((int16_t)((y_value & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE +
                                                  (x_value & (INTER_TAB_SIZE - 1))))
                                       << 2;
                        x_value = HPC::utils::saturate_cast<uint16_t>(x_value >> INTER_BITS);
                        y_value = HPC::utils::saturate_cast<uint16_t>(y_value >> INTER_BITS);
                        int32_t sx0 = clip(x_value, 0, inWidth - 1);
                        int32_t sy0 = clip(y_value, 0, inHeight - 1);
                        int32_t sx1 = clip((x_value + 1), 0, inWidth - 1);
                        int32_t sy1 = clip((y_value + 1), 0, inHeight - 1);
                        xy_int_p[x1 * 4] = sy0 * inWidthStride + sx0 * cn;
                        xy_int_p[x1 * 4 + 1] = sy0 * inWidthStride + sx1 * cn;
                        xy_int_p[x1 * 4 + 2] = sy1 * inWidthStride + sx0 * cn;
                        xy_int_p[x1 * 4 + 3] = sy1 * inWidthStride + sx1 * cn;
                    }
                }
                float32x4_t v_scale = vdupq_n_f32(INTER_REMAP_COEF_SCALE);
                for (int32_t y1 = 0; y1 < bh; ++y1) {
                    int32_t dstY = y1 + y;
                    int32_t x1 = 0;
                    int32_t *xy_int_p = XY_INT + ((y1 * bw) << 2);
                    int32_t dstIndex = dstY * outWidthStride + x * cn;
                    int16_t *xy_dec_p = XY_DEC + y1 * bw;

                    if (cn == 1) {
                        for (; x1 < bw; x1 += 4) {
                            float32x4_t p_s_t0 = vld1q_f32(short_tab + xy_dec_p[x1]);
                            float32x4_t p_s_t1 = vld1q_f32(short_tab + xy_dec_p[x1 + 1]);
                            float32x4_t p_s_t2 = vld1q_f32(short_tab + xy_dec_p[x1 + 2]);
                            float32x4_t p_s_t3 = vld1q_f32(short_tab + xy_dec_p[x1 + 3]);
                            float32x4_t src0 = {
                                src[xy_int_p[x1 << 2]], src[xy_int_p[(x1 << 2) + 1]], src[xy_int_p[(x1 << 2) + 2]], src[xy_int_p[(x1 << 2) + 3]]};
                            float32x4_t src1 = {
                                src[xy_int_p[((x1 + 1) << 2)]], src[xy_int_p[((x1 + 1) << 2) + 1]], src[xy_int_p[((x1 + 1) << 2) + 2]], src[xy_int_p[((x1 + 1) << 2) + 3]]};
                            float32x4_t src2 = {
                                src[xy_int_p[((x1 + 2) << 2)]], src[xy_int_p[((x1 + 2) << 2) + 1]], src[xy_int_p[((x1 + 2) << 2) + 2]], src[xy_int_p[((x1 + 2) << 2) + 3]]};
                            float32x4_t src3 = {
                                src[xy_int_p[((x1 + 3) << 2)]], src[xy_int_p[((x1 + 3) << 2) + 1]], src[xy_int_p[((x1 + 3) << 2) + 2]], src[xy_int_p[((x1 + 3) << 2) + 3]]};
                            src0 = vmulq_f32(src0, p_s_t0);
                            src1 = vmulq_f32(src1, p_s_t1);
                            src2 = vmulq_f32(src2, p_s_t2);
                            src3 = vmulq_f32(src3, p_s_t3);
                            float32x4_t res = {vaddvq_f32(src0), vaddvq_f32(src1), vaddvq_f32(src2), vaddvq_f32(src3)};
                            vst1q_f32(dst + dstIndex, vdivq_f32(res, v_scale));
                            dstIndex += 4;
                        }
                        for (; x1 < bw; ++x1) {
                            float32x4_t p_s_t = vld1q_f32(short_tab + xy_dec_p[x1]);
                            float32x4_t sum = {src[xy_int_p[(x1 << 2)]],
                                               src[xy_int_p[(x1 << 2) + 1]],
                                               src[xy_int_p[(x1 << 2) + 2]],
                                               src[xy_int_p[(x1 << 2) + 3]]};
                            sum = vmulq_f32(p_s_t, sum);
                            dst[dstIndex] = vaddvq_f32(sum) / INTER_REMAP_COEF_SCALE;
                            dstIndex++;
                        }
                    } else if (cn == 4) {
                        for (int32_t x1 = 0; x1 < bw; ++x1) {
                            float32x4_t p_s_t = vld1q_f32(short_tab + xy_dec_p[x1]);
                            float32x4x4_t sum = {vld1q_f32(src + xy_int_p[(x1 << 2)]),
                                                 vld1q_f32(src + xy_int_p[(x1 << 2) + 1]),
                                                 vld1q_f32(src + xy_int_p[(x1 << 2) + 2]),
                                                 vld1q_f32(src + xy_int_p[(x1 << 2) + 3])};
                            sum.val[0] = vmulq_laneq_f32(sum.val[0], p_s_t, 0);
                            sum.val[1] = vmulq_laneq_f32(sum.val[1], p_s_t, 1);
                            sum.val[2] = vmulq_laneq_f32(sum.val[2], p_s_t, 2);
                            sum.val[3] = vmulq_laneq_f32(sum.val[3], p_s_t, 3);
                            vst1q_f32(dst + dstIndex, vdivq_f32(vaddq_f32(vaddq_f32(sum.val[0], sum.val[1]), vaddq_f32(sum.val[2], sum.val[3])), v_scale));
                            dstIndex += 4;
                        }
                    } else {
                        for (int32_t x1 = 0; x1 < bw; ++x1) {
                            float32x4_t p_s_t = vld1q_f32(short_tab + xy_dec_p[x1]);
                            float32x4x4_t sum = {vld1q_f32(src + xy_int_p[(x1 << 2)]),
                                                 vld1q_f32(src + xy_int_p[(x1 << 2) + 1]),
                                                 vld1q_f32(src + xy_int_p[(x1 << 2) + 2]),
                                                 vld1q_f32(src + xy_int_p[(x1 << 2) + 3])};
                            sum.val[0] = vmulq_laneq_f32(sum.val[0], p_s_t, 0);
                            sum.val[1] = vmulq_laneq_f32(sum.val[1], p_s_t, 1);
                            sum.val[2] = vmulq_laneq_f32(sum.val[2], p_s_t, 2);
                            sum.val[3] = vmulq_laneq_f32(sum.val[3], p_s_t, 3);
                            float32x4_t res = vdivq_f32(vaddq_f32(vaddq_f32(sum.val[0], sum.val[1]), vaddq_f32(sum.val[2], sum.val[3])), v_scale);
                            vst1_f32(dst + dstIndex, vget_low_f32(res));
                            dst[dstIndex + 2] = vgetq_lane_f32(res, 2);
                            dstIndex += 3;
                        }
                    }
                }
            }
        }
    }

    free(short_tab);
    free(adelta);
    free(bdelta);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode WarpAffineNearestPoint<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData,
    const float *affineMatrix,
    BorderType border_type,
    float borderValue)
{
    if (inData == nullptr || outData == nullptr || affineMatrix == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || inWidthStride < inWidth || outHeight <= 0 || outWidth <= 0 || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return warpAffine_nearest<float, 1>(outData, inData, inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, affineMatrix, border_type, borderValue);
}

template <>
::ppl::common::RetCode WarpAffineNearestPoint<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData,
    const float *affineMatrix,
    BorderType border_type,
    float borderValue)
{
    if (inData == nullptr || outData == nullptr || affineMatrix == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || inWidthStride < inWidth || outHeight <= 0 || outWidth <= 0 || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return warpAffine_nearest<float, 3>(outData, inData, inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, affineMatrix, border_type, borderValue);
}

template <>
::ppl::common::RetCode WarpAffineNearestPoint<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData,
    const float *affineMatrix,
    BorderType border_type,
    float borderValue)
{
    if (inData == nullptr || outData == nullptr || affineMatrix == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || inWidthStride < inWidth || outHeight <= 0 || outWidth <= 0 || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return warpAffine_nearest<float, 4>(outData, inData, inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, affineMatrix, border_type, borderValue);
}

template <>
::ppl::common::RetCode WarpAffineLinear<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData,
    const float *affineMatrix,
    BorderType border_type,
    float borderValue)
{
    if (inData == nullptr || outData == nullptr || affineMatrix == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || inWidthStride < inWidth || outHeight <= 0 || outWidth <= 0 || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return warpAffine_linear_float(outData, inData, inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, affineMatrix, 1, border_type, borderValue);
}

template <>
::ppl::common::RetCode WarpAffineLinear<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData,
    const float *affineMatrix,
    BorderType border_type,
    float borderValue)
{
    if (inData == nullptr || outData == nullptr || affineMatrix == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || inWidthStride < inWidth || outHeight <= 0 || outWidth <= 0 || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return warpAffine_linear_float(outData, inData, inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, affineMatrix, 3, border_type, borderValue);
}

template <>
::ppl::common::RetCode WarpAffineLinear<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float *outData,
    const float *affineMatrix,
    BorderType border_type,
    float borderValue)
{
    if (inData == nullptr || outData == nullptr || affineMatrix == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight <= 0 || inWidth <= 0 || inWidthStride < inWidth || outHeight <= 0 || outWidth <= 0 || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_CONSTANT && border_type != BORDER_REPLICATE && border_type != BORDER_TRANSPARENT) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return warpAffine_linear_float(outData, inData, inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, affineMatrix, 4, border_type, borderValue);
}

}
}
} // namespace ppl::cv::arm
