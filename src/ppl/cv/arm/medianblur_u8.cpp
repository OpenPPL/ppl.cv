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

#include "ppl/cv/types.h"
#include <arm_neon.h>
#include <string.h>
#include <limits>
#include <float.h>
#include "ppl/cv/arm/medianblur.h"
#include "ppl/cv/arm/copymakeborder.h"
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

namespace ppl {
namespace cv {
namespace arm {

static inline int32_t findIndex(int32_t p, int32_t len, int32_t borderType)
{
    if (p >= 0 && p < len) {
        return p;
    }
    if (borderType == BORDER_REPLICATE) {
        p = p < 0 ? 0 : len - 1;
    } else if (borderType == BORDER_REFLECT_101) {
        p = p < 0 ? (-p) : 2 * len - p - 2;
    } else if (borderType == BORDER_REFLECT) {
        p = p < 0 ? (-p - 1) : 2 * len - p - 1;
    } else if (borderType == BORDER_CONSTANT) {
        p = -1;
    }
    return p;
}

template <typename T>
static T findKth(T* a, int32_t n, int32_t k)
{
    T x = a[0];
    int32_t i = 0, j = n - 1, pos = 0;
    while (i < j) {
        while (i < j && a[j] >= x) {
            --j;
        }
        if (i < j) {
            a[pos] = a[j];
            pos = j;
        }
        while (i < j && a[i] <= x) {
            ++i;
        }
        if (i < j) {
            a[pos] = a[i];
            pos = i;
        }
    }
    a[pos] = x;
    if (pos == k - 1) {
        return a[pos];
    }
    if (pos < k - 1) {
        return findKth(a + pos + 1, n - pos - 1, k - pos - 1);
    } else
        return findKth(a, pos + 1, k);
}

#define inline __inline__
#define align(x) __attribute__((aligned(x)))

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

typedef struct align(16)
{
    uint16_t coarse[16];
    uint16_t fine[16][16];
}
Histogram;

#define HOP(h, x, op)    \
    h.coarse[x >> 4] op; \
    *((uint16_t*)h.fine + x) op;

#define COP(c, j, x, op)                      \
    h_coarse[16 * (n * c + j) + (x >> 4)] op; \
    h_fine[16 * (n * (16 * c + (x >> 4)) + j) + (x & 0xF)] op;

static inline void histogram_add(const uint16_t x[16], uint16_t y[16])
{
    uint16x8_t vx0 = vld1q_u16(x);
    uint16x8_t vx1 = vld1q_u16(x + 8);
    uint16x8_t vy0 = vld1q_u16(y);
    uint16x8_t vy1 = vld1q_u16(y + 8);
    vst1q_u16(y, vaddq_u16(vy0, vx0));
    vst1q_u16(y + 8, vaddq_u16(vy1, vx1));
}

static inline void histogram_sub(const uint16_t x[16], uint16_t y[16])
{
    uint16x8_t vx0 = vld1q_u16(x);
    uint16x8_t vx1 = vld1q_u16(x + 8);
    uint16x8_t vy0 = vld1q_u16(y);
    uint16x8_t vy1 = vld1q_u16(y + 8);
    vst1q_u16(y, vsubq_u16(vy0, vx0));
    vst1q_u16(y + 8, vsubq_u16(vy1, vx1));
}

static inline void histogram_muladd(const uint16_t a, const uint16_t x[16], uint16_t y[16])
{
    uint16x8_t vx0 = vld1q_u16(x);
    uint16x8_t vx1 = vld1q_u16(x + 8);
    uint16x8_t vy0 = vld1q_u16(y);
    uint16x8_t vy1 = vld1q_u16(y + 8);
    vy0 = vmlaq_n_u16(vy0, vx0, a);
    vy1 = vmlaq_n_u16(vy1, vx1, a);
    vst1q_u16(y, vy0);
    vst1q_u16(y + 8, vy1);
}

static void MedianBlur_CTMF(
    const uint8_t* const src,
    uint8_t* const dst,
    const int32_t width,
    const int32_t height,
    const int32_t src_step,
    const int32_t dst_step,
    const int32_t r,
    const int32_t cn,
    BorderType border_type)
{
    const int32_t m = height, n = width;
    int32_t i, j, k, c;
    const uint8_t *p, *q;

    Histogram H[4];
    uint16_t *h_coarse, *h_fine, luc[4][16];

    h_coarse = (uint16_t*)calloc(1 * 16 * n * cn, sizeof(uint16_t));
    h_fine = (uint16_t*)calloc(16 * 16 * n * cn, sizeof(uint16_t));

    /* First row initialization */
    for (i = -r; i <= 0; i++) {
        int32_t idx = findIndex(i, height, border_type);
        for (j = 0; j < n; ++j) {
            for (c = 0; c < cn; ++c) {
                COP(c, j, src[src_step * idx + cn * j + c], ++);
            }
        }
    }
    for (i = 0; i < r; ++i) {
        for (j = 0; j < n; ++j) {
            for (c = 0; c < cn; ++c) {
                COP(c, j, src[src_step * i + cn * j + c], ++);
            }
        }
    }

    for (i = 0; i < m; ++i) {
        /* Update column histograms for entire row. */
        p = src + src_step * findIndex(i - r - 1, height, border_type);
        q = p + cn * n;
        for (j = 0; p != q; ++j) {
            for (c = 0; c < cn; ++c, ++p) {
                COP(c, j, *p, --);
            }
        }

        p = src + src_step * findIndex(i + r, height, border_type);
        q = p + cn * n;
        for (j = 0; p != q; ++j) {
            for (c = 0; c < cn; ++c, ++p) {
                COP(c, j, *p, ++);
            }
        }

        /* First column initialization */
        memset(H, 0, cn * sizeof(H[0]));
        memset(luc, 0, cn * sizeof(luc[0]));
        for (j = -r; j < 0; ++j) {
            int32_t idx = findIndex(j, width, border_type);
            for (c = 0; c < cn; ++c) {
                histogram_add(&h_coarse[16 * (n * c + idx)], H[c].coarse);
            }
        }
        for (j = 0; j < r; ++j) {
            for (c = 0; c < cn; ++c) {
                histogram_add(&h_coarse[16 * (n * c + j)], H[c].coarse);
            }
        }
        for (c = 0; c < cn; ++c) {
            for (k = 0; k < 16; ++k) {
                histogram_muladd(2 * r + 1, &h_fine[16 * n * (16 * c + k)], &H[c].fine[k][0]);
            }
        }

        for (j = 0; j < n; ++j) {
            for (c = 0; c < cn; ++c) {
                const uint16_t t = 2 * r * r + 2 * r;
                uint16_t sum = 0, *segment;
                int32_t b;

                int32_t idx = findIndex(j + r, width, border_type);
                histogram_add(&h_coarse[16 * (n * c + idx)], H[c].coarse);

                /* Find median at coarse level */
                for (k = 0; k < 16; ++k) {
                    sum += H[c].coarse[k];
                    if (sum > t) {
                        sum -= H[c].coarse[k];
                        break;
                    }
                }
                // assert(k < 16);

                /* Update corresponding histogram segment */
                if (luc[c][k] <= j - r) {
                    memset(&H[c].fine[k], 0, 16 * sizeof(uint16_t));
                    for (luc[c][k] = j - r; luc[c][k] < MIN(j + r + 1, n); ++luc[c][k]) {
                        histogram_add(&h_fine[16 * (n * (16 * c + k) + luc[c][k])], H[c].fine[k]);
                    }
                    if (luc[c][k] < j + r + 1) {
                        histogram_muladd(j + r + 1 - n, &h_fine[16 * (n * (16 * c + k) + (n - 1))], &H[c].fine[k][0]);
                        luc[c][k] = j + r + 1;
                    }
                } else {
                    for (; luc[c][k] < j + r + 1; ++luc[c][k]) {
                        idx = findIndex(luc[c][k] - 2 * r - 1, width, border_type);
                        histogram_sub(&h_fine[16 * (n * (16 * c + k) + idx)], H[c].fine[k]);
                        idx = findIndex(luc[c][k], width, border_type);
                        histogram_add(&h_fine[16 * (n * (16 * c + k) + idx)], H[c].fine[k]);
                    }
                }
                idx = findIndex(j - r, width, border_type);
                histogram_sub(&h_coarse[16 * (n * c + idx)], H[c].coarse);

                /* Find median in segment */
                segment = H[c].fine[k];
                for (b = 0; b < 16; ++b) {
                    sum += segment[b];
                    if (sum > t) {
                        dst[dst_step * i + cn * j + c] = 16 * k + b;
                        break;
                    }
                }
                // assert(b < 16);
            }
        }
    }

    free(h_coarse);
    free(h_fine);
}

///////////////////////////////////////////special case for 3x3 & 5x5 using bitonic sort//////////////////////////////////////

#define SORT_VECU8(a, b)         \
    {                            \
        uint8x16_t a_temp = a;   \
        a = vminq_u8(a, b);      \
        b = vmaxq_u8(a_temp, b); \
    }

static inline void CalcMedian_3x3u8x16(uint8x16_t& v0, uint8x16_t& v1, uint8x16_t& v2, uint8x16_t& v3, uint8x16_t& v4, uint8x16_t& v5, uint8x16_t& v6, uint8x16_t& v7, uint8x16_t& v8)
{
    SORT_VECU8(v1, v2);
    SORT_VECU8(v4, v5);
    SORT_VECU8(v7, v8);
    SORT_VECU8(v0, v1); // copied from opencv
    SORT_VECU8(v3, v4);
    SORT_VECU8(v6, v7);
    SORT_VECU8(v1, v2);
    SORT_VECU8(v4, v5); // this is not sort
    SORT_VECU8(v7, v8);
    SORT_VECU8(v0, v3);
    SORT_VECU8(v5, v8);
    SORT_VECU8(v4, v7); // just make sure v4
    SORT_VECU8(v3, v6);
    SORT_VECU8(v1, v4);
    SORT_VECU8(v2, v5);
    SORT_VECU8(v4, v7); // is the median value
    SORT_VECU8(v4, v2);
    SORT_VECU8(v6, v4);
    SORT_VECU8(v4, v2);
}

template <int32_t cn>
static inline uint8x16_t getLeftBorder(const uint8_t* src, int32_t width, int32_t radius, BorderType border_type)
{
    uint8_t temp[16];
    for (int32_t i = 1; i <= radius; i++) {
        const int32_t idx = findIndex(-i, width, border_type);
        for (int32_t c = 0; c < cn; c++) {
            temp[16 - i * cn + c] = src[idx * cn + c];
        }
    }
    return vld1q_u8(temp);
}

template <int32_t cn>
static inline uint8x16_t getRightBorder(const uint8_t* src, int32_t width, int32_t radius, BorderType border_type)
{
    uint8_t temp[16];
    for (int32_t i = 0; i < radius; i++) {
        const int32_t idx = findIndex(width + i, width, border_type);
        for (int32_t c = 0; c < cn; c++) {
            temp[i * cn + c] = src[idx * cn + c];
        }
    }
    return vld1q_u8(temp);
}

template <int32_t cn>
static inline uint8x16_t getRightData(
    const uint8_t* src,
    int32_t width,
    int32_t bytes_remain,
    int32_t radius,
    BorderType border_type)
{
    uint8x16_t v_border = getRightBorder<cn>(src, width, radius, border_type);
    if (bytes_remain <= 0) {
        return v_border;
    }
    uint8x16_t v_mid = vld1q_u8(src + width * cn - 16);
    switch (bytes_remain) {
        case 1: return vextq_u8(v_mid, v_border, 16 - 1);
        case 2: return vextq_u8(v_mid, v_border, 16 - 2);
        case 3: return vextq_u8(v_mid, v_border, 16 - 3);
        case 4: return vextq_u8(v_mid, v_border, 16 - 4);
        case 5: return vextq_u8(v_mid, v_border, 16 - 5);
        case 6: return vextq_u8(v_mid, v_border, 16 - 6);
        case 7: return vextq_u8(v_mid, v_border, 16 - 7);
        case 8: return vextq_u8(v_mid, v_border, 16 - 8);
        case 9: return vextq_u8(v_mid, v_border, 16 - 9);
        case 10: return vextq_u8(v_mid, v_border, 16 - 10);
        case 11: return vextq_u8(v_mid, v_border, 16 - 11);
        case 12: return vextq_u8(v_mid, v_border, 16 - 12);
        case 13: return vextq_u8(v_mid, v_border, 16 - 13);
        case 14: return vextq_u8(v_mid, v_border, 16 - 14);
        case 15: return vextq_u8(v_mid, v_border, 16 - 15);
        default: return v_border;
    }
}

template <int32_t cn>
static void MedianBlur_3x3u8(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type)
{
    const uint8_t *srows0, *srows1, *srows2;
    uint8_t* drow;
    uint8x16_t v_prev0, v_curr0, v_next0; // for srows0
    uint8x16_t v_prev1, v_curr1, v_next1; // for srows1
    uint8x16_t v_prev2, v_curr2, v_next2; // for srows2
    uint8x16_t v00, v01, v02; // for srows0
    uint8x16_t v10, v11, v12; // for srows1
    uint8x16_t v20, v21, v22; // for srows2

    for (int32_t i = 0; i < height; i++) {
        srows0 = inData + inWidthStride * findIndex(i - 1, height, border_type);
        srows1 = inData + inWidthStride * findIndex(i, height, border_type);
        srows2 = inData + inWidthStride * findIndex(i + 1, height, border_type);
        drow = outData + i * outWidthStride;
        // left border
        v_curr0 = getLeftBorder<cn>(srows0, width, 1, border_type);
        v_curr1 = getLeftBorder<cn>(srows1, width, 1, border_type);
        v_curr2 = getLeftBorder<cn>(srows2, width, 1, border_type);
        v_next0 = vld1q_u8(srows0);
        v_next1 = vld1q_u8(srows1);
        v_next2 = vld1q_u8(srows2);
        int32_t j = 0;
        // center
        for (; j + 32 <= width * cn; j += 16) {
            v_prev0 = v_curr0;
            v_prev1 = v_curr1;
            v_prev2 = v_curr2;
            v_curr0 = v_next0;
            v_curr1 = v_next1;
            v_curr2 = v_next2;
            v_next0 = vld1q_u8(srows0 + j + 16);
            v_next1 = vld1q_u8(srows1 + j + 16);
            v_next2 = vld1q_u8(srows2 + j + 16);
            v00 = vextq_u8(v_prev0, v_curr0, 16 - cn);
            v10 = vextq_u8(v_prev1, v_curr1, 16 - cn);
            v20 = vextq_u8(v_prev2, v_curr2, 16 - cn);
            v01 = v_curr0;
            v11 = v_curr1;
            v21 = v_curr2;
            v02 = vextq_u8(v_curr0, v_next0, cn);
            v12 = vextq_u8(v_curr1, v_next1, cn);
            v22 = vextq_u8(v_curr2, v_next2, cn);
            CalcMedian_3x3u8x16(v00, v01, v02, v10, v11, v12, v20, v21, v22);
            uint8x16_t& v_result = v11;
            vst1q_u8(drow + j, v_result);
        }
        // right border
        v_prev0 = v_curr0;
        v_prev1 = v_curr1;
        v_prev2 = v_curr2;
        v_curr0 = v_next0;
        v_curr1 = v_next1;
        v_curr2 = v_next2;
        const int32_t bytes_remain = width * cn - (j + 16);
        v_next0 = getRightData<cn>(srows0, width, bytes_remain, 1, border_type);
        v_next1 = getRightData<cn>(srows1, width, bytes_remain, 1, border_type);
        v_next2 = getRightData<cn>(srows2, width, bytes_remain, 1, border_type);
        v00 = vextq_u8(v_prev0, v_curr0, 16 - cn);
        v10 = vextq_u8(v_prev1, v_curr1, 16 - cn);
        v20 = vextq_u8(v_prev2, v_curr2, 16 - cn);
        v01 = v_curr0;
        v11 = v_curr1;
        v21 = v_curr2;
        v02 = vextq_u8(v_curr0, v_next0, cn);
        v12 = vextq_u8(v_curr1, v_next1, cn);
        v22 = vextq_u8(v_curr2, v_next2, cn);
        CalcMedian_3x3u8x16(v00, v01, v02, v10, v11, v12, v20, v21, v22);
        uint8x16_t& v_result = v11;
        vst1q_u8(drow + j, v_result);
        j += 16;
        if (j >= width * cn) {
            continue;
        }
        // not aligned to 16, still has remainig
        v_prev0 = v_curr0;
        v_prev1 = v_curr1;
        v_prev2 = v_curr2;
        v_curr0 = v_next0;
        v_curr1 = v_next1;
        v_curr2 = v_next2;
        v00 = vextq_u8(v_prev0, v_curr0, 16 - cn);
        v10 = vextq_u8(v_prev1, v_curr1, 16 - cn);
        v20 = vextq_u8(v_prev2, v_curr2, 16 - cn);
        v01 = v_curr0;
        v11 = v_curr1;
        v21 = v_curr2;
        v02 = vextq_u8(v_curr0, v_next0, cn);
        v12 = vextq_u8(v_curr1, v_next1, cn);
        v22 = vextq_u8(v_curr2, v_next2, cn);
        CalcMedian_3x3u8x16(v00, v01, v02, v10, v11, v12, v20, v21, v22);
        uint8_t temp[16];
        vst1q_u8(temp, v_result);
        for (int32_t k = 0; j < width * cn; j++, k++) {
            drow[j] = temp[k];
        }
    }
}

static inline void CalcMedian_5x5u8x16(uint8x16_t v[25])
{
    // just found median, nor sort
    SORT_VECU8(v[1], v[2]);
    SORT_VECU8(v[0], v[1]);
    SORT_VECU8(v[1], v[2]);
    SORT_VECU8(v[4], v[5]);
    SORT_VECU8(v[3], v[4]);
    SORT_VECU8(v[4], v[5]);
    SORT_VECU8(v[0], v[3]);
    SORT_VECU8(v[2], v[5]);
    SORT_VECU8(v[2], v[3]);
    SORT_VECU8(v[1], v[4]);
    SORT_VECU8(v[1], v[2]);
    SORT_VECU8(v[3], v[4]);
    SORT_VECU8(v[7], v[8]);
    SORT_VECU8(v[6], v[7]);
    SORT_VECU8(v[7], v[8]);
    SORT_VECU8(v[10], v[11]);
    SORT_VECU8(v[9], v[10]);
    SORT_VECU8(v[10], v[11]);
    SORT_VECU8(v[6], v[9]);
    SORT_VECU8(v[8], v[11]);
    SORT_VECU8(v[8], v[9]);
    SORT_VECU8(v[7], v[10]);
    SORT_VECU8(v[7], v[8]);
    SORT_VECU8(v[9], v[10]);
    SORT_VECU8(v[0], v[6]);
    SORT_VECU8(v[4], v[10]);
    SORT_VECU8(v[4], v[6]);
    SORT_VECU8(v[2], v[8]);
    SORT_VECU8(v[2], v[4]);
    SORT_VECU8(v[6], v[8]);
    SORT_VECU8(v[1], v[7]);
    SORT_VECU8(v[5], v[11]);
    SORT_VECU8(v[5], v[7]);
    SORT_VECU8(v[3], v[9]);
    SORT_VECU8(v[3], v[5]);
    SORT_VECU8(v[7], v[9]);
    SORT_VECU8(v[1], v[2]);
    SORT_VECU8(v[3], v[4]);
    SORT_VECU8(v[5], v[6]);
    SORT_VECU8(v[7], v[8]);
    SORT_VECU8(v[9], v[10]);
    SORT_VECU8(v[13], v[14]);
    SORT_VECU8(v[12], v[13]);
    SORT_VECU8(v[13], v[14]);
    SORT_VECU8(v[16], v[17]);
    SORT_VECU8(v[15], v[16]);
    SORT_VECU8(v[16], v[17]);
    SORT_VECU8(v[12], v[15]);
    SORT_VECU8(v[14], v[17]);
    SORT_VECU8(v[14], v[15]);
    SORT_VECU8(v[13], v[16]);
    SORT_VECU8(v[13], v[14]);
    SORT_VECU8(v[15], v[16]);
    SORT_VECU8(v[19], v[20]);
    SORT_VECU8(v[18], v[19]);
    SORT_VECU8(v[19], v[20]);
    SORT_VECU8(v[21], v[22]);
    SORT_VECU8(v[23], v[24]);
    SORT_VECU8(v[21], v[23]);
    SORT_VECU8(v[22], v[24]);
    SORT_VECU8(v[22], v[23]);
    SORT_VECU8(v[18], v[21]);
    SORT_VECU8(v[20], v[23]);
    SORT_VECU8(v[20], v[21]);
    SORT_VECU8(v[19], v[22]);
    SORT_VECU8(v[22], v[24]);
    SORT_VECU8(v[19], v[20]);
    SORT_VECU8(v[21], v[22]);
    SORT_VECU8(v[23], v[24]);
    SORT_VECU8(v[12], v[18]);
    SORT_VECU8(v[16], v[22]);
    SORT_VECU8(v[16], v[18]);
    SORT_VECU8(v[14], v[20]);
    SORT_VECU8(v[20], v[24]);
    SORT_VECU8(v[14], v[16]);
    SORT_VECU8(v[18], v[20]);
    SORT_VECU8(v[22], v[24]);
    SORT_VECU8(v[13], v[19]);
    SORT_VECU8(v[17], v[23]);
    SORT_VECU8(v[17], v[19]);
    SORT_VECU8(v[15], v[21]);
    SORT_VECU8(v[15], v[17]);
    SORT_VECU8(v[19], v[21]);
    SORT_VECU8(v[13], v[14]);
    SORT_VECU8(v[15], v[16]);
    SORT_VECU8(v[17], v[18]);
    SORT_VECU8(v[19], v[20]);
    SORT_VECU8(v[21], v[22]);
    SORT_VECU8(v[23], v[24]);
    SORT_VECU8(v[0], v[12]);
    SORT_VECU8(v[8], v[20]);
    SORT_VECU8(v[8], v[12]);
    SORT_VECU8(v[4], v[16]);
    SORT_VECU8(v[16], v[24]);
    SORT_VECU8(v[12], v[16]);
    SORT_VECU8(v[2], v[14]);
    SORT_VECU8(v[10], v[22]);
    SORT_VECU8(v[10], v[14]);
    SORT_VECU8(v[6], v[18]);
    SORT_VECU8(v[6], v[10]);
    SORT_VECU8(v[10], v[12]);
    SORT_VECU8(v[1], v[13]);
    SORT_VECU8(v[9], v[21]);
    SORT_VECU8(v[9], v[13]);
    SORT_VECU8(v[5], v[17]);
    SORT_VECU8(v[13], v[17]);
    SORT_VECU8(v[3], v[15]);
    SORT_VECU8(v[11], v[23]);
    SORT_VECU8(v[11], v[15]);
    SORT_VECU8(v[7], v[19]);
    SORT_VECU8(v[7], v[11]);
    SORT_VECU8(v[11], v[13]);
    SORT_VECU8(v[11], v[12]);
}

template <int32_t cn>
static void MedianBlur_5x5u8(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    BorderType border_type)
{
    const uint8_t *srows0, *srows1, *srows2, *srows3, *srows4;
    uint8_t* drow;

    for (int32_t i = 0; i < height; i++) {
        srows0 = inData + inWidthStride * findIndex(i - 2, height, border_type);
        srows1 = inData + inWidthStride * findIndex(i - 1, height, border_type);
        srows2 = inData + inWidthStride * findIndex(i, height, border_type);
        srows3 = inData + inWidthStride * findIndex(i + 1, height, border_type);
        srows4 = inData + inWidthStride * findIndex(i + 2, height, border_type);
        drow = outData + i * outWidthStride;
        uint8x16_t v[25];

        int32_t j = 0;
        for (; j + 32 <= width * cn; j += 16) {
            // mid
            v[0 * 5 + 2] = vld1q_u8(srows0 + j);
            v[1 * 5 + 2] = vld1q_u8(srows1 + j);
            v[2 * 5 + 2] = vld1q_u8(srows2 + j);
            v[3 * 5 + 2] = vld1q_u8(srows3 + j);
            v[4 * 5 + 2] = vld1q_u8(srows4 + j);
            // left
            if (j >= 16) { // normal case
                v[0 * 5 + 0] = vld1q_u8(srows0 + j - 2 * cn);
                v[1 * 5 + 0] = vld1q_u8(srows1 + j - 2 * cn);
                v[2 * 5 + 0] = vld1q_u8(srows2 + j - 2 * cn);
                v[3 * 5 + 0] = vld1q_u8(srows3 + j - 2 * cn);
                v[4 * 5 + 0] = vld1q_u8(srows4 + j - 2 * cn);
                v[0 * 5 + 1] = vld1q_u8(srows0 + j - cn);
                v[1 * 5 + 1] = vld1q_u8(srows1 + j - cn);
                v[2 * 5 + 1] = vld1q_u8(srows2 + j - cn);
                v[3 * 5 + 1] = vld1q_u8(srows3 + j - cn);
                v[4 * 5 + 1] = vld1q_u8(srows4 + j - cn);
            } else { // left border
                uint8x16_t v_border0 = getLeftBorder<cn>(srows0, width, 2, border_type);
                uint8x16_t v_border1 = getLeftBorder<cn>(srows1, width, 2, border_type);
                uint8x16_t v_border2 = getLeftBorder<cn>(srows2, width, 2, border_type);
                uint8x16_t v_border3 = getLeftBorder<cn>(srows3, width, 2, border_type);
                uint8x16_t v_border4 = getLeftBorder<cn>(srows4, width, 2, border_type);
                v[0 * 5 + 0] = vextq_u8(v_border0, v[0 * 5 + 2], 16 - 2 * cn);
                v[0 * 5 + 1] = vextq_u8(v_border0, v[0 * 5 + 2], 16 - cn);
                v[1 * 5 + 0] = vextq_u8(v_border1, v[1 * 5 + 2], 16 - 2 * cn);
                v[1 * 5 + 1] = vextq_u8(v_border1, v[1 * 5 + 2], 16 - cn);
                v[2 * 5 + 0] = vextq_u8(v_border2, v[2 * 5 + 2], 16 - 2 * cn);
                v[2 * 5 + 1] = vextq_u8(v_border2, v[2 * 5 + 2], 16 - cn);
                v[3 * 5 + 0] = vextq_u8(v_border3, v[3 * 5 + 2], 16 - 2 * cn);
                v[3 * 5 + 1] = vextq_u8(v_border3, v[3 * 5 + 2], 16 - cn);
                v[4 * 5 + 0] = vextq_u8(v_border4, v[4 * 5 + 2], 16 - 2 * cn);
                v[4 * 5 + 1] = vextq_u8(v_border4, v[4 * 5 + 2], 16 - cn);
            }
            // right
            v[0 * 5 + 3] = vld1q_u8(srows0 + j + cn);
            v[1 * 5 + 3] = vld1q_u8(srows1 + j + cn);
            v[2 * 5 + 3] = vld1q_u8(srows2 + j + cn);
            v[3 * 5 + 3] = vld1q_u8(srows3 + j + cn);
            v[4 * 5 + 3] = vld1q_u8(srows4 + j + cn);
            v[0 * 5 + 4] = vld1q_u8(srows0 + j + 2 * cn);
            v[1 * 5 + 4] = vld1q_u8(srows1 + j + 2 * cn);
            v[2 * 5 + 4] = vld1q_u8(srows2 + j + 2 * cn);
            v[3 * 5 + 4] = vld1q_u8(srows3 + j + 2 * cn);
            v[4 * 5 + 4] = vld1q_u8(srows4 + j + 2 * cn);
            CalcMedian_5x5u8x16(v);
            uint8x16_t& v_result = v[2 * 5 + 2];
            vst1q_u8(drow + j, v_result);
        }
        // right border
        // mid
        v[0 * 5 + 2] = vld1q_u8(srows0 + j);
        v[1 * 5 + 2] = vld1q_u8(srows1 + j);
        v[2 * 5 + 2] = vld1q_u8(srows2 + j);
        v[3 * 5 + 2] = vld1q_u8(srows3 + j);
        v[4 * 5 + 2] = vld1q_u8(srows4 + j);
        // left
        v[0 * 5 + 0] = vld1q_u8(srows0 + j - 2 * cn);
        v[1 * 5 + 0] = vld1q_u8(srows1 + j - 2 * cn);
        v[2 * 5 + 0] = vld1q_u8(srows2 + j - 2 * cn);
        v[3 * 5 + 0] = vld1q_u8(srows3 + j - 2 * cn);
        v[4 * 5 + 0] = vld1q_u8(srows4 + j - 2 * cn);
        v[0 * 5 + 1] = vld1q_u8(srows0 + j - cn);
        v[1 * 5 + 1] = vld1q_u8(srows1 + j - cn);
        v[2 * 5 + 1] = vld1q_u8(srows2 + j - cn);
        v[3 * 5 + 1] = vld1q_u8(srows3 + j - cn);
        v[4 * 5 + 1] = vld1q_u8(srows4 + j - cn);
        // right
        const int32_t bytes_remain = width * cn - (j + 16);
        uint8x16_t v_border0 = getRightData<cn>(srows0, width, bytes_remain, 2, border_type);
        uint8x16_t v_border1 = getRightData<cn>(srows1, width, bytes_remain, 2, border_type);
        uint8x16_t v_border2 = getRightData<cn>(srows2, width, bytes_remain, 2, border_type);
        uint8x16_t v_border3 = getRightData<cn>(srows3, width, bytes_remain, 2, border_type);
        uint8x16_t v_border4 = getRightData<cn>(srows4, width, bytes_remain, 2, border_type);
        v[0 * 5 + 3] = vextq_u8(v[0 * 5 + 2], v_border0, cn);
        v[0 * 5 + 4] = vextq_u8(v[0 * 5 + 2], v_border0, 2 * cn);
        v[1 * 5 + 3] = vextq_u8(v[1 * 5 + 2], v_border1, cn);
        v[1 * 5 + 4] = vextq_u8(v[1 * 5 + 2], v_border1, 2 * cn);
        v[2 * 5 + 3] = vextq_u8(v[2 * 5 + 2], v_border2, cn);
        v[2 * 5 + 4] = vextq_u8(v[2 * 5 + 2], v_border2, 2 * cn);
        v[3 * 5 + 3] = vextq_u8(v[3 * 5 + 2], v_border3, cn);
        v[3 * 5 + 4] = vextq_u8(v[3 * 5 + 2], v_border3, 2 * cn);
        v[4 * 5 + 3] = vextq_u8(v[4 * 5 + 2], v_border4, cn);
        v[4 * 5 + 4] = vextq_u8(v[4 * 5 + 2], v_border4, 2 * cn);
        CalcMedian_5x5u8x16(v);
        uint8x16_t& v_result = v[2 * 5 + 2];
        vst1q_u8(drow + j, v_result);
        j += 16;
        if (j >= width * cn) {
            continue;
        }
        // not aligned to 16, still has remaining
        j /= cn;
        for (; j < width; j++) {
            for (int32_t c = 0; c < cn; c++) {
                uint8_t temp[25];
                for (int32_t kx = 0; kx < 5; kx++) {
                    int32_t idx = findIndex(j + kx - 2, width, border_type);
                    temp[kx] = srows0[idx * cn + c];
                    temp[kx + 5] = srows1[idx * cn + c];
                    temp[kx + 10] = srows2[idx * cn + c];
                    temp[kx + 15] = srows3[idx * cn + c];
                    temp[kx + 20] = srows4[idx * cn + c];
                }
                drow[j * cn + c] = findKth<uint8_t>(temp, 25, 13);
            }
        }
    }
}

template <int32_t cn>
static ::ppl::common::RetCode MedianBlurUchar(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t ksize,
    BorderType border_type)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride < width) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != BORDER_REFLECT_101 && border_type != BORDER_REFLECT && border_type != BORDER_REPLICATE) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (ksize == 3) {
        MedianBlur_3x3u8<cn>(height, width, inWidthStride, inData, outWidthStride, outData, border_type);
    } else if (ksize == 5) {
        MedianBlur_5x5u8<cn>(height, width, inWidthStride, inData, outWidthStride, outData, border_type);
    } else {
        MedianBlur_CTMF(inData, outData, width, height, inWidthStride, outWidthStride, ksize / 2, cn, border_type);
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode MedianBlur<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t ksize,
    BorderType border_type)
{
    return MedianBlurUchar<1>(height, width, inWidthStride, inData, outWidthStride, outData, ksize, border_type);
}

template <>
::ppl::common::RetCode MedianBlur<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t ksize,
    BorderType border_type)
{
    return MedianBlurUchar<3>(height, width, inWidthStride, inData, outWidthStride, outData, ksize, border_type);
}

template <>
::ppl::common::RetCode MedianBlur<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData,
    int32_t ksize,
    BorderType border_type)
{
    MedianBlurUchar<4>(height, width, inWidthStride, inData, outWidthStride, outData, ksize, border_type);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm
