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

#include "ppl/cv/x86/arithmetic.h"
#include "ppl/cv/x86/morph.hpp"

#include <immintrin.h>
#include <assert.h>
#include <stdio.h>
#include <cmath>
#include "string.h"

namespace ppl {
namespace cv {
namespace x86 {

#define VLEN 16 // 16 bytes = 128 bits for SSE reg
template <typename T>
inline T *getRowPtr(T *base, int32_t stride, int32_t row)
{
    float *baseRaw = const_cast<float *>(reinterpret_cast<const float *>(base));
    return reinterpret_cast<T *>(baseRaw + row * stride);
}

template <class morphOp, int32_t nc, int32_t kernel_len>
inline void MorphRow(__m128 *tprev, __m128 &tcurr, __m128 *tnext, const float *srcCenterRow, int32_t srcStride, float *drow, int32_t rowIdx, int32_t rowIdxInv, int32_t colIdx, int32_t colIdxInv, float borderValue = 0)
{
    morphOp vop;
    __m128 v_border                   = _mm_set1_ps(borderValue);
    constexpr int32_t kernel_radius   = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num  = (kernel_radius * nc * sizeof(float) + VLEN - 1) / VLEN;
    constexpr int32_t v_elem          = VLEN / sizeof(float) / nc;
    constexpr int8_t invalid_byte_len = VLEN % (nc * sizeof(float));
    switch (kernel_len) {
        case 3: {
            __m128 v_up, v_mid, v_down;
            __m128 t_left, t_mid, t_right;
            __m128i tcurr_tmp = _mm_slli_si128(_mm_castps_si128(tcurr), invalid_byte_len);
            __m128i tprev_tmp = _mm_slli_si128(_mm_castps_si128(tprev[0]), invalid_byte_len);

            v_up   = rowIdx == 0 ? v_border : _mm_loadu_ps(srcCenterRow - srcStride);
            v_mid  = _mm_loadu_ps(srcCenterRow);
            v_down = rowIdxInv == 0 ? v_border : _mm_loadu_ps(srcCenterRow + srcStride);

            tnext[0] = vop(vop(v_up, v_mid), v_down);

            t_left  = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tcurr), tprev_tmp, VLEN - nc * sizeof(float)));
            t_mid   = tcurr;
            t_right = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tnext[0]), tcurr_tmp, nc * sizeof(float) + invalid_byte_len));

            t_mid = vop(t_left, vop(t_mid, t_right));

            _mm_storeu_ps(drow, t_mid);
        } break;
        case 5: {
            __m128 v_up0, v_up1, v_mid, v_down0, v_down1;
            v_up0   = rowIdx < 2 ? v_border : _mm_loadu_ps(srcCenterRow - 2 * srcStride + v_elem * nc * (radius_vec_num - 1));
            v_up1   = rowIdx < 1 ? v_border : _mm_loadu_ps(srcCenterRow - 1 * srcStride + v_elem * nc * (radius_vec_num - 1));
            v_mid   = _mm_loadu_ps(srcCenterRow + v_elem * nc * (radius_vec_num - 1));
            v_down0 = rowIdxInv < 1 ? v_border : _mm_loadu_ps(srcCenterRow + 1 * srcStride + v_elem * nc * (radius_vec_num - 1));
            v_down1 = rowIdxInv < 2 ? v_border : _mm_loadu_ps(srcCenterRow + 2 * srcStride + v_elem * nc * (radius_vec_num - 1));

            tnext[radius_vec_num - 1] = vop(vop(vop(v_up0, v_up1), v_mid), vop(v_down0, v_down1));

            __m128 t_left0, t_left1, t_mid, t_right0, t_right1;
            if (radius_vec_num == 1) {
                __m128i tcurr_tmp = _mm_slli_si128(_mm_castps_si128(tcurr), invalid_byte_len);
                __m128i tprev_tmp = _mm_slli_si128(_mm_castps_si128(tprev[0]), invalid_byte_len);

                t_left0  = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tcurr), tprev_tmp, VLEN - nc * 2 * static_cast<int32_t>(sizeof(float))));
                t_left1  = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tcurr), tprev_tmp, VLEN - nc * 1 * sizeof(float)));
                t_mid    = tcurr;
                t_right0 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tnext[0]), tcurr_tmp, nc * 1 * sizeof(float) + invalid_byte_len));
                t_right1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tnext[0]), tcurr_tmp, nc * 2 * sizeof(float) + invalid_byte_len));
            } else {
                t_left0  = tprev[0];
                t_left1  = tprev[1];
                t_mid    = tcurr;
                t_right0 = tnext[0];
                t_right1 = tnext[1];
            }

            t_mid = vop(vop(t_mid, vop(t_left0, t_left1)), vop(t_right0, t_right1));
            _mm_storeu_ps(drow, t_mid);
        } break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
inline void MorphRowLast(__m128 *tprev, __m128 &tcurr, __m128 *tnext, const float *srcCenterRow, int32_t srcStride, float *drow, int32_t rowIdx, int32_t rowIdxInv, int32_t colIdx, int32_t colIdxInv, float borderValue = 0)
{
    morphOp vop;
    __m128 v_border                   = _mm_set1_ps(borderValue);
    constexpr int32_t kernel_radius   = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num  = (kernel_radius * nc * sizeof(float) + VLEN - 1) / VLEN;
    constexpr int32_t v_elem          = VLEN / sizeof(float) / nc;
    constexpr int8_t invalid_byte_len = VLEN % (nc * sizeof(float));

    switch (kernel_len) {
        case 3: {
            __m128 v_up, v_mid, v_down;
            __m128 t_left, t_mid, t_right;
            __m128i tcurr_tmp = _mm_slli_si128(_mm_castps_si128(tcurr), invalid_byte_len);
            __m128i tprev_tmp = _mm_slli_si128(_mm_castps_si128(tprev[0]), invalid_byte_len);

            if (colIdxInv + 1 - v_elem >= kernel_radius) { // #remaining is sufficient for current kernel.
                v_up     = rowIdx == 0 ? v_border : _mm_loadu_ps(srcCenterRow - srcStride);
                v_mid    = _mm_loadu_ps(srcCenterRow);
                v_down   = rowIdxInv == 0 ? v_border : _mm_loadu_ps(srcCenterRow + srcStride);
                tnext[0] = vop(vop(v_up, v_mid), v_down);
            } else {
                tnext[0] = v_border;
            }

            t_left  = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tcurr), tprev_tmp, VLEN - nc * sizeof(float)));
            t_mid   = tcurr;
            t_right = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tnext[0]), tcurr_tmp, nc * sizeof(float) + invalid_byte_len));

            t_mid = vop(t_left, vop(t_mid, t_right));

            if (VLEN % nc * sizeof(float) == 0 || (colIdxInv + 1) * nc * sizeof(float) >= VLEN) {
                _mm_storeu_ps(drow, t_mid);
            } else {
                int32_t i = 0;
                for (; i <= VLEN - invalid_byte_len - 4; i += 4) {
                    drow[i >> 2] = *((float *)&t_mid + (i >> 2));
                }
                for (; i < VLEN - invalid_byte_len; i++) {
                    ((uint8_t *)drow)[i] = ((uint8_t *)&t_mid)[i];
                }
            }
        } break;
        case 5: {
            __m128 v_up0, v_up1, v_mid, v_down0, v_down1;
            __m128 t_left0, t_left1, t_mid, t_right0, t_right1;
            __m128i tcurr_tmp = _mm_slli_si128(_mm_castps_si128(tcurr), invalid_byte_len);
            __m128i tprev_tmp = _mm_slli_si128(_mm_castps_si128(tprev[0]), invalid_byte_len);

            v_up0 = rowIdx < 2 ? v_border : _mm_loadu_ps(srcCenterRow - 2 * srcStride + v_elem * nc * (radius_vec_num - 1));
            v_up1 = rowIdx < 1 ? v_border : _mm_loadu_ps(srcCenterRow - 1 * srcStride + v_elem * nc * (radius_vec_num - 1));
            v_mid = _mm_loadu_ps(srcCenterRow + v_elem * nc * (radius_vec_num - 1));
            v_down0 = rowIdxInv < 1 ? v_border : _mm_loadu_ps(srcCenterRow + 1 * srcStride + v_elem * nc * (radius_vec_num - 1));
            v_down1 = rowIdxInv < 2 ? v_border : _mm_loadu_ps(srcCenterRow + 2 * srcStride + v_elem * nc * (radius_vec_num - 1));
            tnext[radius_vec_num - 1] = vop(vop(vop(v_up0, v_up1), v_mid), vop(v_down0, v_down1));

            for (int32_t i = colIdxInv + 1 - v_elem - (radius_vec_num - 1) * v_elem; i < kernel_radius - (radius_vec_num - 1) * v_elem; i++) {
                for (int32_t j = 0; j < nc; j++) {
                    ((float *)&(tnext[radius_vec_num - 1]))[i * nc + j] = borderValue;
                }
            }

            if (radius_vec_num == 1) {
                t_left0  = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tcurr), tprev_tmp, VLEN - nc * 2 * static_cast<int32_t>(sizeof(float))));
                t_left1  = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tcurr), tprev_tmp, VLEN - nc * 1 * sizeof(float)));
                t_mid    = tcurr;
                t_right0 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tnext[0]), tcurr_tmp, nc * 1 * sizeof(float) + invalid_byte_len));
                t_right1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(tnext[0]), tcurr_tmp, nc * 2 * sizeof(float) + invalid_byte_len));
            } else {
                t_left0  = tprev[0];
                t_left1  = tprev[1];
                t_mid    = tcurr;
                t_right0 = tnext[0];
                t_right1 = tnext[1];
            }
            t_mid = vop(vop(t_mid, vop(t_left0, t_left1)), vop(t_right0, t_right1));

            if (VLEN % nc * sizeof(float) == 0 || (colIdxInv + 1) * nc * sizeof(float) >= VLEN) {
                _mm_storeu_ps(drow, t_mid);
            } else {
                int32_t i = 0;
                for (; i <= VLEN - invalid_byte_len - 4; i += 4) {
                    drow[i >> 2] = *((float *)&t_mid + (i >> 2));
                }
                for (; i < VLEN - invalid_byte_len; i++) {
                    ((uint8_t *)drow)[i] = ((uint8_t *)&t_mid)[i];
                }
            }
        } break;
        default:
            break;
    }
}


template <class morphOp, int32_t nc, int32_t kernel_len>
inline void MorphFirstCol(__m128 &tcurr, __m128 *tnext, const float *srcCenterRow, int32_t srcStride, int32_t rowIdx, int32_t rowIdxInv, float borderValue = 0)
{
    morphOp vop;
    __m128 v_border                  = _mm_set1_ps(borderValue);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = (kernel_radius * nc * sizeof(float) + VLEN - 1) / VLEN;
    constexpr int32_t v_elem         = VLEN / sizeof(float) / nc;
    switch (kernel_len) {
        case 3: {
            __m128 v_up   = rowIdx == 0 ? v_border : _mm_loadu_ps(srcCenterRow - 1 * srcStride);
            __m128 v_mid  = _mm_loadu_ps(srcCenterRow);
            __m128 v_down = rowIdxInv == 0 ? v_border : _mm_loadu_ps(srcCenterRow + 1 * srcStride);

            tnext[0] = vop(vop(v_up, v_mid), v_down);
            tcurr    = v_border;
        } break;
        case 5: {
            for (int32_t i = 0; i < radius_vec_num; i++) {
                __m128 v_up0   = rowIdx < 2 ? v_border : _mm_loadu_ps(srcCenterRow - 2 * srcStride + v_elem * nc * i);
                __m128 v_up1   = rowIdx < 1 ? v_border : _mm_loadu_ps(srcCenterRow - 1 * srcStride + v_elem * nc * i);
                __m128 v_mid   = _mm_loadu_ps(srcCenterRow + v_elem * nc * i);
                __m128 v_down0 = rowIdxInv < 1 ? v_border : _mm_loadu_ps(srcCenterRow + 1 * srcStride + v_elem * nc * i);
                __m128 v_down1 = rowIdxInv < 2 ? v_border : _mm_loadu_ps(srcCenterRow + 2 * srcStride + v_elem * nc * i);
                tnext[i]       = vop(vop(vop(v_up0, v_up1), v_mid), vop(v_down0, v_down1));
            }
            tcurr = v_border;
        } break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
void morphScalar(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    int32_t startX,
    int32_t endX,
    int32_t startY,
    int32_t endY,
    BorderType border_type,
    float borderValue)
{
    if (startX >= endX || startY >= endY) return;
    morphOp vop;
    switch (kernel_len) {
        case 3: {
            for (int32_t y = startY; y < endY; ++y) {
                const float *srow1 = getRowPtr(srcBase, srcStride, y);
                const float *srow0 = y == 0 ? nullptr : getRowPtr(srcBase, srcStride, y - 1);
                const float *srow2 = y == (height - 1) ? nullptr : getRowPtr(srcBase, srcStride, y + 1);
                float *drow        = getRowPtr(dstBase, dstStride, y);

                int32_t x = startX;

                float temp_buffer[12] = {0.0f};
                float *prevx          = temp_buffer;
                float *currx          = temp_buffer + 4;
                float *nextx          = temp_buffer + 8;

                for (; x <= endX; x++) {
                    if (x >= width) break;

                    int32_t cur_idx  = x;
                    int32_t prev_idx = cur_idx - 1;
                    int32_t next_idx = cur_idx + 1;

                    if (x > startX) {
                        for (int32_t idx_c = 0; idx_c < nc; idx_c++) {
                            if (next_idx >= width) {
                                nextx[idx_c] = borderValue;
                            } else {
                                nextx[idx_c] = vop(vop(srow2 ? srow2[next_idx * nc + idx_c] : borderValue,
                                                       srow0 ? srow0[next_idx * nc + idx_c] : borderValue),
                                                   srow1[next_idx * nc + idx_c]);
                            }

                            drow[cur_idx * nc + idx_c] = vop(prevx[idx_c], vop(currx[idx_c], nextx[idx_c]));
                        }
                    } else {
                        for (int32_t idx_c = 0; idx_c < nc; idx_c++) {
                            currx[idx_c] = vop(srow2 ? srow2[cur_idx * nc + idx_c] : borderValue, vop(srow1[cur_idx * nc + idx_c], srow0 ? srow0[cur_idx * nc + idx_c] : borderValue));

                            if (prev_idx < 0) {
                                prevx[idx_c] = borderValue;
                            } else {
                                prevx[idx_c] = vop(srow1[prev_idx * nc + idx_c],
                                                   vop(srow2 ? srow2[prev_idx * nc + idx_c] : borderValue,
                                                       srow0 ? srow0[prev_idx * nc + idx_c] : borderValue));
                            }

                            if (next_idx >= width) {
                                nextx[idx_c] = borderValue;
                            } else {
                                nextx[idx_c] = vop(vop(srow2 ? srow2[next_idx * nc + idx_c] : borderValue,
                                                       srow0 ? srow0[next_idx * nc + idx_c] : borderValue),
                                                   srow1[next_idx * nc + idx_c]);
                            }

                            drow[cur_idx * nc + idx_c] = vop(prevx[idx_c], vop(currx[idx_c], nextx[idx_c]));
                        }
                    }
                    float *tmp_ptr = prevx;
                    prevx          = currx;
                    currx          = nextx;
                    nextx          = tmp_ptr;
                }
            }
        } break;
        case 5:
            for (int32_t y = startY; y < endY; ++y) {
                const float *srow2 = getRowPtr(srcBase, srcStride, y);
                const float *srow0 = (y == 0 || y == 1) ? nullptr : srow2 - 2 * srcStride;
                const float *srow1 = y == 0 ? nullptr : srow2 - 1 * srcStride;
                const float *srow3 = (y + 1) == height ? nullptr : srow2 + 1 * srcStride;
                const float *srow4 = ((y + 2) == height || (y + 1) == height) ? nullptr : srow2 + 2 * srcStride;
                float *drow        = getRowPtr(dstBase, dstStride, y);

                int32_t x = startX;

                float temp_buffer[20] = {0.0f};
                float *prev0x         = temp_buffer;
                float *prev1x         = temp_buffer + 4;
                float *currx          = temp_buffer + 8;
                float *next0x         = temp_buffer + 12;
                float *next1x         = temp_buffer + 16;

                for (; x <= endX; x++) {
                    if (x >= width) break;

                    int32_t cur_idx   = x;
                    int32_t prev0_idx = cur_idx - 2;
                    int32_t prev1_idx = cur_idx - 1;
                    int32_t next0_idx = cur_idx + 1;
                    int32_t next1_idx = cur_idx + 2;

                    if (x > startX) {
                        for (int32_t idx_c = 0; idx_c < nc; idx_c++) {
                            if (next1_idx >= width)
                                next1x[idx_c] = borderValue;
                            else
                                next1x[idx_c] = vop(
                                    vop(srow2[next1_idx * nc + idx_c],
                                        vop(srow1 ? srow1[next1_idx * nc + idx_c] : borderValue, srow0 ? srow0[next1_idx * nc + idx_c] : borderValue)),
                                    vop(srow3 ? srow3[next1_idx * nc + idx_c] : borderValue, srow4 ? srow4[next1_idx * nc + idx_c] : borderValue));

                            drow[cur_idx * nc + idx_c] = vop(vop(prev0x[idx_c], next0x[idx_c]), vop(prev1x[idx_c], vop(currx[idx_c], next1x[idx_c])));
                        }
                    } else {
                        for (int32_t idx_c = 0; idx_c < nc; idx_c++) {
                            currx[idx_c] = vop(
                                vop(srow2[cur_idx * nc + idx_c],
                                    vop(srow1 ? srow1[cur_idx * nc + idx_c] : borderValue, srow0 ? srow0[cur_idx * nc + idx_c] : borderValue)),
                                vop(srow3 ? srow3[cur_idx * nc + idx_c] : borderValue, srow4 ? srow4[cur_idx * nc + idx_c] : borderValue));

                            if (prev0_idx < 0)
                                prev0x[idx_c] = borderValue;
                            else
                                prev0x[idx_c] = vop(
                                    vop(srow2[prev0_idx * nc + idx_c],
                                        vop(srow1 ? srow1[prev0_idx * nc + idx_c] : borderValue, srow0 ? srow0[prev0_idx * nc + idx_c] : borderValue)),
                                    vop(srow3 ? srow3[prev0_idx * nc + idx_c] : borderValue, srow4 ? srow4[prev0_idx * nc + idx_c] : borderValue));

                            if (prev1_idx < 0)
                                prev1x[idx_c] = borderValue;
                            else
                                prev1x[idx_c] = vop(
                                    vop(srow2[prev1_idx * nc + idx_c],
                                        vop(srow1 ? srow1[prev1_idx * nc + idx_c] : borderValue, srow0 ? srow0[prev1_idx * nc + idx_c] : borderValue)),
                                    vop(srow3 ? srow3[prev1_idx * nc + idx_c] : borderValue, srow4 ? srow4[prev1_idx * nc + idx_c] : borderValue));

                            if (next0_idx >= width)
                                next0x[idx_c] = borderValue;
                            else
                                next0x[idx_c] = vop(
                                    vop(srow2[next0_idx * nc + idx_c],
                                        vop(srow1 ? srow1[next0_idx * nc + idx_c] : borderValue, srow0 ? srow0[next0_idx * nc + idx_c] : borderValue)),
                                    vop(srow3 ? srow3[next0_idx * nc + idx_c] : borderValue, srow4 ? srow4[next0_idx * nc + idx_c] : borderValue));

                            if (next1_idx >= width)
                                next1x[idx_c] = borderValue;
                            else
                                next1x[idx_c] = vop(
                                    vop(srow2[next1_idx * nc + idx_c],
                                        vop(srow1 ? srow1[next1_idx * nc + idx_c] : borderValue, srow0 ? srow0[next1_idx * nc + idx_c] : borderValue)),
                                    vop(srow3 ? srow3[next1_idx * nc + idx_c] : borderValue, srow4 ? srow4[next1_idx * nc + idx_c] : borderValue));

                            drow[cur_idx * nc + idx_c] = vop(vop(prev0x[idx_c], next0x[idx_c]), vop(prev1x[idx_c], vop(currx[idx_c], next1x[idx_c])));
                        }
                    }
                    float *tmp_ptr = prev0x;
                    prev0x         = prev1x;
                    prev1x         = currx;
                    currx          = next0x;
                    next0x         = next1x;
                    next1x         = tmp_ptr;
                }
            }
            break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
void morph_f32(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue)
{
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t v_elem         = VLEN / sizeof(float) / nc;
    constexpr int32_t radius_vec_num = (kernel_radius * nc * sizeof(float) + VLEN - 1) / VLEN;
    int32_t y = 0;
    __m128 tcurr, tprev[radius_vec_num], tnext[radius_vec_num];
    __m128 v_border = _mm_set1_ps(borderValue);
    for (; y + kernel_radius < height; ++y) {
        for (int32_t i = 1; i < radius_vec_num; i++) {
            tprev[i] = v_border;
        }
        const float *srow = getRowPtr(srcBase, srcStride, y);
        float *drow       = getRowPtr(dstBase, dstStride, y);
        MorphFirstCol<morphOp, nc, kernel_len>(tcurr, tnext, srow, srcStride, y, height - 1 - y, borderValue);

        int32_t x = v_elem;
        for (; x <= width - v_elem * radius_vec_num; x += v_elem) {
            // shift
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tprev[i - 1] = tprev[i];
            }
            tprev[radius_vec_num - 1] = tcurr;
            tcurr                     = tnext[0];
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tnext[i - 1] = tnext[i];
            }

            MorphRow<morphOp, nc, kernel_len>(tprev, tcurr, tnext, srow + x * nc, srcStride, drow, y, height - 1 - y, x - v_elem, width - 1 - (x - v_elem), borderValue);
            drow += v_elem * nc;
        }

        if (x <= width) {
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tprev[i - 1] = tprev[i];
            }
            tprev[radius_vec_num - 1] = tcurr;
            tcurr                     = tnext[0];
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tnext[i - 1] = tnext[i];
            }

            MorphRowLast<morphOp, nc, kernel_len>(tprev, tcurr, tnext, srow + x * nc, srcStride, drow, y, height - 1 - y, x - v_elem, width - 1 - (x - v_elem), borderValue);
            drow += v_elem * nc;
        }
        int32_t startX = std::max(x, 0), endX = width;
        int32_t startY = y, endY = y + 1;
        morphScalar<morphOp, nc, kernel_len>(height, width, srcStride, srcBase, dstStride, dstBase, startX, endX, startY, endY, border_type, borderValue);
    }
    for (; y < height; ++y) {
        for (int32_t i = 1; i < radius_vec_num; i++) {
            tprev[i] = v_border;
        }
        const float *srow = getRowPtr(srcBase, srcStride, y);
        float *drow       = getRowPtr(dstBase, dstStride, y);
        MorphFirstCol<morphOp, nc, kernel_len>(tcurr, tnext, srow, srcStride, y, height - 1 - y, borderValue);

        int32_t x = v_elem;
        for (; x <= width - v_elem * radius_vec_num; x += v_elem) {
            // shift
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tprev[i - 1] = tprev[i];
            }
            tprev[radius_vec_num - 1] = tcurr;
            tcurr                     = tnext[0];
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tnext[i - 1] = tnext[i];
            }

            MorphRow<morphOp, nc, kernel_len>(tprev, tcurr, tnext, srow + x * nc, srcStride, drow, y, height - 1 - y, x - v_elem, width - 1 - (x - v_elem), borderValue);
            drow += v_elem * nc;
        }

        x -= v_elem;
        int32_t startX = std::max<int32_t>(x, 0), endX = width;
        int32_t startY = y, endY = y + 1;
        morphScalar<morphOp, nc, kernel_len>(height, width, srcStride, srcBase, dstStride, dstBase, startX, endX, startY, endY, border_type, borderValue);
    }
}

template void morph_f32<DilateVecOp, 1, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 1, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 3, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 3, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 4, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<DilateVecOp, 4, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);

template void morph_f32<ErodeVecOp, 1, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 1, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 3, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 3, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 4, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);
template void morph_f32<ErodeVecOp, 4, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type,
    float borderValue);

}
}
} // namespace ppl::cv::x86

