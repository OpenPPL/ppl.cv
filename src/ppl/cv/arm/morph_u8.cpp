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

#include "ppl/cv/arm/arithmetic.h"
#include "ppl/cv/arm/morph.hpp"
#include <stdio.h>
#include <cmath>
#include <float.h>
#include "string.h"
#include <arm_neon.h>
#include "ppl/common/log.h"
#include "common.hpp"

namespace ppl {
namespace cv {
namespace arm {

#define VLEN 16 // 16 bytes = 128 bits for  reg
template <typename T>
inline T *getRowPtr(T *base, int32_t stride, int32_t row)
{
    T *baseRaw = const_cast<T *>(reinterpret_cast<const T *>(base));
    return reinterpret_cast<T *>(baseRaw + row * stride);
}

template <class morphOp, typename T, int32_t nc, int32_t kernel_len>
inline void MorphRow(uint8x16_t &tprev, uint8x16_t &tcurr, uint8x16_t &tnext, const T *srcCenterRow, int32_t srcStride, T *drow, int32_t rowIdx, int32_t rowIdxInv, int32_t colIdx, int32_t colIdxInv, T borderValue = 0)
{
    constexpr int32_t v_elem         = VLEN / sizeof(T);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = (nc * kernel_radius - 1) / v_elem + 1;
    uint8x16_t v_border              = vdupq_n_u8(borderValue);
    morphOp vop;
    switch (kernel_len) {
        case 3: {
            uint8x16_t v_up, v_mid, v_down;
            uint8x16_t t_left, t_mid, t_right;
            v_up   = rowIdx == 0 ? v_border : vld1q_u8(srcCenterRow - srcStride);
            v_mid  = vld1q_u8(srcCenterRow);
            v_down = rowIdxInv == 0 ? v_border : vld1q_u8(srcCenterRow + srcStride);

            tnext   = vop(vop(v_up, v_mid), v_down);
            t_left  = vextq_u8(tprev, tcurr, VLEN / sizeof(T) - nc);
            t_mid   = tcurr;
            t_right = vextq_u8(tcurr, tnext, nc);

            t_mid = vop(vop(t_left, t_right), t_mid);
            vst1q_u8(drow, t_mid);
        } break;
        case 5: {
            uint8x16_t v_up0, v_up1, v_mid, v_down0, v_down1;
            v_up0   = rowIdx < 2 ? v_border : vld1q_u8(srcCenterRow - 2 * srcStride + v_elem * (radius_vec_num - 1));
            v_up1   = rowIdx < 1 ? v_border : vld1q_u8(srcCenterRow - 1 * srcStride + v_elem * (radius_vec_num - 1));
            v_mid   = vld1q_u8(srcCenterRow + v_elem * (radius_vec_num - 1));
            v_down0 = rowIdxInv < 1 ? v_border : vld1q_u8(srcCenterRow + 1 * srcStride + v_elem * (radius_vec_num - 1));
            v_down1 = rowIdxInv < 2 ? v_border : vld1q_u8(srcCenterRow + 2 * srcStride + v_elem * (radius_vec_num - 1));

            tnext = vop(vop(v_up0, vop(v_up1, v_mid)), vop(v_down0, v_down1));
            ;

            uint8x16_t t_left0, t_left1, t_mid, t_right0, t_right1;

            t_left0  = vextq_u8(tprev, tcurr, VLEN / sizeof(T) - 2 * nc);
            t_left1  = vextq_u8(tprev, tcurr, VLEN / sizeof(T) - 1 * nc);
            t_mid    = tcurr;
            t_right0 = vextq_u8(tcurr, tnext, 1 * nc);
            t_right1 = vextq_u8(tcurr, tnext, 2 * nc);

            t_mid = vop(vop(vop(t_left0, t_left1), vop(t_mid, t_right0)), t_right1);
            vst1q_u8(drow, t_mid);
        } break;
        default:
            break;
    }
}

template <class morphOp, typename T, int32_t nc, int32_t kernel_len>
inline void MorphRowLast(uint8x16_t &tprev, uint8x16_t &tcurr, uint8x16_t &tnext, const T *srcCenterRow, int32_t srcStride, T *drow, int32_t rowIdx, int32_t rowIdxInv, int32_t colIdx, int32_t colIdxInv, uint8_t borderValue = 0)
{
    uint8x16_t v_border = vdupq_n_u8(borderValue);

    morphOp vop;
    int32_t bias                     = colIdxInv + 1;
    constexpr int32_t v_elem         = VLEN / sizeof(T);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = (nc * kernel_radius - 1) / v_elem + 1;
    switch (kernel_len) {
        case 3: {
            int32_t lane = bias % 16;
            uint8x16_t v_up, v_mid, v_down, t_last;
            v_up   = rowIdx < 1 ? v_border : vld1q_u8(srcCenterRow - srcStride);
            v_mid  = vld1q_u8(srcCenterRow + v_elem * (radius_vec_num - 1));
            v_down = rowIdxInv < 1 ? v_border : vld1q_u8(srcCenterRow + srcStride);

            t_last = vop(vop(v_up, v_mid), v_down);
            switch (lane) {
                case 1: {
                    t_last = vextq_u8(v_border, t_last, 1);
                    t_last = vextq_u8(t_last, v_border, 15);
                } break;
                case 2: {
                    t_last = vextq_u8(v_border, t_last, 2);
                    t_last = vextq_u8(t_last, v_border, 14);
                } break;
                case 3: {
                    t_last = vextq_u8(v_border, t_last, 3);
                    t_last = vextq_u8(t_last, v_border, 13);
                } break;
                case 4: {
                    t_last = vextq_u8(v_border, t_last, 4);
                    t_last = vextq_u8(t_last, v_border, 12);
                } break;
                case 5: {
                    t_last = vextq_u8(v_border, t_last, 5);
                    t_last = vextq_u8(t_last, v_border, 11);
                } break;
                case 6: {
                    t_last = vextq_u8(v_border, t_last, 6);
                    t_last = vextq_u8(t_last, v_border, 10);
                } break;
                case 7: {
                    t_last = vextq_u8(v_border, t_last, 7);
                    t_last = vextq_u8(t_last, v_border, 9);
                } break;
                case 8: {
                    t_last = vextq_u8(v_border, t_last, 8);
                    t_last = vextq_u8(t_last, v_border, 8);
                } break;
                case 9: {
                    t_last = vextq_u8(v_border, t_last, 9);
                    t_last = vextq_u8(t_last, v_border, 7);
                } break;
                case 10: {
                    t_last = vextq_u8(v_border, t_last, 10);
                    t_last = vextq_u8(t_last, v_border, 6);
                } break;
                case 11: {
                    t_last = vextq_u8(v_border, t_last, 11);
                    t_last = vextq_u8(t_last, v_border, 5);
                } break;
                case 12: {
                    t_last = vextq_u8(v_border, t_last, 12);
                    t_last = vextq_u8(t_last, v_border, 4);
                } break;
                case 13: {
                    t_last = vextq_u8(v_border, t_last, 13);
                    t_last = vextq_u8(t_last, v_border, 3);
                } break;
                case 14: {
                    t_last = vextq_u8(v_border, t_last, 14);
                    t_last = vextq_u8(t_last, v_border, 2);
                } break;
                case 15: {
                    t_last = vextq_u8(v_border, t_last, 15);
                    t_last = vextq_u8(t_last, v_border, 1);
                } break;
                case 0: {
                } break;
                default: break;
            }
            uint8x16_t t_left, t_mid, t_right, t_res;
            t_left  = vextq_u8(tcurr, tnext, VLEN / sizeof(T) - nc);
            t_mid   = tnext;
            t_right = vextq_u8(tnext, t_last, nc);
            t_res   = vop(vop(t_left, t_right), t_mid);
            vst1q_u8(drow, t_res);

            t_left  = vextq_u8(tnext, t_last, VLEN / sizeof(T) - nc);
            t_mid   = t_last;
            t_right = vextq_u8(t_last, v_border, nc);

            t_mid = vop(vop(t_left, t_mid), t_right);

            switch (lane) {
                case 1: {
                    t_res = vextq_u8(t_res, t_mid, 1);
                } break;
                case 2: {
                    t_res = vextq_u8(t_res, t_mid, 2);
                } break;
                case 3: {
                    t_res = vextq_u8(t_res, t_mid, 3);
                } break;
                case 4: {
                    t_res = vextq_u8(t_res, t_mid, 4);
                } break;
                case 5: {
                    t_res = vextq_u8(t_res, t_mid, 5);
                } break;
                case 6: {
                    t_res = vextq_u8(t_res, t_mid, 6);
                } break;
                case 7: {
                    t_res = vextq_u8(t_res, t_mid, 7);
                } break;
                case 8: {
                    t_res = vextq_u8(t_res, t_mid, 8);
                } break;
                case 9: {
                    t_res = vextq_u8(t_res, t_mid, 9);
                } break;
                case 10: {
                    t_res = vextq_u8(t_res, t_mid, 10);
                } break;
                case 11: {
                    t_res = vextq_u8(t_res, t_mid, 11);
                } break;
                case 12: {
                    t_res = vextq_u8(t_res, t_mid, 12);
                } break;
                case 13: {
                    t_res = vextq_u8(t_res, t_mid, 13);
                } break;
                case 14: {
                    t_res = vextq_u8(t_res, t_mid, 14);
                } break;
                case 15: {
                    t_res = vextq_u8(t_res, t_mid, 15);
                } break;
                case 0: {
                    t_res = t_mid;
                    lane  = 16;
                } break;
                default: break;
            }
            vst1q_u8(drow + lane, t_res);

        } break;
        case 5: {
            int32_t lane = bias % 16;
            uint8x16_t v_up0, v_up1, v_mid, v_down0, v_down1, t_last;
            v_up0   = rowIdx < 2 ? v_border : vld1q_u8(srcCenterRow - 2 * srcStride);
            v_up1   = rowIdx < 1 ? v_border : vld1q_u8(srcCenterRow - 1 * srcStride);
            v_mid   = vld1q_u8(srcCenterRow + v_elem * (radius_vec_num - 1));
            v_down0 = rowIdxInv < 1 ? v_border : vld1q_u8(srcCenterRow + 1 * srcStride);
            v_down1 = rowIdxInv < 2 ? v_border : vld1q_u8(srcCenterRow + 2 * srcStride);

            t_last = vop(vop(vop(v_up1, v_up0), vop(v_down0, v_down1)), v_mid);
            switch (lane) {
                case 1: {
                    t_last = vextq_u8(v_border, t_last, 1);
                    t_last = vextq_u8(t_last, v_border, 15);
                } break;
                case 2: {
                    t_last = vextq_u8(v_border, t_last, 2);
                    t_last = vextq_u8(t_last, v_border, 14);
                } break;
                case 3: {
                    t_last = vextq_u8(v_border, t_last, 3);
                    t_last = vextq_u8(t_last, v_border, 13);
                } break;
                case 4: {
                    t_last = vextq_u8(v_border, t_last, 4);
                    t_last = vextq_u8(t_last, v_border, 12);
                } break;
                case 5: {
                    t_last = vextq_u8(v_border, t_last, 5);
                    t_last = vextq_u8(t_last, v_border, 11);
                } break;
                case 6: {
                    t_last = vextq_u8(v_border, t_last, 6);
                    t_last = vextq_u8(t_last, v_border, 10);
                } break;
                case 7: {
                    t_last = vextq_u8(v_border, t_last, 7);
                    t_last = vextq_u8(t_last, v_border, 9);
                } break;
                case 8: {
                    t_last = vextq_u8(v_border, t_last, 8);
                    t_last = vextq_u8(t_last, v_border, 8);
                } break;
                case 9: {
                    t_last = vextq_u8(v_border, t_last, 9);
                    t_last = vextq_u8(t_last, v_border, 7);
                } break;
                case 10: {
                    t_last = vextq_u8(v_border, t_last, 10);
                    t_last = vextq_u8(t_last, v_border, 6);
                } break;
                case 11: {
                    t_last = vextq_u8(v_border, t_last, 11);
                    t_last = vextq_u8(t_last, v_border, 5);
                } break;
                case 12: {
                    t_last = vextq_u8(v_border, t_last, 12);
                    t_last = vextq_u8(t_last, v_border, 4);
                } break;
                case 13: {
                    t_last = vextq_u8(v_border, t_last, 13);
                    t_last = vextq_u8(t_last, v_border, 3);
                } break;
                case 14: {
                    t_last = vextq_u8(v_border, t_last, 14);
                    t_last = vextq_u8(t_last, v_border, 2);
                } break;
                case 15: {
                    t_last = vextq_u8(v_border, t_last, 15);
                    t_last = vextq_u8(t_last, v_border, 1);
                } break;
                case 0: {
                } break;
                default: break;
            }
            uint8x16_t t_left0, t_left1, t_mid, t_right0, t_right1, t_res;

            t_left0  = vextq_u8(tcurr, tnext, VLEN / sizeof(T) - 2 * nc);
            t_left1  = vextq_u8(tcurr, tnext, VLEN / sizeof(T) - 1 * nc);
            t_mid    = tnext;
            t_right0 = vextq_u8(tnext, t_last, 1 * nc);
            t_right1 = vextq_u8(tnext, t_last, 2 * nc);

            t_res = vop(vop(vop(t_left0, t_left1), vop(t_mid, t_right0)), t_right1);
            vst1q_u8(drow, t_res);

            t_left0  = vextq_u8(tnext, t_last, VLEN / sizeof(T) - 2 * nc);
            t_left1  = vextq_u8(tnext, t_last, VLEN / sizeof(T) - 1 * nc);
            t_mid    = t_last;
            t_right0 = vextq_u8(t_last, v_border, 1 * nc);
            t_right1 = vextq_u8(t_last, v_border, 2 * nc);
            // p(t_last)

            t_mid = vop(vop(vop(t_left0, t_left1), vop(t_mid, t_right0)), t_right1);

            switch (lane) {
                case 1: {
                    t_res = vextq_u8(t_res, t_mid, 1);
                } break;
                case 2: {
                    t_res = vextq_u8(t_res, t_mid, 2);
                } break;
                case 3: {
                    t_res = vextq_u8(t_res, t_mid, 3);
                } break;
                case 4: {
                    t_res = vextq_u8(t_res, t_mid, 4);
                } break;
                case 5: {
                    t_res = vextq_u8(t_res, t_mid, 5);
                } break;
                case 6: {
                    t_res = vextq_u8(t_res, t_mid, 6);
                } break;
                case 7: {
                    t_res = vextq_u8(t_res, t_mid, 7);
                } break;
                case 8: {
                    t_res = vextq_u8(t_res, t_mid, 8);
                } break;
                case 9: {
                    t_res = vextq_u8(t_res, t_mid, 9);
                } break;
                case 10: {
                    t_res = vextq_u8(t_res, t_mid, 10);
                } break;
                case 11: {
                    t_res = vextq_u8(t_res, t_mid, 11);
                } break;
                case 12: {
                    t_res = vextq_u8(t_res, t_mid, 12);
                } break;
                case 13: {
                    t_res = vextq_u8(t_res, t_mid, 13);
                } break;
                case 14: {
                    t_res = vextq_u8(t_res, t_mid, 14);
                } break;
                case 15: {
                    t_res = vextq_u8(t_res, t_mid, 15);
                } break;
                case 0: {
                    t_res = t_mid;
                    lane  = 16;
                } break;
                default: break;
            }
            vst1q_u8(drow + lane, t_res);
        } break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
inline void MorphFirstCol(uint8x16_t &tcurr, uint8x16_t &tnext, const uint8_t *srcCenterRow, int32_t srcStride, int32_t rowIdx, int32_t rowIdxInv, uint8_t borderValue = 0)
{
    uint8x16_t v_border = vdupq_n_u8(borderValue);
    morphOp vop;
    switch (kernel_len) {
        case 3: {
            uint8x16_t v_up, v_mid, v_down;
            v_up   = rowIdx < 1 ? v_border : vld1q_u8(srcCenterRow - srcStride);
            v_mid  = vld1q_u8(srcCenterRow);
            v_down = rowIdxInv < 1 ? v_border : vld1q_u8(srcCenterRow + srcStride);
            tnext  = vop(vop(v_up, v_mid), v_down);
            tcurr  = v_border;
        } break;
        case 5: {
            uint8x16_t v_up0, v_up1, v_mid, v_down0, v_down1;

            v_up0   = rowIdx < 2 ? v_border : vld1q_u8(srcCenterRow - 2 * srcStride);
            v_up1   = rowIdx < 1 ? v_border : vld1q_u8(srcCenterRow - 1 * srcStride);
            v_mid   = vld1q_u8(srcCenterRow);
            v_down0 = rowIdxInv < 1 ? v_border : vld1q_u8(srcCenterRow + 1 * srcStride);
            v_down1 = rowIdxInv < 2 ? v_border : vld1q_u8(srcCenterRow + 2 * srcStride);
            tnext   = vop(vop(v_up0, vop(v_up1, v_mid)), vop(v_down0, v_down1));
            tcurr   = v_border;
        } break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
void morph_u8(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue)
{
    constexpr int32_t v_elem         = VLEN / sizeof(uint8_t);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = (nc * kernel_radius - 1) / v_elem + 1;

    uint8x16_t tcurr, tprev, tnext;
    uint8x16_t v_border = vdupq_n_u8(borderValue);

    for (int32_t y = 0; y < height; y++) {
        const uint8_t *srow = getRowPtr(srcBase, srcStride, y);
        uint8_t *drow       = getRowPtr(dstBase, dstStride, y);
        MorphFirstCol<morphOp, nc, kernel_len>(tcurr, tnext, srow, srcStride, y, height - 1 - y, borderValue);
        tprev     = v_border;
        int32_t x = v_elem;
        for (int32_t i = 0; i < nc; i++) {
            prefetch(srow + (i + 1) * srcStride);
            prefetch(srow - (i + 1) * srcStride);
        }
        prefetch(srow);
        prefetch(drow);
        for (; x < width * nc - v_elem * radius_vec_num; x += v_elem) {
            tprev = tcurr;
            tcurr = tnext;
            MorphRow<morphOp, uint8_t, nc, kernel_len>(tprev, tcurr, tnext, srow + x, srcStride, drow, y, height - 1 - y, x - v_elem, width * nc - 1 - (x - v_elem), borderValue);
            drow += v_elem;
        }
        MorphRowLast<morphOp, uint8_t, nc, kernel_len>(tprev, tcurr, tnext, srow + x, srcStride, drow, y, height - 1 - y, x - v_elem, width * nc - 1 - (x - v_elem), borderValue);
    }
}

template void morph_u8<DilateVecOp, 1, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<DilateVecOp, 1, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<DilateVecOp, 3, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<DilateVecOp, 3, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<DilateVecOp, 4, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<DilateVecOp, 4, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);

template void morph_u8<ErodeVecOp, 1, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<ErodeVecOp, 1, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<ErodeVecOp, 3, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<ErodeVecOp, 3, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<ErodeVecOp, 4, 3>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
template void morph_u8<ErodeVecOp, 4, 5>(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type,
    uint8_t borderValue);
}
}
} // namespace ppl::cv::arm
