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
inline void MorphRow(float32x4_t *tprev, float32x4_t &tcurr, float32x4_t *tnext, const T *srcCenterRow, int32_t srcStride, T *drow, int32_t rowIdx, int32_t rowIdxInv, int32_t colIdx, int32_t colIdxInv, T borderValue = 0)
{
    constexpr int32_t v_elem         = VLEN / sizeof(T);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = (nc * kernel_radius - 1) / v_elem + 1;

    float32x4_t v_border = vdupq_n_f32(borderValue);
    morphOp vop;
    switch (kernel_len) {
        case 3: {
            float32x4_t v_up, v_mid, v_down;
            float32x4_t t_left, t_mid, t_right;
            v_up   = rowIdx == 0 ? v_border : vld1q_f32(srcCenterRow - srcStride);
            v_mid  = vld1q_f32(srcCenterRow);
            v_down = rowIdxInv == 0 ? v_border : vld1q_f32(srcCenterRow + srcStride);

            tnext[0] = vop(vop(v_up, v_mid), v_down);
            if (nc == 4) {
                t_left  = tprev[0];
                t_mid   = tcurr;
                t_right = tnext[0];
            } else if (nc == 3) {
                t_left  = vextq_f32(tprev[0], tcurr, VLEN / sizeof(T) - 3);
                t_mid   = tcurr;
                t_right = vextq_f32(tcurr, tnext[0], 3);
            } else if (nc == 1) {
                t_left  = vextq_f32(tprev[0], tcurr, VLEN / sizeof(T) - 1);
                t_mid   = tcurr;
                t_right = vextq_f32(tcurr, tnext[0], 1);
            }

            t_mid = vop(vop(t_left, t_mid), t_right);
            vst1q_f32(drow, t_mid);
        } break;
        case 5: {
            float32x4_t v_up0, v_up1, v_mid, v_down0, v_down1;
            v_up0   = rowIdx < 2 ? v_border : vld1q_f32(srcCenterRow - 2 * srcStride + v_elem * (radius_vec_num - 1));
            v_up1   = rowIdx < 1 ? v_border : vld1q_f32(srcCenterRow - 1 * srcStride + v_elem * (radius_vec_num - 1));
            v_mid   = vld1q_f32(srcCenterRow + v_elem * (radius_vec_num - 1));
            v_down0 = rowIdxInv < 1 ? v_border : vld1q_f32(srcCenterRow + 1 * srcStride + v_elem * (radius_vec_num - 1));
            v_down1 = rowIdxInv < 2 ? v_border : vld1q_f32(srcCenterRow + 2 * srcStride + v_elem * (radius_vec_num - 1));

            tnext[radius_vec_num - 1] = vop(vop(v_up0, vop(v_up1, v_mid)), vop(v_down0, v_down1));
            ;

            float32x4_t t_left0, t_left1, t_mid, t_right0, t_right1;

            if (nc == 4) {
                t_left0  = tprev[0];
                t_left1  = tprev[1];
                t_mid    = tcurr;
                t_right0 = colIdxInv < 4 ? v_border : tnext[0];
                t_right1 = colIdxInv < 4 ? v_border : tnext[1];
            } else if (nc == 3) {
                t_left0  = vextq_f32(tprev[0], tprev[1], 2);
                t_left1  = vextq_f32(tprev[1], tcurr, 1);
                t_mid    = tcurr;
                t_right0 = vextq_f32(tcurr, tnext[0], 3);
                t_right1 = vextq_f32(tnext[0], tnext[1], 2);
            } else {
                t_left0  = vextq_f32(tprev[0], tcurr, 2);
                t_left1  = vextq_f32(tprev[0], tcurr, 3);
                t_mid    = tcurr;
                t_right0 = vextq_f32(tcurr, tnext[0], 1);
                t_right1 = vextq_f32(tcurr, tnext[0], 2);
            }

            t_mid = vop(vop(vop(t_left0, t_left1), t_mid), vop(t_right0, t_right1));

            vst1q_f32(drow, t_mid);
        } break;
        default:
            break;
    }
}

template <class morphOp, typename T, int32_t nc, int32_t kernel_len>
inline void MorphRowLast(float32x4_t *tprev, float32x4_t &tcurr, float32x4_t *tnext, const T *srcCenterRow, int32_t srcStride, T *drow, int32_t rowIdx, int32_t rowIdxInv, int32_t colIdx, int32_t colIdxInv, float borderValue = 0)
{
    float32x4_t v_border = vdupq_n_f32(borderValue);
    morphOp vop;
    constexpr int32_t v_elem         = VLEN / sizeof(T);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = (nc * kernel_radius - 1) / v_elem + 1;
    int32_t bias                     = colIdxInv + 1;
    switch (kernel_len) {
        case 3: {
            int32_t lane = bias % 4;
            float32x4_t v_up, v_mid, v_down, t_last;
            v_up   = rowIdx < 1 ? v_border : vld1q_f32(srcCenterRow - srcStride);
            v_mid  = vld1q_f32(srcCenterRow + v_elem * (radius_vec_num - 1));
            v_down = rowIdxInv < 1 ? v_border : vld1q_f32(srcCenterRow + srcStride);

            t_last = vop(vop(v_up, v_mid), v_down);

            if (lane == 3) t_last = vsetq_lane_f32(borderValue, t_last, 3);
            if (lane == 2) {
                t_last = vsetq_lane_f32(borderValue, t_last, 3);
                t_last = vsetq_lane_f32(borderValue, t_last, 2);
            };
            if (lane == 1) {
                t_last = vsetq_lane_f32(borderValue, t_last, 3);
                t_last = vsetq_lane_f32(borderValue, t_last, 2);
                t_last = vsetq_lane_f32(borderValue, t_last, 1);
            };
            float32x4_t t_left, t_mid, t_right, t_res;
            if (nc == 4) {
                t_left  = tcurr;
                t_mid   = tnext[0];
                t_right = t_last;
            } else if(nc == 3){
                t_left  = vextq_f32(tcurr, tnext[0], VLEN / sizeof(T) - 3);
                t_mid   = tnext[0];
                t_right = vextq_f32(tnext[0], t_last, 3);
            } else if(nc == 1) {
                t_left  = vextq_f32(tcurr, tnext[0], VLEN / sizeof(T) - 1);
                t_mid   = tnext[0];
                t_right = vextq_f32(tnext[0], t_last, 1);
            }

            t_res = vop(vop(t_left, t_mid), t_right);
            vst1q_f32(drow, t_res);
            if (nc == 4) {
                t_left  = tnext[0];
                t_mid   = t_last;
                t_right = v_border;
            } else if (nc == 3){
                t_left  = vextq_f32(tnext[0], t_last, VLEN / sizeof(T) - 3);
                t_mid   = t_last;
                t_right = vextq_f32(t_last, v_border, 3);
            } else if (nc == 1) {
                t_left  = vextq_f32(tnext[0], t_last, VLEN / sizeof(T) - 1);
                t_mid   = t_last;
                t_right = vextq_f32(t_last, v_border, 1);
            }

            t_mid = vop(vop(t_left, t_mid), t_right);

            if (lane == 3) t_res = vextq_f32(t_res, t_mid, 3);
            if (lane == 2) t_res = vextq_f32(t_res, t_mid, 2);
            if (lane == 1) t_res = vextq_f32(t_res, t_mid, 1);
            if (lane == 0) {
                t_res = t_mid;
                lane  = 4;
            }
            vst1q_f32(drow + lane, t_res);
        } break;
        case 5: {
            int32_t lane = bias % 4;
            float32x4_t v_up0, v_up1, v_mid, v_down0, v_down1, t_last;
            v_up0   = rowIdx < 2 ? v_border : vld1q_f32(srcCenterRow - 2 * srcStride + v_elem * (radius_vec_num - 1));
            v_up1   = rowIdx < 1 ? v_border : vld1q_f32(srcCenterRow - 1 * srcStride + v_elem * (radius_vec_num - 1));
            v_mid   = vld1q_f32(srcCenterRow + v_elem * (radius_vec_num - 1));
            v_down0 = rowIdxInv < 1 ? v_border : vld1q_f32(srcCenterRow + 1 * srcStride + v_elem * (radius_vec_num - 1));
            v_down1 = rowIdxInv < 2 ? v_border : vld1q_f32(srcCenterRow + 2 * srcStride + v_elem * (radius_vec_num - 1));

            t_last = vop(vop(v_up0, vop(v_up1, v_mid)), vop(v_down0, v_down1));
            ;
            if (lane == 3) t_last = vsetq_lane_f32(borderValue, t_last, 3);
            if (lane == 2) {
                t_last = vsetq_lane_f32(borderValue, t_last, 3);
                t_last = vsetq_lane_f32(borderValue, t_last, 2);
            };
            if (lane == 1) {
                t_last = vsetq_lane_f32(borderValue, t_last, 3);
                t_last = vsetq_lane_f32(borderValue, t_last, 2);
                t_last = vsetq_lane_f32(borderValue, t_last, 1);
            };
            float32x4_t t_left0, t_left1, t_mid, t_right0, t_right1, t_res;

            float32x4_t vnext[4] = {t_last, v_border, v_border, v_border};
            for (int32_t j = 0; j < 1 + radius_vec_num; j++) {
                for (int32_t i = 1; i < radius_vec_num; i++) {
                    tprev[i - 1] = tprev[i];
                }
                tprev[radius_vec_num - 1] = tcurr;
                tcurr                     = tnext[0];
                for (int32_t i = 1; i < radius_vec_num; i++) {
                    tnext[i - 1] = tnext[i];
                }
                tnext[radius_vec_num - 1] = vnext[j];
                {
                    if (nc == 4) {
                        t_left0  = tprev[0];
                        t_left1  = tprev[1];
                        t_mid    = tcurr;
                        t_right0 = tnext[0];
                        t_right1 = tnext[1];
                    } else if (nc == 3) {
                        t_left0  = vextq_f32(tprev[0], tprev[1], 2);
                        t_left1  = vextq_f32(tprev[1], tcurr, 1);
                        t_mid    = tcurr;
                        t_right0 = vextq_f32(tcurr, tnext[0], 3);
                        t_right1 = vextq_f32(tnext[0], tnext[1], 2);
                    } else {
                        t_left0  = vextq_f32(tprev[0], tcurr, 2);
                        t_left1  = vextq_f32(tprev[0], tcurr, 3);
                        t_mid    = tcurr;
                        t_right0 = vextq_f32(tcurr, tnext[0], 1);
                        t_right1 = vextq_f32(tcurr, tnext[0], 2);
                    }
                    t_mid = vop(vop(vop(t_left0, t_left1), t_mid), vop(t_right0, t_right1));
                }
                if (j != radius_vec_num) {
                    t_res = t_mid;
                    vst1q_f32(drow, t_res);
                    drow += v_elem;
                } else {
                    if (lane == 3) t_res = vextq_f32(t_res, t_mid, 3);
                    if (lane == 2) t_res = vextq_f32(t_res, t_mid, 2);
                    if (lane == 1) t_res = vextq_f32(t_res, t_mid, 1);
                    if (lane == 0) {
                        t_res = t_mid;
                        lane  = 4;
                    }
                    vst1q_f32(drow - v_elem + lane, t_res);
                }
            }
        } break;
        default:
            break;
    }
}

template <class morphOp, int32_t nc, int32_t kernel_len>
inline void MorphFirstCol(float32x4_t &tcurr, float32x4_t *tnext, const float *srcCenterRow, int32_t srcStride, int32_t rowIdx, int32_t rowIdxInv, float borderValue = 0)
{
    constexpr int32_t v_elem         = VLEN / sizeof(float32_t);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = (nc * kernel_radius - 1) / v_elem + 1;

    float32x4_t v_border = vdupq_n_f32(borderValue);
    morphOp vop;
    switch (kernel_len) {
        case 3: {
            float32x4_t v_up, v_mid, v_down;
            v_up     = rowIdx < 1 ? v_border : vld1q_f32(srcCenterRow - srcStride);
            v_mid    = vld1q_f32(srcCenterRow);
            v_down   = rowIdxInv < 1 ? v_border : vld1q_f32(srcCenterRow + srcStride);
            tnext[0] = vop(vop(v_up, v_mid), v_down);
            tcurr    = v_border;
        } break;
        case 5: {
            float32x4_t v_up0, v_up1, v_mid, v_down0, v_down1;

            for (int32_t i = 0; i < radius_vec_num; i++) {
                v_up0    = rowIdx < 2 ? v_border : vld1q_f32(srcCenterRow - 2 * srcStride + i * v_elem);
                v_up1    = rowIdx < 1 ? v_border : vld1q_f32(srcCenterRow - 1 * srcStride + i * v_elem);
                v_mid    = vld1q_f32(srcCenterRow + i * v_elem);
                v_down0  = rowIdxInv < 1 ? v_border : vld1q_f32(srcCenterRow + 1 * srcStride + i * v_elem);
                v_down1  = rowIdxInv < 2 ? v_border : vld1q_f32(srcCenterRow + 2 * srcStride + i * v_elem);
                tnext[i] = vop(vop(v_up0, vop(v_up1, v_mid)), vop(v_down0, v_down1));
            }
            tcurr = v_border;
        } break;
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
    constexpr int32_t v_elem         = VLEN / sizeof(float);
    constexpr int32_t kernel_radius  = (kernel_len - 1) / 2;
    constexpr int32_t radius_vec_num = (nc * kernel_radius - 1) / v_elem + 1;
    float32x4_t tcurr, tprev[radius_vec_num], tnext[radius_vec_num];
    float32x4_t v_border = vdupq_n_f32(borderValue);

    for (int32_t y = 0; y < height; y++) {
        const float *srow = getRowPtr(srcBase, srcStride, y);
        float *drow       = getRowPtr(dstBase, dstStride, y);
        MorphFirstCol<morphOp, nc, kernel_len>(tcurr, tnext, srow, srcStride, y, height - 1 - y, borderValue);
        tprev[radius_vec_num - 1] = v_border;
        int32_t x                 = v_elem;
        for (int32_t i = 0; i < nc; i++) {
            prefetch(srow + (i + 1) * srcStride);
            prefetch(srow - (i + 1) * srcStride);
        }
        prefetch(srow);
        prefetch(drow);
        for (; x < width * nc - v_elem * radius_vec_num; x += v_elem) {
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tprev[i - 1] = tprev[i];
            }
            tprev[radius_vec_num - 1] = tcurr;
            tcurr                     = tnext[0];
            for (int32_t i = 1; i < radius_vec_num; i++) {
                tnext[i - 1] = tnext[i];
            }
            MorphRow<morphOp, float, nc, kernel_len>(tprev, tcurr, tnext, srow + x, srcStride, drow, y, height - 1 - y, x - v_elem, width * nc - 1 - (x - v_elem), borderValue);
            drow += v_elem;
        }
        MorphRowLast<morphOp, float, nc, kernel_len>(tprev, tcurr, tnext, srow + x, srcStride, drow, y, height - 1 - y, x - v_elem, width * nc - 1 - (x - v_elem), borderValue);
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
} // namespace ppl::cv::arm
