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

#define BLOCK_SIZE (64)
#define INTER_TABLE_BITS (5)
#define INTER_TABLE_SIZE (32)

namespace ppl::cv::arm {

template <typename T>
static inline T clip(T x, T a, T b)
{
    return std::max(a, std::min(x, b));
}

void WarpPerspective_CoordCompute_Nearest_Line(const double *M, int16_t *coord_map, int bw, double X0, double Y0, double W0) {
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

            vCoordX0123i = vcvtaq_s32_f32(v_X01f);
            vCoordY0123i = vcvtaq_s32_f32(v_Y01f);
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

            vCoordX4567i = vcvtaq_s32_f32(v_X01f);
            vCoordY4567i = vcvtaq_s32_f32(v_Y01f);
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

            vCoordX89abi = vcvtaq_s32_f32(v_X01f);
            vCoordY89abi = vcvtaq_s32_f32(v_Y01f);
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

            vCoordXcdefi = vcvtaq_s32_f32(v_X01f);
            vCoordYcdefi = vcvtaq_s32_f32(v_Y01f);
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
        W = W ? 1./W : 0;
        double X = std::max((double)SHRT_MIN, std::min((double)SHRT_MAX, (X0 + M[0] * jj) * W));
        double Y = std::max((double)SHRT_MIN, std::min((double)SHRT_MAX, (Y0 + M[3] * jj) * W));
        int16_t Xh = std::round(X);
        int16_t Yh = std::round(Y);

        coord_map[jj * 2] = Xh;
        coord_map[jj * 2 + 1] = Yh;
    }
}

void WarpPerspective_CoordCompute_Linear_Line(const double *M, int16_t *coord_map, int16_t *alpha_map, int bw, double X0, double Y0, double W0) {
    float64x2_t v_baseXd = {0.0, 1.0};
    const float64x2_t v_2d = vdupq_n_f64(2.0);
    const float64x2_t v_W0d_c = vdupq_n_f64(W0);
    const float64x2_t v_X0d_c = vdupq_n_f64(X0);
    const float64x2_t v_Y0d_c = vdupq_n_f64(Y0);
    const float64x2_t v_M0d = vdupq_n_f64(M[0]);
    const float64x2_t v_M3d = vdupq_n_f64(M[3]);
    const float64x2_t v_M6d = vdupq_n_f64(M[6]);
    const float64x2_t v_interTblSzd = vdupq_n_f64((double)INTER_TABLE_SIZE);
    const int32x4_t v_interTblMaski = vdupq_n_s32(INTER_TABLE_SIZE - 1);
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
            v_X0d = vmulq_f64(vdivq_f64(v_X0d, v_W0d), v_interTblSzd);
            v_Y0d = vmulq_f64(vdivq_f64(v_Y0d, v_W0d), v_interTblSzd);
            v_X1d = vmulq_f64(vdivq_f64(v_X1d, v_W1d), v_interTblSzd);
            v_Y1d = vmulq_f64(vdivq_f64(v_Y1d, v_W1d), v_interTblSzd);

            // assume that the number won't overflow float
            // this almost always holds and permits less clip
            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_intminf), v_intmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_intminf), v_intmaxf);

            vCoordX0123i = vcvtaq_s32_f32(v_X01f);
            vCoordY0123i = vcvtaq_s32_f32(v_Y01f);
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

            v_X0d = vmulq_f64(vdivq_f64(v_X0d, v_W0d), v_interTblSzd);
            v_Y0d = vmulq_f64(vdivq_f64(v_Y0d, v_W0d), v_interTblSzd);
            v_X1d = vmulq_f64(vdivq_f64(v_X1d, v_W1d), v_interTblSzd);
            v_Y1d = vmulq_f64(vdivq_f64(v_Y1d, v_W1d), v_interTblSzd);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_intminf), v_intmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_intminf), v_intmaxf);

            vCoordX4567i = vcvtaq_s32_f32(v_X01f);
            vCoordY4567i = vcvtaq_s32_f32(v_Y01f);
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

            v_X0d = vmulq_f64(vdivq_f64(v_X0d, v_W0d), v_interTblSzd);
            v_Y0d = vmulq_f64(vdivq_f64(v_Y0d, v_W0d), v_interTblSzd);
            v_X1d = vmulq_f64(vdivq_f64(v_X1d, v_W1d), v_interTblSzd);
            v_Y1d = vmulq_f64(vdivq_f64(v_Y1d, v_W1d), v_interTblSzd);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_intminf), v_intmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_intminf), v_intmaxf);

            vCoordX89abi = vcvtaq_s32_f32(v_X01f);
            vCoordY89abi = vcvtaq_s32_f32(v_Y01f);
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

            v_X0d = vmulq_f64(vdivq_f64(v_X0d, v_W0d), v_interTblSzd);
            v_Y0d = vmulq_f64(vdivq_f64(v_Y0d, v_W0d), v_interTblSzd);
            v_X1d = vmulq_f64(vdivq_f64(v_X1d, v_W1d), v_interTblSzd);
            v_Y1d = vmulq_f64(vdivq_f64(v_Y1d, v_W1d), v_interTblSzd);

            float32x2_t v_X0f = vcvtx_f32_f64(v_X0d);
            float32x4_t v_X01f = vcvtx_high_f32_f64(v_X0f, v_X1d);
            float32x2_t v_Y0f = vcvtx_f32_f64(v_Y0d);
            float32x4_t v_Y01f = vcvtx_high_f32_f64(v_Y0f, v_Y1d);
            v_X01f = vminq_f32(vmaxq_f32(v_X01f, v_intminf), v_intmaxf);
            v_Y01f = vminq_f32(vmaxq_f32(v_Y01f, v_intminf), v_intmaxf);

            vCoordXcdefi = vcvtaq_s32_f32(v_X01f);
            vCoordYcdefi = vcvtaq_s32_f32(v_Y01f);
        }

        int32x4_t v_Alpha0i = vaddq_s32(vshlq_n_s32(vandq_s32(vCoordY0123i, v_interTblMaski), INTER_TABLE_BITS), vandq_s32(vCoordX0123i, v_interTblMaski));
        int32x4_t v_Alpha1i = vaddq_s32(vshlq_n_s32(vandq_s32(vCoordY4567i, v_interTblMaski), INTER_TABLE_BITS), vandq_s32(vCoordX4567i, v_interTblMaski));
        int32x4_t v_Alpha2i = vaddq_s32(vshlq_n_s32(vandq_s32(vCoordY89abi, v_interTblMaski), INTER_TABLE_BITS), vandq_s32(vCoordX89abi, v_interTblMaski));
        int32x4_t v_Alpha3i = vaddq_s32(vshlq_n_s32(vandq_s32(vCoordYcdefi, v_interTblMaski), INTER_TABLE_BITS), vandq_s32(vCoordXcdefi, v_interTblMaski));
        
        int16x4_t v_Alpha01Halfh = vmovn_s32(v_Alpha0i);
        int16x8_t v_Alpha01h = vmovn_high_s32(v_Alpha01Halfh, v_Alpha1i);
        int16x4_t v_Alpha23Halfh = vmovn_s32(v_Alpha2i);
        int16x8_t v_Alpha23h = vmovn_high_s32(v_Alpha23Halfh, v_Alpha3i);
        
        vst1q_s16(alpha_map + jj, v_Alpha01h);
        vst1q_s16(alpha_map + jj + 8, v_Alpha23h);

        vCoordX0123i = vshrq_n_s32(vCoordX0123i, INTER_TABLE_BITS);
        vCoordX4567i = vshrq_n_s32(vCoordX4567i, INTER_TABLE_BITS);
        vCoordX89abi = vshrq_n_s32(vCoordX89abi, INTER_TABLE_BITS);
        vCoordXcdefi = vshrq_n_s32(vCoordXcdefi, INTER_TABLE_BITS);
        vCoordY0123i = vshrq_n_s32(vCoordY0123i, INTER_TABLE_BITS);
        vCoordY4567i = vshrq_n_s32(vCoordY4567i, INTER_TABLE_BITS);
        vCoordY89abi = vshrq_n_s32(vCoordY89abi, INTER_TABLE_BITS);
        vCoordYcdefi = vshrq_n_s32(vCoordYcdefi, INTER_TABLE_BITS);

        int16x8x2_t vCoord0, vCoord1;
        int16x4_t vCoord0XHalf = vqmovn_s32(vCoordX0123i);
        vCoord0.val[0] = vqmovn_high_s32(vCoord0XHalf, vCoordX4567i);
        int16x4_t vCoord0YHalf = vqmovn_s32(vCoordY0123i);
        vCoord0.val[1] = vqmovn_high_s32(vCoord0YHalf, vCoordY4567i);
        vst2q_s16(coord_map + jj * 2, vCoord0);

        int16x4_t vCoord1XHalf = vqmovn_s32(vCoordX89abi);
        vCoord1.val[0] = vqmovn_high_s32(vCoord1XHalf, vCoordXcdefi);
        int16x4_t vCoord1YHalf = vqmovn_s32(vCoordY89abi);
        vCoord1.val[1] = vqmovn_high_s32(vCoord1YHalf, vCoordYcdefi);
        vst2q_s16(coord_map + jj * 2 + 16, vCoord1);
    }

    for (; jj < bw; jj++) {
        double W = W0 + M[6] * jj;
        W = W ? INTER_TABLE_SIZE/W : 0;
        double Xd = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * jj) * W));
        double Yd = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * jj) * W));
        int X = std::round(Xd);
        int Y = std::round(Yd);

        coord_map[jj * 2] = clip((X >> INTER_TABLE_BITS), SHRT_MIN, SHRT_MAX);
        coord_map[jj * 2 + 1] = clip((Y >> INTER_TABLE_BITS), SHRT_MIN, SHRT_MAX);
        alpha_map[jj] = (short)(((Y & (INTER_TABLE_SIZE-1)) << INTER_TABLE_BITS) + (X & (INTER_TABLE_SIZE-1)));
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
    int16_t *coord_map = (int16_t *)&coord_data[0];
    
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
                WarpPerspective_CoordCompute_Nearest_Line(&M[0][0], coord_map + BLOCK_SIZE * bi * 2, bw, baseX, baseY, baseW);
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

static uint8_t sat_cast(int32_t data)
{
    int32_t val;
    val = data > 255 ? 255 : data;
    val = data < 0 ? 0 : val;
    return val;
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
    int16_t *coord_map = (int16_t *)&coord_data[0];
    int16_t *alpha_map = (int16_t *)&alpha_data[0];
    
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
                WarpPerspective_CoordCompute_Linear_Line(&M[0][0], coord_map + BLOCK_SIZE * bi * 2, alpha_map + BLOCK_SIZE * bi, bw, baseX, baseY, baseW);
            }

            for (int32_t bi = 0; bi < bh; bi++) {
                for (int32_t bj = 0; bj < bw; bj++) {
                    int ii = i + bi;
                    int jj = j + bj;
                    int32_t sx0 = coord_map[bi * BLOCK_SIZE * 2 + bj * 2];
                    int32_t sy0 = coord_map[bi * BLOCK_SIZE * 2 + bj * 2 + 1];
                    int32_t alpha = alpha_map[bi * BLOCK_SIZE + bj];
                    
                    float u = (alpha & (INTER_TABLE_SIZE - 1)) / (1.0 * INTER_TABLE_SIZE);
                    float v = ((alpha >> INTER_TABLE_BITS) & (INTER_TABLE_SIZE - 1)) / (1.0 * INTER_TABLE_SIZE);
                    
                    float tab[4];
                    float taby[2], tabx[2];
                    taby[0] = 1.0f - 1.0f * v;
                    taby[1] = v;
                    tabx[0] = 1.0f - u;
                    tabx[1] = u;

                    tab[0] = taby[0] * tabx[0];
                    tab[1] = taby[0] * tabx[1];
                    tab[2] = taby[1] * tabx[0];
                    tab[3] = taby[1] * tabx[1];

                    int32_t idxDst = ii * outWidthStride + jj * nc;
                    
                    if (borderMode == ppl::cv::BORDER_CONSTANT) {
                        bool flag0 = (sx0 >= 0 && sx0 < inWidth && sy0 >= 0 && sy0 < inHeight);
                        bool flag1 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 >= 0 && sy0 < inHeight);
                        bool flag2 = (sx0 >= 0 && sx0 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                        bool flag3 = (sx0 + 1 >= 0 && sx0 + 1 < inWidth && sy0 + 1 >= 0 && sy0 + 1 < inHeight);
                        for (int32_t k = 0; k < nc; k++) {
                            int32_t position1 = (sy0 * inWidthStride + sx0 * nc);
                            int32_t position2 = ((sy0 + 1) * inWidthStride + sx0 * nc);
                            float v0, v1, v2, v3;
                            v0 = flag0 ? src[position1 + k] : delta;
                            v1 = flag1 ? src[position1 + nc + k] : delta;
                            v2 = flag2 ? src[position2 + k] : delta;
                            v3 = flag3 ? src[position2 + nc + k] : delta;
                            float sum = 0;
                            sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                            // dst[idxDst + k] = static_cast<T>(sum);
                            dst[idxDst + k] = (std::is_same<float, T>::value) ? static_cast<T>(sum) : sat_cast(std::round(sum));
                        }
                    } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                        int32_t sx1 = sx0 + 1;
                        int32_t sy1 = sy0 + 1;
                        sx0 = clip(sx0, 0, inWidth - 1);
                        sx1 = clip(sx1, 0, inWidth - 1);
                        sy0 = clip(sy0, 0, inHeight - 1);
                        sy1 = clip(sy1, 0, inHeight - 1);
                        const T* t0 = src + sy0 * inWidthStride + sx0 * nc;
                        const T* t1 = src + sy0 * inWidthStride + sx1 * nc;
                        const T* t2 = src + sy1 * inWidthStride + sx0 * nc;
                        const T* t3 = src + sy1 * inWidthStride + sx1 * nc;
                        for (int32_t k = 0; k < nc; ++k) {
                            float sum = 0;
                            sum += t0[k] * tab[0] + t1[k] * tab[1] + t2[k] * tab[2] + t3[k] * tab[3];
                            // dst[idxDst + k] = static_cast<T>(sum);
                            dst[idxDst + k] = (std::is_same<float, T>::value) ? static_cast<T>(sum) : sat_cast(std::round(sum));
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
                                float v0, v1, v2, v3;
                                v0 = src[position1 + k];
                                v1 = src[position1 + nc + k];
                                v2 = src[position2 + k];
                                v3 = src[position2 + nc + k];
                                float sum = 0;
                                sum += v0 * tab[0] + v1 * tab[1] + v2 * tab[2] + v3 * tab[3];
                                // dst[idxDst + k] = static_cast<T>(sum);
                                dst[idxDst + k] = (std::is_same<float, T>::value) ? static_cast<T>(sum) : sat_cast(std::round(sum));
                            }
                        } else {
                            continue;
                        }
                    }
                }
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

} // namespace ppl::cv::arm
