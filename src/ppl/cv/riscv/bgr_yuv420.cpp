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

#include "ppl/cv/riscv/cvtcolor.h"

#include "ppl/cv/types.h"
#include "ppl/cv/riscv/util.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/cv/riscv/typetraits.h"

#include <string.h>
#include <cmath>

#include <limits.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace riscv {

///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////

const int32_t CY_coeff = 1220542;
const int32_t CUB_coeff = 2116026;
const int32_t CUG_coeff = -409993;
const int32_t CVG_coeff = -852492;
const int32_t CVR_coeff = 1673527;
const int32_t SHIFT = 20;

// Coefficients for RGB to YUV420p conversion
const int32_t CRY_coeff = 269484;
const int32_t CGY_coeff = 528482;
const int32_t CBY_coeff = 102760;
const int32_t CRU_coeff = -155188;
const int32_t CGU_coeff = -305135;
const int32_t CBU_coeff = 460324;
const int32_t CGV_coeff = -385875;
const int32_t CBV_coeff = -74448;
struct YUV420p2RGB_u8 {
    YUV420p2RGB_u8(int32_t _bIdx)
        : bIdx(_bIdx) {}

    ::ppl::common::RetCode operator()(
        int32_t height,
        int32_t width,
        int32_t yStride,
        const uint8_t *y1,
        int32_t uStride,
        const uint8_t *u1,
        int32_t vStride,
        const uint8_t *v1,
        int32_t outWidthStride,
        uint8_t *dst) const
    {
        if (nullptr == y1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == u1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == v1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst) {
            return ppl::common::RC_INVALID_VALUE;
        }

        const size_t vl = vsetvlmax_e32m4();
        const int32_t num_unroll = vl;

        for (int32_t j = 0; j < height; j += 2, y1 += yStride * 2, u1 += uStride, v1 += vStride) {
            uint8_t *row1 = dst + j * outWidthStride;
            uint8_t *row2 = dst + (j + 1) * outWidthStride;
            const uint8_t *y2 = y1 + yStride;

            for (int32_t i = 0; i < width / 2; i += num_unroll, row1 += 6 * num_unroll, row2 += 6 * num_unroll) {
                const size_t vl = vsetvl_e32m4(width / 2 - i);

                vint32m4_t u_v = vadd_vx_i32m4(vreinterpret_v_u32m4_i32m4(vlbu_v_u32m4((uint32_t *)(u1 + i), vl)), -128, vl);
                vint32m4_t v_v = vadd_vx_i32m4(vreinterpret_v_u32m4_i32m4(vlbu_v_u32m4((uint32_t *)(v1 + i), vl)), -128, vl);

                vint32m4_t ruv_v = vadd_vx_i32m4(vmul_vx_i32m4(v_v, CVR_coeff, vl), (1 << (SHIFT - 1)), vl);
                vint32m4_t guv_v = vadd_vx_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(v_v, CVG_coeff, vl), CUG_coeff, u_v, vl), (1 << (SHIFT - 1)), vl);
                vint32m4_t buv_v = vadd_vx_i32m4(vmul_vx_i32m4(u_v, CUB_coeff, vl), (1 << (SHIFT - 1)), vl);

                vint32m4_t y00_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(y1 + 2 * i + 0), 2 * sizeof(uint8_t), vl));
                vint32m4_t y01_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(y1 + 2 * i + 1), 2 * sizeof(uint8_t), vl));
                vint32m4_t y10_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(y2 + 2 * i + 0), 2 * sizeof(uint8_t), vl));
                vint32m4_t y11_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(y2 + 2 * i + 1), 2 * sizeof(uint8_t), vl));

                y00_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y00_v, -16, vl), 0, vl), CY_coeff, vl);
                y01_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y01_v, -16, vl), 0, vl), CY_coeff, vl);
                y10_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y10_v, -16, vl), 0, vl), CY_coeff, vl);
                y11_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y11_v, -16, vl), 0, vl), CY_coeff, vl);

                vsse8_v_u8m1(row1 + 2 - bIdx, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row1 + 1, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row1 + bIdx, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

                vsse8_v_u8m1(row1 + 5 - bIdx, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row1 + 4, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row1 + 3 + bIdx, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

                vsse8_v_u8m1(row2 + 2 - bIdx, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row2 + 1, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row2 + bIdx, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

                vsse8_v_u8m1(row2 + 5 - bIdx, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row2 + 4, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row2 + 3 + bIdx, 2 * 3 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            }
        }
        return ppl::common::RC_SUCCESS;
    }
    int32_t bIdx;
};

struct YUV420p2RGBA_u8 {
    YUV420p2RGBA_u8(int32_t _bIdx)
        : bIdx(_bIdx) {}

    ::ppl::common::RetCode operator()(
        int32_t height,
        int32_t width,
        int32_t yStride,
        const uint8_t *y1,
        int32_t uStride,
        const uint8_t *u1,
        int32_t vStride,
        const uint8_t *v1,
        int32_t outWidthStride,
        uint8_t *dst) const
    {
        if (nullptr == y1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == u1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == v1) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst) {
            return ppl::common::RC_INVALID_VALUE;
        }

        const size_t vl = vsetvlmax_e32m4();
        const int32_t num_unroll = vl;
        const vuint8m1_t alpha_v = vmv_v_x_u8m1(255, vl);

        for (int32_t j = 0; j < height; j += 2, y1 += yStride * 2, u1 += uStride, v1 += vStride) {
            uint8_t *row1 = dst + j * outWidthStride;
            uint8_t *row2 = dst + (j + 1) * outWidthStride;
            const uint8_t *y2 = y1 + yStride;

            for (int32_t i = 0; i < width / 2; i += num_unroll, row1 += 8 * num_unroll, row2 += 8 * num_unroll) {
                const size_t vl = vsetvl_e32m4(width / 2 - i);

                vint32m4_t u_v = vadd_vx_i32m4(vreinterpret_v_u32m4_i32m4(vlbu_v_u32m4((uint32_t *)(u1 + i), vl)), -128, vl);
                vint32m4_t v_v = vadd_vx_i32m4(vreinterpret_v_u32m4_i32m4(vlbu_v_u32m4((uint32_t *)(v1 + i), vl)), -128, vl);

                vint32m4_t ruv_v = vadd_vx_i32m4(vmul_vx_i32m4(v_v, CVR_coeff, vl), (1 << (SHIFT - 1)), vl);
                vint32m4_t guv_v = vadd_vx_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(v_v, CVG_coeff, vl), CUG_coeff, u_v, vl), (1 << (SHIFT - 1)), vl);
                vint32m4_t buv_v = vadd_vx_i32m4(vmul_vx_i32m4(u_v, CUB_coeff, vl), (1 << (SHIFT - 1)), vl);

                vint32m4_t y00_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(y1 + 2 * i + 0), 2 * sizeof(uint8_t), vl));
                vint32m4_t y01_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(y1 + 2 * i + 1), 2 * sizeof(uint8_t), vl));
                vint32m4_t y10_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(y2 + 2 * i + 0), 2 * sizeof(uint8_t), vl));
                vint32m4_t y11_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(y2 + 2 * i + 1), 2 * sizeof(uint8_t), vl));

                y00_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y00_v, -16, vl), 0, vl), CY_coeff, vl);
                y01_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y01_v, -16, vl), 0, vl), CY_coeff, vl);
                y10_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y10_v, -16, vl), 0, vl), CY_coeff, vl);
                y11_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y11_v, -16, vl), 0, vl), CY_coeff, vl);

                vsse8_v_u8m1(row1 + 2 - bIdx, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row1 + 1, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row1 + bIdx, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

                vsse8_v_u8m1(row1 + 6 - bIdx, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row1 + 5, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row1 + 4 + bIdx, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

                vsse8_v_u8m1(row2 + 2 - bIdx, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row2 + 1, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row2 + bIdx, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

                vsse8_v_u8m1(row2 + 6 - bIdx, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row2 + 5, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(row2 + 4 + bIdx, 2 * 4 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

                vsse8_v_u8m1(row2 + 3, 2 * 4 * sizeof(uint8_t), alpha_v, vl);
                vsse8_v_u8m1(row1 + 3, 2 * 4 * sizeof(uint8_t), alpha_v, vl);
                vsse8_v_u8m1(row2 + 3 + 4, 2 * 4 * sizeof(uint8_t), alpha_v, vl);
                vsse8_v_u8m1(row1 + 3 + 4, 2 * 4 * sizeof(uint8_t), alpha_v, vl);
            }
        }
        return ppl::common::RC_SUCCESS;
    }
    int32_t bIdx;
};

struct RGBtoYUV420p_u8 {
    RGBtoYUV420p_u8(int32_t _bIdx)
        : bIdx(_bIdx) {}

    ::ppl::common::RetCode operator()(
        int32_t height,
        int32_t width,
        int32_t cn,
        int32_t inWidthStride,
        const uint8_t *src,
        int32_t yStride,
        uint8_t *dst_y,
        int32_t uStride,
        uint8_t *dst_u,
        int32_t vStride,
        uint8_t *dst_v) const
    {
        if (nullptr == src) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst_y) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst_u) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == dst_v) {
            return ppl::common::RC_INVALID_VALUE;
        }
        int32_t w = width;
        int32_t h = height;

        const size_t vl = vsetvlmax_e32m4();
        const int32_t num_unroll = vl;

        const int32_t shifted16 = (16 << SHIFT);
        const int32_t shifted128 = (128 << SHIFT);

        for (int32_t i = 0; i < h / 2; i++) {
            const uint8_t *row0 = src + i * 2 * inWidthStride;
            const uint8_t *row1 = src + (i * 2 + 1) * inWidthStride;

            uint8_t *y = dst_y + i * 2 * yStride;
            uint8_t *u = dst_u + i * uStride;
            uint8_t *v = dst_v + i * vStride;

            for (int32_t j = 0, k = 0; j < w * cn; j += 2 * cn * num_unroll, k += num_unroll) {
                const size_t vl = vsetvl_e32m4((w - 2 * k + 1) / 2);

                {
                    vint32m4_t r10_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row1 + j + 2 - bIdx), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t g10_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row1 + j + 1), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t b10_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row1 + j + bIdx), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t r11_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row1 + j + 2 - bIdx + cn), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t g11_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row1 + j + 1 + cn), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t b11_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row1 + j + bIdx + cn), 2 * cn * sizeof(uint8_t), vl), 255, vl);

                    vint32m4_t y10_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r10_v, CRY_coeff, vl), CGY_coeff, g10_v, vl), vadd_vx_i32m4(vmul_vx_i32m4(b10_v, CBY_coeff, vl), shifted16, vl), vl);
                    vint32m4_t y11_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r11_v, CRY_coeff, vl), CGY_coeff, g11_v, vl), vadd_vx_i32m4(vmul_vx_i32m4(b11_v, CBY_coeff, vl), shifted16, vl), vl);

                    vint16m2_t out10_v = vmin_vx_i16m2(vnclip_wx_i16m2(vmax_vx_i32m4(y10_v, 0, vl), SHIFT, vl), 255, vl);
                    vint16m2_t out11_v = vsll_vx_i16m2(vmin_vx_i16m2(vnclip_wx_i16m2(vmax_vx_i32m4(y11_v, 0, vl), SHIFT, vl), 255, vl), 8, vl);
                    vse16_v_i16m2((int16_t *)(y + 2 * k + yStride), vor_vv_i16m2(out10_v, out11_v, vl), vl);
                }
                {
                    vint32m4_t r01_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row0 + j + 2 - bIdx + cn), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t g01_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row0 + j + 1 + cn), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t b01_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row0 + j + bIdx + cn), 2 * cn * sizeof(uint8_t), vl), 255, vl);

                    vint32m4_t r00_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row0 + j + 2 - bIdx), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t g00_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row0 + j + 1), 2 * cn * sizeof(uint8_t), vl), 255, vl);
                    vint32m4_t b00_v = vand_vx_i32m4(vlsb_v_i32m4((int32_t *)(row0 + j + bIdx), 2 * cn * sizeof(uint8_t), vl), 255, vl);

                    vint32m4_t y00_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r00_v, CRY_coeff, vl), CGY_coeff, g00_v, vl), vadd_vx_i32m4(vmul_vx_i32m4(b00_v, CBY_coeff, vl), shifted16, vl), vl);
                    vint32m4_t y01_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r01_v, CRY_coeff, vl), CGY_coeff, g01_v, vl), vadd_vx_i32m4(vmul_vx_i32m4(b01_v, CBY_coeff, vl), shifted16, vl), vl);

                    vint32m4_t u00_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r00_v, CRU_coeff, vl), CGU_coeff, g00_v, vl), vadd_vx_i32m4(vmul_vx_i32m4(b00_v, CBU_coeff, vl), shifted128, vl), vl);
                    vint32m4_t v00_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r00_v, CBU_coeff, vl), CGV_coeff, g00_v, vl), vadd_vx_i32m4(vmul_vx_i32m4(b00_v, CBV_coeff, vl), shifted128, vl), vl);

                    vint16m2_t out00_v = vmin_vx_i16m2(vnclip_wx_i16m2(vmax_vx_i32m4(y00_v, 0, vl), SHIFT, vl), 255, vl);
                    vint16m2_t out01_v = vsll_vx_i16m2(vmin_vx_i16m2(vnclip_wx_i16m2(vmax_vx_i32m4(y01_v, 0, vl), SHIFT, vl), 255, vl), 8, vl);
                    vse16_v_i16m2((int16_t *)(y + 2 * k), vor_vv_i16m2(out00_v, out01_v, vl), vl);

                    vse8_v_u8m1(u + k, vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(u00_v, 0, vl)), SHIFT, vl), 0, vl), vl);
                    vse8_v_u8m1(v + k, vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(v00_v, 0, vl)), SHIFT, vl), 0, vl), vl);
                }
            }
        }
        return ppl::common::RC_SUCCESS;
    }

    int32_t bIdx;
};

template <int32_t dcn, int32_t bIdx>
::ppl::common::RetCode YUV420ptoRGB(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (width % 2 != 0 || height % 2 != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (dcn == 3) {
        YUV420p2RGB_u8 s = YUV420p2RGB_u8(bIdx);
        return s.operator()(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    } else if (dcn == 4) {
        YUV420p2RGBA_u8 s = YUV420p2RGBA_u8(bIdx);
        return s.operator()(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
    }
}

template <int32_t scn, int32_t bIdx>
::ppl::common::RetCode RGBtoYUV420p(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (width % 2 != 0 || height % 2 != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    RGBtoYUV420p_u8 s = RGBtoYUV420p_u8(bIdx);
    return s.operator()(height, width, scn, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

template <>
::ppl::common::RetCode I4202BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataU = inData + height * inWidthStride;
    const uint8_t *inDataV = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    return YUV420ptoRGB<3, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode YV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    return YUV420ptoRGB<3, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode I4202BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataU = inData + height * inWidthStride;
    const uint8_t *inDataV = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    return YUV420ptoRGB<4, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode YV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    return YUV420ptoRGB<4, 0>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
}

template <>
::ppl::common::RetCode BGR2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataU = outData + height * outWidthStride;
    uint8_t *outDataV = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}
template <>
::ppl::common::RetCode BGR2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode BGRA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataU = outData + height * outWidthStride;
    uint8_t *outDataV = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<4, 0>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}
template <>
::ppl::common::RetCode BGRA2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<4, 0>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode I4202RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataU = inData + height * inWidthStride;
    const uint8_t *inDataV = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    return YUV420ptoRGB<3, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode YV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    return YUV420ptoRGB<3, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode I4202RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataU = inData + height * inWidthStride;
    const uint8_t *inDataV = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    return YUV420ptoRGB<4, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode YV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const uint8_t *inDataY = inData;
    const uint8_t *inDataV = inData + height * inWidthStride;
    const uint8_t *inDataU = inData + height * inWidthStride + (height / 2) * (inWidthStride / 2);
    return YUV420ptoRGB<4, 2>(height, width, inWidthStride, inDataY, inWidthStride / 2, inDataU, inWidthStride / 2, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode RGB2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataU = outData + height * outWidthStride;
    uint8_t *outDataV = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode RGB2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode RGBA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataU = outData + height * outWidthStride;
    uint8_t *outDataV = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<4, 2>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

template <>
::ppl::common::RetCode RGBA2YV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *outDataY = outData;
    uint8_t *outDataV = outData + height * outWidthStride;
    uint8_t *outDataU = outData + height * outWidthStride + (height / 2) * (outWidthStride / 2);
    return RGBtoYUV420p<4, 2>(height, width, inWidthStride, inData, outWidthStride, outDataY, outWidthStride / 2, outDataU, outWidthStride / 2, outDataV);
}

// multiple plane implement
template <>
::ppl::common::RetCode I4202BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inDataY && nullptr == inDataU && nullptr == inDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUStride == 0 || inVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return YUV420ptoRGB<3, 0>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode I4202BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inDataY && nullptr == inDataU && nullptr == inDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUStride == 0 || inVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return YUV420ptoRGB<4, 0>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGR2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataY && nullptr == outDataU && nullptr == outDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUStride == 0 || outVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return RGBtoYUV420p<3, 0>(height, width, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

template <>
::ppl::common::RetCode BGRA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataY && nullptr == outDataU && nullptr == outDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUStride == 0 || outVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return RGBtoYUV420p<4, 0>(height, width, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

template <>
::ppl::common::RetCode I4202RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inDataY && nullptr == inDataU && nullptr == inDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUStride == 0 || inVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return YUV420ptoRGB<3, 2>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode I4202RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inDataY,
    int32_t inUStride,
    const uint8_t *inDataU,
    int32_t inVStride,
    const uint8_t *inDataV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inDataY && nullptr == inDataU && nullptr == inDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUStride == 0 || inVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return YUV420ptoRGB<4, 2>(height, width, inYStride, inDataY, inUStride, inDataU, inVStride, inDataV, outWidthStride, outData);
}
template <>
::ppl::common::RetCode RGB2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataY && nullptr == outDataU && nullptr == outDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUStride == 0 || outVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return RGBtoYUV420p<3, 2>(height, width, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

template <>
::ppl::common::RetCode RGBA2I420<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outDataY,
    int32_t outUStride,
    uint8_t *outDataU,
    int32_t outVStride,
    uint8_t *outDataV)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataY && nullptr == outDataU && nullptr == outDataV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUStride == 0 || outVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return RGBtoYUV420p<4, 2>(height, width, inWidthStride, inData, outYStride, outDataY, outUStride, outDataU, outVStride, outDataV);
}

}
}
} // namespace ppl::cv::riscv
