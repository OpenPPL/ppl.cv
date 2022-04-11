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
#include "ppl/cv/riscv/util.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/cv/riscv/typetraits.h"
#include <string.h>
#include <cmath>

#include <limits.h>
#include <algorithm>

namespace ppl {
namespace cv {
namespace riscv {

#define CY_coeff  1220542
#define CUB_coeff 2116026
#define CUG_coeff -409993
#define CVG_coeff -852492
#define CVR_coeff 1673527
#define SHIFT     20

// Coefficients for RGB to YUV420p conversion
#define CRY_coeff 269484
#define CGY_coeff 528482
#define CBY_coeff 102760
#define CRU_coeff -155188
#define CGU_coeff -305135
#define CBU_coeff 460324
#define CGV_coeff -385875
#define CBV_coeff -74448

#define DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))
template <int32_t srccn, int32_t bIdx, bool isUV>
void rgb_2_nv(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    const size_t vl = vsetvlmax_e32m4();
    const int32_t num_unroll = vl;

    const int32_t shifted16 = (16 << SHIFT);
    const int32_t shifted128 = (128 << SHIFT);
    const int32_t halfShift = (1 << (SHIFT - 1));
    const vint32m4_t b16_v = vmv_v_x_i32m4(halfShift + shifted16, vl);
    const vint32m4_t b128_v = vmv_v_x_i32m4(halfShift + shifted128, vl);

    for (int32_t i = 0; i < height; i += 2) {
        const uint8_t *src0 = inData + i * inWidthStride;
        const uint8_t *src1 = inData + (i + 1) * inWidthStride;
        uint8_t *dst0 = outY + i * outYStride;
        uint8_t *dst1 = outY + (i + 1) * outYStride;
        uint8_t *dst2 = outUV + (i / 2) * outUVStride;
        for (int32_t j = 0; j < width / 2; j += num_unroll, src0 += 2 * srccn * num_unroll, src1 += 2 * srccn * num_unroll) {
            const size_t vl = vsetvl_e32m4(width / 2 - j);

            vint32m4_t r00_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src0 + 2 - bIdx), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t g00_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src0 + 1), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t b00_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src0 + bIdx), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t r01_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src0 + 2 - bIdx + srccn), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t g01_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src0 + 1 + srccn), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t b01_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src0 + bIdx + srccn), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t r10_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src1 + 2 - bIdx), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t g10_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src1 + 1), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t b10_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src1 + bIdx), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t r11_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src1 + 2 - bIdx + srccn), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t g11_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src1 + 1 + srccn), 2 * srccn * sizeof(uint8_t), vl));
            vint32m4_t b11_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src1 + bIdx + srccn), 2 * srccn * sizeof(uint8_t), vl));

            vint32m4_t y00_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r00_v, CRY_coeff, vl), CGY_coeff, g00_v, vl), vmacc_vx_i32m4(b16_v, CBY_coeff, b00_v, vl), vl);
            vint32m4_t y01_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r01_v, CRY_coeff, vl), CGY_coeff, g01_v, vl), vmacc_vx_i32m4(b16_v, CBY_coeff, b01_v, vl), vl);
            vint32m4_t y10_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r10_v, CRY_coeff, vl), CGY_coeff, g10_v, vl), vmacc_vx_i32m4(b16_v, CBY_coeff, b10_v, vl), vl);
            vint32m4_t y11_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r11_v, CRY_coeff, vl), CGY_coeff, g11_v, vl), vmacc_vx_i32m4(b16_v, CBY_coeff, b11_v, vl), vl);

            vsse8_v_u8m1(dst0 + 2 * j + 0, 2 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(y00_v, 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst0 + 2 * j + 1, 2 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(y01_v, 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst1 + 2 * j + 0, 2 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(y10_v, 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst1 + 2 * j + 1, 2 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(y11_v, 0, vl)), SHIFT, vl), 0, vl), vl);

            vint32m4_t u00_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r00_v, CRU_coeff, vl), CGU_coeff, g00_v, vl), vmacc_vx_i32m4(b128_v, CBU_coeff, b00_v, vl), vl);
            vint32m4_t v00_v = vadd_vv_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(r00_v, CBU_coeff, vl), CGV_coeff, g00_v, vl), vmacc_vx_i32m4(b128_v, CBV_coeff, b00_v, vl), vl);

            if (isUV) {
                vsse8_v_u8m1(dst2 + 2 * j + 0, 2 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(u00_v, 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(dst2 + 2 * j + 1, 2 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(v00_v, 0, vl)), SHIFT, vl), 0, vl), vl);
            } else {
                vsse8_v_u8m1(dst2 + 2 * j + 0, 2 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(v00_v, 0, vl)), SHIFT, vl), 0, vl), vl);
                vsse8_v_u8m1(dst2 + 2 * j + 1, 2 * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(u00_v, 0, vl)), SHIFT, vl), 0, vl), vl);
            }
        }
    }
}

template <int32_t dstcn, int32_t blueIdx, bool isUV>
void nv_2_rgb(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    const uint8_t delta_uv = 128, alpha = 255;
    const size_t vl = vsetvlmax_e32m4();
    const int32_t num_unroll = vl;

    const vuint8m1_t alpha_v = vmv_v_x_u8m1(alpha, vl);
    for (int32_t i = 0; i < height; i += 2) {
        const uint8_t *src0 = inY + i * inYStride;
        const uint8_t *src1 = inY + (i + 1) * inYStride;
        const uint8_t *src2 = inUV + (i / 2) * inUVStride;
        uint8_t *dst0 = outData + i * outWidthStride;
        uint8_t *dst1 = outData + (i + 1) * outWidthStride;
        for (int32_t j = 0; j < width; j += 2 * num_unroll, dst0 += 2 * dstcn * num_unroll, dst1 += 2 * dstcn * num_unroll) {
            const size_t vl = vsetvl_e32m4((width - j + 1) / 2);

            vint32m4_t y00_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src0 + j + 0), 2 * sizeof(uint8_t), vl));
            vint32m4_t y01_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src0 + j + 1), 2 * sizeof(uint8_t), vl));
            vint32m4_t y10_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src1 + j + 0), 2 * sizeof(uint8_t), vl));
            vint32m4_t y11_v = vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src1 + j + 1), 2 * sizeof(uint8_t), vl));

            y00_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y00_v, -16, vl), 0, vl), CY_coeff, vl);
            y01_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y01_v, -16, vl), 0, vl), CY_coeff, vl);
            y10_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y10_v, -16, vl), 0, vl), CY_coeff, vl);
            y11_v = vmul_vx_i32m4(vmax_vx_i32m4(vadd_vx_i32m4(y11_v, -16, vl), 0, vl), CY_coeff, vl);

            vint32m4_t u_v;
            vint32m4_t v_v;

            if (isUV) {
                u_v = vadd_vx_i32m4(vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src2 + j + 0), 2 * sizeof(uint8_t), vl)), -delta_uv, vl);
                v_v = vadd_vx_i32m4(vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src2 + j + 1), 2 * sizeof(uint8_t), vl)), -delta_uv, vl);
            } else {
                v_v = vadd_vx_i32m4(vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src2 + j + 0), 2 * sizeof(uint8_t), vl)), -delta_uv, vl);
                u_v = vadd_vx_i32m4(vreinterpret_v_u32m4_i32m4(vlsbu_v_u32m4((uint32_t *)(src2 + j + 1), 2 * sizeof(uint8_t), vl)), -delta_uv, vl);
            }

            vint32m4_t ruv_v = vadd_vx_i32m4(vmul_vx_i32m4(v_v, CVR_coeff, vl), (1 << (SHIFT - 1)), vl);
            vint32m4_t guv_v = vadd_vx_i32m4(vmacc_vx_i32m4(vmul_vx_i32m4(v_v, CVG_coeff, vl), CUG_coeff, u_v, vl), (1 << (SHIFT - 1)), vl);
            vint32m4_t buv_v = vadd_vx_i32m4(vmul_vx_i32m4(u_v, CUB_coeff, vl), (1 << (SHIFT - 1)), vl);

            vsse8_v_u8m1(dst0 + blueIdx, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst0 + 1, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst0 + (blueIdx ^ 2), 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y00_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

            vsse8_v_u8m1(dst1 + blueIdx, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst1 + 1, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst1 + (blueIdx ^ 2), 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y10_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

            vsse8_v_u8m1(dst0 + blueIdx + dstcn, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst0 + 1 + dstcn, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst0 + (blueIdx ^ 2) + dstcn, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y01_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

            vsse8_v_u8m1(dst1 + blueIdx + dstcn, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, buv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst1 + 1 + dstcn, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, guv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);
            vsse8_v_u8m1(dst1 + (blueIdx ^ 2) + dstcn, 2 * dstcn * sizeof(uint8_t), vnclipu_wx_u8m1(vnclipu_wx_u16m2(vreinterpret_v_i32m4_u32m4(vmax_vx_i32m4(vadd_vv_i32m4(y11_v, ruv_v, vl), 0, vl)), SHIFT, vl), 0, vl), vl);

            if (dstcn == 4) {
                vsse8_v_u8m1(dst1 + 3, 2 * dstcn * sizeof(uint8_t), alpha_v, vl);
                vsse8_v_u8m1(dst0 + 3, 2 * dstcn * sizeof(uint8_t), alpha_v, vl);
                vsse8_v_u8m1(dst1 + 3 + dstcn, 2 * dstcn * sizeof(uint8_t), alpha_v, vl);
                vsse8_v_u8m1(dst0 + 3 + dstcn, 2 * dstcn * sizeof(uint8_t), alpha_v, vl);
            }
        }
    }
}

template <>
::ppl::common::RetCode BGR2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<3, 0, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 0, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, true>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<3, 2, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 2, true>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<3, 0, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 0, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV12<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, true>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<3, 2, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV122RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 2, true>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<3, 0, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 0, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, false>(height, width, inWidthStride, inData, outWidthStride, outData, outWidthStride, outData + height * outWidthStride);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<3, 2, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inData && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 2, false>(height, width, inWidthStride, inData, inWidthStride, inData + height * inWidthStride, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 0, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGRA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 0, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<3, 0, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 0, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGB2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<3, 2, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode RGBA2NV21<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outYStride,
    uint8_t *outY,
    int32_t outUVStride,
    uint8_t *outUV)
{
    if (nullptr == inData && nullptr == outY && nullptr == outUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride == 0 || outYStride == 0 || outUVStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    rgb_2_nv<4, 2, false>(height, width, inWidthStride, inData, outYStride, outY, outUVStride, outUV);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGB<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<3, 2, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode NV212RGBA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const uint8_t *inY,
    int32_t inUVStride,
    const uint8_t *inUV,
    int32_t outWidthStride,
    uint8_t *outData)
{
    if (nullptr == inY && nullptr == inUV) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inYStride == 0 || inUVStride == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    nv_2_rgb<4, 2, false>(height, width, inYStride, inY, inUVStride, inUV, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}
}
}
} // namespace ppl::cv::riscv
