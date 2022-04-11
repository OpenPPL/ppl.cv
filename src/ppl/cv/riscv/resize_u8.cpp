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
#include "ppl/common/retcode.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "ppl/cv/riscv/resize.h"
#include "ppl/cv/riscv/resize_common.h"
#include "ppl/cv/types.h"
#include "util.h"
#include <vector>

namespace ppl {
namespace cv {
namespace riscv {
template <>
struct ResizeNearestPointFunc<uint8_t, 1> {
    static void resize_nearest_point(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const uint8_t* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        uint8_t* outData)
    {
        constexpr int32_t num_unroll = 16;

        double scale_y = (double)inHeight / outHeight;
        double scale_x = (double)inWidth / outWidth;

        int32_t* buffer = (int32_t*)malloc((outWidth + outHeight) * sizeof(int32_t));

        int32_t* in_x_ofs = buffer;
        int32_t* in_y_ofs = in_x_ofs + outWidth;

        cal_resize_nearest_point_ofs<false>(outWidth, 1, scale_x, in_x_ofs);
        cal_resize_nearest_point_ofs<true>(outHeight, inWidthStride, scale_y, in_y_ofs);

        auto out = outData;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            int32_t in_y = in_y_ofs[out_y];
            auto in = inData + in_y;

            int32_t out_x = 0;
            for (; out_x <= outWidth - num_unroll; out_x += num_unroll) {
                out[out_x + 0] = in[in_x_ofs[out_x + 0]];
                out[out_x + 1] = in[in_x_ofs[out_x + 1]];
                out[out_x + 2] = in[in_x_ofs[out_x + 2]];
                out[out_x + 3] = in[in_x_ofs[out_x + 3]];
                out[out_x + 4] = in[in_x_ofs[out_x + 4]];
                out[out_x + 5] = in[in_x_ofs[out_x + 5]];
                out[out_x + 6] = in[in_x_ofs[out_x + 6]];
                out[out_x + 7] = in[in_x_ofs[out_x + 7]];
                out[out_x + 8] = in[in_x_ofs[out_x + 8]];
                out[out_x + 9] = in[in_x_ofs[out_x + 9]];
                out[out_x + 10] = in[in_x_ofs[out_x + 10]];
                out[out_x + 11] = in[in_x_ofs[out_x + 11]];
                out[out_x + 12] = in[in_x_ofs[out_x + 12]];
                out[out_x + 13] = in[in_x_ofs[out_x + 13]];
                out[out_x + 14] = in[in_x_ofs[out_x + 14]];
                out[out_x + 15] = in[in_x_ofs[out_x + 15]];
            }
            for (; out_x <= outWidth - 4; out_x += 4) {
                out[out_x + 0] = in[in_x_ofs[out_x + 0]];
                out[out_x + 1] = in[in_x_ofs[out_x + 1]];
                out[out_x + 2] = in[in_x_ofs[out_x + 2]];
                out[out_x + 3] = in[in_x_ofs[out_x + 3]];
            }
            for (; out_x < outWidth; ++out_x) {
                int32_t in_x = in_x_ofs[out_x];
                out[out_x] = in[in_x];
            }
            out += outWidthStride;
        }
        free(buffer);
    }
};

template <>
struct ResizeNearestPointFunc<uint8_t, 3> {
    static void resize_nearest_point(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const uint8_t* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        uint8_t* outData)
    {
        constexpr int32_t channels = 3;
        const int32_t num_unroll = 4;

        double scale_y = (double)inHeight / outHeight;
        double scale_x = (double)inWidth / outWidth;

        int32_t* buffer = (int32_t*)malloc((outWidth + outHeight) * sizeof(int32_t));

        int32_t* in_x_ofs = buffer;
        int32_t* in_y_ofs = in_x_ofs + outWidth;

        cal_resize_nearest_point_ofs<true>(outWidth, channels, scale_x, in_x_ofs);
        cal_resize_nearest_point_ofs<true>(outHeight, inWidthStride, scale_y, in_y_ofs);

        auto out = outData;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            int32_t in_y = in_y_ofs[out_y];
            auto in = inData + in_y;

            int32_t out_x = 0, out_xc = 0;
            for (; out_x <= outWidth - 1 - num_unroll; out_x += num_unroll, out_xc += channels * num_unroll) {
                *((uint32_t*)(out + out_xc + channels * 0)) = *((uint32_t*)(in + in_x_ofs[out_x + 0]));
                *((uint32_t*)(out + out_xc + channels * 1)) = *((uint32_t*)(in + in_x_ofs[out_x + 1]));
                *((uint32_t*)(out + out_xc + channels * 2)) = *((uint32_t*)(in + in_x_ofs[out_x + 2]));
                *((uint32_t*)(out + out_xc + channels * 3)) = *((uint32_t*)(in + in_x_ofs[out_x + 3]));
            }
            for (; out_x < outWidth; ++out_x, out_xc += channels) {
                int32_t in_xc = in_x_ofs[out_x];
                *((uint16_t*)(out + out_xc)) = *((uint16_t*)(in + in_xc));
                *(out + out_xc + 2) = *(in + in_xc + 2);
            }
            out += outWidthStride;
        }
        free(buffer);
    }
};

template <>
struct ResizeNearestPointFunc<uint8_t, 4> {
    static void resize_nearest_point(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const uint8_t* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        uint8_t* outData)
    {
        constexpr int32_t channels = 4;
        constexpr int32_t num_unroll = 8;

        double scale_y = (double)inHeight / outHeight;
        double scale_x = (double)inWidth / outWidth;

        int32_t* buffer = (int32_t*)malloc((outWidth + outHeight) * sizeof(int32_t));

        int32_t* in_x_ofs = buffer;
        int32_t* in_y_ofs = in_x_ofs + outWidth;

        cal_resize_nearest_point_ofs<true>(outWidth, channels, scale_x, in_x_ofs);
        cal_resize_nearest_point_ofs<true>(outHeight, inWidthStride, scale_y, in_y_ofs);

        auto out = outData;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            int32_t in_y = in_y_ofs[out_y];
            auto in = inData + in_y;

            int32_t out_x = 0;
            for (; out_x <= outWidth - num_unroll; out_x += num_unroll) {
                *(((uint32_t*)out + out_x) + 0) = *((uint32_t*)(in + in_x_ofs[out_x + 0]));
                *(((uint32_t*)out + out_x) + 1) = *((uint32_t*)(in + in_x_ofs[out_x + 1]));
                *(((uint32_t*)out + out_x) + 2) = *((uint32_t*)(in + in_x_ofs[out_x + 2]));
                *(((uint32_t*)out + out_x) + 3) = *((uint32_t*)(in + in_x_ofs[out_x + 3]));
                *(((uint32_t*)out + out_x) + 4) = *((uint32_t*)(in + in_x_ofs[out_x + 4]));
                *(((uint32_t*)out + out_x) + 5) = *((uint32_t*)(in + in_x_ofs[out_x + 5]));
                *(((uint32_t*)out + out_x) + 6) = *((uint32_t*)(in + in_x_ofs[out_x + 6]));
                *(((uint32_t*)out + out_x) + 7) = *((uint32_t*)(in + in_x_ofs[out_x + 7]));
            }
            for (; out_x < outWidth; ++out_x) {
                int32_t in_x = in_x_ofs[out_x];
                *((uint32_t*)out + out_x) = *((uint32_t*)(in + in_x));
            }
            out += outWidthStride;
        }
        free(buffer);
    }
};

template <>
void resize_linear_shrink2<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    const size_t vl = vsetvlmax_e8m4();
    const int32_t num_unroll = vl;

    const int32_t scale_x = 2;
    const int32_t scale_y = 2;

    vuint8m4_t data0_v, data1_v, data2_v, data3_v;

    auto in = inData;
    auto out = outData;
    for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
        auto in0 = in;
        auto in1 = in + inWidthStride;

        int32_t out_x = 0, in_x = 0;
        for (; out_x < outWidth; out_x += num_unroll, in_x += num_unroll * scale_x) {
            const size_t vl = vsetvl_e8m4(outWidth - out_x);

            vlseg2e8_v_u8m4(&data0_v, &data1_v, in0 + in_x, vl);
            vlseg2e8_v_u8m4(&data2_v, &data3_v, in1 + in_x, vl);

            vuint16m8_t psum0_v = vwaddu_vv_u16m8(data0_v, data1_v, vl);
            vuint16m8_t psum1_v = vwaddu_vv_u16m8(data2_v, data3_v, vl);
            vuint16m8_t sum_v = vadd_vv_u16m8(psum0_v, psum1_v, vl);
            vuint8m4_t out_v = vnclipu_wx_u8m4(vsrl_vx_u16m8(sum_v, 2, vl), 0, vl);

            vse8_v_u8m4(out + out_x, out_v, vl);
        }

        in += inWidthStride * scale_y;
        out += outWidthStride;
    }
}

template <>
void resize_linear_shrink2<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    const int32_t channels = 3;
    const size_t vl = vsetvlmax_e8m1();
    const int32_t num_unroll = vl;

    const int32_t scale_x = 2;
    const int32_t scale_y = 2;

    vuint8m1_t data0_c0_v, data1_c0_v, data2_c0_v, data3_c0_v,
        data0_c1_v, data1_c1_v, data2_c1_v, data3_c1_v,
        data0_c2_v, data1_c2_v, data2_c2_v, data3_c2_v;

    auto in = inData;
    auto out = outData;
    for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
        auto in0 = in;
        auto in1 = in + inWidthStride;

        int32_t out_x = 0, in_xc = 0, out_xc = 0;
        for (; out_x < outWidth; out_x += num_unroll, in_xc += num_unroll * scale_x * channels, out_xc += num_unroll * channels) {
            const size_t vl = vsetvl_e8m1(outWidth - out_x);

            vlseg6e8_v_u8m1(&data0_c0_v, &data0_c1_v, &data0_c2_v, &data1_c0_v, &data1_c1_v, &data1_c2_v, in0 + in_xc, vl);
            vlseg6e8_v_u8m1(&data2_c0_v, &data2_c1_v, &data2_c2_v, &data3_c0_v, &data3_c1_v, &data3_c2_v, in1 + in_xc, vl);

            vuint16m2_t sum_c0_v = vadd_vv_u16m2(vwaddu_vv_u16m2(data0_c0_v, data1_c0_v, vl), vwaddu_vv_u16m2(data2_c0_v, data3_c0_v, vl), vl);
            vuint16m2_t sum_c1_v = vadd_vv_u16m2(vwaddu_vv_u16m2(data0_c1_v, data1_c1_v, vl), vwaddu_vv_u16m2(data2_c1_v, data3_c1_v, vl), vl);
            vuint16m2_t sum_c2_v = vadd_vv_u16m2(vwaddu_vv_u16m2(data0_c2_v, data1_c2_v, vl), vwaddu_vv_u16m2(data2_c2_v, data3_c2_v, vl), vl);

            vuint8m1_t out_c0_v = vnclipu_wx_u8m1(vsrl_vx_u16m2(sum_c0_v, 2, vl), 0, vl);
            vuint8m1_t out_c1_v = vnclipu_wx_u8m1(vsrl_vx_u16m2(sum_c1_v, 2, vl), 0, vl);
            vuint8m1_t out_c2_v = vnclipu_wx_u8m1(vsrl_vx_u16m2(sum_c2_v, 2, vl), 0, vl);

            vsseg3e8_v_u8m1(out + out_xc, out_c0_v, out_c1_v, out_c2_v, vl);
        }

        in += inWidthStride * scale_y;
        out += outWidthStride;
    }
}

template <>
void resize_linear_shrink2<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    const int32_t channels = 4;
    const size_t vl = vsetvlmax_e32m4();
    const int32_t num_unroll = vl;

    const int32_t scale_x = 2;
    const int32_t scale_y = 2;
    const int32_t in_data_stride = scale_x * channels * sizeof(uint8_t);

    vuint8m4_t data0_v, data1_v, data2_v, data3_v;

    auto in = inData;
    auto out = outData;
    for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
        auto in0 = in;
        auto in1 = in + inWidthStride;

        int32_t out_x = 0, in_xc = 0, out_xc = 0;
        for (; out_x < outWidth; out_x += num_unroll, in_xc += num_unroll * scale_x * channels, out_xc += num_unroll * channels) {
            const size_t vl = vsetvl_e8m4(outWidth * channels - out_xc);
            const size_t e32_vl = vl / channels;

            data0_v = vreinterpret_v_u32m4_u8m4(vlse32_v_u32m4((const uint32_t*)(in0 + in_xc + 0), in_data_stride, e32_vl));
            data1_v = vreinterpret_v_u32m4_u8m4(vlse32_v_u32m4((const uint32_t*)(in0 + in_xc + channels), in_data_stride, e32_vl));
            data2_v = vreinterpret_v_u32m4_u8m4(vlse32_v_u32m4((const uint32_t*)(in1 + in_xc + 0), in_data_stride, e32_vl));
            data3_v = vreinterpret_v_u32m4_u8m4(vlse32_v_u32m4((const uint32_t*)(in1 + in_xc + channels), in_data_stride, e32_vl));

            vuint16m8_t psum0_v = vwaddu_vv_u16m8(data0_v, data1_v, vl);
            vuint16m8_t psum1_v = vwaddu_vv_u16m8(data2_v, data3_v, vl);
            vuint16m8_t sum_v = vadd_vv_u16m8(psum0_v, psum1_v, vl);
            vuint8m4_t out_v = vnclipu_wx_u8m4(vsrl_vx_u16m8(sum_v, 2, vl), 0, vl);

            vse8_v_u8m4(out + out_xc, out_v, vl);
        }

        in += inWidthStride * scale_y;
        out += outWidthStride;
    }
}

template <int32_t channels>
struct ResizeLinearKernelFunc<uint8_t, __fp16, channels> {
    inline static void resize_linear_kernel(
        const uint32_t* x0_ofs,
        const uint32_t* x1_ofs,
        const uint32_t* y0_ofs,
        const uint32_t* y1_ofs,
        const __fp16* alpha0_lst,
        const __fp16* alpha1_lst,
        const __fp16* beta0_lst,
        const __fp16* beta1_lst,
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const uint8_t* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        uint8_t* outData)
    {
        const size_t vl = vsetvlmax_e8m1();
        const int32_t num_unroll = vl;

        auto out = outData;
        outWidth *= channels;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            auto in0 = inData + y0_ofs[out_y];
            auto in1 = inData + y1_ofs[out_y];
            float beta0 = beta0_lst[out_y];
            float beta1 = beta1_lst[out_y];

            vfloat16m2_t beta0_v = vfncvt_f_f_w_f16m2(vfmv_v_f_f32m4(beta0, vl), vl); // patch
            vfloat16m2_t beta1_v = vfncvt_f_f_w_f16m2(vfmv_v_f_f32m4(beta1, vl), vl); // patch

            int32_t out_x = 0;
            for (; out_x < outWidth; out_x += num_unroll) {
                const size_t vl = vsetvl_e8m1(outWidth - out_x);

                vuint32m4_t in_x0_v = vle32_v_u32m4(x0_ofs + out_x, vl);
                vuint32m4_t in_x1_v = vle32_v_u32m4(x1_ofs + out_x, vl);

                vfloat16m2_t data0_v = vfncvt_f_xu_w_f16m2(vlxbu_v_u32m4((uint32_t*)in0, in_x0_v, vl), vl);
                vfloat16m2_t data1_v = vfncvt_f_xu_w_f16m2(vlxbu_v_u32m4((uint32_t*)in0, in_x1_v, vl), vl);
                vfloat16m2_t data2_v = vfncvt_f_xu_w_f16m2(vlxbu_v_u32m4((uint32_t*)in1, in_x0_v, vl), vl);
                vfloat16m2_t data3_v = vfncvt_f_xu_w_f16m2(vlxbu_v_u32m4((uint32_t*)in1, in_x1_v, vl), vl);

                vfloat16m2_t alpha0_v = vle16_v_f16m2(alpha0_lst + out_x, vl);
                vfloat16m2_t alpha1_v = vle16_v_f16m2(alpha1_lst + out_x, vl);

                vfloat16m2_t p0_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(alpha0_v, data0_v, vl), alpha1_v, data1_v, vl);
                vfloat16m2_t p1_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(alpha0_v, data2_v, vl), alpha1_v, data3_v, vl);
                vfloat16m2_t out_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(p0_v, beta0_v, vl), beta1_v, p1_v, vl);

                vse8_v_u8m1(out + out_x, vfncvt_xu_f_w_u8m1(out_v, vl), vl);
            }
            out += outWidthStride;
        }
    }
};

template <bool resize_area, int32_t channels>
struct ResizeLinearFunc<uint8_t, resize_area, channels> {
    static void resize_linear(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const uint8_t* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        uint8_t* outData)
    {
        typedef __fp16 facT;
        typedef uint8_t eT;

        const int32_t dim0 = outHeight;
        const int32_t dim1 = outWidth * channels;
        const int32_t x_tabs_len = dim1 * 2;
        const int32_t y_tabs_len = dim0 * 2;

        uint32_t* buffer = (uint32_t*)malloc((x_tabs_len + y_tabs_len) * (sizeof(uint32_t) + sizeof(facT)));
        uint32_t *x_ofs = buffer, *y_ofs = buffer + x_tabs_len;
        facT *alpha_lst = (facT*)(y_ofs + y_tabs_len), *beta_lst = alpha_lst + x_tabs_len;

        uint32_t *x0_ofs = x_ofs, *x1_ofs = x0_ofs + dim1;
        uint32_t *y0_ofs = y_ofs, *y1_ofs = y_ofs + dim0;
        facT *alpha0_lst = alpha_lst, *alpha1_lst = alpha_lst + dim1;
        facT *beta0_lst = beta_lst, *beta1_lst = beta_lst + dim0;

        const int32_t x_ofs_factor = sizeof(eT) / sizeof(int8_t);
        const int32_t y_ofs_factor = inWidthStride;

        cal_resize_linear_ofs<facT, resize_area, true, false>(inWidth, outWidth, channels, x_ofs_factor, x0_ofs, x1_ofs, alpha0_lst, alpha1_lst);
        cal_resize_linear_ofs<facT, resize_area, false, true>(inHeight, outHeight, 1, y_ofs_factor, y0_ofs, y1_ofs, beta0_lst, beta1_lst);

        ResizeLinearKernelFunc<eT, facT, channels>::resize_linear_kernel(
            x0_ofs,
            x1_ofs,
            y0_ofs,
            y1_ofs,
            alpha0_lst,
            alpha1_lst,
            beta0_lst,
            beta1_lst,
            inHeight,
            inWidth,
            inWidthStride,
            inData,
            outHeight,
            outWidth,
            outWidthStride,
            outData);

        free(buffer);
    }
};

// template <>
// struct ResizeLinearKernelFunc<uint8_t, __fp16, 3>
// {
//     inline static void resize_linear_kernel(
//         const uint32_t *x0_ofs,
//         const uint32_t *x1_ofs,
//         const uint32_t *y0_ofs,
//         const uint32_t *y1_ofs,
//         const __fp16 *alpha0_lst,
//         const __fp16 *alpha1_lst,
//         const __fp16 *beta0_lst,
//         const __fp16 *beta1_lst,
//         int32_t inHeight,
//         int32_t inWidth,
//         int32_t inWidthStride,
//         const uint8_t* inData,
//         int32_t outHeight,
//         int32_t outWidth,
//         int32_t outWidthStride,
//         uint8_t* outData
//     )
//     {
//         const int32_t channels = 3;
//         const int32_t vir_channels = 4;
//         const size_t e32_vl = vsetvlmax_e32m1();
//         const int32_t num_unroll = e32_vl;

//         __fp16 alpha0_lst_dup[outWidth * vir_channels], alpha1_lst_dup[outWidth * vir_channels];
//         for (int32_t i  = 0, ic = 0; i < outWidth; ++i, ic += vir_channels) {
//             alpha0_lst_dup[ic + 0] = alpha0_lst[i], alpha0_lst_dup[ic + 1] = alpha0_lst[i];
//             alpha0_lst_dup[ic + 2] = alpha0_lst[i], alpha0_lst_dup[ic + 3] = alpha0_lst[i];
//             alpha1_lst_dup[ic + 0] = alpha1_lst[i], alpha1_lst_dup[ic + 1] = alpha1_lst[i];
//             alpha1_lst_dup[ic + 2] = alpha1_lst[i], alpha1_lst_dup[ic + 3] = alpha1_lst[i];
//         }

//         auto out = outData;
//         for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
//             auto in0 = inData + y0_ofs[out_y];
//             auto in1 = inData + y1_ofs[out_y];
//             float beta0 = beta0_lst[out_y];
//             float beta1 = beta1_lst[out_y];

//             const size_t vl = vsetvlmax_e8m1();
//             vfloat16m2_t beta0_v = vfncvt_f_f_w_f16m2(vfmv_v_f_f32m4(beta0, vl), vl); // patch
//             vfloat16m2_t beta1_v = vfncvt_f_f_w_f16m2(vfmv_v_f_f32m4(beta1, vl), vl); // patch

//             int32_t out_x = 0, out_xc = 0;
//             for (; out_x < outWidth; out_x += num_unroll, out_xc += num_unroll * channels) {
//                 const size_t vl = vsetvl_e8m1((outWidth - out_x) * vir_channels);
//                 const size_t e32_vl = vl / vir_channels;

//                 vuint32m1_t in_x0_v = vle32_v_u32m1(x0_ofs + out_x, e32_vl);
//                 vuint32m1_t in_x1_v = vle32_v_u32m1(x1_ofs + out_x, e32_vl);

//                 vfloat16m2_t data0_v = vfwcvt_f_xu_v_f16m2(vreinterpret_v_u32m1_u8m1(vlxwu_v_u32m1((uint32_t*)in0, in_x0_v, e32_vl)), vl);
//                 vfloat16m2_t data1_v = vfwcvt_f_xu_v_f16m2(vreinterpret_v_u32m1_u8m1(vlxwu_v_u32m1((uint32_t*)in0, in_x1_v, e32_vl)), vl);
//                 vfloat16m2_t data2_v = vfwcvt_f_xu_v_f16m2(vreinterpret_v_u32m1_u8m1(vlxwu_v_u32m1((uint32_t*)in1, in_x0_v, e32_vl)), vl);
//                 vfloat16m2_t data3_v = vfwcvt_f_xu_v_f16m2(vreinterpret_v_u32m1_u8m1(vlxwu_v_u32m1((uint32_t*)in1, in_x1_v, e32_vl)), vl);

//                 vfloat16m2_t alpha0_v = vle16_v_f16m2(alpha0_lst_dup + out_xc, vl);
//                 vfloat16m2_t alpha1_v = vle16_v_f16m2(alpha1_lst_dup + out_xc, vl);

//                 vfloat16m2_t p0_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(data0_v, alpha0_v, vl), alpha1_v, data1_v, vl);
//                 vfloat16m2_t p1_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(data2_v, alpha0_v, vl), alpha1_v, data3_v, vl);
//                 vfloat16m2_t out_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(p0_v, beta0_v, vl), beta1_v, p1_v, vl);

//                 vse8_v_u8m1(out + out_xc, vfncvt_xu_f_w_u8m1(out_v, vl), vl);
//             }
//             out += outWidthStride;
//         }
//     }
// };

// template <bool resize_area>
// struct ResizeLinearFunc<uint8_t, resize_area, 3>
// {
//     static void resize_linear(
//         int32_t inHeight,
//         int32_t inWidth,
//         int32_t inWidthStride,
//         const uint8_t* inData,
//         int32_t outHeight,
//         int32_t outWidth,
//         int32_t outWidthStride,
//         uint8_t* outData)
//     {
//         typedef __fp16 facT;
//         typedef uint8_t eT;
//         constexpr int32_t channels = 3;

//         const int32_t dim0 = outHeight;
//         const int32_t dim1 = outWidth;
//         const int32_t x_tabs_len = dim1 * 2;
//         const int32_t y_tabs_len = dim0 * 2;

//         uint32_t *buffer = (uint32_t*)malloc((x_tabs_len + y_tabs_len) * (sizeof(uint32_t) + sizeof(facT)));
//         uint32_t *x_ofs = buffer, *y_ofs = buffer + x_tabs_len;
//         facT *alpha_lst = (facT*)(y_ofs + y_tabs_len), *beta_lst = alpha_lst + x_tabs_len;

//         uint32_t *x0_ofs = x_ofs, *x1_ofs = x0_ofs + dim1;
//         uint32_t *y0_ofs = y_ofs, *y1_ofs = y_ofs + dim0;
//         facT *alpha0_lst = alpha_lst, *alpha1_lst = alpha_lst + dim1;
//         facT *beta0_lst = beta_lst, *beta1_lst = beta_lst + dim0;

//         const int32_t x_ofs_factor = channels;
//         const int32_t y_ofs_factor = inWidthStride;

//         cal_resize_linear_ofs<facT, resize_area, false, true>(inWidth, outWidth, channels, x_ofs_factor, x0_ofs, x1_ofs, alpha0_lst, alpha1_lst);
//         cal_resize_linear_ofs<facT, resize_area, false, true>(inHeight, outHeight, 1, y_ofs_factor, y0_ofs, y1_ofs, beta0_lst, beta1_lst);

//         ResizeLinearKernelFunc<eT, facT, channels>::resize_linear_kernel(
//             x0_ofs,
//             x1_ofs,
//             y0_ofs,
//             y1_ofs,
//             alpha0_lst,
//             alpha1_lst,
//             beta0_lst,
//             beta1_lst,
//             inHeight,
//             inWidth,
//             inWidthStride,
//             inData,
//             outHeight,
//             outWidth,
//             outWidthStride,
//             outData
//         );

//         free(buffer);
//     }
// };

template <>
struct ResizeLinearKernelFunc<uint8_t, __fp16, 4> {
    inline static void resize_linear_kernel(
        const uint32_t* x0_ofs,
        const uint32_t* x1_ofs,
        const uint32_t* y0_ofs,
        const uint32_t* y1_ofs,
        const __fp16* alpha0_lst,
        const __fp16* alpha1_lst,
        const __fp16* beta0_lst,
        const __fp16* beta1_lst,
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const uint8_t* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        uint8_t* outData)
    {
        const int32_t channels = 4;
        const size_t e32_vl = vsetvlmax_e32m1();
        const int32_t num_unroll = e32_vl;

        __fp16 alpha0_lst_dup[outWidth * channels], alpha1_lst_dup[outWidth * channels];
        for (int32_t i = 0, ic = 0; i < outWidth; ++i, ic += channels) {
            alpha0_lst_dup[ic + 0] = alpha0_lst[i], alpha0_lst_dup[ic + 1] = alpha0_lst[i];
            alpha0_lst_dup[ic + 2] = alpha0_lst[i], alpha0_lst_dup[ic + 3] = alpha0_lst[i];
            alpha1_lst_dup[ic + 0] = alpha1_lst[i], alpha1_lst_dup[ic + 1] = alpha1_lst[i];
            alpha1_lst_dup[ic + 2] = alpha1_lst[i], alpha1_lst_dup[ic + 3] = alpha1_lst[i];
        }

        auto out = outData;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            auto in0 = inData + y0_ofs[out_y];
            auto in1 = inData + y1_ofs[out_y];
            float beta0 = beta0_lst[out_y];
            float beta1 = beta1_lst[out_y];

            const size_t vl = vsetvlmax_e8m1();
            vfloat16m2_t beta0_v = vfncvt_f_f_w_f16m2(vfmv_v_f_f32m4(beta0, vl), vl); // patch
            vfloat16m2_t beta1_v = vfncvt_f_f_w_f16m2(vfmv_v_f_f32m4(beta1, vl), vl); // patch

            int32_t out_x = 0, out_xc = 0;
            for (; out_x < outWidth; out_x += num_unroll, out_xc += num_unroll * channels) {
                const size_t vl = vsetvl_e8m1(outWidth * channels - out_xc);
                const size_t e32_vl = vl / channels;

                vuint32m1_t in_x0_v = vle32_v_u32m1(x0_ofs + out_x, e32_vl);
                vuint32m1_t in_x1_v = vle32_v_u32m1(x1_ofs + out_x, e32_vl);

                vfloat16m2_t data0_v = vfwcvt_f_xu_v_f16m2(vreinterpret_v_u32m1_u8m1(vlxwu_v_u32m1((uint32_t*)in0, in_x0_v, e32_vl)), vl);
                vfloat16m2_t data1_v = vfwcvt_f_xu_v_f16m2(vreinterpret_v_u32m1_u8m1(vlxwu_v_u32m1((uint32_t*)in0, in_x1_v, e32_vl)), vl);
                vfloat16m2_t data2_v = vfwcvt_f_xu_v_f16m2(vreinterpret_v_u32m1_u8m1(vlxwu_v_u32m1((uint32_t*)in1, in_x0_v, e32_vl)), vl);
                vfloat16m2_t data3_v = vfwcvt_f_xu_v_f16m2(vreinterpret_v_u32m1_u8m1(vlxwu_v_u32m1((uint32_t*)in1, in_x1_v, e32_vl)), vl);

                vfloat16m2_t alpha0_v = vle16_v_f16m2(alpha0_lst_dup + out_xc, vl);
                vfloat16m2_t alpha1_v = vle16_v_f16m2(alpha1_lst_dup + out_xc, vl);

                vfloat16m2_t p0_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(data0_v, alpha0_v, vl), alpha1_v, data1_v, vl);
                vfloat16m2_t p1_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(data2_v, alpha0_v, vl), alpha1_v, data3_v, vl);
                vfloat16m2_t out_v = vfmacc_vv_f16m2(vfmul_vv_f16m2(p0_v, beta0_v, vl), beta1_v, p1_v, vl);

                vse8_v_u8m1(out + out_xc, vfncvt_xu_f_w_u8m1(out_v, vl), vl);
            }
            out += outWidthStride;
        }
    }
};

template <bool resize_area>
struct ResizeLinearFunc<uint8_t, resize_area, 4> {
    static void resize_linear(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const uint8_t* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        uint8_t* outData)
    {
        typedef __fp16 facT;
        typedef uint8_t eT;
        constexpr int32_t channels = 4;

        const int32_t dim0 = outHeight;
        const int32_t dim1 = outWidth;
        const int32_t x_tabs_len = dim1 * 2;
        const int32_t y_tabs_len = dim0 * 2;

        uint32_t* buffer = (uint32_t*)malloc((x_tabs_len + y_tabs_len) * (sizeof(uint32_t) + sizeof(facT)));
        uint32_t *x_ofs = buffer, *y_ofs = buffer + x_tabs_len;
        facT *alpha_lst = (facT*)(y_ofs + y_tabs_len), *beta_lst = alpha_lst + x_tabs_len;

        uint32_t *x0_ofs = x_ofs, *x1_ofs = x0_ofs + dim1;
        uint32_t *y0_ofs = y_ofs, *y1_ofs = y_ofs + dim0;
        facT *alpha0_lst = alpha_lst, *alpha1_lst = alpha_lst + dim1;
        facT *beta0_lst = beta_lst, *beta1_lst = beta_lst + dim0;

        const int32_t x_ofs_factor = channels;
        const int32_t y_ofs_factor = inWidthStride;

        cal_resize_linear_ofs<facT, resize_area, false, true>(inWidth, outWidth, channels, x_ofs_factor, x0_ofs, x1_ofs, alpha0_lst, alpha1_lst);
        cal_resize_linear_ofs<facT, resize_area, false, true>(inHeight, outHeight, 1, y_ofs_factor, y0_ofs, y1_ofs, beta0_lst, beta1_lst);

        ResizeLinearKernelFunc<eT, facT, channels>::resize_linear_kernel(
            x0_ofs,
            x1_ofs,
            y0_ofs,
            y1_ofs,
            alpha0_lst,
            alpha1_lst,
            beta0_lst,
            beta1_lst,
            inHeight,
            inWidth,
            inWidthStride,
            inData,
            outHeight,
            outWidth,
            outWidthStride,
            outData);

        free(buffer);
    }
};

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    ResizeNearestPointFunc<uint8_t, 1>::resize_nearest_point(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    ResizeNearestPointFunc<uint8_t, 3>::resize_nearest_point(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    ResizeNearestPointFunc<uint8_t, 4>::resize_nearest_point(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<uint8_t, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        ResizeLinearFunc<uint8_t, false, 1>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<uint8_t, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        ResizeLinearFunc<uint8_t, false, 3>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}
template <>
::ppl::common::RetCode ResizeLinear<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<uint8_t, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        ResizeLinearFunc<uint8_t, false, 4>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<uint8_t, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else if (inHeight <= outHeight || inWidth <= outWidth) {
        ResizeLinearFunc<uint8_t, true, 1>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        resize_area<uint8_t, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<uint8_t, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else if (inHeight <= outHeight || inWidth <= outWidth) {
        ResizeLinearFunc<uint8_t, true, 3>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        resize_area<uint8_t, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<uint8_t, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else if (inHeight <= outHeight || inWidth <= outWidth) {
        ResizeLinearFunc<uint8_t, true, 4>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        resize_area<uint8_t, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

}
}
} // namespace ppl::cv::riscv