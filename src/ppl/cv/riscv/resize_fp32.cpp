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

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "ppl/cv/riscv/resize.h"
#include "ppl/cv/riscv/resize_common.h"
#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"
#include "util.h"
#include <algorithm>

namespace ppl {
namespace cv {
namespace riscv {

template <>
struct ResizeNearestPointFunc<float, 1> {
    static void resize_nearest_point(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const float* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        float* outData)
    {
        const int32_t num_unroll = 4;

        double scale_y = (double)inHeight / outHeight;
        float scale_x = (double)inWidth / outWidth;

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
struct ResizeNearestPointFunc<float, 3> {
    static void resize_nearest_point(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const float* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        float* outData)
    {
        constexpr int32_t channels = 3;

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

            for (int32_t out_x = 0, out_xc = 0; out_x < outWidth; ++out_x, out_xc += channels) {
                int32_t in_xc = in_x_ofs[out_x];
                out[out_xc] = in[in_xc];
                out[out_xc + 1] = in[in_xc + 1];
                out[out_xc + 2] = in[in_xc + 2];
            }
            out += outWidthStride;
        }
        free(buffer);
    }
};

template <>
struct ResizeNearestPointFunc<float, 4> {
    static void resize_nearest_point(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const float* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        float* outData)
    {
        const int32_t channels = 4;

        double scale_y = (double)inHeight / outHeight;
        double scale_x = (double)inWidth / outWidth;

        int32_t* buffer = (int32_t*)malloc((outWidth + outHeight) * sizeof(int32_t));

        int32_t* in_x_ofs = buffer;
        int32_t* in_y_ofs = in_x_ofs + outWidth;

        cal_resize_nearest_point_ofs<true>(outWidth, channels, scale_x, in_x_ofs);
        cal_resize_nearest_point_ofs<true>(outHeight, inWidthStride, scale_y, in_y_ofs);

        auto out = outData;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            auto in = inData + in_y_ofs[out_y];

            for (int32_t out_x = 0, out_xc = 0; out_x < outWidth; ++out_x, out_xc += channels) {
                int32_t in_xc = in_x_ofs[out_x];
                out[out_xc] = in[in_xc];
                out[out_xc + 1] = in[in_xc + 1];
                out[out_xc + 2] = in[in_xc + 2];
                out[out_xc + 3] = in[in_xc + 3];
            }
            out += outWidthStride;
        }
        free(buffer);
    }
};

template <>
void resize_linear_shrink2<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    const size_t vl = vsetvlmax_e32m4();
    const int32_t num_unroll = vl;

    const int32_t scale_x = 2;
    const int32_t scale_y = 2;
    const int32_t in_data_stride = scale_x * sizeof(float);
    const float avg_factor = 1. / (scale_x * scale_y);

    auto in = inData;
    auto out = outData;
    for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
        auto in0 = in;
        auto in1 = in + inWidthStride;

        int32_t out_x = 0, in_x = 0;
        for (; out_x < outWidth; out_x += num_unroll, in_x += num_unroll * scale_x) {
            const size_t vl = vsetvl_e32m4(outWidth - num_unroll);

            vfloat32m4_t data0_v = vlse32_v_f32m4(in0 + in_x + 0, in_data_stride, vl);
            vfloat32m4_t data1_v = vlse32_v_f32m4(in0 + in_x + 1, in_data_stride, vl);
            vfloat32m4_t data2_v = vlse32_v_f32m4(in1 + in_x + 0, in_data_stride, vl);
            vfloat32m4_t data3_v = vlse32_v_f32m4(in1 + in_x + 1, in_data_stride, vl);

            vfloat32m4_t sum_v = vfadd_vv_f32m4(vfadd_vv_f32m4(vfadd_vv_f32m4(data0_v, data1_v, vl), data2_v, vl), data3_v, vl);
            vfloat32m4_t out_v = vfmul_vf_f32m4(sum_v, avg_factor, vl);

            vse32_v_f32m4(out + out_x, out_v, vl);
        }

        in += inWidthStride * scale_y;
        out += outWidthStride;
    }
}

template <>
void resize_linear_shrink2<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    const int32_t channels = 3;
    const size_t vl = vsetvl_e32m1(channels);
    const int32_t num_unroll = 2;

    const int32_t scale_x = 2;
    const int32_t scale_y = 2;
    const float avg_factor = 1. / (scale_x * scale_y);

    auto in = inData;
    auto out = outData;
    for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
        auto in0 = in;
        auto in1 = in + inWidthStride;

        int32_t out_x = 0, in_xc = 0, out_xc = 0;
        for (; out_x <= outWidth - num_unroll; out_x += num_unroll, in_xc += scale_x * channels * num_unroll, out_xc += channels * num_unroll) {
            vfloat32m1_t data0_v = vle32_v_f32m1(in0 + in_xc + 0, vl);
            vfloat32m1_t data1_v = vle32_v_f32m1(in0 + in_xc + channels, vl);
            vfloat32m1_t data2_v = vle32_v_f32m1(in1 + in_xc + 0, vl);
            vfloat32m1_t data3_v = vle32_v_f32m1(in1 + in_xc + channels, vl);

            vfloat32m1_t sum0_v = vfadd_vv_f32m1(vfadd_vv_f32m1(data0_v, data1_v, vl), vfadd_vv_f32m1(data2_v, data3_v, vl), vl);
            vfloat32m1_t out0_v = vfmul_vf_f32m1(sum0_v, avg_factor, vl);

            vfloat32m1_t data4_v = vle32_v_f32m1(in0 + in_xc + num_unroll * channels + 0, vl);
            vfloat32m1_t data5_v = vle32_v_f32m1(in0 + in_xc + num_unroll * channels + channels, vl);
            vfloat32m1_t data6_v = vle32_v_f32m1(in1 + in_xc + num_unroll * channels + 0, vl);
            vfloat32m1_t data7_v = vle32_v_f32m1(in1 + in_xc + num_unroll * channels + channels, vl);

            vfloat32m1_t sum1_v = vfadd_vv_f32m1(vfadd_vv_f32m1(data4_v, data5_v, vl), vfadd_vv_f32m1(data6_v, data7_v, vl), vl);
            vfloat32m1_t out1_v = vfmul_vf_f32m1(sum1_v, avg_factor, vl);

            vse32_v_f32m1(out + out_xc, out0_v, vl);
            vse32_v_f32m1(out + out_xc + channels, out1_v, vl);
        }
        for (; out_x < outWidth; ++out_x, in_xc += scale_x * channels, out_xc += channels) {
            vfloat32m1_t data0_v = vle32_v_f32m1(in0 + in_xc + 0, vl);
            vfloat32m1_t data1_v = vle32_v_f32m1(in0 + in_xc + channels, vl);
            vfloat32m1_t data2_v = vle32_v_f32m1(in1 + in_xc + 0, vl);
            vfloat32m1_t data3_v = vle32_v_f32m1(in1 + in_xc + channels, vl);

            vfloat32m1_t sum_v = vfadd_vv_f32m1(vfadd_vv_f32m1(data0_v, data1_v, vl), vfadd_vv_f32m1(data2_v, data3_v, vl), vl);
            vfloat32m1_t out_v = vfmul_vf_f32m1(sum_v, avg_factor, vl);

            vse32_v_f32m1(out + out_xc, out_v, vl);
        }

        in += inWidthStride * scale_y;
        out += outWidthStride;
    }
}

template <>
void resize_linear_shrink2<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    const int32_t channels = 4;
    const size_t vl = vsetvl_e32m1(channels);
    const int32_t num_unroll = 2;

    const int32_t scale_x = 2;
    const int32_t scale_y = 2;
    const float avg_factor = 1. / (scale_x * scale_y);

    auto in = inData;
    auto out = outData;
    for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
        auto in0 = in;
        auto in1 = in + inWidthStride;

        int32_t out_x = 0, in_xc = 0, out_xc = 0;
        for (; out_x <= outWidth - num_unroll; out_x += num_unroll, in_xc += scale_x * channels * num_unroll, out_xc += channels * num_unroll) {
            vfloat32m1_t data0_v = vle32_v_f32m1(in0 + in_xc + 0, vl);
            vfloat32m1_t data1_v = vle32_v_f32m1(in0 + in_xc + channels, vl);
            vfloat32m1_t data2_v = vle32_v_f32m1(in1 + in_xc + 0, vl);
            vfloat32m1_t data3_v = vle32_v_f32m1(in1 + in_xc + channels, vl);

            vfloat32m1_t sum0_v = vfadd_vv_f32m1(vfadd_vv_f32m1(data0_v, data1_v, vl), vfadd_vv_f32m1(data2_v, data3_v, vl), vl);
            vfloat32m1_t out0_v = vfmul_vf_f32m1(sum0_v, avg_factor, vl);

            vfloat32m1_t data4_v = vle32_v_f32m1(in0 + in_xc + num_unroll * channels + 0, vl);
            vfloat32m1_t data5_v = vle32_v_f32m1(in0 + in_xc + num_unroll * channels + channels, vl);
            vfloat32m1_t data6_v = vle32_v_f32m1(in1 + in_xc + num_unroll * channels + 0, vl);
            vfloat32m1_t data7_v = vle32_v_f32m1(in1 + in_xc + num_unroll * channels + channels, vl);

            vfloat32m1_t sum1_v = vfadd_vv_f32m1(vfadd_vv_f32m1(data4_v, data5_v, vl), vfadd_vv_f32m1(data6_v, data7_v, vl), vl);
            vfloat32m1_t out1_v = vfmul_vf_f32m1(sum1_v, avg_factor, vl);

            vse32_v_f32m1(out + out_xc, out0_v, vl);
            vse32_v_f32m1(out + out_xc + channels, out1_v, vl);
        }
        for (; out_x < outWidth; ++out_x, in_xc += scale_x * channels, out_xc += channels) {
            vfloat32m1_t data0_v = vle32_v_f32m1(in0 + in_xc + 0, vl);
            vfloat32m1_t data1_v = vle32_v_f32m1(in0 + in_xc + channels, vl);
            vfloat32m1_t data2_v = vle32_v_f32m1(in1 + in_xc + 0, vl);
            vfloat32m1_t data3_v = vle32_v_f32m1(in1 + in_xc + channels, vl);

            vfloat32m1_t sum_v = vfadd_vv_f32m1(vfadd_vv_f32m1(data0_v, data1_v, vl), vfadd_vv_f32m1(data2_v, data3_v, vl), vl);
            vfloat32m1_t out_v = vfmul_vf_f32m1(sum_v, avg_factor, vl);

            vse32_v_f32m1(out + out_xc, out_v, vl);
        }

        in += inWidthStride * scale_y;
        out += outWidthStride;
    }
}

template <int32_t channels>
struct ResizeLinearKernelFunc<float, float, channels> {
    inline static void resize_linear_kernel(
        const uint32_t* x0_ofs,
        const uint32_t* x1_ofs,
        const uint32_t* y0_ofs,
        const uint32_t* y1_ofs,
        const float* alpha0_lst,
        const float* alpha1_lst,
        const float* beta0_lst,
        const float* beta1_lst,
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const float* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        float* outData)
    {
        const size_t vl = vsetvlmax_e32m4();
        const int32_t num_unroll = vl;

        vuint32m4_t in_x0_v, in_x1_v;
        vfloat32m4_t data0_v, data1_v, data2_v, data3_v;
        vfloat32m4_t alpha0_v, alpha1_v;
        vfloat32m4_t p0_v, p1_v;
        vfloat32m4_t out_v;

        auto out = outData;
        outWidth *= channels;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            auto in0 = inData + y0_ofs[out_y];
            auto in1 = inData + y1_ofs[out_y];
            float beta0 = beta0_lst[out_y];
            float beta1 = beta1_lst[out_y];

            int32_t out_x = 0;
            for (; out_x < outWidth; out_x += num_unroll) {
                const size_t vl = vsetvl_e32m4(outWidth - out_x);

                in_x0_v = vle32_v_u32m4(x0_ofs + out_x, vl);
                in_x1_v = vle32_v_u32m4(x1_ofs + out_x, vl);

                data0_v = vloxei32_v_f32m4(in0, in_x0_v, vl);
                data1_v = vloxei32_v_f32m4(in0, in_x1_v, vl);
                data2_v = vloxei32_v_f32m4(in1, in_x0_v, vl);
                data3_v = vloxei32_v_f32m4(in1, in_x1_v, vl);

                alpha0_v = vle32_v_f32m4(alpha0_lst + out_x, vl);
                alpha1_v = vle32_v_f32m4(alpha1_lst + out_x, vl);

                p0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(alpha0_v, data0_v, vl), alpha1_v, data1_v, vl);
                p1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(alpha0_v, data2_v, vl), alpha1_v, data3_v, vl);
                out_v = vfmacc_vf_f32m4(vfmul_vf_f32m4(p0_v, beta0, vl), beta1, p1_v, vl);

                vse32_v_f32m4(out + out_x, out_v, vl);
            }
            out += outWidthStride;
        }
    }
};

template <>
struct ResizeLinearKernelFunc<float, float, 3> {
    inline static void resize_linear_kernel(
        const uint32_t* x0_ofs,
        const uint32_t* x1_ofs,
        const uint32_t* y0_ofs,
        const uint32_t* y1_ofs,
        const float* alpha0_lst,
        const float* alpha1_lst,
        const float* beta0_lst,
        const float* beta1_lst,
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const float* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        float* outData)
    {
        constexpr int32_t channels = 3;
        const size_t vl = vsetvl_e32m1(channels);
        const int32_t num_unroll = 2;

        vfloat32m1_t data0_v, data1_v, data2_v, data3_v;
        vfloat32m1_t data4_v, data5_v, data6_v, data7_v;
        vfloat32m1_t p0_v, p1_v, p2_v, p3_v;
        vfloat32m1_t out0_v, out1_v;

        auto out = outData;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            auto in0 = inData + y0_ofs[out_y];
            auto in1 = inData + y1_ofs[out_y];
            float beta0 = beta0_lst[out_y];
            float beta1 = beta1_lst[out_y];

            int32_t out_x = 0;
            for (; out_x <= outWidth - num_unroll; out_x += num_unroll) {
                float alpha0 = alpha0_lst[out_x];
                float alpha1 = alpha1_lst[out_x];
                float alpha2 = alpha0_lst[out_x + 1];
                float alpha3 = alpha1_lst[out_x + 1];

                data0_v = vle32_v_f32m1(in0 + x0_ofs[out_x], vl);
                data1_v = vle32_v_f32m1(in0 + x1_ofs[out_x], vl);
                data2_v = vle32_v_f32m1(in1 + x0_ofs[out_x], vl);
                data3_v = vle32_v_f32m1(in1 + x1_ofs[out_x], vl);
                data4_v = vle32_v_f32m1(in0 + x0_ofs[out_x + 1], vl);
                data5_v = vle32_v_f32m1(in0 + x1_ofs[out_x + 1], vl);
                data6_v = vle32_v_f32m1(in1 + x0_ofs[out_x + 1], vl);
                data7_v = vle32_v_f32m1(in1 + x1_ofs[out_x + 1], vl);

                p0_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data0_v, alpha0, vl), alpha1, data1_v, vl);
                p1_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data2_v, alpha0, vl), alpha1, data3_v, vl);
                out0_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(p0_v, beta0, vl), beta1, p1_v, vl);
                p2_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data4_v, alpha2, vl), alpha3, data5_v, vl);
                p3_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data6_v, alpha2, vl), alpha3, data7_v, vl);
                out1_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(p2_v, beta0, vl), beta1, p3_v, vl);

                vse32_v_f32m1(out + (out_x * channels), out0_v, vl);
                vse32_v_f32m1(out + ((out_x + 1) * channels), out1_v, vl);
            }
            for (; out_x < outWidth; ++out_x) {
                float alpha0 = alpha0_lst[out_x];
                float alpha1 = alpha1_lst[out_x];

                data0_v = vle32_v_f32m1(in0 + x0_ofs[out_x], vl);
                data1_v = vle32_v_f32m1(in0 + x1_ofs[out_x], vl);
                data2_v = vle32_v_f32m1(in1 + x0_ofs[out_x], vl);
                data3_v = vle32_v_f32m1(in1 + x1_ofs[out_x], vl);

                p0_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data0_v, alpha0, vl), alpha1, data1_v, vl);
                p1_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data2_v, alpha0, vl), alpha1, data3_v, vl);
                out0_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(p0_v, beta0, vl), beta1, p1_v, vl);

                vse32_v_f32m1(out + (out_x * channels), out0_v, vl);
            }
            out += outWidthStride;
        }
    }
};

template <bool resize_area>
struct ResizeLinearFunc<float, resize_area, 3> {
    static void resize_linear(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const float* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        float* outData)
    {
        constexpr int32_t channels = 3;

        const int32_t dim0 = outHeight;
        const int32_t dim1 = outWidth;
        const int32_t x_tabs_len = dim1 * 2;
        const int32_t y_tabs_len = dim0 * 2;

        uint32_t* buffer = (uint32_t*)malloc((x_tabs_len + y_tabs_len) * (sizeof(uint32_t) + sizeof(float)));
        uint32_t *x_ofs = buffer, *y_ofs = buffer + x_tabs_len;
        float *alpha_lst = (float*)(y_ofs + y_tabs_len), *beta_lst = alpha_lst + x_tabs_len;

        uint32_t *x0_ofs = x_ofs, *x1_ofs = x0_ofs + dim1;
        uint32_t *y0_ofs = y_ofs, *y1_ofs = y_ofs + dim0;
        float *alpha0_lst = alpha_lst, *alpha1_lst = alpha_lst + dim1;
        float *beta0_lst = beta_lst, *beta1_lst = beta_lst + dim0;

        const int32_t x_ofs_factor = channels;
        const int32_t y_ofs_factor = inWidthStride;

        cal_resize_linear_ofs<float, resize_area, false, true>(inWidth, outWidth, channels, x_ofs_factor, x0_ofs, x1_ofs, alpha0_lst, alpha1_lst);
        cal_resize_linear_ofs<float, resize_area, false, true>(inHeight, outHeight, 1, y_ofs_factor, y0_ofs, y1_ofs, beta0_lst, beta1_lst);

        ResizeLinearKernelFunc<float, float, channels>::resize_linear_kernel(
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
struct ResizeLinearKernelFunc<float, float, 4> {
    inline static void resize_linear_kernel(
        const uint32_t* x0_ofs,
        const uint32_t* x1_ofs,
        const uint32_t* y0_ofs,
        const uint32_t* y1_ofs,
        const float* alpha0_lst,
        const float* alpha1_lst,
        const float* beta0_lst,
        const float* beta1_lst,
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const float* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        float* outData)
    {
        constexpr int32_t channels = 4;
        const size_t vl = vsetvl_e32m1(channels);
        const int32_t num_unroll = 2;

        vfloat32m1_t data0_v, data1_v, data2_v, data3_v;
        vfloat32m1_t data4_v, data5_v, data6_v, data7_v;
        vfloat32m1_t p0_v, p1_v, p2_v, p3_v;
        vfloat32m1_t out0_v, out1_v;

        auto out = outData;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            auto in0 = inData + y0_ofs[out_y];
            auto in1 = inData + y1_ofs[out_y];
            float beta0 = beta0_lst[out_y];
            float beta1 = beta1_lst[out_y];

            int32_t out_x = 0;
            for (; out_x <= outWidth - num_unroll; out_x += num_unroll) {
                float alpha0 = alpha0_lst[out_x];
                float alpha1 = alpha1_lst[out_x];
                float alpha2 = alpha0_lst[out_x + 1];
                float alpha3 = alpha1_lst[out_x + 1];

                data0_v = vle32_v_f32m1(in0 + x0_ofs[out_x], vl);
                data1_v = vle32_v_f32m1(in0 + x1_ofs[out_x], vl);
                data2_v = vle32_v_f32m1(in1 + x0_ofs[out_x], vl);
                data3_v = vle32_v_f32m1(in1 + x1_ofs[out_x], vl);
                data4_v = vle32_v_f32m1(in0 + x0_ofs[out_x + 1], vl);
                data5_v = vle32_v_f32m1(in0 + x1_ofs[out_x + 1], vl);
                data6_v = vle32_v_f32m1(in1 + x0_ofs[out_x + 1], vl);
                data7_v = vle32_v_f32m1(in1 + x1_ofs[out_x + 1], vl);

                p0_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data0_v, alpha0, vl), alpha1, data1_v, vl);
                p1_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data2_v, alpha0, vl), alpha1, data3_v, vl);
                out0_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(p0_v, beta0, vl), beta1, p1_v, vl);
                p2_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data4_v, alpha2, vl), alpha3, data5_v, vl);
                p3_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data6_v, alpha2, vl), alpha3, data7_v, vl);
                out1_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(p2_v, beta0, vl), beta1, p3_v, vl);

                vse32_v_f32m1(out + (out_x * channels), out0_v, vl);
                vse32_v_f32m1(out + ((out_x + 1) * channels), out1_v, vl);
            }
            for (; out_x < outWidth; ++out_x) {
                float alpha0 = alpha0_lst[out_x];
                float alpha1 = alpha1_lst[out_x];

                data0_v = vle32_v_f32m1(in0 + x0_ofs[out_x], vl);
                data1_v = vle32_v_f32m1(in0 + x1_ofs[out_x], vl);
                data2_v = vle32_v_f32m1(in1 + x0_ofs[out_x], vl);
                data3_v = vle32_v_f32m1(in1 + x1_ofs[out_x], vl);

                p0_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data0_v, alpha0, vl), alpha1, data1_v, vl);
                p1_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(data2_v, alpha0, vl), alpha1, data3_v, vl);
                out0_v = vfmacc_vf_f32m1(vfmul_vf_f32m1(p0_v, beta0, vl), beta1, p1_v, vl);

                vse32_v_f32m1(out + (out_x * channels), out0_v, vl);
            }
            out += outWidthStride;
        }
    }
};

template <bool resize_area>
struct ResizeLinearFunc<float, resize_area, 4> {
    static void resize_linear(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const float* inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        float* outData)
    {
        constexpr int32_t channels = 4;

        const int32_t dim0 = outHeight;
        const int32_t dim1 = outWidth;
        const int32_t x_tabs_len = dim1 * 2;
        const int32_t y_tabs_len = dim0 * 2;

        uint32_t* buffer = (uint32_t*)malloc((x_tabs_len + y_tabs_len) * (sizeof(uint32_t) + sizeof(float)));
        uint32_t *x_ofs = buffer, *y_ofs = buffer + x_tabs_len;
        float *alpha_lst = (float*)(y_ofs + y_tabs_len), *beta_lst = alpha_lst + x_tabs_len;

        uint32_t *x0_ofs = x_ofs, *x1_ofs = x0_ofs + dim1;
        uint32_t *y0_ofs = y_ofs, *y1_ofs = y_ofs + dim0;
        float *alpha0_lst = alpha_lst, *alpha1_lst = alpha_lst + dim1;
        float *beta0_lst = beta_lst, *beta1_lst = beta_lst + dim0;

        const int32_t x_ofs_factor = channels;
        const int32_t y_ofs_factor = inWidthStride;

        cal_resize_linear_ofs<float, resize_area, false, true>(inWidth, outWidth, channels, x_ofs_factor, x0_ofs, x1_ofs, alpha0_lst, alpha1_lst);
        cal_resize_linear_ofs<float, resize_area, false, true>(inHeight, outHeight, 1, y_ofs_factor, y0_ofs, y1_ofs, beta0_lst, beta1_lst);

        ResizeLinearKernelFunc<float, float, channels>::resize_linear_kernel(
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
::ppl::common::RetCode ResizeNearestPoint<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    ResizeNearestPointFunc<float, 1>::resize_nearest_point(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    ResizeNearestPointFunc<float, 3>::resize_nearest_point(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeNearestPoint<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    ResizeNearestPointFunc<float, 4>::resize_nearest_point(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode ResizeLinear<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<float, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        ResizeLinearFunc<float, false, 1>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeLinear<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<float, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        ResizeLinearFunc<float, false, 3>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}
template <>
::ppl::common::RetCode ResizeLinear<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<float, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        ResizeLinearFunc<float, false, 4>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<float, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else if (inHeight <= outHeight || inWidth <= outWidth) {
        ResizeLinearFunc<float, true, 1>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        resize_area<float, 1>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<float, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else if (inHeight <= outHeight || inWidth <= outWidth) {
        ResizeLinearFunc<float, true, 3>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        resize_area<float, 3>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode ResizeArea<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData)
{
    if (nullptr == outData || nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 || inWidthStride < inWidth || outWidthStride < outWidth) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (inHeight / outHeight == 2 && inWidth / outWidth == 2 && inHeight % outHeight == 0 && inWidth % outWidth == 0) {
        resize_linear_shrink2<float, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else if (inHeight <= outHeight || inWidth <= outWidth) {
        ResizeLinearFunc<float, true, 4>::resize_linear(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    } else {
        resize_area<float, 4>(inHeight, inWidth, inWidthStride, inData, outHeight, outWidth, outWidthStride, outData);
        return ppl::common::RC_SUCCESS;
    }
}

}
}
} // namespace ppl::cv::riscv