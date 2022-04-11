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

#ifndef PPL_CV_RISCV_RESIZE_COMMON_H_
#define PPL_CV_RISCV_RESIZE_COMMON_H_

#include <type_traits>
#include <string.h>
#include "ppl/cv/types.h"
#include "ppl/cv/riscv/util.h"
#include "ppl/cv/riscv/typetraits.h"

namespace ppl {
namespace cv {
namespace riscv {

#define MAX_ESIZE 16

static inline void wr_fcsr_rm(const uint32_t rm)
{
    asm volatile(
        "fsrm %[RM] \n\t"
        :
        : [RM] "r"(rm)
        :);
}

template <bool mul_alpha>
inline static void cal_resize_nearest_point_ofs(const int32_t outLength, const int32_t alpha, const float scale, int32_t *ofs)
{
    const int32_t num_unroll = 4;
    int32_t out_x = 0;

    for (; out_x <= outLength - num_unroll; out_x += num_unroll) {
        int32_t data0 = (int32_t)((out_x + 0) * scale);
        int32_t data1 = (int32_t)((out_x + 1) * scale);
        int32_t data2 = (int32_t)((out_x + 2) * scale);
        int32_t data3 = (int32_t)((out_x + 3) * scale);
        ofs[out_x + 0] = mul_alpha ? data0 * alpha : data0;
        ofs[out_x + 1] = mul_alpha ? data1 * alpha : data1;
        ofs[out_x + 2] = mul_alpha ? data2 * alpha : data2;
        ofs[out_x + 3] = mul_alpha ? data3 * alpha : data3;
    }
    for (; out_x < outLength; ++out_x) {
        int32_t data = (int32_t)(out_x * scale);
        ofs[out_x] = mul_alpha ? data * alpha : data;
    }
}

template <typename eT, int32_t channels>
struct ResizeNearestPointFunc {
    static void resize_nearest_point(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const eT *inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        eT *outData)
    {
        double scale_y = (double)inHeight / outHeight;
        double scale_x = (double)inWidth / outWidth;

        int32_t *buffer = (int32_t *)malloc((outWidth + outHeight) * sizeof(int32_t));

        int32_t *in_x_ofs = buffer;
        int32_t *in_y_ofs = in_x_ofs + outWidth;

        cal_resize_nearest_point_ofs<true>(outWidth, channels, scale_x, in_x_ofs);
        cal_resize_nearest_point_ofs<true>(outHeight, inWidthStride, scale_y, in_y_ofs);

        auto out = outData;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            int32_t in_y = in_y_ofs[out_y];
            auto in = inData + in_y;

            for (int32_t out_x = 0, out_xc = 0; out_x < outWidth; ++out_x, out_xc += channels) {
                int32_t in_xc = in_x_ofs[out_x];
                for (int32_t c = 0; c < channels; ++c) {
                    out[out_xc + c] = in[in_xc + c];
                }
            }
            out += outWidthStride;
        }
        free(buffer);
    }
};

template <typename eT, int32_t channels>
void resize_linear_shrink2(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const eT *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    eT *outData)
{
    auto in = inData;
    auto out = outData;
    for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
        auto in0 = in;
        auto in1 = in + inWidthStride;

        for (int32_t out_x = 0, out_xc = 0; out_x < outWidth; ++out_x, out_xc += channels) {
            int32_t x0 = out_xc * 2;
            int32_t x1 = x0 + channels;

            for (int32_t c = 0; c < channels; ++c) {
                if (std::is_same<eT, float>::value) {
                    out[out_xc + c] = (in0[x0 + c] + in0[x1 + c] + in1[x0 + c] + in1[x1 + c]) * 0.25;
                } else if (std::is_same<eT, uint8_t>::value) {
                    out[out_xc + c] = (uint16_t)((uint16_t)in0[x0 + c] + in0[x1 + c] + in1[x0 + c] + in1[x1 + c]) >> 2;
                } else {
                    out[out_xc + c] = ((float)in0[x0 + c] + in0[x1 + c] + in1[x0 + c] + in1[x1 + c]) * 0.25;
                }
            }
        }

        in += inWidthStride * 2;
        out += outWidthStride;
    }
}

template <typename facT, bool resize_area, bool with_channels, bool with_fac>
inline static void cal_resize_linear_ofs(
    const int32_t inLength,
    const int32_t outLength,
    const int32_t channels,
    const int32_t factor,
    uint32_t *x0_ofs,
    uint32_t *x1_ofs,
    facT *alpha0_lst,
    facT *alpha1_lst)
{
    const double scale = (double)inLength / outLength;
    const double inv_scale = 1. / scale;

    for (int32_t out_x = 0; out_x < outLength; ++out_x) {
        int32_t in_sx;
        float dx;

        if (resize_area) {
            in_sx = img_floor(out_x * scale);
            dx = (float)((out_x + 1) - (in_sx + 1) * inv_scale);
            dx = (dx <= 0) ? 0.f : dx - img_floor(dx);
        } else {
            float in_fx = (out_x + 0.5) * scale - 0.5;
            in_sx = img_floor(in_fx);
            dx = in_fx - in_sx;
        }

        int32_t in_sx0 = in_sx;
        if (in_sx0 < 0) dx = 0, in_sx0 = 0;

        int32_t in_sx1 = in_sx0 + 1;
        if (in_sx1 >= inLength) dx = 0, in_sx1 = inLength - 1;

        if (with_channels) {
            for (int32_t c = 0; c < channels; ++c) {
                x0_ofs[out_x * channels + c] = (int32_t)(in_sx0 * channels + c) * (with_fac ? factor : 1);
                x1_ofs[out_x * channels + c] = (int32_t)(in_sx1 * channels + c) * (with_fac ? factor : 1);
                alpha0_lst[out_x * channels + c] = 1 - dx;
                alpha1_lst[out_x * channels + c] = dx;
            }
        } else {
            x0_ofs[out_x] = (int32_t)in_sx0 * (with_fac ? factor : 1);
            x1_ofs[out_x] = (int32_t)in_sx1 * (with_fac ? factor : 1);
            alpha0_lst[out_x] = 1 - dx;
            alpha1_lst[out_x] = dx;
        }
    }
}

template <typename eT, typename facT, int32_t channels>
struct ResizeLinearKernelFunc {
    inline static void resize_linear_kernel(
        const uint32_t *x0_ofs,
        const uint32_t *x1_ofs,
        const uint32_t *y0_ofs,
        const uint32_t *y1_ofs,
        const facT *alpha0_lst,
        const facT *alpha1_lst,
        const facT *beta0_lst,
        const facT *beta1_lst,
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const eT *inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        eT *outData)
    {
        constexpr int32_t num_unroll = 4;
        facT p0[num_unroll], p1[num_unroll];

        auto out = outData;
        outWidth *= channels;
        for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
            auto in0 = inData + y0_ofs[out_y];
            auto in1 = inData + y1_ofs[out_y];
            facT beta0 = beta0_lst[out_y];
            facT beta1 = beta1_lst[out_y];

            int32_t out_x = 0;
            for (; out_x <= outWidth - num_unroll; out_x += num_unroll) {
                eT data0[] = {in0[x0_ofs[out_x + 0]], in0[x0_ofs[out_x + 1]], in0[x0_ofs[out_x + 2]], in0[x0_ofs[out_x + 3]]};
                eT data1[] = {in0[x1_ofs[out_x + 0]], in0[x1_ofs[out_x + 1]], in0[x1_ofs[out_x + 2]], in0[x1_ofs[out_x + 3]]};
                eT data2[] = {in1[x0_ofs[out_x + 0]], in1[x0_ofs[out_x + 1]], in1[x0_ofs[out_x + 2]], in1[x0_ofs[out_x + 3]]};
                eT data3[] = {in1[x1_ofs[out_x + 0]], in1[x1_ofs[out_x + 1]], in1[x1_ofs[out_x + 2]], in1[x1_ofs[out_x + 3]]};

                p0[0] = alpha0_lst[out_x + 0] * data0[0] + alpha1_lst[out_x + 0] * data1[0];
                p0[1] = alpha0_lst[out_x + 1] * data0[1] + alpha1_lst[out_x + 1] * data1[1];
                p0[2] = alpha0_lst[out_x + 2] * data0[2] + alpha1_lst[out_x + 2] * data1[2];
                p0[3] = alpha0_lst[out_x + 3] * data0[3] + alpha1_lst[out_x + 3] * data1[3];

                p1[0] = alpha0_lst[out_x + 0] * data2[0] + alpha1_lst[out_x + 0] * data3[0];
                p1[1] = alpha0_lst[out_x + 1] * data2[1] + alpha1_lst[out_x + 1] * data3[1];
                p1[2] = alpha0_lst[out_x + 2] * data2[2] + alpha1_lst[out_x + 2] * data3[2];
                p1[3] = alpha0_lst[out_x + 3] * data2[3] + alpha1_lst[out_x + 3] * data3[3];

                out[out_x + 0] = img_sat_cast(beta0 * p0[0] + beta1 * p1[0]);
                out[out_x + 1] = img_sat_cast(beta0 * p0[1] + beta1 * p1[1]);
                out[out_x + 2] = img_sat_cast(beta0 * p0[2] + beta1 * p1[2]);
                out[out_x + 3] = img_sat_cast(beta0 * p0[3] + beta1 * p1[3]);
            }
            for (; out_x < outWidth; ++out_x) {
                p0[0] = alpha0_lst[out_x] * in0[x0_ofs[out_x]] + alpha1_lst[out_x] * in0[x1_ofs[out_x]];
                p1[0] = alpha0_lst[out_x] * in1[x0_ofs[out_x]] + alpha1_lst[out_x] * in1[x1_ofs[out_x]];
                out[out_x] = img_sat_cast(beta0 * p0[0] + beta1 * p1[0]);
            }
            out += outWidthStride;
        }
    }
};

template <typename eT, bool resize_area, int32_t channels>
struct ResizeLinearFunc {
    static void resize_linear(
        int32_t inHeight,
        int32_t inWidth,
        int32_t inWidthStride,
        const eT *inData,
        int32_t outHeight,
        int32_t outWidth,
        int32_t outWidthStride,
        eT *outData)
    {
        typedef float facT;

        const int32_t dim0 = outHeight;
        const int32_t dim1 = outWidth * channels;
        const int32_t x_tabs_len = dim1 * 2;
        const int32_t y_tabs_len = dim0 * 2;

        uint32_t *buffer = (uint32_t *)malloc((x_tabs_len + y_tabs_len) * (sizeof(uint32_t) + sizeof(facT)));
        uint32_t *x_ofs = buffer, *y_ofs = buffer + x_tabs_len;
        facT *alpha_lst = (facT *)(y_ofs + y_tabs_len), *beta_lst = alpha_lst + x_tabs_len;

        uint32_t *x0_ofs = x_ofs, *x1_ofs = x0_ofs + dim1;
        uint32_t *y0_ofs = y_ofs, *y1_ofs = y_ofs + dim0;
        facT *alpha0_lst = alpha_lst, *alpha1_lst = alpha_lst + dim1;
        facT *beta0_lst = beta_lst, *beta1_lst = beta_lst + dim0;

        const int32_t x_ofs_factor = sizeof(int32_t) / sizeof(int8_t);
        const int32_t y_ofs_factor = inWidthStride;

        cal_resize_linear_ofs<facT, resize_area, true, std::is_same<eT, float>::value>(inWidth, outWidth, channels, x_ofs_factor, x0_ofs, x1_ofs, alpha0_lst, alpha1_lst);
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

struct ResizeAreaTabs {
    int32_t in_start, in_end;
    float alpha_start, alpha_end;
    void load(double fstart, double fend)
    {
        in_start = (int32_t)fstart;
        alpha_start = 1. - (fstart - in_start);
        in_end = (int32_t)fend;
        alpha_end = fend - in_end;
        if (alpha_end == 0) {
            in_end -= 1;
            alpha_end = 1;
        }
        if (in_start == in_end) {
            alpha_start = alpha_end + alpha_start - 1;
        }
    }
};

template <bool mul_channels>
static float load_resize_area_tabs(
    int32_t inLength,
    int32_t outLength,
    int32_t channels,
    ResizeAreaTabs *tabs)
{
    double scale_len = (double)inLength / outLength;

    double fstart = 0, fend = 0;
    for (int32_t i = 0; i < outLength - 1; ++i) {
        fend = fstart + scale_len;
        tabs[i].load(fstart, fend);
        fstart = fend;
        if (mul_channels) {
            tabs[i].in_end *= channels;
            tabs[i].in_start *= channels;
        }
    }
    tabs[outLength - 1].load(fstart, inLength);
    if (mul_channels) {
        tabs[outLength - 1].in_end *= channels;
        tabs[outLength - 1].in_start *= channels;
    }
    return scale_len;
}

template <typename eT, bool y_start, bool with_beta>
inline void cal_resize_area_out_line(
    const eT *in_line,
    const int32_t outWidth,
    const int32_t channels,
    const float beta,
    const ResizeAreaTabs *xtabs,
    float *sum_buf)
{
    for (int32_t out_x = 0; out_x < outWidth; ++out_x) {
        const ResizeAreaTabs &xtab = xtabs[out_x];
        auto sum_buf_dst = sum_buf + out_x * channels;

        {
            for (int32_t c = 0; c < channels; ++c) {
                auto start_elem = in_line[xtab.in_start + c] * xtab.alpha_start;
                start_elem = with_beta ? start_elem * beta : start_elem;
                if (y_start) {
                    sum_buf_dst[c] = start_elem;
                } else {
                    sum_buf_dst[c] += start_elem;
                }
            }
        }
        if (xtab.in_start != xtab.in_end) {
            for (int32_t j = xtab.in_start + channels; j < xtab.in_end; j += channels) {
                for (int32_t c = 0; c < channels; ++c) {
                    sum_buf_dst[c] += with_beta ? in_line[j + c] * beta : in_line[j + c];
                }
            }
            {
                for (int32_t c = 0; c < channels; ++c) {
                    auto end_elem = in_line[xtab.in_end + c] * xtab.alpha_end;
                    sum_buf_dst[c] += with_beta ? end_elem * beta : end_elem;
                }
            }
        }
    }
}

template <typename eT, int32_t channels>
void resize_area(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const eT *inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    eT *outData)
{
    ResizeAreaTabs *ytabs = (ResizeAreaTabs *)malloc(outHeight * sizeof(ResizeAreaTabs));
    ResizeAreaTabs *xtabs = (ResizeAreaTabs *)malloc(outWidth * sizeof(ResizeAreaTabs));
    float *sum = (float *)malloc(outWidth * channels * sizeof(float));

    float scale_y = load_resize_area_tabs<false>(inHeight, outHeight, channels, ytabs);
    float scale_x = load_resize_area_tabs<true>(inWidth, outWidth, channels, xtabs);
    float avg_gamma = 1. / (scale_y * scale_x);

    for (int32_t out_y = 0; out_y < outHeight; ++out_y) {
        ResizeAreaTabs &ytab = ytabs[out_y];
        auto out = outData + out_y * outWidthStride;

        {
            auto in = inData + ytab.in_start * inWidthStride;
            cal_resize_area_out_line<eT, true, true>(in, outWidth, channels, ytab.alpha_start, xtabs, sum);
        }
        if (ytab.in_start != ytab.in_end) {
            for (int32_t in_y = ytab.in_start + 1; in_y < ytab.in_end; ++in_y) {
                auto in = inData + in_y * inWidthStride;
                cal_resize_area_out_line<eT, false, false>(in, outWidth, channels, 1, xtabs, sum);
            }
            {
                auto in = inData + ytab.in_end * inWidthStride;
                cal_resize_area_out_line<eT, false, true>(in, outWidth, channels, ytab.alpha_end, xtabs, sum);
            }
        }
        for (int32_t j = 0; j < outWidth * channels; ++j) {
            out[j] = (eT)(sum[j] * avg_gamma);
        }
    }

    free(ytabs);
    free(xtabs);
    free(sum);
}

}
}
} // namespace ppl::cv::riscv

#endif //! __ST_HPC_PPL_CV_RISCV_RISIZE_COMMON_H_