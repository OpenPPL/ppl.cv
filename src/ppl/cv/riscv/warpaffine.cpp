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

#include "ppl/cv/riscv/warpaffine.h"
#include "ppl/cv/riscv/typetraits.h"
#include "ppl/cv/riscv/util.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include <string.h>
#include <cmath>

#include <functional>
#include <vector>
#include <limits.h>

namespace ppl {
namespace cv {
namespace riscv {

static inline void wr_fcsr_rm(const uint32_t rm)
{
    asm volatile(
        "fsrm %[RM] \n\t"
        :
        : [RM] "r"(rm)
        :);
}

static inline int32_t rd_fcsr_rm()
{
    int32_t rm;
    asm volatile(
        "frrm %[RM] \n\t"
        : [RM] "=r"(rm)
        :
        :);
    return rm;
}

static inline int32_t saturate_cast(double value)
{
    int32_t round2zero = (int32_t)value;
    if (value >= 0) {
        return (value - round2zero != 0.5) ? (int32_t)(value + 0.5) : round2zero % 2 == 0 ? round2zero
                                                                                          : round2zero + 1;
    } else {
        return (round2zero - value != 0.5) ? (int32_t)(value - 0.5) : round2zero % 2 == 0 ? round2zero
                                                                                          : round2zero - 1;
    }
}
static inline int32_t floor(float value)
{
    int32_t i = (int32_t)value;
    return i - (i > value);
}

static inline short saturate_cast(int32_t v)
{
    return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX
                                                                               : SHRT_MIN);
}
template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
struct WarpaffineNearestCalIdx {
    static inline void f(
        const int32_t* sx_buf,
        const int32_t* sy_buf,
        const int32_t base_x,
        const int32_t base_y,
        const int32_t inHeight,
        const int32_t inWidth,
        const int32_t inWidthStride,
        int32_t* idx_src_buf,
        const int32_t idx_src_buf_len)
    {
        const size_t vl = vsetvlmax_e32m4();
        const int32_t num_cal_idx_unroll = vl;

        for (int32_t j = 0; j < idx_src_buf_len; j += num_cal_idx_unroll) {
            const size_t vl = vsetvl_e32m4(idx_src_buf_len - j);

            vint32m4_t sx_v = vsra_vx_i32m4(vadd_vx_i32m4(vle32_v_i32m4(sx_buf + j, vl), base_x, vl), 10, vl);
            vint32m4_t sy_v = vsra_vx_i32m4(vadd_vx_i32m4(vle32_v_i32m4(sy_buf + j, vl), base_y, vl), 10, vl);

            if (borderMode == ppl::cv::BORDER_CONSTANT || borderMode == ppl::cv::BORDER_TRANSPARENT) {
                vint32m4_t idx_src_v = vmacc_vx_i32m4(vmul_vx_i32m4(sy_v, inWidthStride, vl), nc, sx_v, vl);
                vbool8_t flag0_v = vmsgt_vx_i32m4_b8(sx_v, -1, vl);
                vbool8_t flag1_v = vmsle_vx_i32m4_b8(sx_v, inWidth - 1, vl);
                vbool8_t flag2_v = vmsgt_vx_i32m4_b8(sy_v, -1, vl);
                vbool8_t flag3_v = vmsle_vx_i32m4_b8(sy_v, inHeight - 1, vl);
                vbool8_t nflag_v = vmnot_m_b8(vmand_mm_b8(vmand_mm_b8(vmand_mm_b8(flag0_v, flag1_v, vl), flag2_v, vl), flag3_v, vl), vl);

                idx_src_v = vmerge_vxm_i32m4(nflag_v, idx_src_v, -1, vl);
                vse32_v_i32m4(idx_src_buf + j, idx_src_v, vl);
            } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                sx_v = vmax_vx_i32m4(vmin_vx_i32m4(sx_v, inWidth - 1, vl), 0, vl);
                sy_v = vmax_vx_i32m4(vmin_vx_i32m4(sy_v, inHeight - 1, vl), 0, vl);
                vint32m4_t idx_src_v = vmacc_vx_i32m4(vmul_vx_i32m4(sy_v, inWidthStride, vl), nc, sx_v, vl);
                vse32_v_i32m4(idx_src_buf + j, idx_src_v, vl);
            }
        }
    }
};

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
struct WarpaffineNearestLoadWithIdx {
    static inline void f(
        const T* src,
        const int32_t* idx_src_buf,
        const int32_t num_idx,
        T* out,
        const T delta)
    {
        for (int32_t j = 0; j < num_idx; j++) {
            if (borderMode == ppl::cv::BORDER_CONSTANT) {
                if (nc > 0) out[j * nc + 0] = idx_src_buf[j] < 0 ? delta : src[idx_src_buf[j] + 0];
                if (nc > 1) out[j * nc + 1] = idx_src_buf[j] < 0 ? delta : src[idx_src_buf[j] + 1];
                if (nc > 2) out[j * nc + 2] = idx_src_buf[j] < 0 ? delta : src[idx_src_buf[j] + 2];
                if (nc > 3) out[j * nc + 3] = idx_src_buf[j] < 0 ? delta : src[idx_src_buf[j] + 3];
            } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                if (nc > 0) out[j * nc + 0] = src[idx_src_buf[j] + 0];
                if (nc > 1) out[j * nc + 1] = src[idx_src_buf[j] + 1];
                if (nc > 2) out[j * nc + 2] = src[idx_src_buf[j] + 2];
                if (nc > 3) out[j * nc + 3] = src[idx_src_buf[j] + 3];
            } else if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                if (idx_src_buf[j] >= 0) {
                    if (nc > 0) out[j * nc + 0] = src[idx_src_buf[j] + 0];
                    if (nc > 1) out[j * nc + 1] = src[idx_src_buf[j] + 1];
                    if (nc > 2) out[j * nc + 2] = src[idx_src_buf[j] + 2];
                    if (nc > 3) out[j * nc + 3] = src[idx_src_buf[j] + 3];
                }
            }
        }
    }
};

template <int32_t nc, ppl::cv::BorderType borderMode>
struct WarpaffineNearestLoadWithIdx<float, nc, borderMode> {
    typedef float T;
    static inline void f(
        const T* src,
        const int32_t* idx_src_buf,
        const int32_t num_idx,
        T* out,
        const T delta)
    {
        bool is_border_constant = borderMode == ppl::cv::BORDER_CONSTANT;
        bool is_border_transparent = borderMode == ppl::cv::BORDER_TRANSPARENT;

        if (nc == 1) {
            int32_t j = 0;
            for (; j <= num_idx - 8; j += 8) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) out[j + 0] = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta : src[idx_src_buf[j + 0]];
                if (!is_border_transparent || (idx_src_buf[j + 1] >= 0)) out[j + 1] = (is_border_constant && (idx_src_buf[j + 1] < 0)) ? delta : src[idx_src_buf[j + 1]];
                if (!is_border_transparent || (idx_src_buf[j + 2] >= 0)) out[j + 2] = (is_border_constant && (idx_src_buf[j + 2] < 0)) ? delta : src[idx_src_buf[j + 2]];
                if (!is_border_transparent || (idx_src_buf[j + 3] >= 0)) out[j + 3] = (is_border_constant && (idx_src_buf[j + 3] < 0)) ? delta : src[idx_src_buf[j + 3]];
                if (!is_border_transparent || (idx_src_buf[j + 4] >= 0)) out[j + 4] = (is_border_constant && (idx_src_buf[j + 4] < 0)) ? delta : src[idx_src_buf[j + 4]];
                if (!is_border_transparent || (idx_src_buf[j + 5] >= 0)) out[j + 5] = (is_border_constant && (idx_src_buf[j + 5] < 0)) ? delta : src[idx_src_buf[j + 5]];
                if (!is_border_transparent || (idx_src_buf[j + 6] >= 0)) out[j + 6] = (is_border_constant && (idx_src_buf[j + 6] < 0)) ? delta : src[idx_src_buf[j + 6]];
                if (!is_border_transparent || (idx_src_buf[j + 7] >= 0)) out[j + 7] = (is_border_constant && (idx_src_buf[j + 7] < 0)) ? delta : src[idx_src_buf[j + 7]];
            }
            for (; j < num_idx; ++j) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) out[j + 0] = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta : src[idx_src_buf[j + 0]];
            }
        } else { // nc == 3 or nc == 4
            vfloat32m1_t delta_v;
            if (is_border_constant) {
                delta_v = vfmv_v_f_f32m1(delta, nc);
            }

            int32_t j = 0;
            for (; j <= num_idx - 2; j += 2) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) vse32_v_f32m1(out + j * nc, (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta_v : vle32_v_f32m1(src + idx_src_buf[j + 0], nc), nc);
                if (!is_border_transparent || (idx_src_buf[j + 1] >= 0)) vse32_v_f32m1(out + j * nc + nc, (is_border_constant && (idx_src_buf[j + 1] < 0)) ? delta_v : vle32_v_f32m1(src + idx_src_buf[j + 1], nc), nc);
            }
            for (; j < num_idx; ++j) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) vse32_v_f32m1(out + j * nc, (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta_v : vle32_v_f32m1(src + idx_src_buf[j + 0], nc), nc);
            }
        }
    }
};

template <int32_t nc, ppl::cv::BorderType borderMode>
struct WarpaffineNearestLoadWithIdx<uint8_t, nc, borderMode> {
    typedef uint8_t T;
    static inline void f(
        const T* src,
        const int32_t* idx_src_buf,
        const int32_t num_idx,
        T* out,
        const T delta)
    {
        bool is_border_constant = borderMode == ppl::cv::BORDER_CONSTANT;
        bool is_border_transparent = borderMode == ppl::cv::BORDER_TRANSPARENT;

        if (nc == 1) {
            int32_t j = 0;
            for (; j <= num_idx - 16; j += 16) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) out[j + 0] = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta : src[idx_src_buf[j + 0]];
                if (!is_border_transparent || (idx_src_buf[j + 1] >= 0)) out[j + 1] = (is_border_constant && (idx_src_buf[j + 1] < 0)) ? delta : src[idx_src_buf[j + 1]];
                if (!is_border_transparent || (idx_src_buf[j + 2] >= 0)) out[j + 2] = (is_border_constant && (idx_src_buf[j + 2] < 0)) ? delta : src[idx_src_buf[j + 2]];
                if (!is_border_transparent || (idx_src_buf[j + 3] >= 0)) out[j + 3] = (is_border_constant && (idx_src_buf[j + 3] < 0)) ? delta : src[idx_src_buf[j + 3]];
                if (!is_border_transparent || (idx_src_buf[j + 4] >= 0)) out[j + 4] = (is_border_constant && (idx_src_buf[j + 4] < 0)) ? delta : src[idx_src_buf[j + 4]];
                if (!is_border_transparent || (idx_src_buf[j + 5] >= 0)) out[j + 5] = (is_border_constant && (idx_src_buf[j + 5] < 0)) ? delta : src[idx_src_buf[j + 5]];
                if (!is_border_transparent || (idx_src_buf[j + 6] >= 0)) out[j + 6] = (is_border_constant && (idx_src_buf[j + 6] < 0)) ? delta : src[idx_src_buf[j + 6]];
                if (!is_border_transparent || (idx_src_buf[j + 7] >= 0)) out[j + 7] = (is_border_constant && (idx_src_buf[j + 7] < 0)) ? delta : src[idx_src_buf[j + 7]];
                if (!is_border_transparent || (idx_src_buf[j + 8] >= 0)) out[j + 8] = (is_border_constant && (idx_src_buf[j + 8] < 0)) ? delta : src[idx_src_buf[j + 8]];
                if (!is_border_transparent || (idx_src_buf[j + 9] >= 0)) out[j + 9] = (is_border_constant && (idx_src_buf[j + 9] < 0)) ? delta : src[idx_src_buf[j + 9]];
                if (!is_border_transparent || (idx_src_buf[j + 10] >= 0)) out[j + 10] = (is_border_constant && (idx_src_buf[j + 10] < 0)) ? delta : src[idx_src_buf[j + 10]];
                if (!is_border_transparent || (idx_src_buf[j + 11] >= 0)) out[j + 11] = (is_border_constant && (idx_src_buf[j + 11] < 0)) ? delta : src[idx_src_buf[j + 11]];
                if (!is_border_transparent || (idx_src_buf[j + 12] >= 0)) out[j + 12] = (is_border_constant && (idx_src_buf[j + 12] < 0)) ? delta : src[idx_src_buf[j + 12]];
                if (!is_border_transparent || (idx_src_buf[j + 13] >= 0)) out[j + 13] = (is_border_constant && (idx_src_buf[j + 13] < 0)) ? delta : src[idx_src_buf[j + 13]];
                if (!is_border_transparent || (idx_src_buf[j + 14] >= 0)) out[j + 14] = (is_border_constant && (idx_src_buf[j + 14] < 0)) ? delta : src[idx_src_buf[j + 14]];
                if (!is_border_transparent || (idx_src_buf[j + 15] >= 0)) out[j + 15] = (is_border_constant && (idx_src_buf[j + 15] < 0)) ? delta : src[idx_src_buf[j + 15]];
            }
            for (; j < num_idx; ++j) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) out[j + 0] = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta : src[idx_src_buf[j + 0]];
            }
        } else if (nc == 3) {
            uint8_t delta_e32[] = {delta, delta, delta, delta};
            int32_t j = 0;
            for (; j <= num_idx - 1 - 4; j += 4) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) *(uint16_t*)(out + (j + 0) * nc) = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? *(uint16_t*)delta_e32 : *(uint16_t*)(src + idx_src_buf[j + 0]);
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) *(uint8_t*)(out + (j + 0) * nc + 2) = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? *(uint8_t*)delta_e32 : *(uint8_t*)(src + idx_src_buf[j + 0] + 2);
                if (!is_border_transparent || (idx_src_buf[j + 1] >= 0)) *(uint16_t*)(out + (j + 1) * nc) = (is_border_constant && (idx_src_buf[j + 1] < 0)) ? *(uint16_t*)delta_e32 : *(uint16_t*)(src + idx_src_buf[j + 1]);
                if (!is_border_transparent || (idx_src_buf[j + 1] >= 0)) *(uint8_t*)(out + (j + 1) * nc + 2) = (is_border_constant && (idx_src_buf[j + 1] < 0)) ? *(uint8_t*)delta_e32 : *(uint8_t*)(src + idx_src_buf[j + 1] + 2);
                if (!is_border_transparent || (idx_src_buf[j + 2] >= 0)) *(uint16_t*)(out + (j + 2) * nc) = (is_border_constant && (idx_src_buf[j + 2] < 0)) ? *(uint16_t*)delta_e32 : *(uint16_t*)(src + idx_src_buf[j + 2]);
                if (!is_border_transparent || (idx_src_buf[j + 2] >= 0)) *(uint8_t*)(out + (j + 2) * nc + 2) = (is_border_constant && (idx_src_buf[j + 2] < 0)) ? *(uint8_t*)delta_e32 : *(uint8_t*)(src + idx_src_buf[j + 2] + 2);
                if (!is_border_transparent || (idx_src_buf[j + 3] >= 0)) *(uint16_t*)(out + (j + 3) * nc) = (is_border_constant && (idx_src_buf[j + 3] < 0)) ? *(uint16_t*)delta_e32 : *(uint16_t*)(src + idx_src_buf[j + 3]);
                if (!is_border_transparent || (idx_src_buf[j + 3] >= 0)) *(uint8_t*)(out + (j + 3) * nc + 2) = (is_border_constant && (idx_src_buf[j + 3] < 0)) ? *(uint8_t*)delta_e32 : *(uint8_t*)(src + idx_src_buf[j + 3] + 2);
            }
            for (; j < num_idx; ++j) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) out[j * nc + 0] = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta : src[idx_src_buf[j] + 0];
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) out[j * nc + 1] = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta : src[idx_src_buf[j] + 1];
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) out[j * nc + 2] = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? delta : src[idx_src_buf[j] + 2];
            }
        } else { // nc == 4
            uint8_t delta_e32[] = {delta, delta, delta, delta};
            int32_t j = 0;
            for (; j <= num_idx - 8; j += 8) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) *(uint32_t*)(out + (j + 0) * nc) = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 0]);
                if (!is_border_transparent || (idx_src_buf[j + 1] >= 0)) *(uint32_t*)(out + (j + 1) * nc) = (is_border_constant && (idx_src_buf[j + 1] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 1]);
                if (!is_border_transparent || (idx_src_buf[j + 2] >= 0)) *(uint32_t*)(out + (j + 2) * nc) = (is_border_constant && (idx_src_buf[j + 2] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 2]);
                if (!is_border_transparent || (idx_src_buf[j + 3] >= 0)) *(uint32_t*)(out + (j + 3) * nc) = (is_border_constant && (idx_src_buf[j + 3] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 3]);
                if (!is_border_transparent || (idx_src_buf[j + 4] >= 0)) *(uint32_t*)(out + (j + 4) * nc) = (is_border_constant && (idx_src_buf[j + 4] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 4]);
                if (!is_border_transparent || (idx_src_buf[j + 5] >= 0)) *(uint32_t*)(out + (j + 5) * nc) = (is_border_constant && (idx_src_buf[j + 5] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 5]);
                if (!is_border_transparent || (idx_src_buf[j + 6] >= 0)) *(uint32_t*)(out + (j + 6) * nc) = (is_border_constant && (idx_src_buf[j + 6] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 6]);
                if (!is_border_transparent || (idx_src_buf[j + 7] >= 0)) *(uint32_t*)(out + (j + 7) * nc) = (is_border_constant && (idx_src_buf[j + 7] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 7]);
            }
            for (; j < num_idx; ++j) {
                if (!is_border_transparent || (idx_src_buf[j + 0] >= 0)) *(uint32_t*)(out + (j + 0) * nc) = (is_border_constant && (idx_src_buf[j + 0] < 0)) ? *(uint32_t*)delta_e32 : *(uint32_t*)(src + idx_src_buf[j + 0]);
            }
        }
    }
};

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_nearest(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* dst,
    const T* src,
    const double* M,
    T delta)
{
    const int32_t HEIGHT_BLK = 8;
    const int32_t WIDTH_BLK = 32 * 4;

    int32_t idx_src_buf[WIDTH_BLK];

    int32_t sx_buf[outWidth], sy_buf[outWidth];
    for (int32_t j = 0; j < outWidth; j++) {
        sx_buf[j] = saturate_cast(M[0] * j * 1024);
        sy_buf[j] = saturate_cast(M[3] * j * 1024);
    }
    int32_t base_x_buf[outHeight], base_y_buf[outHeight];
    for (int32_t i = 0; i < outHeight; i++) {
        base_x_buf[i] = saturate_cast((M[1] * i + M[2]) * 1024) + 512;
        base_y_buf[i] = saturate_cast((M[4] * i + M[5]) * 1024) + 512;
    }

    auto out = dst;
    for (int32_t h_beg = 0; h_beg < outHeight; h_beg += HEIGHT_BLK, out += HEIGHT_BLK * outWidthStride) {
        int32_t real_height_blk = std::min(HEIGHT_BLK, outHeight - h_beg);

        for (int32_t w_beg = 0; w_beg < outWidth; w_beg += WIDTH_BLK) {
            int32_t real_width_blk = std::min(WIDTH_BLK, outWidth - w_beg);

            auto out_ = out;
            for (int32_t i = h_beg; i < h_beg + real_height_blk; i++, out_ += outWidthStride) {
                int32_t base_x = base_x_buf[i];
                int32_t base_y = base_y_buf[i];

                WarpaffineNearestCalIdx<T, nc, borderMode>::f(sx_buf + w_beg, sy_buf + w_beg, base_x, base_y, inHeight, inWidth, inWidthStride, idx_src_buf, real_width_blk);
                WarpaffineNearestLoadWithIdx<T, nc, borderMode>::f(src, idx_src_buf, real_width_blk, out_ + w_beg * nc, delta);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
struct WarpaffineLinearCalIdx {
    static inline vbool8_t check_border(
        vint32m4_t sx_v,
        vint32m4_t sy_v,
        const int32_t x_min, // included
        const int32_t x_max, // excluded
        const int32_t y_min, // included
        const int32_t y_max, // excluded
        const size_t vl)
    {
        vbool8_t flag0_v = vmsgt_vx_i32m4_b8(sx_v, x_min - 1, vl);
        vbool8_t flag1_v = vmsle_vx_i32m4_b8(sx_v, x_max - 1, vl);
        vbool8_t flag2_v = vmsgt_vx_i32m4_b8(sy_v, y_min - 1, vl);
        vbool8_t flag3_v = vmsle_vx_i32m4_b8(sy_v, y_max - 1, vl);
        vbool8_t flag_v = vmand_mm_b8(vmand_mm_b8(vmand_mm_b8(flag0_v, flag1_v, vl), flag2_v, vl), flag3_v, vl);
        return flag_v;
    }

    static inline void f(
        const int32_t num_idx,
        const int32_t inHeight,
        const int32_t inWidth,
        const int32_t inWidthStride,
        const float base_x,
        const float base_y,
        const float* offset_x_buf,
        const float* offset_y_buf,
        int32_t** idx_bufs,
        float** tab_bufs,
        bool* mask_buf)
    {
        const size_t vl = vsetvlmax_e32m4();
        const int32_t num_cal_idx_unroll = vl;

        for (int32_t j = 0; j < num_idx; j += num_cal_idx_unroll) {
            const size_t vl = vsetvl_e32m4(num_idx - j);

            vfloat32m4_t x_v = vfadd_vf_f32m4(vle32_v_f32m4(offset_x_buf + j, vl), base_x, vl);
            vfloat32m4_t y_v = vfadd_vf_f32m4(vle32_v_f32m4(offset_y_buf + j, vl), base_y, vl);
            wr_fcsr_rm(2); // round to negive infinity
            vint32m4_t sx0_v = vfcvt_x_f_v_i32m4(x_v, vl);
            vint32m4_t sy0_v = vfcvt_x_f_v_i32m4(y_v, vl);
            vfloat32m4_t u_v = vfsub_vv_f32m4(x_v, vfcvt_f_x_v_f32m4(sx0_v, vl), vl);
            vfloat32m4_t v_v = vfsub_vv_f32m4(y_v, vfcvt_f_x_v_f32m4(sy0_v, vl), vl);
            {
                vfloat32m4_t taby0_v = vfrsub_vf_f32m4(v_v, 1.0f, vl);
                vfloat32m4_t tabx0_v = vfrsub_vf_f32m4(u_v, 1.0f, vl);

                vse32_v_f32m4(tab_bufs[0] + j, vfmul_vv_f32m4(taby0_v, tabx0_v, vl), vl);
                vse32_v_f32m4(tab_bufs[1] + j, vfmul_vv_f32m4(taby0_v, u_v, vl), vl);
                vse32_v_f32m4(tab_bufs[2] + j, vfmul_vv_f32m4(v_v, tabx0_v, vl), vl);
                vse32_v_f32m4(tab_bufs[3] + j, vfmul_vv_f32m4(v_v, u_v, vl), vl);
            }
            {
                if (borderMode == ppl::cv::BORDER_CONSTANT || borderMode == ppl::cv::BORDER_TRANSPARENT) {
                    vbool8_t flag0_v = check_border(sx0_v, sy0_v, 0, inWidth, 0, inHeight, vl);
                    vbool8_t flag1_v = check_border(sx0_v, sy0_v, 0 - 1, inWidth - 1, 0, inHeight, vl);
                    vbool8_t flag2_v = check_border(sx0_v, sy0_v, 0, inWidth, 0 - 1, inHeight - 1, vl);
                    vbool8_t flag3_v = check_border(sx0_v, sy0_v, 0 - 1, inWidth - 1, 0 - 1, inHeight - 1, vl);

                    vint32m4_t idx0_v = vmacc_vx_i32m4(vmul_vx_i32m4(sy0_v, inWidthStride, vl), nc, sx0_v, vl);
                    vint32m4_t idx1_v = vadd_vx_i32m4(idx0_v, nc, vl);
                    vint32m4_t idx2_v = vadd_vx_i32m4(idx0_v, inWidthStride, vl);
                    vint32m4_t idx3_v = vadd_vx_i32m4(idx2_v, nc, vl);
                    vse32_v_i32m4(idx_bufs[0] + j, vmerge_vxm_i32m4(vmnot_m_b8(flag0_v, vl), idx0_v, -1, vl), vl);
                    vse32_v_i32m4(idx_bufs[1] + j, vmerge_vxm_i32m4(vmnot_m_b8(flag1_v, vl), idx1_v, -1, vl), vl);
                    vse32_v_i32m4(idx_bufs[2] + j, vmerge_vxm_i32m4(vmnot_m_b8(flag2_v, vl), idx2_v, -1, vl), vl);
                    vse32_v_i32m4(idx_bufs[3] + j, vmerge_vxm_i32m4(vmnot_m_b8(flag3_v, vl), idx3_v, -1, vl), vl);

                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vbool8_t mask = vmand_mm_b8(vmand_mm_b8(vmand_mm_b8(flag0_v, flag1_v, vl), flag2_v, vl), flag3_v, vl);
                        ;
                        vse8_v_u8m1((uint8_t*)(mask_buf + j), vmerge_vxm_u8m1(mask, vmv_v_x_u8m1(0, vl), 1, vl), vl);
                    }
                } else if (borderMode == ppl::cv::BORDER_REPLICATE) {
                    vint32m4_t sx1_v = vadd_vx_i32m4(sx0_v, 1, vl);
                    vint32m4_t sy1_v = vadd_vx_i32m4(sy0_v, 1, vl);

                    sx0_v = vmax_vx_i32m4(vmin_vx_i32m4(sx0_v, inWidth - 1, vl), 0, vl);
                    sx1_v = vmax_vx_i32m4(vmin_vx_i32m4(sx1_v, inWidth - 1, vl), 0, vl);
                    sy0_v = vmax_vx_i32m4(vmin_vx_i32m4(sy0_v, inHeight - 1, vl), 0, vl);
                    sy1_v = vmax_vx_i32m4(vmin_vx_i32m4(sy1_v, inHeight - 1, vl), 0, vl);

                    vint32m4_t idx0_v = vmacc_vx_i32m4(vmul_vx_i32m4(sy0_v, inWidthStride, vl), nc, sx0_v, vl);
                    vint32m4_t idx1_v = vmacc_vx_i32m4(vmul_vx_i32m4(sy0_v, inWidthStride, vl), nc, sx1_v, vl);
                    vint32m4_t idx2_v = vmacc_vx_i32m4(vmul_vx_i32m4(sy1_v, inWidthStride, vl), nc, sx0_v, vl);
                    vint32m4_t idx3_v = vmacc_vx_i32m4(vmul_vx_i32m4(sy1_v, inWidthStride, vl), nc, sx1_v, vl);

                    vse32_v_i32m4(idx_bufs[0] + j, idx0_v, vl);
                    vse32_v_i32m4(idx_bufs[1] + j, idx1_v, vl);
                    vse32_v_i32m4(idx_bufs[2] + j, idx2_v, vl);
                    vse32_v_i32m4(idx_bufs[3] + j, idx3_v, vl);
                }
            }
        }
    }
};

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
struct WarpaffineLinearCalLinear {
    static inline void f(
        int32_t length,
        T** a,
        float** b,
        const bool* mask,
        T* out)
    {
        for (int32_t i = 0; i < length; ++i) {
            if ((borderMode == ppl::cv::BORDER_TRANSPARENT) && (!mask[i])) {
                continue;
            }
            for (int32_t j = 0; j < nc; ++j) {
                float sum = a[0][i * nc + j] * b[0][i] + a[1][i * nc + j] * b[1][i] + a[2][i * nc + j] * b[2][i] + a[3][i * nc + j] * b[3][i];
                out[i * nc + j] = static_cast<T>(sum);
            }
        }
    }
};

template <int32_t nc, ppl::cv::BorderType borderMode>
struct WarpaffineLinearCalLinear<float, nc, borderMode> {
    typedef float T;
    static inline void f(
        int32_t length,
        T** a,
        float** b,
        const bool* mask,
        T* out)
    {
        if (nc == 1) {
            const size_t vl = vsetvlmax_e32m4();
            const int32_t num_unroll = vl;

            for (int32_t i = 0; i < length; i += num_unroll) {
                const size_t vl = vsetvl_e32m4(length - i);

                vfloat32m4_t a0_v = vle32_v_f32m4(a[0] + i, vl);
                vfloat32m4_t a1_v = vle32_v_f32m4(a[1] + i, vl);
                vfloat32m4_t b0_v = vle32_v_f32m4(b[0] + i, vl);
                vfloat32m4_t b1_v = vle32_v_f32m4(b[1] + i, vl);
                vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                vfloat32m4_t a2_v = vle32_v_f32m4(a[2] + i, vl);
                vfloat32m4_t a3_v = vle32_v_f32m4(a[3] + i, vl);
                vfloat32m4_t b2_v = vle32_v_f32m4(b[2] + i, vl);
                vfloat32m4_t b3_v = vle32_v_f32m4(b[3] + i, vl);
                vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                vfloat32m4_t sum_v = vfadd_vv_f32m4(sum0_v, sum1_v, vl);
                if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                    vbool8_t mask_v = vmsne_vx_u8m1_b8(vle8_v_u8m1((uint8_t*)(mask + i), vl), 0, vl);
                    vse32_v_f32m4_m(mask_v, out + i, sum_v, vl);
                } else {
                    vse32_v_f32m4(out + i, sum_v, vl);
                }
            }
        } else { // nc == 3 or nc == 4
            const size_t vl = vsetvlmax_e32m4();
            const int32_t num_unroll = vl;

            for (int32_t i = 0; i < length; i += num_unroll) {
                const size_t vl = vsetvl_e32m4(length - i);
                vbool8_t mask_v;
                if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                    mask_v = vmsne_vx_u8m1_b8(vle8_v_u8m1((uint8_t*)(mask + i), vl), 0, vl);
                }

                vfloat32m4_t b0_v = vle32_v_f32m4(b[0] + i, vl);
                vfloat32m4_t b1_v = vle32_v_f32m4(b[1] + i, vl);
                vfloat32m4_t b2_v = vle32_v_f32m4(b[2] + i, vl);
                vfloat32m4_t b3_v = vle32_v_f32m4(b[3] + i, vl);

                // c0
                {
                    vfloat32m4_t a0_v = vlse32_v_f32m4(a[0] + i * nc + 0, nc * sizeof(T), vl);
                    vfloat32m4_t a1_v = vlse32_v_f32m4(a[1] + i * nc + 0, nc * sizeof(T), vl);
                    vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                    vfloat32m4_t a2_v = vlse32_v_f32m4(a[2] + i * nc + 0, nc * sizeof(T), vl);
                    vfloat32m4_t a3_v = vlse32_v_f32m4(a[3] + i * nc + 0, nc * sizeof(T), vl);
                    vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                    vfloat32m4_t sum_v = vfadd_vv_f32m4(sum0_v, sum1_v, vl);
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vsse32_v_f32m4_m(mask_v, out + i * nc + 0, nc * sizeof(T), sum_v, vl);
                    } else {
                        vsse32_v_f32m4(out + i * nc + 0, nc * sizeof(T), sum_v, vl);
                    }
                }
                // c1
                {
                    vfloat32m4_t a0_v = vlse32_v_f32m4(a[0] + i * nc + 1, nc * sizeof(T), vl);
                    vfloat32m4_t a1_v = vlse32_v_f32m4(a[1] + i * nc + 1, nc * sizeof(T), vl);
                    vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                    vfloat32m4_t a2_v = vlse32_v_f32m4(a[2] + i * nc + 1, nc * sizeof(T), vl);
                    vfloat32m4_t a3_v = vlse32_v_f32m4(a[3] + i * nc + 1, nc * sizeof(T), vl);
                    vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                    vfloat32m4_t sum_v = vfadd_vv_f32m4(sum0_v, sum1_v, vl);
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vsse32_v_f32m4_m(mask_v, out + i * nc + 1, nc * sizeof(T), sum_v, vl);
                    } else {
                        vsse32_v_f32m4(out + i * nc + 1, nc * sizeof(T), sum_v, vl);
                    }
                }
                // c2
                {
                    vfloat32m4_t a0_v = vlse32_v_f32m4(a[0] + i * nc + 2, nc * sizeof(T), vl);
                    vfloat32m4_t a1_v = vlse32_v_f32m4(a[1] + i * nc + 2, nc * sizeof(T), vl);
                    vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                    vfloat32m4_t a2_v = vlse32_v_f32m4(a[2] + i * nc + 2, nc * sizeof(T), vl);
                    vfloat32m4_t a3_v = vlse32_v_f32m4(a[3] + i * nc + 2, nc * sizeof(T), vl);
                    vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                    vfloat32m4_t sum_v = vfadd_vv_f32m4(sum0_v, sum1_v, vl);
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vsse32_v_f32m4_m(mask_v, out + i * nc + 2, nc * sizeof(T), sum_v, vl);
                    } else {
                        vsse32_v_f32m4(out + i * nc + 2, nc * sizeof(T), sum_v, vl);
                    }
                }
                // c3
                if (nc == 4) {
                    vfloat32m4_t a0_v = vlse32_v_f32m4(a[0] + i * nc + 3, nc * sizeof(T), vl);
                    vfloat32m4_t a1_v = vlse32_v_f32m4(a[1] + i * nc + 3, nc * sizeof(T), vl);
                    vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                    vfloat32m4_t a2_v = vlse32_v_f32m4(a[2] + i * nc + 3, nc * sizeof(T), vl);
                    vfloat32m4_t a3_v = vlse32_v_f32m4(a[3] + i * nc + 3, nc * sizeof(T), vl);
                    vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                    vfloat32m4_t sum_v = vfadd_vv_f32m4(sum0_v, sum1_v, vl);
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vsse32_v_f32m4_m(mask_v, out + i * nc + 3, nc * sizeof(T), sum_v, vl);
                    } else {
                        vsse32_v_f32m4(out + i * nc + 3, nc * sizeof(T), sum_v, vl);
                    }
                }
            }
        }
    }
};

template <int32_t nc, ppl::cv::BorderType borderMode>
struct WarpaffineLinearCalLinear<uint8_t, nc, borderMode> {
    typedef uint8_t T;
    static inline vfloat32m4_t load_fp32_from_u8_ptr_with_stride(
        const uint8_t* src,
        const int32_t stride,
        const size_t vl)
    {
        return vfcvt_f_xu_v_f32m4(vlsbu_v_u32m4((uint32_t*)src, stride, vl), vl);
    }
    static inline vuint8m1_t cvt_f32_u8(
        vfloat32m4_t data_v,
        const size_t vl)
    {
        return vnclipu_wx_u8m1(vfncvt_xu_f_w_u16m2(data_v, vl), 0, vl);
    }
    static inline void f(
        int32_t length,
        T** a,
        float** b,
        const bool* mask,
        T* out)
    {
        if (nc == 1) {
            const size_t vl = vsetvlmax_e32m4();
            const int32_t num_unroll = vl;

            for (int32_t i = 0; i < length; i += num_unroll) {
                const size_t vl = vsetvl_e32m4(length - i);

                vfloat32m4_t a0_v = load_fp32_from_u8_ptr_with_stride(a[0] + i, 1 * sizeof(uint8_t), vl);
                vfloat32m4_t a1_v = load_fp32_from_u8_ptr_with_stride(a[1] + i, 1 * sizeof(uint8_t), vl);
                vfloat32m4_t b0_v = vle32_v_f32m4(b[0] + i, vl);
                vfloat32m4_t b1_v = vle32_v_f32m4(b[1] + i, vl);
                vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                vfloat32m4_t a2_v = load_fp32_from_u8_ptr_with_stride(a[2] + i, 1 * sizeof(uint8_t), vl);
                vfloat32m4_t a3_v = load_fp32_from_u8_ptr_with_stride(a[3] + i, 1 * sizeof(uint8_t), vl);
                vfloat32m4_t b2_v = vle32_v_f32m4(b[2] + i, vl);
                vfloat32m4_t b3_v = vle32_v_f32m4(b[3] + i, vl);
                vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                vuint8m1_t sum_v = cvt_f32_u8(vfadd_vv_f32m4(sum0_v, sum1_v, vl), vl);
                if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                    vbool8_t mask_v = vmsne_vx_u8m1_b8(vle8_v_u8m1((uint8_t*)(mask + i), vl), 0, vl);
                    vse8_v_u8m1_m(mask_v, out + i, sum_v, vl);
                } else {
                    vse8_v_u8m1(out + i, sum_v, vl);
                }
            }
        } else { // nc == 3 or nc == 4
            const size_t vl = vsetvlmax_e32m4();
            const int32_t num_unroll = vl;

            for (int32_t i = 0; i < length; i += num_unroll) {
                const size_t vl = vsetvl_e32m4(length - i);
                vbool8_t mask_v;
                if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                    mask_v = vmsne_vx_u8m1_b8(vle8_v_u8m1((uint8_t*)(mask + i), vl), 0, vl);
                }

                vfloat32m4_t b0_v = vle32_v_f32m4(b[0] + i, vl);
                vfloat32m4_t b1_v = vle32_v_f32m4(b[1] + i, vl);
                vfloat32m4_t b2_v = vle32_v_f32m4(b[2] + i, vl);
                vfloat32m4_t b3_v = vle32_v_f32m4(b[3] + i, vl);

                // c0
                {
                    vfloat32m4_t a0_v = load_fp32_from_u8_ptr_with_stride(a[0] + i * nc + 0, nc * sizeof(T), vl);
                    vfloat32m4_t a1_v = load_fp32_from_u8_ptr_with_stride(a[1] + i * nc + 0, nc * sizeof(T), vl);
                    vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                    vfloat32m4_t a2_v = load_fp32_from_u8_ptr_with_stride(a[2] + i * nc + 0, nc * sizeof(T), vl);
                    vfloat32m4_t a3_v = load_fp32_from_u8_ptr_with_stride(a[3] + i * nc + 0, nc * sizeof(T), vl);
                    vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                    vuint8m1_t sum_v = cvt_f32_u8(vfadd_vv_f32m4(sum0_v, sum1_v, vl), vl);
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vsse8_v_u8m1_m(mask_v, out + i * nc + 0, nc * sizeof(T), sum_v, vl);
                    } else {
                        vsse8_v_u8m1(out + i * nc + 0, nc * sizeof(T), sum_v, vl);
                    }
                }
                // c1
                {
                    vfloat32m4_t a0_v = load_fp32_from_u8_ptr_with_stride(a[0] + i * nc + 1, nc * sizeof(T), vl);
                    vfloat32m4_t a1_v = load_fp32_from_u8_ptr_with_stride(a[1] + i * nc + 1, nc * sizeof(T), vl);
                    vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                    vfloat32m4_t a2_v = load_fp32_from_u8_ptr_with_stride(a[2] + i * nc + 1, nc * sizeof(T), vl);
                    vfloat32m4_t a3_v = load_fp32_from_u8_ptr_with_stride(a[3] + i * nc + 1, nc * sizeof(T), vl);
                    vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                    vuint8m1_t sum_v = cvt_f32_u8(vfadd_vv_f32m4(sum0_v, sum1_v, vl), vl);
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vsse8_v_u8m1_m(mask_v, out + i * nc + 1, nc * sizeof(T), sum_v, vl);
                    } else {
                        vsse8_v_u8m1(out + i * nc + 1, nc * sizeof(T), sum_v, vl);
                    }
                }
                // c2
                {
                    vfloat32m4_t a0_v = load_fp32_from_u8_ptr_with_stride(a[0] + i * nc + 2, nc * sizeof(T), vl);
                    vfloat32m4_t a1_v = load_fp32_from_u8_ptr_with_stride(a[1] + i * nc + 2, nc * sizeof(T), vl);
                    vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                    vfloat32m4_t a2_v = load_fp32_from_u8_ptr_with_stride(a[2] + i * nc + 2, nc * sizeof(T), vl);
                    vfloat32m4_t a3_v = load_fp32_from_u8_ptr_with_stride(a[3] + i * nc + 2, nc * sizeof(T), vl);
                    vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                    vuint8m1_t sum_v = cvt_f32_u8(vfadd_vv_f32m4(sum0_v, sum1_v, vl), vl);
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vsse8_v_u8m1_m(mask_v, out + i * nc + 2, nc * sizeof(T), sum_v, vl);
                    } else {
                        vsse8_v_u8m1(out + i * nc + 2, nc * sizeof(T), sum_v, vl);
                    }
                }
                // c3
                if (nc == 4) {
                    vfloat32m4_t a0_v = load_fp32_from_u8_ptr_with_stride(a[0] + i * nc + 3, nc * sizeof(T), vl);
                    vfloat32m4_t a1_v = load_fp32_from_u8_ptr_with_stride(a[1] + i * nc + 3, nc * sizeof(T), vl);
                    vfloat32m4_t sum0_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a0_v, b0_v, vl), a1_v, b1_v, vl);

                    vfloat32m4_t a2_v = load_fp32_from_u8_ptr_with_stride(a[2] + i * nc + 3, nc * sizeof(T), vl);
                    vfloat32m4_t a3_v = load_fp32_from_u8_ptr_with_stride(a[3] + i * nc + 3, nc * sizeof(T), vl);
                    vfloat32m4_t sum1_v = vfmacc_vv_f32m4(vfmul_vv_f32m4(a2_v, b2_v, vl), a3_v, b3_v, vl);

                    vuint8m1_t sum_v = cvt_f32_u8(vfadd_vv_f32m4(sum0_v, sum1_v, vl), vl);
                    if (borderMode == ppl::cv::BORDER_TRANSPARENT) {
                        vsse8_v_u8m1_m(mask_v, out + i * nc + 3, nc * sizeof(T), sum_v, vl);
                    } else {
                        vsse8_v_u8m1(out + i * nc + 3, nc * sizeof(T), sum_v, vl);
                    }
                }
            }
        }
    }
};

template <typename T, int32_t nc, ppl::cv::BorderType borderMode>
::ppl::common::RetCode warpaffine_linear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* dst,
    const T* src,
    const double* M,
    T delta)
{
    constexpr int32_t WIDTH_BLK = 128;
    constexpr int32_t NUM_TAB = 4;
    float tab_buf[WIDTH_BLK * NUM_TAB];
    int32_t idx_buf[WIDTH_BLK * NUM_TAB];
    T data_buf[WIDTH_BLK * nc * NUM_TAB];
    bool mask_buf[WIDTH_BLK];

    int32_t* idx_bufs[NUM_TAB];
    float* tab_bufs[NUM_TAB];
    T* data_bufs[NUM_TAB];
    {
        for (int32_t i = 0; i < NUM_TAB; ++i) {
            tab_bufs[i] = (float*)(tab_buf + WIDTH_BLK * i);
            idx_bufs[i] = (int32_t*)(idx_buf + WIDTH_BLK * i);
            data_bufs[i] = (T*)(data_buf + WIDTH_BLK * nc * i);
        }
    }

    float base_x_buf[outHeight], base_y_buf[outHeight];
    float offset_x_buf[outWidth], offset_y_buf[outWidth];
    {
        for (int32_t i = 0; i < outHeight; i++) {
            base_x_buf[i] = M[1] * i + M[2];
            base_y_buf[i] = M[4] * i + M[5];
        }
        for (int32_t j = 0; j < outWidth; j++) {
            offset_x_buf[j] = M[0] * j;
            offset_y_buf[j] = M[3] * j;
        }
    }

    int32_t rm = rd_fcsr_rm();

    auto out = dst;
    for (int32_t i = 0; i < outHeight; i++, out += outWidthStride) {
        float base_x = base_x_buf[i];
        float base_y = base_y_buf[i];

        for (int32_t w_beg = 0; w_beg < outWidth; w_beg += WIDTH_BLK) {
            int32_t real_width_blk = std::min(outWidth - w_beg, WIDTH_BLK);
            WarpaffineLinearCalIdx<T, nc, borderMode>::f(real_width_blk, inHeight, inWidth, inWidthStride, base_x, base_y, offset_x_buf + w_beg, offset_y_buf + w_beg, idx_bufs, tab_bufs, mask_buf);
            // TODO: merge 'LoadWithIdx' and 'CalLinear'
            {
                WarpaffineNearestLoadWithIdx<T, nc, borderMode>::f(src, idx_bufs[0], real_width_blk, data_bufs[0], delta);
                WarpaffineNearestLoadWithIdx<T, nc, borderMode>::f(src, idx_bufs[1], real_width_blk, data_bufs[1], delta);
                WarpaffineNearestLoadWithIdx<T, nc, borderMode>::f(src, idx_bufs[2], real_width_blk, data_bufs[2], delta);
                WarpaffineNearestLoadWithIdx<T, nc, borderMode>::f(src, idx_bufs[3], real_width_blk, data_bufs[3], delta);
                WarpaffineLinearCalLinear<T, nc, borderMode>::f(real_width_blk, data_bufs, tab_bufs, mask_buf, out + w_beg * nc);
            }
        }
    }

    wr_fcsr_rm(rm);
    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t nc>
::ppl::common::RetCode WarpAffineNearestPoint(
    int32_t inHeight,
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
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        return warpaffine_nearest<T, nc, ppl::cv::BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        return warpaffine_nearest<T, nc, ppl::cv::BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
    } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
        return warpaffine_nearest<T, nc, ppl::cv::BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode WarpAffineNearestPoint<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpAffineNearestPoint<float, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpAffineNearestPoint<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpAffineNearestPoint<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpAffineNearestPoint<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpAffineNearestPoint<uint8_t, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpAffineNearestPoint<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpAffineNearestPoint<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template <typename T, int32_t nc>
::ppl::common::RetCode WarpAffineLinear(
    int32_t inHeight,
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
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        return warpaffine_linear<T, nc, ppl::cv::BORDER_CONSTANT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        return warpaffine_linear<T, nc, ppl::cv::BORDER_REPLICATE>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
    } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
        return warpaffine_linear<T, nc, ppl::cv::BORDER_TRANSPARENT>(inHeight, inWidth, inWidthStride, outHeight, outWidth, outWidthStride, outData, inData, affineMatrix, border_value);
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode WarpAffineLinear<float, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpAffineLinear<float, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpAffineLinear<float, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpAffineLinear<float, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const float* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    float* outData,
    const double* affineMatrix,
    BorderType border_type,
    float border_value);

template ::ppl::common::RetCode WarpAffineLinear<uint8_t, 1>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpAffineLinear<uint8_t, 2>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpAffineLinear<uint8_t, 3>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

template ::ppl::common::RetCode WarpAffineLinear<uint8_t, 4>(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    uint8_t* outData,
    const double* affineMatrix,
    BorderType border_type,
    uint8_t border_value);

}
}
} // namespace ppl::cv::riscv
