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

#ifndef __ST_HPC_PPL_CV_RISCV_MORPH_H_
#define __ST_HPC_PPL_CV_RISCV_MORPH_H_

#include <algorithm>
#include "ppl/cv/types.h"
#include "ppl/cv/riscv/typetraits.h"
#include <cstring>

namespace ppl {
namespace cv {
namespace riscv {

enum MorphType {
    MORPH_DILATE = 0,
    MORPH_ERODE = 1
};

template <typename T, MorphType morph>
inline T morph_scalar_op(T a, T b);

template <>
inline float morph_scalar_op<float, MORPH_DILATE>(float a, float b)
{
    return a > b ? a : b;
}

template <>
inline float morph_scalar_op<float, MORPH_ERODE>(float a, float b)
{
    return a > b ? b : a;
}

template <>
inline uint8_t morph_scalar_op<uint8_t, MORPH_DILATE>(uint8_t a, uint8_t b)
{
    return a > b ? a : b;
}

template <>
inline uint8_t morph_scalar_op<uint8_t, MORPH_ERODE>(uint8_t a, uint8_t b)
{
    return a > b ? b : a;
}

template <typename T, int32_t lmul, MorphType morph>
inline vdataxmy_t<T, lmul> morph_vec_op(vdataxmy_t<T, lmul> op1, vdataxmy_t<T, lmul> op2, size_t vl);

template <>
inline vdataxmy_t<float, 4> morph_vec_op<float, 4, MORPH_DILATE>(vdataxmy_t<float, 4> op1, vdataxmy_t<float, 4> op2, size_t vl)
{
    return vmax_vv_exmy<float, 4>(op1, op2, vl);
}

template <>
inline vdataxmy_t<float, 4> morph_vec_op<float, 4, MORPH_ERODE>(vdataxmy_t<float, 4> op1, vdataxmy_t<float, 4> op2, size_t vl)
{
    return vmin_vv_exmy<float, 4>(op1, op2, vl);
}

template <>
inline vdataxmy_t<uint8_t, 4> morph_vec_op<uint8_t, 4, MORPH_DILATE>(vdataxmy_t<uint8_t, 4> op1, vdataxmy_t<uint8_t, 4> op2, size_t vl)
{
    return vmax_vv_exmy<uint8_t, 4>(op1, op2, vl);
}

template <>
inline vdataxmy_t<uint8_t, 4> morph_vec_op<uint8_t, 4, MORPH_ERODE>(vdataxmy_t<uint8_t, 4> op1, vdataxmy_t<uint8_t, 4> op2, size_t vl)
{
    return vmin_vv_exmy<uint8_t, 4>(op1, op2, vl);
}

template <BorderType border_mode, bool first_side>
static inline int32_t border_interpolate(int32_t p, int32_t len)
{
    if (border_mode == ppl::cv::BORDER_REFLECT_101) {
        p = first_side ? (-p) : 2 * len - p - 2;
    } else if (border_mode == ppl::cv::BORDER_REFLECT) {
        p = first_side ? (-p - 1) : 2 * len - p - 1;
    } else if (border_mode == ppl::cv::BORDER_REPLICATE) {
        p = first_side ? 0 : len - 1;
    } else if (border_mode == ppl::cv::BORDER_CONSTANT) {
        p = 0;
    }
    return p;
}

template <ppl::cv::BorderType border_mode>
struct FilterWin {
    void init(
        const int32_t num_valid_elem,
        const int32_t in_width_stride,
        const int32_t kernelx_len,
        const int32_t kernely_len,
        const int32_t *kernel_bases,
        const int32_t *kernel_offsets,
        const uint8_t *kernel,
        const int32_t padding_top,
        const int32_t padding_bottom)
    {
        num_elem = num_valid_elem;
        buffer = (int32_t *)malloc(num_elem * 6 * sizeof(int32_t));

        bases = buffer + num_elem * 0;
        offsets = buffer + num_elem * 1;
        left_beg_lst = buffer + num_elem * 2;
        left_end_lst = buffer + num_elem * 3;
        right_beg_lst = buffer + num_elem * 4;
        right_end_lst = buffer + num_elem * 5;

        // top
        int32_t i = 0, wi = 0;
        auto ker = kernel;
        for (; i < padding_top; ++i, ker += kernelx_len) {
            int32_t pad_idx = border_interpolate<border_mode, true>(-padding_top + i, kernely_len);
            for (int32_t j = 0; j < kernelx_len; ++j) {
                if (ker[j]) {
                    if (border_mode != ppl::cv::BORDER_CONSTANT) {
                        bases[wi] = kernel_bases[pad_idx * kernelx_len + j];
                        offsets[wi] = kernel_offsets[pad_idx * kernelx_len + j];
                        ++wi;
                    }
                    padding = true;
                }
            }
        }
        for (; i < kernely_len - padding_bottom; ++i, ker += kernelx_len) {
            for (int32_t j = 0; j < kernelx_len; ++j) {
                if (ker[j]) {
                    bases[wi] = kernel_bases[(i - padding_top) * kernelx_len + j];
                    offsets[wi] = kernel_offsets[(i - padding_top) * kernelx_len + j];
                    ++wi;
                }
            }
        }
        // bottom
        for (; i < kernely_len; ++i, ker += kernelx_len) {
            int32_t last_line_idx = kernely_len - padding_bottom - 1;
            int32_t pad_idx = border_interpolate<border_mode, false>(i + padding_bottom, kernely_len);
            for (int32_t j = 0; j < kernelx_len; ++j) {
                if (ker[j]) {
                    if (border_mode != ppl::cv::BORDER_CONSTANT) {
                        bases[wi] = kernel_bases[last_line_idx * kernelx_len + j] + (pad_idx - kernely_len + 1) * in_width_stride;
                        offsets[wi] = kernel_offsets[last_line_idx * kernelx_len + j] + (pad_idx - kernely_len + 1) * in_width_stride;
                        ++wi;
                    }
                    padding = true;
                }
            }
        }
        num_elem = wi;
    }
    void cal_row_seg(
        const int32_t width,
        const int32_t channels,
        const int32_t kernelx_len)
    {
        num_left_elem = kernelx_len >> 1;
        num_right_elem = kernelx_len >> 1;
        mid_len = std::max(0, width - num_left_elem - num_right_elem) * channels;

        for (int32_t i = 0; i < num_elem; ++i) {
            int32_t padding_left = std::max(0, (bases[i] - offsets[i]) / channels);
            int32_t padding_right = std::max(0, (offsets[i] - bases[i]) / channels);
            int32_t valid_seg_len = width - padding_right - padding_left;
            int32_t left_len = std::min(num_left_elem - padding_left, valid_seg_len);
            int32_t right_len = std::min(num_right_elem - padding_right, valid_seg_len - left_len);
            left_beg_lst[i] = padding_left * channels;
            left_end_lst[i] = left_beg_lst[i] + left_len * channels;
            right_beg_lst[i] = left_end_lst[i] + mid_len;
            right_end_lst[i] = right_beg_lst[i] + right_len * channels;
        }
    }

    ~FilterWin()
    {
        free(buffer);
    }

    int32_t num_elem = 0;
    int32_t *bases = nullptr;
    int32_t *offsets = nullptr;
    bool padding = false;

    int32_t num_left_elem;
    int32_t num_right_elem;
    int32_t *left_beg_lst;
    int32_t *left_end_lst;
    int32_t *right_beg_lst;
    int32_t *right_end_lst;
    int32_t mid_len;

private:
    int32_t *buffer;
};

template <ppl::cv::BorderType border_mode>
inline void init_morph_win(
    const int32_t height,
    const int32_t width,
    const int32_t inWidthStride,
    const int32_t channels,
    const uint8_t *kernel,
    const int32_t num_valid_elem,
    const int32_t kernelx_len,
    const int32_t kernely_len,
    FilterWin<border_mode> *filter_wins)
{
    int32_t kernel_bases[kernelx_len * kernely_len];
    int32_t kernel_offsets[kernelx_len * kernely_len];

    int32_t num_top_pad = kernely_len >> 1;
    int32_t num_bottom_pad = kernely_len - 1 - num_top_pad;
    int32_t num_left_pad = num_top_pad;

    int32_t index = 0;
    for (int32_t ky = 0; ky < kernely_len; ++ky) {
        for (int32_t kx = 0; kx < kernelx_len; ++kx, ++index) {
            kernel_bases[index] = ky * inWidthStride;
            kernel_offsets[index] = kernel_bases[index] + (kx - num_left_pad) * channels;
        }
    }

    // top
    for (int32_t i = 0; i < num_top_pad; ++i, ++filter_wins) {
        int32_t num_top_row = num_top_pad - i;
        int32_t num_bottom_row = std::max(kernely_len - num_top_row - height, 0);
        (*filter_wins).init(num_valid_elem, inWidthStride, kernelx_len, kernely_len, kernel_bases, kernel_offsets, kernel, num_top_row, num_bottom_row);
        (*filter_wins).cal_row_seg(width, channels, kernelx_len);
    }
    {
        (*filter_wins).init(num_valid_elem, inWidthStride, kernelx_len, kernely_len, kernel_bases, kernel_offsets, kernel, 0, 0);
        (*filter_wins).cal_row_seg(width, channels, kernelx_len);
        ++filter_wins;
    }
    // bottom
    for (int32_t i = 0; i < num_bottom_pad; ++i, ++filter_wins) {
        (*filter_wins).init(num_valid_elem, inWidthStride, kernelx_len, kernely_len, kernel_bases, kernel_offsets, kernel, 0, i + 1);
        (*filter_wins).cal_row_seg(width, channels, kernelx_len);
    }
}

template <typename T, MorphType morph_type, ppl::cv::BorderType border_mode, int32_t channels, bool with_pad, bool is_first>
struct MorphRowFunc {
    inline static void f(
        const T *src,
        const int32_t width,
        const FilterWin<border_mode> &win,
        const T border_value,
        T *dst)
    {
        constexpr int32_t RVV_LMUL = 4;
        const size_t vl = vsetvlmax_exmy<T, RVV_LMUL>();
        const int32_t num_unroll = vl;

        // top & bottom
        if (border_mode == ppl::cv::BORDER_CONSTANT && win.padding) {
            auto pad_value_v = vmv_v_x_exmy<T, RVV_LMUL>(border_value, vl);
            if (is_first) {
                if (win.num_elem == 0) {
                    for (int32_t i = 0; i < width * channels; i += num_unroll) {
                        const size_t vl = vsetvl_exmy<T, RVV_LMUL>(width * channels - i);
                        vsex_v_my<T, RVV_LMUL>(dst + i, pad_value_v, vl);
                    }
                }
            } else {
                if (win.num_elem > 0) {
                    for (int32_t i = 0; i < width * channels; i += num_unroll) {
                        const size_t vl = vsetvl_exmy<T, RVV_LMUL>(width * channels - i);
                        auto out_v = vlex_v_my<T, RVV_LMUL>(dst + i, vl);
                        out_v = morph_vec_op<T, RVV_LMUL, morph_type>(out_v, pad_value_v, vl);
                        vsex_v_my<T, RVV_LMUL>(dst + i, out_v, vl);
                    }
                }
            }
        }

        int32_t win_idx_beg = is_first ? 0 : 1;
        int32_t win_idx_end = is_first ? std::min(1, win.num_elem) : win.num_elem;

        // left
        for (int32_t i = win_idx_beg; i < win_idx_end; ++i) {
            if (win.offsets[i] < win.bases[i]) {
                int32_t in_x = win.offsets[i], out_x = 0;
                for (; in_x < win.bases[i]; in_x += channels, out_x += channels) {
                    int32_t real_in_x = win.bases[i] + border_interpolate<border_mode, true>((in_x - win.bases[i]) / channels, width) * channels;

                    T in0, in1, in2, in3;
                    if (channels > 0) in0 = (border_mode == ppl::cv::BORDER_CONSTANT) ? border_value : src[real_in_x + 0];
                    if (channels > 1) in1 = (border_mode == ppl::cv::BORDER_CONSTANT) ? border_value : src[real_in_x + 1];
                    if (channels > 2) in2 = (border_mode == ppl::cv::BORDER_CONSTANT) ? border_value : src[real_in_x + 2];
                    if (channels > 3) in3 = (border_mode == ppl::cv::BORDER_CONSTANT) ? border_value : src[real_in_x + 3];
                    if (channels > 0) dst[out_x + 0] = is_first ? in0 : morph_scalar_op<T, morph_type>(dst[out_x + 0], in0);
                    if (channels > 1) dst[out_x + 1] = is_first ? in1 : morph_scalar_op<T, morph_type>(dst[out_x + 1], in1);
                    if (channels > 2) dst[out_x + 2] = is_first ? in2 : morph_scalar_op<T, morph_type>(dst[out_x + 2], in2);
                    if (channels > 3) dst[out_x + 3] = is_first ? in3 : morph_scalar_op<T, morph_type>(dst[out_x + 3], in3);
                }
            }
        }

        for (int32_t i = win_idx_beg; i < win_idx_end; ++i) {
            int32_t out_x = win.left_beg_lst[i];
            int32_t in_x = win.offsets[i] + out_x;
            auto in_x_end = win.offsets[i] + win.left_end_lst[i];
            for (; in_x < in_x_end; in_x += num_unroll, out_x += num_unroll) {
                const size_t vl = vsetvl_exmy<T, RVV_LMUL>(in_x_end - in_x);
                auto out_v = vlex_v_my<T, RVV_LMUL>(src + in_x, vl);
                if (!is_first) {
                    out_v = morph_vec_op<T, RVV_LMUL, morph_type>(vlex_v_my<T, RVV_LMUL>(dst + out_x, vl), out_v, vl);
                }
                vsex_v_my<T, RVV_LMUL>(dst + out_x, out_v, vl);
            }
        }

        if (is_first) {
            for (int32_t i = win_idx_beg; i < win_idx_end; ++i) {
                int32_t out_x = win.left_end_lst[i];
                memcpy(dst + out_x, src + win.offsets[i] + out_x, win.mid_len * sizeof(T));
            }
        } else {
            int32_t out_x_beg = win.num_left_elem * channels;
            int32_t out_x_end = out_x_beg + win.mid_len;
            for (int32_t out_x = out_x_beg; out_x < out_x_end; out_x += num_unroll) {
                const size_t vl = vsetvl_exmy<T, RVV_LMUL>(out_x_end - out_x);
                auto out_v = vlex_v_my<T, RVV_LMUL>(dst + out_x, vl);
                int32_t i = win_idx_beg;
                for (; i <= win_idx_end - 4; i += 4) {
                    out_v = morph_vec_op<T, RVV_LMUL, morph_type>(vlex_v_my<T, RVV_LMUL>(src + win.offsets[i + 0] + out_x, vl), out_v, vl);
                    out_v = morph_vec_op<T, RVV_LMUL, morph_type>(vlex_v_my<T, RVV_LMUL>(src + win.offsets[i + 1] + out_x, vl), out_v, vl);
                    out_v = morph_vec_op<T, RVV_LMUL, morph_type>(vlex_v_my<T, RVV_LMUL>(src + win.offsets[i + 2] + out_x, vl), out_v, vl);
                    out_v = morph_vec_op<T, RVV_LMUL, morph_type>(vlex_v_my<T, RVV_LMUL>(src + win.offsets[i + 3] + out_x, vl), out_v, vl);
                }
                for (; i < win_idx_end; ++i) {
                    out_v = morph_vec_op<T, RVV_LMUL, morph_type>(vlex_v_my<T, RVV_LMUL>(src + win.offsets[i] + out_x, vl), out_v, vl);
                }
                vsex_v_my<T, RVV_LMUL>(dst + out_x, out_v, vl);
            }
        }
        for (int32_t i = win_idx_beg; i < win_idx_end; ++i) {
            int32_t out_x = win.right_beg_lst[i];
            int32_t in_x = win.offsets[i] + out_x;
            auto in_x_end = win.offsets[i] + win.right_end_lst[i];
            for (; in_x < in_x_end; in_x += num_unroll, out_x += num_unroll) {
                const size_t vl = vsetvl_exmy<T, RVV_LMUL>(in_x_end - in_x);
                auto out_v = vlex_v_my<T, RVV_LMUL>(src + in_x, vl);
                if (!is_first) {
                    out_v = morph_vec_op<T, RVV_LMUL, morph_type>(vlex_v_my<T, RVV_LMUL>(dst + out_x, vl), out_v, vl);
                }
                vsex_v_my<T, RVV_LMUL>(dst + out_x, out_v, vl);
            }
        }
        // right
        for (int32_t i = win_idx_beg; i < win_idx_end; ++i) {
            if (win.offsets[i] > win.bases[i]) {
                int32_t in_x = win.bases[i] + width * channels, out_x = win.bases[i] - win.offsets[i] + width * channels;
                auto in_x_end = win.offsets[i] + width * channels;
                for (; in_x < in_x_end; in_x += channels, out_x += channels) {
                    int32_t real_in_x = win.bases[i] + border_interpolate<border_mode, false>((in_x - win.bases[i]) / channels, width) * channels;

                    T in0, in1, in2, in3;
                    if (channels > 0) in0 = (border_mode == ppl::cv::BORDER_CONSTANT) ? border_value : src[real_in_x + 0];
                    if (channels > 1) in1 = (border_mode == ppl::cv::BORDER_CONSTANT) ? border_value : src[real_in_x + 1];
                    if (channels > 2) in2 = (border_mode == ppl::cv::BORDER_CONSTANT) ? border_value : src[real_in_x + 2];
                    if (channels > 3) in3 = (border_mode == ppl::cv::BORDER_CONSTANT) ? border_value : src[real_in_x + 3];
                    if (channels > 0) dst[out_x + 0] = is_first ? in0 : morph_scalar_op<T, morph_type>(dst[out_x + 0], in0);
                    if (channels > 1) dst[out_x + 1] = is_first ? in1 : morph_scalar_op<T, morph_type>(dst[out_x + 1], in1);
                    if (channels > 2) dst[out_x + 2] = is_first ? in2 : morph_scalar_op<T, morph_type>(dst[out_x + 2], in2);
                    if (channels > 3) dst[out_x + 3] = is_first ? in3 : morph_scalar_op<T, morph_type>(dst[out_x + 3], in3);
                }
            }
        }
    }
};

template <typename T, MorphType morph_type, ppl::cv::BorderType border_mode, int32_t channels>
::ppl::common::RetCode morph(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T *inData,
    int32_t kernelx_len,
    int32_t kernely_len,
    const uint8_t *kernel,
    int32_t outWidthStride,
    T *outData,
    T border_value)
{
    int32_t num_top_pad = kernely_len >> 1;
    int32_t num_bottom_pad = kernely_len - 1 - num_top_pad;
    int32_t num_left_pad = kernelx_len >> 1;

    if (num_top_pad > height - 1 || num_left_pad > width - 1) {
        return ppl::common::RC_UNSUPPORTED;
    }

    int32_t num_kernel_elem = 0;
    for (int32_t i = 0; i < kernelx_len * kernely_len; ++i) {
        num_kernel_elem += (bool)kernel[i];
    }
    if (num_kernel_elem == 0) {
        return ppl::common::RC_SUCCESS;
    }

    FilterWin<border_mode> filter_wins[num_top_pad + num_bottom_pad + 1];
    init_morph_win<border_mode>(height, width, inWidthStride, channels, kernel, num_kernel_elem, kernelx_len, kernely_len, filter_wins);

    int32_t dst_y = 0;
    auto src = inData;
    auto dst = outData;
    for (; dst_y < num_top_pad; ++dst_y, dst += outWidthStride) {
        MorphRowFunc<T, morph_type, border_mode, channels, true, true>::f(src, width, filter_wins[dst_y], border_value, dst);
        MorphRowFunc<T, morph_type, border_mode, channels, true, false>::f(src, width, filter_wins[dst_y], border_value, dst);
    }
    for (; dst_y < height - num_bottom_pad; ++dst_y, src += inWidthStride, dst += outWidthStride) {
        MorphRowFunc<T, morph_type, border_mode, channels, false, true>::f(src, width, filter_wins[num_top_pad], border_value, dst);
        MorphRowFunc<T, morph_type, border_mode, channels, false, false>::f(src, width, filter_wins[num_top_pad], border_value, dst);
    }
    for (int pad_i = 0; dst_y < height; ++dst_y, src += inWidthStride, dst += outWidthStride, ++pad_i) {
        MorphRowFunc<T, morph_type, border_mode, channels, true, true>::f(src, width, filter_wins[num_top_pad + 1 + pad_i], border_value, dst);
        MorphRowFunc<T, morph_type, border_mode, channels, true, false>::f(src, width, filter_wins[num_top_pad + 1 + pad_i], border_value, dst);
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::riscv
#endif //__ST_HPC_PPL_CV_RISCV_MORPH_H_
