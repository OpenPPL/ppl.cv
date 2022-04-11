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
#include "ppl/cv/riscv/typetraits.h"
#include "ppl/cv/types.h"
#include <algorithm>
#include <complex>
#include <string.h>

namespace ppl {
namespace cv {
namespace riscv {

::ppl::common::RetCode cvt_color_bgra2bgr_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    constexpr int32_t src_nc = 4;
    constexpr int32_t dst_nc = 3;

    const int32_t num_unroll = 8;

    auto in = src;
    auto out = dst;
    for (int32_t k = 0; k < height; k++, in += srcStride, out += dstStride) {
        int32_t i = 0;
        for (; i <= width - num_unroll; i += num_unroll) {
            out[i * dst_nc + dst_nc * 0 + 0] = in[i * src_nc + src_nc * 0 + 0];
            out[i * dst_nc + dst_nc * 0 + 1] = in[i * src_nc + src_nc * 0 + 1];
            out[i * dst_nc + dst_nc * 0 + 2] = in[i * src_nc + src_nc * 0 + 2];
            out[i * dst_nc + dst_nc * 1 + 0] = in[i * src_nc + src_nc * 1 + 0];
            out[i * dst_nc + dst_nc * 1 + 1] = in[i * src_nc + src_nc * 1 + 1];
            out[i * dst_nc + dst_nc * 1 + 2] = in[i * src_nc + src_nc * 1 + 2];
            out[i * dst_nc + dst_nc * 2 + 0] = in[i * src_nc + src_nc * 2 + 0];
            out[i * dst_nc + dst_nc * 2 + 1] = in[i * src_nc + src_nc * 2 + 1];
            out[i * dst_nc + dst_nc * 2 + 2] = in[i * src_nc + src_nc * 2 + 2];
            out[i * dst_nc + dst_nc * 3 + 0] = in[i * src_nc + src_nc * 3 + 0];
            out[i * dst_nc + dst_nc * 3 + 1] = in[i * src_nc + src_nc * 3 + 1];
            out[i * dst_nc + dst_nc * 3 + 2] = in[i * src_nc + src_nc * 3 + 2];
            out[i * dst_nc + dst_nc * 4 + 0] = in[i * src_nc + src_nc * 4 + 0];
            out[i * dst_nc + dst_nc * 4 + 1] = in[i * src_nc + src_nc * 4 + 1];
            out[i * dst_nc + dst_nc * 4 + 2] = in[i * src_nc + src_nc * 4 + 2];
            out[i * dst_nc + dst_nc * 5 + 0] = in[i * src_nc + src_nc * 5 + 0];
            out[i * dst_nc + dst_nc * 5 + 1] = in[i * src_nc + src_nc * 5 + 1];
            out[i * dst_nc + dst_nc * 5 + 2] = in[i * src_nc + src_nc * 5 + 2];
            out[i * dst_nc + dst_nc * 6 + 0] = in[i * src_nc + src_nc * 6 + 0];
            out[i * dst_nc + dst_nc * 6 + 1] = in[i * src_nc + src_nc * 6 + 1];
            out[i * dst_nc + dst_nc * 6 + 2] = in[i * src_nc + src_nc * 6 + 2];
            out[i * dst_nc + dst_nc * 7 + 0] = in[i * src_nc + src_nc * 7 + 0];
            out[i * dst_nc + dst_nc * 7 + 1] = in[i * src_nc + src_nc * 7 + 1];
            out[i * dst_nc + dst_nc * 7 + 2] = in[i * src_nc + src_nc * 7 + 2];
        }
        for (; i < width; ++i) {
            out[i * dst_nc + 0] = in[i * src_nc + 0];
            out[i * dst_nc + 1] = in[i * src_nc + 1];
            out[i * dst_nc + 2] = in[i * src_nc + 2];
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode cvt_color_bgr2bgra_u8(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const uint8_t* src,
    const int32_t dstStride,
    uint8_t* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    constexpr int32_t src_nc = 3;
    constexpr int32_t dst_nc = 4;

    const size_t vl = vsetvlmax_e8m1();
    const uint8_t uvl = (uint8_t)vl;
    const int32_t num_unroll = vl / dst_nc;

    const uint8_t data_idx_lst[] = {0, 1, 2, uvl, 3, 4, 5, uvl, 6, 7, 8, uvl, 9, 10, 11, uvl};
    const uint8_t const_idx_lst[] = {uvl, uvl, uvl, 3, uvl, uvl, uvl, 7, uvl, uvl, uvl, 11, uvl, uvl, uvl, 15};

    vuint8m1_t data_idx_v = vle8_v_u8m1(data_idx_lst, vl);
    vuint8m1_t const_idx_v = vle8_v_u8m1(const_idx_lst, vl);
    vuint8m1_t const_v = vrgather_vv_u8m1(vmv_v_x_u8m1(255, vl), const_idx_v, vl);

    auto in = src;
    auto out = dst;
    for (int32_t k = 0; k < height; k++, in += srcStride, out += dstStride) {
        int32_t i = 0;
        for (; i <= width - num_unroll; i += num_unroll) {
            vuint8m1_t in_v = vrgather_vv_u8m1(vle8_v_u8m1(in + i * src_nc, vl), data_idx_v, vl);
            vuint8m1_t out_v = vor_vv_u8m1(in_v, const_v, vl);
            vse8_v_u8m1(out + i * dst_nc, out_v, vl);
        }
        for (; i < width; ++i) {
            *(uint32_t*)(out + i * dst_nc) = *(uint32_t*)(in + i * src_nc);
            out[i * dst_nc + 3] = 255;
        }
    }

    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode cvt_color_bgr2bgra_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float* src,
    const int32_t dstStride,
    float* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    constexpr int32_t src_nc = 3;
    constexpr int32_t dst_nc = 4;

    const size_t vl = vsetvlmax_e32m1();
    const int32_t num_unroll = vl / dst_nc * 3;

    const uint32_t const_idx_lst[] = {0, 0, 0, 1};
    vbool32_t merge_idx_v = vmsne_vx_u32m1_b32(vle32_v_u32m1(const_idx_lst, vl), 0, vl);

    auto in = src;
    auto out = dst;
    for (int32_t k = 0; k < height; k++) {
        int32_t i = 0;
        for (; i <= width - num_unroll; i += num_unroll, out += dst_nc * num_unroll, in += src_nc * num_unroll) {
            vse32_v_f32m1(out, vfmerge_vfm_f32m1(merge_idx_v, vle32_v_f32m1(in, vl), 1.f, vl), vl);
            vse32_v_f32m1(out + dst_nc, vfmerge_vfm_f32m1(merge_idx_v, vle32_v_f32m1(in + src_nc, vl), 1.f, vl), vl);
            vse32_v_f32m1(out + 2 * dst_nc, vfmerge_vfm_f32m1(merge_idx_v, vle32_v_f32m1(in + 2 * src_nc, vl), 1.f, vl), vl);
        }
        for (; i < width; ++i, out += dst_nc, in += src_nc) {
            vse32_v_f32m1(out, vfmerge_vfm_f32m1(merge_idx_v, vle32_v_f32m1(in, vl), 1.f, vl), vl);
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode cvt_color_bgra2bgr_f32(
    const int32_t height,
    const int32_t width,
    const int32_t srcStride,
    const float* src,
    const int32_t dstStride,
    float* dst)
{
    if (!src || !dst || height == 0 || width == 0 || srcStride == 0 || dstStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    constexpr int32_t src_nc = 4;
    constexpr int32_t dst_nc = 3;

    auto in = src;
    auto out = dst;
    for (int32_t k = 0; k < height; k++, in += srcStride, out += dstStride) {
        int32_t i = 0;
        for (; i < width; ++i) {
            out[i * dst_nc + 0] = in[i * src_nc + 0];
            out[i * dst_nc + 1] = in[i * src_nc + 1];
            out[i * dst_nc + 2] = in[i * src_nc + 2];
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode BGR2BGRA<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_bgr2bgra_u8(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGR2BGRA<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_bgr2bgra_f32(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGRA2BGR<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outData)
{
    return cvt_color_bgra2bgr_u8(height, width, inWidthStride, inData, outWidthStride, outData);
}
template <>
::ppl::common::RetCode BGRA2BGR<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outData)
{
    return cvt_color_bgra2bgr_f32(height, width, inWidthStride, inData, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::riscv
