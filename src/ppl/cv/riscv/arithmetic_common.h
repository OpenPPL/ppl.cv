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

#include "ppl/cv/types.h"
#include "ppl/cv/riscv/util.h"
#include "ppl/cv/riscv/typetraits.h"

#ifndef PPL_CV_RISCV_ARITHMETIC_COMMON_H_
#define PPL_CV_RISCV_ARITHMETIC_COMMON_H_

namespace ppl {
namespace cv {
namespace riscv {

enum arithmetic_op_type_t {
    ARITHMETIC_ADD = 0,
    ARITHMETIC_SUB = 1,
    ARITHMETIC_MUL = 2,
    ARITHMETIC_DIV = 3
};

template <typename eT, int32_t lmul, arithmetic_op_type_t op>
inline vdataxmy_t<eT, lmul> arithmetic_vector(vdataxmy_t<eT, lmul> a, vdataxmy_t<eT, lmul> b, size_t vl);

template <>
inline vfloat32m1_t arithmetic_vector<float, 1, ARITHMETIC_ADD>(vfloat32m1_t a, vfloat32m1_t b, size_t vl)
{
    return vfadd_vv_f32m1(a, b, vl);
}
template <>
inline vfloat32m1_t arithmetic_vector<float, 1, ARITHMETIC_SUB>(vfloat32m1_t a, vfloat32m1_t b, size_t vl)
{
    return vfsub_vv_f32m1(a, b, vl);
}
template <>
inline vfloat32m1_t arithmetic_vector<float, 1, ARITHMETIC_MUL>(vfloat32m1_t a, vfloat32m1_t b, size_t vl)
{
    return vfmul_vv_f32m1(a, b, vl);
}
template <>
inline vfloat32m1_t arithmetic_vector<float, 1, ARITHMETIC_DIV>(vfloat32m1_t a, vfloat32m1_t b, size_t vl)
{
    return vfdiv_vv_f32m1(a, b, vl);
}
template <>
inline vuint8m1_t arithmetic_vector<uint8_t, 1, ARITHMETIC_ADD>(vuint8m1_t a, vuint8m1_t b, size_t vl)
{
    return vsaddu_vv_u8m1(a, b, vl);
}
template <>
inline vuint8m1_t arithmetic_vector<uint8_t, 1, ARITHMETIC_SUB>(vuint8m1_t a, vuint8m1_t b, size_t vl)
{
    return vssubu_vv_u8m1(a, b, vl);
}
template <>
inline vuint8m1_t arithmetic_vector<uint8_t, 1, ARITHMETIC_MUL>(vuint8m1_t a, vuint8m1_t b, size_t vl)
{
    return vnclipu_wx_u8m1(vwmulu_vv_u16m2(a, b, vl), 0, vl);
}
template <>
inline vuint8m1_t arithmetic_vector<uint8_t, 1, ARITHMETIC_DIV>(vuint8m1_t a, vuint8m1_t b, size_t vl)
{
    return vmerge_vxm_u8m1(vmseq_vx_u8m1_b8(b, 0, vl), vdivu_vv_u8m1(a, b, vl), 0, vl);
}

template <>
inline vfloat32m4_t arithmetic_vector<float, 4, ARITHMETIC_ADD>(vfloat32m4_t a, vfloat32m4_t b, size_t vl)
{
    return vfadd_vv_f32m4(a, b, vl);
}
template <>
inline vfloat32m4_t arithmetic_vector<float, 4, ARITHMETIC_SUB>(vfloat32m4_t a, vfloat32m4_t b, size_t vl)
{
    return vfsub_vv_f32m4(a, b, vl);
}
template <>
inline vfloat32m4_t arithmetic_vector<float, 4, ARITHMETIC_MUL>(vfloat32m4_t a, vfloat32m4_t b, size_t vl)
{
    return vfmul_vv_f32m4(a, b, vl);
}
template <>
inline vfloat32m4_t arithmetic_vector<float, 4, ARITHMETIC_DIV>(vfloat32m4_t a, vfloat32m4_t b, size_t vl)
{
    return vfdiv_vv_f32m4(a, b, vl);
}
template <>
inline vuint8m4_t arithmetic_vector<uint8_t, 4, ARITHMETIC_ADD>(vuint8m4_t a, vuint8m4_t b, size_t vl)
{
    return vsaddu_vv_u8m4(a, b, vl);
}
template <>
inline vuint8m4_t arithmetic_vector<uint8_t, 4, ARITHMETIC_SUB>(vuint8m4_t a, vuint8m4_t b, size_t vl)
{
    return vssubu_vv_u8m4(a, b, vl);
}
template <>
inline vuint8m4_t arithmetic_vector<uint8_t, 4, ARITHMETIC_MUL>(vuint8m4_t a, vuint8m4_t b, size_t vl)
{
    return vnclipu_wx_u8m4(vwmulu_vv_u16m8(a, b, vl), 0, vl);
}
template <>
inline vuint8m4_t arithmetic_vector<uint8_t, 4, ARITHMETIC_DIV>(vuint8m4_t a, vuint8m4_t b, size_t vl)
{
    return vmerge_vxm_u8m4(vmseq_vx_u8m4_b2(b, 0, vl), vdivu_vv_u8m4(a, b, vl), 0, vl);
}

template <typename eT, arithmetic_op_type_t op>
::ppl::common::RetCode arithmetic_op(
    int32_t height,
    int32_t width,
    int32_t channels,
    int32_t inWidthStride0,
    const eT *inData0,
    int32_t inWidthStride1,
    const eT *inData1,
    int32_t outWidthStride,
    eT *outData)
{
    if (nullptr == inData0 && nullptr == inData1 && nullptr == outData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width == 0 || height == 0 || inWidthStride0 == 0 || inWidthStride1 == 0 || outWidthStride == 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    constexpr int32_t RVV_LMUL = 4;
    size_t vl = vsetvlmax_exmy<eT, RVV_LMUL>();
    const int32_t num_vec_elem = vl;

    width *= channels;
    for (int32_t i = 0; i < height; ++i) {
        int32_t j = 0;
        const eT *in0 = inData0 + i * inWidthStride0;
        const eT *in1 = inData1 + i * inWidthStride1;
        eT *out = outData + i * outWidthStride;

        for (; j <= width - 2 * num_vec_elem; j += 2 * num_vec_elem) {
            vdataxmy_t<eT, RVV_LMUL> data0_in0_v = vlex_v_my<eT, RVV_LMUL>(in0 + j, vl);
            vdataxmy_t<eT, RVV_LMUL> data0_in1_v = vlex_v_my<eT, RVV_LMUL>(in1 + j, vl);
            vdataxmy_t<eT, RVV_LMUL> data1_in0_v = vlex_v_my<eT, RVV_LMUL>(in0 + num_vec_elem + j, vl);
            vdataxmy_t<eT, RVV_LMUL> data1_in1_v = vlex_v_my<eT, RVV_LMUL>(in1 + num_vec_elem + j, vl);
            vdataxmy_t<eT, RVV_LMUL> data0_out_v = arithmetic_vector<eT, RVV_LMUL, op>(data0_in0_v, data0_in1_v, vl);
            vdataxmy_t<eT, RVV_LMUL> data1_out_v = arithmetic_vector<eT, RVV_LMUL, op>(data1_in0_v, data1_in1_v, vl);
            vsex_v_my<eT, RVV_LMUL>(out + j, data0_out_v, vl);
            vsex_v_my<eT, RVV_LMUL>(out + num_vec_elem + j, data1_out_v, vl);
        }
        for (; j < width; j += num_vec_elem) {
            const size_t vl = vsetvl_exmy<eT, RVV_LMUL>(width - j);
            vdataxmy_t<eT, RVV_LMUL> data0_in0_v = vlex_v_my<eT, RVV_LMUL>(in0 + j, vl);
            vdataxmy_t<eT, RVV_LMUL> data0_in1_v = vlex_v_my<eT, RVV_LMUL>(in1 + j, vl);
            vdataxmy_t<eT, RVV_LMUL> data0_out_v = arithmetic_vector<eT, RVV_LMUL, op>(data0_in0_v, data0_in1_v, vl);
            vsex_v_my<eT, RVV_LMUL>(out + j, data0_out_v, vl);
        }
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::riscv

#endif //! __ST_HPC_PPL_CV_RISCV_ARITHMETIC_COMMON_H_