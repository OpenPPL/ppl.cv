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

#ifndef __ST_HPC_PPL_CV_RISCV_TYPETRAITS_H_
#define __ST_HPC_PPL_CV_RISCV_TYPETRAITS_H_
#include <stdint.h>
#include <riscv_vector.h>

namespace ppl {
namespace cv {
namespace riscv {

template <typename eT>
int32_t get_num_vec_elem(size_t vl)
{
    return vl / sizeof(eT);
}

// vector type
template <typename eT, int32_t lmul>
struct vdataxmy_t_helper;

template <typename eT, int32_t lmul>
using vdataxmy_t = typename vdataxmy_t_helper<eT, lmul>::U;

template <>
struct vdataxmy_t_helper<float, 1> {
    typedef vfloat32m1_t U;
};

template <>
struct vdataxmy_t_helper<float, 4> {
    typedef vfloat32m4_t U;
};

template <>
struct vdataxmy_t_helper<uint8_t, 1> {
    typedef vuint8m1_t U;
};

template <>
struct vdataxmy_t_helper<uint8_t, 4> {
    typedef vuint8m4_t U;
};

// vsetvl
template <typename eT, int32_t lmul>
inline size_t vsetvl_exmy(size_t avl);

template <>
inline size_t vsetvl_exmy<float, 1>(size_t avl)
{
    return vsetvl_e32m1(avl);
}

template <>
inline size_t vsetvl_exmy<float, 4>(size_t avl)
{
    return vsetvl_e32m4(avl);
}

template <>
inline size_t vsetvl_exmy<uint8_t, 1>(size_t avl)
{
    return vsetvl_e8m1(avl);
}

template <>
inline size_t vsetvl_exmy<uint8_t, 4>(size_t avl)
{
    return vsetvl_e8m4(avl);
}

template <typename eT, int32_t lmul>
inline size_t vsetvlmax_exmy();

template <>
inline size_t vsetvlmax_exmy<float, 1>()
{
    return vsetvlmax_e32m1();
}

template <>
inline size_t vsetvlmax_exmy<float, 2>()
{
    return vsetvlmax_e32m2();
}

template <>
inline size_t vsetvlmax_exmy<float, 4>()
{
    return vsetvlmax_e32m4();
}

template <>
inline size_t vsetvlmax_exmy<uint8_t, 1>()
{
    return vsetvlmax_e8m1();
}

template <>
inline size_t vsetvlmax_exmy<uint8_t, 4>()
{
    return vsetvlmax_e8m4();
}

// load
template <typename eT, int32_t lmul>
inline vdataxmy_t<eT, lmul> vlex_v_my(const eT *base, size_t vl);

template <>
inline vfloat32m1_t vlex_v_my<float, 1>(const float *base, size_t vl)
{
    return vle32_v_f32m1(base, vl);
}

template <>
inline vfloat32m4_t vlex_v_my<float, 4>(const float *base, size_t vl)
{
    return vle32_v_f32m4(base, vl);
}

template <>
inline vuint8m1_t vlex_v_my<uint8_t, 1>(const uint8_t *base, size_t vl)
{
    return vle8_v_u8m1(base, vl);
}

template <>
inline vuint8m4_t vlex_v_my<uint8_t, 4>(const uint8_t *base, size_t vl)
{
    return vle8_v_u8m4(base, vl);
}

// store
template <typename eT, int32_t lmul>
inline void vsex_v_my(eT *base, vdataxmy_t<eT, lmul> value, size_t vl);

template <>
inline void vsex_v_my<float, 1>(float *base, vfloat32m1_t value, size_t vl)
{
    vse32_v_f32m1(base, value, vl);
}

template <>
inline void vsex_v_my<float, 4>(float *base, vfloat32m4_t value, size_t vl)
{
    vse32_v_f32m4(base, value, vl);
}

template <>
inline void vsex_v_my<uint8_t, 1>(uint8_t *base, vuint8m1_t value, size_t vl)
{
    vse8_v_u8m1(base, value, vl);
}

template <>
inline void vsex_v_my<uint8_t, 4>(uint8_t *base, vuint8m4_t value, size_t vl)
{
    vse8_v_u8m4(base, value, vl);
}

// mv
template <typename eT, int32_t lmul>
inline vdataxmy_t<eT, lmul> vmv_v_x_exmy(eT src, size_t vl);

template <>
inline vdataxmy_t<float, 4> vmv_v_x_exmy<float, 4>(float src, size_t vl)
{
    return vfmv_v_f_f32m4(src, vl);
}

template <>
inline vdataxmy_t<uint8_t, 4> vmv_v_x_exmy<uint8_t, 4>(uint8_t src, size_t vl)
{
    return vmv_v_x_u8m4(src, vl);
}

// max
template <typename eT, int32_t lmul>
inline vdataxmy_t<eT, lmul> vmax_vv_exmy(vdataxmy_t<eT, lmul> op1, vdataxmy_t<eT, lmul> op2, size_t vl);

template <>
inline vdataxmy_t<float, 4> vmax_vv_exmy<float, 4>(vdataxmy_t<float, 4> op1, vdataxmy_t<float, 4> op2, size_t vl)
{
    return vfmax_vv_f32m4(op1, op2, vl);
}

template <>
inline vdataxmy_t<uint8_t, 4> vmax_vv_exmy<uint8_t, 4>(vdataxmy_t<uint8_t, 4> op1, vdataxmy_t<uint8_t, 4> op2, size_t vl)
{
    return vmaxu_vv_u8m4(op1, op2, vl);
}

// min
template <typename eT, int32_t lmul>
inline vdataxmy_t<eT, lmul> vmin_vv_exmy(vdataxmy_t<eT, lmul> op1, vdataxmy_t<eT, lmul> op2, size_t vl);

template <>
inline vdataxmy_t<float, 4> vmin_vv_exmy<float, 4>(vdataxmy_t<float, 4> op1, vdataxmy_t<float, 4> op2, size_t vl)
{
    return vfmin_vv_f32m4(op1, op2, vl);
}

template <>
inline vdataxmy_t<uint8_t, 4> vmin_vv_exmy<uint8_t, 4>(vdataxmy_t<uint8_t, 4> op1, vdataxmy_t<uint8_t, 4> op2, size_t vl)
{
    return vminu_vv_u8m4(op1, op2, vl);
}

}
}
} // namespace ppl::cv::riscv

#endif //! __ST_HPC_PPL_CV_RISCV_TYPETRAITS_H_