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

#ifndef __INTRINUTILS_NEON_H__
#define __INTRINUTILS_NEON_H__

#include <algorithm>

#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"

#include <arm_neon.h>

namespace ppl::cv::arm {

static inline void neon_transpose_f32_4x4(float32x4_t &va, float32x4_t &vb, float32x4_t &vc, float32x4_t &vd)
{
    float32x4_t v02_0 = vzip1q_f32(va, vc);
    float32x4_t v13_0 = vzip1q_f32(vb, vd);
    float32x4_t v02_2 = vzip2q_f32(va, vc);
    float32x4_t v13_2 = vzip2q_f32(vb, vd);
    va = vzip1q_f32(v02_0, v13_0);
    vb = vzip2q_f32(v02_0, v13_0);
    vc = vzip1q_f32(v02_2, v13_2);
    vd = vzip2q_f32(v02_2, v13_2);
}

static inline void neon_transpose_u32_4x4(uint32x4_t &va, uint32x4_t &vb, uint32x4_t &vc, uint32x4_t &vd)
{
    uint32x4_t v02_0 = vzip1q_u32(va, vc);
    uint32x4_t v13_0 = vzip1q_u32(vb, vd);
    uint32x4_t v02_2 = vzip2q_u32(va, vc);
    uint32x4_t v13_2 = vzip2q_u32(vb, vd);
    va = vzip1q_u32(v02_0, v13_0);
    vb = vzip2q_u32(v02_0, v13_0);
    vc = vzip1q_u32(v02_2, v13_2);
    vd = vzip2q_u32(v02_2, v13_2);
}

static inline void neon_transpose_u8_8x8(uint8x8_t &va,
                                         uint8x8_t &vb,
                                         uint8x8_t &vc,
                                         uint8x8_t &vd,
                                         uint8x8_t &ve,
                                         uint8x8_t &vf,
                                         uint8x8_t &vg,
                                         uint8x8_t &vh)
{
    uint8x8_t v04_0 = vzip1_u8(va, ve);
    uint8x8_t v15_0 = vzip1_u8(vb, vf);
    uint8x8_t v26_0 = vzip1_u8(vc, vg);
    uint8x8_t v37_0 = vzip1_u8(vd, vh);
    uint8x8_t v04_4 = vzip2_u8(va, ve);
    uint8x8_t v15_4 = vzip2_u8(vb, vf);
    uint8x8_t v26_4 = vzip2_u8(vc, vg);
    uint8x8_t v37_4 = vzip2_u8(vd, vh);

    uint8x8_t v0246_0 = vzip1_u8(v04_0, v26_0);
    uint8x8_t v1357_0 = vzip1_u8(v15_0, v37_0);
    uint8x8_t v0246_2 = vzip2_u8(v04_0, v26_0);
    uint8x8_t v1357_2 = vzip2_u8(v15_0, v37_0);
    uint8x8_t v0246_4 = vzip1_u8(v04_4, v26_4);
    uint8x8_t v1357_4 = vzip1_u8(v15_4, v37_4);
    uint8x8_t v0246_6 = vzip2_u8(v04_4, v26_4);
    uint8x8_t v1357_6 = vzip2_u8(v15_4, v37_4);

    va = vzip1_u8(v0246_0, v1357_0);
    vb = vzip2_u8(v0246_0, v1357_0);
    vc = vzip1_u8(v0246_2, v1357_2);
    vd = vzip2_u8(v0246_2, v1357_2);
    ve = vzip1_u8(v0246_4, v1357_4);
    vf = vzip2_u8(v0246_4, v1357_4);
    vg = vzip1_u8(v0246_6, v1357_6);
    vh = vzip2_u8(v0246_6, v1357_6);
}

static inline void neon_transpose_u8_8x8_new_device(uint8x8_t &va,
                                                    uint8x8_t &vb,
                                                    uint8x8_t &vc,
                                                    uint8x8_t &vd,
                                                    uint8x8_t &ve,
                                                    uint8x8_t &vf,
                                                    uint8x8_t &vg,
                                                    uint8x8_t &vh)
{
    uint8x16_t v04 = vzip1q_u8(vcombine_u8(va, va), vcombine_u8(ve, ve));
    uint8x16_t v15 = vzip1q_u8(vcombine_u8(vb, vb), vcombine_u8(vf, vf));
    uint8x16_t v26 = vzip1q_u8(vcombine_u8(vc, vc), vcombine_u8(vg, vg));
    uint8x16_t v37 = vzip1q_u8(vcombine_u8(vd, vd), vcombine_u8(vh, vh));

    uint8x16_t v0246_0 = vzip1q_u8(v04, v26);
    uint8x16_t v1357_0 = vzip1q_u8(v15, v37);
    uint8x16_t v0246_4 = vzip2q_u8(v04, v26);
    uint8x16_t v1357_4 = vzip2q_u8(v15, v37);

    uint8x16_t vab = vzip1q_u8(v0246_0, v1357_0);
    uint8x16_t vcd = vzip2q_u8(v0246_0, v1357_0);
    uint8x16_t vef = vzip1q_u8(v0246_4, v1357_4);
    uint8x16_t vgh = vzip2q_u8(v0246_4, v1357_4);

    va = vget_low_u8(vab);
    vb = vget_high_u8(vab);
    vc = vget_low_u8(vcd);
    vd = vget_high_u8(vcd);
    ve = vget_low_u8(vef);
    vf = vget_high_u8(vef);
    vg = vget_low_u8(vgh);
    vh = vget_high_u8(vgh);
}

static inline uint8x8_t neon_reverse_u8x8(uint8x8_t va)
{
    return vrev64_u8(va);
}

static inline uint8x16_t neon_reverse_u8x16(uint8x16_t va)
{
    uint8x16_t rev_inlane = vrev64q_u8(va);
    return vcombine_u8(vget_high_u8(rev_inlane), vget_low_u8(rev_inlane));
}

static inline float32x2_t neon_reverse_f32x2(float32x2_t va)
{
    return vrev64_f32(va);
}

static inline float32x4_t neon_reverse_f32x4(float32x4_t va)
{
    float32x4_t rev_inlane = vrev64q_f32(va);
    return vcombine_f32(vget_high_f32(rev_inlane), vget_low_f32(rev_inlane));
}

static inline uint32x4_t neon_reverse_u32x4(uint32x4_t va)
{
    uint32x4_t rev_inlane = vrev64q_u32(va);
    return vcombine_u32(vget_high_u32(rev_inlane), vget_low_u32(rev_inlane));
}

static inline void neon_rotate90_f32_4x4(float32x4_t &va, float32x4_t &vb, float32x4_t &vc, float32x4_t &vd)
{
    neon_transpose_f32_4x4(va, vb, vc, vd);
    va = neon_reverse_f32x4(va);
    vb = neon_reverse_f32x4(vb);
    vc = neon_reverse_f32x4(vc);
    vd = neon_reverse_f32x4(vd);
}

static inline void neon_rotate90_u32_4x4(uint32x4_t &va, uint32x4_t &vb, uint32x4_t &vc, uint32x4_t &vd)
{
    neon_transpose_u32_4x4(va, vb, vc, vd);
    va = neon_reverse_u32x4(va);
    vb = neon_reverse_u32x4(vb);
    vc = neon_reverse_u32x4(vc);
    vd = neon_reverse_u32x4(vd);
}

static inline void neon_rotate270_f32_4x4(float32x4_t &va, float32x4_t &vb, float32x4_t &vc, float32x4_t &vd)
{
    neon_transpose_f32_4x4(va, vb, vc, vd);
    std::swap(va, vd);
    std::swap(vb, vc);
}

static inline void neon_rotate270_u32_4x4(uint32x4_t &va, uint32x4_t &vb, uint32x4_t &vc, uint32x4_t &vd)
{
    neon_transpose_u32_4x4(va, vb, vc, vd);
    std::swap(va, vd);
    std::swap(vb, vc);
}

static inline void neon_rotate90_u8_8x8(uint8x8_t &va,
                                        uint8x8_t &vb,
                                        uint8x8_t &vc,
                                        uint8x8_t &vd,
                                        uint8x8_t &ve,
                                        uint8x8_t &vf,
                                        uint8x8_t &vg,
                                        uint8x8_t &vh)
{
    neon_transpose_u8_8x8(va, vb, vc, vd, ve, vf, vg, vh);
    va = neon_reverse_u8x8(va);
    vb = neon_reverse_u8x8(vb);
    vc = neon_reverse_u8x8(vc);
    vd = neon_reverse_u8x8(vd);
    ve = neon_reverse_u8x8(ve);
    vf = neon_reverse_u8x8(vf);
    vg = neon_reverse_u8x8(vg);
    vh = neon_reverse_u8x8(vh);
}

static inline void neon_rotate270_u8_8x8(uint8x8_t &va,
                                         uint8x8_t &vb,
                                         uint8x8_t &vc,
                                         uint8x8_t &vd,
                                         uint8x8_t &ve,
                                         uint8x8_t &vf,
                                         uint8x8_t &vg,
                                         uint8x8_t &vh)
{
    neon_transpose_u8_8x8(va, vb, vc, vd, ve, vf, vg, vh);
    std::swap(va, vh);
    std::swap(vb, vg);
    std::swap(vc, vf);
    std::swap(vd, ve);
}

static inline uint32x4_t neon_scan_across_vector_u32x4(uint32x4_t vIn) {
    uint32x4_t vZero = vdupq_n_u32(0);

    uint32x4_t vResShft1 = vextq_u32(vZero, vIn, 3);
    uint32x4_t vResStep1 = vaddq_u32(vIn, vResShft1);
    
    uint32x4_t vResShft2 = vextq_u32(vZero, vResStep1, 2);
    uint32x4_t vResStep2 = vaddq_u32(vResStep1, vResShft2);

    return vResStep2;
}

static inline float32x4_t neon_scan_across_vector_f32x4(float32x4_t vIn) {
    float32x4_t vZero = vdupq_n_f32(0);

    float32x4_t vResShft1 = vextq_f32(vZero, vIn, 3);
    float32x4_t vResStep1 = vaddq_f32(vIn, vResShft1);
    
    float32x4_t vResShft2 = vextq_f32(vZero, vResStep1, 2);
    float32x4_t vResStep2 = vaddq_f32(vResStep1, vResShft2);

    return vResStep2;
}

} // namespace ppl::cv::arm
#endif