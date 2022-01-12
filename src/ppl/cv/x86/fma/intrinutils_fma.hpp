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

#ifndef __INTRINUTILS_FMA_H__
#define __INTRINUTILS_FMA_H__

#include "ppl/cv/types.h"
#include <immintrin.h>
#include <stdio.h>

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

inline void v_load_deinterleave(const uint8_t *ptr, __m256i &a, __m256i &b, __m256i &c)
{
    __m256i bgr0 = _mm256_loadu_si256((const __m256i *)ptr);
    __m256i bgr1 = _mm256_loadu_si256((const __m256i *)(ptr + 32));
    __m256i bgr2 = _mm256_loadu_si256((const __m256i *)(ptr + 64));

    __m256i s02_low  = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
    __m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

    const __m256i m0 = _mm256_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    const __m256i m1 = _mm256_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);

    __m256i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), bgr1, m1);
    __m256i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), bgr1, m0);
    __m256i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, m0), s02_high, m1);

    const __m256i
        sh_b = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13),
        sh_g = _mm256_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14),
        sh_r = _mm256_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
    a        = _mm256_shuffle_epi8(b0, sh_b);
    b        = _mm256_shuffle_epi8(g0, sh_g);
    c        = _mm256_shuffle_epi8(r0, sh_r);
}

inline void v_load_deinterleave(const uchar *ptr, __m256i &a, __m256i &b, __m256i &c, __m256i &d)
{
    __m256i bgr0     = _mm256_loadu_si256((const __m256i *)ptr);
    __m256i bgr1     = _mm256_loadu_si256((const __m256i *)(ptr + 32));
    __m256i bgr2     = _mm256_loadu_si256((const __m256i *)(ptr + 64));
    __m256i bgr3     = _mm256_loadu_si256((const __m256i *)(ptr + 96));
    const __m256i sh = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    __m256i p0 = _mm256_shuffle_epi8(bgr0, sh);
    __m256i p1 = _mm256_shuffle_epi8(bgr1, sh);
    __m256i p2 = _mm256_shuffle_epi8(bgr2, sh);
    __m256i p3 = _mm256_shuffle_epi8(bgr3, sh);

    __m256i p01l = _mm256_unpacklo_epi32(p0, p1);
    __m256i p01h = _mm256_unpackhi_epi32(p0, p1);
    __m256i p23l = _mm256_unpacklo_epi32(p2, p3);
    __m256i p23h = _mm256_unpackhi_epi32(p2, p3);

    __m256i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2 * 16);
    __m256i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3 * 16);
    __m256i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2 * 16);
    __m256i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3 * 16);

    a = _mm256_unpacklo_epi32(pll, plh);
    b = _mm256_unpackhi_epi32(pll, plh);
    c = _mm256_unpacklo_epi32(phl, phh);
    d = _mm256_unpackhi_epi32(phl, phh);
}

inline void v_load_deinterleave(const float *ptr, __m256 &a, __m256 &b, __m256 &c, __m256 &d)
{
    __m256i p0 = _mm256_loadu_si256((const __m256i *)ptr);
    __m256i p1 = _mm256_loadu_si256((const __m256i *)(ptr + 8));
    __m256i p2 = _mm256_loadu_si256((const __m256i *)(ptr + 16));
    __m256i p3 = _mm256_loadu_si256((const __m256i *)(ptr + 24));

    __m256i p01l = _mm256_unpacklo_epi32(p0, p1);
    __m256i p01h = _mm256_unpackhi_epi32(p0, p1);
    __m256i p23l = _mm256_unpacklo_epi32(p2, p3);
    __m256i p23h = _mm256_unpackhi_epi32(p2, p3);

    __m256i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2 * 16);
    __m256i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3 * 16);
    __m256i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2 * 16);
    __m256i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3 * 16);

    __m256i b0 = _mm256_unpacklo_epi32(pll, plh);
    __m256i g0 = _mm256_unpackhi_epi32(pll, plh);
    __m256i r0 = _mm256_unpacklo_epi32(phl, phh);
    __m256i a0 = _mm256_unpackhi_epi32(phl, phh);

    a = _mm256_castsi256_ps(b0);
    b = _mm256_castsi256_ps(g0);
    c = _mm256_castsi256_ps(r0);
    d = _mm256_castsi256_ps(a0);
}

}
}
}
} // namespace ppl::cv::x86::fma

#endif
