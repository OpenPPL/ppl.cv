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

#ifndef __INTRINUTILS_H__
#define __INTRINUTILS_H__

#include <immintrin.h>
#include <stdio.h>
#include "ppl/cv/types.h"

inline void v_load_deinterleave(const uint8_t* ptr, __m128i& a, __m128i& b, __m128i& c)
{
    const __m128i m0   = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i m1   = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    __m128i s0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i s1 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i s2 = _mm_loadu_si128((const __m128i*)(ptr + 32));
    __m128i a0 = _mm_blendv_epi8(_mm_blendv_epi8(s0, s1, m0), s2, m1);
    __m128i b0 = _mm_blendv_epi8(_mm_blendv_epi8(s1, s2, m0), s0, m1);
    __m128i c0 = _mm_blendv_epi8(_mm_blendv_epi8(s2, s0, m0), s1, m1);
    const __m128i sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
    const __m128i sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
    const __m128i sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
    a = _mm_shuffle_epi8(a0, sh_b);
    b = _mm_shuffle_epi8(b0, sh_g);
    c = _mm_shuffle_epi8(c0, sh_r);
}

inline void v_load_deinterleave(const uint8_t* ptr, __m128i& a, __m128i& b, __m128i& c, __m128i& d)
{
    __m128i u0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0 c0 d0 a1 b1 c1 d1 ...
    __m128i u1 = _mm_loadu_si128((const __m128i*)(ptr + 16)); // a4 b4 c4 d4 ...
    __m128i u2 = _mm_loadu_si128((const __m128i*)(ptr + 32)); // a8 b8 c8 d8 ...
    __m128i u3 = _mm_loadu_si128((const __m128i*)(ptr + 48)); // a12 b12 c12 d12 ...

    __m128i v0 = _mm_unpacklo_epi8(u0, u2); // a0 a8 b0 b8 ...
    __m128i v1 = _mm_unpackhi_epi8(u0, u2); // a2 a10 b2 b10 ...
    __m128i v2 = _mm_unpacklo_epi8(u1, u3); // a4 a12 b4 b12 ...
    __m128i v3 = _mm_unpackhi_epi8(u1, u3); // a6 a14 b6 b14 ...

    u0 = _mm_unpacklo_epi8(v0, v2); // a0 a4 a8 a12 ...
    u1 = _mm_unpacklo_epi8(v1, v3); // a2 a6 a10 a14 ...
    u2 = _mm_unpackhi_epi8(v0, v2); // a1 a5 a9 a13 ...
    u3 = _mm_unpackhi_epi8(v1, v3); // a3 a7 a11 a15 ...

    v0 = _mm_unpacklo_epi8(u0, u1); // a0 a2 a4 a6 ...
    v1 = _mm_unpacklo_epi8(u2, u3); // a1 a3 a5 a7 ...
    v2 = _mm_unpackhi_epi8(u0, u1); // c0 c2 c4 c6 ...
    v3 = _mm_unpackhi_epi8(u2, u3); // c1 c3 c5 c7 ...

    a = _mm_unpacklo_epi8(v0, v1);
    b = _mm_unpackhi_epi8(v0, v1);
    c = _mm_unpacklo_epi8(v2, v3);
    d = _mm_unpackhi_epi8(v2, v3);
}

inline void _mm_interleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i v_mask = _mm_set1_epi16(0x00ff);

    __m128i layer4_chunk0 = _mm_packus_epi16(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer4_chunk3 = _mm_packus_epi16(_mm_srli_epi16(v_r0, 8), _mm_srli_epi16(v_r1, 8));
    __m128i layer4_chunk1 = _mm_packus_epi16(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer4_chunk4 = _mm_packus_epi16(_mm_srli_epi16(v_g0, 8), _mm_srli_epi16(v_g1, 8));
    __m128i layer4_chunk2 = _mm_packus_epi16(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer4_chunk5 = _mm_packus_epi16(_mm_srli_epi16(v_b0, 8), _mm_srli_epi16(v_b1, 8));

    __m128i layer3_chunk0 = _mm_packus_epi16(_mm_and_si128(layer4_chunk0, v_mask), _mm_and_si128(layer4_chunk1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk0, 8), _mm_srli_epi16(layer4_chunk1, 8));
    __m128i layer3_chunk1 = _mm_packus_epi16(_mm_and_si128(layer4_chunk2, v_mask), _mm_and_si128(layer4_chunk3, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk2, 8), _mm_srli_epi16(layer4_chunk3, 8));
    __m128i layer3_chunk2 = _mm_packus_epi16(_mm_and_si128(layer4_chunk4, v_mask), _mm_and_si128(layer4_chunk5, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk4, 8), _mm_srli_epi16(layer4_chunk5, 8));

    __m128i layer2_chunk0 = _mm_packus_epi16(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk0, 8), _mm_srli_epi16(layer3_chunk1, 8));
    __m128i layer2_chunk1 = _mm_packus_epi16(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk2, 8), _mm_srli_epi16(layer3_chunk3, 8));
    __m128i layer2_chunk2 = _mm_packus_epi16(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk4, 8), _mm_srli_epi16(layer3_chunk5, 8));

    __m128i layer1_chunk0 = _mm_packus_epi16(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk0, 8), _mm_srli_epi16(layer2_chunk1, 8));
    __m128i layer1_chunk1 = _mm_packus_epi16(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk2, 8), _mm_srli_epi16(layer2_chunk3, 8));
    __m128i layer1_chunk2 = _mm_packus_epi16(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk4, 8), _mm_srli_epi16(layer2_chunk5, 8));

    v_r0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk0, 8), _mm_srli_epi16(layer1_chunk1, 8));
    v_r1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk2, 8), _mm_srli_epi16(layer1_chunk3, 8));
    v_g0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_b1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk4, 8), _mm_srli_epi16(layer1_chunk5, 8));
}

inline void _mm_interleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i v_mask = _mm_set1_epi16(0x00ff);

    __m128i layer4_chunk0 = _mm_packus_epi16(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer4_chunk4 = _mm_packus_epi16(_mm_srli_epi16(v_r0, 8), _mm_srli_epi16(v_r1, 8));
    __m128i layer4_chunk1 = _mm_packus_epi16(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer4_chunk5 = _mm_packus_epi16(_mm_srli_epi16(v_g0, 8), _mm_srli_epi16(v_g1, 8));
    __m128i layer4_chunk2 = _mm_packus_epi16(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer4_chunk6 = _mm_packus_epi16(_mm_srli_epi16(v_b0, 8), _mm_srli_epi16(v_b1, 8));
    __m128i layer4_chunk3 = _mm_packus_epi16(_mm_and_si128(v_a0, v_mask), _mm_and_si128(v_a1, v_mask));
    __m128i layer4_chunk7 = _mm_packus_epi16(_mm_srli_epi16(v_a0, 8), _mm_srli_epi16(v_a1, 8));

    __m128i layer3_chunk0 = _mm_packus_epi16(_mm_and_si128(layer4_chunk0, v_mask), _mm_and_si128(layer4_chunk1, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk0, 8), _mm_srli_epi16(layer4_chunk1, 8));
    __m128i layer3_chunk1 = _mm_packus_epi16(_mm_and_si128(layer4_chunk2, v_mask), _mm_and_si128(layer4_chunk3, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk2, 8), _mm_srli_epi16(layer4_chunk3, 8));
    __m128i layer3_chunk2 = _mm_packus_epi16(_mm_and_si128(layer4_chunk4, v_mask), _mm_and_si128(layer4_chunk5, v_mask));
    __m128i layer3_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk4, 8), _mm_srli_epi16(layer4_chunk5, 8));
    __m128i layer3_chunk3 = _mm_packus_epi16(_mm_and_si128(layer4_chunk6, v_mask), _mm_and_si128(layer4_chunk7, v_mask));
    __m128i layer3_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk6, 8), _mm_srli_epi16(layer4_chunk7, 8));

    __m128i layer2_chunk0 = _mm_packus_epi16(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk0, 8), _mm_srli_epi16(layer3_chunk1, 8));
    __m128i layer2_chunk1 = _mm_packus_epi16(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk2, 8), _mm_srli_epi16(layer3_chunk3, 8));
    __m128i layer2_chunk2 = _mm_packus_epi16(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk4, 8), _mm_srli_epi16(layer3_chunk5, 8));
    __m128i layer2_chunk3 = _mm_packus_epi16(_mm_and_si128(layer3_chunk6, v_mask), _mm_and_si128(layer3_chunk7, v_mask));
    __m128i layer2_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk6, 8), _mm_srli_epi16(layer3_chunk7, 8));

    __m128i layer1_chunk0 = _mm_packus_epi16(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk0, 8), _mm_srli_epi16(layer2_chunk1, 8));
    __m128i layer1_chunk1 = _mm_packus_epi16(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk2, 8), _mm_srli_epi16(layer2_chunk3, 8));
    __m128i layer1_chunk2 = _mm_packus_epi16(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk4, 8), _mm_srli_epi16(layer2_chunk5, 8));
    __m128i layer1_chunk3 = _mm_packus_epi16(_mm_and_si128(layer2_chunk6, v_mask), _mm_and_si128(layer2_chunk7, v_mask));
    __m128i layer1_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk6, 8), _mm_srli_epi16(layer2_chunk7, 8));

    v_r0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_b0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk0, 8), _mm_srli_epi16(layer1_chunk1, 8));
    v_r1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk2, 8), _mm_srli_epi16(layer1_chunk3, 8));
    v_g0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_a0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk4, 8), _mm_srli_epi16(layer1_chunk5, 8));
    v_g1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk6, v_mask), _mm_and_si128(layer1_chunk7, v_mask));
    v_a1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk6, 8), _mm_srli_epi16(layer1_chunk7, 8));
}

inline void v_load_deinterleave(const float* ptr, __m128& a, __m128& b, __m128& c)
{
    __m128 t0 = _mm_loadu_ps(ptr + 0);
    __m128 t1 = _mm_loadu_ps(ptr + 4);
    __m128 t2 = _mm_loadu_ps(ptr + 8);

    __m128 at12 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(0, 1, 0, 2));
    a = _mm_shuffle_ps(t0, at12, _MM_SHUFFLE(2, 0, 3, 0));

    __m128 bt01 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 bt12 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(0, 2, 0, 3));
    b = _mm_shuffle_ps(bt01, bt12, _MM_SHUFFLE(2, 0, 2, 0));

    __m128 ct01 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(0, 1, 0, 2));
    c = _mm_shuffle_ps(ct01, t2, _MM_SHUFFLE(3, 0, 2, 0));
}

inline void v_load_deinterleave(const float* ptr, __m128& a, __m128& b, __m128& c, __m128& d)
{
    __m128 t0 = _mm_loadu_ps(ptr +  0);
    __m128 t1 = _mm_loadu_ps(ptr +  4);
    __m128 t2 = _mm_loadu_ps(ptr +  8);
    __m128 t3 = _mm_loadu_ps(ptr + 12);
    __m128 t02lo = _mm_unpacklo_ps(t0, t2);
    __m128 t13lo = _mm_unpacklo_ps(t1, t3);
    __m128 t02hi = _mm_unpackhi_ps(t0, t2);
    __m128 t13hi = _mm_unpackhi_ps(t1, t3);
    a = _mm_unpacklo_ps(t02lo, t13lo);
    b = _mm_unpackhi_ps(t02lo, t13lo);
    c = _mm_unpacklo_ps(t02hi, t13hi);
    d = _mm_unpackhi_ps(t02hi, t13hi);
}

inline void _mm_interleave_epi16(__m128i& v_r0, __m128i& v_r1, __m128i& v_g0, __m128i& v_g1)
{
    __m128i v_mask = _mm_set1_epi32(0x0000ffff);

    __m128i layer3_chunk0 = _mm_packus_epi32(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer3_chunk2 = _mm_packus_epi32(_mm_srli_epi32(v_r0, 16), _mm_srli_epi32(v_r1, 16));
    __m128i layer3_chunk1 = _mm_packus_epi32(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi32(_mm_srli_epi32(v_g0, 16), _mm_srli_epi32(v_g1, 16));

    __m128i layer2_chunk0 = _mm_packus_epi32(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk2 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk0, 16), _mm_srli_epi32(layer3_chunk1, 16));
    __m128i layer2_chunk1 = _mm_packus_epi32(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk2, 16), _mm_srli_epi32(layer3_chunk3, 16));

    __m128i layer1_chunk0 = _mm_packus_epi32(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk2 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk0, 16), _mm_srli_epi32(layer2_chunk1, 16));
    __m128i layer1_chunk1 = _mm_packus_epi32(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk2, 16), _mm_srli_epi32(layer2_chunk3, 16));

    v_r0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk0, 16), _mm_srli_epi32(layer1_chunk1, 16));
    v_r1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_g1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk2, 16), _mm_srli_epi32(layer1_chunk3, 16));
}
#endif
