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

#include<immintrin.h>
#include<stdio.h>

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

inline void _mm_interleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
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

