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

#ifndef __INTRINUTILS_AVX_H__
#define __INTRINUTILS_AVX_H__

#include<immintrin.h>
#include<stdio.h>

static inline void _mm256_deinterleave_ps(const float *p, __m256 & v_r0, __m256 & v_g0, __m256 & v_b0)
{
//	__m128 *m = (__m128*) p;
	__m256 m03 = _mm256_loadu_ps(p);
	__m256 m14 = _mm256_loadu_ps(p+4);
	__m256 m25 = _mm256_loadu_ps(p+8);

	__m128 m3 = _mm_loadu_ps(p+12);
	__m128 m4 = _mm_loadu_ps(p+16);
	__m128 m5 = _mm_loadu_ps(p+20);
	m03 = _mm256_insertf128_ps(m03,m3,1);
	m14 = _mm256_insertf128_ps(m14,m4,1);
	m25 = _mm256_insertf128_ps(m25,m5,1);


	__m256 xy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2, 1, 3, 2)); // upper x's and y's
	__m256 yz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1, 0, 2, 1)); // lower y's and z's
	v_r0 = _mm256_shuffle_ps(m03, xy, _MM_SHUFFLE(2, 0, 3, 0));
	v_g0 = _mm256_shuffle_ps(yz, xy, _MM_SHUFFLE(3, 1, 2, 0));
	v_b0 = _mm256_shuffle_ps(yz, m25, _MM_SHUFFLE(3, 0, 3, 1));
}

static inline void _mm256_deinterleave_ps(const float *p, __m256 & v_r0, __m256 & v_g0, __m256 & v_b0, __m256 & v_a0)
{
	__m256 m04 = _mm256_loadu_ps(p);
	__m256 m15 = _mm256_loadu_ps(p+4);
	__m256 m26 = _mm256_loadu_ps(p+8);
	__m256 m37 = _mm256_loadu_ps(p+12);

	__m128 m4 = _mm_loadu_ps(p+16);
	__m128 m5 = _mm_loadu_ps(p+20);
	__m128 m6 = _mm_loadu_ps(p+24);
	__m128 m7 = _mm_loadu_ps(p+28);

	m04 = _mm256_insertf128_ps(m04,m4,1);
	m15 = _mm256_insertf128_ps(m15,m5,1);
	m26 = _mm256_insertf128_ps(m26,m6,1);
	m37 = _mm256_insertf128_ps(m37,m7,1);

	__m256 ma = _mm256_shuffle_ps(m04,m15,_MM_SHUFFLE(2,0,2,0));
	__m256 mb = _mm256_shuffle_ps(m04,m15,_MM_SHUFFLE(3,1,3,1));
	__m256 mc = _mm256_shuffle_ps(m26,m37,_MM_SHUFFLE(2,0,2,0));
	__m256 md = _mm256_shuffle_ps(m26,m37,_MM_SHUFFLE(3,1,3,1));


	v_r0 = _mm256_shuffle_ps(ma,mc,_MM_SHUFFLE(2,0,2,0));
	v_b0 = _mm256_shuffle_ps(ma,mc,_MM_SHUFFLE(3,1,3,1));
	v_g0 = _mm256_shuffle_ps(mb,md,_MM_SHUFFLE(2,0,2,0));
	v_a0 = _mm256_shuffle_ps(mb,md,_MM_SHUFFLE(3,1,3,1));
}
static inline void _mm256_interleave_ps(float *p, __m256 & x, __m256 & y, __m256 & z)
{

	__m128 *m = (__m128*) p;
	__m256 rxy = _mm256_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
	__m256 ryz = _mm256_shuffle_ps(y, z, _MM_SHUFFLE(3, 1, 3, 1));
	__m256 rzx = _mm256_shuffle_ps(z, x, _MM_SHUFFLE(3, 1, 2, 0));

	__m256 r03 = _mm256_shuffle_ps(rxy, rzx, _MM_SHUFFLE(2, 0, 2, 0));
	__m256 r14 = _mm256_shuffle_ps(ryz, rxy, _MM_SHUFFLE(3, 1, 2, 0));
	__m256 r25 = _mm256_shuffle_ps(rzx, ryz, _MM_SHUFFLE(3, 1, 3, 1));

	m[0] = *((__m128 *)&r03);
	m[1] = *((__m128 *)&r14);
	m[2] = *((__m128 *)&r25);
	m[3] = _mm256_extractf128_ps(r03, 1);
	m[4] = _mm256_extractf128_ps(r14, 1);
	m[5] = _mm256_extractf128_ps(r25, 1);

}
static inline void _mm256_interleave1_ps(float *p, __m256 & x, __m256 & y, __m256 & z)
{

	//__m128 *m = (__m128*) p;
	__m256 rxy = _mm256_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
	__m256 ryz = _mm256_shuffle_ps(y, z, _MM_SHUFFLE(3, 1, 3, 1));
	__m256 rzx = _mm256_shuffle_ps(z, x, _MM_SHUFFLE(3, 1, 2, 0));

	__m256 r03 = _mm256_shuffle_ps(rxy, rzx, _MM_SHUFFLE(2, 0, 2, 0));
	__m256 r14 = _mm256_shuffle_ps(ryz, rxy, _MM_SHUFFLE(3, 1, 2, 0));
	__m256 r25 = _mm256_shuffle_ps(rzx, ryz, _MM_SHUFFLE(3, 1, 3, 1));

	__m128 m0 = _mm256_castps256_ps128(r03);
	__m128 m1= _mm256_castps256_ps128(r14);
	__m128 m2= _mm256_castps256_ps128(r25);
	__m128 m3= _mm256_extractf128_ps(r03, 1);
	__m128 m4= _mm256_extractf128_ps(r14, 1);
	__m128 m5= _mm256_extractf128_ps(r25, 1);
	_mm_storeu_ps(p,m0);
	_mm_storeu_ps(p+4,m1);
	_mm_storeu_ps(p+8,m2);
	_mm_storeu_ps(p+12,m3);
	_mm_storeu_ps(p+16,m4);
	_mm_storeu_ps(p+20,m5);
}
static inline void _mm256_interleave1_ps(float *p, __m256 & x, __m256 & y, __m256 & z, __m256 & h)
{

	//__m128 *m = (__m128*) p;


	__m256 ma = _mm256_shuffle_ps(x,y,_MM_SHUFFLE(2,0,2,0));
	__m256 mb = _mm256_shuffle_ps(x,y,_MM_SHUFFLE(3,1,3,1));
	__m256 mc = _mm256_shuffle_ps(z,h,_MM_SHUFFLE(2,0,2,0));
	__m256 md = _mm256_shuffle_ps(z,h,_MM_SHUFFLE(3,1,3,1));


	__m256 m04 = _mm256_shuffle_ps(ma,mc,_MM_SHUFFLE(2,0,2,0));
	__m256 m26 = _mm256_shuffle_ps(ma,mc,_MM_SHUFFLE(3,1,3,1));
	__m256 m15 = _mm256_shuffle_ps(mb,md,_MM_SHUFFLE(2,0,2,0));
	__m256 m37 = _mm256_shuffle_ps(mb,md,_MM_SHUFFLE(3,1,3,1));

	__m128 m0 = _mm256_castps256_ps128(m04);
	__m128 m1 = _mm256_castps256_ps128(m15);
	__m128 m2 = _mm256_castps256_ps128(m26);
	__m128 m3 = _mm256_castps256_ps128(m37);
	__m128 m4 = _mm256_extractf128_ps(m04, 1);
	__m128 m5 = _mm256_extractf128_ps(m15, 1);
	__m128 m6 = _mm256_extractf128_ps(m26, 1);
	__m128 m7 = _mm256_extractf128_ps(m37, 1);
	_mm_storeu_ps(p,m0);
	_mm_storeu_ps(p+4,m1);
	_mm_storeu_ps(p+8,m2);
	_mm_storeu_ps(p+12,m3);
	_mm_storeu_ps(p+16,m4);
	_mm_storeu_ps(p+20,m5);
	_mm_storeu_ps(p+24,m6);
	_mm_storeu_ps(p+28,m7);
}

#endif

