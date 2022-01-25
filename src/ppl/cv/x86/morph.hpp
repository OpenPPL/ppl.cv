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

#ifndef __ST_HPC_PPL_CV_X86_MORPH_HPP_
#define __ST_HPC_PPL_CV_X86_MORPH_HPP_
#include <algorithm>
#include "ppl/cv/types.h"
#include <immintrin.h>

namespace ppl {
namespace cv {
namespace x86 {

struct ErodeVecOp {
    inline __m128i operator()(__m128i a, __m128i b) const
    {
        return _mm_min_epu8(a, b);
    }

    inline __m128 operator()(__m128 a, __m128 b) const
    {
        return _mm_min_ps(a, b);
    }

    inline uint8_t operator()(uint8_t a, uint8_t b) const
    {
        return a > b ? b : a;
    }

    inline float operator()(float a, float b) const
    {
        return a > b ? b : a;
    }
};

struct DilateVecOp {
    inline __m128i operator()(__m128i a, __m128i b) const
    {
        return _mm_max_epu8(a, b);
    }

    inline __m128 operator()(__m128 a, __m128 b) const
    {
        return _mm_max_ps(a, b);
    }

    inline uint8_t operator()(uint8_t a, uint8_t b) const
    {
        return a > b ? a : b;
    }

    inline float operator()(float a, float b) const
    {
        return a > b ? a : b;
    }
};

//support erode or dilate
template <class morphOp, int32_t nc, int32_t kernel_len>
void morph_u8(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const uint8_t *srcBase,
    int32_t dstStride,
    uint8_t *dstBase,
    BorderType border_type = BORDER_CONSTANT,
    uint8_t borderValue    = 0);
template <class morphOp, int32_t nc, int32_t kernel_len>
void morph_f32(
    const int32_t height,
    const int32_t width,
    int32_t srcStride,
    const float *srcBase,
    int32_t dstStride,
    float *dstBase,
    BorderType border_type = BORDER_CONSTANT,
    float borderValue      = 0);
}
}
} // namespace ppl::cv::x86
#endif //__ST_HPC_PPL_CV_X86_MORPH_HPP_
