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

#include "internal_fma.hpp"
#include "ppl/cv/x86/avx/intrinutils_avx.hpp"
#include "ppl/common/sys.h"

#include <stdint.h>
#include <immintrin.h>

#include <vector>
#include <algorithm>
#include <cstring>

namespace ppl {
namespace cv {
namespace x86 {
namespace fma {

::ppl::common::RetCode BGR2GRAY(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* in,
    int32_t outWidthStride,
    float* out,
    bool reverse_channel)
{
    float r_coeff = 0.299f;
    float g_coeff = 0.587f;
    float b_coeff = 0.114f;

    if (reverse_channel) {
        std::swap(r_coeff, b_coeff);
    }
    __m256 v_cb = _mm256_set1_ps(b_coeff);
    __m256 v_cg = _mm256_set1_ps(g_coeff);
    __m256 v_cr = _mm256_set1_ps(r_coeff);

    for (int32_t h = 0; h < height; ++h) {
        const float* base_in = in + h * inWidthStride;
        float* base_out      = out + h * outWidthStride;
        int32_t w            = 0;
        for (; w <= width - 16; w += 16) {
            __m256 v_gray0, vr0, vb0, vg0;
            __m256 v_gray1, vr1, vb1, vg1;
            _mm256_deinterleave_ps(base_in + w * 3, vb0, vg0, vr0);
            _mm256_deinterleave_ps(base_in + w * 3 + 24, vb1, vg1, vr1);
            v_gray0 = _mm256_mul_ps(vr0, v_cr);
            v_gray0 = _mm256_fmadd_ps(vg0, v_cg, v_gray0);
            v_gray0 = _mm256_fmadd_ps(vb0, v_cb, v_gray0);
            v_gray1 = _mm256_mul_ps(vr1, v_cr);
            v_gray1 = _mm256_fmadd_ps(vg1, v_cg, v_gray1);
            v_gray1 = _mm256_fmadd_ps(vb1, v_cb, v_gray1);
            _mm256_storeu_ps(base_out + w, v_gray0);
            _mm256_storeu_ps(base_out + w + 8, v_gray1);
        }
        for (; w < width; w++) {
            base_out[w] = base_in[w * 3] * b_coeff + base_in[w * 3 + 1] * g_coeff + base_in[w * 3 + 2] * r_coeff;
        }
    }
    return ppl::common::RC_SUCCESS;
}

}
}
}
} // namespace ppl::cv::x86::fma
