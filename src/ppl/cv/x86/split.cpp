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

#include "ppl/cv/x86/split.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "intrinutils.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>
#include <immintrin.h>

namespace ppl {
namespace cv {
namespace x86 {

template <>
::ppl::common::RetCode Split3Channels<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outDataChannel0,
    uint8_t* outDataChannel1,
    uint8_t* outDataChannel2)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        uint8_t* outData[3] = {outDataChannel0, outDataChannel1, outDataChannel2};
        return fma::splitAOS2SOA<uint8_t, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
    } else {
        for (int32_t h = 0; h < height; h++) {
            int32_t w = 0;
            const uint8_t* src_ptr = inData + h * inWidthStride;
            uint8_t* dst0_ptr = outDataChannel0 + h * outWidthStride;
            uint8_t* dst1_ptr = outDataChannel1 + h * outWidthStride;
            uint8_t* dst2_ptr = outDataChannel2 + h * outWidthStride;
            for (; w <= width - 16; w += 16) {
                __m128i vr, vb, vg;
                v_load_deinterleave(src_ptr, vr, vb, vg);
                _mm_storeu_si128((__m128i*)dst0_ptr, vr);
                _mm_storeu_si128((__m128i*)dst1_ptr, vb);
                _mm_storeu_si128((__m128i*)dst2_ptr, vg);
                src_ptr += 16 * 3;
                dst0_ptr += 16;
                dst1_ptr += 16;
                dst2_ptr += 16;
            }
            for (int32_t i = 0; w < width; w++, i++) {
                dst0_ptr[i] = src_ptr[i * 3 + 0];
                dst1_ptr[i] = src_ptr[i * 3 + 1];
                dst2_ptr[i] = src_ptr[i * 3 + 2];
            }
        }
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode Split3Channels<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outDataChannel0,
    float* outDataChannel1,
    float* outDataChannel2)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        float* outData[3] = {outDataChannel0, outDataChannel1, outDataChannel2};
        return fma::splitAOS2SOA<float, 3>(height, width, inWidthStride, inData, outWidthStride, outData);
    } else {
        for (int32_t h = 0; h < height; h++) {
            int32_t w = 0;
            const float* src_ptr = inData + h * inWidthStride;
            float* dst0_ptr = outDataChannel0 + h * outWidthStride;
            float* dst1_ptr = outDataChannel1 + h * outWidthStride;
            float* dst2_ptr = outDataChannel2 + h * outWidthStride;
            for (; w <= width - 4; w += 4) {
                __m128 vr, vb, vg;
                v_load_deinterleave(src_ptr, vr, vb, vg);
                _mm_storeu_ps(dst0_ptr, vr);
                _mm_storeu_ps(dst1_ptr, vb);
                _mm_storeu_ps(dst2_ptr, vg);
                src_ptr += 4 * 3;
                dst0_ptr += 4;
                dst1_ptr += 4;
                dst2_ptr += 4;
            }
            for (int32_t i = 0; w < width; w++, i++) {
                dst0_ptr[i] = src_ptr[i * 3 + 0];
                dst1_ptr[i] = src_ptr[i * 3 + 1];
                dst2_ptr[i] = src_ptr[i * 3 + 2];
            }
        }
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode Split4Channels<uint8_t>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    uint8_t* outDataChannel0,
    uint8_t* outDataChannel1,
    uint8_t* outDataChannel2,
    uint8_t* outDataChannel3)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2 || nullptr == outDataChannel3) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t* outData[4] = {outDataChannel0, outDataChannel1, outDataChannel2, outDataChannel3};
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::splitAOS2SOA<uint8_t, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
    } else {
        for (int32_t h = 0; h < height; h++) {
            int32_t w = 0;
            const uint8_t* src_ptr = inData + h * inWidthStride;
            uint8_t* dst0_ptr = outDataChannel0 + h * outWidthStride;
            uint8_t* dst1_ptr = outDataChannel1 + h * outWidthStride;
            uint8_t* dst2_ptr = outDataChannel2 + h * outWidthStride;
            uint8_t* dst3_ptr = outDataChannel3 + h * outWidthStride;
            for (; w <= width - 16; w += 16) {
                __m128i vr, vb, vg, va;
                v_load_deinterleave(src_ptr, vr, vb, vg, va);
                _mm_storeu_si128((__m128i*)dst0_ptr, vr);
                _mm_storeu_si128((__m128i*)dst1_ptr, vb);
                _mm_storeu_si128((__m128i*)dst2_ptr, vg);
                _mm_storeu_si128((__m128i*)dst3_ptr, va);
                src_ptr += 16 * 4;
                dst0_ptr += 16;
                dst1_ptr += 16;
                dst2_ptr += 16;
                dst3_ptr += 16;
            }
            for (int32_t i = 0; w < width; w++, i++) {
                dst0_ptr[i] = src_ptr[i * 4 + 0];
                dst1_ptr[i] = src_ptr[i * 4 + 1];
                dst2_ptr[i] = src_ptr[i * 4 + 2];
                dst3_ptr[i] = src_ptr[i * 4 + 3];
            }
        }
        return ppl::common::RC_SUCCESS;
    }
}

template <>
::ppl::common::RetCode Split4Channels<float>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inData,
    int32_t outWidthStride,
    float* outDataChannel0,
    float* outDataChannel1,
    float* outDataChannel2,
    float* outDataChannel3)
{
    if (nullptr == inData) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (nullptr == outDataChannel0 || nullptr == outDataChannel1 || nullptr == outDataChannel2 || nullptr == outDataChannel3) {
        return ppl::common::RC_INVALID_VALUE;
    }
    float* outData[4] = {outDataChannel0, outDataChannel1, outDataChannel2, outDataChannel3};
    if (ppl::common::CpuSupports(ppl::common::ISA_X86_FMA)) {
        return fma::splitAOS2SOA<float, 4>(height, width, inWidthStride, inData, outWidthStride, outData);
    } else {
        for (int32_t h = 0; h < height; h++) {
            int32_t w = 0;
            const float* src_ptr = inData + h * inWidthStride;
            float* dst0_ptr = outData[0] + h * outWidthStride;
            float* dst1_ptr = outData[1] + h * outWidthStride;
            float* dst2_ptr = outData[2] + h * outWidthStride;
            float* dst3_ptr = outData[3] + h * outWidthStride;
            for (; w <= width - 4; w += 4) {
                __m128 vr, vb, vg, va;
                v_load_deinterleave(src_ptr, vr, vb, vg, va);
                _mm_storeu_ps(dst0_ptr, vr);
                _mm_storeu_ps(dst1_ptr, vb);
                _mm_storeu_ps(dst2_ptr, vg);
                _mm_storeu_ps(dst3_ptr, va);
                src_ptr += 4 * 4;
                dst0_ptr += 4;
                dst1_ptr += 4;
                dst2_ptr += 4;
                dst3_ptr += 4;
            }
            for (int32_t i = 0; w < width; w++, i++) {
                dst0_ptr[i] = src_ptr[i * 4 + 0];
                dst1_ptr[i] = src_ptr[i * 4 + 1];
                dst2_ptr[i] = src_ptr[i * 4 + 2];
                dst3_ptr[i] = src_ptr[i * 4 + 3];
            }
        }
        return ppl::common::RC_SUCCESS;
    }
}

}
}
} // namespace ppl::cv::x86
