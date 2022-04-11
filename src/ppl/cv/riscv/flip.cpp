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

#include "ppl/cv/riscv/flip.h"
#include "ppl/cv/riscv/typetraits.h"

#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"

#include <string.h>
#include <cmath>
#include <limits.h>
#include <memory>

namespace ppl {
namespace cv {
namespace riscv {

template <typename eT, int32_t channels>
struct FlipHorizontalChannelsUnrollFunc {
    inline static void f(const eT *src, eT *dst)
    {
        *dst = *src;
        FlipHorizontalChannelsUnrollFunc<eT, channels - 1>::f(src + 1, dst + 1);
    }
};

template <typename eT>
struct FlipHorizontalChannelsUnrollFunc<eT, 0> {
    inline static void f(const eT *src, eT *dst) {}
};

template <typename eT, int32_t num_unroll, int32_t channels>
struct FlipHorizontalWidthUnrollFunc {
    inline static void f(const eT *src, eT *dst)
    {
        FlipHorizontalChannelsUnrollFunc<eT, channels>::f(src, dst);
        FlipHorizontalWidthUnrollFunc<eT, num_unroll - 1, channels>::f(src + channels, dst - channels);
    }
};

template <typename eT, int32_t channels>
struct FlipHorizontalWidthUnrollFunc<eT, 0, channels> {
    inline static void f(const eT *src, eT *dst) {}
};

template <typename eT, bool flip_horizontal, int32_t channels>
struct FlipGenericKernelFunc {
    static constexpr int32_t num_unroll = 16 / sizeof(eT);
    inline static void flip_kernel(
        int32_t width,
        const eT *src,
        eT *dst)
    {
        if (flip_horizontal) {
            int32_t out_x = 0, out_xc = (width - 1) * channels, in_xc = 0;
            for (; out_x <= width - num_unroll; out_x += num_unroll, out_xc -= channels * num_unroll, in_xc += channels * num_unroll) {
                FlipHorizontalWidthUnrollFunc<eT, num_unroll, channels>::f(src + in_xc, dst + out_xc);
            }
            for (; out_x < width; ++out_x, out_xc -= channels, in_xc += channels) {
                FlipHorizontalChannelsUnrollFunc<eT, channels>::f(src + in_xc, dst + out_xc);
            }
        } else {
            memcpy(dst, src, width * channels * sizeof(eT));
        }
    }
}; // struct FlipGenericFunc

template <typename eT, bool flip_vertical, bool flip_horizontal, int32_t channels>
struct FlipGenericFunc {
    static ::ppl::common::RetCode flip(
        int32_t height,
        int32_t width,
        int32_t inWidthStride,
        const eT *inData,
        int32_t outWidthStride,
        eT *outData)
    {
        if (nullptr == inData) {
            return ppl::common::RC_INVALID_VALUE;
        }
        if (nullptr == outData) {
            return ppl::common::RC_INVALID_VALUE;
        }

        auto in = inData;
        auto out = flip_vertical ? (outData + (height - 1) * inWidthStride) : outData;
        for (int32_t i = 0; i < height; ++i) {
            FlipGenericKernelFunc<eT, flip_horizontal, channels>::flip_kernel(width, in, out);
            in += inWidthStride;
            out += flip_vertical ? (-inWidthStride) : inWidthStride;
        }
        return ppl::common::RC_SUCCESS;
    }
}; // struct FlipGenericFunc

template <>
::ppl::common::RetCode Flip<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t flipCode)
{
    if (flipCode == 0) {
        return FlipGenericFunc<float, true, false, 1>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else if (flipCode > 0) {
        return FlipGenericFunc<float, false, true, 1>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else { //! flipCode < 0
        return FlipGenericFunc<float, true, true, 1>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<float, 2>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t flipCode)
{
    if (flipCode == 0) {
        return FlipGenericFunc<float, true, false, 2>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else if (flipCode > 0) {
        return FlipGenericFunc<float, false, true, 2>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else { //! flipCode < 0
        return FlipGenericFunc<float, true, true, 2>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t flipCode)
{
    if (flipCode == 0) {
        return FlipGenericFunc<float, true, false, 3>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else if (flipCode > 0) {
        return FlipGenericFunc<float, false, true, 3>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else { //! flipCode < 0
        return FlipGenericFunc<float, true, true, 3>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float *inData,
    int32_t outWidthStride,
    float *outData,
    int32_t flipCode)
{
    if (flipCode == 0) {
        return FlipGenericFunc<float, true, false, 4>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else if (flipCode > 0) {
        return FlipGenericFunc<float, false, true, 4>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else { //! flipCode < 0
        return FlipGenericFunc<float, true, true, 4>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    int32_t flipCode)
{
    if (flipCode == 0) {
        return FlipGenericFunc<uint8_t, true, false, 1>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else if (flipCode > 0) {
        return FlipGenericFunc<uint8_t, false, true, 1>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else { //! flipCode < 0
        return FlipGenericFunc<uint8_t, true, true, 1>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<uint8_t, 2>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    int32_t flipCode)
{
    if (flipCode == 0) {
        return FlipGenericFunc<uint8_t, true, false, 2>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else if (flipCode > 0) {
        return FlipGenericFunc<uint8_t, false, true, 2>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else { //! flipCode < 0
        return FlipGenericFunc<uint8_t, true, true, 2>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    int32_t flipCode)
{
    if (flipCode == 0) {
        return FlipGenericFunc<uint8_t, true, false, 3>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else if (flipCode > 0) {
        return FlipGenericFunc<uint8_t, false, true, 3>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else { //! flipCode < 0
        return FlipGenericFunc<uint8_t, true, true, 3>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

template <>
::ppl::common::RetCode Flip<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t *inData,
    int32_t outWidthStride,
    uint8_t *outData,
    int32_t flipCode)
{
    if (flipCode == 0) {
        return FlipGenericFunc<uint8_t, true, false, 4>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else if (flipCode > 0) {
        return FlipGenericFunc<uint8_t, false, true, 4>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    } else { //! flipCode < 0
        return FlipGenericFunc<uint8_t, true, true, 4>::flip(height, width, inWidthStride, inData, outWidthStride, outData);
    }
}

}
}
} // namespace ppl::cv::riscv
