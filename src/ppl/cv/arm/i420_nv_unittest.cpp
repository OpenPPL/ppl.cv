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

#include "ppl/common/retcode.h"
#include "ppl/common/sys.h"
#include "ppl/cv/arm/cvtcolor.h"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "ppl/cv/debug.h"
#include "ppl/cv/arm/test.h"

namespace ppl {
namespace cv {
namespace arm_test {

::ppl::common::RetCode I4202NV21(int32_t height,
                                 int32_t width,
                                 int32_t inStrideY,
                                 const uint8_t* inDataY,
                                 int32_t inStrideU,
                                 const uint8_t* inDataU,
                                 int32_t inStrideV,
                                 const uint8_t* inDataV,
                                 int32_t outStrideY,
                                 uint8_t* outDataY,
                                 int32_t outStrideVU,
                                 uint8_t* outDataVU)
{
    assert((height % 2) == 0);
    assert((width % 2) == 0);
    assert(height != 0 && width != 0);
    assert(inStrideY != 0 && inStrideV != 0 && inStrideU != 0);
    assert(outStrideY != 0 && outStrideVU != 0);
    assert(inDataY != NULL && inDataU != NULL && inDataV != NULL);
    assert(outDataY != NULL && outDataVU != NULL);

    // memcpy y plane
    for (int32_t i = 0; i < height; ++i) {
        memcpy(outDataY + i * outStrideY, inDataY + i * inStrideY, sizeof(uint8_t) * width);
    }

    // memcpy u,v plane
    for (int32_t i = 0; i < height / 2; ++i) {
        for (int32_t j = 0; j < width / 2; ++j) {
            outDataVU[i * outStrideVU + j * 2] = inDataV[i * inStrideV + j];
            outDataVU[i * outStrideVU + j * 2 + 1] = inDataU[i * inStrideU + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode I4202NV12(int32_t height,
                                 int32_t width,
                                 int32_t inStrideY,
                                 const uint8_t* inDataY,
                                 int32_t inStrideU,
                                 const uint8_t* inDataU,
                                 int32_t inStrideV,
                                 const uint8_t* inDataV,
                                 int32_t outStrideY,
                                 uint8_t* outDataY,
                                 int32_t outStrideUV,
                                 uint8_t* outDataUV)
{
    assert((height % 2) == 0);
    assert((width % 2) == 0);
    assert(height != 0 && width != 0);
    assert(inStrideY != 0 && inStrideV != 0 && inStrideU != 0);
    assert(outStrideY != 0 && outStrideUV != 0);
    assert(inDataY != NULL && inDataU != NULL && inDataV != NULL);
    assert(outDataY != NULL && outDataUV != NULL);

    // memcpy y plane
    for (int32_t i = 0; i < height; ++i) {
        memcpy(outDataY + i * outStrideY, inDataY + i * inStrideY, sizeof(uint8_t) * width);
    }

    // memcpy u,v plane
    for (int32_t i = 0; i < height / 2; ++i) {
        for (int32_t j = 0; j < width / 2; ++j) {
            outDataUV[i * outStrideUV + j * 2] = inDataU[i * inStrideU + j];
            outDataUV[i * outStrideUV + j * 2 + 1] = inDataV[i * inStrideV + j];
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode NV212I420(int32_t height,
                                 int32_t width,
                                 int32_t inStrideY,
                                 const uint8_t* inDataY,
                                 int32_t inStrideVU,
                                 const uint8_t* inDataVU,
                                 int32_t outStrideY,
                                 uint8_t* outDataY,
                                 int32_t outStrideU,
                                 uint8_t* outDataU,
                                 int32_t outStrideV,
                                 uint8_t* outDataV)
{
    assert((height % 2) == 0);
    assert((width % 2) == 0);
    assert(height != 0 && width != 0);
    assert(outStrideY != 0 && outStrideV != 0 && outStrideU != 0);
    assert(inStrideY != 0 && inStrideVU != 0);
    assert(outDataY != NULL && outDataU != NULL && outDataV != NULL);
    assert(inDataY != NULL && inDataVU != NULL);

    // memcpy y plane
    for (int32_t i = 0; i < height; ++i) {
        memcpy(outDataY + i * outStrideY, inDataY + i * inStrideY, sizeof(uint8_t) * width);
    }

    // memcpy u,v plane
    for (int32_t i = 0; i < height / 2; ++i) {
        for (int32_t j = 0; j < width / 2; ++j) {
            outDataV[i * outStrideV + j] = inDataVU[i * inStrideVU + 2 * j];
            outDataU[i * outStrideU + j] = inDataVU[i * inStrideVU + 2 * j + 1];
        }
    }
    return ppl::common::RC_SUCCESS;
}

::ppl::common::RetCode NV122I420(int32_t height,
                                 int32_t width,
                                 int32_t inStrideY,
                                 const uint8_t* inDataY,
                                 int32_t inStrideUV,
                                 const uint8_t* inDataUV,
                                 int32_t outStrideY,
                                 uint8_t* outDataY,
                                 int32_t outStrideU,
                                 uint8_t* outDataU,
                                 int32_t outStrideV,
                                 uint8_t* outDataV)
{
    assert((height % 2) == 0);
    assert((width % 2) == 0);
    assert(height != 0 && width != 0);
    assert(outStrideY != 0 && outStrideV != 0 && outStrideU != 0);
    assert(inStrideY != 0 && inStrideUV != 0);
    assert(outDataY != NULL && outDataU != NULL && outDataV != NULL);
    assert(inDataY != NULL && inDataUV != NULL);

    // memcpy y plane
    for (int32_t i = 0; i < height; ++i) {
        memcpy(outDataY + i * outStrideY, inDataY + i * inStrideY, sizeof(uint8_t) * width);
    }

    // memcpy u,v plane
    for (int32_t i = 0; i < height / 2; ++i) {
        for (int32_t j = 0; j < width / 2; ++j) {
            outDataU[i * outStrideU + j] = inDataUV[i * inStrideUV + 2 * j];
            outDataV[i * outStrideV + j] = inDataUV[i * inStrideUV + 2 * j + 1];
        }
    }
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::arm_test

template <typename T, int32_t input_channels, int32_t output_channels>
class NV2I420 : public ::testing::TestWithParam<std::tuple<Size, float, int32_t>> {
public:
    using NV2I420Param = std::tuple<Size, float, int32_t>;
    NV2I420()
    {
    }

    ~NV2I420()
    {
    }

    void NV2I420Aapply(const NV2I420Param& param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);
        int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * 3 / 2]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * 3 / 2, 0, 255);

        if (1 == mode) {
            ppl::cv::arm_test::NV122I420(size.height,
                                         size.width,
                                         size.width,
                                         src.get(),
                                         size.width,
                                         src.get() + size.height * size.width,
                                         size.width,
                                         dst_ref.get(),
                                         size.width / 2,
                                         dst_ref.get() + size.height * size.width,
                                         size.width / 2,
                                         dst_ref.get() + size.height * size.width + size.height * size.width / 4);

            ppl::cv::arm::NV122I420<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
            checkResult<T, output_channels>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        } else {
            ppl::cv::arm_test::NV212I420(size.height,
                                         size.width,
                                         size.width,
                                         src.get(),
                                         size.width,
                                         src.get() + size.height * size.width,
                                         size.width,
                                         dst_ref.get(),
                                         size.width / 2,
                                         dst_ref.get() + size.height * size.width,
                                         size.width / 2,
                                         dst_ref.get() + size.height * size.width + size.height * size.width / 4);

            ppl::cv::arm::NV212I420<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, output_channels>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }
    }

    void I4202NVAapply(const NV2I420Param& param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);
        int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * 3 / 2]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * 3 / 2, 0, 255);

        if (1 == mode) {
            ppl::cv::arm_test::I4202NV21(size.height,
                                         size.width,
                                         size.width,
                                         src.get(),
                                         size.width / 4,
                                         src.get() + size.height * size.width,
                                         size.width / 4,
                                         src.get() + size.height * size.width + size.height * size.width / 4,
                                         size.width,
                                         dst_ref.get(),
                                         size.width / 2,
                                         dst_ref.get() + size.height * size.width);

            ppl::cv::arm::I4202NV21<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
            checkResult<T, output_channels>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        } else {
            ppl::cv::arm_test::I4202NV12(size.height,
                                         size.width,
                                         size.width,
                                         src.get(),
                                         size.width / 4,
                                         src.get() + size.height * size.width,
                                         size.width / 4,
                                         src.get() + size.height * size.width + size.height * size.width / 4,
                                         size.width,
                                         dst_ref.get(),
                                         size.width / 2,
                                         dst_ref.get() + size.height * size.width);

            ppl::cv::arm::I4202NV12<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());

            checkResult<T, output_channels>(
                dst_ref.get(),
                dst.get(),
                size.height,
                size.width,
                size.width * output_channels,
                size.width * output_channels,
                diff);
        }
    }
};

constexpr int32_t c1 = 1;
constexpr int32_t c2 = 2;
constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

#define R1(name, t, ic, oc, diff, mode)  \
    using name = NV2I420<t, ic, oc>;     \
    TEST_P(name, abc)                    \
    {                                    \
        this->NV2I420Aapply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));

R1(UT_NV122I420_uint8_t_aarch64, uint8_t, c1, c1, 1.01, 1)

R1(UT_NV212I420_uint8_t_aarch64, uint8_t, c1, c1, 1.01, 2)

#define R2(name, t, ic, oc, diff, mode)  \
    using name = NV2I420<t, ic, oc>;     \
    TEST_P(name, abc)                    \
    {                                    \
        this->I4202NVAapply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));

R2(UT_I4202NV21_uint8_t_aarch64, uint8_t, c1, c1, 1.01, 1)

R2(UT_I4202NV12_uint8_t_aarch64, uint8_t, c1, c1, 1.01, 2)
