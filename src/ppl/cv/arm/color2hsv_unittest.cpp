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

#include "ppl/cv/arm/cvtcolor.h"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "ppl/cv/debug.h"
#include "ppl/cv/arm/test.h"

template <typename T, int32_t input_channels, int32_t output_channels>
class RGB2HSV : public ::testing::TestWithParam<std::tuple<Size, float, int32_t>> {
public:
    using RGB2HSVParam = std::tuple<Size, float, int32_t>;
    RGB2HSV()
    {
    }

    ~RGB2HSV()
    {
    }

    void RGB2HSVAapply(const RGB2HSVParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);
        int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        if (1 == mode) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2HSV);

            ppl::cv::arm::RGB2HSV<T>(
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
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2HSV);

            ppl::cv::arm::BGR2HSV<T>(
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

    void HSV2RGBAapply(const RGB2HSVParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);
        int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        if (1 == mode) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_HSV2RGB);

            ppl::cv::arm::HSV2RGB<T>(
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
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_HSV2BGR);

            ppl::cv::arm::HSV2BGR<T>(
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
    using name = RGB2HSV<t, ic, oc>;     \
    TEST_P(name, abc)                    \
    {                                    \
        this->RGB2HSVAapply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));

R1(UT_RGB2HSV_uint8_t_aarch64, uint8_t, c3, c3, 1.01, 1)
R1(UT_RGB2HSV_float32_t_aarch64, float32_t, c3, c3, 1e-4, 1)

R1(UT_BGR2HSV_uint8_t_aarch64, uint8_t, c3, c3, 1.01, 2)
R1(UT_BGR2HSV_float32_t_aarch64, float32_t, c3, c3, 1e-4, 2)

#define R2(name, t, ic, oc, diff, mode)  \
    using name = RGB2HSV<t, ic, oc>;     \
    TEST_P(name, abc)                    \
    {                                    \
        this->HSV2RGBAapply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));

R2(UT_HSV2RGB_uint8_t_aarch64, uint8_t, c3, c3, 1.01, 1)
R2(UT_HSV2RGB_float32_t_aarch64, float32_t, c3, c3, 1e-2, 1)

R2(UT_HSV2BGR_uint8_t_aarch64, uint8_t, c3, c3, 1.01, 2)
R2(UT_HSV2BGR_float32_t_aarch64, float32_t, c3, c3, 1e-2, 2)
