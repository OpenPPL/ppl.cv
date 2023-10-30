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
class RGB2LAB : public ::testing::TestWithParam<std::tuple<Size, float, int32_t>> {
public:
    using RGB2LABParam = std::tuple<Size, float, int32_t>;
    RGB2LAB()
    {
    }

    ~RGB2LAB()
    {
    }

    void RGB2LABAapply(const RGB2LABParam &param)
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
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2Lab);

            ppl::cv::arm::RGB2LAB<T>(
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
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2Lab);

            ppl::cv::arm::BGR2LAB<T>(
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

    void LAB2RGBAapply(const RGB2LABParam &param)
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
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_Lab2RGB);

            ppl::cv::arm::LAB2RGB<T>(
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
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_Lab2BGR);

            ppl::cv::arm::LAB2BGR<T>(
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

#define R1(name, t, ic, oc, diff, mode)    \
    using name = RGB2LAB<t, ic, oc>;     \
    TEST_P(name, abc)                      \
    {                                      \
        this->RGB2LABAapply(GetParam()); \
    }                                      \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));

R1(UT_RGB2LAB_uint8_t_aarch64, uint8_t, c3, c3, 3.01, 1)
R1(UT_BGR2LAB_uint8_t_aarch64, uint8_t, c3, c3, 3.01, 2)

#define R2(name, t, ic, oc, diff, mode)    \
    using name = RGB2LAB<t, ic, oc>;     \
    TEST_P(name, abc)                      \
    {                                      \
        this->LAB2RGBAapply(GetParam()); \
    }                                      \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));

R2(UT_LAB2RGB_uint8_t_aarch64, uint8_t, c3, c3, 3.01, 1)
R2(UT_LAB2BGR_uint8_t_aarch64, uint8_t, c3, c3, 3.01, 2)
