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
class YUV422Convert : public ::testing::TestWithParam<std::tuple<Size, float>> {
public:
    using YUV422ConvertParam = std::tuple<Size, float>;
    YUV422Convert()
    {
    }

    ~YUV422Convert()
    {
    }

    void YUV2GRAYAapply(const YUV422ConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels * 3 / 2]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels * 3 / 2, 0, 255);

        cv::Mat src_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2GRAY_I420);

        ppl::cv::arm::YUV2GRAY<T>(
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

    void UYVY2GRAYAapply(const YUV422ConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2GRAY_UYVY);

        ppl::cv::arm::UYVY2GRAY<T>(
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

    void YUYV2GRAYAapply(const YUV422ConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2GRAY_YUYV);

        ppl::cv::arm::YUYV2GRAY<T>(
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

    void YUV2BGRAapply(const YUV422ConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels * 3 / 2]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels * 3 / 2, 0, 255);

        cv::Mat src_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_I420);

        ppl::cv::arm::YUV2BGR<T>(
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

    void UYVY2BGRAapply(const YUV422ConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_UYVY);

        ppl::cv::arm::UYVY2BGR<T>(
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

    void YUYV2BGRAapply(const YUV422ConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels * 3 / 2]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels * 3 / 2, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_YUYV);

        ppl::cv::arm::YUYV2BGR<T>(
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
};

constexpr int32_t c1 = 1;
constexpr int32_t c2 = 2;
constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

// YUV UYVY YUYV --> GRAY
#define R1(name, t, ic, oc, diff)          \
    using name = YUV422Convert<t, ic, oc>; \
    TEST_P(name, abc)                      \
    {                                      \
        this->YUV2GRAYAapply(GetParam());  \
    }                                      \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 240}, Size{720, 480}), ::testing::Values(diff)));

R1(UT_YUV2GRAY_uint8_t_aarch64, uint8_t, c1, c1, 1.01)

#define R2(name, t, ic, oc, diff)          \
    using name = YUV422Convert<t, ic, oc>; \
    TEST_P(name, abc)                      \
    {                                      \
        this->UYVY2GRAYAapply(GetParam()); \
    }                                      \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R2(UT_UYVY2GRAY_uint8_t_aarch64, uint8_t, c2, c1, 1.01)

#define R3(name, t, ic, oc, diff)          \
    using name = YUV422Convert<t, ic, oc>; \
    TEST_P(name, abc)                      \
    {                                      \
        this->YUYV2GRAYAapply(GetParam()); \
    }                                      \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R3(UT_YUYV2GRAY_uint8_t_aarch64, uint8_t, c2, c1, 1.01)

#define R4(name, t, ic, oc, diff)          \
    using name = YUV422Convert<t, ic, oc>; \
    TEST_P(name, abc)                      \
    {                                      \
        this->YUV2BGRAapply(GetParam());   \
    }                                      \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 240}, Size{720, 480}), ::testing::Values(diff)));

R4(UT_YUV2BGR_uint8_t_aarch64, uint8_t, c1, c3, 1.01)

#define R5(name, t, ic, oc, diff)          \
    using name = YUV422Convert<t, ic, oc>; \
    TEST_P(name, abc)                      \
    {                                      \
        this->UYVY2BGRAapply(GetParam());  \
    }                                      \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 240}, Size{720, 480}), ::testing::Values(diff)));

R5(UT_UYVY2BGR_uint8_t_aarch64, uint8_t, c2, c3, 1.01)

#define R6(name, t, ic, oc, diff)          \
    using name = YUV422Convert<t, ic, oc>; \
    TEST_P(name, abc)                      \
    {                                      \
        this->YUYV2BGRAapply(GetParam());  \
    }                                      \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 240}, Size{720, 480}), ::testing::Values(diff)));

R6(UT_YUYV2BGR_uint8_t_aarch64, uint8_t, c2, c3, 1.01)
