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
class RGBInnerConvert : public ::testing::TestWithParam<std::tuple<Size, float>> {
public:
    using RGBInnerConvertParam = std::tuple<Size, float>;
    RGBInnerConvert()
    {
    }

    ~RGBInnerConvert()
    {
    }

    // RGB-> RGBA BGR BGRA
    void RGB2RGBAapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2RGBA);

        ppl::cv::arm::RGB2RGBA<T>(
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

    void RGB2BGRapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2BGR);

        ppl::cv::arm::RGB2BGR<T>(
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

    void RGB2BGRAapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2BGRA);

        ppl::cv::arm::RGB2BGRA<T>(
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
    // RGBA ---> RGB BGR BGRA
    void RGBA2RGBapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2RGB);

        ppl::cv::arm::RGBA2RGB<T>(
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

    void RGBA2BGRapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2BGR);

        ppl::cv::arm::RGBA2BGR<T>(
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

    void RGBA2BGRAapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2BGRA);

        ppl::cv::arm::RGBA2BGRA<T>(
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
    // BGR ---> RGB RGBA BGRA
    void BGR2RGBapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2RGB);

        ppl::cv::arm::BGR2RGB<T>(
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
    void BGR2RGBAapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2RGBA);

        ppl::cv::arm::BGR2RGBA<T>(
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
    void BGR2BGRAapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2BGRA);

        ppl::cv::arm::BGR2BGRA<T>(
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
    // BGRA ---> RGB RGBA BGR
    void BGRA2RGBapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2RGB);

        ppl::cv::arm::BGRA2RGB<T>(
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
    void BGRA2RGBAapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2RGBA);

        ppl::cv::arm::BGRA2RGBA<T>(
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
    void BGRA2BGRapply(const RGBInnerConvertParam &param)
    {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2BGR);

        ppl::cv::arm::BGRA2BGR<T>(
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

constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

// RGB --> RGBA BGR BGRA
#define R1(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->RGB2RGBAapply(GetParam());     \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R1(UT_RGB2RGBA_float_aarch64, float32_t, c3, c4, 1e-5)
R1(UT_RGB2RGBA_uint8_t_aarch64, uint8_t, c3, c4, 1.01)

#define R2(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->RGB2BGRapply(GetParam());      \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R2(UT_RGB2BGR_float_aarch64, float32_t, c3, c3, 1e-5)
R2(UT_RGB2BGR_uint8_t_aarch64, uint8_t, c3, c3, 1.01)

#define R3(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->RGB2BGRAapply(GetParam());     \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R3(UT_RGB2BGRA_float_aarch64, float32_t, c3, c4, 1e-5)
R3(UT_RGB2BGRA_uint8_t_aarch64, uint8_t, c3, c4, 1.01)

// RGBA --> RGB BGR BGRA
#define R4(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->RGBA2RGBapply(GetParam());     \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R4(UT_RGBA2RGB_float_aarch64, float32_t, c4, c3, 1e-5)
R4(UT_RGBA2RGB_uint8_t_aarch64, uint8_t, c4, c3, 1.01)

#define R5(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->RGBA2BGRapply(GetParam());     \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R5(UT_RGBA2BGR_float_aarch64, float32_t, c4, c3, 1e-5)
R5(UT_RGBA2BGR_uint8_t_aarch64, uint8_t, c4, c3, 1.01)

#define R6(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->RGBA2BGRAapply(GetParam());    \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R6(UT_RGBA2BGRA_float_aarch64, float32_t, c4, c4, 1e-5)
R6(UT_RGBA2BGRA_uint8_t_aarch64, uint8_t, c4, c4, 1.01)

// BGR --> RGB RGBA BGRA
#define R7(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->BGR2RGBapply(GetParam());      \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R7(UT_BGR2RGB_float_aarch64, float32_t, c3, c3, 1e-5)
R7(UT_BGR2RGB_uint8_t_aarch64, uint8_t, c3, c3, 1.01)

#define R8(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->BGR2RGBAapply(GetParam());     \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R8(UT_BGR2RGBA_float_aarch64, float32_t, c3, c4, 1e-5)
R8(UT_BGR2RGBA_uint8_t_aarch64, uint8_t, c3, c4, 1.01)

#define R9(name, t, ic, oc, diff)            \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->BGR2BGRAapply(GetParam());     \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R9(UT_BGR2BGRA_float_aarch64, float32_t, c3, c4, 1e-5)
R9(UT_BGR2BGRA_uint8_t_aarch64, uint8_t, c3, c4, 1.01)

// BGRA --> RGB RGBA BGR
#define R10(name, t, ic, oc, diff)           \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->BGRA2RGBapply(GetParam());     \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R10(UT_BGRA2RGB_float_aarch64, float32_t, c4, c3, 1e-5)
R10(UT_BGRA2RGB_uint8_t_aarch64, uint8_t, c4, c3, 1.01)

#define R11(name, t, ic, oc, diff)           \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->BGRA2RGBAapply(GetParam());    \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R11(UT_BGRA2RGBA_float_aarch64, float32_t, c4, c4, 1e-5)
R11(UT_BGRA2RGBA_uint8_t_aarch64, uint8_t, c4, c4, 1.01)

#define R12(name, t, ic, oc, diff)           \
    using name = RGBInnerConvert<t, ic, oc>; \
    TEST_P(name, abc)                        \
    {                                        \
        this->BGRA2BGRapply(GetParam());     \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R12(UT_BGRA2BGR_float_aarch64, float32_t, c4, c3, 1e-5)
R12(UT_BGRA2BGR_uint8_t_aarch64, uint8_t, c4, c3, 1.01)
