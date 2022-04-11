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

#include "ppl/cv/riscv/cvtcolor.h"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "ppl/cv/debug.h"
#include "ppl/cv/riscv/test.h"

template <typename T, int32_t input_channels, int32_t output_channels>
class BGR2BGRA : public ::testing::TestWithParam<std::tuple<Size, float>> {
public:
    using BGR2BGRAParam = std::tuple<Size, float>;
    BGR2BGRA()
    {
    }

    ~BGR2BGRA()
    {
    }

    void BGR2BGRAapply(const BGR2BGRAParam &param)
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

        ppl::cv::riscv::BGR2BGRA<T>(
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
    void BGRA2BGRapply(const BGR2BGRAParam &param)
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

        ppl::cv::riscv::BGRA2BGR<T>(
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

#define R1(name, t, ic, oc, diff)        \
    using name = BGR2BGRA<t, ic, oc>;    \
    TEST_P(name, abc)                    \
    {                                    \
        this->BGR2BGRAapply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R1(UT_BGR2BGRA_float_riscv, float, c3, c4, 1e-5)
R1(UT_BGR2BGRA_uint8_t_riscv, uint8_t, c3, c4, 1.01)

#define R2(name, t, ic, oc, diff)        \
    using name = BGR2BGRA<t, ic, oc>;    \
    TEST_P(name, abc)                    \
    {                                    \
        this->BGRA2BGRapply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff)));

R2(UT_BGRA2BGR_float_riscv, float, c4, c3, 1e-5)
R2(UT_BGRA2BGR_uint8_t_riscv, uint8_t, c4, c3, 1.01)
