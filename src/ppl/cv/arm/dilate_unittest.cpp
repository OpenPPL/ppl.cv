// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for subitional information
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

#include "ppl/cv/arm/dilate.h"
#include "morph.hpp"
#include "ppl/cv/arm/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include <iostream>

template <typename T, int32_t val>
static void randomRangeData(T *data, const size_t num, int32_t maxNum = 255)
{
    size_t tmp;

    for (size_t i = 0; i < num; i++) {
        tmp     = rand() % maxNum;
        data[i] = (T)((float)tmp / (float)val);
    }
}

template <typename T, int32_t nc>
class Dilate : public ::testing::TestWithParam<std::tuple<int32_t, int32_t, int32_t, int32_t, float>> {
public:
    using DilateParam = std::tuple<int32_t, int32_t, int32_t, int32_t, float>;
    Dilate()
    {
    }

    ~Dilate()
    {
    }

    void apply(const DilateParam &param)
    {
        int32_t width       = std::get<0>(param);
        int32_t height      = std::get<1>(param);
        int32_t kernel_size = std::get<2>(param);
        int32_t borderType  = std::get<3>(param);
        float diff      = std::get<4>(param);

        std::unique_ptr<T[]> src(new T[width * height * nc]);
        std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
        std::unique_ptr<T[]> dst(new T[width * height * nc]);
        std::unique_ptr<uchar[]> kernel(new uchar[kernel_size * kernel_size]);
        ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
        ppl::cv::debug::randomFill<T>(dst.get(), width * height * nc, 1, 1);
        ppl::cv::debug::randomFill<T>(dst_ref.get(), width * height * nc, 1, 1);
        ppl::cv::debug::randomFill<uchar>(kernel.get(), kernel_size * kernel_size, 1, 1);
        memcpy(dst.get(), dst_ref.get(), width * height * nc * sizeof(T));
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);
        cv::Mat kernel_opencv(kernel_size, kernel_size, CV_8U, kernel.get());

        double border_value;
        ppl::cv::debug::randomFill<double>(&border_value, 1, 0, 255);
        cv::Scalar borderValue = {border_value, border_value, border_value, border_value};

        cv::dilate(src_opencv, dst_opencv, kernel_opencv, cv::Point(-1, -1), 1, borderType, borderValue);

        ppl::cv::arm::Dilate<T, nc>(
            height,
            width,
            width * nc,
            src.get(),
            kernel_size,
            kernel_size,
            kernel.get(),
            width * nc,
            dst.get(),
            (ppl::cv::BorderType)borderType,
            (T)border_value);

        checkResult<T, nc>(
            dst_ref.get(),
            dst.get(),
            height,
            width,
            width * nc,
            width * nc,
            diff);
    };
};

#define R1(name, t, c, diff)     \
    using name = Dilate<t, c>;   \
    TEST_P(name, abc)            \
    {                            \
        this->apply(GetParam()); \
    }                            \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254), ::testing::Values(10), ::testing::Values(3, 5), ::testing::Values(ppl::cv::BORDER_CONSTANT, ppl::cv::BORDER_REFLECT, ppl::cv::BORDER_REFLECT101, ppl::cv::BORDER_REPLICATE), ::testing::Values(diff)));

R1(Dilate_f32c1, float, 1, 1.01)
R1(Dilate_f32c3, float, 3, 1.01)
R1(Dilate_f32c4, float, 4, 1.01)

R1(Dilate_u8c1, uint8_t, 1, 1.01)
R1(Dilate_u8c3, uint8_t, 3, 1.01)
R1(Dilate_u8c4, uint8_t, 4, 1.01)
