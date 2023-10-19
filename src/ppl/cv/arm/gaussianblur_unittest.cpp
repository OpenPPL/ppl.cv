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

#include "ppl/cv/arm/gaussianblur.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, int, ppl::cv::BorderType, cv::Size>;
inline std::string convertToStringGaussianBlur(const Parameters& parameters)
{
    std::ostringstream formatted;

    int ksize = std::get<0>(parameters);
    formatted << "Ksize" << ksize << "_";

    int int_sigma = std::get<1>(parameters);
    formatted << "Sigma" << int_sigma << "_";

    ppl::cv::BorderType border_type = std::get<2>(parameters);
    if (border_type == ppl::cv::BORDER_REPLICATE) {
        formatted << "BORDER_REPLICATE"
                  << "_";
    } else if (border_type == ppl::cv::BORDER_REFLECT) {
        formatted << "BORDER_REFLECT"
                  << "_";
    } else if (border_type == ppl::cv::BORDER_REFLECT_101) {
        formatted << "BORDER_REFLECT_101"
                  << "_";
    } else { // border_type == ppl::cv::BORDER_DEFAULT
        formatted << "BORDER_DEFAULT"
                  << "_";
    }

    cv::Size size = std::get<3>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

template <typename T, int channels>
class PplCvArmGaussianBlurTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmGaussianBlurTest()
    {
        const Parameters& parameters = GetParam();
        ksize = std::get<0>(parameters);
        sigma = std::get<1>(parameters) / 10.f;
        border_type = std::get<2>(parameters);
        size = std::get<3>(parameters);
    }

    ~PplCvArmGaussianBlurTest() {}

    bool apply();

private:
    int ksize;
    float sigma;
    ppl::cv::BorderType border_type;
    cv::Size size;
};

template <typename T, int channels>
bool PplCvArmGaussianBlurTest<T, channels>::apply()
{
    cv::Mat src;
    src = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat cv_dst(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
    if (border_type == ppl::cv::BORDER_REPLICATE) {
        cv_border = cv::BORDER_REPLICATE;
    } else if (border_type == ppl::cv::BORDER_REFLECT) {
        cv_border = cv::BORDER_REFLECT;
    } else if (border_type == ppl::cv::BORDER_REFLECT_101) {
        cv_border = cv::BORDER_REFLECT_101;
    } else {
    }
    cv::GaussianBlur(src, cv_dst, cv::Size(ksize, ksize), sigma, sigma, cv_border);

    ppl::cv::arm::GaussianBlur<T, channels>(src.rows,
                                            src.cols,
                                            src.step / sizeof(T),
                                            (T*)src.data,
                                            ksize,
                                            sigma,
                                            dst.step / sizeof(T),
                                            (T*)dst.data,
                                            border_type);

    float epsilon;
    if (sizeof(T) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    bool identity = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(T, channels)                                                                                   \
    using PplCvArmGaussianBlurTest_##T##_##channels = PplCvArmGaussianBlurTest<T, channels>;                    \
    TEST_P(PplCvArmGaussianBlurTest_##T##_##channels, Standard)                                                 \
    {                                                                                                           \
        bool identity = this->apply();                                                                          \
        EXPECT_TRUE(identity);                                                                                  \
    }                                                                                                           \
                                                                                                                \
    INSTANTIATE_TEST_CASE_P(                                                                                    \
        IsEqual,                                                                                                \
        PplCvArmGaussianBlurTest_##T##_##channels,                                                              \
        ::testing::Combine(                                                                                     \
            ::testing::Values(1, 3, 5, 13, 31),                                                                 \
            ::testing::Values(0, 1, 7, 43),                                                                     \
            ::testing::Values(ppl::cv::BORDER_REPLICATE, ppl::cv::BORDER_REFLECT, ppl::cv::BORDER_REFLECT_101), \
            ::testing::Values(cv::Size{321, 240},                                                               \
                              cv::Size{321, 245},                                                               \
                              cv::Size{647, 493},                                                               \
                              cv::Size{654, 486},                                                               \
                              cv::Size{1280, 720},                                                              \
                              cv::Size{1920, 1080})),                                                           \
        [](const testing::TestParamInfo<PplCvArmGaussianBlurTest_##T##_##channels::ParamType>& info) {          \
            return convertToStringGaussianBlur(info.param);                                                     \
        });

UNITTEST(uint8_t, 1)
UNITTEST(uint8_t, 3)
UNITTEST(uint8_t, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
