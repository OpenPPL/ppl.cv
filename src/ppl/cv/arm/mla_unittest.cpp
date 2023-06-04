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

#include "ppl/cv/arm/arithmetic.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<cv::Size>;
inline std::string convertToString(const Parameters& parameters)
{
    std::ostringstream formatted;

    cv::Size size = std::get<0>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

template <typename T, int channels>
class PplCvArmMlaTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmMlaTest()
    {
        const Parameters& parameters = GetParam();
        size = std::get<0>(parameters);
    }

    ~PplCvArmMlaTest() {}

    bool apply();

private:
    cv::Size size;
};

template <typename T, int channels>
bool PplCvArmMlaTest<T, channels>::apply()
{
    cv::Mat src0 = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat src1 = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat cv_dst(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    cv::Mat temp0(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat temp1(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    dst.copyTo(temp0);
    cv::multiply(src0, src1, temp1);
    cv::add(temp0, temp1, cv_dst);

    ppl::cv::arm::Mla<T, channels>(src0.rows,
                                   src0.cols,
                                   src0.step / sizeof(T),
                                   (T*)src0.data,
                                   src1.step / sizeof(T),
                                   (T*)src1.data,
                                   dst.step / sizeof(T),
                                   (T*)dst.data);

    float epsilon;
    if (sizeof(T) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    bool identity = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(T, channels)                                                                                     \
    using PplCvArmMlaTest_##T##_##channels = PplCvArmMlaTest<T, channels>;                                        \
    TEST_P(PplCvArmMlaTest_##T##_##channels, Standard)                                                            \
    {                                                                                                             \
        bool identity = this->apply();                                                                            \
        EXPECT_TRUE(identity);                                                                                    \
    }                                                                                                             \
                                                                                                                  \
    INSTANTIATE_TEST_CASE_P(IsEqual,                                                                              \
                            PplCvArmMlaTest_##T##_##channels,                                                     \
                            ::testing::Values(cv::Size{321, 240},                                                 \
                                              cv::Size{642, 480},                                                 \
                                              cv::Size{1283, 720},                                                \
                                              cv::Size{1934, 1080},                                               \
                                              cv::Size{320, 240},                                                 \
                                              cv::Size{640, 480},                                                 \
                                              cv::Size{1280, 720},                                                \
                                              cv::Size{1920, 1080}),                                              \
                            [](const testing::TestParamInfo<PplCvArmMlaTest_##T##_##channels::ParamType>& info) { \
                                return convertToString(info.param);                                               \
                            });

UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
