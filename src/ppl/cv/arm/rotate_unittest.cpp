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

#include "ppl/cv/arm/rotate.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, cv::Size>;
inline std::string convertToStringRotate(const Parameters& parameters)
{
    std::ostringstream formatted;

    int degree = std::get<0>(parameters);
    formatted << "Degree" << degree << "_";

    cv::Size size = std::get<1>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

template <typename T, int channels>
class PplCvArmRotateTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmRotateTest()
    {
        const Parameters& parameters = GetParam();
        degree = std::get<0>(parameters);
        size = std::get<1>(parameters);
    }

    ~PplCvArmRotateTest() {}

    bool apply();

private:
    int degree;
    cv::Size size;
};

template <typename T, int channels>
bool PplCvArmRotateTest<T, channels>::apply()
{
    int dst_height, dst_width;
    cv::RotateFlags cv_rotate_flag;
    if (degree == 90) {
        dst_height = size.width;
        dst_width = size.height;
        cv_rotate_flag = cv::ROTATE_90_CLOCKWISE;
    } else if (degree == 180) {
        dst_height = size.height;
        dst_width = size.width;
        cv_rotate_flag = cv::ROTATE_180;
    } else if (degree == 270) {
        dst_height = size.width;
        dst_width = size.height;
        cv_rotate_flag = cv::ROTATE_90_COUNTERCLOCKWISE;
    } else {
        return false;
    }

    cv::Mat src = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(dst_height, dst_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat cv_dst(dst_height, dst_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    cv::rotate(src, cv_dst, cv_rotate_flag);

    ppl::cv::arm::Rotate<T, channels>(size.height,
                                      size.width,
                                      src.step / sizeof(T),
                                      (T*)src.data,
                                      dst_height,
                                      dst_width,
                                      dst.step / sizeof(T),
                                      (T*)dst.data,
                                      degree);

    float epsilon;
    if (sizeof(T) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    bool identity = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(T, channels)                                                                                        \
    using PplCvArmRotateTest_##T##_##channels = PplCvArmRotateTest<T, channels>;                                     \
    TEST_P(PplCvArmRotateTest_##T##_##channels, Standard)                                                            \
    {                                                                                                                \
        bool identity = this->apply();                                                                               \
        EXPECT_TRUE(identity);                                                                                       \
    }                                                                                                                \
                                                                                                                     \
    INSTANTIATE_TEST_CASE_P(IsEqual,                                                                                 \
                            PplCvArmRotateTest_##T##_##channels,                                                     \
                            ::testing::Combine(::testing::Values(90, 180, 270),                                      \
                                               ::testing::Values(cv::Size{320, 240},                                 \
                                                                 cv::Size{321, 245},                                 \
                                                                 cv::Size{642, 484},                                 \
                                                                 cv::Size{647, 493},                                 \
                                                                 cv::Size{654, 486},                                 \
                                                                 cv::Size{1920, 1080})),                             \
                            [](const testing::TestParamInfo<PplCvArmRotateTest_##T##_##channels::ParamType>& info) { \
                                return convertToStringRotate(info.param);                                            \
                            });

UNITTEST(uint8_t, 1)
UNITTEST(uint8_t, 3)
UNITTEST(uint8_t, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)