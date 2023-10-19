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

#include "ppl/cv/arm/transpose.h"

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
class PplCvArmTransposeTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmTransposeTest()
    {
        const Parameters& parameters = GetParam();
        size = std::get<0>(parameters);
    }

    ~PplCvArmTransposeTest() {}

    bool apply();

private:
    cv::Size size;
};

template <typename T, int channels>
bool PplCvArmTransposeTest<T, channels>::apply()
{
    cv::Mat src = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(size.width, size.height, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat cv_dst(size.width, size.height, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    cv::transpose(src, cv_dst);

    ppl::cv::arm::Transpose<T, channels>(
        src.rows, src.cols, src.step / sizeof(T), (T*)src.data, dst.step / sizeof(T), (T*)dst.data);

    float epsilon;
    if (sizeof(T) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    bool identity = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(T, channels)                                                                       \
    using PplCvArmTransposeTest_##T##_##channels = PplCvArmTransposeTest<T, channels>;              \
    TEST_P(PplCvArmTransposeTest_##T##_##channels, Standard)                                        \
    {                                                                                               \
        bool identity = this->apply();                                                              \
        EXPECT_TRUE(identity);                                                                      \
    }                                                                                               \
                                                                                                    \
    INSTANTIATE_TEST_CASE_P(                                                                        \
        IsEqual,                                                                                    \
        PplCvArmTransposeTest_##T##_##channels,                                                     \
        ::testing::Values(cv::Size{320, 240},                                                       \
                          cv::Size{321, 245},                                                       \
                          cv::Size{642, 484},                                                       \
                          cv::Size{647, 493},                                                       \
                          cv::Size{654, 486},                                                       \
                          cv::Size{1920, 1080}),                                                    \
        [](const testing::TestParamInfo<PplCvArmTransposeTest_##T##_##channels::ParamType>& info) { \
            return convertToString(info.param);                                                     \
        });

UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
UNITTEST(uint8_t, 1)
UNITTEST(uint8_t, 3)
UNITTEST(uint8_t, 4)
