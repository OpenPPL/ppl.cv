/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "ppl/cv/arm/convertto.h"

#include <tuple>
#include <sstream>
#include <cmath>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "ppl/cv/debug.h"
#include "ppl/cv/arm/test.h"
#include "utility/infrastructure.hpp"

#define BASE  (50)
#define SCALE (10.f)

using Parameters = std::tuple<float, float, cv::Size>;
inline std::string convertToStringConvertto(const Parameters& parameters)
{
    std::ostringstream formatted;

    int int_alpha = std::get<0>(parameters);
    formatted << "IntAlpha" << int_alpha << "_";

    int int_beta = std::get<1>(parameters);
    formatted << "IntBeta" << int_beta << "_";

    cv::Size size = std::get<2>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

template <typename Tsrc, typename Tdst, int channels>
class PplCvArmConvertToTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmConvertToTest()
    {
        const Parameters& parameters = GetParam();
        alpha = (std::get<0>(parameters) - BASE) / SCALE;
        beta = (std::get<1>(parameters) - BASE) / SCALE;
        size = std::get<2>(parameters);
    }

    ~PplCvArmConvertToTest() {}

    bool apply();

private:
    float alpha;
    float beta;
    cv::Size size;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvArmConvertToTest<Tsrc, Tdst, channels>::apply()
{
    int width = size.width;
    int height = size.height;
    cv::Mat src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
    cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
    cv::Mat cv_dst(height, width, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
    src.convertTo(cv_dst, cv_dst.type(), alpha, beta);

    ppl::cv::arm::ConvertTo<Tsrc, Tdst, channels>(src.rows,
                                                  src.cols,
                                                  src.step / sizeof(Tsrc),
                                                  (Tsrc*)src.data,
                                                  dst.step / sizeof(Tdst),
                                                  (Tdst*)dst.data,
                                                  alpha,
                                                  beta);

    float epsilon;
    if (sizeof(Tdst) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    bool identity = checkMatricesIdentity<Tdst>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(Tsrc, Tdst, channels)                                                                             \
    using PplCvArmConvertToTest_##Tsrc##_To_##Tdst##_##channels = PplCvArmConvertToTest<Tsrc, Tdst, channels>;     \
    TEST_P(PplCvArmConvertToTest_##Tsrc##_To_##Tdst##_##channels, Standard)                                        \
    {                                                                                                              \
        bool identity = this->apply();                                                                             \
        EXPECT_TRUE(identity);                                                                                     \
    }                                                                                                              \
                                                                                                                   \
    INSTANTIATE_TEST_CASE_P(                                                                                       \
        IsEqual,                                                                                                   \
        PplCvArmConvertToTest_##Tsrc##_To_##Tdst##_##channels,                                                     \
        ::testing::Combine(::testing::Values(37, 60, 65, 2560),                                                    \
                           ::testing::Values(13, 50, 89),                                                          \
                           ::testing::Values(cv::Size{320, 240},                                                   \
                                             cv::Size{1280, 720},                                                  \
                                             cv::Size{1920, 1080},                                                 \
                                             cv::Size{321, 245},                                                   \
                                             cv::Size{647, 493},                                                   \
                                             cv::Size{654, 486})),                                                 \
        [](const testing::TestParamInfo<PplCvArmConvertToTest_##Tsrc##_To_##Tdst##_##channels::ParamType>& info) { \
            return convertToStringConvertto(info.param);                                                           \
        });

UNITTEST(uint8_t, uint8_t, 1)
UNITTEST(uint8_t, uint8_t, 3)
UNITTEST(uint8_t, uint8_t, 4)
UNITTEST(uint8_t, float, 1)
UNITTEST(uint8_t, float, 3)
UNITTEST(uint8_t, float, 4)
UNITTEST(float, uint8_t, 1)
UNITTEST(float, uint8_t, 3)
UNITTEST(float, uint8_t, 4)
UNITTEST(float, float, 1)
UNITTEST(float, float, 3)
UNITTEST(float, float, 4)
