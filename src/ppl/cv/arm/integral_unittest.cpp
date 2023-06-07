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

#include "ppl/cv/arm/integral.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
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

template <typename Tsrc, typename Tdst, int channels>
class PplCvArmIntegralTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmIntegralTest()
    {
        const Parameters& parameters = GetParam();
        size = std::get<0>(parameters);
    }

    ~PplCvArmIntegralTest() {}

    bool apply();

private:
    cv::Size size;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvArmIntegralTest<Tsrc, Tdst, channels>::apply()
{
    cv::Mat src = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
    cv::Mat dst(size.height + 1, size.width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
    cv::Mat cv_dst(size.height + 1, size.width + 1, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

    cv::integral(src, cv_dst, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
    if (dst.rows == src.rows && dst.cols == src.cols) {
        cv::Rect roi(1, 1, size.width, size.height);
        cv::Mat croppedImage = cv_dst(roi);
        cv::Mat tmp;
        croppedImage.copyTo(tmp);
        cv_dst = tmp;
    }

    ppl::cv::arm::Integral<Tsrc, Tdst, channels>(src.rows,
                                                 src.cols,
                                                 src.step / sizeof(Tsrc),
                                                 (Tsrc*)src.data,
                                                 dst.rows,
                                                 dst.cols,
                                                 dst.step / sizeof(Tdst),
                                                 (Tdst*)dst.data);

    float epsilon;
    if (sizeof(Tdst) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    bool identity = checkMatricesIdentity<Tdst>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(Tsrc, Tdst, channels)                                                                         \
    using PplCvArmIntegralTest_##Tsrc##_##Tdst##_##channels = PplCvArmIntegralTest<Tsrc, Tdst, channels>;      \
    TEST_P(PplCvArmIntegralTest_##Tsrc##_##Tdst##_##channels, Standard)                                        \
    {                                                                                                          \
        bool identity = this->apply();                                                                         \
        EXPECT_TRUE(identity);                                                                                 \
    }                                                                                                          \
                                                                                                               \
    INSTANTIATE_TEST_CASE_P(                                                                                   \
        IsEqual,                                                                                               \
        PplCvArmIntegralTest_##Tsrc##_##Tdst##_##channels,                                                     \
        ::testing::Values(cv::Size{321, 240},                                                                  \
                          cv::Size{642, 480},                                                                  \
                          cv::Size{1283, 720},                                                                 \
                          cv::Size{1934, 1080},                                                                \
                          cv::Size{320, 240},                                                                  \
                          cv::Size{640, 480},                                                                  \
                          cv::Size{1280, 720},                                                                 \
                          cv::Size{1920, 1080}),                                                               \
        [](const testing::TestParamInfo<PplCvArmIntegralTest_##Tsrc##_##Tdst##_##channels::ParamType>& info) { \
            return convertToString(info.param);                                                                \
        });

UNITTEST(uint8_t, int32_t, 1)
UNITTEST(uint8_t, int32_t, 3)
UNITTEST(uint8_t, int32_t, 4)
UNITTEST(float, float, 1)
UNITTEST(float, float, 3)
UNITTEST(float, float, 4)
