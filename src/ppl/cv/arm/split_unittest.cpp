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

#include "ppl/cv/arm/split.h"

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
class PplCvArmSplitTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmSplitTest()
    {
        const Parameters& parameters = GetParam();
        size = std::get<0>(parameters);
    }

    ~PplCvArmSplitTest() {}

    bool apply();

private:
    cv::Size size;
};

template <typename T, int channels>
bool PplCvArmSplitTest<T, channels>::apply()
{
    cv::Mat src = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst0(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst1(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst2(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat dst3(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat cv_dst0(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat cv_dst1(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat cv_dst2(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
    cv::Mat cv_dst3(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));

    if (channels == 3) {
        cv::Mat dsts[3] = {cv_dst0, cv_dst1, cv_dst2};
        cv::split(src, dsts);
        ppl::cv::arm::Split3Channels<T>(src.rows,
                                        src.cols,
                                        src.step / sizeof(T),
                                        (T*)src.data,
                                        dst0.step / sizeof(T),
                                        (T*)dst0.data,
                                        (T*)dst1.data,
                                        (T*)dst2.data);
    } else { // channels == 4
        cv::Mat dsts[4] = {cv_dst0, cv_dst1, cv_dst2, cv_dst3};
        cv::split(src, dsts);
        ppl::cv::arm::Split4Channels<T>(src.rows,
                                        src.cols,
                                        src.step / sizeof(T),
                                        (T*)src.data,
                                        dst0.step / sizeof(T),
                                        (T*)dst0.data,
                                        (T*)dst1.data,
                                        (T*)dst2.data,
                                        (T*)dst3.data);
    }

    float epsilon;
    if (sizeof(T) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    bool identity0, identity1, identity2, identity3;
    identity0 = checkMatricesIdentity<T>(cv_dst0, dst0, epsilon);
    identity1 = checkMatricesIdentity<T>(cv_dst1, dst1, epsilon);
    identity2 = checkMatricesIdentity<T>(cv_dst2, dst2, epsilon);
    if (channels == 4) { identity3 = checkMatricesIdentity<T>(cv_dst3, dst3, epsilon); }

    if (channels == 3) {
        return (identity0 && identity1 && identity2);
    } else {
        return (identity0 && identity1 && identity2 && identity3);
    }
}

#define UNITTEST(T, channels)                                                                                       \
    using PplCvArmSplitTest_##T##_##channels = PplCvArmSplitTest<T, channels>;                                      \
    TEST_P(PplCvArmSplitTest_##T##_##channels, Standard)                                                            \
    {                                                                                                               \
        bool identity = this->apply();                                                                              \
        EXPECT_TRUE(identity);                                                                                      \
    }                                                                                                               \
                                                                                                                    \
    INSTANTIATE_TEST_CASE_P(IsEqual,                                                                                \
                            PplCvArmSplitTest_##T##_##channels,                                                     \
                            ::testing::Values(cv::Size{320, 240},                                                   \
                                              cv::Size{321, 245},                                                   \
                                              cv::Size{647, 493},                                                   \
                                              cv::Size{654, 486},                                                   \
                                              cv::Size{1280, 720},                                                  \
                                              cv::Size{1920, 1080}),                                                \
                            [](const testing::TestParamInfo<PplCvArmSplitTest_##T##_##channels::ParamType>& info) { \
                                return convertToString(info.param);                                                 \
                            });

UNITTEST(uint8_t, 3)
UNITTEST(float, 3)
UNITTEST(uint8_t, 4)
UNITTEST(float, 4)