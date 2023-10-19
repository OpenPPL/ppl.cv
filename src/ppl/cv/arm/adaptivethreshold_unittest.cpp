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

#include "ppl/cv/arm/adaptivethreshold.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, int, int, int, int, ppl::cv::BorderType, cv::Size>;
inline std::string convertToStringThreshold(const Parameters& parameters)
{
    std::ostringstream formatted;

    int ksize = std::get<0>(parameters);
    formatted << "Ksize" << ksize << "_";

    int adaptive_method = std::get<1>(parameters);
    formatted << (adaptive_method == ppl::cv::ADAPTIVE_THRESH_MEAN_C ? "METHOD_MEAN" : "METHOD_GAUSSIAN") << "_";

    int threshold_type = std::get<2>(parameters);
    formatted << (threshold_type == ppl::cv::THRESH_BINARY ? "THRESH_BINARY" : "THRESH_BINARY_INV") << "_";

    int max_value = std::get<3>(parameters);
    formatted << "MaxValue" << max_value << "_";

    int int_delta = std::get<4>(parameters);
    formatted << "IntDelta" << int_delta << "_";

    ppl::cv::BorderType border_type = std::get<5>(parameters);
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

    cv::Size size = std::get<6>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

template <typename T, int channels>
class PplCvArmAdaptiveThresholdTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmAdaptiveThresholdTest()
    {
        const Parameters& parameters = GetParam();
        ksize = std::get<0>(parameters);
        adaptive_method = std::get<1>(parameters);
        threshold_type = std::get<2>(parameters);
        max_value = std::get<3>(parameters) / 10.f;
        delta = std::get<4>(parameters) / 10.f;
        border_type = std::get<5>(parameters);
        size = std::get<6>(parameters);

        max_value -= 1.f;
    }

    ~PplCvArmAdaptiveThresholdTest() {}

    bool apply();

private:
    int ksize;
    int adaptive_method;
    int threshold_type;
    float max_value;
    float delta;
    ppl::cv::BorderType border_type;
    cv::Size size;
};

template <typename T, int channels>
bool PplCvArmAdaptiveThresholdTest<T, channels>::apply()
{
    cv::Mat src = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat cv_dst(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

    cv::AdaptiveThresholdTypes cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
    if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_MEAN_C) {
        cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
    } else if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C) {
        cv_adaptive_method = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
    } else {
    }

    cv::ThresholdTypes cv_threshold_type = cv::THRESH_BINARY;
    if (threshold_type == ppl::cv::THRESH_BINARY) {
        cv_threshold_type = cv::THRESH_BINARY;
    } else if (threshold_type == ppl::cv::THRESH_BINARY_INV) {
        cv_threshold_type = cv::THRESH_BINARY_INV;
    }
    cv::adaptiveThreshold(src, cv_dst, max_value, cv_adaptive_method, cv_threshold_type, ksize, delta);

    ppl::cv::arm::AdaptiveThreshold(src.rows,
                                    src.cols,
                                    src.step,
                                    (uint8_t*)src.data,
                                    dst.step,
                                    (uint8_t*)dst.data,
                                    max_value,
                                    adaptive_method,
                                    threshold_type,
                                    ksize,
                                    delta,
                                    border_type);

    float epsilon;
    if (sizeof(T) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    // can't perfectly align, possibly because we used associative law to optimize GaussianBlur
    // todo: maybe we should use better criteria like percentage of error pixels?
    bool identity = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(T, channels)                                                                                       \
    using PplCvArmAdaptiveThresholdTest_##T##_##channels = PplCvArmAdaptiveThresholdTest<T, channels>;              \
    TEST_P(PplCvArmAdaptiveThresholdTest_##T##_##channels, Standard)                                                \
    {                                                                                                               \
        bool identity = this->apply();                                                                              \
        EXPECT_TRUE(identity);                                                                                      \
    }                                                                                                               \
                                                                                                                    \
    INSTANTIATE_TEST_CASE_P(                                                                                        \
        IsEqual,                                                                                                    \
        PplCvArmAdaptiveThresholdTest_##T##_##channels,                                                             \
        ::testing::Combine(::testing::Values(3, 5, 13, 31, 43),                                                     \
                           ::testing::Values(ppl::cv::ADAPTIVE_THRESH_MEAN_C, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C), \
                           ::testing::Values(ppl::cv::THRESH_BINARY, ppl::cv::THRESH_BINARY_INV),                   \
                           ::testing::Values(0, 70, 1587, 3784),                                                    \
                           ::testing::Values(0, 70, 1587, 3784),                                                    \
                           ::testing::Values(ppl::cv::BORDER_REPLICATE),                                            \
                           ::testing::Values(cv::Size{321, 240},                                                    \
                                             cv::Size{323, 245},                                                    \
                                             cv::Size{320, 240},                                                    \
                                             cv::Size{642, 480},                                                    \
                                             cv::Size{647, 493},                                                    \
                                             cv::Size{654, 486},                                                    \
                                             cv::Size{1280, 720},                                                   \
                                             cv::Size{1920, 1080})),                                                \
        [](const testing::TestParamInfo<PplCvArmAdaptiveThresholdTest_##T##_##channels::ParamType>& info) {         \
            return convertToStringThreshold(info.param);                                                            \
        });

UNITTEST(uint8_t, 1)
