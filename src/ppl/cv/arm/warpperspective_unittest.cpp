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

#include "ppl/cv/arm/warpperspective.h"

#include <tuple>
#include <sstream>
#include <random>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

enum Scaling {
    kHalfSize,
    kSameSize,
    kDoubleSize,
};

using Parameters = std::tuple<Scaling, ppl::cv::InterpolationType, ppl::cv::BorderType, cv::Size>;
inline std::string convertToStringWarpPerspect(const Parameters& parameters)
{
    std::ostringstream formatted;

    Scaling scale = std::get<0>(parameters);
    if (scale == kHalfSize) {
        formatted << "HalfSize"
                  << "_";
    } else if (scale == kSameSize) {
        formatted << "SameSize"
                  << "_";
    } else if (scale == kDoubleSize) {
        formatted << "DoubleSize"
                  << "_";
    } else {
    }

    ppl::cv::InterpolationType inter_type = std::get<1>(parameters);
    if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
        formatted << "InterLinear"
                  << "_";
    } else if (inter_type == ppl::cv::INTERPOLATION_NEAREST_POINT) {
        formatted << "InterNearest"
                  << "_";
    } else {
    }

    ppl::cv::BorderType border_type = std::get<2>(parameters);
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        formatted << "BORDER_CONSTANT"
                  << "_";
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        formatted << "BORDER_REPLICATE"
                  << "_";
    } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
        formatted << "BORDER_TRANSPARENT"
                  << "_";
    } else {
    }

    cv::Size size = std::get<3>(parameters);
    formatted << size.width << "x";
    formatted << size.height;

    return formatted.str();
}

template <typename T, int channels>
class PplCvArmWarpPerspectiveTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmWarpPerspectiveTest()
    {
        const Parameters& parameters = GetParam();
        scale = std::get<0>(parameters);
        inter_type = std::get<1>(parameters);
        border_type = std::get<2>(parameters);
        size = std::get<3>(parameters);
    }

    ~PplCvArmWarpPerspectiveTest() {}

    bool apply();

private:
    Scaling scale;
    ppl::cv::InterpolationType inter_type;
    ppl::cv::BorderType border_type;
    cv::Size size;
};

cv::Mat getRandomPerspectiveMat(cv::Mat src, cv::Mat dst)
{
    constexpr int offsetLimit = 32;
    cv::Point2f srcPoints[4];
    cv::Point2f dstPoints[4];

    srcPoints[0] = cv::Point2f(0 + rand() % offsetLimit, 0 + rand() % offsetLimit);
    srcPoints[1] = cv::Point2f(src.rows - 1 - rand() % offsetLimit, 0 + rand() % offsetLimit);
    srcPoints[2] = cv::Point2f(src.rows - 1 - rand() % offsetLimit, src.cols - 1 - rand() % offsetLimit);
    srcPoints[3] = cv::Point2f(0 + rand() % offsetLimit, src.cols - 1 - rand() % offsetLimit);

    dstPoints[0] = cv::Point2f(0 + rand() % offsetLimit, 0 + rand() % offsetLimit);
    dstPoints[1] = cv::Point2f(dst.rows - 1 - rand() % offsetLimit, 0 + rand() % offsetLimit);
    dstPoints[2] = cv::Point2f(dst.rows - 1 - rand() % offsetLimit, dst.cols - 1 - rand() % offsetLimit);
    dstPoints[3] = cv::Point2f(0 + rand() % offsetLimit, dst.cols - 1 - rand() % offsetLimit);

    std::mt19937 rng(rand());
    std::shuffle(&dstPoints[0], &dstPoints[0] + 4, rng);

    // inverse map
    cv::Mat M = getPerspectiveTransform(dstPoints, srcPoints);
    return M;
}

template <typename T, int channels>
bool PplCvArmWarpPerspectiveTest<T, channels>::apply()
{
    float scale_coeff;
    if (scale == kHalfSize) {
        scale_coeff = 0.5f;
    } else if (scale == kDoubleSize) {
        scale_coeff = 2.0f;
    } else {
        scale_coeff = 1.0f;
    }
    int dst_height = size.height * scale_coeff;
    int dst_width = size.width * scale_coeff;
    cv::Mat src = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat dst = createSourceImage(dst_height, dst_width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
    cv::Mat cv_dst = dst.clone();
    cv::Mat M = getRandomPerspectiveMat(src, dst);

    int cv_iterpolation;
    if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
        cv_iterpolation = cv::INTER_LINEAR;
    } else {
        cv_iterpolation = cv::INTER_NEAREST;
    }

    cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        cv_border = cv::BORDER_CONSTANT;
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        cv_border = cv::BORDER_REPLICATE;
    } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
        cv_border = cv::BORDER_TRANSPARENT;
    } else {
    }

    int border_value = 5;
    cv::warpPerspective(src,
                        cv_dst,
                        M,
                        cv::Size(dst_width, dst_height),
                        cv_iterpolation | cv::WARP_INVERSE_MAP,
                        cv_border,
                        cv::Scalar(border_value, border_value, border_value, border_value));

    ppl::cv::arm::WarpPerspective<T, channels>(src.rows,
                                               src.cols,
                                               src.step / sizeof(T),
                                               (T*)src.data,
                                               dst_height,
                                               dst_width,
                                               dst.step / sizeof(T),
                                               (T*)dst.data,
                                               (double*)M.data,
                                               inter_type,
                                               border_type,
                                               border_value);

    float epsilon;
    if (sizeof(T) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E6;
    }
    bool identity = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(T, channels)                                                                                    \
    using PplCvArmWarpPerspectiveTest_##T##_##channels = PplCvArmWarpPerspectiveTest<T, channels>;               \
    TEST_P(PplCvArmWarpPerspectiveTest_##T##_##channels, Standard)                                               \
    {                                                                                                            \
        bool identity = this->apply();                                                                           \
        EXPECT_TRUE(identity);                                                                                   \
    }                                                                                                            \
                                                                                                                 \
    INSTANTIATE_TEST_CASE_P(                                                                                     \
        IsEqual,                                                                                                 \
        PplCvArmWarpPerspectiveTest_##T##_##channels,                                                            \
        ::testing::Combine(                                                                                      \
            ::testing::Values(kHalfSize, kSameSize, kDoubleSize),                                                \
            ::testing::Values(ppl::cv::INTERPOLATION_LINEAR, ppl::cv::INTERPOLATION_NEAREST_POINT),              \
            ::testing::Values(ppl::cv::BORDER_CONSTANT, ppl::cv::BORDER_REPLICATE, ppl::cv::BORDER_TRANSPARENT), \
            ::testing::Values(cv::Size{321, 240},                                                                \
                              cv::Size{321, 245},                                                                \
                              cv::Size{647, 493},                                                                \
                              cv::Size{654, 486},                                                                \
                              cv::Size{1280, 720},                                                               \
                              cv::Size{1920, 1080})),                                                            \
        [](const testing::TestParamInfo<PplCvArmWarpPerspectiveTest_##T##_##channels::ParamType>& info) {        \
            return convertToStringWarpPerspect(info.param);                                                      \
        });

UNITTEST(uint8_t, 1)
UNITTEST(uint8_t, 3)
UNITTEST(uint8_t, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
