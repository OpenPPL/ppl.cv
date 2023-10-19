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

#include "ppl/cv/arm/filter2d.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, int, ppl::cv::BorderType, cv::Size>;
inline std::string convertToStringFilter2D(const Parameters& parameters)
{
    std::ostringstream formatted;

    int ksize = std::get<0>(parameters);
    formatted << "Ksize" << ksize << "_";

    int int_delta = std::get<1>(parameters);
    formatted << "Delta" << int_delta << "_";

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

template <typename Tsrc, typename Tdst, int channels>
class PplCvArmFilter2DTest : public ::testing::TestWithParam<Parameters> {
public:
    PplCvArmFilter2DTest()
    {
        const Parameters& parameters = GetParam();
        ksize = std::get<0>(parameters);
        delta = std::get<1>(parameters) / 10.f;
        border_type = std::get<2>(parameters);
        size = std::get<3>(parameters);
    }

    ~PplCvArmFilter2DTest() {}

    bool apply();

private:
    int ksize;
    float delta;
    ppl::cv::BorderType border_type;
    cv::Size size;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvArmFilter2DTest<Tsrc, Tdst, channels>::apply()
{
    cv::Mat src = createSourceImage(size.height, size.width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
    cv::Mat kernel0 = createSourceImage(ksize, ksize, CV_MAKETYPE(cv::DataType<float>::depth, 1));
    cv::Mat dst(size.height, size.width, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
    cv::Mat cv_dst(size.height, size.width, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

    cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
    if (border_type == ppl::cv::BORDER_REPLICATE) {
        cv_border = cv::BORDER_REPLICATE;
    } else if (border_type == ppl::cv::BORDER_REFLECT) {
        cv_border = cv::BORDER_REFLECT;
    } else if (border_type == ppl::cv::BORDER_REFLECT_101) {
        cv_border = cv::BORDER_REFLECT_101;
    } else {
    }
    cv::filter2D(src, cv_dst, cv_dst.depth(), kernel0, cv::Point(-1, -1), delta, cv_border);

    ppl::cv::arm::Filter2D<Tsrc, channels>(src.rows,
                                           src.cols,
                                           src.step / sizeof(Tsrc),
                                           (Tsrc*)src.data,
                                           ksize,
                                           (float*)kernel0.data,
                                           dst.step / sizeof(Tdst),
                                           (Tdst*)dst.data,
                                           delta,
                                           border_type);

    float epsilon;
    if (sizeof(Tsrc) == 1) {
        epsilon = EPSILON_1F;
    } else {
        epsilon = EPSILON_E3; // todo: change to E6 after implemention of FFT filter
    }
    bool identity = checkMatricesIdentity<Tdst>(cv_dst, dst, epsilon);

    return identity;
}

#define UNITTEST(Tsrc, Tdst, channels)                                                                                 \
    using PplCvArmFilter2DTest_##Tsrc##_##channels = PplCvArmFilter2DTest<Tsrc, Tdst, channels>;                       \
    TEST_P(PplCvArmFilter2DTest_##Tsrc##_##channels, Standard)                                                         \
    {                                                                                                                  \
        bool identity = this->apply();                                                                                 \
        EXPECT_TRUE(identity);                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    INSTANTIATE_TEST_CASE_P(                                                                                           \
        IsEqual,                                                                                                       \
        PplCvArmFilter2DTest_##Tsrc##_##channels,                                                                      \
        ::testing::Combine(                                                                                            \
            ::testing::Values(1, 3, 5, 17, 25, 31, 43),                                                                \
            ::testing::Values(0, 43),                                                                                  \
            ::testing::Values(ppl::cv::BORDER_REPLICATE, ppl::cv::BORDER_REFLECT, ppl::cv::BORDER_REFLECT_101),        \
            ::testing::Values(                                                                                         \
                cv::Size{320, 240}, cv::Size{321, 245}, cv::Size{647, 493}, cv::Size{654, 486}, cv::Size{1280, 720})), \
        [](const testing::TestParamInfo<PplCvArmFilter2DTest_##Tsrc##_##channels::ParamType>& info) {                  \
            return convertToStringFilter2D(info.param);                                                                \
        });

UNITTEST(uint8_t, uint8_t, 1)
UNITTEST(uint8_t, uint8_t, 3)
UNITTEST(uint8_t, uint8_t, 4)
UNITTEST(float, float, 1)
UNITTEST(float, float, 3)
UNITTEST(float, float, 4)
