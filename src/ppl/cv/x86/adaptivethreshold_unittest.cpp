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

#include "ppl/cv/x86/adaptivethreshold.h"
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include "ppl/cv/types.h"
#include "ppl/cv/x86/test.h"
#include <memory>

template<typename Tsrc, int32_t c>
class AdaptiveThreshold : public ::testing::TestWithParam<std::tuple<Size, int32_t, int32_t, int32_t, double, double>> {
public:
    using AdaptiveThresholdParam = std::tuple<Size, int32_t, int32_t, int32_t, double, double>;
    AdaptiveThreshold()
    {
    }

    ~AdaptiveThreshold()
    {
    }

    void apply(const AdaptiveThresholdParam &param)
    {
        Size size = std::get<0>(param);
        int32_t max_value = std::get<1>(param);
        int32_t adaptive_method = std::get<2>(param);
        int32_t threshold_type = std::get<3>(param);
        int32_t ksize = std::get<4>(param);
        double delta = std::get<5>(param);

        std::unique_ptr<Tsrc[]> src(new Tsrc[size.width * size.height * c]);
        ppl::cv::debug::randomFill<Tsrc>(src.get(), size.width * size.height * c, 0, 255);

        std::unique_ptr<Tsrc[]> dst(new Tsrc[size.width * size.height * c]);
        std::unique_ptr<Tsrc[]> dst_opencv(new Tsrc[size.width * size.height * c]);

        ppl::cv::x86::AdaptiveThreshold(size.height, size.width, size.width,src.get(),
            size.width, dst.get(), max_value, adaptive_method, threshold_type, ksize, delta,
            ppl::cv::BORDER_REPLICATE);

        ::cv::Mat iMat(size.height, size.width, T2CvType<Tsrc, c>::type, src.get());
        ::cv::Mat oMat(size.height, size.width, T2CvType<Tsrc, c>::type, dst_opencv.get());

        ::cv::adaptiveThreshold(iMat, oMat, max_value, adaptive_method, threshold_type, ksize, delta);

        Tsrc *ptr_dst = dst.get();
        Tsrc *ptr_opencv = dst_opencv.get();
        for (int32_t i = 0; i < size.height; ++i) {
            for (int32_t j = 0; j < size.width * c; ++j) {
                if (ptr_dst[i * size.width * c + j] - ptr_opencv[i * size.width * c + j] > 1 ||
                    ptr_dst[i * size.width * c + j] - ptr_opencv[i * size.width * c + j] < -1) {
                    printf("Diff at (%d, %d), ppl.cv: %f, opencv: %f\n",
                        i, j, (float)ptr_dst[i * size.width * c + j],
                        (float)ptr_opencv[i * size.width * c + j]);
                }
            }
        }
    }
};

#define R(name, ts, c)\
    using name = AdaptiveThreshold<ts, c>;\
    TEST_P(name, abc)\
    {\
        this->apply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
        ::testing::Combine(::testing::Values(Size{320, 240}, Size{100, 100}),\
                           ::testing::Values(155),\
                           ::testing::Values(ppl::cv::ADAPTIVE_THRESH_MEAN_C, ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C),\
                           ::testing::Values(ppl::cv::CV_THRESH_BINARY, ppl::cv::CV_THRESH_BINARY_INV),\
                           ::testing::Values(3, 5, 7),\
                           ::testing::Values(0, 5, 13, 18)));

R(AdaptiveThreshold_u8c1, uint8_t, 1)
