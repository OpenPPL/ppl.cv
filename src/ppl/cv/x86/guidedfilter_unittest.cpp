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

#include "ppl/cv/x86/guidedfilter.h"
#include <gtest/gtest.h>
#include <opencv2/ximgproc.hpp>
#include "ppl/cv/debug.h"
#include "ppl/cv/x86/test.h"

template<typename T, int32_t c_src, int32_t c_guide>
class GuidedFilter : public ::testing::TestWithParam<std::tuple<Size, float>> {
public:
    using GuidedFilterParam = std::tuple<Size, float>;
    GuidedFilter()
    {
    }

    ~GuidedFilter()
    {
    }

    void apply(const GuidedFilterParam &param) {
        Size size = std::get<0>(param);
        const float diff = std::get<1>(param);

        int32_t r = 8;
        float eps = 0.02 * 0.02 * 255 * 255;

        std::unique_ptr<T[]> src(new T[size.width * size.height * c_src]);
        std::unique_ptr<T[]> guided(new T[size.width * size.height * c_guide]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * c_src]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * c_src]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * c_src, 0, 255);
        ppl::cv::debug::randomFill<T>(guided.get(), size.width * size.height * c_guide, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, c_src), src.get(), sizeof(T) * size.width * c_src);
        cv::Mat guided_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, c_guide), guided.get(), sizeof(T) * size.width * c_guide);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, c_src), dst_ref.get(), sizeof(T) * size.width * c_src);

        cv::ximgproc::guidedFilter(guided_opencv, src_opencv, dst_opencv, r, eps, -1);

        ppl::cv::x86::GuidedFilter<T, c_src, c_guide>(
            size.height,
            size.width,
            size.width * c_src,
            src.get(),
            size.width * c_guide,
            guided.get(),
            size.width * c_src,
            dst.get(),
            r,
            eps,
            ppl::cv::BORDER_REFLECT);

        checkResult<T, c_src>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * c_src,
            size.width * c_src,
            diff);
    }

};

#define R(name, t, c_src, c_guide, diff)\
    using name = GuidedFilter<t, c_src, c_guide>;\
    TEST_P(name, abc)\
    {\
        this->apply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
        ::testing::Combine(::testing::Values(Size{320,240}, Size{640,480}),\
                           ::testing::Values(diff)));

R(GuidedFilter_f32c11, float, 1, 1, 1e-3)
R(GuidedFilter_f32c33, float, 3, 3, 1e-3)
R(GuidedFilter_f32c13, float, 1, 3, 1e-3)
R(GuidedFilter_u8c11, uint8_t, 1, 1, 2)
R(GuidedFilter_u8c33, uint8_t, 3, 3, 2)
R(GuidedFilter_u8c13, uint8_t, 1, 3, 2)
