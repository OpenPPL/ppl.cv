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

#include "ppl/cv/x86/gaussianblur.h"
#include "ppl/cv/debug.h"
#include "ppl/cv/x86/test.h"
#include <gtest/gtest.h>

template<typename T, ppl::cv::BorderType border_type, int c>
class gaussblur_ : public  ::testing::TestWithParam<std::tuple<Size, int>> {
public:
    using GaussblurParameter = std::tuple<Size, int>;
    gaussblur_()
    {
    }
    ~gaussblur_()
    {
    }

    void apply(const GaussblurParameter &param)
    {
        Size size = std::get<0>(param);
        int kernel = std::get<1>(param);
        std::unique_ptr<T[]> src(new T[size.width * size.height * c]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * c]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * c]);
        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * c, 0, 255);
        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, c), src.get(), sizeof(T) * size.width * c);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, c), dst_ref.get(), sizeof(T) * size.width * c);
        int cv_bordertype = 4;
        if(border_type == ppl::cv::BORDER_REFLECT_101) {
            cv_bordertype = 4;
        } else if(border_type == ppl::cv::BORDER_REFLECT) {
            cv_bordertype = 2;
        } else if(border_type == ppl::cv::BORDER_REPLICATE) {
            cv_bordertype = 1;
        }
        cv::GaussianBlur(src_opencv, dst_opencv, cv::Size(kernel, kernel), 0, 0, cv_bordertype);
        ppl::cv::x86::GaussianBlur<T, c>(size.height, size.width, size.width * c, src.get(), kernel, 0.0f, size.width * c, dst.get(), border_type);

        checkResult<T, c>(dst_ref.get(), dst.get(),
                        size.height, size.width,
                        size.width * c, size.width * c, 1.01f);
    }
};

#define R(name, t, b, c) \
    using name = gaussblur_<t, b, c>; \
    TEST_P(name, abc) \
    { \
        this->apply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
                            ::testing::Combine(\
                            ::testing::Values(Size{320, 240}, Size{640, 480}, Size{321, 241}, Size{319, 239}),\
                            ::testing::Values(3, 5, 7)));

R(gaussianblur_f32c1_reflect_101, float, ppl::cv::BORDER_REFLECT_101, 1)
R(gaussianblur_f32c3_reflect_101, float, ppl::cv::BORDER_REFLECT_101, 3)
R(gaussianblur_f32c4_reflect_101, float, ppl::cv::BORDER_REFLECT_101, 4)
R(gaussianblur_u8c1_reflect_101, uint8_t, ppl::cv::BORDER_REFLECT_101, 1)
R(gaussianblur_u8c3_reflect_101, uint8_t, ppl::cv::BORDER_REFLECT_101, 3)
R(gaussianblur_u8c4_reflect_101, uint8_t, ppl::cv::BORDER_REFLECT_101, 4)

R(gaussianblur_f32c1_reflect, float, ppl::cv::BORDER_REFLECT, 1)
R(gaussianblur_f32c3_reflect, float, ppl::cv::BORDER_REFLECT, 3)
R(gaussianblur_f32c4_reflect, float, ppl::cv::BORDER_REFLECT, 4)
R(gaussianblur_u8c1_reflect, uint8_t, ppl::cv::BORDER_REFLECT, 1)
R(gaussianblur_u8c3_reflect, uint8_t, ppl::cv::BORDER_REFLECT, 3)
R(gaussianblur_u8c4_reflect, uint8_t, ppl::cv::BORDER_REFLECT, 4)

R(gaussianblur_f32c1_replicate, float, ppl::cv::BORDER_REPLICATE, 1)
R(gaussianblur_f32c3_replicate, float, ppl::cv::BORDER_REPLICATE, 3)
R(gaussianblur_f32c4_replicate, float, ppl::cv::BORDER_REPLICATE, 4)
R(gaussianblur_u8c1_replicate, uint8_t, ppl::cv::BORDER_REPLICATE, 1)
R(gaussianblur_u8c3_replicate, uint8_t, ppl::cv::BORDER_REPLICATE, 3)
R(gaussianblur_u8c4_replicate, uint8_t, ppl::cv::BORDER_REPLICATE, 4)
