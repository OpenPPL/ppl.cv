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

#include "ppl/cv/x86/pyrdown.h"
#include "ppl/cv/x86/test.h"
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include <memory>

template<typename T, int32_t c>
class PyrDown : public ::testing::TestWithParam<std::tuple<Size, Size>> {
public:
    using PyrDownParam = std::tuple<Size, Size>;
    PyrDown()
    {
    }

    ~PyrDown()
    {
    }

    void apply(const PyrDownParam &param) {
        Size isize = std::get<0>(param);
        Size osize = std::get<1>(param);

        std::unique_ptr<T> src(new T[isize.width * isize.height * c]);
        ppl::cv::debug::randomFill<T>(src.get(), isize.width * isize.height * c, 0, 255);
        std::unique_ptr<T> dst(new T[osize.width * osize.height * c]);
        std::unique_ptr<T> dst_opencv(new T[osize.width * osize.height * c]);

        ppl::cv::x86::PyrDown<T, c>(isize.height, isize.width, isize.width * c, src.get(),
            osize.width * c, dst.get(), ppl::cv::BORDER_DEFAULT);

        ::cv::Mat iMat(isize.height, isize.width, T2CvType<T, c>::type, src.get());
        ::cv::Mat oMat(osize.height, osize.width, T2CvType<T, c>::type, dst_opencv.get());

        ::cv::pyrDown(iMat, oMat, ::cv::Size(osize.width, osize.height));
        checkResult<T, c>(dst.get(), dst_opencv.get(), osize.height, osize.width, osize.width * c, osize.width * c, 1e-3);
    }
};

#define R(name, t, c)\
    using name = PyrDown<t, c>;\
    TEST_P(name, t ## c)\
    {\
        this->apply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
        ::testing::Combine(::testing::Values(Size{6, 8}),\
                           ::testing::Values(Size{3, 4})));

R(PyrDown_x86_f32c1, float, 1)
R(PyrDown_x86_f32c3, float, 3)
R(PyrDown_x86_f32c4, float, 4)

R(PyrDown_x86_u8c1, uint8_t, 1)
R(PyrDown_x86_u8c3, uint8_t, 3)
R(PyrDown_x86_u8c4, uint8_t, 4)
