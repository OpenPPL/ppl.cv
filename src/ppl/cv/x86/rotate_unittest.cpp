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

#include "ppl/cv/x86/rotate.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include "ppl/common/retcode.h"

template <typename T, int val>
void randomRangeData(T *data, const size_t num, int maxNum = 255)
{
    size_t tmp;

    for (size_t i = 0; i < num; i++) {
        tmp     = rand() % maxNum;
        data[i] = (T)((float)tmp / (float)val);
    }
}

template <typename T, int c>
class Rotate : public ::testing::TestWithParam<std::tuple<Size, int, float>> {
public:
    using RotateParam = std::tuple<Size, int, float>;
    Rotate()
    {
    }

    ~Rotate()
    {
    }

    void apply(const RotateParam &param)
    {
        Size size        = std::get<0>(param);
        //int degree = std::get<1>(param);
        const float diff = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * c]);
        std::unique_ptr<T[]> src1(new T[size.width * size.height * c]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * c]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * c]);

        randomRangeData<T, 1>(src.get(), size.width * size.height * c);

        ppl::cv::x86::RotateNx90degree<T, c>(
            size.height,
            size.width,
            size.width * c,
            src.get(),
            size.width,
            size.height,
            size.height * c,
            src1.get(),
            90);

        ppl::cv::x86::RotateNx90degree<T, c>(
            size.width,
            size.height,
            size.height * c,
            src1.get(),
            size.height,
            size.width,
            size.width * c,
            dst_ref.get(),
            90);

        ppl::cv::x86::RotateNx90degree<T, c>(
            size.height,
            size.width,
            size.width * c,
            src.get(),
            size.height,
            size.width,
            size.width * c,
            dst.get(),
            180);

        checkResult<T, c>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * c,
            size.width * c,
            diff);

        ppl::cv::x86::RotateNx90degree<T, c>(
            size.height,
            size.width,
            size.width * c,
            src.get(),
            size.width,
            size.height,
            size.height * c,
            src1.get(),
            270);

        ppl::cv::x86::RotateNx90degree<T, c>(
            size.height,
            size.width,
            size.width * c,
            dst_ref.get(),
            size.width,
            size.height,
            size.height * c,
            dst.get(),
            90);

        checkResult<T, c>(
            dst.get(),
            src1.get(),
            size.width,
            size.height,
            size.height * c,
            size.height * c,
            diff);
    }
};

#define R1(name, t, c, diff)     \
    using name = Rotate<t, c>;   \
    TEST_P(name, abc)            \
    {                            \
        this->apply(GetParam()); \
    }                            \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}, Size{723, 483}, Size{777, 333}, Size{31, 31}, Size{17, 101}, Size{65, 65}, Size{97, 107}), ::testing::Values(90), ::testing::Values(diff)));
R1(Rotate_f32c1, float, 1, 1e-5)
R1(Rotate_f32c2, float, 2, 1e-5)
R1(Rotate_f32c3, float, 3, 1e-5)
R1(Rotate_f32c4, float, 4, 1e-5)

R1(Rotate_u8c1, uint8_t, 1, 1)
R1(Rotate_u8c2, uint8_t, 2, 1)
R1(Rotate_u8c3, uint8_t, 3, 1)
R1(Rotate_u8c4, uint8_t, 4, 1)