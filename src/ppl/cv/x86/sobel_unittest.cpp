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

#include "ppl/cv/x86/sobel.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include "ppl/common/retcode.h"

template <typename Tsrc, int c, typename Tdst>
class Sobel : public ::testing::TestWithParam<std::tuple<Size, int, int, int, double, double>> {
public:
    using SobelParam = std::tuple<Size, int, int, int, double, double>;
    Sobel()
    {
    }

    ~Sobel()
    {
    }

    void apply(const SobelParam &param)
    {
        Size size    = std::get<0>(param);
        int dx       = std::get<1>(param);
        int dy       = std::get<2>(param);
        int ksize    = std::get<3>(param);
        double scale = std::get<4>(param);
        double delta = std::get<5>(param);

        std::unique_ptr<Tsrc[]> src(new Tsrc[size.width * size.height * c]);
        ppl::cv::debug::randomFill<Tsrc>(src.get(), size.width * size.height * c, 0, 255);

        std::unique_ptr<Tdst[]> dst(new Tdst[size.width * size.height * c]);
        std::unique_ptr<Tdst[]> dst_opencv(new Tdst[size.width * size.height * c]);

        ppl::cv::x86::Sobel<Tsrc, Tdst, c>(size.height, size.width, size.width * c, src.get(), size.width * c, dst.get(), dx, dy, ksize, scale, delta, ppl::cv::BORDER_DEFAULT);

        ::cv::Mat iMat(size.height, size.width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, c), src.get());
        ::cv::Mat oMat(size.height, size.width, CV_MAKETYPE(cv::DataType<Tdst>::depth, c), dst_opencv.get());

        ::cv::Sobel(iMat, oMat, oMat.depth(), dx, dy, ksize, scale, delta, ::cv::BORDER_REFLECT_101);
        checkResult<Tdst, c>(
            dst.get(),
            dst_opencv.get(),
            size.width,
            size.height,
            size.height * c,
            size.height * c,
            1.01f);
    }
};

#define R(name, ts, c, td)         \
    using name = Sobel<ts, c, td>; \
    TEST_P(name, abc)              \
    {                              \
        this->apply(GetParam());   \
    }                              \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{10, 8}), ::testing::Values(1, 2), ::testing::Values(1, 2), ::testing::Values(1, 3, 5), ::testing::Values(1.0), ::testing::Values(0.0)));

R(Sobel_f32c1, float, 1, float)
R(Sobel_f32c3, float, 3, float)
R(Sobel_f32c4, float, 4, float)

R(Sobel_u8c1, uint8_t, 1, int16_t)
R(Sobel_u8c3, uint8_t, 3, int16_t)
R(Sobel_u8c4, uint8_t, 4, int16_t)
