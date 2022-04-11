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

#include "ppl/cv/riscv/resize.h"
#include "ppl/cv/riscv/test.h"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "ppl/cv/debug.h"

struct Size_p {
    int inWidth;
    int inHeight;
    int outWidth;
    int outHeight;
};

template <typename T, int c>
class Resize : public ::testing::TestWithParam<std::tuple<Size_p, float>> {
public:
    using ResizeParam = std::tuple<Size_p, float>;
    Resize()
    {
    }

    ~Resize()
    {
    }

    void Linearapply(const ResizeParam &param)
    {
        Size_p size = std::get<0>(param);
        const float diff = std::get<1>(param);
        std::cout << __FUNCTION__ << std::endl;

        std::unique_ptr<T[]> src(new T[size.inWidth * size.inHeight * c]);
        std::unique_ptr<T[]> dst_ref(new T[size.outWidth * size.outHeight * c]);
        std::unique_ptr<T[]> dst(new T[size.outWidth * size.outHeight * c]);

        ppl::cv::debug::randomFill<T>(src.get(), size.inWidth * size.inHeight * c, 0, 255);
        cv::Mat src_opencv(size.inHeight, size.inWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), src.get(), sizeof(T) * size.inWidth * c);
        cv::Mat dst_opencv(size.outHeight, size.outWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), dst_ref.get(), sizeof(T) * size.outWidth * c);

        cv::resize(src_opencv, dst_opencv, cv::Size(size.outWidth, size.outHeight), 0, 0, cv::INTER_LINEAR);

        ppl::cv::riscv::ResizeLinear<T, c>(
            size.inHeight,
            size.inWidth,
            size.inWidth * c,
            src.get(),
            size.outHeight,
            size.outWidth,
            size.outWidth * c,
            dst.get());

        checkResult<T, c>(
            dst_ref.get(),
            dst.get(),
            size.outHeight,
            size.outWidth,
            size.outWidth * c,
            size.outWidth * c,
            diff);
    }

    void NearestPointapply(const ResizeParam &param)
    {
        Size_p size = std::get<0>(param);
        const float diff = std::get<1>(param);
        std::cout << __FUNCTION__ << std::endl;

        std::unique_ptr<T[]> src(new T[size.inWidth * size.inHeight * c]);
        std::unique_ptr<T[]> dst_ref(new T[size.outWidth * size.outHeight * c]);
        std::unique_ptr<T[]> dst(new T[size.outWidth * size.outHeight * c]);

        ppl::cv::debug::randomFill<T>(src.get(), size.inWidth * size.inHeight * c, 0, 255);
        cv::Mat src_opencv(size.inHeight, size.inWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), src.get(), sizeof(T) * size.inWidth * c);
        cv::Mat dst_opencv(size.outHeight, size.outWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), dst_ref.get(), sizeof(T) * size.outWidth * c);

        cv::resize(src_opencv, dst_opencv, cv::Size(size.outWidth, size.outHeight), 0, 0, cv::INTER_NEAREST);

        ppl::cv::riscv::ResizeNearestPoint<T, c>(
            size.inHeight,
            size.inWidth,
            size.inWidth * c,
            src.get(),
            size.outHeight,
            size.outWidth,
            size.outWidth * c,
            dst.get());

        checkResult<T, c>(
            dst_ref.get(),
            dst.get(),
            size.outHeight,
            size.outWidth,
            size.outWidth * c,
            size.outWidth * c,
            diff);
    }

    void Areaapply(const ResizeParam &param)
    {
        Size_p size = std::get<0>(param);
        const float diff = std::get<1>(param);
        std::cout << __FUNCTION__ << std::endl;

        std::unique_ptr<T[]> src(new T[size.inWidth * size.inHeight * c]);
        std::unique_ptr<T[]> dst_ref(new T[size.outWidth * size.outHeight * c]);
        std::unique_ptr<T[]> dst(new T[size.outWidth * size.outHeight * c]);

        ppl::cv::debug::randomFill<T>(src.get(), size.inWidth * size.inHeight * c, 0, 255);
        cv::Mat src_opencv(size.inHeight, size.inWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), src.get(), sizeof(T) * size.inWidth * c);
        cv::Mat dst_opencv(size.outHeight, size.outWidth, CV_MAKETYPE(cv::DataType<T>::depth, c), dst_ref.get(), sizeof(T) * size.outWidth * c);

        cv::resize(src_opencv, dst_opencv, cv::Size(size.outWidth, size.outHeight), 0, 0, cv::INTER_AREA);

        ppl::cv::riscv::ResizeArea<T, c>(
            size.inHeight,
            size.inWidth,
            size.inWidth * c,
            src.get(),
            size.outHeight,
            size.outWidth,
            size.outWidth * c,
            dst.get());

        checkResult<T, c>(
            dst_ref.get(),
            dst.get(),
            size.outHeight,
            size.outWidth,
            size.outWidth * c,
            size.outWidth * c,
            diff);
    }
};

#define R1(name, t, c, diff)           \
    using name = Resize<t, c>;         \
    TEST_P(name, abc)                  \
    {                                  \
        this->Linearapply(GetParam()); \
    }                                  \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size_p{320, 240, 640, 480}, Size_p{640, 480, 320, 240}, Size_p{800, 600, 1280, 720}, Size_p{1280, 720, 800, 600}, Size_p{1080, 1920, 270, 480}, Size_p{1080, 1920, 180, 320}), ::testing::Values(diff)));
R1(ResizeLinear_f32c1, float, 1, 1e-1f)
R1(ResizeLinear_f32c3, float, 3, 1e-1f)
R1(ResizeLinear_f32c4, float, 4, 1e-1f)

R1(ResizeLinear_u8c1, uint8_t, 1, 2.01f)
R1(ResizeLinear_u8c3, uint8_t, 3, 1.01f)
R1(ResizeLinear_u8c4, uint8_t, 4, 2.01f)

#define R2(name, t, c, diff)                 \
    using name = Resize<t, c>;               \
    TEST_P(name, abc)                        \
    {                                        \
        this->NearestPointapply(GetParam()); \
    }                                        \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size_p{320, 240, 640, 480}, Size_p{640, 480, 320, 240}, Size_p{800, 600, 1280, 720}, Size_p{1280, 720, 800, 600}), ::testing::Values(diff)));
R2(ResizeNearestPoint_f32c1, float, 1, 1e-5f)
R2(ResizeNearestPoint_f32c3, float, 3, 1e-5f)
R2(ResizeNearestPoint_f32c4, float, 4, 1e-5f)

R2(ResizeNearestPoint_u8c1, uint8_t, 1, 1.01f)
R2(ResizeNearestPoint_u8c3, uint8_t, 3, 1.01f)
R2(ResizeNearestPoint_u8c4, uint8_t, 4, 1.01f)

#define R3(name, t, c, diff)         \
    using name = Resize<t, c>;       \
    TEST_P(name, abc)                \
    {                                \
        this->Areaapply(GetParam()); \
    }                                \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size_p{320, 240, 640, 480}, Size_p{640, 480, 320, 240}, Size_p{800, 600, 1280, 720}, Size_p{1280, 720, 800, 600}), ::testing::Values(diff)));

R3(ResizeArea_u8c1, uint8_t, 1, 1.01f)
R3(ResizeArea_u8c3, uint8_t, 3, 1.01f)
R3(ResizeArea_u8c4, uint8_t, 4, 1.01f)
R3(ResizeArea_f32c1, float, 1, 1.01f)
R3(ResizeArea_f32c3, float, 3, 1.01f)
R3(ResizeArea_f32c4, float, 4, 1.01f)