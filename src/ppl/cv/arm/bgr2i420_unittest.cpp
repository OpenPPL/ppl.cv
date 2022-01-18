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
#include "ppl/cv/arm/cvtcolor.h"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "ppl/cv/debug.h"
#include "ppl/cv/arm/test.h"

template <typename T, int32_t input_channels, int32_t output_channels>
class BGR2I420 : public ::testing::TestWithParam<std::tuple<Size, float, int32_t>> {
public:
    using BGR2I420Param = std::tuple<Size, float, int32_t>;
    BGR2I420()
    {
    }

    ~BGR2I420()
    {
    }

    void BGR2I420apply(const BGR2I420Param &param)
    {
        Size size          = std::get<0>(param);
        const float diff   = std::get<1>(param);
        const int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 3]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * 3 / 2]);

        ppl::cv::debug::randomFill<uint8_t>(src.get(), size.width * size.height * 3, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get());
        cv::Mat dst_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get());

        if (mode == 0) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2YUV_I420);

            ppl::cv::arm::BGR2I420<T>(
                size.height,
                size.width,
                size.width * 3,
                src.get(),
                size.width * 1,
                dst.get());
        } else if (mode == 2) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2YUV_I420);

            ppl::cv::arm::RGB2I420<T>(
                size.height,
                size.width,
                size.width * 3,
                src.get(),
                size.width * 1,
                dst.get());
        } else if (mode == 1) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2YUV_I420);

            ppl::cv::arm::BGR2I420<T>(
                size.height,
                size.width,
                size.width * 3,
                src.get(),
                size.width * 1,
                dst.get(),
                size.width / 2,
                dst.get() + size.height * size.width,
                size.width / 2,
                dst.get() + size.height * size.width + (size.height / 2) * (size.width / 2));
        } else if (mode == 3) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2YUV_I420);

            ppl::cv::arm::RGB2I420<T>(
                size.height,
                size.width,
                size.width * 3,
                src.get(),
                size.width * 1,
                dst.get(),
                size.width / 2,
                dst.get() + size.height * size.width,
                size.width / 2,
                dst.get() + size.height * size.width + (size.height / 2) * (size.width / 2));
        }

#ifdef USE_QUANTIZED
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height * 3 / 2,
            size.width,
            size.width * 1,
            size.width * 1,
            diff,
            0.05);
#else
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height * 3 / 2,
            size.width,
            size.width * 1,
            size.width * 1,
            diff);
#endif
    }

    void I4202BGRapply(const BGR2I420Param &param)
    {
        Size size          = std::get<0>(param);
        const float diff   = std::get<1>(param);
        const int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 3 / 2]); //?
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * 3]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * 3]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * 3 / 2, 0, 255);

        cv::Mat src_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        if (mode == 0) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_I420);

            ppl::cv::arm::I4202BGR<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        } else if (mode == 2) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2RGB_I420);

            ppl::cv::arm::I4202RGB<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        } else if (mode == 1) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_I420);

            ppl::cv::arm::I4202BGR<T>(
                size.height,
                size.width,
                size.width,
                src.get(),
                size.width / 2,
                src.get() + size.height * size.width,
                size.width / 2,
                src.get() + size.height * size.width + (size.height / 2) * (size.width / 2),
                size.width * 3,
                dst.get());
        } else if (mode == 3) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2RGB_I420);

            ppl::cv::arm::I4202RGB<T>(
                size.height,
                size.width,
                size.width,
                src.get(),
                size.width / 2,
                src.get() + size.height * size.width,
                size.width / 2,
                src.get() + size.height * size.width + (size.height / 2) * (size.width / 2),
                size.width * 3,
                dst.get());
        }

#ifdef USE_QUANTIZED
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff,
            0.05);
#else
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff);
#endif
    }

    void BGRA2I420apply(const BGR2I420Param &param)
    {
        Size size          = std::get<0>(param);
        const float diff   = std::get<1>(param);
        const int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * input_channels]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * 3 / 2]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * input_channels, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        if (mode == 0) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2YUV_I420);

            ppl::cv::arm::BGRA2I420<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        } else if (mode == 2) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2YUV_I420);

            ppl::cv::arm::RGBA2I420<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        } else if (mode == 1) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2YUV_I420);

            ppl::cv::arm::BGRA2I420<T>(
                size.height,
                size.width,
                size.width * 4,
                src.get(),
                size.width * 1,
                dst.get(),
                size.width / 2,
                dst.get() + size.height * size.width,
                size.width / 2,
                dst.get() + size.height * size.width + (size.height / 2) * (size.width / 2));
        } else if (mode == 3) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2YUV_I420);

            ppl::cv::arm::RGBA2I420<T>(
                size.height,
                size.width,
                size.width * 4,
                src.get(),
                size.width * 1,
                dst.get(),
                size.width / 2,
                dst.get() + size.height * size.width,
                size.width / 2,
                dst.get() + size.height * size.width + (size.height / 2) * (size.width / 2));
        }

#ifdef USE_QUANTIZED
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height * 3 / 2,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff,
            0.05);
#else
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height * 3 / 2,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff);
#endif
    }

    void I4202BGRAapply(const BGR2I420Param &param)
    {
        Size size          = std::get<0>(param);
        const float diff   = std::get<1>(param);
        const int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * 3 / 2, 0, 255);

        cv::Mat src_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        if (mode == 0) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGRA_I420);

            ppl::cv::arm::I4202BGRA<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        } else if (mode == 2) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2RGBA_I420);

            ppl::cv::arm::I4202RGBA<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        } else if (mode == 1) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGRA_I420);

            ppl::cv::arm::I4202BGRA<T>(
                size.height,
                size.width,
                size.width,
                src.get(),
                size.width / 2,
                src.get() + size.height * size.width,
                size.width / 2,
                src.get() + size.height * size.width + (size.height / 2) * (size.width / 2),
                size.width * 4,
                dst.get());
        } else if (mode == 3) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2RGBA_I420);

            ppl::cv::arm::I4202RGBA<T>(
                size.height,
                size.width,
                size.width,
                src.get(),
                size.width / 2,
                src.get() + size.height * size.width,
                size.width / 2,
                src.get() + size.height * size.width + (size.height / 2) * (size.width / 2),
                size.width * 4,
                dst.get());
        }

#ifdef USE_QUANTIZED
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff,
            0.05);
#else
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff);
#endif
    }

    void BGR2YV12apply(const BGR2I420Param &param)
    {
        Size size          = std::get<0>(param);
        const float diff   = std::get<1>(param);
        const int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 3]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * 3 / 2]);

        ppl::cv::debug::randomFill<uint8_t>(src.get(), size.width * size.height * 3, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get());
        cv::Mat dst_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get());

        if (mode == 0) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2YUV_YV12);

            ppl::cv::arm::BGR2YV12<T>(
                size.height,
                size.width,
                size.width * 3,
                src.get(),
                size.width * 1,
                dst.get());
        } else {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2YUV_YV12);

            ppl::cv::arm::RGB2YV12<T>(
                size.height,
                size.width,
                size.width * 3,
                src.get(),
                size.width * 1,
                dst.get());
        }

#ifdef USE_QUANTIZED
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height * 3 / 2,
            size.width,
            size.width * 1,
            size.width * 1,
            diff,
            0.05);
#else
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height * 3 / 2,
            size.width,
            size.width * 1,
            size.width * 1,
            diff);
#endif
    }

    void YV122BGRapply(const BGR2I420Param &param)
    {
        Size size          = std::get<0>(param);
        const float diff   = std::get<1>(param);
        const int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 3 / 2]); //?
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * 3]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * 3]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * 3 / 2, 0, 255);

        cv::Mat src_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        if (mode == 0) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_YV12);

            ppl::cv::arm::YV122BGR<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        } else {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2RGB_YV12);

            ppl::cv::arm::YV122RGB<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        }

#ifdef USE_QUANTIZED
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff,
            0.05);
#else
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff);
#endif
    }

    void BGRA2YV12apply(const BGR2I420Param &param)
    {
        Size size          = std::get<0>(param);
        const float diff   = std::get<1>(param);
        const int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 4]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * 3 / 2]);

        ppl::cv::debug::randomFill<uint8_t>(src.get(), size.width * size.height * 4, 0, 255);

        cv::Mat src_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get());
        cv::Mat dst_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get());

        if (mode == 0) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2YUV_YV12);

            ppl::cv::arm::BGRA2YV12<T>(
                size.height,
                size.width,
                size.width * 4,
                src.get(),
                size.width * 1,
                dst.get());
        } else {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2YUV_YV12);

            ppl::cv::arm::RGBA2YV12<T>(
                size.height,
                size.width,
                size.width * 4,
                src.get(),
                size.width * 1,
                dst.get());
        }

#ifdef USE_QUANTIZED
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height * 3 / 2,
            size.width,
            size.width * 1,
            size.width * 1,
            diff,
            0.05);
#else
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height * 3 / 2,
            size.width,
            size.width * 1,
            size.width * 1,
            diff);
#endif
    }

    void YV122BGRAapply(const BGR2I420Param &param)
    {
        Size size          = std::get<0>(param);
        const float diff   = std::get<1>(param);
        const int32_t mode = std::get<2>(param);

        std::unique_ptr<T[]> src(new T[size.width * size.height * 3 / 2]);
        std::unique_ptr<T[]> dst_ref(new T[size.width * size.height * output_channels]);
        std::unique_ptr<T[]> dst(new T[size.width * size.height * output_channels]);

        ppl::cv::debug::randomFill<T>(src.get(), size.width * size.height * 3 / 2, 0, 255);

        cv::Mat src_opencv(size.height * 3 / 2, size.width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), src.get(), sizeof(T) * size.width * input_channels);
        cv::Mat dst_opencv(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dst_ref.get(), sizeof(T) * size.width * output_channels);

        if (mode == 0) {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGRA_YV12);

            ppl::cv::arm::YV122BGRA<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        } else {
            cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2RGBA_YV12);

            ppl::cv::arm::YV122RGBA<T>(
                size.height,
                size.width,
                size.width * input_channels,
                src.get(),
                size.width * output_channels,
                dst.get());
        }

#ifdef USE_QUANTIZED
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff,
            0.05);
#else
        checkResult<T, output_channels>(
            dst_ref.get(),
            dst.get(),
            size.height,
            size.width,
            size.width * output_channels,
            size.width * output_channels,
            diff);
#endif
    }
};

constexpr int32_t c1 = 1;
constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

#define R1(name, t, ic, oc, diff, mode)  \
    using name = BGR2I420<t, ic, oc>;    \
    TEST_P(name, abc)                    \
    {                                    \
        this->BGR2I420apply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));
R1(UT_BGR2I420_uchar_aarch64, uint8_t, c3, c1, 2.01f, 0)
R1(UT_RGB2I420_uchar_aarch64, uint8_t, c3, c1, 2.01f, 2)
R1(UT_BGR2I420MultiPlane_uchar_aarch64, uint8_t, c3, c1, 2.01f, 1)
R1(UT_RGB2I420MultiPlane_uchar_aarch64, uint8_t, c3, c1, 2.01f, 3)

#define R2(name, t, ic, oc, diff, mode)  \
    using name = BGR2I420<t, ic, oc>;    \
    TEST_P(name, abc)                    \
    {                                    \
        this->I4202BGRapply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));
R2(UT_I4202BGR_uchar_aarch64, uint8_t, c1, c3, 2.01f, 0)
R2(UT_I4202RGB_uchar_aarch64, uint8_t, c1, c3, 2.01f, 2)
R2(UT_I4202BGRMultiPlane_uchar_aarch64, uint8_t, c1, c3, 2.01f, 1)
R2(UT_I4202RGBMultiPlane_uchar_aarch64, uint8_t, c1, c3, 2.01f, 3)

#define R3(name, t, ic, oc, diff, mode)   \
    using name = BGR2I420<t, ic, oc>;     \
    TEST_P(name, abc)                     \
    {                                     \
        this->BGRA2I420apply(GetParam()); \
    }                                     \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));
R3(UT_BGRA2I420_uchar_aarch64, uint8_t, c4, c1, 2.01f, 0)
R3(UT_RGBA2I420_uchar_aarch64, uint8_t, c4, c1, 2.01f, 2)
R3(UT_BGRA2I420MultiPlane_uchar_aarch64, uint8_t, c4, c1, 2.01f, 1)
R3(UT_RGBA2I420MultiPlane_uchar_aarch64, uint8_t, c4, c1, 2.01f, 3)

#define R4(name, t, ic, oc, diff, mode)   \
    using name = BGR2I420<t, ic, oc>;     \
    TEST_P(name, abc)                     \
    {                                     \
        this->I4202BGRAapply(GetParam()); \
    }                                     \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));
R4(UT_I4202BGRA_uchar_aarch64, uint8_t, c1, c4, 2.01f, 0)
R4(UT_I4202RGBA_uchar_aarch64, uint8_t, c1, c4, 2.01f, 2)
R4(UT_I4202BGRAMultiPlane_uchar_aarch64, uint8_t, c1, c4, 2.01f, 1)
R4(UT_I4202RGBAMultiPlane_uchar_aarch64, uint8_t, c1, c4, 2.01f, 3)

#define R5(name, t, ic, oc, diff, mode)  \
    using name = BGR2I420<t, ic, oc>;    \
    TEST_P(name, abc)                    \
    {                                    \
        this->BGR2YV12apply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));
R5(UT_BGR2YV12_uchar_aarch64, uint8_t, c3, c1, 2.01f, 0)
R5(UT_RGB2YV12_uchar_aarch64, uint8_t, c3, c1, 2.01f, 2)

#define R6(name, t, ic, oc, diff, mode)  \
    using name = BGR2I420<t, ic, oc>;    \
    TEST_P(name, abc)                    \
    {                                    \
        this->YV122BGRapply(GetParam()); \
    }                                    \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));
R6(UT_YV122BGR_uchar_aarch64, uint8_t, c1, c3, 2.01f, 0)
R6(UT_YV122RGB_uchar_aarch64, uint8_t, c1, c3, 2.01f, 2)

#define R7(name, t, ic, oc, diff, mode)   \
    using name = BGR2I420<t, ic, oc>;     \
    TEST_P(name, abc)                     \
    {                                     \
        this->BGRA2YV12apply(GetParam()); \
    }                                     \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));
R7(UT_BGRA2YV12_uchar_aarch64, uint8_t, c4, c1, 2.01f, 0)
R7(UT_RGBA2YV12_uchar_aarch64, uint8_t, c4, c1, 2.01f, 2)

#define R8(name, t, ic, oc, diff, mode)   \
    using name = BGR2I420<t, ic, oc>;     \
    TEST_P(name, abc)                     \
    {                                     \
        this->YV122BGRAapply(GetParam()); \
    }                                     \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{320, 256}, Size{720, 480}), ::testing::Values(diff), ::testing::Values(mode)));
R8(UT_YV122BGRA_uchar_aarch64, uint8_t, c1, c4, 2.01f, 0)
R8(UT_YV122RGBA_uchar_aarch64, uint8_t, c1, c4, 2.01f, 2)
