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

#include "ppl/cv/riscv/cvtcolor.h"
#include "ppl/cv/riscv/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

enum Color2YUV420Mode { RGB2I420_MODE,
                        BGR2I420_MODE,
                        RGB2YV12_MODE,
                        BGR2YV12_MODE };

template <Color2YUV420Mode mode>
void Color2YUV420Test(int32_t height, int32_t width)
{
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 3]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 3 / 2]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 3, 0, 255);

    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 3), src.get());
    cv::Mat dstMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), dst_ref.get());
    if (mode == RGB2I420_MODE) {
        ppl::cv::riscv::RGB2I420<uint8_t>(height, width, width * 3, src.get(), width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_RGB2YUV_I420);
    } else if (mode == RGB2YV12_MODE) {
        ppl::cv::riscv::RGB2YV12<uint8_t>(height, width, width * 3, src.get(), width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_RGB2YUV_YV12);
    } else if (mode == BGR2I420_MODE) {
        ppl::cv::riscv::BGR2I420<uint8_t>(height, width, width * 3, src.get(), width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_BGR2YUV_I420);
    } else if (mode == BGR2YV12_MODE) {
        ppl::cv::riscv::BGR2YV12<uint8_t>(height, width, width * 3, src.get(), width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_BGR2YUV_YV12);
    }
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), 3 * height / 2, width, width, width, 1.01f);
}

template <Color2YUV420Mode mode>
void Color2YUV420MultiPlaneTest(int32_t height, int32_t width)
{
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 3]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 3 / 2]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 3, 0, 255);

    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 3), src.get());
    cv::Mat dstMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), dst_ref.get());
    if (mode == RGB2I420_MODE) {
        ppl::cv::riscv::RGB2I420<uint8_t>(height, width, width * 3, src.get(), width, dst.get(), width / 2, dst.get() + height * width, width / 2, dst.get() + height * width + (height / 2) * (width / 2));
        cv::cvtColor(srcMat, dstMat, cv::COLOR_RGB2YUV_I420);
    } else if (mode == BGR2I420_MODE) {
        ppl::cv::riscv::BGR2I420<uint8_t>(height, width, width * 3, src.get(), width, dst.get(), width / 2, dst.get() + height * width, width / 2, dst.get() + height * width + (height / 2) * (width / 2));
        cv::cvtColor(srcMat, dstMat, cv::COLOR_BGR2YUV_I420);
    }
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), 3 * height / 2, width, width, width, 1.01f);
}

enum ColorAlpha2YUV420Mode { RGBA2I420_MODE,
                             BGRA2I420_MODE,
                             RGBA2YV12_MODE,
                             BGRA2YV12_MODE };

template <ColorAlpha2YUV420Mode mode>
void ColorAlpha2YUV420Test(int32_t height, int32_t width)
{
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 4]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 3 / 2]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 4, 0, 255);

    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 4), src.get());
    cv::Mat dstMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), dst_ref.get());
    if (mode == RGBA2I420_MODE) {
        ppl::cv::riscv::RGBA2I420<uint8_t>(height, width, width * 4, src.get(), width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_RGBA2YUV_I420);
    } else if (mode == RGBA2YV12_MODE) {
        ppl::cv::riscv::RGBA2YV12<uint8_t>(height, width, width * 4, src.get(), width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_RGBA2YUV_YV12);
    } else if (mode == BGRA2I420_MODE) {
        ppl::cv::riscv::BGRA2I420<uint8_t>(height, width, width * 4, src.get(), width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_BGRA2YUV_I420);
    } else if (mode == BGRA2YV12_MODE) {
        ppl::cv::riscv::BGRA2YV12<uint8_t>(height, width, width * 4, src.get(), width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_BGRA2YUV_YV12);
    }
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), 3 * height / 2, width, width, width, 1.01f);
}

template <ColorAlpha2YUV420Mode mode>
void ColorAlpha2YUV420MultiPlaneTest(int32_t height, int32_t width)
{
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 4]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 3 / 2]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 4, 0, 255);

    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 4), src.get());
    cv::Mat dstMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), dst_ref.get());
    if (mode == RGBA2I420_MODE) {
        ppl::cv::riscv::RGBA2I420<uint8_t>(height, width, width * 4, src.get(), width, dst.get(), width / 2, dst.get() + height * width, width / 2, dst.get() + height * width + (height / 2) * (width / 2));
        cv::cvtColor(srcMat, dstMat, cv::COLOR_RGBA2YUV_I420);
    } else if (mode == BGRA2I420_MODE) {
        ppl::cv::riscv::BGRA2I420<uint8_t>(height, width, width * 4, src.get(), width, dst.get(), width / 2, dst.get() + height * width, width / 2, dst.get() + height * width + (height / 2) * (width / 2));
        cv::cvtColor(srcMat, dstMat, cv::COLOR_BGRA2YUV_I420);
    }
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), 3 * height / 2, width, width, width, 1.01f);
}

enum YUV4202ColorMode { I4202RGB_MODE,
                        I4202BGR_MODE,
                        YV122RGB_MODE,
                        YV122BGR_MODE };

template <YUV4202ColorMode mode>
void YUV4202ColorTest(int32_t height, int32_t width)
{
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 3]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * 3]);

    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 3 / 2, 0, 255);

    cv::Mat srcMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 3), dst_ref.get());
    if (mode == I4202RGB_MODE) {
        ppl::cv::riscv::I4202RGB<uint8_t>(height, width, width, src.get(), 3 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2RGB_I420);
    } else if (mode == I4202BGR_MODE) {
        ppl::cv::riscv::I4202BGR<uint8_t>(height, width, width, src.get(), 3 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2BGR_I420);
    } else if (mode == YV122RGB_MODE) {
        ppl::cv::riscv::YV122RGB<uint8_t>(height, width, width, src.get(), 3 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2RGB_YV12);
    } else if (mode == YV122BGR_MODE) {
        ppl::cv::riscv::YV122BGR<uint8_t>(height, width, width, src.get(), 3 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2BGR_YV12);
    }
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), height, width, 3 * width, 3 * width, 1.01f);
}

template <YUV4202ColorMode mode>
void YUV4202ColorMultiPlaneTest(int32_t height, int32_t width)
{
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 3]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * 3]);

    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 3 / 2, 0, 255);

    cv::Mat srcMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 3), dst_ref.get());
    if (mode == I4202RGB_MODE) {
        ppl::cv::riscv::I4202RGB<uint8_t>(height, width, width, src.get(), width / 2, src.get() + height * width, width / 2, src.get() + height * width + (height / 2) * (width / 2), 3 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2RGB_I420);
    } else if (mode == I4202BGR_MODE) {
        ppl::cv::riscv::I4202BGR<uint8_t>(height, width, width, src.get(), width / 2, src.get() + height * width, width / 2, src.get() + height * width + (height / 2) * (width / 2), 3 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2BGR_I420);
    }
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), height, width, 3 * width, 3 * width, 1.01f);
}

enum YUV4202ColorAlphaMode { I4202RGBA_MODE,
                             I4202BGRA_MODE,
                             YV122RGBA_MODE,
                             YV122BGRA_MODE };

template <YUV4202ColorAlphaMode mode>
void YUV4202ColorAlphaTest(int32_t height, int32_t width)
{
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 4]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * 4]);

    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 3 / 2, 0, 255);

    cv::Mat srcMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 4), dst_ref.get());
    if (mode == I4202RGBA_MODE) {
        ppl::cv::riscv::I4202RGBA<uint8_t>(height, width, width, src.get(), 4 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2RGBA_I420);
    } else if (mode == I4202BGRA_MODE) {
        ppl::cv::riscv::I4202BGRA<uint8_t>(height, width, width, src.get(), 4 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2BGRA_I420);
    } else if (mode == YV122RGBA_MODE) {
        ppl::cv::riscv::YV122RGBA<uint8_t>(height, width, width, src.get(), 4 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2RGBA_YV12);
    } else if (mode == YV122BGRA_MODE) {
        ppl::cv::riscv::YV122BGRA<uint8_t>(height, width, width, src.get(), 4 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2BGRA_YV12);
    }
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), height, width, 4 * width, 4 * width, 1.01f);
}

template <YUV4202ColorAlphaMode mode>
void YUV4202ColorAlphaMultiPlaneTest(int32_t height, int32_t width)
{
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * 3 / 2]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * 4]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * 4]);

    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * 3 / 2, 0, 255);

    cv::Mat srcMat(3 * height / 2, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 4), dst_ref.get());
    if (mode == I4202RGBA_MODE) {
        ppl::cv::riscv::I4202RGBA<uint8_t>(height, width, width, src.get(), width / 2, src.get() + height * width, width / 2, src.get() + height * width + (height / 2) * (width / 2), 4 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2RGBA_I420);
    } else if (mode == I4202BGRA_MODE) {
        ppl::cv::riscv::I4202BGRA<uint8_t>(height, width, width, src.get(), width / 2, src.get() + height * width, width / 2, src.get() + height * width + (height / 2) * (width / 2), 4 * width, dst.get());
        cv::cvtColor(srcMat, dstMat, cv::COLOR_YUV2BGRA_I420);
    }
    checkResult<uint8_t, 1>(dst.get(), dst_ref.get(), height, width, 4 * width, 4 * width, 1.01f);
}

TEST(RGB2YV12, riscv)
{
    Color2YUV420Test<RGB2YV12_MODE>(640, 720);
    Color2YUV420Test<RGB2YV12_MODE>(720, 1080);
    Color2YUV420Test<RGB2YV12_MODE>(1080, 1920);
}

TEST(BGR2YV12, riscv)
{
    Color2YUV420Test<BGR2YV12_MODE>(640, 720);
    Color2YUV420Test<BGR2YV12_MODE>(720, 1080);
    Color2YUV420Test<BGR2YV12_MODE>(1080, 1920);
}

TEST(RGB2I420, riscv)
{
    Color2YUV420Test<RGB2I420_MODE>(640, 720);
    Color2YUV420Test<RGB2I420_MODE>(720, 1080);
    Color2YUV420Test<RGB2I420_MODE>(1080, 1920);
    Color2YUV420MultiPlaneTest<RGB2I420_MODE>(640, 720);
    Color2YUV420MultiPlaneTest<RGB2I420_MODE>(720, 1080);
    Color2YUV420MultiPlaneTest<RGB2I420_MODE>(1080, 1920);
}

TEST(BGR2I420, riscv)
{
    Color2YUV420Test<BGR2I420_MODE>(32, 32);
    Color2YUV420Test<BGR2I420_MODE>(640, 720);
    Color2YUV420Test<BGR2I420_MODE>(720, 1080);
    Color2YUV420Test<BGR2I420_MODE>(1080, 1920);
    Color2YUV420MultiPlaneTest<BGR2I420_MODE>(640, 720);
    Color2YUV420MultiPlaneTest<BGR2I420_MODE>(720, 1080);
    Color2YUV420MultiPlaneTest<BGR2I420_MODE>(1080, 1920);
}

TEST(RGBA2YV12, riscv)
{
    ColorAlpha2YUV420Test<RGBA2YV12_MODE>(640, 720);
    ColorAlpha2YUV420Test<RGBA2YV12_MODE>(720, 1080);
    ColorAlpha2YUV420Test<RGBA2YV12_MODE>(1080, 1920);
}

TEST(BGRA2YV12, riscv)
{
    ColorAlpha2YUV420Test<BGRA2YV12_MODE>(640, 720);
    ColorAlpha2YUV420Test<BGRA2YV12_MODE>(720, 1080);
    ColorAlpha2YUV420Test<BGRA2YV12_MODE>(1080, 1920);
}

TEST(RGBA2I420, riscv)
{
    ColorAlpha2YUV420Test<RGBA2I420_MODE>(640, 720);
    ColorAlpha2YUV420Test<RGBA2I420_MODE>(720, 1080);
    ColorAlpha2YUV420Test<RGBA2I420_MODE>(1080, 1920);
    ColorAlpha2YUV420MultiPlaneTest<RGBA2I420_MODE>(640, 720);
    ColorAlpha2YUV420MultiPlaneTest<RGBA2I420_MODE>(720, 1080);
    ColorAlpha2YUV420MultiPlaneTest<RGBA2I420_MODE>(1080, 1920);
}

TEST(BGRA2I420, riscv)
{
    ColorAlpha2YUV420Test<BGRA2I420_MODE>(640, 720);
    ColorAlpha2YUV420Test<BGRA2I420_MODE>(720, 1080);
    ColorAlpha2YUV420Test<BGRA2I420_MODE>(1080, 1920);
    ColorAlpha2YUV420MultiPlaneTest<BGRA2I420_MODE>(640, 720);
    ColorAlpha2YUV420MultiPlaneTest<BGRA2I420_MODE>(720, 1080);
    ColorAlpha2YUV420MultiPlaneTest<BGRA2I420_MODE>(1080, 1920);
}

TEST(YV122RGB, riscv)
{
    YUV4202ColorTest<YV122RGB_MODE>(640, 720);
    YUV4202ColorTest<YV122RGB_MODE>(720, 1080);
    YUV4202ColorTest<YV122RGB_MODE>(1080, 1920);
    YUV4202ColorTest<YV122RGB_MODE>(642, 722);
    YUV4202ColorTest<YV122RGB_MODE>(722, 1082);
    YUV4202ColorTest<YV122RGB_MODE>(1082, 1922);
}

TEST(YV122BGR, riscv)
{
    YUV4202ColorTest<YV122BGR_MODE>(640, 720);
    YUV4202ColorTest<YV122BGR_MODE>(720, 1080);
    YUV4202ColorTest<YV122BGR_MODE>(1080, 1920);
    YUV4202ColorTest<YV122BGR_MODE>(642, 722);
    YUV4202ColorTest<YV122BGR_MODE>(722, 1082);
    YUV4202ColorTest<YV122BGR_MODE>(1082, 1922);
}

TEST(I4202RGB, riscv)
{
    YUV4202ColorTest<I4202RGB_MODE>(640, 720);
    YUV4202ColorTest<I4202RGB_MODE>(720, 1080);
    YUV4202ColorTest<I4202RGB_MODE>(1080, 1920);
    YUV4202ColorTest<I4202RGB_MODE>(642, 722);
    YUV4202ColorTest<I4202RGB_MODE>(722, 1082);
    YUV4202ColorTest<I4202RGB_MODE>(1082, 1922);
    YUV4202ColorMultiPlaneTest<I4202RGB_MODE>(640, 720);
    YUV4202ColorMultiPlaneTest<I4202RGB_MODE>(720, 1080);
    YUV4202ColorMultiPlaneTest<I4202RGB_MODE>(1080, 1920);
    YUV4202ColorMultiPlaneTest<I4202RGB_MODE>(642, 722);
    YUV4202ColorMultiPlaneTest<I4202RGB_MODE>(722, 1082);
    YUV4202ColorMultiPlaneTest<I4202RGB_MODE>(1082, 1922);
}

TEST(I4202BGR, riscv)
{
    YUV4202ColorTest<I4202BGR_MODE>(640, 720);
    YUV4202ColorTest<I4202BGR_MODE>(720, 1080);
    YUV4202ColorTest<I4202BGR_MODE>(1080, 1920);
    YUV4202ColorTest<I4202BGR_MODE>(642, 722);
    YUV4202ColorTest<I4202BGR_MODE>(722, 1082);
    YUV4202ColorTest<I4202BGR_MODE>(1082, 1922);
    YUV4202ColorMultiPlaneTest<I4202BGR_MODE>(642, 722);
    YUV4202ColorMultiPlaneTest<I4202BGR_MODE>(722, 1082);
    YUV4202ColorMultiPlaneTest<I4202BGR_MODE>(1082, 1922);
}

TEST(YV122RGBA, riscv)
{
    YUV4202ColorAlphaTest<YV122RGBA_MODE>(640, 720);
    YUV4202ColorAlphaTest<YV122RGBA_MODE>(720, 1080);
    YUV4202ColorAlphaTest<YV122RGBA_MODE>(1080, 1920);
}

TEST(YV122BGRA, riscv)
{
    YUV4202ColorAlphaTest<YV122BGRA_MODE>(640, 720);
    YUV4202ColorAlphaTest<YV122BGRA_MODE>(720, 1080);
    YUV4202ColorAlphaTest<YV122BGRA_MODE>(1080, 1920);
}

TEST(I4202RGBA, riscv)
{
    YUV4202ColorAlphaTest<I4202RGBA_MODE>(640, 720);
    YUV4202ColorAlphaTest<I4202RGBA_MODE>(720, 1080);
    YUV4202ColorAlphaTest<I4202RGBA_MODE>(1080, 1920);
    YUV4202ColorAlphaMultiPlaneTest<I4202RGBA_MODE>(640, 720);
    YUV4202ColorAlphaMultiPlaneTest<I4202RGBA_MODE>(720, 1080);
    YUV4202ColorAlphaMultiPlaneTest<I4202RGBA_MODE>(1080, 1920);
}

TEST(I4202BGRA, riscv)
{
    YUV4202ColorAlphaTest<I4202BGRA_MODE>(640, 720);
    YUV4202ColorAlphaTest<I4202BGRA_MODE>(720, 1080);
    YUV4202ColorAlphaTest<I4202BGRA_MODE>(1080, 1920);
    YUV4202ColorAlphaMultiPlaneTest<I4202BGRA_MODE>(640, 720);
    YUV4202ColorAlphaMultiPlaneTest<I4202BGRA_MODE>(720, 1080);
    YUV4202ColorAlphaMultiPlaneTest<I4202BGRA_MODE>(1080, 1920);
}
