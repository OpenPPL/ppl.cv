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

#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

enum Color2GrayMode {BGR2GRAY_MODE, RGB2GRAY_MODE};
template<typename T, int32_t nc, Color2GrayMode mode>
void Color2GRAYTest(int32_t height, int32_t width) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height]);
    std::unique_ptr<T[]> dst(new T[width * height]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), dst_ref.get());
    if (nc == 3) {
        if (mode == BGR2GRAY_MODE) {
            ppl::cv::x86::BGR2GRAY<T>(height, width, width * nc, src.get(), width, dst.get());
            cv::cvtColor(srcMat, dstMat, cv::COLOR_BGR2GRAY);
        }
        if (mode == RGB2GRAY_MODE) {
            ppl::cv::x86::RGB2GRAY<T>(height, width, width * nc, src.get(), width, dst.get());
            cv::cvtColor(srcMat, dstMat, cv::COLOR_RGB2GRAY);
        }
    } else if (nc == 4) {
        if (mode == BGR2GRAY_MODE) {
            ppl::cv::x86::BGRA2GRAY<T>(height, width, width * nc, src.get(), width, dst.get());
            cv::cvtColor(srcMat, dstMat, cv::COLOR_BGRA2GRAY);
        }
        if (mode == RGB2GRAY_MODE) {
            ppl::cv::x86::RGBA2GRAY<T>(height, width, width * nc, src.get(), width, dst.get());
            cv::cvtColor(srcMat, dstMat, cv::COLOR_RGBA2GRAY);
        }
    }
    checkResult<T, 1>(dst.get(), dst_ref.get(), height, width, width, width, 1.01f);
}
	
enum Gray2ColorMode {GRAY2BGR_MODE, GRAY2RGB_MODE};
template<typename T, int32_t nc, Gray2ColorMode mode>
void GRAY2ColorTest(int32_t height, int32_t width) {
    std::unique_ptr<T[]> src(new T[width * height]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height, 0, 255);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get());
    if (nc == 3) {
        if (mode == GRAY2BGR_MODE) {
            ppl::cv::x86::GRAY2BGR<T>(height, width, width, src.get(), width * nc, dst.get());
            cv::cvtColor(srcMat, dstMat, cv::COLOR_GRAY2BGR);
        }
        if (mode == GRAY2RGB_MODE) {
            ppl::cv::x86::GRAY2RGB<T>(height, width, width, src.get(), width * nc, dst.get());
            cv::cvtColor(srcMat, dstMat, cv::COLOR_GRAY2RGB);
        }
    } else if (nc == 4) {
        if (mode == GRAY2BGR_MODE) {
            ppl::cv::x86::GRAY2BGRA<T>(height, width, width, src.get(), width * nc, dst.get());
            cv::cvtColor(srcMat, dstMat, cv::COLOR_GRAY2BGRA);
        }
        if (mode == GRAY2RGB_MODE) {
            ppl::cv::x86::GRAY2RGBA<T>(height, width, width, src.get(), width * nc, dst.get());
            cv::cvtColor(srcMat, dstMat, cv::COLOR_GRAY2RGBA);
        }
    }
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

TEST(RGB2GRAY_FP32, x86)
{
    Color2GRAYTest<float, 3, RGB2GRAY_MODE>(640, 720);
    Color2GRAYTest<float, 3, RGB2GRAY_MODE>(720, 1080);
    Color2GRAYTest<float, 4, RGB2GRAY_MODE>(640, 720);
    Color2GRAYTest<float, 4, RGB2GRAY_MODE>(720, 1080);
}

TEST(BGR2GRAY_FP32, x86)
{
    Color2GRAYTest<float, 3, BGR2GRAY_MODE>(640, 720);
    Color2GRAYTest<float, 3, BGR2GRAY_MODE>(720, 1080);
    Color2GRAYTest<float, 4, BGR2GRAY_MODE>(640, 720);
    Color2GRAYTest<float, 4, BGR2GRAY_MODE>(720, 1080);
}

TEST(RGB2GRAY_UINT8, x86)
{
    Color2GRAYTest<uint8_t, 3, RGB2GRAY_MODE>(640, 720);
    Color2GRAYTest<uint8_t, 3, RGB2GRAY_MODE>(720, 1080);
    Color2GRAYTest<uint8_t, 4, RGB2GRAY_MODE>(640, 720);
    Color2GRAYTest<uint8_t, 4, RGB2GRAY_MODE>(720, 1080);
}

TEST(BGR2GRAY_UINT8, x86)
{
    Color2GRAYTest<uint8_t, 3, BGR2GRAY_MODE>(640, 720);
    Color2GRAYTest<uint8_t, 3, BGR2GRAY_MODE>(720, 1080);
    Color2GRAYTest<uint8_t, 4, BGR2GRAY_MODE>(640, 720);
    Color2GRAYTest<uint8_t, 4, BGR2GRAY_MODE>(720, 1080);
}

TEST(GRAY2RGB_FP32, x86)
{
    GRAY2ColorTest<float, 3, GRAY2RGB_MODE>(640, 720);
    GRAY2ColorTest<float, 3, GRAY2RGB_MODE>(720, 1080);
    GRAY2ColorTest<float, 4, GRAY2RGB_MODE>(640, 720);
    GRAY2ColorTest<float, 4, GRAY2RGB_MODE>(720, 1080);
}

TEST(GRAY2BGR_FP32, x86)
{
    GRAY2ColorTest<float, 3, GRAY2BGR_MODE>(640, 720);
    GRAY2ColorTest<float, 3, GRAY2BGR_MODE>(720, 1080);
    GRAY2ColorTest<float, 4, GRAY2BGR_MODE>(640, 720);
    GRAY2ColorTest<float, 4, GRAY2BGR_MODE>(720, 1080);
}

TEST(GRAY2RGB_UINT8, x86)
{
    GRAY2ColorTest<uint8_t, 3, GRAY2RGB_MODE>(640, 720);
    GRAY2ColorTest<uint8_t, 3, GRAY2RGB_MODE>(720, 1080);
    GRAY2ColorTest<uint8_t, 4, GRAY2RGB_MODE>(640, 720);
    GRAY2ColorTest<uint8_t, 4, GRAY2RGB_MODE>(720, 1080);
}

TEST(GRAY2BGR_UINT8, x86)
{
    GRAY2ColorTest<uint8_t, 3, GRAY2BGR_MODE>(640, 720);
    GRAY2ColorTest<uint8_t, 3, GRAY2BGR_MODE>(720, 1080);
    GRAY2ColorTest<uint8_t, 4, GRAY2BGR_MODE>(640, 720);
    GRAY2ColorTest<uint8_t, 4, GRAY2BGR_MODE>(720, 1080);
}

