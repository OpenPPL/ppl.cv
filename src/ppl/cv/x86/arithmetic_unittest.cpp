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

#include "ppl/cv/x86/arithmetic.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include <opencv2/imgproc.hpp>

template<typename T, int32_t nc>
void ADD_Test(int32_t height, int32_t width) {
    std::unique_ptr<T[]> src0(new T[width * height * nc]);
    std::unique_ptr<T[]> src1(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src0.get(), width * height * nc, 1, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), width * height * nc, 1, 255);
    cv::Mat adder0(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src0.get());
    cv::Mat adder1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src1.get());
    cv::Mat result(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get());
    cv::add(adder0, adder1, result);
    ppl::cv::x86::Add<T, nc>(height, width, width * nc, src0.get(), width * nc, src1.get(), width * nc, dst.get());
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

template<typename T, int32_t nc>
void SUB_Test(int32_t height, int32_t width) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> inScalar(new T[nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 1, 255);
    ppl::cv::debug::randomFill<T>(inScalar.get(), nc, 1, 255);
    cv::Mat suber(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get());
    cv::Mat result(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get());
    cv::Scalar scl;
    for (int32_t i = 0; i < nc; i++)
    {
        scl[i] = inScalar.get()[i];
    }
    cv::subtract(suber, scl, result);
    ppl::cv::x86::Subtract<T, nc>(height, width, width * nc, src.get(), inScalar.get(), width * nc, dst.get());
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

template<typename T, int32_t nc>
void MUL_Test(int32_t height, int32_t width) {
    std::unique_ptr<T[]> src0(new T[width * height * nc]);
    std::unique_ptr<T[]> src1(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src0.get(), width * height * nc, 1, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), width * height * nc, 1, 255);
    cv::Mat mulplier0(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src0.get());
    cv::Mat mulplier1(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src1.get());
    cv::Mat result(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get());
    cv::multiply(mulplier0, mulplier1, result);
    ppl::cv::x86::Mul<T, nc>(height, width, width * nc, src0.get(), width * nc, src1.get(), width * nc, dst.get());
    checkResult<T, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

template<int32_t nc>
void DIV_Test(int32_t height, int32_t width) {
    std::unique_ptr<float[]> src0(new float[width * height * nc]);
    std::unique_ptr<float[]> src1(new float[width * height * nc]);
    std::unique_ptr<float[]> dst_ref(new float[width * height * nc]);
    std::unique_ptr<float[]> dst(new float[width * height * nc]);
    ppl::cv::debug::randomFill<float>(src0.get(), width * height * nc, 1, 255);
    ppl::cv::debug::randomFill<float>(src1.get(), width * height * nc, 1, 255);
    cv::Mat div0(height, width, CV_MAKETYPE(cv::DataType<float>::depth, nc), src0.get());
    cv::Mat div1(height, width, CV_MAKETYPE(cv::DataType<float>::depth, nc), src1.get());
    cv::Mat result(height, width, CV_MAKETYPE(cv::DataType<float>::depth, nc), dst_ref.get());
    cv::divide(div0, div1, result);
    ppl::cv::x86::Div<float, nc>(height, width, width * nc, src0.get(), width * nc, src1.get(), width * nc, dst.get());
    checkResult<float, nc>(dst.get(), dst_ref.get(), height, width, width * nc, width * nc, 1.01f);
}

template <typename T>
void NaiveMla(T *out, const T *src0, const T *src1, int32_t len) {
    for (int32_t i = 0; i < len; ++i) {
        out[i] += src0[i] * src1[i];
    }
}

template<typename T>
void Mla_Test(int32_t len) {
    std::unique_ptr<T[]> src0(new T[len]);
    std::unique_ptr<T[]> src1(new T[len]);
    std::unique_ptr<T[]> dst(new T[len]);
    std::unique_ptr<T[]> dst_ref(new T[len]);
    ppl::cv::debug::randomFill<T>(src0.get(), len, 0, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), len, 0, 255);
    ppl::cv::debug::randomFill<T>(dst.get(), len, 0, 255);
    memcpy(dst_ref.get(), dst.get(), sizeof(T) * len);
    ppl::cv::x86::Mla<float, 1>(1, len, len, src0.get(), len, src1.get(), len, dst.get());
    NaiveMla(dst_ref.get(), src0.get(), src1.get(), len);
    checkResult<T, 1>(dst.get(), dst_ref.get(), 1, len, len, len, 1e-3);
}

template <typename T>
void NaiveMls(T *out, const T *src0, const T *src1, int32_t len) {
    for (int32_t i = 0; i < len; ++i) {
        out[i] -= src0[i] * src1[i];
    }
}

template<typename T>
void Mls_Test(int32_t len) {
    std::unique_ptr<T[]> src0(new T[len]);
    std::unique_ptr<T[]> src1(new T[len]);
    std::unique_ptr<T[]> dst(new T[len]);
    std::unique_ptr<T[]> dst_ref(new T[len]);
    ppl::cv::debug::randomFill<T>(src0.get(), len, 0, 255);
    ppl::cv::debug::randomFill<T>(src1.get(), len, 0, 255);
    ppl::cv::debug::randomFill<T>(dst.get(), len, 0, 255);
    memcpy(dst_ref.get(), dst.get(), sizeof(T) * len);
    ppl::cv::x86::Mls<float, 1>(1, len, len, src0.get(), len, src1.get(), len, dst.get());
    NaiveMls(dst_ref.get(), src0.get(), src1.get(), len);
    checkResult<T, 1>(dst.get(), dst_ref.get(), 1, len, len, len, 1e-3);
}

TEST(ADD_FP32, x86)
{
    ADD_Test<float, 1>(640, 720);
    ADD_Test<float, 3>(640, 720);
    ADD_Test<float, 4>(640, 720);

    ADD_Test<float, 1>(720, 1080);
    ADD_Test<float, 3>(720, 1080);
    ADD_Test<float, 4>(720, 1080);
}

TEST(ADD_UINT8, x86)
{
    ADD_Test<uint8_t, 1>(640, 720);
    ADD_Test<uint8_t, 3>(640, 720);
    ADD_Test<uint8_t, 4>(640, 720);

    ADD_Test<uint8_t, 1>(720, 1080);
    ADD_Test<uint8_t, 3>(720, 1080);
    ADD_Test<uint8_t, 4>(720, 1080);
}

TEST(SUB_FP32, x86)
{
    SUB_Test<float, 1>(640, 720);
    SUB_Test<float, 3>(640, 720);
    SUB_Test<float, 4>(640, 720);

    SUB_Test<float, 1>(720, 1080);
    SUB_Test<float, 3>(720, 1080);
    SUB_Test<float, 4>(720, 1080);
}

TEST(SUB_UINT8, x86)
{
    SUB_Test<uint8_t, 1>(640, 720);
    SUB_Test<uint8_t, 3>(640, 720);
    SUB_Test<uint8_t, 4>(640, 720);

    SUB_Test<uint8_t, 1>(720, 1080);
    SUB_Test<uint8_t, 3>(720, 1080);
    SUB_Test<uint8_t, 4>(720, 1080);
}

TEST(MUL_FP32, x86)
{
    MUL_Test<float, 1>(640, 720);
    MUL_Test<float, 3>(640, 720);
    MUL_Test<float, 4>(640, 720);

    MUL_Test<float, 1>(720, 1080);
    MUL_Test<float, 3>(720, 1080);
    MUL_Test<float, 4>(720, 1080);
}

TEST(MUL_UINT8, x86)
{
    MUL_Test<uint8_t, 1>(640, 720);
    MUL_Test<uint8_t, 3>(640, 720);
    MUL_Test<uint8_t, 4>(640, 720);

    MUL_Test<uint8_t, 1>(720, 1080);
    MUL_Test<uint8_t, 3>(720, 1080);
    MUL_Test<uint8_t, 4>(720, 1080);
}


TEST(DIV, x86)
{
    DIV_Test<1>(640, 720);
    DIV_Test<3>(640, 720);
    DIV_Test<4>(640, 720);

    DIV_Test<1>(720, 1080);
    DIV_Test<3>(720, 1080);
    DIV_Test<4>(720, 1080);
}

TEST(MlaTestFP32, x86)
{
    for (int32_t i = 1024; i < 1280; ++i) {
        Mla_Test<float>(i);
    }
}

TEST(MlsTestFP32, x86)
{
    for (int32_t i = 1024; i < 1280; ++i) {
        Mls_Test<float>(i);
    }
}
