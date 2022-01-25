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

#include "ppl/cv/x86/remap.h"
#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <typename T, int nc, ppl::cv::BorderType border_type, bool inter_linear>
void RemapTest(int inHeight, int inWidth, int outHeight, int outWidth, T diff)
{
    std::unique_ptr<T[]> src(new T[inWidth * inHeight * nc]);
    std::unique_ptr<T[]> dst_ref(new T[inWidth * inHeight * nc]);
    std::unique_ptr<T[]> dst(new T[outWidth * outHeight * nc]);
    std::unique_ptr<float[]> map_x(new float[outWidth * outHeight]);
    std::unique_ptr<float[]> map_y(new float[outWidth * outHeight]);
    ppl::cv::debug::randomFill<T>(src.get(), inWidth * inHeight * nc, 0, 255);
    memcpy(dst_ref.get(), dst.get(), outHeight * outWidth * nc * sizeof(T));
    ppl::cv::debug::randomFill<float>(map_x.get(), outWidth * outHeight, 1, inWidth - 2);
    ppl::cv::debug::randomFill<float>(map_y.get(), outWidth * outHeight, 1, inHeight - 2);

    // init mat
    cv::Mat srcMat(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * inWidth * nc);
    cv::Mat dstMat(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * outWidth * nc);

    cv::Mat yMat(outHeight, outWidth, CV_MAKETYPE(cv::DataType<float>::depth, 1), map_y.get(), sizeof(float) * outWidth);
    cv::Mat xMat(outHeight, outWidth, CV_MAKETYPE(cv::DataType<float>::depth, 1), map_x.get(), sizeof(float) * outWidth);

    if (inter_linear) {
        if (border_type == ppl::cv::BORDER_CONSTANT) {
            ppl::cv::x86::RemapLinear<T, nc>(inHeight, inWidth, inWidth * nc, src.get(), outHeight, outWidth, outWidth * nc, dst.get(), map_x.get(), map_y.get(), ppl::cv::BORDER_CONSTANT);
            cv::remap(srcMat, dstMat, xMat, yMat, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        } else if (border_type == ppl::cv::BORDER_REPLICATE) {
            ppl::cv::x86::RemapLinear<T, nc>(inHeight, inWidth, inWidth * nc, src.get(), outHeight, outWidth, outWidth * nc, dst.get(), map_x.get(), map_y.get(), ppl::cv::BORDER_REPLICATE);
            cv::remap(srcMat, dstMat, xMat, yMat, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
            ppl::cv::x86::RemapLinear<T, nc>(inHeight, inWidth, inWidth * nc, src.get(), outHeight, outWidth, outWidth * nc, dst.get(), map_x.get(), map_y.get(), ppl::cv::BORDER_TRANSPARENT);
            cv::remap(srcMat, dstMat, xMat, yMat, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        }
    } else {
        if (border_type == ppl::cv::BORDER_CONSTANT) {
            ppl::cv::x86::RemapNearestPoint<T, nc>(inHeight, inWidth, inWidth * nc, src.get(), outHeight, outWidth, outWidth * nc, dst.get(), map_x.get(), map_y.get(), ppl::cv::BORDER_CONSTANT);
            cv::remap(srcMat, dstMat, xMat, yMat, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
        } else if (border_type == ppl::cv::BORDER_REPLICATE) {
            ppl::cv::x86::RemapNearestPoint<T, nc>(inHeight, inWidth, inWidth * nc, src.get(), outHeight, outWidth, outWidth * nc, dst.get(), map_x.get(), map_y.get(), ppl::cv::BORDER_REPLICATE);
            cv::remap(srcMat, dstMat, xMat, yMat, cv::INTER_NEAREST, cv::BORDER_REPLICATE);
        } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
            ppl::cv::x86::RemapNearestPoint<T, nc>(inHeight, inWidth, inWidth * nc, src.get(), outHeight, outWidth, outWidth * nc, dst.get(), map_x.get(), map_y.get(), ppl::cv::BORDER_TRANSPARENT);
            cv::remap(srcMat, dstMat, xMat, yMat, cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
        }
    }
    checkResult<T, nc>(dst.get(), dst_ref.get(), outHeight, outWidth, outWidth * nc, outWidth * nc, 1.01f);
}

TEST(REMAP_UINT8_LINEAR, x86)
{
    RemapTest<uint8_t, 1, ppl::cv::BORDER_CONSTANT, true>(480, 640, 480, 640, 1.01f);
    RemapTest<uint8_t, 3, ppl::cv::BORDER_CONSTANT, true>(480, 640, 480, 640, 1.01f);
    RemapTest<uint8_t, 4, ppl::cv::BORDER_CONSTANT, true>(480, 640, 480, 640, 1.01f);

    RemapTest<uint8_t, 1, ppl::cv::BORDER_REPLICATE, true>(480, 640, 480, 640, 1.01f);
    RemapTest<uint8_t, 3, ppl::cv::BORDER_REPLICATE, true>(480, 640, 480, 640, 1.01f);
    RemapTest<uint8_t, 4, ppl::cv::BORDER_REPLICATE, true>(480, 640, 480, 640, 1.01f);

    RemapTest<uint8_t, 1, ppl::cv::BORDER_TRANSPARENT, true>(480, 640, 480, 640, 1.01f);
    RemapTest<uint8_t, 3, ppl::cv::BORDER_TRANSPARENT, true>(480, 640, 480, 640, 1.01f);
    RemapTest<uint8_t, 4, ppl::cv::BORDER_TRANSPARENT, true>(480, 640, 480, 640, 1.01f);
}

TEST(REMAP_FP32_LINEAR, x86)
{
    RemapTest<float, 1, ppl::cv::BORDER_CONSTANT, true>(480, 640, 480, 640, 1.01f);
    RemapTest<float, 3, ppl::cv::BORDER_CONSTANT, true>(480, 640, 480, 640, 1.01f);
    RemapTest<float, 4, ppl::cv::BORDER_CONSTANT, true>(480, 640, 480, 640, 1.01f);

    RemapTest<float, 1, ppl::cv::BORDER_REPLICATE, true>(480, 640, 480, 640, 1.01f);
    RemapTest<float, 3, ppl::cv::BORDER_REPLICATE, true>(480, 640, 480, 640, 1.01f);
    RemapTest<float, 4, ppl::cv::BORDER_REPLICATE, true>(480, 640, 480, 640, 1.01f);

    RemapTest<float, 1, ppl::cv::BORDER_TRANSPARENT, true>(480, 640, 480, 640, 1.01f);
    RemapTest<float, 3, ppl::cv::BORDER_TRANSPARENT, true>(480, 640, 480, 640, 1.01f);
    RemapTest<float, 4, ppl::cv::BORDER_TRANSPARENT, true>(480, 640, 480, 640, 1.01f);
}

TEST(REMAP_UINT8_NEAREST, x86)
{
    RemapTest<uint8_t, 1, ppl::cv::BORDER_CONSTANT, false>(48, 64, 48, 64, 1.01f);
    RemapTest<uint8_t, 3, ppl::cv::BORDER_CONSTANT, false>(48, 64, 48, 64, 1.01f);
    RemapTest<uint8_t, 4, ppl::cv::BORDER_CONSTANT, false>(48, 64, 48, 64, 1.01f);

    RemapTest<uint8_t, 1, ppl::cv::BORDER_REPLICATE, false>(48, 64, 48, 64, 1.01f);
    RemapTest<uint8_t, 3, ppl::cv::BORDER_REPLICATE, false>(48, 64, 48, 64, 1.01f);
    RemapTest<uint8_t, 4, ppl::cv::BORDER_REPLICATE, false>(48, 64, 48, 64, 1.01f);

    RemapTest<uint8_t, 1, ppl::cv::BORDER_TRANSPARENT, false>(48, 64, 48, 64, 1.01f);
    RemapTest<uint8_t, 3, ppl::cv::BORDER_TRANSPARENT, false>(48, 64, 48, 64, 1.01f);
    RemapTest<uint8_t, 4, ppl::cv::BORDER_TRANSPARENT, false>(48, 64, 48, 64, 1.01f);
}

TEST(REMAP_FP32_NEAREST, x86)
{
    RemapTest<float, 1, ppl::cv::BORDER_CONSTANT, false>(48, 64, 48, 64, 1.01f);
    RemapTest<float, 3, ppl::cv::BORDER_CONSTANT, false>(48, 64, 48, 64, 1.01f);
    RemapTest<float, 4, ppl::cv::BORDER_CONSTANT, false>(48, 64, 48, 64, 1.01f);

    RemapTest<float, 1, ppl::cv::BORDER_REPLICATE, false>(48, 64, 48, 64, 1.01f);
    RemapTest<float, 3, ppl::cv::BORDER_REPLICATE, false>(48, 64, 48, 64, 1.01f);
    RemapTest<float, 4, ppl::cv::BORDER_REPLICATE, false>(48, 64, 48, 64, 1.01f);

    RemapTest<float, 1, ppl::cv::BORDER_TRANSPARENT, false>(48, 64, 48, 64, 1.01f);
    RemapTest<float, 3, ppl::cv::BORDER_TRANSPARENT, false>(48, 64, 48, 64, 1.01f);
    RemapTest<float, 4, ppl::cv::BORDER_TRANSPARENT, false>(48, 64, 48, 64, 1.01f);
}
