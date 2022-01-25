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

#include "ppl/cv/x86/warpperspective.h"
#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <typename T, int channels>
void WarpPerspectiveTest(int inHeight, int inWidth, int outHeight, int outWidth, ppl::cv::InterpolationType mode, ppl::cv::BorderType border_type)
{
    std::unique_ptr<T[]> src(new T[inWidth * inHeight * channels]);
    std::unique_ptr<T[]> dst_ref(new T[outWidth * outHeight * channels]);
    std::unique_ptr<T[]> dst(new T[outWidth * outHeight * channels]);
    std::unique_ptr<double[]> affine_matrix(new double[9]);
    ppl::cv::debug::randomFill<T>(src.get(), channels * inWidth * inHeight, 0, 255);
    ppl::cv::debug::randomFill<T>(dst.get(), channels * outWidth * outHeight, 0, 255);
    memcpy(dst_ref.get(), dst.get(), sizeof(T) * channels * outWidth * outHeight);
    ppl::cv::debug::randomFill<double>(affine_matrix.get(), 9, 0, 1);
    cv::Mat srcMat(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), src.get(), sizeof(T) * inWidth * channels);
    cv::Mat dstMat(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dst_ref.get(), sizeof(T) * outWidth * channels);
    cv::Mat affineMat(3, 3, CV_MAKETYPE(cv::DataType<double>::depth, 1), affine_matrix.get());
    if (mode == ppl::cv::INTERPOLATION_LINEAR) {
        if (border_type == ppl::cv::BORDER_REPLICATE) {
            ppl::cv::x86::WarpPerspectiveLinear<T, channels>(inHeight, inWidth, inWidth * channels, src.get(), outHeight, outWidth, outWidth * channels, dst.get(), affine_matrix.get(), border_type);
            cv::warpPerspective(srcMat, dstMat, affineMat, cv::Size(outWidth, outHeight), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
        } else if (border_type == ppl::cv::BORDER_CONSTANT) {
            ppl::cv::x86::WarpPerspectiveLinear<T, channels>(inHeight, inWidth, inWidth * channels, src.get(), outHeight, outWidth, outWidth * channels, dst.get(), affine_matrix.get(), border_type);
            cv::warpPerspective(srcMat, dstMat, affineMat, cv::Size(outWidth, outHeight), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
        } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
            ppl::cv::x86::WarpPerspectiveLinear<T, channels>(inHeight, inWidth, inWidth * channels, src.get(), outHeight, outWidth, outWidth * channels, dst.get(), affine_matrix.get(), border_type);
            cv::warpPerspective(srcMat, dstMat, affineMat, cv::Size(outWidth, outHeight), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_TRANSPARENT);
        }
    } else if (mode == ppl::cv::INTERPOLATION_NEAREST_POINT) {
        if (border_type == ppl::cv::BORDER_REPLICATE) {
            ppl::cv::x86::WarpPerspectiveNearestPoint<T, channels>(inHeight, inWidth, inWidth * channels, src.get(), outHeight, outWidth, outWidth * channels, dst.get(), affine_matrix.get(), border_type);
            cv::warpPerspective(srcMat, dstMat, affineMat, cv::Size(outWidth, outHeight), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
        } else if (border_type == ppl::cv::BORDER_CONSTANT) {
            ppl::cv::x86::WarpPerspectiveNearestPoint<T, channels>(inHeight, inWidth, inWidth * channels, src.get(), outHeight, outWidth, outWidth * channels, dst.get(), affine_matrix.get(), border_type);
            cv::warpPerspective(srcMat, dstMat, affineMat, cv::Size(outWidth, outHeight), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
        } else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
            ppl::cv::x86::WarpPerspectiveNearestPoint<T, channels>(inHeight, inWidth, inWidth * channels, src.get(), outHeight, outWidth, outWidth * channels, dst.get(), affine_matrix.get(), border_type);
            cv::warpPerspective(srcMat, dstMat, affineMat, cv::Size(outWidth, outHeight), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_TRANSPARENT);
        }
    }
    checkResult<T, channels>(dst.get(), dst_ref.get(), outHeight, outWidth, outWidth * channels, outWidth * channels, 1.01f);
}

TEST(WarpPerspectiveLinear_FP32_BORDER_CONSTANT, x86)
{
    WarpPerspectiveTest<float, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT);
    WarpPerspectiveTest<float, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT);
    WarpPerspectiveTest<float, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT);
}

TEST(WarpPerspectiveLinear_FP32_BORDER_REPLICATE, x86)
{
    WarpPerspectiveTest<float, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE);
    WarpPerspectiveTest<float, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE);
    WarpPerspectiveTest<float, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE);
}

TEST(WarpPerspectiveLinear_FP32_BORDER_TRANSPARENT, x86)
{
    WarpPerspectiveTest<float, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT);
    WarpPerspectiveTest<float, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT);
    WarpPerspectiveTest<float, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT);
}

TEST(WarpPerspectiveLinear_UCHAR_BORDER_CONSTANT, x86)
{
    WarpPerspectiveTest<uchar, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT);
    WarpPerspectiveTest<uchar, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT);
    WarpPerspectiveTest<uchar, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT);
}

TEST(WarpPerspectiveLinear_UCHAR_BORDER_REPLICATE, x86)
{
    WarpPerspectiveTest<uchar, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE);
    WarpPerspectiveTest<uchar, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE);
    WarpPerspectiveTest<uchar, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE);
}

TEST(WarpPerspectiveLinear_UCHAR_BORDER_TRANSPARENT, x86)
{
    WarpPerspectiveTest<uchar, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT);
    WarpPerspectiveTest<uchar, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT);
    WarpPerspectiveTest<uchar, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT);
}

TEST(WarpPerspectiveNearest_FP32_BORDER_CONSTANT, x86)
{
    WarpPerspectiveTest<float, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT);
    WarpPerspectiveTest<float, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT);
    WarpPerspectiveTest<float, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT);
}

TEST(WarpPerspectiveNearest_FP32_BORDER_REPLICATE, x86)
{
    WarpPerspectiveTest<float, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE);
    WarpPerspectiveTest<float, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE);
    WarpPerspectiveTest<float, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE);
}

TEST(WarpPerspectiveNearest_FP32_BORDER_TRANSPARENT, x86)
{
    WarpPerspectiveTest<float, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT);
    WarpPerspectiveTest<float, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT);
    WarpPerspectiveTest<float, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT);
}

TEST(WarpPerspectiveNearest_UCHAR_BORDER_CONSTANT, x86)
{
    WarpPerspectiveTest<uchar, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT);
    WarpPerspectiveTest<uchar, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT);
    WarpPerspectiveTest<uchar, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_CONSTANT);
}

TEST(WarpPerspectiveNearest_UCHAR_BORDER_REPLICATE, x86)
{
    WarpPerspectiveTest<uchar, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE);
    WarpPerspectiveTest<uchar, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE);
    WarpPerspectiveTest<uchar, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_REPLICATE);
}

TEST(WarpPerspectiveNearest_UCHAR_BORDER_TRANSPARENT, x86)
{
    WarpPerspectiveTest<uchar, 1>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT);
    WarpPerspectiveTest<uchar, 3>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT);
    WarpPerspectiveTest<uchar, 4>(640, 720, 640, 720, ppl::cv::INTERPOLATION_NEAREST_POINT, ppl::cv::BORDER_TRANSPARENT);
}
