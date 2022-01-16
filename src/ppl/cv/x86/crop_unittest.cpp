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

#include "ppl/cv/x86/crop.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename T, int32_t nc>
void CropTest(int32_t inHeight, int32_t inWidth, int32_t outHeight, int32_t outWidth,
                int32_t left, int32_t top, T diff) {
    std::unique_ptr<T[]> src(new T[inWidth * inHeight * nc]);
    std::unique_ptr<T[]> dst_ref(new T[inWidth * inHeight * nc]);
    std::unique_ptr<T[]> dst(new T[outWidth * outHeight * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), inWidth * inHeight * nc, 0, 255);
    ppl::cv::x86::Crop<T, nc>(inHeight, inWidth, inWidth * nc, src.get(),
                            outHeight, outWidth, outWidth * nc, dst.get(),
                            left, top, 1.0f);
    cv::Mat srcMat(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * inWidth * nc);
    cv::Mat dstMat(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * outWidth * nc);
    cv::Rect roi(left, top, outWidth, outHeight);
    cv::Mat croppedImage = srcMat(roi);
    croppedImage.copyTo(dstMat);
    checkResult<T, nc>(dst.get(), dst_ref.get(), outHeight, outWidth, outWidth * nc, outWidth * nc, 1.01f);
}


TEST(CROP_FP32, x86)
{
    CropTest<float, 1>(720, 1080, 360, 540, 20, 40, 1e-3);
    CropTest<float, 3>(720, 1080, 360, 540, 20, 40, 1e-3);
    CropTest<float, 4>(720, 1080, 360, 540, 20, 40, 1e-3);
}

TEST(CROP_UINT8, x86)
{
    CropTest<uint8_t, 1>(720, 1080, 360, 540, 20, 40, 1e-3);
    CropTest<uint8_t, 3>(720, 1080, 360, 540, 20, 40, 1e-3);
    CropTest<uint8_t, 4>(720, 1080, 360, 540, 20, 40, 1e-3);
}
