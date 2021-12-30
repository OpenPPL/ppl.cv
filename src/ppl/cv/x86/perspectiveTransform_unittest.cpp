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

#include "ppl/cv/x86/perspectivetransform.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/types.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"


template<typename T, int32_t ncSrc, int32_t ncDst>
void PerspectiveTransfromTest(int32_t height, int32_t width) {
    std::unique_ptr<T[]> src(new T[width * height * ncSrc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * ncDst]);
    std::unique_ptr<T[]> dst(new T[width * height * ncDst]);
    std::unique_ptr<T[]> affineMatrix(new T[(ncSrc + 1) * (ncDst + 1)]);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, ncSrc), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, ncDst), dst_ref.get());
    cv::Mat affineMat(ncDst + 1, ncSrc + 1, CV_MAKETYPE(cv::DataType<T>::depth, 1), affineMatrix.get());
    ppl::cv::debug::randomFill<T>(src.get(), width * height * ncSrc, 0, 1024);
    ppl::cv::debug::randomFill<T>(affineMatrix.get(), (ncDst + 1) * (ncSrc + 1), 0, 1024);
    T *affineMatrixPtr = affineMatrix.get();
    ppl::cv::x86::PerspectiveTransform<T, ncSrc, ncDst>(height, width, width * ncSrc, src.get(), width * ncDst, dst.get(), affineMatrixPtr);
    cv::perspectiveTransform(srcMat, dstMat, affineMat);
    checkResult<T, ncDst>(dst.get(), dst_ref.get(), height, width, width * ncDst, width * ncDst, 1e-3);
}



TEST(PerspectiveTransformFP32, x86)
{
    PerspectiveTransfromTest<float, 3, 3>(240, 320);
    PerspectiveTransfromTest<float, 3, 3>(640, 720);
    PerspectiveTransfromTest<float, 3, 3>(720, 1080);
    PerspectiveTransfromTest<float, 3, 3>(1280, 1920);

    PerspectiveTransfromTest<float, 2, 2>(240, 320);
    PerspectiveTransfromTest<float, 2, 2>(640, 720);
    PerspectiveTransfromTest<float, 2, 2>(720, 1080);
    PerspectiveTransfromTest<float, 2, 2>(1280, 1920);
}

