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

#include "ppl/cv/x86/calchist.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"
#include <opencv2/imgproc.hpp>


template<typename T>
void CalcHistTest(int height, int width) {

    constexpr int c = 1;
    int histSize = 256;
    std::unique_ptr<uint8_t> src(new uint8_t[width * height * c]);
    std::unique_ptr<uint8_t> mask(new uint8_t[width * height * c]);
    std::unique_ptr<int> dst(new int[histSize]); //for uint8_t

    //init 
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * c, 0, 255);
    ppl::cv::debug::randomFill<uint8_t>(mask.get(), width * height * c, 0, 2);
    memset(dst.get(), 0, sizeof(int)*histSize);

    //without mask
    ppl::cv::x86::CalcHist<uint8_t>(height, width, width * c, src.get(), dst.get());

    //opencv
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), src.get());
    cv::Mat maskMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), mask.get());
    cv::Mat dstMat_opencv;

    int channels = 0;
    float data_range[2] = {0,256};
    const float* ranges[1] = {data_range};
    cv::calcHist(&srcMat, 1, &channels, cv::Mat(), dstMat_opencv, 1, &histSize, ranges, true, false);

    //check
    int* hist = dst.get();
    for(int i = 0; i < 256; i++){
        float hist_opencv = dstMat_opencv.at<float>(i);
        if(abs(hist_opencv - hist[i]) > 1e-6)
        {
            FAIL() << "hist " << i << " error!!!" << "\n";
        }
    }

    //with mask
    memset(dst.get(), 0, sizeof(int)*255);
    ppl::cv::x86::CalcHist<uint8_t>(height, width, width * c, src.get(), dst.get(), width * c, mask.get());
    //opencv
    cv::Mat dstMat_opencv_mask;
    cv::calcHist(&srcMat, 1, &channels, maskMat, dstMat_opencv_mask, 1, &histSize, ranges, true, false);

    //check
    hist = dst.get();
    for(int i = 0; i < 256; i++){
        float hist_opencv = dstMat_opencv_mask.at<float>(i);
        if(abs(hist_opencv - hist[i]) > 1e-6)
        {
            FAIL() << "mask hist " << i << " error!!!" << "\n";
        }
    }
}

TEST(CalcHistTest_UINT8, x86)
{
    CalcHistTest<uint8_t>(640, 720);
    CalcHistTest<uint8_t>(720, 1080);
    CalcHistTest<uint8_t>(1080, 1920);
}

