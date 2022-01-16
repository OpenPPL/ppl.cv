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

#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "ppl/cv/x86/distancetransform.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/debug.h"
#include "ppl/cv/types.h"

#define CHECK_RESULT(a, b) \
        EXPECT_LT(abs(a-b), diff_THR);

template<typename T>
class distanceTransform : public  ::testing::TestWithParam<std::tuple<Size, float>> {
public:
    using Parameter = std::tuple<Size, float>;
    distanceTransform(){
    }
    ~distanceTransform(){
    }

    void apply(const Parameter &param){
        constexpr int32_t c = 1; // must be 1
        Size size       = std::get<0>(param);
        float diff_THR   = std::get<1>(param);
        int32_t height = size.height;
        int32_t width  = size.width;
        int32_t stride = size.width * c;

        std::unique_ptr<uint8_t[]> inData(new uint8_t[stride * height]);
        std::unique_ptr<T[]> pplcv_outData(new T[stride * height]);
        std::unique_ptr<T[]> opencv_outData(new T[stride * height]);
        ppl::cv::debug::randomFill<uint8_t>(inData.get(), stride * height, 0, 255);
        ppl::cv::debug::randomFill<T>(pplcv_outData.get(), stride * height, 0, 255);
        ppl::cv::debug::randomFill<T>(opencv_outData.get(), stride * height, 0, 255);

        //pplcv
        ppl::cv::x86::DistanceTransform(height, width, stride, inData.get(), stride, pplcv_outData.get(),
            ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE);

        //opencv
        cv::Mat iMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, c), inData.get());
        cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, c), opencv_outData.get());
        cv::distanceTransform(iMat, oMat, 2/*CV_DIST_L2*/, cv::DIST_MASK_PRECISE);

        //check result
        for(int32_t i = 0; i < height; i++)
        {
            T* ptrOut0 = pplcv_outData.get() + i * width;
            T* ptrOut1 = opencv_outData.get() + i * width;
            for (int32_t j = 0; j < width; j++)
            {
                CHECK_RESULT(ptrOut0[j], ptrOut1[j]);
            }
        }
    }
};

#define R(name, t, diff) \
    using name = distanceTransform<t>; \
    TEST_P(name, abc) \
    { \
        this->apply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
                            ::testing::Combine(\
                            ::testing::Values(Size{320, 240}, Size{640, 480}),\
                            ::testing::Values(diff)));

R(distanceTransform_f32, float, 1e-4)
