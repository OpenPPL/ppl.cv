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
#include "ppl/cv/arm/medianblur.h"
#include "ppl/cv/debug.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "ppl/cv/arm/test.h"

template<typename T, int c>
class medianblur_ : public  ::testing::TestWithParam<std::tuple<Size, int>> {
public:
    using MedianBlurParameter = std::tuple<Size, int>;
    medianblur_()
    {
    }
    ~medianblur_()
    {
    }

    void apply(const MedianBlurParameter &param)
    {
        Size size = std::get<0>(param);
        int kernel = std::get<1>(param);

        if((std::is_same<T, float>::value) && (kernel != 3) && (kernel != 5)) {
            std::cout << "for fp32 datatype, opencv only support 3x3 & 5x5, " 
                      << "so this test case seen as passed" << std::endl;
            SUCCEED();
            return;
        }

        T *src = (T*)malloc(size.width * size.height * c * sizeof(T));
        ppl::cv::debug::randomFill<T>(src, size.width * size.height * c, 0, 255);
        T *dst = (T*)malloc(size.width * size.height * c * sizeof(T));

        ppl::cv::arm::MedianBlur<T, c>(size.height, size.width, size.width * c,
            src, size.width * c, dst, kernel, ppl::cv::BORDER_REPLICATE);

        T* dst_opencv = (T*)malloc(size.width * size.height * c * sizeof(T));
        cv::Mat iMat(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, c), src);
        cv::Mat oMat(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, c), dst_opencv);

        cv::medianBlur(iMat, oMat, kernel);

        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                for (int k = 0; k < c; ++k) {
                    if (fabs(dst[i * size.width * c + j * c + k] -
                             dst_opencv[i * size.width * c + j * c + k]) > 1e-3) {
                        FAIL() << "Diff at (h, w, c) = (" << i << ", " << j <<
                            ", " << k << "), ppl.cv: " << (float)dst[i * size.width * c + j * c + k] <<
                            ", opencv: " << (float)dst_opencv[i * size.width * c + j * c + k] << "\n";
                    }
                }
            }
        }

        free(src);
        free(dst);
    }
};

#define R(name, t, c) \
    using name = medianblur_<t, c>; \
    TEST_P(name, abc) \
    { \
        this->apply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
                            ::testing::Combine(\
                            ::testing::Values(Size{320, 240}, Size{640, 480}, Size{321, 241}, Size{639, 479}),\
                            ::testing::Values(3, 5, 7)));

R(MedianBlur_f32c1, float, 1)
R(MedianBlur_f32c3, float, 3)
R(MedianBlur_f32c4, float, 4)
R(MedianBlur_u8c1, uint8_t, 1)
R(MedianBlur_u8c3, uint8_t, 3)
R(MedianBlur_u8c4, uint8_t, 4)
