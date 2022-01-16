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

#include "ppl/cv/x86/invert.h"
#include "ppl/cv/x86/test.h"
#include "ppl/cv/debug.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>


template<typename T, int32_t c>
class Invert_ : public  ::testing::TestWithParam<std::tuple<Size, int32_t, float>> {
public:
    using vecOperatorParameter = std::tuple<Size, int32_t, float>;
    int32_t width;
    int32_t height;
    Invert_()
    {
    }
    ~Invert_()
    {
    }

    void apply(const vecOperatorParameter &param)
    {
        Size size       = std::get<0>(param);
        int32_t  method     = std::get<1>(param);
        float diff_THR  = std::get<2>(param);
        width = size.width;
        height= size.height;
        int32_t sstep = width *  c;
        int32_t dstep = width *  c;

        T *input = (T*)malloc(width * height * c * sizeof(T));
        T *output0 = (T*)malloc(width * height * c * sizeof(T));
        T *output1 = (T*)malloc(width * height * c * sizeof(T));
        ppl::cv::debug::randomFill<T>(input, width * height * c, -255, 255);
        ppl::cv::debug::randomFill<T>(output0, width * height * c, -255, 255);
        memcpy(output1, output0, size.width * size.height * c * sizeof(T));

        //make positive definite matrix(method DECOMP_CHOLESKY needed)
        /*
         * |a|b|c|d|
         * |b|e|f|g|
         * |c|f|h|i|
         * |d|g|i|j|
         */
        if(method == ppl::cv::DECOMP_CHOLESKY) {
            for(int32_t i = 0; i < height; i++){
                for(int32_t j = 0; j < width - i; j++){
                    input[i * width + width - 1 - j] = input[(width - 1 - j) * width + i];
                }
            }
        }
        //ppl.cv
        ppl::cv::x86::Invert<T>(height, width, sstep, input, dstep, output0, ppl::cv::DECOMP_CHOLESKY);

        cv::Mat src_opencv(height, width,
                CV_MAKETYPE(cv::DataType<T>::depth, c), input);
        cv::Mat dst_opencv(height, width,
                CV_MAKETYPE(cv::DataType<T>::depth, c), output1);
        switch(method) {
            case ppl::cv::DECOMP_CHOLESKY:
                //opencv
                cv::invert(src_opencv, dst_opencv, 2/*DECOMP_CHOLESKY*/);
                break;
            default:
                std::cout << "[ERROR] method do not support yet!!!" << std::endl;
                free(input);
                free(output0);
                free(output1);
                return;
        }

        checkResult<T, 1>(output0, output1,
                        height, width, sstep, dstep, diff_THR);


        free(input);
        free(output0);
        free(output1);
    }
};

constexpr int32_t c1 = 1;
constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

#define R(name, t, c, diff) \
    using name = Invert_<t, c>; \
    TEST_P(name, x86) \
    { \
        this->apply(GetParam());\
    }\
    INSTANTIATE_TEST_CASE_P(standard, name,\
                            ::testing::Combine(\
                            ::testing::Values(Size{64, 64}),\
                            ::testing::Values(ppl::cv::DECOMP_CHOLESKY),\
                            ::testing::Values(diff)));

R(Invert_float_c1,  float, c1, 8e-2)
R(Invert_double_c1, double, c1, 8e-2)
