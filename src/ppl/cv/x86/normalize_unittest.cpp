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

#include "ppl/cv/x86/normalize.h"
#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <typename Tsrc, int c>
class Normalize : public ::testing::TestWithParam<std::tuple<Size, double, double, ppl::cv::NormTypes, int>> {
public:
    using NormalizeParam = std::tuple<Size, double, double, ppl::cv::NormTypes, int>;
    Normalize()
    {
    }

    ~Normalize()
    {
    }

    void apply(const NormalizeParam &param)
    {
        Size size                    = std::get<0>(param);
        double alpha                 = std::get<1>(param);
        double beta                  = std::get<2>(param);
        ppl::cv::NormTypes norm_type = std::get<3>(param);
        int use_mask                 = std::get<4>(param);

        std::unique_ptr<Tsrc[]> src(new Tsrc[size.width * size.height * c]);
        ppl::cv::debug::randomFill<Tsrc>(src.get(), size.width * size.height * c, 0, 255);

        std::unique_ptr<uchar[]> mask(new uchar[size.width * size.height]);
        for (int i = 0; i < size.height * size.width; ++i) {
            mask.get()[i] = (std::rand()) % 2;
        }

        std::unique_ptr<float[]> dst(new float[size.width * size.height * c]);
        std::unique_ptr<float[]> dst_opencv(new float[size.width * size.height * c]);

        ::cv::Mat iMat(size.height, size.width, T2CvType<Tsrc, c>::type, src.get());
        ::cv::Mat oMat(size.height, size.width, T2CvType<float, c>::type, dst_opencv.get());
        ::cv::Mat maskMat(size.height, size.width, T2CvType<uchar, 1>::type, mask.get());
        if (use_mask == 0) {
            ppl::cv::x86::Normalize<Tsrc, c>(size.height, size.width, size.width * c, src.get(), size.width * c, dst.get(), alpha, beta, norm_type);
            ::cv::normalize(iMat, oMat, alpha, beta, norm_type, CV_32F);
        } else if (use_mask == 1) {
            if (norm_type == ppl::cv::NORM_MINMAX && c > 1) {
                return;
            } else {
                ppl::cv::x86::Normalize<Tsrc, c>(size.height, size.width, size.width * c, src.get(), size.width * c, dst.get(), alpha, beta, norm_type, size.width, mask.get());
                ::cv::normalize(iMat, oMat, alpha, beta, norm_type, CV_32F, maskMat);
            }
        }
        float *ptr_dst    = dst.get();
        float *ptr_opencv = dst_opencv.get();
        uchar *ptr_mask   = mask.get();
        for (int i = 0; i < size.height; ++i) {
            if (mask != nullptr && norm_type == ppl::cv::NORM_MINMAX && c > 1) {
                continue;
            }
            for (int j = 0; j < size.width; ++j) {
                if (ptr_mask[i * size.width + j] != 0) {
                    for (int k = 0; k < c; ++k) {
                        if (std::abs(ptr_dst[i * size.width * c + j * c + k] - ptr_opencv[i * size.width * c + j * c + k]) > 0.001) {
                            FAIL() << "Diff at ( " << i << ", " << j << "), ppl.cv:" << ptr_dst[i * size.width * c + j * c + k] << ", opencv: " << ptr_opencv[i * size.width * c + j * c + k] << "\n";
                        }
                    }
                }
            }
        }
    }
};

#define R(name, ts, c)             \
    using name = Normalize<ts, c>; \
    TEST_P(name, abc)              \
    {                              \
        this->apply(GetParam());   \
    }                              \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Combine(::testing::Values(Size{5, 5}, Size{100, 100}, Size{240, 320}, Size{640, 720}), ::testing::Values(1, 1.7, 4.2), ::testing::Values(0, 0.3, 1.5), ::testing::Values(ppl::cv::NORM_L1, ppl::cv::NORM_L2, ppl::cv::NORM_INF, ppl::cv::NORM_MINMAX), ::testing::Values(0, 1)));

R(Normalize_u8c1, uint8_t, 1)
R(Normalize_u8c3, uint8_t, 3)
R(Normalize_u8c4, uint8_t, 4)
R(Normalize_f32c1, float, 1)
R(Normalize_f32c3, float, 3)
R(Normalize_f32c4, float, 4)
