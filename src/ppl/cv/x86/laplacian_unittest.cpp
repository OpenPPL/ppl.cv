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

#include "ppl/cv/x86/laplacian.h"
#include "ppl/cv/x86/test.h"
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template<typename T, int32_t filter_size, int32_t nc>
void LaplacianTest(int32_t height, int32_t width, double scale, double delta, T diff) {
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * width * nc);

    cv::Laplacian(src_opencv, dst_opencv, cv::DataType<T>::depth, filter_size, scale, delta);
    ppl::cv::x86::Laplacian<T, nc>(height, width, width * nc, src.get(), width * nc, dst.get(),
                            filter_size, scale, delta, ppl::cv::BORDER_REFLECT_101);

    checkResult<T, nc>(dst_ref.get(), dst.get(),
                    height, width,
                    width * nc, width * nc,
                    diff);
}


TEST(Laplacian_FP32, x86)
{
    LaplacianTest<float, 1, 1>(720, 1080, 2.0, 1.0, 1);
    LaplacianTest<float, 1, 3>(720, 1080, 2.0, 1.0, 1);
    LaplacianTest<float, 1, 4>(720, 1080, 2.0, 1.0, 1);
    LaplacianTest<float, 3, 1>(720, 1080, 2.0, 1.0, 1);
    LaplacianTest<float, 3, 3>(720, 1080, 2.0, 1.0, 1);
    LaplacianTest<float, 3, 4>(720, 1080, 2.0, 1.0, 1);
}

TEST(Laplacian_UINT8, x86)
{
    LaplacianTest<uint8_t, 1, 1>(720, 1080, 2, 1, 1);
    LaplacianTest<uint8_t, 1, 3>(720, 1080, 2, 1, 1);
    LaplacianTest<uint8_t, 1, 4>(720, 1080, 2, 1, 1);
    LaplacianTest<uint8_t, 3, 1>(720, 1080, 2, 1, 1);
    LaplacianTest<uint8_t, 3, 3>(720, 1080, 2, 1, 1);
    LaplacianTest<uint8_t, 3, 4>(720, 1080, 2, 1, 1);
}
