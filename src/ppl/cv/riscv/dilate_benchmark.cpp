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

#include "ppl/cv/riscv/dilate.h"
#include "ppl/cv/debug.h"
#include "ppl/cv/types.h"
#include <memory>
#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>

namespace {

template <typename T, int32_t nc, int32_t kernel_size>
void BM_Dilate_ppl_riscv(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    std::unique_ptr<unsigned char[]> kernel(new unsigned char[kernel_size * kernel_size]);
    memset(kernel.get(), 1, kernel_size * kernel_size);

    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 1);
    for (auto _ : state) {
        ppl::cv::riscv::Dilate<T, nc>(height, width, width * nc, src.get(), kernel_size, kernel_size, kernel.get(), width * nc, dst.get(), ppl::cv::BORDER_CONSTANT, 0);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, float, c1, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, float, c3, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, float, c4, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, float, c1, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, float, c3, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, float, c4, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, uchar, c1, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, uchar, c3, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, uchar, c4, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, uchar, c1, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, uchar, c3, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_ppl_riscv, uchar, c4, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename T, int32_t nc, int32_t kernel_size>
void BM_Dilate_opencv_riscv(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    cv::Mat iMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
    cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst.get(), sizeof(T) * width * nc);
    std::unique_ptr<unsigned char[]> kernel(new unsigned char[kernel_size * kernel_size]);
    memset(kernel.get(), 1, kernel_size * kernel_size);
    cv::Mat eleMat(kernel_size, kernel_size, CV_MAKETYPE(CV_8U, 1), kernel.get());

    for (auto _ : state) {
        cv::dilate(iMat, oMat, eleMat);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, float, c1, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, float, c3, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, float, c4, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, float, c1, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, float, c3, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, float, c4, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, uchar, c1, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, uchar, c3, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, uchar, c4, k3x3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, uchar, c1, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, uchar, c3, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Dilate_opencv_riscv, uchar, c4, k5x5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
} // namespace
