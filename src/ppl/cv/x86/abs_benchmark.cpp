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

#include "ppl/cv/x86/abs.h"
#include "ppl/cv/debug.h"
#include <memory>
#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
namespace {

template<typename T, int nc>
void BM_Abs_ppl_x86(benchmark::State &state) {
    int width = state.range(0);
    int height = state.range(1);
    std::unique_ptr<float[]> src(new float[width * height * nc]);
    std::unique_ptr<float[]> dst(new float[width * height * nc]);
    ppl::cv::debug::randomFill<float>(src.get(), width * height * nc, 0, 1);
    for (auto _ : state) {
        ppl::cv::x86::Abs<float, nc>(height, width, width * nc, src.get(),
                                                    width * nc, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}


using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Abs_ppl_x86, float, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_ppl_x86, int8_t, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_ppl_x86, int8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_ppl_x86, int8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
using namespace ppl::cv::debug;
template<typename T, int32_t nc>
void BM_Abs_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * nc]);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * nc, 0, 255);
    cv::Mat iMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * width * nc);
    cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst.get(), sizeof(T) * width * nc);
    for (auto _ : state) {
        oMat = cv::abs(iMat);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Abs_opencv_x86, float, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_opencv_x86, float, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_opencv_x86, float, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_opencv_x86, int8_t, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_opencv_x86, int8_t, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Abs_opencv_x86, int8_t, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
}
