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

#include <benchmark/benchmark.h>
#include "ppl/cv/x86/erode.h"
#include "ppl/cv/debug.h"
#include <opencv2/imgproc.hpp>
#include <memory>

namespace {

template<typename T, int32_t channels, int32_t erode_size>
void BM_Erode_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                         cv::Size(erode_size, erode_size));
    for (auto _ : state) {
        ppl::cv::x86::Erode<T, channels>(height, width, width * channels, src.get(),
                                            erode_size, erode_size,
                                            element.ptr<uint8_t>(), width * channels,
                                            dst.get(), ppl::cv::BORDER_CONSTANT);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, float, c1, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, float, c3, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, float, c4, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, float, c1, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, float, c3, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, float, c4, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, uint8_t, c1, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, uint8_t, c3, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, uint8_t, c4, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, uint8_t, c1, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, uint8_t, c3, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_ppl_x86, uint8_t, c4, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t channels, int32_t erode_size>
static void BM_Erode_opencv_x86(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                         cv::Size(erode_size, erode_size));
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dst.get());
    for (auto _ : state) {
        cv::erode(srcMat, dstMat, element);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, float, c1, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, float, c3, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, float, c4, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, float, c1, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, float, c3, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, float, c4, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, uint8_t, c1, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, uint8_t, c3, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, uint8_t, c4, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, uint8_t, c1, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, uint8_t, c3, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Erode_opencv_x86, uint8_t, c4, 5)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
}
