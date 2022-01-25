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

#include "ppl/cv/x86/remap.h"
#include "ppl/cv/debug.h"
#include <benchmark/benchmark.h>
#include <memory>
namespace {

template <typename T, int channels>
void BM_REMAP_ppl_x86(benchmark::State &state)
{
    int width  = state.range(0);
    int height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    std::unique_ptr<float[]> map_x(new float[width * height]);
    std::unique_ptr<float[]> map_y(new float[width * height]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    ppl::cv::debug::randomFill<float>(map_x.get(), width * height, 0, width - 1);
    ppl::cv::debug::randomFill<float>(map_y.get(), width * height, 0, height - 1);

    for (auto _ : state) {
        ppl::cv::x86::RemapLinear<T, channels>(height, width, width * channels, src.get(), height, width, width * channels, dst.get(), map_x.get(), map_y.get(), ppl::cv::BORDER_CONSTANT);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_REMAP_ppl_x86, float, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_ppl_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPL3CV_BENCHMARK_OPENCV

template <typename T, int channels>
void BM_REMAP_opencv_x86(benchmark::State &state)
{
    int width  = state.range(0);
    int height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    std::unique_ptr<float[]> map_x(new float[width * height]);
    std::unique_ptr<float[]> map_y(new float[width * height]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    ppl::cv::debug::randomFill<float>(map_x.get(), width * height, 0, width - 1);
    ppl::cv::debug::randomFill<float>(map_y.get(), width * height, 0, height - 1);
    cv::Mat yMat(height, width, CV_MAKETYPE(cv::DataType<float>::depth, 1), map_y.get(), sizeof(float) * width);
    cv::Mat xMat(height, width, CV_MAKETYPE(cv::DataType<float>::depth, 1), map_x.get(), sizeof(float) * width);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), src.get(), sizeof(T) * width * channels);
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dst.get(), sizeof(T) * width * channels);

    for (auto _ : state) {
        cv::remap(srcMat, dstMat, xMat, yMat, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
BENCHMARK_TEMPLATE(BM_REMAP_opencv_x86, float, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_opencv_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_opencv_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_opencv_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_REMAP_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPL3CV_BENCHMARK_OPENCV
} // namespace