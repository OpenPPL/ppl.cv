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
#include "ppl/cv/x86/convertto.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {

template <typename T, int32_t nc>
void BM_CONVERTTO_FP32_To_Uint8_ppl_x86(benchmark::State &state)
{
    float scale    = 255.0f;
    int32_t width  = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<float[]> src(new float[width * height * nc]);
    std::unique_ptr<uint8_t[]> dst_ref(new uint8_t[width * height * nc]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * nc]);
    ppl::cv::debug::randomFill<float>(src.get(), width * height * nc, 0, 1);

    for (auto _ : state) {
        ppl::cv::x86::ConvertTo<float, nc, uint8_t>(height, width, width * nc, src.get(), scale, width * nc, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t nc>
void BM_CONVERTTO_Uint8_To_FP32_ppl_x86(benchmark::State &state)
{
    float scale    = 1.0f / 255.0f;
    int32_t width  = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * nc]);
    std::unique_ptr<float[]> dst_ref(new float[width * height * nc]);
    std::unique_ptr<float[]> dst(new float[width * height * nc]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * nc, 0, 255);

    for (auto _ : state) {
        ppl::cv::x86::ConvertTo<uint8_t, nc, float>(height, width, width * nc, src.get(), scale, width * nc, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_CONVERTTO_FP32_To_Uint8_ppl_x86, float, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_FP32_To_Uint8_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_FP32_To_Uint8_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_Uint8_To_FP32_ppl_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_Uint8_To_FP32_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_Uint8_To_FP32_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV

template <typename T, int32_t nc>
void BM_CONVERTTO_FP32_To_Uint8_opencv_x86(benchmark::State &state)
{
    float scale    = 255.0f;
    int32_t width  = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<float[]> src(new float[width * height * nc]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height * nc]);
    ppl::cv::debug::randomFill<float>(src.get(), width * height * nc, 0, 1);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<float>::depth, nc), src.get(), sizeof(float) * width);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, nc), dst.get(), sizeof(uint8_t) * width);

    for (auto _ : state) {
        src_opencv.convertTo(dst_opencv, CV_8U, scale);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t nc>
void BM_CONVERTTO_Uint8_To_FP32_opencv_x86(benchmark::State &state)
{
    float scale    = 1.0f / 255.0f;
    int32_t width  = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height * nc]);
    std::unique_ptr<float[]> dst(new float[width * height * nc]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * nc, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, nc), src.get(), sizeof(uint8_t) * width);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<float>::depth, nc), dst.get(), sizeof(float) * width);

    for (auto _ : state) {
        src_opencv.convertTo(dst_opencv, CV_32F, scale);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_CONVERTTO_FP32_To_Uint8_opencv_x86, float, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_FP32_To_Uint8_opencv_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_FP32_To_Uint8_opencv_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_Uint8_To_FP32_opencv_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_Uint8_To_FP32_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CONVERTTO_Uint8_To_FP32_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
} // namespace
