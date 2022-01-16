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
#include "ppl/cv/x86/integral.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {

template<typename TSrc, typename TDst, int32_t nc>
void BM_Integral_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    int32_t outHeight = height + 1;
    int32_t outWidth = width + 1;
    std::unique_ptr<TSrc[]> src(new TSrc[width * height * nc]);
    std::unique_ptr<TDst[]> dst(new TDst[outHeight * outWidth * nc]);
    ppl::cv::debug::randomFill<TSrc>(src.get(), width * height * nc, 0, 255);
    for (auto _ : state) {
        ppl::cv::x86::Integral<TSrc, TDst, nc>(height, width, width * nc, src.get(), outHeight, outWidth, outWidth * nc, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Integral_ppl_x86, float, float, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_ppl_x86, float, float, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_ppl_x86, float, float, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_ppl_x86, uint8_t, int32_t, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_ppl_x86, uint8_t, int32_t, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_ppl_x86, uint8_t, int32_t, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename TSrc, typename TDst, int32_t nc>
void BM_Integral_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    int32_t outHeight = height + 1;
    int32_t outWidth = width + 1;
    std::unique_ptr<TSrc[]> src(new TSrc[width * height * nc]);
    std::unique_ptr<TDst[]> dst_ref(new TDst[outHeight * outWidth * nc]);
    ppl::cv::debug::randomFill<TSrc>(src.get(), width * height * nc, 0, 255);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<TSrc>::depth, nc), src.get());
    cv::Mat dstMat(outHeight, outWidth, CV_MAKETYPE(cv::DataType<TDst>::depth, nc), dst_ref.get());
    for (auto _ : state) {
        cv::integral(srcMat, dstMat, CV_MAKETYPE(cv::DataType<TDst>::depth, nc));
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Integral_opencv_x86, float, float, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86, float, float, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86, float, float, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86, uint8_t, int32_t, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86, uint8_t, int32_t, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Integral_opencv_x86, uint8_t, int32_t, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
}
