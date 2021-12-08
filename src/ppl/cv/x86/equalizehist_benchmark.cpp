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

#include "ppl/cv/x86/equalizehist.h"
#include "ppl/cv/types.h"
#include "ppl/cv/debug.h"
#include <memory>
#include <benchmark/benchmark.h>

namespace {

void BM_EqualizeHist_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height, 0, 255);
    for (auto _ : state) {
        ppl::cv::x86::EqualizeHist(height, width, width, src.get(), width, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK(BM_EqualizeHist_ppl_x86)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
static void BM_EqualizeHist_opencv_x86(benchmark::State &state)
{
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<uint8_t[]> src(new uint8_t[width * height]);
    std::unique_ptr<uint8_t[]> dst(new uint8_t[width * height]);
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height, 0, 255);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), src.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), dst.get());
    for (auto _ : state) {
        cv::equalizeHist(srcMat, dstMat);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK(BM_EqualizeHist_opencv_x86)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
}

