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

#include "ppl/cv/x86/calchist.h"
#include "ppl/cv/types.h"
#include "ppl/cv/debug.h"
#include <memory>
#include <benchmark/benchmark.h>

namespace {

template<typename T, bool with_mask>
void BM_CalcHist_ppl_x86(benchmark::State &state) {
    int width = state.range(0);
    int height = state.range(1);
    constexpr int c = 1;
    int histSize = 256;
    std::unique_ptr<uint8_t> src(new uint8_t[width * height * c]);
    std::unique_ptr<uint8_t> mask(new uint8_t[width * height * c]);
    std::unique_ptr<int> dst(new int[histSize]); //for uint8_t

    //init 
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * c, 0, 255);
    ppl::cv::debug::randomFill<uint8_t>(mask.get(), width * height * c, 0, 2);
    memset(dst.get(), 0, sizeof(int)*histSize);
    if (!with_mask) {
        for (auto _ : state) {
            ppl::cv::x86::CalcHist<uint8_t>(height, width, width * c, src.get(), dst.get());
        }
        state.SetItemsProcessed(state.iterations() * 1);
    } else {
        memset(dst.get(), 0, sizeof(int)*255);
        for (auto _ : state) {
            ppl::cv::x86::CalcHist<uint8_t>(height, width, width * c, src.get(), dst.get(), width * c, mask.get());
        }
        state.SetItemsProcessed(state.iterations() * 1);
    }
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_CalcHist_ppl_x86, uint8_t, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CalcHist_ppl_x86, uint8_t, true)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, bool with_mask>
static void BM_CalcHist_opencv_x86(benchmark::State &state)
{
    int width = state.range(0);
    int height = state.range(1);
    constexpr int c = 1;
    int histSize = 256;
    std::unique_ptr<uint8_t> src(new uint8_t[width * height * c]);
    std::unique_ptr<uint8_t> mask(new uint8_t[width * height * c]);
    std::unique_ptr<int> dst(new int[histSize]); //for uint8_t

    //init 
    ppl::cv::debug::randomFill<uint8_t>(src.get(), width * height * c, 0, 255);
    ppl::cv::debug::randomFill<uint8_t>(mask.get(), width * height * c, 0, 2);
    memset(dst.get(), 0, sizeof(int)*histSize);
    cv::Mat srcMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), src.get());
    cv::Mat maskMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), mask.get());
    cv::Mat dstMat_opencv;

    int channels = 0;
    float data_range[2] = {0,256};
    const float* ranges[1] = {data_range};
    if (!with_mask) {
        for (auto _ : state) {
            cv::calcHist(&srcMat, 1, &channels, cv::Mat(), dstMat_opencv, 1, &histSize, ranges, true, false);
        }
        state.SetItemsProcessed(state.iterations() * 1);
    } else {
        cv::Mat dstMat_opencv_mask;
        for (auto _ : state) {
            cv::calcHist(&srcMat, 1, &channels, maskMat, dstMat_opencv_mask, 1, &histSize, ranges, true, false);
        }
        state.SetItemsProcessed(state.iterations() * 1);
    }
}

BENCHMARK_TEMPLATE(BM_CalcHist_opencv_x86, uint8_t, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_CalcHist_opencv_x86, uint8_t, true)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
}

