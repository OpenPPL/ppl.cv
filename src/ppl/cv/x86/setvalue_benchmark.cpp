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
#include "ppl/cv/x86/setvalue.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {

template<typename T, int32_t nc, bool use_mask, int32_t mask_nc = 0>
void BM_SetTo_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    std::unique_ptr<uint8_t[]> mask(new uint8_t[width * height * mask_nc]);
    ppl::cv::debug::randomFill<T>(dst.get(), width * height * nc, 0, 255);
    for (int32_t i = 0; i < height * width * mask_nc; ++i) {
        mask.get()[i] = std::rand() % 2;
    }
    T value = static_cast<T>(17);
    if(use_mask) {
        for (auto _ : state) {
            ppl::cv::x86::SetTo<T, nc, mask_nc>(height, width, width * nc, dst.get(), static_cast<T>(value), width * mask_nc, mask.get());
        }
    }else {
        for (auto _ : state) {
            ppl::cv::x86::SetTo<T, nc>(height, width, width * nc, dst.get(), static_cast<T>(value), 0, nullptr);
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, float, 1, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, float, 3, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, float, 4, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, uint8_t, 1, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, uint8_t, 3, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, uint8_t, 4, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, float, 1, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, float, 3, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, float, 4, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, uint8_t, 1, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, uint8_t, 3, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_ppl_x86, uint8_t, 4, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t nc, bool use_mask, int32_t mask_nc = 0>
void BM_SetTo_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    std::unique_ptr<uint8_t[]> mask(new uint8_t[width * height * mask_nc]);
    cv::Mat maskMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, mask_nc), mask.get());
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst.get());
    ppl::cv::debug::randomFill<T>(dst.get(), width * height * nc, 0, 255);
    for (int32_t i = 0; i < height * width * mask_nc; ++i) {
        mask.get()[i] = std::rand() % 2;
    }
    T value = static_cast<T>(17);
    if(use_mask) {
        for (auto _ : state) {
            dstMat.setTo(value, maskMat);
        }
    }else {
        for (auto _ : state) {
            dstMat.setTo(value);
        }
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, float, 1, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, float, 3, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, float, 4, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, uint8_t, 1, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, uint8_t, 3, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, uint8_t, 4, true, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, float, 1, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, float, 3, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, float, 4, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, uint8_t, 1, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, uint8_t, 3, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_SetTo_opencv_x86, uint8_t, 4, false)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV

template<typename T, int32_t nc>
void BM_Ones_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);   
    for (auto _ : state) {
        ppl::cv::x86::Ones<T, nc>(height, width, width * nc, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Ones_ppl_x86, float, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_ppl_x86, float, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_ppl_x86, float, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_ppl_x86, uint8_t, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_ppl_x86, uint8_t, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_ppl_x86, uint8_t, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});


#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t nc>
void BM_Ones_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst.get());
    for (auto _ : state) {
        dstMat = cv::Mat::ones(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc));
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Ones_opencv_x86, float, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_opencv_x86, float, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_opencv_x86, float, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_opencv_x86, uint8_t, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_opencv_x86, uint8_t, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Ones_opencv_x86, uint8_t, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV

template<typename T, int32_t nc>
void BM_Zeros_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);   
    for (auto _ : state) {
        ppl::cv::x86::Zeros<T, nc>(height, width, width * nc, dst.get());
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_Zeros_ppl_x86, float, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_ppl_x86, float, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_ppl_x86, float, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_ppl_x86, uint8_t, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_ppl_x86, uint8_t, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_ppl_x86, uint8_t, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});


#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t nc>
void BM_Zeros_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    std::unique_ptr<T[]> dst(new T[width * height * nc]);
    cv::Mat dstMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst.get());
    for (auto _ : state) {
        dstMat = cv::Mat::zeros(height, width, CV_MAKETYPE(cv::DataType<T>::depth, nc));
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Zeros_opencv_x86, float, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_opencv_x86, float, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_opencv_x86, float, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_opencv_x86, uint8_t, 1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_opencv_x86, uint8_t, 3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Zeros_opencv_x86, uint8_t, 4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
}
