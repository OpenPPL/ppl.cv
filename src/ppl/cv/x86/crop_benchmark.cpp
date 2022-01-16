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
#include "ppl/cv/x86/crop.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {

template<typename T, int32_t nc>
void BM_Crop_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    int32_t inHeight = height;
    int32_t inWidth = width;
    int32_t outHeight = height / 2;
    int32_t outWidth = width / 2;
    int32_t left = 20;
    int32_t top = 20;
    std::unique_ptr<T[]> src(new T[inWidth * inHeight * nc]);
    std::unique_ptr<T[]> dst(new T[outWidth * outHeight * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), inWidth * inHeight * nc, 0, 255);
    for (auto _ : state) {
        ppl::cv::x86::Crop<T, nc>(inHeight, inWidth, inWidth * nc, src.get(),
                                outHeight, outWidth, outWidth * nc, dst.get(),
                                left, top, 1.0f);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;
BENCHMARK_TEMPLATE(BM_Crop_ppl_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_ppl_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_ppl_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_ppl_x86, float, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_ppl_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_ppl_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});


#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t nc>
void BM_Crop_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    int32_t inHeight = height;
    int32_t inWidth = width;
    int32_t outHeight = height / 2;
    int32_t outWidth = width / 2;
    int32_t left = 20;
    int32_t top = 20;
    std::unique_ptr<T[]> src(new T[inWidth * inHeight * nc]);
    std::unique_ptr<T[]> dst(new T[outWidth * outHeight * nc]);
    cv::Mat srcMat(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * inWidth * nc);
    cv::Mat dstMat(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst.get(), sizeof(T) * outWidth * nc);
    cv::Rect roi(left, top, outWidth, outHeight);
    for (auto _ : state) {
        cv::Rect roi(left, top, outWidth, outHeight);
        cv::Mat croppedImage = srcMat(roi);
        croppedImage.copyTo(dstMat);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Crop_opencv_x86, uint8_t, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86, uint8_t, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86, uint8_t, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86, float, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86, float, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86, float, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
}
