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
#include "ppl/cv/x86/distancetransform.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {

template<typename T>
void BM_DistanceTransform_ppl_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    int32_t stride = width; 
    std::unique_ptr<uint8_t[]> inData(new uint8_t[stride * height]);
    std::unique_ptr<T[]> pplcv_outData(new T[stride * height]);
    ppl::cv::debug::randomFill<uint8_t>(inData.get(), stride * height, 0, 255);
    ppl::cv::debug::randomFill<T>(pplcv_outData.get(), stride * height, 0, 255);
    for (auto _ : state) {
        ppl::cv::x86::DistanceTransform(height, width, stride, inData.get(), stride, pplcv_outData.get(), 
            ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;

BENCHMARK_TEMPLATE(BM_DistanceTransform_ppl_x86, float)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T>
void BM_DistanceTransform_opencv_x86(benchmark::State &state) {
    int32_t width = state.range(0);
    int32_t height = state.range(1);
    int32_t stride = width; 
    std::unique_ptr<uint8_t[]> inData(new uint8_t[stride * height]);
    std::unique_ptr<T[]> opencv_outData(new T[stride * height]);
    ppl::cv::debug::randomFill<uint8_t>(inData.get(), stride * height, 0, 255);
    ppl::cv::debug::randomFill<T>(opencv_outData.get(), stride * height, 0, 255);
    cv::Mat iMat(height, width, CV_MAKETYPE(cv::DataType<uint8_t>::depth, 1), inData.get());
    cv::Mat oMat(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1), opencv_outData.get());
    for (auto _ : state) {
        cv::distanceTransform(iMat, oMat, 2/*CV_DIST_L2*/, cv::DIST_MASK_PRECISE);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_DistanceTransform_opencv_x86, float)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#endif //! PPLCV_BENCHMARK_OPENCV
}
