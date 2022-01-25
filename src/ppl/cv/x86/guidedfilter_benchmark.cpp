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
#include <opencv2/ximgproc.hpp>
#include "ppl/cv/x86/guidedfilter.h"
#include "ppl/cv/debug.h"
#include "ppl/cv/types.h"

namespace {
template<typename T, int32_t channels>
class GuidedFilterBenchmark {
public:
    T* dev_iImage = nullptr;
    T* dev_oImage = nullptr;
    T* dev_guidedImage = nullptr;
    int32_t inWidth;
    int32_t inHeight;
    int32_t outWidth;
    int32_t outHeight;
    GuidedFilterBenchmark(int32_t inWidth, int32_t inHeight, int32_t outWidth, int32_t outHeight)
        : inWidth(inWidth)
        , inHeight(inHeight)
        , outWidth(outWidth)
        , outHeight(outHeight)
    {
        dev_iImage = (T*)malloc(inWidth * inHeight * channels * sizeof(T));
        dev_guidedImage = (T*)malloc(inWidth * inHeight * channels * sizeof(T));
        dev_oImage = (T*)malloc(outWidth * outHeight * channels * sizeof(T));
        memset(this->dev_iImage, 0, inWidth * inHeight * channels * sizeof(T));
        memset(this->dev_guidedImage, 0, inWidth * inHeight * channels * sizeof(T));
    }

    void apply() {
        int32_t r = 8;
        float eps = 0.4 * 0.4;
        ppl::cv::x86::GuidedFilter<T, channels, channels>(
            this->inHeight,
            this->inWidth,
            this->inWidth * channels,
            this->dev_iImage,
            this->inWidth * channels,
            this->dev_guidedImage,
            this->outWidth * channels,
            this->dev_oImage,
            r,
            eps,
            ppl::cv::BORDER_DEFAULT);
    }

    void apply_opencv() {
        int32_t r = 8;
        float eps = 0.4 * 0.4;
        cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage, sizeof(T) * inWidth * channels);
        cv::Mat guided_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_guidedImage, sizeof(T) * inWidth * channels);
        cv::Mat dst_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage, sizeof(T) * inWidth * channels);

        cv::ximgproc::guidedFilter(guided_opencv, src_opencv, dst_opencv, r, eps, -1);
    }

    ~GuidedFilterBenchmark() {
        free(this->dev_iImage);
        free(this->dev_oImage);
        free(this->dev_guidedImage);
    }
};
}

template<typename T, int32_t channels>
static void BM_GuidedFilter_ppl_x86(benchmark::State &state) {
    GuidedFilterBenchmark<T, channels> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _: state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
}

template<typename T, int32_t channels>
static void BM_GuidedFilter_opencv_x86(benchmark::State &state) {
    GuidedFilterBenchmark<T, channels> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _: state) {
        bm.apply_opencv();
    }
    state.SetItemsProcessed(state.iterations());
}

//ppl.cv
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_x86, float, 3)->Args({320, 240, 320, 320});
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_x86, uint8_t, 3)->Args({320, 240, 320, 320});

//opencv
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86, float, 3)->Args({320, 240, 320, 320});
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86, uint8_t, 3)->Args({320, 240, 320, 320});

