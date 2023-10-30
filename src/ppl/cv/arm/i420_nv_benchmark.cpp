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
#include "ppl/cv/arm/cvtcolor.h"
#include "ppl/cv/debug.h"

namespace {
template <typename T, int32_t input_channels, int32_t output_channels>
class I420_NV_Benchmark {
public:
    T *dev_iImage;
    T *dev_oImage;
    int32_t width;
    int32_t height;

    I420_NV_Benchmark(int32_t width, int32_t height)
        : dev_iImage(nullptr)
        , dev_oImage(nullptr)
        , width(width)
        , height(height)
    {
        dev_iImage = (T *)malloc(sizeof(T) * width * height * input_channels * 3 / 2);
        dev_oImage = (T *)malloc(sizeof(T) * width * height * output_channels * 3 / 2);
    }

    ~I420_NV_Benchmark()
    {
        free(dev_iImage);
        free(dev_oImage);
    }

    void NV122I420apply()
    {
        ppl::cv::arm::NV122I420<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }

    void NV212I420apply()
    {
        ppl::cv::arm::NV212I420<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }

    void I4202NV12apply()
    {
        ppl::cv::arm::I4202NV12<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }

    void I4202NV21apply()
    {
        ppl::cv::arm::I4202NV21<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
};
} // namespace
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_NV122I420_ppl_aarch64(benchmark::State &state)
{
    I420_NV_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.NV122I420apply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_NV212I420_ppl_aarch64(benchmark::State &state)
{
    I420_NV_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.NV212I420apply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_I4202NV12_ppl_aarch64(benchmark::State &state)
{
    I420_NV_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.I4202NV12apply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_I4202NV21_ppl_aarch64(benchmark::State &state)
{
    I420_NV_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.I4202NV21apply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

constexpr int32_t c1 = 1;

// pplcv
BENCHMARK_TEMPLATE(BM_NV122I420_ppl_aarch64, uint8_t, c1, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_NV212I420_ppl_aarch64, uint8_t, c1, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_I4202NV12_ppl_aarch64, uint8_t, c1, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_I4202NV21_ppl_aarch64, uint8_t, c1, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
