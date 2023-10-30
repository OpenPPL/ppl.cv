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
class LAB_RGB_Benchmark {
public:
    T *dev_iImage;
    T *dev_oImage;
    int32_t width;
    int32_t height;

    LAB_RGB_Benchmark(int32_t width, int32_t height)
        : dev_iImage(nullptr)
        , dev_oImage(nullptr)
        , width(width)
        , height(height)
    {
        dev_iImage = (T *)malloc(sizeof(T) * width * height * input_channels);
        dev_oImage = (T *)malloc(sizeof(T) * width * height * output_channels);
    }

    ~LAB_RGB_Benchmark()
    {
        free(dev_iImage);
        free(dev_oImage);
    }

    void RGB2LABapply()
    {
        ppl::cv::arm::RGB2LAB<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void RGB2LABapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2Lab);
    }

    void BGR2LABapply()
    {
        ppl::cv::arm::RGB2LAB<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGR2LABapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2Lab);
    }

    void LAB2RGBapply()
    {
        ppl::cv::arm::LAB2RGB<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void LAB2RGBapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_Lab2RGB);
    }

    void LAB2BGRapply()
    {
        ppl::cv::arm::LAB2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void LAB2BGRapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_Lab2BGR);
    }
};
} // namespace
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2LAB_ppl_aarch64(benchmark::State &state)
{
    LAB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2LABapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2LAB_aarch64_opencv(benchmark::State &state)
{
    LAB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2LABapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2LAB_ppl_aarch64(benchmark::State &state)
{
    LAB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2LABapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2LAB_aarch64_opencv(benchmark::State &state)
{
    LAB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2LABapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_LAB2RGB_ppl_aarch64(benchmark::State &state)
{
    LAB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.LAB2RGBapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_LAB2RGB_aarch64_opencv(benchmark::State &state)
{
    LAB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.LAB2RGBapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_LAB2BGR_ppl_aarch64(benchmark::State &state)
{
    LAB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.LAB2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_LAB2BGR_aarch64_opencv(benchmark::State &state)
{
    LAB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.LAB2BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

// pplcv
BENCHMARK_TEMPLATE(BM_RGB2LAB_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2LAB_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_LAB2RGB_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_LAB2BGR_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

// opencv
BENCHMARK_TEMPLATE(BM_RGB2LAB_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2LAB_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_LAB2RGB_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_LAB2BGR_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
