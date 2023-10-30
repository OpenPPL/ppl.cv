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
class RGB_Benchmark {
public:
    T *dev_iImage;
    T *dev_oImage;
    int32_t width;
    int32_t height;

    RGB_Benchmark(int32_t width, int32_t height)
        : dev_iImage(nullptr)
        , dev_oImage(nullptr)
        , width(width)
        , height(height)
    {
        dev_iImage = (T *)malloc(sizeof(T) * width * height * input_channels);
        dev_oImage = (T *)malloc(sizeof(T) * width * height * output_channels);
    }

    ~RGB_Benchmark()
    {
        free(dev_iImage);
        free(dev_oImage);
    }

    void RGB2RGBAapply()
    {
        ppl::cv::arm::RGB2RGBA<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void RGB2RGBAapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2RGBA);
    }
    void RGB2BGRapply()
    {
        ppl::cv::arm::RGB2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void RGB2BGRapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2BGR);
    }
    void RGB2BGRAapply()
    {
        ppl::cv::arm::RGB2BGRA<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void RGB2BGRAapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2BGRA);
    }

    void RGBA2RGBapply()
    {
        ppl::cv::arm::RGBA2RGB<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void RGBA2RGBapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2RGB);
    }
    void RGBA2BGRapply()
    {
        ppl::cv::arm::RGBA2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void RGBA2BGRapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2BGR);
    }
    void RGBA2BGRAapply()
    {
        ppl::cv::arm::RGBA2BGRA<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void RGBA2BGRAapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGBA2BGRA);
    }

    void BGR2RGBapply()
    {
        ppl::cv::arm::BGR2RGB<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGR2RGBapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2RGB);
    }
    void BGR2RGBAapply()
    {
        ppl::cv::arm::BGR2RGBA<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGR2RGBAapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2RGBA);
    }
    void BGR2BGRAapply()
    {
        ppl::cv::arm::BGR2BGRA<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGR2BGRAapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2BGRA);
    }

    void BGRA2RGBapply()
    {
        ppl::cv::arm::BGR2RGB<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGRA2RGBapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2RGB);
    }
    void BGRA2RGBAapply()
    {
        ppl::cv::arm::BGRA2RGBA<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGRA2RGBAapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2RGBA);
    }
    void BGRA2BGRapply()
    {
        ppl::cv::arm::BGRA2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGRA2BGRapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGRA2BGR);
    }
};
} // namespace

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2RGBA_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2RGBAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2RGBA_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2RGBAapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2BGR_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2BGR_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2BGRA_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2BGRAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2BGRA_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2BGRAapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGBA2RGB_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGBA2RGBapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGBA2RGB_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGBA2RGBapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGBA2BGR_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGBA2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGBA2BGR_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGBA2BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGBA2BGRA_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGBA2BGRAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGBA2BGRA_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGBA2BGRAapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2RGB_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2RGBapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2RGB_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2RGBapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2RGBA_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2RGBAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2RGBA_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2RGBAapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2BGRA_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2BGRAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2BGRA_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2BGRAapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2RGB_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2RGBapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2RGB_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2RGBapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2RGBA_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2RGBAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2RGBA_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2RGBAapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2BGR_ppl_aarch64(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2BGR_aarch64_opencv(benchmark::State &state)
{
    RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

// pplcv
BENCHMARK_TEMPLATE(BM_RGB2RGBA_ppl_aarch64, float, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2RGBA_ppl_aarch64, uint8_t, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2BGR_ppl_aarch64, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2BGR_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2BGRA_ppl_aarch64, float, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2BGRA_ppl_aarch64, uint8_t, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_RGBA2RGB_ppl_aarch64, float, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2RGB_ppl_aarch64, uint8_t, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2BGR_ppl_aarch64, float, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2BGR_ppl_aarch64, uint8_t, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2BGRA_ppl_aarch64, float, c4, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2BGRA_ppl_aarch64, uint8_t, c4, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_BGR2RGB_ppl_aarch64, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2RGB_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2RGBA_ppl_aarch64, float, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2RGBA_ppl_aarch64, uint8_t, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2BGRA_ppl_aarch64, float, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2BGRA_ppl_aarch64, uint8_t, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_BGRA2RGB_ppl_aarch64, float, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2RGB_ppl_aarch64, uint8_t, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2RGBA_ppl_aarch64, float, c4, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2RGBA_ppl_aarch64, uint8_t, c4, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2BGR_ppl_aarch64, float, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2BGR_ppl_aarch64, uint8_t, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

// opencv
BENCHMARK_TEMPLATE(BM_RGB2RGBA_aarch64_opencv, float, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2RGBA_aarch64_opencv, uint8_t, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2BGR_aarch64_opencv, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2BGR_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2BGRA_aarch64_opencv, float, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2BGRA_aarch64_opencv, uint8_t, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_RGBA2RGB_aarch64_opencv, float, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2RGB_aarch64_opencv, uint8_t, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2BGR_aarch64_opencv, float, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2BGR_aarch64_opencv, uint8_t, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2BGRA_aarch64_opencv, float, c4, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGBA2BGRA_aarch64_opencv, uint8_t, c4, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_BGR2RGB_aarch64_opencv, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2RGB_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2RGBA_aarch64_opencv, float, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2RGBA_aarch64_opencv, uint8_t, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2BGRA_aarch64_opencv, float, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2BGRA_aarch64_opencv, uint8_t, c3, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_BGRA2RGB_aarch64_opencv, float, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2RGB_aarch64_opencv, uint8_t, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2RGBA_aarch64_opencv, float, c4, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2RGBA_aarch64_opencv, uint8_t, c4, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2BGR_aarch64_opencv, float, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2BGR_aarch64_opencv, uint8_t, c4, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
