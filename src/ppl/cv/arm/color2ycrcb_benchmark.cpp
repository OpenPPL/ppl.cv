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
class YCRCB_RGB_Benchmark {
public:
    T *dev_iImage;
    T *dev_oImage;
    int32_t width;
    int32_t height;

    YCRCB_RGB_Benchmark(int32_t width, int32_t height)
        : dev_iImage(nullptr)
        , dev_oImage(nullptr)
        , width(width)
        , height(height)
    {
        dev_iImage = (T *)malloc(sizeof(T) * width * height * input_channels);
        dev_oImage = (T *)malloc(sizeof(T) * width * height * output_channels);
    }

    ~YCRCB_RGB_Benchmark()
    {
        free(dev_iImage);
        free(dev_oImage);
    }

    void RGB2YCRCBapply()
    {
        ppl::cv::arm::RGB2YCrCb<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void RGB2YCRCBapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_RGB2YCrCb);
    }

    void BGR2YCRCBapply()
    {
        ppl::cv::arm::RGB2YCrCb<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGR2YCRCBapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_BGR2YCrCb);
    }

    void YCRCB2RGBapply()
    {
        ppl::cv::arm::YCrCb2RGB<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void YCRCB2RGBapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YCrCb2RGB);
    }

    void YCRCB2BGRapply()
    {
        ppl::cv::arm::YCrCb2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void YCRCB2BGRapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YCrCb2BGR);
    }
};
} // namespace
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2YCRCB_ppl_aarch64(benchmark::State &state)
{
    YCRCB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2YCRCBapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_RGB2YCRCB_aarch64_opencv(benchmark::State &state)
{
    YCRCB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.RGB2YCRCBapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2YCRCB_ppl_aarch64(benchmark::State &state)
{
    YCRCB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2YCRCBapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2YCRCB_aarch64_opencv(benchmark::State &state)
{
    YCRCB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2YCRCBapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YCRCB2RGB_ppl_aarch64(benchmark::State &state)
{
    YCRCB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YCRCB2RGBapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YCRCB2RGB_aarch64_opencv(benchmark::State &state)
{
    YCRCB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YCRCB2RGBapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YCRCB2BGR_ppl_aarch64(benchmark::State &state)
{
    YCRCB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YCRCB2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YCRCB2BGR_aarch64_opencv(benchmark::State &state)
{
    YCRCB_RGB_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YCRCB2BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

// pplcv
BENCHMARK_TEMPLATE(BM_RGB2YCRCB_ppl_aarch64, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2YCRCB_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2YCRCB_ppl_aarch64, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2YCRCB_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YCRCB2RGB_ppl_aarch64, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YCRCB2RGB_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YCRCB2BGR_ppl_aarch64, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YCRCB2BGR_ppl_aarch64, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

// opencv
BENCHMARK_TEMPLATE(BM_RGB2YCRCB_aarch64_opencv, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_RGB2YCRCB_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2YCRCB_aarch64_opencv, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2YCRCB_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YCRCB2RGB_aarch64_opencv, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YCRCB2RGB_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YCRCB2BGR_aarch64_opencv, float, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YCRCB2BGR_aarch64_opencv, uint8_t, c3, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
