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
template <typename T, int32_t input_channels, int32_t output_channels, int32_t mode>
class YUV_RGB_Benchmark {
public:
    T *dev_iImage;
    T *dev_oImage;
    int32_t width;
    int32_t height;

    YUV_RGB_Benchmark(int32_t width, int32_t height)
        : dev_iImage(nullptr)
        , dev_oImage(nullptr)
        , width(width)
        , height(height)
    {
        if (mode == 2) {
            dev_iImage = (T *)malloc(sizeof(T) * width * height * input_channels * 3 / 2);
            dev_oImage = (T *)malloc(sizeof(T) * width * height * output_channels);
        } else {
            dev_iImage = (T *)malloc(sizeof(T) * width * height * input_channels);
            dev_oImage = (T *)malloc(sizeof(T) * width * height * output_channels);
        }
    }

    ~YUV_RGB_Benchmark()
    {
        free(dev_iImage);
        free(dev_oImage);
    }

    void YUV2GRAYapply()
    {
        ppl::cv::arm::YUV2GRAY<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void YUV2GRAYapply_opencv()
    {
        cv::Mat src_opencv(height * 3 / 2, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2GRAY_I420);
    }

    void UYVY2GRAYapply()
    {
        ppl::cv::arm::UYVY2GRAY<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void UYVY2GRAYapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2GRAY_UYVY);
    }

    void YUYV2GRAYapply()
    {
        ppl::cv::arm::YUYV2GRAY<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void YUYV2GRAYapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2GRAY_YUYV);
    }

    void YUV2BGRapply()
    {
        ppl::cv::arm::YUV2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void YUV2BGRapply_opencv()
    {
        cv::Mat src_opencv(height * 3 / 2, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_I420);
    }

    void UYVY2BGRapply()
    {
        ppl::cv::arm::UYVY2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void UYVY2BGRapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_UYVY);
    }

    void YUYV2BGRapply()
    {
        ppl::cv::arm::YUYV2BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void YUYV2BGRapply_opencv()
    {
        cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, input_channels), dev_iImage, sizeof(T) * width * input_channels);
        cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, output_channels), dev_oImage, sizeof(T) * width * output_channels);

        cv::cvtColor(src_opencv, dst_opencv, cv::COLOR_YUV2BGR_YUYV);
    }
};
} // namespace

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YUV2GRAY_ppl_aarch64(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 2> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YUV2GRAYapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YUV2GRAY_aarch64_opencv(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 2> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YUV2GRAYapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_UYVY2GRAY_ppl_aarch64(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 1> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.UYVY2GRAYapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_UYVY2GRAY_aarch64_opencv(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 1> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.UYVY2GRAYapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YUYV2GRAY_ppl_aarch64(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 1> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YUYV2GRAYapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YUYV2GRAY_aarch64_opencv(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 1> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YUYV2GRAYapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YUV2BGR_ppl_aarch64(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 2> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YUV2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YUV2BGR_aarch64_opencv(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 2> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YUV2BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_UYVY2BGR_ppl_aarch64(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 1> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.UYVY2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_UYVY2BGR_aarch64_opencv(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 1> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.UYVY2BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YUYV2BGR_ppl_aarch64(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 1> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YUYV2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_YUYV2BGR_aarch64_opencv(benchmark::State &state)
{
    YUV_RGB_Benchmark<T, input_channels, output_channels, 1> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.YUYV2BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

constexpr int32_t c1 = 1;
constexpr int32_t c2 = 2;
constexpr int32_t c3 = 3;

// pplcv
BENCHMARK_TEMPLATE(BM_YUV2GRAY_ppl_aarch64, uint8_t, c1, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_UYVY2GRAY_ppl_aarch64, uint8_t, c2, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YUYV2GRAY_ppl_aarch64, uint8_t, c2, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YUV2BGR_ppl_aarch64, uint8_t, c1, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_UYVY2BGR_ppl_aarch64, uint8_t, c2, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YUYV2BGR_ppl_aarch64, uint8_t, c2, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

// opencv
BENCHMARK_TEMPLATE(BM_YUV2GRAY_aarch64_opencv, uint8_t, c1, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_UYVY2GRAY_aarch64_opencv, uint8_t, c2, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YUYV2GRAY_aarch64_opencv, uint8_t, c2, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YUV2BGR_aarch64_opencv, uint8_t, c1, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_UYVY2BGR_aarch64_opencv, uint8_t, c2, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_YUYV2BGR_aarch64_opencv, uint8_t, c2, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});