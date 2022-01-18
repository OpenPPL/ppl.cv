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
#include <opencv2/imgproc.hpp>
#include "ppl/cv/arm/cvtcolor.h"
#include "ppl/cv/debug.h"

namespace {
template <typename T, int32_t input_channels, int32_t output_channels>
class BGR_I420_Benchmark {
public:
    T *dev_iImage;
    T *dev_oImage;
    T *dev_Image_y;
    T *dev_Image_u;
    T *dev_Image_v;
    int32_t width;
    int32_t height;

    BGR_I420_Benchmark(int32_t width, int32_t height)
        : dev_iImage(nullptr)
        , dev_oImage(nullptr)
        , width(width)
        , height(height)
    {
        dev_iImage = (T *)malloc(sizeof(T) * width * height * 4); //max malloc
        dev_oImage = (T *)malloc(sizeof(T) * width * height * 4);

        memset(dev_oImage, 0, sizeof(T) * width * height * 4);
        dev_Image_y = (T *)malloc(sizeof(T) * width * height);
        dev_Image_u = (T *)malloc(sizeof(T) * width * height / 4);
        dev_Image_v = (T *)malloc(sizeof(T) * width * height / 4);
        memset(dev_iImage, 0, sizeof(T) * width * height * input_channels);
    }

    ~BGR_I420_Benchmark()
    {
        free(dev_iImage);
        free(dev_oImage);
        free(dev_Image_y);
        free(dev_Image_u);
        free(dev_Image_v);
    }

    void BGR2I420apply()
    {
        ppl::cv::arm::BGR2I420<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGR2I420apply_opencv()
    {
        cv::setNumThreads(0);
        cv::Mat iMat(height, width, CV_8UC3, dev_iImage);
        cv::Mat oMat(height * 3 / 2, width, CV_8UC1, dev_oImage);
        cv::cvtColor(iMat, oMat, cv::COLOR_BGR2YUV_I420);
    }
    void I4202BGRapply()
    {
        ppl::cv::arm::I4202BGR<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void I4202BGRapply_opencv()
    {
        cv::setNumThreads(0);
        cv::Mat iMat(height * 3 / 2, width, CV_8UC1, dev_iImage);
        cv::Mat oMat(height, width, CV_8UC3, dev_oImage);
        cv::cvtColor(iMat, oMat, cv::COLOR_YUV2BGR_I420);
    }
    void BGRA2I420apply()
    {
        ppl::cv::arm::BGRA2I420<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void BGRA2I420apply_opencv()
    {
        cv::setNumThreads(0);
        cv::Mat iMat(height, width, CV_8UC3, dev_iImage);
        cv::Mat oMat(height * 3 / 2, width, CV_8UC1, dev_oImage);
        cv::cvtColor(iMat, oMat, cv::COLOR_BGRA2YUV_I420);
    }
    void I4202BGRAapply()
    {
        ppl::cv::arm::I4202BGRA<T>(height, width, width * input_channels, dev_iImage, width * output_channels, dev_oImage);
    }
    void I4202BGRAapply_opencv()
    {
        cv::setNumThreads(0);
        cv::Mat iMat(height * 3 / 2, width, CV_8UC1, dev_iImage);
        cv::Mat oMat(height, width, CV_8UC3, dev_oImage);
        cv::cvtColor(iMat, oMat, cv::COLOR_YUV2BGRA_I420);
    }

    void BGR2I420_3channelsapply()
    {
        ppl::cv::arm::BGR2I420<T>(
            height,
            width,
            width * input_channels,
            dev_iImage,
            width * 1,
            dev_Image_y,
            width / 2,
            dev_Image_u,
            width / 2,
            dev_Image_v);
    }
    void I420_3channels2BGRapply()
    {
        ppl::cv::arm::I4202BGR<T>(
            height,
            width,
            width * 1,
            dev_Image_y,
            width / 2,
            dev_Image_u,
            width / 2,
            dev_Image_v,
            width * output_channels,
            dev_oImage);
    }
    void BGRA2I420_3channelsapply()
    {
        ppl::cv::arm::BGRA2I420<T>(
            height,
            width,
            width * input_channels,
            dev_iImage,
            width * 1,
            dev_Image_y,
            width / 2,
            dev_Image_u,
            width / 2,
            dev_Image_v);
    }
    void I420_3channels2BGRAapply()
    {
        ppl::cv::arm::I4202BGRA<T>(
            height,
            width,
            width * 1,
            dev_Image_y,
            width / 2,
            dev_Image_u,
            width / 2,
            dev_Image_v,
            width * output_channels,
            dev_oImage);
    }
};
} // namespace
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2I420_ppl_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2I420apply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2I420_opencv_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2I420apply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_I4202BGR_ppl_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.I4202BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_I4202BGR_opencv_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.I4202BGRapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2I420_ppl_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2I420apply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2I420_opencv_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2I420apply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_I4202BGRA_ppl_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.I4202BGRAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_I4202BGRA_opencv_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.I4202BGRAapply_opencv();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGR2I420_3channels_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGR2I420_3channelsapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_I420_3channels2BGR_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.I420_3channels2BGRapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_BGRA2I420_3channels_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.BGRA2I420_3channelsapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}
template <typename T, int32_t input_channels, int32_t output_channels>
void BM_I420_3channels2BGRA_aarch64(benchmark::State &state)
{
    BGR_I420_Benchmark<T, input_channels, output_channels> bm(state.range(0), state.range(1));
    for (auto _ : state) {
        bm.I420_3channels2BGRAapply();
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

constexpr int32_t c1 = 1;
constexpr int32_t c3 = 3;
constexpr int32_t c4 = 4;

BENCHMARK_TEMPLATE(BM_BGR2I420_ppl_aarch64, uint8_t, c3, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGR2I420_opencv_aarch64, uint8_t, c3, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2I420_ppl_aarch64, uint8_t, c4, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_BGRA2I420_opencv_aarch64, uint8_t, c4, c1)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_I4202BGR_ppl_aarch64, uint8_t, c1, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_I4202BGR_opencv_aarch64, uint8_t, c1, c3)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_I4202BGRA_ppl_aarch64, uint8_t, c1, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_I4202BGRA_opencv_aarch64, uint8_t, c1, c4)->Args({320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
