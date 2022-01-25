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
#include "ppl/cv/arm/resize.h"
#include "ppl/cv/debug.h"
#include "ppl/cv/types.h"

namespace {

template <typename T, int channels, int mode>
class ResizeBenchmark {
public:
    T* dev_iImage = nullptr;
    T* dev_oImage = nullptr;
    int inWidth;
    int inHeight;
    int outWidth;
    int outHeight;
    ResizeBenchmark(int inWidth, int inHeight, int outWidth, int outHeight)
        : inWidth(inWidth)
        , inHeight(inHeight)
        , outWidth(outWidth)
        , outHeight(outHeight)
    {
        dev_iImage = (T*)malloc(inWidth * inHeight * channels * sizeof(T));
        dev_oImage = (T*)malloc(outWidth * outHeight * channels * sizeof(T));
        memset(this->dev_iImage, 0, inWidth * inHeight * channels * sizeof(T));
    }

    void apply()
    {
        if (mode == ppl::cv::INTERPOLATION_LINEAR) {
            ppl::cv::arm::ResizeLinear<T, channels>(this->inHeight,
                                                        this->inWidth,
                                                        this->inWidth * channels,
                                                        this->dev_iImage,
                                                        this->outHeight,
                                                        this->outWidth,
                                                        this->outWidth * channels,
                                                        this->dev_oImage);
        } else if (mode == ppl::cv::INTERPOLATION_NEAREST_POINT) {
            ppl::cv::arm::ResizeNearestPoint<T, channels>(this->inHeight,
                                                              this->inWidth,
                                                              this->inWidth * channels,
                                                              this->dev_iImage,
                                                              this->outHeight,
                                                              this->outWidth,
                                                              this->outWidth * channels,
                                                              this->dev_oImage);
        } else if (mode == ppl::cv::INTERPOLATION_AREA) {
            ppl::cv::arm::ResizeArea<T, channels>(this->inHeight,
                                                      this->inWidth,
                                                      this->inWidth * channels,
                                                      this->dev_iImage,
                                                      this->outHeight,
                                                      this->outWidth,
                                                      this->outWidth * channels,
                                                      this->dev_oImage);
        }
    }

    void apply_opencv()
    {
        cv::setNumThreads(0);
        if (mode == ppl::cv::INTERPOLATION_LINEAR) {
            cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage);
            cv::Mat dst_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage);

            cv::resize(src_opencv, dst_opencv, cv::Size(outWidth, outHeight), 0, 0, cv::INTER_LINEAR);
        } else if (mode == ppl::cv::INTERPOLATION_NEAREST_POINT) {
            cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage);
            cv::Mat dst_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage);

            cv::resize(src_opencv, dst_opencv, cv::Size(outWidth, outHeight), 0, 0, cv::INTER_NEAREST);
        } else if (mode == ppl::cv::INTERPOLATION_AREA) {
            cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage);
            cv::Mat dst_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage);

            cv::resize(src_opencv, dst_opencv, cv::Size(outWidth, outHeight), 0, 0, cv::INTER_AREA);
        }
    }

    ~ResizeBenchmark()
    {
        free(this->dev_iImage);
        free(this->dev_oImage);
    }
};

} // namespace

template <typename T, int channels, int mode>
static void BM_Resize_ppl_arm(benchmark::State& state)
{
    ResizeBenchmark<T, channels, mode> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _ : state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
}

template <typename T, int channels, int mode>
static void BM_Resize_opencv_arm(benchmark::State& state)
{
    ResizeBenchmark<T, channels, mode> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _ : state) {
        bm.apply_opencv();
    }
    state.SetItemsProcessed(state.iterations());
}

using namespace ppl::cv::debug;
using ppl::cv::INTERPOLATION_AREA;
using ppl::cv::INTERPOLATION_LINEAR;
using ppl::cv::INTERPOLATION_NEAREST_POINT;

BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c1, INTERPOLATION_LINEAR)->Args({1080, 1920, 180, 320})->Args({1080, 1920, 270, 480});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c1, INTERPOLATION_LINEAR)->Args({1080, 1920, 180, 320})->Args({1080, 1920, 270, 480});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c3, INTERPOLATION_LINEAR)->Args({1080, 1920, 180, 320})->Args({1080, 1920, 270, 480});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c3, INTERPOLATION_LINEAR)->Args({1080, 1920, 180, 320})->Args({1080, 1920, 270, 480});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c4, INTERPOLATION_LINEAR)->Args({1080, 1920, 180, 320})->Args({1080, 1920, 270, 480});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c4, INTERPOLATION_LINEAR)->Args({1080, 1920, 180, 320})->Args({1080, 1920, 270, 480});

//pplcv
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c1, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c3, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c4, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
//opencv
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c1, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c3, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c4, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
//pplcv
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c1, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c3, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c4, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
//opencv
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c1, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c3, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c4, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
//pplcv
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c1, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c3, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, float, c4, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
//opencv
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c1, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c3, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, float, c4, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});

BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c1, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c3, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c4, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
//opencv
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c1, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c3, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c4, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});

BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c1, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c3, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c4, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});

BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c1, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c3, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c4, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});

BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c1, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c3, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_ppl_arm, uint8_t, c4, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});

BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c1, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c3, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
BENCHMARK_TEMPLATE(BM_Resize_opencv_arm, uint8_t, c4, INTERPOLATION_AREA)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240})->Args({1280, 720, 800, 600})->Args({800, 600, 1280, 720});
