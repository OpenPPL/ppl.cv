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

#include "ppl/cv/arm/warpaffine.h"
#include "ppl/cv/debug.h"
#include <memory>
#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>
#include "ppl/cv/types.h"

namespace {
template <typename T, int32_t val>
void randomRangeData(T* data, const size_t num, int32_t maxNum = 255)
{
    size_t tmp;

    for (size_t i = 0; i < num; i++) {
        tmp     = rand() % maxNum;
        data[i] = (T)((float)tmp / (float)val);
    }
}

template <typename T, int32_t channels, int32_t mode, int32_t borderType>
class WarpAffineBenchmark {
public:
    T* dev_iImage;
    T* dev_oImage;
    int32_t inHeight;
    int32_t inWidth;
    int32_t outHeight;
    int32_t outWidth;
    float affineMatrix0[6];
    float* affineMatrix;
    cv::Mat src_opencv;
    cv::Mat dst_opencv;
    cv::Mat affineMatrix_opencv;

    WarpAffineBenchmark(int32_t inWidth, int32_t inHeight, int32_t outWidth, int32_t outHeight)
        : inHeight(inHeight)
        , inWidth(inWidth)
        , outHeight(outHeight)
        , outWidth(outWidth)
        , affineMatrix0{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}
    {
        const int32_t N0 = inWidth * inHeight * channels;
        const int32_t N1 = outWidth * outHeight * channels;
        dev_iImage   = (T*)malloc(N0 * sizeof(T));
        dev_oImage   = (T*)malloc(N1 * sizeof(T));
        memset(dev_iImage, 0, N0 * sizeof(T));
        affineMatrix = (float*)malloc(6 * sizeof(float));
        // randomRangeData<T, 1>(dev_iImage, N0);
        // randomRangeData<float, 128>(affineMatrix0, 6);

        memcpy(this->affineMatrix, this->affineMatrix0, 6 * sizeof(float));

        cv::Mat src_m(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage);
        cv::Mat dst_m(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage);
        cv::Mat affineMatrix_m(2, 3, CV_32FC1, affineMatrix);
        src_opencv          = src_m.clone();
        dst_opencv          = dst_m.clone();
        affineMatrix_opencv = affineMatrix_m.clone();
    }

    void apply()
    {
        if (mode == ppl::cv::INTERPOLATION_LINEAR) {
            ppl::cv::arm::WarpAffineLinear<T, channels>(this->inHeight,
                                                            this->inWidth,
                                                            this->inWidth * channels,
                                                            this->dev_iImage,
                                                            this->outHeight,
                                                            this->outWidth,
                                                            this->outWidth * channels,
                                                            this->dev_oImage,
                                                            this->affineMatrix,
                                                            (ppl::cv::BorderType)borderType);
        } else {
            ppl::cv::arm::WarpAffineNearestPoint<T, channels>(this->inHeight,
                                                                  this->inWidth,
                                                                  this->inWidth * channels,
                                                                  this->dev_iImage,
                                                                  this->outHeight,
                                                                  this->outWidth,
                                                                  this->outWidth * channels,
                                                                  this->dev_oImage,
                                                                  this->affineMatrix,
                                                                  (ppl::cv::BorderType)borderType);
        }
    }

    void apply_opencv()
    {
        cv::setNumThreads(0);
        if (mode == ppl::cv::INTERPOLATION_LINEAR) {
            cv::warpAffine(src_opencv, dst_opencv, affineMatrix_opencv, dst_opencv.size(), 17 /*CV_WARP_INVERSE_MAP + CV_INTER_LINEAR*/, borderType);
        } else {
            cv::warpAffine(src_opencv, dst_opencv, affineMatrix_opencv, dst_opencv.size(), 16 /*CV_WARP_INVERSE_MAP + cv::INTER_NEAREST*/, borderType);
        }
    }

    ~WarpAffineBenchmark()
    {
        free(this->dev_iImage);
        free(this->dev_oImage);
        free(this->affineMatrix);
    }
};

using namespace ppl::cv;

template <typename T, int32_t channels, int32_t mode, int32_t borderType>
static void BM_WarpAffine_ppl_arm(benchmark::State& state)
{
    WarpAffineBenchmark<T, channels, mode, borderType> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _ : state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
}

template <typename T, int32_t channels, int32_t mode, int32_t borderType>
static void BM_WarpAffine_opencv_arm(benchmark::State& state)
{
    WarpAffineBenchmark<T, channels, mode, borderType> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _ : state) {
        bm.apply_opencv();
    }
    state.SetItemsProcessed(state.iterations());
}

using namespace ppl::cv::debug;
#define c1 1
#define c2 2
#define c3 3
#define c4 4
//linear float
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c1,   INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c3,   INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c4,   INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c1,   INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c3,   INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c4,   INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c1,   INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c3,   INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c4,   INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// // linear float
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c1,   INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c3,   INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c4,   INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c1,   INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c3,   INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c4,   INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c1,   INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c3,   INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c4,   INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

//linear float
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c1,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c3,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c4,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c1,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c3,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c4,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c1,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c3,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, float, c4,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// // linear float
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c1,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c3,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c4,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c1,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c3,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c4,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c1,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c3,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, float, c4,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c1, INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c3, INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c4, INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c1, INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c3, INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c4, INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c1, INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c3, INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c4, INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// linear float
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c1, INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c3, INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c4, INTERPOLATION_NEAREST_POINT, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c1, INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c3, INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c4, INTERPOLATION_NEAREST_POINT, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c1, INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c3, INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c4, INTERPOLATION_NEAREST_POINT, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c1,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c3,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c4,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c1,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c3,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c4,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c1,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c3,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_arm, uchar, c4,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// // linear float
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c1,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c3,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c4,   INTERPOLATION_LINEAR, BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c1,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c3,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c4,   INTERPOLATION_LINEAR, BORDER_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c1,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c3,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
// BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_arm, uchar, c4,   INTERPOLATION_LINEAR, BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

} // namespace
