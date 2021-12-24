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

#include "ppl/cv/x86/warpperspective.h"
#include "ppl/cv/debug.h"
#include <benchmark/benchmark.h>
#include <memory>
namespace {

template <typename T, int32_t channels, int32_t mode, ppl::cv::BorderType border_type>
class WarpperspectiveBenchmark {
public:
    T* dev_iImage       = nullptr;
    T* dev_oImage       = nullptr;
    double* inv_warpMat = nullptr;
    int32_t inWidth;
    int32_t inHeight;
    int32_t outWidth;
    int32_t outHeight;
    WarpperspectiveBenchmark(int32_t inWidth, int32_t inHeight, int32_t outWidth, int32_t outHeight)
        : inWidth(inWidth)
        , inHeight(inHeight)
        , outWidth(outWidth)
        , outHeight(outHeight)
    {
        dev_iImage  = (T*)malloc(inWidth * inHeight * channels * sizeof(T));
        dev_oImage  = (T*)malloc(outWidth * outHeight * channels * sizeof(T));
        inv_warpMat = (double*)malloc(6 * sizeof(double));
        memset(this->dev_iImage, 0, inWidth * inHeight * channels * sizeof(T));
        memset(this->dev_oImage, 0, outWidth * outHeight * channels * sizeof(T));
        ppl::cv::debug::randomFill<T>(this->dev_iImage, inWidth * inHeight * channels, 0, 255);
        ppl::cv::debug::randomFill<double>(inv_warpMat, 6, 0, 2);
    }

    void apply()
    {
        if (mode == ppl::cv::INTERPOLATION_TYPE_LINEAR) {
            ppl::cv::x86::WarpPerspectiveLinear<T, channels>(this->inHeight, this->inWidth, this->inWidth * channels, this->dev_iImage, this->outHeight, this->outWidth, this->outWidth * channels, this->dev_oImage, this->inv_warpMat, border_type);
        } else if (mode == ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT) {
            ppl::cv::x86::WarpPerspectiveNearestPoint<T, channels>(this->inHeight, this->inWidth, this->inWidth * channels, this->dev_iImage, this->outHeight, this->outWidth, this->outWidth * channels, this->dev_oImage, this->inv_warpMat, border_type);
        }
    }

    void apply_opencv()
    {
        cv::BorderTypes cv_border_type;
        if (border_type == ppl::cv::BORDER_TYPE_CONSTANT) {
            cv_border_type = cv::BORDER_CONSTANT;
        } else if (border_type == ppl::cv::BORDER_TYPE_REPLICATE) {
            cv_border_type = cv::BORDER_REPLICATE;
        } else if (border_type == ppl::cv::BORDER_TYPE_TRANSPARENT) {
            cv_border_type = cv::BORDER_TRANSPARENT;
        }
        cv::setNumThreads(0);
        if (mode == ppl::cv::INTERPOLATION_TYPE_LINEAR) {
            cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage, sizeof(T) * inWidth * channels);
            cv::Mat dst_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage, sizeof(T) * outWidth * channels);
            cv::Mat inv_mat(2, 3, CV_64FC1, this->inv_warpMat);
            cv::warpAffine(src_opencv, dst_opencv, inv_mat, dst_opencv.size(), cv::WARP_INVERSE_MAP | cv::INTER_LINEAR, cv_border_type);
        } else if (mode == ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT) {
            cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage, sizeof(T) * inWidth * channels);
            cv::Mat dst_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage, sizeof(T) * outWidth * channels);
            cv::Mat inv_mat(2, 3, CV_64FC1, this->inv_warpMat);
            cv::warpAffine(src_opencv, dst_opencv, inv_mat, dst_opencv.size(), cv::WARP_INVERSE_MAP | cv::INTER_NEAREST, cv_border_type);
        }
    }

    ~WarpperspectiveBenchmark()
    {
        free(this->dev_iImage);
        free(this->dev_oImage);
    }
};

} // namespace

template <typename T, int32_t channels, int32_t mode, ppl::cv::BorderType border_type>
static void BM_Warpperspective_ppl_x86(benchmark::State& state)
{
    WarpperspectiveBenchmark<T, channels, mode, border_type> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _ : state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
}

template <typename T, int32_t channels, int32_t mode, ppl::cv::BorderType border_type>
static void BM_Warpperspective_opencv_x86(benchmark::State& state)
{
    WarpperspectiveBenchmark<T, channels, mode, border_type> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _ : state) {
        bm.apply_opencv();
    }
    state.SetItemsProcessed(state.iterations());
}

using namespace ppl::cv::debug;
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_ppl_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

// opencv
#ifdef PPL3CV_BENCHMARK_OPENCV
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_LINEAR, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});

BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, float, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c1, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c3, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
BENCHMARK_TEMPLATE(BM_Warpperspective_opencv_x86, uint8_t, c4, ppl::cv::INTERPOLATION_TYPE_NEAREST_POINT, ppl::cv::BORDER_TYPE_TRANSPARENT)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480})->Args({1280, 720, 1280, 720})->Args({1920, 1080, 1920, 1080})->Args({3840, 2160, 3840, 2160});
#endif //! PPL3CV_BENCHMARK_OPENCV