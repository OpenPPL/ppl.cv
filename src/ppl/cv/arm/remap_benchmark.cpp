// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for mulitional information
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
#include "ppl/cv/arm/remap.h"
#include "ppl/cv/debug.h"
#include "ppl/cv/types.h"

namespace {
template<typename T, int32_t channels, int32_t mode>
class RemapBenchmark {
public:
    T* dev_iImage = nullptr;
    T* dev_oImage = nullptr;
    float* dev_mapX = nullptr;
    float* dev_mapY = nullptr;
    int32_t inWidth;
    int32_t inHeight;
    int32_t outWidth;
    int32_t outHeight;
    RemapBenchmark(int32_t inWidth, int32_t inHeight, int32_t outWidth, int32_t outHeight)
        : inWidth(inWidth)
        , inHeight(inHeight)
        , outWidth(outWidth)
        , outHeight(outHeight)
    {
        dev_iImage = (T*)malloc(inWidth * inHeight * channels * sizeof(T));
        dev_oImage = (T*)malloc(outWidth * outHeight * channels * sizeof(T));
        dev_mapX = (float*)malloc(this->outWidth * outHeight * channels * sizeof(float));
        dev_mapY = (float*)malloc(this->outWidth * outHeight * channels * sizeof(float));
        memset(this->dev_iImage, 0, inWidth * inHeight * channels * sizeof(T));
        ppl::cv::debug::randomFill<float>(this->dev_mapX, outWidth * outHeight * 1, -10, inWidth+10);
        ppl::cv::debug::randomFill<float>(this->dev_mapY, outWidth * outHeight * 1, -10, inHeight+10);
    }

    void apply() {
        if (mode == ppl::cv::INTERPOLATION_LINEAR) {
            ppl::cv::arm::RemapLinear<T, channels>(
                                         this->inHeight,
                                         this->inWidth,
                                         this->inWidth * channels,
                                         this->dev_iImage,
                                         this->outHeight,
                                         this->outWidth,
                                         this->outWidth * channels,
                                         this->dev_oImage,
                                         this->dev_mapX,
                                         this->dev_mapY,
                                         ppl::cv::BORDER_CONSTANT);
        } else {
            ppl::cv::arm::RemapNearestPoint<T, channels>(
                                         this->inHeight,
                                         this->inWidth,
                                         this->inWidth * channels,
                                         this->dev_iImage,
                                         this->outHeight,
                                         this->outWidth,
                                         this->outWidth * channels,
                                         this->dev_oImage,
                                         this->dev_mapX,
                                         this->dev_mapY,
                                         ppl::cv::BORDER_CONSTANT);
        }
    }

#ifdef PPLCV_BENCHMARK_OPENCV
    void apply_opencv() {
        cv::Mat src_opencv(inHeight, inWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_iImage, sizeof(T) * inWidth * channels);
        cv::Mat dst_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<T>::depth, channels), dev_oImage, sizeof(T) * outWidth * channels);
        cv::Mat mapy_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<float>::depth, 1), dev_mapX, sizeof(float) * outWidth);
        cv::Mat mapx_opencv(outHeight, outWidth, CV_MAKETYPE(cv::DataType<float>::depth, 1), dev_mapY, sizeof(float) * outWidth);
        cv::Scalar borderValue = {0, 0, 0, 0};
        if (mode == ppl::cv::INTERPOLATION_LINEAR) {
            cv::remap(src_opencv, dst_opencv, mapx_opencv, mapy_opencv, cv::INTER_LINEAR, 
                      cv::BORDER_CONSTANT, borderValue);
        } else {
            cv::remap(src_opencv, dst_opencv, mapx_opencv, mapy_opencv, cv::INTER_NEAREST, 
                      cv::BORDER_CONSTANT, borderValue);
        }
    }
#endif // PPLCV_BENCHMARK_OPENCV

    ~RemapBenchmark() {
        free(this->dev_iImage);
        free(this->dev_oImage);
        free(this->dev_mapX);
        free(this->dev_mapY);
    }
};
}

template<typename T, int32_t channels, int32_t mode>
static void BM_Remap_ppl_aarch64(benchmark::State &state) {
    RemapBenchmark<T, channels, mode> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _: state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
}

using namespace ppl::cv::debug;
using ppl::cv::INTERPOLATION_LINEAR;
using ppl::cv::INTERPOLATION_NEAREST_POINT;

BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, float, c1, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, float, c3, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, float, c4, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, uint8_t, c1, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, uint8_t, c3, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, uint8_t, c4, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});

BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, float, c1, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, float, c3, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, float, c4, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, uint8_t, c1, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, uint8_t, c3, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_ppl_aarch64, uint8_t, c4, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});

#ifdef PPLCV_BENCHMARK_OPENCV
template<typename T, int32_t channels, int32_t mode>
static void BM_Remap_opencv_aarch64(benchmark::State &state) {
    RemapBenchmark<T, channels, mode> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    if (mode == ppl::cv::INTERPOLATION_LINEAR) {
        for (auto _: state) {
            bm.apply_opencv();
        }
    }
    else if (mode == ppl::cv::INTERPOLATION_NEAREST_POINT) {
        for (auto _: state) {
            bm.apply_opencv();
        }
    }
    state.SetItemsProcessed(state.iterations());

}

BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, float, c1, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, float, c3, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, float, c4, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, uint8_t, c1, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, uint8_t, c3, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, uint8_t, c4, INTERPOLATION_LINEAR)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});

BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, float, c1, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, float, c3, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, float, c4, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, uint8_t, c1, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, uint8_t, c3, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
BENCHMARK_TEMPLATE(BM_Remap_opencv_aarch64, uint8_t, c4, INTERPOLATION_NEAREST_POINT)->Args({320, 240, 640, 480})->Args({640, 480, 320, 240});
#endif //! PPLCV_BENCHMARK_OPENCV
