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
#include "ppl/cv/arm/medianblur.h"
#include "ppl/cv/debug.h"

namespace {
template<typename T, int channels, int kernel_len>
class MedianBlurBenchmark {
public:
    int device = 0;
    T* dev_iImage = nullptr;
    T* dev_oImage = nullptr;
    unsigned char* element;
    int height;
    int width;
    MedianBlurBenchmark(int height, int width)
        : height(height)
        , width(width)
    {
        dev_iImage = (T*)malloc(height * width * channels * sizeof(T));
        dev_oImage = (T*)malloc(height * width * channels * sizeof(T));
        memset(this->dev_iImage, 0, height * width * channels * sizeof(T));
    }

    void apply() {
        ppl::cv::arm::MedianBlur<T, channels>(
            this->height,
            this->width,
            this->width * channels,
            this->dev_iImage,
            this->width * channels,
            this->dev_oImage,
            kernel_len,
            ppl::cv::BORDER_REPLICATE);
    }

    ~MedianBlurBenchmark() {
        free(this->dev_oImage);
        free(this->dev_iImage);
    }
};
}

using namespace ppl::cv::debug;
template<typename T, int channels, int kernel>
static void BM_MedianBlur_ppl_arm(benchmark::State &state) {
    MedianBlurBenchmark<T, channels, kernel> bm(state.range(1), state.range(0));
    for (auto _: state) {
        bm.apply();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(T) * channels);
}

BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c1, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c3, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c4, k3x3)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c1, k5x5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c3, k5x5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c4, k5x5)->Args({320, 240})->Args({640, 480}); 
// BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c1, k7x7)->Args({320, 240})->Args({640, 480});
// BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c3, k7x7)->Args({320, 240})->Args({640, 480});
// BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, float, c4, k7x7)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c1, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c3, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c4, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c1, k5x5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c3, k5x5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c4, k5x5)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c1, k7x7)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c3, k7x7)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_arm, uint8_t, c4, k7x7)->Args({320, 240})->Args({640, 480}); 

#ifdef PPLCV_BENCHMARK_OPENCV
#include <opencv2/imgproc.hpp>
template<typename T, int channels, int kernel>
static void BM_MedianBlur_opencv_arm(benchmark::State &state) {
    int width = state.range(0);
    int height = state.range(1);
    std::unique_ptr<T[]> src(new T[width * height * channels]);
    std::unique_ptr<T[]> dst(new T[width * height * channels]);
    ppl::cv::debug::randomFill<T>(src.get(), width * height * channels, 0, 255);
    cv::Mat src_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), src.get(), sizeof(T) * width * channels);
    cv::Mat dst_opencv(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels), dst.get(), sizeof(T) * width * channels);

    for (auto _ : state) {
        cv::medianBlur(src_opencv, dst_opencv, kernel);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c1, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c3, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c4, k3x3)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c1, k5x5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c3, k5x5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c4, k5x5)->Args({320, 240})->Args({640, 480}); 
// BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c1, k7x7)->Args({320, 240})->Args({640, 480});
// BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c3, k7x7)->Args({320, 240})->Args({640, 480});
// BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, float, c4, k7x7)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c1, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c3, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c4, k3x3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c1, k5x5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c3, k5x5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c4, k5x5)->Args({320, 240})->Args({640, 480}); 
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c1, k7x7)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c3, k7x7)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_arm, uint8_t, c4, k7x7)->Args({320, 240})->Args({640, 480}); 
#endif //! PPLCV_BENCHMARK_OPENCV

