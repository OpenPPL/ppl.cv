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
#include "ppl/cv/x86/rotate.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {
template <typename T, int channels>
class RotateBenchmark {
public:
    int height;
    int width;
    T* dev_iImage;
    T* dev_oImage;
    int outHeight;
    int outWidth;

    RotateBenchmark(int height, int width, int outHeight, int outWidth)
        : height(height)
        , width(width)
        , outHeight(outHeight)
        , outWidth(outWidth)
    {
        dev_iImage = (T*)malloc(width * height * channels * sizeof(T));
        dev_oImage = (T*)malloc(width * height * channels * sizeof(T));
        memset(dev_iImage, 0, width * height * channels * sizeof(T));
    }

    void apply(int degree)
    {
        ppl::cv::x86::RotateNx90degree<T, channels>(this->height,
                                                    this->width,
                                                    this->width * channels,
                                                    this->dev_iImage,
                                                    this->outHeight,
                                                    this->outWidth,
                                                    this->outWidth * channels,
                                                    this->dev_oImage,
                                                    degree);
    }

    /*
    void applyNV12orNV21(int degree) {
        ppl::cv::x86::RotateNx90degree_YUV420<T>(this->height,
                                                 this->width,
                                                 this->width * 3,
                                                 this->dev_iImage,
                                                 this->outHeight,
                                                 this->outWidth,
                                                 this->outWidth * 3,
                                                 this->dev_oImage,
                                                 degree,
                                                 0);
    }

    void applyI420(int degree) {
        ppl::cv::x86::RotateNx90degree_YUV420<T>(this->height,
                                                 this->width,
                                                 this->width * 3,
                                                 this->dev_iImage,
                                                 this->outHeight,
                                                 this->outWidth,
                                                 this->outWidth * 3,
                                                 this->dev_oImage,
                                                 degree,
                                                 1);
    }
    */

    ~RotateBenchmark()
    {
        free(this->dev_iImage);
        free(this->dev_oImage);
    }
};
} // namespace

using namespace ppl::cv::debug;

template <typename T, int channels, int degree>
static void BM_Rotate_ppl_x86(benchmark::State& state)
{
    RotateBenchmark<T, channels> bm(state.range(0), state.range(1), state.range(2), state.range(3));
    for (auto _ : state) {
        bm.apply(degree);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c1, 90)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c1, 180)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c1, 270)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c2, 90)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c2, 180)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c2, 270)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c3, 90)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c3, 180)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c3, 270)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c4, 90)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c4, 180)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, float, c4, 270)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c1, 90)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c1, 180)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c1, 270)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c2, 90)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c2, 180)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c2, 270)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c3, 90)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c3, 180)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c3, 270)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c4, 90)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c4, 180)->Args({320, 240, 320, 240})->Args({640, 480, 640, 480});
BENCHMARK_TEMPLATE(BM_Rotate_ppl_x86, uint8_t, c4, 270)->Args({320, 240, 240, 320})->Args({640, 480, 480, 640});
