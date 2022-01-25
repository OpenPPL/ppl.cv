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
#include "ppl/cv/x86/sobel.h"
#include <memory>
#include "ppl/cv/debug.h"

namespace {
template <typename Tsrc, int channels, typename Tdst>
class SobelBenchmark {
public:
    int height;
    int width;
    Tsrc *inData;
    Tdst *outData;

    SobelBenchmark(int height, int width)
        : height(height)
        , width(width)
    {
        inData  = NULL;
        outData = NULL;

        inData = (Tsrc *)malloc(height * width * channels * sizeof(Tsrc));
        memset(inData, 0, height * width * channels * sizeof(Tsrc));
        outData = (Tdst *)malloc(height * width * channels * sizeof(Tdst));

        if (!inData || !outData) {
            if (inData) {
                free(inData);
            }
            if (outData) {
                free(outData);
            }
            return;
        }
    }

    void apply(int dx, int dy, int ksize)
    {
        ppl::cv::x86::Sobel<Tsrc, Tdst, channels>(height, width, width * channels, inData, width * channels, outData, dx, dy, ksize, 1.0, 0.0, ppl::cv::BORDER_DEFAULT);
    }

    ~SobelBenchmark()
    {
        free(inData);
        free(outData);
    }
};
} // namespace

using namespace ppl::cv::debug;

template <typename Tsrc, int channels, typename Tdst, int dx, int dy, int ksize>
static void BM_Sobel_ppl_x86(benchmark::State &state)
{
    SobelBenchmark<Tsrc, channels, Tdst> bm(state.range(1), state.range(0));
    for (auto _ : state) {
        bm.apply(dx, dy, ksize);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(Tsrc) * channels);
}

BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c1, float, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c1, float, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c1, float, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c1, float, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c1, float, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c1, float, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c1, float, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c1, float, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c3, float, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c3, float, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c3, float, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c3, float, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c3, float, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c3, float, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c3, float, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c3, float, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c4, float, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c4, float, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c4, float, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c4, float, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c4, float, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c4, float, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c4, float, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, float, c4, float, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c1, int16_t, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c1, int16_t, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c1, int16_t, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c1, int16_t, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c1, int16_t, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c1, int16_t, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c1, int16_t, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c1, int16_t, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c3, int16_t, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c3, int16_t, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c3, int16_t, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c3, int16_t, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c3, int16_t, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c3, int16_t, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c3, int16_t, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c3, int16_t, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c4, int16_t, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c4, int16_t, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c4, int16_t, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c4, int16_t, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c4, int16_t, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c4, int16_t, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c4, int16_t, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_ppl_x86, uint8_t, c4, int16_t, 1, 0, 5)->Args({320, 240})->Args({640, 480});

#ifdef PPLCV_BENCHMARK_OPENCV
template <typename Tsrc, int channels, typename Tdst>
class SobelBenchmark_OPENCV {
public:
    int height;
    int width;
    Tsrc *inData;
    Tdst *outData;
    ::cv::Mat iMat;
    ::cv::Mat oMat;

    SobelBenchmark_OPENCV(int height, int width)
        : height(height)
        , width(width)
    {
        inData  = NULL;
        outData = NULL;

        inData = (Tsrc *)malloc(height * width * channels * sizeof(Tsrc));
        memset(inData, 0, height * width * channels * sizeof(Tsrc));
        outData = (Tdst *)malloc(height * width * channels * sizeof(Tdst));

        if (!inData || !outData) {
            if (inData) {
                free(inData);
            }
            if (outData) {
                free(outData);
            }
            return;
        }
        iMat(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels), inData);
        oMat(height, width, CV_MAKETYPE(cv::DataType<Tdst>::depth, channels), outData);
    }

    void apply(int dx, int dy, int ksize)
    {
        ::cv::Sobel(iMat, oMat, oMat.depth(), dx, dy, ksize, 1.0, 0.0, ::cv::BORDER_REFLECT_101);
    }

    ~SobelBenchmark_OPENCV()
    {
        free(inData);
        free(outData);
    }
};

using namespace ppl::cv::debug;

template <typename Tsrc, int channels, typename Tdst, int dx, int dy, int ksize>
static void BM_Sobel_opencv_x86(benchmark::State &state)
{
    SobelBenchmark<Tsrc, channels, Tdst> bm(state.range(1), state.range(0));
    for (auto _ : state) {
        bm.apply(dx, dy, ksize);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * sizeof(Tsrc) * channels);
}

BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c1, float, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c1, float, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c1, float, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c1, float, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c1, float, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c1, float, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c1, float, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c1, float, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c3, float, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c3, float, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c3, float, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c3, float, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c3, float, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c3, float, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c3, float, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c3, float, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c4, float, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c4, float, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c4, float, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c4, float, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c4, float, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c4, float, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c4, float, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, float, c4, float, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c1, int16_t, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c1, int16_t, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c1, int16_t, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c1, int16_t, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c1, int16_t, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c1, int16_t, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c1, int16_t, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c1, int16_t, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c3, int16_t, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c3, int16_t, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c3, int16_t, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c3, int16_t, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c3, int16_t, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c3, int16_t, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c3, int16_t, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c3, int16_t, 1, 0, 5)->Args({320, 240})->Args({640, 480});

BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c4, int16_t, 1, 0, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c4, int16_t, 0, 1, -1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c4, int16_t, 0, 1, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c4, int16_t, 1, 0, 1)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c4, int16_t, 0, 1, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c4, int16_t, 1, 0, 3)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c4, int16_t, 0, 1, 5)->Args({320, 240})->Args({640, 480});
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86, uint8_t, c4, int16_t, 1, 0, 5)->Args({320, 240})->Args({640, 480});

#endif //! PPLCV_BENCHMARK_OPENCV
