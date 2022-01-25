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

#include "ppl/cv/arm/copymakeborder.h"
#include "ppl/cv/debug.h"
#include <benchmark/benchmark.h>
#include <memory>
namespace {

template <typename T, int32_t nc, ppl::cv::BorderType border_type>
void BM_Copymakeborder_ppl_aarch64(benchmark::State &state)
{
    int32_t width         = state.range(0);
    int32_t height        = state.range(1);
    int32_t padding       = 2;
    int32_t input_height  = height;
    int32_t input_width   = width;
    int32_t output_height = height + 2 * padding;
    int32_t output_width  = width + 2 * padding;
    std::unique_ptr<T[]> src(new T[input_width * input_height * nc]);
    std::unique_ptr<T[]> dst(new T[output_width * output_height * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), input_width * input_height * nc, 0, 255);
    for (auto _ : state) {
        ppl::cv::arm::CopyMakeBorder<T, nc>(input_height, input_width, input_width * nc, src.get(), output_height, output_width, output_width * nc, dst.get(), border_type);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

using namespace ppl::cv::debug;
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c1, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c3, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c4, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c1, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c3, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c4, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c1, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c3, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c4, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c1, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c3, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c4, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c1, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c3, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c4, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c1, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c3, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c4, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c1, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c3, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, float, c4, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c1, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c3, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_ppl_aarch64, uint8_t, c4, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

#ifdef PPLCV_BENCHMARK_OPENCV

template <typename T, int32_t nc, ppl::cv::BorderType border_type>
void BM_Copymakeborder_opencv_aarch64(benchmark::State &state)
{
    int32_t width         = state.range(0);
    int32_t height        = state.range(1);
    int32_t padding       = 2;
    int32_t input_height  = height;
    int32_t input_width   = width;
    int32_t output_height = height + 2 * padding;
    int32_t output_width  = width + 2 * padding;
    std::unique_ptr<T[]> src(new T[input_width * input_height * nc]);
    std::unique_ptr<T[]> dst(new T[output_width * output_height * nc]);
    std::unique_ptr<T[]> dst_ref(new T[output_height * output_width * nc]);
    ppl::cv::debug::randomFill<T>(src.get(), input_width * input_height * nc, 0, 255);
    cv::Mat src_opencv(input_height, input_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), src.get(), sizeof(T) * input_width * nc);
    cv::Mat dst_opencv(output_height, output_width, CV_MAKETYPE(cv::DataType<T>::depth, nc), dst_ref.get(), sizeof(T) * output_width * nc);
    cv::BorderTypes cv_border_type;
    if (border_type == ppl::cv::BORDER_CONSTANT) {
        cv_border_type = cv::BORDER_CONSTANT;
    } else if (border_type == ppl::cv::BORDER_REPLICATE) {
        cv_border_type = cv::BORDER_REPLICATE;
    } else if (border_type == ppl::cv::BORDER_REFLECT) {
        cv_border_type = cv::BORDER_REFLECT;
    } else if (border_type == ppl::cv::BORDER_REFLECT101) {
        cv_border_type = cv::BORDER_REFLECT101;
    }
    for (auto _ : state) {
        cv::copyMakeBorder(src_opencv, dst_opencv, padding, padding, padding, padding, cv_border_type);
    }
    state.SetItemsProcessed(state.iterations() * 1);
}

BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c1, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c3, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c4, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c1, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c3, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c4, ppl::cv::BORDER_CONSTANT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c1, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c3, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c4, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c1, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c3, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c4, ppl::cv::BORDER_REPLICATE)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c1, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c3, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c4, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c1, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c3, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c4, ppl::cv::BORDER_REFLECT)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});

BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c1, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c3, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, float, c4, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c1, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c3, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
BENCHMARK_TEMPLATE(BM_Copymakeborder_opencv_aarch64, uint8_t, c4, ppl::cv::BORDER_REFLECT101)->Args({320, 240, 320, 240})->Args({640, 480})->Args({1280, 720})->Args({1920, 1080})->Args({3840, 2160});
#endif //! PPLCV_BENCHMARK_OPENCV
} // namespace
