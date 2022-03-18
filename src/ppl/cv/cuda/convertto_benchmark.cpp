/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "ppl/cv/cuda/convertto.h"

#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename Tsrc, typename Tdst, int channels>
void BM_ConvertTo_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth,
                          channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<Tdst>::depth,
              channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  float alpha = 3.f;
  float beta  = 10.f;

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::ConvertTo<Tsrc, Tdst, channels>(0, gpu_src.rows,
        gpu_src.cols, gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data,
        gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data, alpha, beta);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::ConvertTo<Tsrc, Tdst, channels>(0, gpu_src.rows,
          gpu_src.cols, gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data,
          gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data, alpha, beta);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename Tsrc, typename Tdst, int channels>
void BM_ConvertTo_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth,
                          channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<Tdst>::depth,
              channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  float alpha = 3.f;
  float beta  = 10.f;

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    gpu_src.convertTo(gpu_dst, gpu_dst.type(), alpha, beta);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      gpu_src.convertTo(gpu_dst, gpu_dst.type(), alpha, beta);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename Tsrc, typename Tdst, int channels>
void BM_ConvertTo_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<Tsrc>::depth,
                          channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<Tdst>::depth,
              channels));

  float alpha = 3.f;
  float beta  = 10.f;

  for (auto _ : state) {
    src.convertTo(dst, dst.type(), alpha, beta);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(Tsrc, Tdst, width, height)                              \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c1)->                 \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c1)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c3)->                 \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c3)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c4)->                 \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c4)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(uchar, uchar, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 1920, 1080)
// RUN_BENCHMARK0(uchar, float, 640, 480)
// RUN_BENCHMARK0(uchar, float, 1920, 1080)
// RUN_BENCHMARK0(float, uchar, 640, 480)
// RUN_BENCHMARK0(float, uchar, 1920, 1080)
// RUN_BENCHMARK0(float, float, 640, 480)
// RUN_BENCHMARK0(float, float, 1920, 1080)

#define RUN_BENCHMARK1(Tsrc, Tdst, width, height)                              \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_x86_cuda, Tsrc, Tdst, c1)->             \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c1)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_x86_cuda, Tsrc, Tdst, c3)->             \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c3)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_x86_cuda, Tsrc, Tdst, c4)->             \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c4)->                    \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(uchar, uchar, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 1920, 1080)
// RUN_BENCHMARK1(uchar, float, 640, 480)
// RUN_BENCHMARK1(uchar, float, 1920, 1080)
// RUN_BENCHMARK1(float, uchar, 640, 480)
// RUN_BENCHMARK1(float, uchar, 1920, 1080)
// RUN_BENCHMARK1(float, float, 640, 480)
// RUN_BENCHMARK1(float, float, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(Tsrc, Tdst)                                  \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c1)->                 \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c3)->                 \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c4)->                 \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c1)->                 \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c3)->                 \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_ConvertTo_opencv_cuda, Tsrc, Tdst, c4)->                 \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(Tsrc, Tdst)                                  \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c1)->Args({640, 480})->  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c3)->Args({640, 480})->  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c4)->Args({640, 480})->  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c1)->Args({1920, 1080})->\
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c3)->Args({1920, 1080})->\
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_ConvertTo_ppl_cuda, Tsrc, Tdst, c4)->Args({1920, 1080})->\
                   UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, float)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, float)
RUN_OPENCV_TYPE_FUNCTIONS(float, uchar)
RUN_OPENCV_TYPE_FUNCTIONS(float, uchar)
RUN_OPENCV_TYPE_FUNCTIONS(float, float)
RUN_OPENCV_TYPE_FUNCTIONS(float, float)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, float)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, float)
RUN_PPL_CV_TYPE_FUNCTIONS(float, uchar)
RUN_PPL_CV_TYPE_FUNCTIONS(float, uchar)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float)
