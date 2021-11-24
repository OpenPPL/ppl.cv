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

#include "ppl/cv/cuda/mean.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

enum MaskType {
  kUnmasked,
  kMasked,
};

template <typename T, int channels, bool channel_wise, MaskType mask_type>
void BM_Mean_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask(mask);
  int dst_size = channels * sizeof(float);
  float* gpu_dst;
  cudaMalloc((void**)&gpu_dst, dst_size);

  int iterations = 1000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    Mean<T, channels>(0, gpu_src.rows, gpu_src.cols, gpu_src.step / sizeof(T),
                      (T*)gpu_src.data, gpu_dst, 0, nullptr, channel_wise);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == kUnmasked) {
        Mean<T, channels>(0, gpu_src.rows, gpu_src.cols,
                          gpu_src.step / sizeof(T), (T*)gpu_src.data,
                          gpu_dst, 0, nullptr, channel_wise);
      }
      else {
        Mean<T, channels>(0, gpu_src.rows, gpu_src.cols,
                          gpu_src.step / sizeof(T), (T*)gpu_src.data,
                          gpu_dst, gpu_mask.step / sizeof(uchar),
                          (uchar*)gpu_mask.data, channel_wise);
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  cudaFree(gpu_dst);
}

template <typename T, int channels, bool channel_wise, MaskType mask_type>
void BM_Mean_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::Scalar mean_value;
  cv::Scalar stddev_value;

  int iterations = 1000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::meanStdDev(gpu_src, mean_value, stddev_value);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::meanStdDev(gpu_src, mean_value, stddev_value);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, bool channel_wise, MaskType mask_type>
void BM_Mean_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask = createSourceImage(height, width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  int dst_size = channels * sizeof(float);
  float* dst = (float*)malloc(dst_size);

  for (auto _ : state) {
    if (mask_type == kUnmasked) {
      cv::Scalar mean_value = cv::mean(src);
    }
    else {
      cv::Scalar mean_value = cv::mean(src, mask);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);

  free(dst);
}

#define RUN_BENCHMARK0(width, height)                                          \
BENCHMARK_TEMPLATE(BM_Mean_opencv_cuda, uchar, 1, true, kUnmasked)->           \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, uchar, 1, true, kUnmasked)->              \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(320, 240)
// RUN_BENCHMARK0(640, 480)
// RUN_BENCHMARK0(1280, 720)
// RUN_BENCHMARK0(1920, 1080)

#define RUN_BENCHMARK1(channel_wise, mask_type, width, height)                 \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, uchar, c1, channel_wise,           \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, uchar, c1, channel_wise,                  \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, uchar, c3, channel_wise,           \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, uchar, c3, channel_wise,                  \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, uchar, c4, channel_wise,           \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, uchar, c4, channel_wise,                  \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, float, c1, channel_wise,           \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, float, c1, channel_wise,                  \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, float, c3, channel_wise,           \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, float, c3, channel_wise,                  \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, float, c4, channel_wise,           \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, float, c4, channel_wise,                  \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);

// RUN_BENCHMARK1(true, kUnmasked, 320, 240)
// RUN_BENCHMARK1(true, kUnmasked, 640, 480)
// RUN_BENCHMARK1(true, kUnmasked, 1280, 720)
// RUN_BENCHMARK1(true, kUnmasked, 1920, 1080)
// RUN_BENCHMARK1(true, kMasked, 320, 240)
// RUN_BENCHMARK1(true, kMasked, 640, 480)
// RUN_BENCHMARK1(true, kMasked, 1280, 720)
// RUN_BENCHMARK1(true, kMasked, 1920, 1080)
// RUN_BENCHMARK1(false, kUnmasked, 320, 240)
// RUN_BENCHMARK1(false, kUnmasked, 640, 480)
// RUN_BENCHMARK1(false, kUnmasked, 1280, 720)
// RUN_BENCHMARK1(false, kUnmasked, 1920, 1080)
// RUN_BENCHMARK1(false, kMasked, 320, 240)
// RUN_BENCHMARK1(false, kMasked, 640, 480)
// RUN_BENCHMARK1(false, kMasked, 1280, 720)
// RUN_BENCHMARK1(false, kMasked, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, channels, channel_wise, mask_type)     \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, type, channels, channel_wise,      \
                   mask_type)->Args({320, 240});                               \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, type, channels, channel_wise,      \
                   mask_type)->Args({640, 480});                               \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, type, channels, channel_wise,      \
                   mask_type)->Args({1280, 720});                              \
BENCHMARK_TEMPLATE(BM_Mean_opencv_x86_cuda, type, channels, channel_wise,      \
                   mask_type)->Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, channels, channel_wise, mask_type)     \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, type, channels, channel_wise, mask_type)->\
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, type, channels, channel_wise, mask_type)->\
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, type, channels, channel_wise, mask_type)->\
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_Mean_ppl_cuda, type, channels, channel_wise, mask_type)->\
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, c1, true, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c3, true, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c4, true, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c1, true, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c3, true, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c4, true, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c1, false, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c3, false, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c4, false, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c1, false, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c3, false, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, c4, false, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c1, true, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c3, true, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c4, true, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c1, true, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c3, true, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c4, true, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c1, false, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c3, false, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c4, false, kMasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c1, false, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c3, false, kUnmasked)
RUN_OPENCV_TYPE_FUNCTIONS(float, c4, false, kUnmasked)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c1, true, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c3, true, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c4, true, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c1, true, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c3, true, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c4, true, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c1, false, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c3, false, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c4, false, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c1, false, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c3, false, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, c4, false, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c1, true, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c3, true, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c4, true, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c1, true, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c3, true, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c4, true, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c1, false, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c3, false, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c4, false, kMasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c1, false, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c3, false, kUnmasked)
RUN_PPL_CV_TYPE_FUNCTIONS(float, c4, false, kUnmasked)
