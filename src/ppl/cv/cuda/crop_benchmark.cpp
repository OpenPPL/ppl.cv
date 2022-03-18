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

#include "ppl/cv/cuda/crop.h"

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, int left, int top, int int_scale>
void BM_Crop_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  int src_width  = width * 2;
  int src_height = height * 2;
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  float scale = int_scale / 10.f;

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Crop<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.rows, gpu_dst.cols,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, left, top, scale);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::Crop<T, channels>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.rows,
          gpu_dst.cols, gpu_dst.step / sizeof(T), (T*)gpu_dst.data, left, top,
          scale);
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

template <typename T, int channels, int left, int top, int int_scale>
void BM_Crop_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  int src_width  = width * 2;
  int src_height = height * 2;
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

  float scale = int_scale / 10.f;

  for (auto _ : state) {
    cv::Rect roi(left, top, width, height);
    cv::Mat croppedImage = src(roi);
    croppedImage.copyTo(dst);
    dst = dst * scale;
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, top, left, scale, width, height)              \
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86_cuda, uchar, channels, top, left,        \
                   scale)->Args({width, height});                              \
BENCHMARK_TEMPLATE(BM_Crop_ppl_cuda, uchar, channels, top, left, scale)->      \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86_cuda, float, channels, top, left,        \
                   scale)->Args({width, height});                              \
BENCHMARK_TEMPLATE(BM_Crop_ppl_cuda, float, channels, top, left, scale)->      \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(c1, 16, 16, 10, 640, 480)
// RUN_BENCHMARK0(c3, 16, 16, 10, 640, 480)
// RUN_BENCHMARK0(c4, 16, 16, 10, 640, 480)
// RUN_BENCHMARK0(c1, 16, 16, 10, 1920, 1080)
// RUN_BENCHMARK0(c3, 16, 16, 10, 1920, 1080)
// RUN_BENCHMARK0(c4, 16, 16, 10, 1920, 1080)

// RUN_BENCHMARK0(c1, 16, 16, 15, 640, 480)
// RUN_BENCHMARK0(c3, 16, 16, 15, 640, 480)
// RUN_BENCHMARK0(c4, 16, 16, 15, 640, 480)
// RUN_BENCHMARK0(c1, 16, 16, 15, 1920, 1080)
// RUN_BENCHMARK0(c3, 16, 16, 15, 1920, 1080)
// RUN_BENCHMARK0(c4, 16, 16, 15, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, top, left, scale)                      \
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86_cuda, type, c1, top, left, scale)->      \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86_cuda, type, c3, top, left, scale)->      \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86_cuda, type, c4, top, left, scale)->      \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86_cuda, type, c1, top, left, scale)->      \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86_cuda, type, c3, top, left, scale)->      \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_Crop_opencv_x86_cuda, type, c4, top, left, scale)->      \
                   Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, top, left, scale)                      \
BENCHMARK_TEMPLATE(BM_Crop_ppl_cuda, type, c1, top, left, scale)->             \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Crop_ppl_cuda, type, c3, top, left, scale)->             \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Crop_ppl_cuda, type, c4, top, left, scale)->             \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Crop_ppl_cuda, type, c1, top, left, scale)->             \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Crop_ppl_cuda, type, c3, top, left, scale)->             \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Crop_ppl_cuda, type, c4, top, left, scale)->             \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, 16, 16, 10)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 16, 16, 15)
RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, 10)
RUN_OPENCV_TYPE_FUNCTIONS(float, 16, 16, 15)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 16, 16, 10)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 16, 16, 15)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, 10)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 16, 16, 15)
