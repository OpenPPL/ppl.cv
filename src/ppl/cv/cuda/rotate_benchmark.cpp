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

#include "ppl/cv/cuda/rotate.h"

#include "opencv2/core.hpp"
#include "opencv2/cudawarping.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, int degree>
void BM_Rotate_ppl_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_height, dst_width;
  if (degree == 90) {
    dst_height = src_width;
    dst_width  = src_height;
  }
  else if (degree == 180) {
    dst_height = src_height;
    dst_width  = src_width;
  }
  else if (degree == 270) {
    dst_height = src_width;
    dst_width  = src_height;
  }
  else {
    return;
  }

  cv::Mat src, dst;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst_height, dst_width, src.type());

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Rotate<T, channels>(0, src_height, src_width,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, degree);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::Rotate<T, channels>(0, src_height, src_width,
          gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
          gpu_dst.step / sizeof(T), (T*)gpu_dst.data, degree);
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

template <typename T, int channels, int degree>
void BM_Rotate_opencv_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_height, dst_width;
  if (degree == 90) {
    dst_height = src_width;
    dst_width  = src_height;
  }
  else if (degree == 180) {
    dst_height = src_height;
    dst_width  = src_width;
  }
  else if (degree == 270) {
    dst_height = src_width;
    dst_width  = src_height;
  }
  else {
    return;
  }

  cv::Mat src, dst;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst_height, dst_width, src.type());

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::rotate(gpu_src, gpu_dst, cv::Size(dst_width, dst_height), degree);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::rotate(gpu_src, gpu_dst, cv::Size(dst_width, dst_height),
                       degree);
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

template <typename T, int channels, int degree>
void BM_Rotate_opencv_x86_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_height, dst_width;
  cv::RotateFlags cv_rotate_flag;
  if (degree == 90) {
    dst_height = src_width;
    dst_width  = src_height;
    cv_rotate_flag = cv::ROTATE_90_CLOCKWISE;
  }
  else if (degree == 180) {
    dst_height = src_height;
    dst_width  = src_width;
    cv_rotate_flag = cv::ROTATE_180;
  }
  else if (degree == 270) {
    dst_height = src_width;
    dst_width  = src_height;
    cv_rotate_flag = cv::ROTATE_90_COUNTERCLOCKWISE;
  }
  else {
    return;
  }

  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());

  for (auto _ : state) {
    cv::rotate(src, dst, cv_rotate_flag);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, degree, width, height)                        \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_cuda, uchar, channels, degree)->           \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, uchar, channels, degree)->              \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_cuda, float, channels, degree)->           \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, float, channels, degree)->              \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(c1, 90, 640, 480)
// RUN_BENCHMARK0(c3, 90, 640, 480)
// RUN_BENCHMARK0(c4, 90, 640, 480)
// RUN_BENCHMARK0(c1, 90, 1920, 1080)
// RUN_BENCHMARK0(c3, 90, 1920, 1080)
// RUN_BENCHMARK0(c4, 90, 1920, 1080)

// RUN_BENCHMARK0(c1, 180, 640, 480)
// RUN_BENCHMARK0(c3, 180, 640, 480)
// RUN_BENCHMARK0(c4, 180, 640, 480)
// RUN_BENCHMARK0(c1, 180, 1920, 1080)
// RUN_BENCHMARK0(c3, 180, 1920, 1080)
// RUN_BENCHMARK0(c4, 180, 1920, 1080)

// RUN_BENCHMARK0(c1, 270, 640, 480)
// RUN_BENCHMARK0(c3, 270, 640, 480)
// RUN_BENCHMARK0(c4, 270, 640, 480)
// RUN_BENCHMARK0(c1, 270, 1920, 1080)
// RUN_BENCHMARK0(c3, 270, 1920, 1080)
// RUN_BENCHMARK0(c4, 270, 1920, 1080)

#define RUN_BENCHMARK1(channels, degree, width, height)                        \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_x86_cuda, uchar, channels, degree)->       \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, uchar, channels, degree)->              \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_x86_cuda, float, channels, degree)->       \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, float, channels, degree)->              \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(c1, 90, 640, 480)
// RUN_BENCHMARK1(c3, 90, 640, 480)
// RUN_BENCHMARK1(c4, 90, 640, 480)
// RUN_BENCHMARK1(c1, 90, 1920, 1080)
// RUN_BENCHMARK1(c3, 90, 1920, 1080)
// RUN_BENCHMARK1(c4, 90, 1920, 1080)

// RUN_BENCHMARK1(c1, 180, 640, 480)
// RUN_BENCHMARK1(c3, 180, 640, 480)
// RUN_BENCHMARK1(c4, 180, 640, 480)
// RUN_BENCHMARK1(c1, 180, 1920, 1080)
// RUN_BENCHMARK1(c3, 180, 1920, 1080)
// RUN_BENCHMARK1(c4, 180, 1920, 1080)

// RUN_BENCHMARK1(c1, 270, 640, 480)
// RUN_BENCHMARK1(c3, 270, 640, 480)
// RUN_BENCHMARK1(c4, 270, 640, 480)
// RUN_BENCHMARK1(c1, 270, 1920, 1080)
// RUN_BENCHMARK1(c3, 270, 1920, 1080)
// RUN_BENCHMARK1(c4, 270, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, degree)                                \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_cuda, type, c1, degree)->                  \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_cuda, type, c3, degree)->                  \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_cuda, type, c4, degree)->                  \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_cuda, type, c1, degree)->                  \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_cuda, type, c3, degree)->                  \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Rotate_opencv_cuda, type, c4, degree)->                  \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, degree)                                \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, type, c1, degree)->                     \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, type, c3, degree)->                     \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, type, c4, degree)->                     \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, type, c1, degree)->                     \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, type, c3, degree)->                     \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Rotate_ppl_cuda, type, c4, degree)->                     \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, 90)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 180)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 270)
RUN_OPENCV_TYPE_FUNCTIONS(float, 90)
RUN_OPENCV_TYPE_FUNCTIONS(float, 180)
RUN_OPENCV_TYPE_FUNCTIONS(float, 270)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 90)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 180)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 270)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 90)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 180)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 270)
