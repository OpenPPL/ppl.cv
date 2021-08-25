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

#include "ppl/cv/cuda/resize.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

enum InterpolationTypes {
  kInterLinear,
  kInterNearest,
  kInterArea,
};

template <typename T, int channels, InterpolationTypes inter_type>
void BM_Resize_ppl_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 3000;
  struct timeval start, end;

  // warp up the GPU
  for (int i = 0; i < iterations; i++) {
    ResizeLinear<T, channels>(0, src.rows, src.cols, gpu_src.step / sizeof(T),
                              (T*)gpu_src.data, dst_height, dst_width,
                              gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (inter_type == kInterLinear) {
        ResizeLinear<T, channels>(0, src.rows, src.cols,
                                  gpu_src.step / sizeof(T), (T*)gpu_src.data,
                                  dst_height, dst_width,
                                  gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else if (inter_type == kInterNearest) {
        ResizeNearestPoint<T, channels>(0, src.rows, src.cols,
                                        gpu_src.step / sizeof(T),
                                        (T*)gpu_src.data, dst_height, dst_width,
                                        gpu_dst.step / sizeof(T),
                                        (T*)gpu_dst.data);
      }
      else if (inter_type == kInterArea) {
        ResizeArea<T, channels>(0, src.rows, src.cols, gpu_src.step / sizeof(T),
                                (T*)gpu_src.data, dst_height, dst_width,
                                gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
      }
      else {
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, InterpolationTypes inter_type>
void BM_Resize_opencv_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 3000;
  struct timeval start, end;

  // warp up the GPU
  for (int i = 0; i < iterations; i++) {
    cv::cuda::resize(gpu_src, gpu_dst, cv::Size(dst_width, dst_height), 0, 0,
                     cv::INTER_LINEAR);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (inter_type == kInterLinear) {
        cv::cuda::resize(gpu_src, gpu_dst, cv::Size(dst_width, dst_height), 0,
                         0, cv::INTER_LINEAR);
      }
      else if (inter_type == kInterNearest) {
        cv::cuda::resize(gpu_src, gpu_dst, cv::Size(dst_width, dst_height), 0,
                         0, cv::INTER_NEAREST);
      }
      else if (inter_type == kInterArea) {
        cv::cuda::resize(gpu_src, gpu_dst, cv::Size(dst_width, dst_height), 0,
                         0, cv::INTER_AREA);
      }
      else {
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, InterpolationTypes inter_type>
void BM_Resize_opencv_x86_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());

  for (auto _ : state) {
    if (inter_type == kInterLinear) {
      cv::resize(src, dst, cv::Size(dst_width, dst_height), 0, 0,
                 cv::INTER_LINEAR);
    }
    else if (inter_type == kInterNearest) {
      cv::resize(src, dst, cv::Size(dst_width, dst_height), 0, 0,
                 cv::INTER_NEAREST);
    }
    else if (inter_type == kInterArea) {
      cv::resize(src, dst, cv::Size(dst_width, dst_height), 0, 0,
                 cv::INTER_AREA);
    }
    else {
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(channels, inter_type, src_width, src_height, dst_width,  \
                      dst_height)                                              \
BENCHMARK_TEMPLATE(BM_Resize_opencv_x86_cuda, uchar, channels, inter_type)->   \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_opencv_cuda, uchar, channels, inter_type)->       \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_cuda, uchar, channels, inter_type)->          \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_x86_cuda, float, channels, inter_type)->   \
                   Args({src_width, src_height, dst_width, dst_height});       \
BENCHMARK_TEMPLATE(BM_Resize_opencv_cuda, float, channels, inter_type)->       \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_cuda, float, channels, inter_type)->          \
                   Args({src_width, src_height, dst_width, dst_height})->      \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK(c1, kInterLinear, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, kInterLinear, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, kInterLinear, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, kInterLinear, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, kInterLinear, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, kInterLinear, 640, 480, 1280, 960)

// RUN_BENCHMARK(c1, kInterNearest, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, kInterNearest, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, kInterNearest, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, kInterNearest, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, kInterNearest, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, kInterNearest, 640, 480, 1280, 960)

// RUN_BENCHMARK(c1, kInterArea, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, kInterArea, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, kInterArea, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, kInterArea, 640, 480, 420, 340)
// RUN_BENCHMARK(c3, kInterArea, 640, 480, 420, 340)
// RUN_BENCHMARK(c4, kInterArea, 640, 480, 420, 340)
// RUN_BENCHMARK(c1, kInterArea, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, kInterArea, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, kInterArea, 640, 480, 1280, 960)

#define RUN_OPENCV_X86_TYPE_FUNCTIONS(type, inter_type)                        \
BENCHMARK_TEMPLATE(BM_Resize_opencv_x86_cuda, type, c1, inter_type)->          \
                   Args({640, 480, 320, 240});                                 \
BENCHMARK_TEMPLATE(BM_Resize_opencv_x86_cuda, type, c1, inter_type)->          \
                   Args({640, 480, 1280, 960});                                \
BENCHMARK_TEMPLATE(BM_Resize_opencv_x86_cuda, type, c3, inter_type)->          \
                   Args({640, 480, 320, 240});                                 \
BENCHMARK_TEMPLATE(BM_Resize_opencv_x86_cuda, type, c3, inter_type)->          \
                   Args({640, 480, 1280, 960});                                \
BENCHMARK_TEMPLATE(BM_Resize_opencv_x86_cuda, type, c4, inter_type)->          \
                   Args({640, 480, 320, 240});                                 \
BENCHMARK_TEMPLATE(BM_Resize_opencv_x86_cuda, type, c4, inter_type)->          \
                   Args({640, 480, 1280, 960});

#define RUN_OPENCV_TYPE_FUNCTIONS(type, inter_type)                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_cuda, type, c1, inter_type)->              \
                   Args({640, 480, 320, 240})->                                \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_cuda, type, c1, inter_type)->              \
                   Args({640, 480, 1280, 960})->                               \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_cuda, type, c3, inter_type)->              \
                   Args({640, 480, 320, 240})->                                \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_cuda, type, c3, inter_type)->              \
                   Args({640, 480, 1280, 960})->                               \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_cuda, type, c4, inter_type)->              \
                   Args({640, 480, 320, 240})->                                \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_opencv_cuda, type, c4, inter_type)->              \
                   Args({640, 480, 1280, 960})->                               \
                   UseManualTime()->Iterations(10);

#define RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(type, inter_type)                       \
BENCHMARK_TEMPLATE(BM_Resize_ppl_cuda, type, c1, inter_type)->                 \
                   Args({640, 480, 320, 240})->                                \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_cuda, type, c1, inter_type)->                 \
                   Args({640, 480, 1280, 960})->                               \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_cuda, type, c3, inter_type)->                 \
                   Args({640, 480, 320, 240})->                                \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_cuda, type, c3, inter_type)->                 \
                   Args({640, 480, 1280, 960})->                               \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_cuda, type, c4, inter_type)->                 \
                   Args({640, 480, 320, 240})->                                \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Resize_ppl_cuda, type, c4, inter_type)->                 \
                   Args({640, 480, 1280, 960})->                               \
                   UseManualTime()->Iterations(10);

// RUN_OPENCV_X86_TYPE_FUNCTIONS(uchar, kInterLinear)
// RUN_OPENCV_X86_TYPE_FUNCTIONS(uchar, kInterNearest)
// RUN_OPENCV_X86_TYPE_FUNCTIONS(uchar, kInterArea)
// RUN_OPENCV_X86_TYPE_FUNCTIONS(float, kInterLinear)
// RUN_OPENCV_X86_TYPE_FUNCTIONS(float, kInterNearest)
// RUN_OPENCV_X86_TYPE_FUNCTIONS(float, kInterArea)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, kInterLinear)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, kInterNearest)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, kInterArea)
RUN_OPENCV_TYPE_FUNCTIONS(float, kInterLinear)
RUN_OPENCV_TYPE_FUNCTIONS(float, kInterNearest)
RUN_OPENCV_TYPE_FUNCTIONS(float, kInterArea)

RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, kInterLinear)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, kInterNearest)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, kInterArea)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, kInterLinear)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, kInterNearest)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, kInterArea)
