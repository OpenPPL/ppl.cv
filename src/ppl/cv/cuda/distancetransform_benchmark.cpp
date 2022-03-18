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

#include "ppl/cv/cuda/distancetransform.h"

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, ppl::cv::DistTypes distance_type,
          ppl::cv::DistanceTransformMasks mask_size>
void BM_DistanceTransform_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createBinaryImage(height, width,
                          CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::DistanceTransform<T>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(uchar), (uchar*)gpu_src.data,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, distance_type, mask_size);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::DistanceTransform<T>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(uchar), (uchar*)gpu_src.data,
          gpu_dst.step / sizeof(T), (T*)gpu_dst.data, distance_type, mask_size);
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

template <typename T, ppl::cv::DistTypes distance_type,
          ppl::cv::DistanceTransformMasks mask_size>
void BM_DistanceTransform_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createBinaryImage(height, width,
                          CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));

  cv::DistanceTypes cv_distance;
  if (distance_type == ppl::cv::DIST_C) {
    cv_distance = cv::DIST_C;
  }
  else if (distance_type == ppl::cv::DIST_L1) {
    cv_distance = cv::DIST_L1;
  }
  else {
    cv_distance = cv::DIST_L2;
  }
  cv::DistanceTransformMasks cv_mask;
  if (mask_size == ppl::cv::DIST_MASK_3) {
    cv_mask = cv::DIST_MASK_3;
  }
  else if (mask_size == ppl::cv::DIST_MASK_5) {
    cv_mask = cv::DIST_MASK_5;
  }
  else {
    cv_mask = cv::DIST_MASK_PRECISE;
  }

  for (auto _ : state) {
    cv::distanceTransform(src, dst, cv_distance, cv_mask);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(distance_type, mask_size, width, height)                 \
BENCHMARK_TEMPLATE(BM_DistanceTransform_opencv_x86_cuda, float, distance_type, \
                   mask_size)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_DistanceTransform_ppl_cuda, float, distance_type,        \
                   mask_size)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);

// RUN_BENCHMARK(ppl::cv::DIST_C, ppl::cv::DIST_MASK_3, 320, 240)
// RUN_BENCHMARK(ppl::cv::DIST_C, ppl::cv::DIST_MASK_3, 640, 480)
// RUN_BENCHMARK(ppl::cv::DIST_C, ppl::cv::DIST_MASK_3, 1280, 720)
// RUN_BENCHMARK(ppl::cv::DIST_C, ppl::cv::DIST_MASK_3, 1920, 1080)

// RUN_BENCHMARK(ppl::cv::DIST_C, ppl::cv::DIST_MASK_5, 320, 240)
// RUN_BENCHMARK(ppl::cv::DIST_C, ppl::cv::DIST_MASK_5, 640, 480)
// RUN_BENCHMARK(ppl::cv::DIST_C, ppl::cv::DIST_MASK_5, 1280, 720)
// RUN_BENCHMARK(ppl::cv::DIST_C, ppl::cv::DIST_MASK_5, 1920, 1080)

// RUN_BENCHMARK(ppl::cv::DIST_L1, ppl::cv::DIST_MASK_3, 320, 240)
// RUN_BENCHMARK(ppl::cv::DIST_L1, ppl::cv::DIST_MASK_3, 640, 480)
// RUN_BENCHMARK(ppl::cv::DIST_L1, ppl::cv::DIST_MASK_3, 1280, 720)
// RUN_BENCHMARK(ppl::cv::DIST_L1, ppl::cv::DIST_MASK_3, 1920, 1080)

// RUN_BENCHMARK(ppl::cv::DIST_L1, ppl::cv::DIST_MASK_5, 320, 240)
// RUN_BENCHMARK(ppl::cv::DIST_L1, ppl::cv::DIST_MASK_5, 640, 480)
// RUN_BENCHMARK(ppl::cv::DIST_L1, ppl::cv::DIST_MASK_5, 1280, 720)
// RUN_BENCHMARK(ppl::cv::DIST_L1, ppl::cv::DIST_MASK_5, 1920, 1080)

// RUN_BENCHMARK(ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE, 320, 240)
// RUN_BENCHMARK(ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE, 640, 480)
// RUN_BENCHMARK(ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE, 1280, 720)
// RUN_BENCHMARK(ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, distance_type, mask_size)              \
BENCHMARK_TEMPLATE(BM_DistanceTransform_opencv_x86_cuda, type, distance_type,  \
                   mask_size)->Args({320, 240});                               \
BENCHMARK_TEMPLATE(BM_DistanceTransform_opencv_x86_cuda, type, distance_type,  \
                   mask_size)->Args({640, 480});                               \
BENCHMARK_TEMPLATE(BM_DistanceTransform_opencv_x86_cuda, type, distance_type,  \
                   mask_size)->Args({1280, 720});                              \
BENCHMARK_TEMPLATE(BM_DistanceTransform_opencv_x86_cuda, type, distance_type,  \
                   mask_size)->Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, distance_type, mask_size)              \
BENCHMARK_TEMPLATE(BM_DistanceTransform_ppl_cuda, type, distance_type,         \
                   mask_size)->Args({320, 240})->UseManualTime()->             \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_DistanceTransform_ppl_cuda, type, distance_type,         \
                   mask_size)->Args({640, 480})->UseManualTime()->             \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_DistanceTransform_ppl_cuda, type, distance_type,         \
                   mask_size)->Args({1280, 720})->UseManualTime()->            \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_DistanceTransform_ppl_cuda, type, distance_type,         \
                   mask_size)->Args({1920, 1080})->UseManualTime()->           \
                   Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::DIST_C, ppl::cv::DIST_MASK_3)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::DIST_C, ppl::cv::DIST_MASK_5)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::DIST_L1, ppl::cv::DIST_MASK_3)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::DIST_L1, ppl::cv::DIST_MASK_5)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE)

RUN_PPL_CV_TYPE_FUNCTIONS(float, ppl::cv::DIST_C, ppl::cv::DIST_MASK_3)
RUN_PPL_CV_TYPE_FUNCTIONS(float, ppl::cv::DIST_C, ppl::cv::DIST_MASK_5)
RUN_PPL_CV_TYPE_FUNCTIONS(float, ppl::cv::DIST_L1, ppl::cv::DIST_MASK_3)
RUN_PPL_CV_TYPE_FUNCTIONS(float, ppl::cv::DIST_L1, ppl::cv::DIST_MASK_5)
RUN_PPL_CV_TYPE_FUNCTIONS(float, ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE)
