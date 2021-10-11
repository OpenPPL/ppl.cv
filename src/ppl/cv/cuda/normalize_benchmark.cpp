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

#include "ppl/cv/cuda/normalize.h"

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
  NO_MASK,
  WITH_MASK,
};

template <typename T, int channels, NormTypes norm_type, MaskType mask_type>
void BM_Normalize_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<float>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask(mask);

  float alpha = 3.f;
  float beta  = 10.f;

  int iterations = 1000;
  struct timeval start, end;

  // Warm up the GPU
  for (int i = 0; i < iterations; i++) {
    Normalize<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data,
        gpu_dst.step / sizeof(float), (float*)gpu_dst.data, alpha, beta,
        norm_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == NO_MASK) {
        Normalize<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(float), (float*)gpu_dst.data, alpha, beta,
            norm_type);
      }
      else {
        Normalize<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data,
            gpu_dst.step / sizeof(float), (float*)gpu_dst.data, alpha, beta,
            norm_type, gpu_mask.step, (uchar*)gpu_mask.data);
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

template <typename T, int channels, NormTypes norm_type, MaskType mask_type>
void BM_Normalize_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<float>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask(mask);

  float alpha = 3.f;
  float beta  = 10.f;

  cv::NormTypes cv_norm_type;
  if (norm_type == NORM_INF) {
    cv_norm_type = cv::NORM_INF;
  }
  else if (norm_type == NORM_L1) {
    cv_norm_type = cv::NORM_L1;
  }
  else if (norm_type == NORM_L2) {
    cv_norm_type = cv::NORM_L2;
  }
  else {  // norm_type == NORM_MINMAX
    cv_norm_type = cv::NORM_MINMAX;
  }

  int iterations = 1000;
  struct timeval start, end;

  // Warm up the GPU
  for (int i = 0; i < iterations; i++) {
    cv::cuda::normalize(gpu_src, gpu_dst, alpha, beta, cv_norm_type,
                        CV_MAT_DEPTH(dst.type()));
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (mask_type == NO_MASK) {
        cv::cuda::normalize(gpu_src, gpu_dst, alpha, beta, cv_norm_type,
                            CV_MAT_DEPTH(dst.type()));
      }
      else {
        cv::cuda::normalize(gpu_src, gpu_dst, alpha, beta, cv_norm_type,
                            CV_MAT_DEPTH(dst.type()), gpu_mask);
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

template <typename T, int channels, NormTypes norm_type, MaskType mask_type>
void BM_Normalize_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, mask;
  src  = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                           channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<float>::depth, channels));
  mask = createSourceImage(height, width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));

  float alpha = 3.f;
  float beta  = 10.f;

  cv::NormTypes cv_norm_type;
  if (norm_type == NORM_INF) {
    cv_norm_type = cv::NORM_INF;
  }
  else if (norm_type == NORM_L1) {
    cv_norm_type = cv::NORM_L1;
  }
  else if (norm_type == NORM_L2) {
    cv_norm_type = cv::NORM_L2;
  }
  else {  // norm_type == NORM_MINMAX
    cv_norm_type = cv::NORM_MINMAX;
  }

  for (auto _ : state) {
    if (mask_type == NO_MASK) {
      cv::normalize(src, dst, alpha, beta, cv_norm_type,
                    CV_MAT_DEPTH(dst.type()));
    }
    else {
      cv::normalize(src, dst, alpha, beta, cv_norm_type,
                    CV_MAT_DEPTH(dst.type()), mask);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, norm_type, mask_type, width, height)          \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_cuda, uchar, channels, norm_type,       \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, uchar, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_cuda, float, channels, norm_type,       \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, float, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);

#define RUN_BENCHMARK1(channels, norm_type, mask_type, width, height)          \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_cuda, uchar, channels, norm_type,       \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, uchar, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);

// RUN_BENCHMARK0(c1, NORM_INF, NO_MASK, 640, 480)
// RUN_BENCHMARK0(c1, NORM_INF, NO_MASK, 1920, 1080)
// RUN_BENCHMARK0(c1, NORM_L1, NO_MASK, 640, 480)
// RUN_BENCHMARK0(c1, NORM_L1, NO_MASK, 1920, 1080)
// RUN_BENCHMARK0(c1, NORM_L2, NO_MASK, 640, 480)
// RUN_BENCHMARK0(c1, NORM_L2, NO_MASK, 1920, 1080)
// RUN_BENCHMARK0(c1, NORM_MINMAX, NO_MASK, 640, 480)
// RUN_BENCHMARK0(c1, NORM_MINMAX, NO_MASK, 1920, 1080)

// RUN_BENCHMARK1(c1, NORM_INF, WITH_MASK, 640, 480)
// RUN_BENCHMARK1(c1, NORM_INF, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK1(c1, NORM_L1, WITH_MASK, 640, 480)
// RUN_BENCHMARK1(c1, NORM_L1, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK1(c1, NORM_L2, WITH_MASK, 640, 480)
// RUN_BENCHMARK1(c1, NORM_L2, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK1(c1, NORM_MINMAX, WITH_MASK, 640, 480)
// RUN_BENCHMARK1(c1, NORM_MINMAX, WITH_MASK, 1920, 1080)

#define RUN_BENCHMARK2(channels, norm_type, mask_type, width, height)          \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, uchar, channels, norm_type,   \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, uchar, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, float, channels, norm_type,   \
                   mask_type)->Args({width, height});                          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, float, channels, norm_type,          \
                   mask_type)->Args({width, height})->UseManualTime()->        \
                   Iterations(10);

// RUN_BENCHMARK2(c1, NORM_INF, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c3, NORM_INF, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c4, NORM_INF, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c1, NORM_INF, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK2(c3, NORM_INF, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK2(c4, NORM_INF, WITH_MASK, 1920, 1080)

// RUN_BENCHMARK2(c1, NORM_L1, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c3, NORM_L1, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c4, NORM_L1, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c1, NORM_L1, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK2(c3, NORM_L1, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK2(c4, NORM_L1, WITH_MASK, 1920, 1080)

// RUN_BENCHMARK2(c1, NORM_L2, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c3, NORM_L2, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c4, NORM_L2, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c1, NORM_L2, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK2(c3, NORM_L2, WITH_MASK, 1920, 1080)
// RUN_BENCHMARK2(c4, NORM_L2, WITH_MASK, 1920, 1080)

// RUN_BENCHMARK2(c1, NORM_MINMAX, WITH_MASK, 640, 480)
// RUN_BENCHMARK2(c1, NORM_MINMAX, WITH_MASK, 1920, 1080)

// RUN_BENCHMARK2(c1, NORM_INF, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c3, NORM_INF, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c4, NORM_INF, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c1, NORM_INF, NO_MASK, 1920, 1080)
// RUN_BENCHMARK2(c3, NORM_INF, NO_MASK, 1920, 1080)
// RUN_BENCHMARK2(c4, NORM_INF, NO_MASK, 1920, 1080)

// RUN_BENCHMARK2(c1, NORM_L1, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c3, NORM_L1, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c4, NORM_L1, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c1, NORM_L1, NO_MASK, 1920, 1080)
// RUN_BENCHMARK2(c3, NORM_L1, NO_MASK, 1920, 1080)
// RUN_BENCHMARK2(c4, NORM_L1, NO_MASK, 1920, 1080)

// RUN_BENCHMARK2(c1, NORM_L2, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c3, NORM_L2, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c4, NORM_L2, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c1, NORM_L2, NO_MASK, 1920, 1080)
// RUN_BENCHMARK2(c3, NORM_L2, NO_MASK, 1920, 1080)
// RUN_BENCHMARK2(c4, NORM_L2, NO_MASK, 1920, 1080)

// RUN_BENCHMARK2(c1, NORM_MINMAX, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c3, NORM_MINMAX, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c4, NORM_MINMAX, NO_MASK, 640, 480)
// RUN_BENCHMARK2(c1, NORM_MINMAX, NO_MASK, 1920, 1080)
// RUN_BENCHMARK2(c3, NORM_MINMAX, NO_MASK, 1920, 1080)
// RUN_BENCHMARK2(c4, NORM_MINMAX, NO_MASK, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS0(type, norm_type, mask_type)                 \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c1, norm_type,          \
                   mask_type)->Args({640, 480})->UseManualTime()->             \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c3, norm_type,          \
                   mask_type)->Args({640, 480})->UseManualTime()->             \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c4, norm_type,          \
                   mask_type)->Args({640, 480})->UseManualTime()->             \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c1, norm_type,          \
                   mask_type)->Args({1920, 1080})->UseManualTime()->           \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c3, norm_type,          \
                   mask_type)->Args({1920, 1080})->UseManualTime()->           \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c4, norm_type,          \
                   mask_type)->Args({1920, 1080})->UseManualTime()->           \
                   Iterations(10);

#define RUN_OPENCV_TYPE_FUNCTIONS1(type, norm_type, mask_type)                 \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c1, norm_type,          \
                   mask_type)->Args({640, 480})->UseManualTime()->             \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Normalize_opencv_x86_cuda, type, c1, norm_type,          \
                   mask_type)->Args({1920, 1080})->UseManualTime()->           \
                   Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS0(type, norm_type, mask_type)                 \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c1, norm_type, mask_type)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c3, norm_type, mask_type)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c4, norm_type, mask_type)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c1, norm_type, mask_type)->    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c3, norm_type, mask_type)->    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);        \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c4, norm_type, mask_type)->    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS1(type, norm_type, mask_type)                 \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c1, norm_type, mask_type)->    \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_Normalize_ppl_cuda, type, c1, norm_type, mask_type)->    \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS0(uchar, NORM_INF, WITH_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, NORM_L1, WITH_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, NORM_L2, WITH_MASK)
RUN_OPENCV_TYPE_FUNCTIONS1(uchar, NORM_MINMAX, WITH_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(float, NORM_INF, WITH_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(float, NORM_L1, WITH_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(float, NORM_L2, WITH_MASK)
RUN_OPENCV_TYPE_FUNCTIONS1(float, NORM_MINMAX, WITH_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, NORM_INF, NO_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, NORM_L1, NO_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, NORM_L2, NO_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(uchar, NORM_MINMAX, NO_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(float, NORM_INF, NO_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(float, NORM_L1, NO_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(float, NORM_L2, NO_MASK)
RUN_OPENCV_TYPE_FUNCTIONS0(float, NORM_MINMAX, NO_MASK)

RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, NORM_INF, WITH_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, NORM_L1, WITH_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, NORM_L2, WITH_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS1(uchar, NORM_MINMAX, WITH_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, NORM_INF, WITH_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, NORM_L1, WITH_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, NORM_L2, WITH_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS1(float, NORM_MINMAX, WITH_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, NORM_INF, NO_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, NORM_L1, NO_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, NORM_L2, NO_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(uchar, NORM_MINMAX, NO_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, NORM_INF, NO_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, NORM_L1, NO_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, NORM_L2, NO_MASK)
RUN_PPL_CV_TYPE_FUNCTIONS0(float, NORM_MINMAX, NO_MASK)
