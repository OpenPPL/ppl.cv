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

#include "ppl/cv/cuda/warpaffine.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

template <typename T, int channels, InterpolationType inter_type,
          BorderType border_type>
void BM_WarpAffine_ppl_cuda(benchmark::State &state) {
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
  cv::Mat M = createSourceImage(2, 3, CV_32FC1);

  int iterations = 3000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    WarpAffine<T, channels>(0, src.rows, src.cols, gpu_src.step / sizeof(T),
        (T*)gpu_src.data, dst_height, dst_width, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data, (float*)M.data, inter_type, border_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      WarpAffine<T, channels>(0, src.rows, src.cols, gpu_src.step / sizeof(T),
          (T*)gpu_src.data, dst_height, dst_width, gpu_dst.step / sizeof(T),
          (T*)gpu_dst.data, (float*)M.data, inter_type, border_type);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, InterpolationType inter_type,
          BorderType border_type>
void BM_WarpAffine_opencv_cuda(benchmark::State &state) {
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
  cv::Mat M = createSourceImage(2, 3, CV_32FC1);

  int cv_iterpolation;
  if (inter_type == INTERPOLATION_LINEAR) {
    cv_iterpolation = cv::INTER_LINEAR;
  }
  else if (inter_type == INTERPOLATION_NEAREST_POINT) {
    cv_iterpolation = cv::INTER_NEAREST;
  }
  else {
  }

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_TRANSPARENT) {
    cv_border = cv::BORDER_TRANSPARENT;
  }
  else {
  }

  int iterations = 3000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::warpAffine(gpu_src, gpu_dst, M, cv::Size(dst_width, dst_height),
                         cv::WARP_INVERSE_MAP | cv_iterpolation, cv_border);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::warpAffine(gpu_src, gpu_dst, M, cv::Size(dst_width, dst_height),
                           cv::WARP_INVERSE_MAP | cv_iterpolation, cv_border);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, InterpolationType inter_type,
          BorderType border_type>
void BM_WarpAffine_opencv_x86_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width, src.type());
  cv::Mat M = createSourceImage(2, 3, CV_32FC1);

  int cv_iterpolation;
  if (inter_type == INTERPOLATION_LINEAR) {
    cv_iterpolation = cv::INTER_LINEAR;
  }
  else if (inter_type == INTERPOLATION_NEAREST_POINT) {
    cv_iterpolation = cv::INTER_NEAREST;
  }
  else {
  }

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_TRANSPARENT) {
    cv_border = cv::BORDER_TRANSPARENT;
  }
  else {
  }

  for (auto _ : state) {
    cv::warpAffine(src, dst, M, cv::Size(dst_width, dst_height),
                   cv::WARP_INVERSE_MAP | cv_iterpolation, cv_border);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(channels, inter_type, src_width, src_height, dst_width,  \
                      dst_height)                                              \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, channels,             \
                   inter_type, BORDER_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height});                        \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, channels,                 \
                   inter_type, BORDER_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, channels,                    \
                   inter_type, BORDER_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, channels,             \
                   inter_type, BORDER_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height});                        \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, channels,                 \
                   inter_type, BORDER_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, channels,                    \
                   inter_type, BORDER_CONSTANT)->Args({src_width,         \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, channels,             \
                   inter_type, BORDER_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height});                        \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, channels,                 \
                   inter_type, BORDER_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, channels,                    \
                   inter_type, BORDER_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, channels,             \
                   inter_type, BORDER_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height});                        \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, channels,                 \
                   inter_type, BORDER_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, channels,                    \
                   inter_type, BORDER_REPLICATE)->Args({src_width,        \
                   src_height, dst_width, dst_height})->                       \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK(c1, INTERPOLATION_LINEAR, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, INTERPOLATION_LINEAR, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, INTERPOLATION_LINEAR, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, INTERPOLATION_LINEAR, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, INTERPOLATION_LINEAR, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, INTERPOLATION_LINEAR, 640, 480, 1280, 960)

// RUN_BENCHMARK(c1, INTERPOLATION_NEAREST_POINT, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, INTERPOLATION_NEAREST_POINT, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, INTERPOLATION_NEAREST_POINT, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, INTERPOLATION_NEAREST_POINT, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, INTERPOLATION_NEAREST_POINT, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, INTERPOLATION_NEAREST_POINT, 640, 480, 1280, 960)

#define RUN_OPENCV_X86_TYPE_FUNCTIONS(inter_type, border_type)                 \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c1, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c1, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c3, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c3, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c4, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, uchar, c4, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c1, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c1, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c3, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c3, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});                  \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c4, inter_type,       \
                   border_type)->Args({640, 480, 320, 240});                   \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_x86_cuda, float, c4, inter_type,       \
                   border_type)->Args({640, 480, 1280, 960});

#define RUN_OPENCV_CUDA_TYPE_FUNCTIONS(inter_type, border_type)                \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c1, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c1, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c3, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c3, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c4, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, uchar, c4, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c1, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c1, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c3, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c3, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c4, inter_type,           \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_opencv_cuda, float, c4, inter_type,           \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);

#define RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(inter_type, border_type)                \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c1, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c1, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c3, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c3, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c4, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, uchar, c4, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c1, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c1, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c3, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c3, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c4, inter_type,              \
                   border_type)->Args({640, 480, 320, 240})->                  \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_WarpAffine_ppl_cuda, float, c4, inter_type,              \
                   border_type)->Args({640, 480, 1280, 960})->                 \
                   UseManualTime()->Iterations(10);

RUN_OPENCV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_LINEAR,
                               BORDER_CONSTANT)
RUN_OPENCV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_LINEAR,
                               BORDER_REPLICATE)
RUN_OPENCV_X86_TYPE_FUNCTIONS(INTERPOLATION_LINEAR,
                              BORDER_TRANSPARENT)
RUN_OPENCV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_NEAREST_POINT,
                               BORDER_CONSTANT)
RUN_OPENCV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_NEAREST_POINT,
                               BORDER_REPLICATE)
RUN_OPENCV_X86_TYPE_FUNCTIONS(INTERPOLATION_NEAREST_POINT,
                              BORDER_TRANSPARENT)

RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_LINEAR,
                               BORDER_CONSTANT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_LINEAR,
                               BORDER_REPLICATE)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_LINEAR,
                               BORDER_TRANSPARENT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_NEAREST_POINT,
                               BORDER_CONSTANT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_NEAREST_POINT,
                               BORDER_REPLICATE)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(INTERPOLATION_NEAREST_POINT,
                               BORDER_TRANSPARENT)
