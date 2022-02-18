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

#include "ppl/cv/cuda/guidedfilter.h"

#ifdef _MSC_VER
#include <time.h>
#else 
#include <sys/time.h>
#endif

#include "opencv2/ximgproc/edge_filter.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

template <typename T, int channels, int radius, int eps>
void BM_GuidedFilter_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat guide(height, width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst(src.rows, src.cols, src.type());
  if (channels == 1) {
    src.copyTo(guide);
  }
  else if (channels == 3) {
    cv::cvtColor(src, guide, cv::COLOR_BGR2GRAY);
  }
  else {
    cv::cvtColor(src, guide, cv::COLOR_BGRA2GRAY);
  }
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_guide(guide);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 500;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    GuidedFilter<T, channels, 1>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_guide.step / sizeof(T),
        (T*)gpu_guide.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data, radius,
        eps, ppl::cv::BORDER_REFLECT);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      GuidedFilter<T, channels, 1>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data,
          gpu_guide.step / sizeof(T), (T*)gpu_guide.data,
          gpu_dst.step / sizeof(T), (T*)gpu_dst.data, radius, eps,
          ppl::cv::BORDER_REFLECT);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, int radius, int eps>
void BM_GuidedFilter_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat guide(height, width,
                CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst(src.rows, src.cols, src.type());
  if (channels == 1) {
    src.copyTo(guide);
  }
  else if (channels == 3) {
    cv::cvtColor(src, guide, cv::COLOR_BGR2GRAY);
  }
  else {
    cv::cvtColor(src, guide, cv::COLOR_BGRA2GRAY);
  }

  for (auto _ : state) {
    cv::ximgproc::guidedFilter(guide, src, dst, radius, eps, -1);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(radius, eps, width, height)                              \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, uchar, c1, radius,         \
                   eps)->Args({width, height});                                \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, uchar, c1, radius, eps)->         \
                   Args({width, height})->UseManualTime()->Iterations(5);      \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, float, c1, radius,         \
                   eps)->Args({width, height});                                \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, float, c1, radius, eps)->         \
                   Args({width, height})->UseManualTime()->Iterations(5);      \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, uchar, c3, radius,         \
                   eps)->Args({width, height});                                \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, uchar, c3, radius, eps)->         \
                   Args({width, height})->UseManualTime()->Iterations(5);      \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, float, c3, radius,         \
                   eps)->Args({width, height});                                \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, float, c3, radius, eps)->         \
                   Args({width, height})->UseManualTime()->Iterations(5);      \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, uchar, c4, radius,         \
                   eps)->Args({width, height});                                \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, uchar, c4, radius, eps)->         \
                   Args({width, height})->UseManualTime()->Iterations(5);      \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, float, c4, radius,         \
                   eps)->Args({width, height});                                \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, float, c4, radius, eps)->         \
                   Args({width, height})->UseManualTime()->Iterations(5);

// RUN_BENCHMARK(3, 50, 320, 240)
// RUN_BENCHMARK(3, 50, 640, 480)
// RUN_BENCHMARK(3, 50, 1280, 720)
// RUN_BENCHMARK(3, 50, 1920, 1080)

// RUN_BENCHMARK(7, 50, 320, 240)
// RUN_BENCHMARK(7, 50, 640, 480)
// RUN_BENCHMARK(7, 50, 1280, 720)
// RUN_BENCHMARK(7, 50, 1920, 1080)

// RUN_BENCHMARK(15, 50, 320, 240)
// RUN_BENCHMARK(15, 50, 640, 480)
// RUN_BENCHMARK(15, 50, 1280, 720)
// RUN_BENCHMARK(15, 50, 1920, 1080)

// RUN_BENCHMARK(22, 50, 320, 240)
// RUN_BENCHMARK(22, 50, 640, 480)
// RUN_BENCHMARK(22, 50, 1280, 720)
// RUN_BENCHMARK(22, 50, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, radius, eps)                           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c1, radius, eps)->   \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c1, radius, eps)->   \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c1, radius, eps)->   \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c1, radius, eps)->   \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c3, radius, eps)->   \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c3, radius, eps)->   \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c3, radius, eps)->   \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c3, radius, eps)->   \
                   Args({1920, 1080});                                         \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c4, radius, eps)->   \
                   Args({320, 240});                                           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c4, radius, eps)->   \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c4, radius, eps)->   \
                   Args({1280, 720});                                          \
BENCHMARK_TEMPLATE(BM_GuidedFilter_opencv_x86_cuda, type, c4, radius, eps)->   \
                   Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, radius, eps)                           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c1, radius, eps)->          \
                   Args({320, 240})->UseManualTime()->Iterations(5);           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c1, radius, eps)->          \
                   Args({640, 480})->UseManualTime()->Iterations(5);           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c1, radius, eps)->          \
                   Args({1280, 720})->UseManualTime()->Iterations(5);          \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c1, radius, eps)->          \
                   Args({1920, 1080})->UseManualTime()->Iterations(5);         \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c3, radius, eps)->          \
                   Args({320, 240})->UseManualTime()->Iterations(5);           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c3, radius, eps)->          \
                   Args({640, 480})->UseManualTime()->Iterations(5);           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c3, radius, eps)->          \
                   Args({1280, 720})->UseManualTime()->Iterations(5);          \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c3, radius, eps)->          \
                   Args({1920, 1080})->UseManualTime()->Iterations(5);         \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c4, radius, eps)->          \
                   Args({320, 240})->UseManualTime()->Iterations(5);           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c4, radius, eps)->          \
                   Args({640, 480})->UseManualTime()->Iterations(5);           \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c4, radius, eps)->          \
                   Args({1280, 720})->UseManualTime()->Iterations(5);          \
BENCHMARK_TEMPLATE(BM_GuidedFilter_ppl_cuda, type, c4, radius, eps)->          \
                   Args({1920, 1080})->UseManualTime()->Iterations(5);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, 3, 50)
RUN_OPENCV_TYPE_FUNCTIONS(float, 3, 50)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 7, 50)
RUN_OPENCV_TYPE_FUNCTIONS(float, 7, 50)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 15, 50)
RUN_OPENCV_TYPE_FUNCTIONS(float, 15, 50)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 22, 50)
RUN_OPENCV_TYPE_FUNCTIONS(float, 22, 50)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 3, 50)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 3, 50)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 7, 50)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 7, 50)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 15, 50)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 15, 50)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 22, 50)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 22, 50)
