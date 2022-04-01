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

#include "ppl/cv/cuda/dilate.h"
#include "ppl/cv/cuda/erode.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

enum Masks {
  kFullyMasked,
  kPartiallyMasked,
};

enum Functions {
  kDilate,
  kErode,
};

template <typename T, int channels, int ksize, Masks mask_type,
          Functions function>
void BM_Dilate_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, dst;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(src.rows, src.cols, src.type());

  cv::Size size(ksize, ksize);
  cv::Mat kernel;
  if (mask_type == kFullyMasked) {
    kernel = cv::getStructuringElement(cv::MORPH_RECT, size);
  }
  else {
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, size);
  }
  uchar* mask = (uchar*)malloc(ksize * ksize * sizeof(uchar));
  int index = 0;
  for (int i = 0; i < ksize; i++) {
    const uchar* data = kernel.ptr<const uchar>(i);
    for (int j = 0; j < ksize; j++) {
      mask[index++] = data[j];
    }
  }

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if ((mask_type == kFullyMasked && height >= 480 && width >= 640) ||
      mask_type == kPartiallyMasked) {
    cudaEventRecord(start, 0);
    size_t size_width = width * channels * sizeof(T);
    size_t ceiled_volume = ppl::cv::cuda::ceil2DVolume(size_width, height);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "activateGpuMemoryPool() time: " << elapsed_time * 1000000
              << " ns" << std::endl;
  }

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Dilate<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, ksize, ksize, mask,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, ppl::cv::BORDER_REFLECT);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      if (function == kDilate) {
        ppl::cv::cuda::Dilate<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, ksize, ksize, mask,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data,
            ppl::cv::BORDER_REFLECT);
      }
      else if (function == kErode) {
        ppl::cv::cuda::Erode<T, channels>(0, gpu_src.rows, gpu_src.cols,
            gpu_src.step / sizeof(T), (T*)gpu_src.data, ksize, ksize, mask,
            gpu_dst.step / sizeof(T), (T*)gpu_dst.data,
            ppl::cv::BORDER_REFLECT);
      }
      else {
      }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  if ((mask_type == kFullyMasked && height >= 480 && width >= 640) ||
      mask_type == kPartiallyMasked) {
    cudaEventRecord(start, 0);
    ppl::cv::cuda::shutDownGpuMemoryPool();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "shutDownGpuMemoryPool() time: " << elapsed_time * 1000000
              << " ns" << std::endl;
  }

  free(mask);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename T, int channels, int ksize, Masks mask_type,
          Functions function>
void BM_Dilate_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, dst;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(src.rows, src.cols, src.type());

  cv::Size size(ksize, ksize);
  cv::Mat kernel;
  if (mask_type == kFullyMasked) {
    kernel = cv::getStructuringElement(cv::MORPH_RECT, size);
  }
  else {
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, size);
  }

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::Ptr<cv::cuda::Filter> filter;
    if (function == kDilate) {
      filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE,
                                                gpu_src.type(), kernel);
    }
    else {
      filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, gpu_src.type(),
                                                kernel);
    }
    filter->apply(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::Ptr<cv::cuda::Filter> filter;
      if (function == kDilate) {
        filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE,
                                                  gpu_src.type(), kernel);
      }
      else {
        filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,
                                                  gpu_src.type(), kernel);
      }
      filter->apply(gpu_src, gpu_dst);
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

template <typename T, int channels, int ksize, Masks mask_type,
          Functions function>
void BM_Dilate_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(src.rows, src.cols, src.type());

  cv::Size size(ksize, ksize);
  cv::Mat kernel;
  if (mask_type == kFullyMasked) {
    kernel = cv::getStructuringElement(cv::MORPH_RECT, size);
  }
  else {
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, size);
  }

  for (auto _ : state) {
    if (function == kDilate) {
      cv::dilate(src, dst, kernel, cv::Point(-1, -1), 1, cv::BORDER_REFLECT);
    }
    else if (function == kErode) {
      cv::erode(src, dst, kernel, cv::Point(-1, -1), 1, cv::BORDER_REFLECT);
    }
    else {
    }
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(channels, ksize, mask_type, function, width, height)    \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_x86_cuda, uchar, channels, ksize,          \
                   mask_type, function)->Args({width, height});                \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_cuda, uchar, channels, ksize,              \
                   mask_type, function)->Args({width, height})->               \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, uchar, channels, ksize,                 \
                   mask_type, function)->Args({width, height})->               \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_x86_cuda, float, channels, ksize,          \
                   mask_type, function)->Args({width, height});                \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_cuda, float, channels, ksize,              \
                   mask_type, function)->Args({width, height})->               \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, float, channels, ksize,                 \
                   mask_type, function)->Args({width, height})->               \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK1(channels, ksize, mask_type, function, width, height)    \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_x86_cuda, uchar, channels, ksize,          \
                   mask_type, function)->Args({width, height});                \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, uchar, channels, ksize,                 \
                   mask_type, function)->Args({width, height})->               \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_x86_cuda, float, channels, ksize,          \
                   mask_type, function)->Args({width, height});                \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, float, channels, ksize,                 \
                   mask_type, function)->Args({width, height})->               \
                   UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(c1, k3x3, kFullyMasked, kDilate, 640, 480)
// RUN_BENCHMARK1(c3, k3x3, kFullyMasked, kDilate, 640, 480)
// RUN_BENCHMARK0(c4, k3x3, kFullyMasked, kDilate, 640, 480)

// RUN_BENCHMARK0(c1, k5x5, kFullyMasked, kDilate, 640, 480)
// RUN_BENCHMARK1(c3, k5x5, kFullyMasked, kDilate, 640, 480)
// RUN_BENCHMARK0(c4, k5x5, kFullyMasked, kDilate, 640, 480)

// RUN_BENCHMARK0(c1, k11x11, kFullyMasked, kDilate, 1920, 1080)
// RUN_BENCHMARK1(c3, k11x11, kFullyMasked, kDilate, 1920, 1080)
// RUN_BENCHMARK0(c4, k11x11, kFullyMasked, kDilate, 1920, 1080)

// RUN_BENCHMARK0(c1, k15x15, kFullyMasked, kDilate, 1920, 1080)
// RUN_BENCHMARK1(c3, k15x15, kFullyMasked, kDilate, 1920, 1080)
// RUN_BENCHMARK0(c4, k15x15, kFullyMasked, kDilate, 1920, 1080)

// RUN_BENCHMARK0(c1, k3x3, kPartiallyMasked, kDilate, 640, 480)
// RUN_BENCHMARK1(c3, k3x3, kPartiallyMasked, kDilate, 640, 480)
// RUN_BENCHMARK0(c4, k3x3, kPartiallyMasked, kDilate, 640, 480)

// RUN_BENCHMARK0(c1, k5x5, kPartiallyMasked, kDilate, 640, 480)
// RUN_BENCHMARK1(c3, k5x5, kPartiallyMasked, kDilate, 640, 480)
// RUN_BENCHMARK0(c4, k5x5, kPartiallyMasked, kDilate, 640, 480)

// RUN_BENCHMARK0(c1, k11x11, kPartiallyMasked, kDilate, 1920, 1080)
// RUN_BENCHMARK1(c3, k11x11, kPartiallyMasked, kDilate, 1920, 1080)
// RUN_BENCHMARK0(c4, k11x11, kPartiallyMasked, kDilate, 1920, 1080)

// RUN_BENCHMARK0(c1, k15x15, kPartiallyMasked, kDilate, 1920, 1080)
// RUN_BENCHMARK1(c3, k15x15, kPartiallyMasked, kDilate, 1920, 1080)
// RUN_BENCHMARK0(c4, k15x15, kPartiallyMasked, kDilate, 1920, 1080)

// RUN_BENCHMARK0(c1, k3x3, kFullyMasked, kErode, 640, 480)
// RUN_BENCHMARK1(c3, k3x3, kFullyMasked, kErode, 640, 480)
// RUN_BENCHMARK0(c4, k3x3, kFullyMasked, kErode, 640, 480)

// RUN_BENCHMARK0(c1, k5x5, kFullyMasked, kErode, 640, 480)
// RUN_BENCHMARK1(c3, k5x5, kFullyMasked, kErode, 640, 480)
// RUN_BENCHMARK0(c4, k5x5, kFullyMasked, kErode, 640, 480)

// RUN_BENCHMARK0(c1, k11x11, kFullyMasked, kErode, 1920, 1080)
// RUN_BENCHMARK1(c3, k11x11, kFullyMasked, kErode, 1920, 1080)
// RUN_BENCHMARK0(c4, k11x11, kFullyMasked, kErode, 1920, 1080)

// RUN_BENCHMARK0(c1, k15x15, kFullyMasked, kErode, 1920, 1080)
// RUN_BENCHMARK1(c3, k15x15, kFullyMasked, kErode, 1920, 1080)
// RUN_BENCHMARK0(c4, k15x15, kFullyMasked, kErode, 1920, 1080)

// RUN_BENCHMARK0(c1, k3x3, kPartiallyMasked, kErode, 640, 480)
// RUN_BENCHMARK1(c3, k3x3, kPartiallyMasked, kErode, 640, 480)
// RUN_BENCHMARK0(c4, k3x3, kPartiallyMasked, kErode, 640, 480)

// RUN_BENCHMARK0(c1, k5x5, kPartiallyMasked, kErode, 640, 480)
// RUN_BENCHMARK1(c3, k5x5, kPartiallyMasked, kErode, 640, 480)
// RUN_BENCHMARK0(c4, k5x5, kPartiallyMasked, kErode, 640, 480)

// RUN_BENCHMARK0(c1, k11x11, kPartiallyMasked, kErode, 1920, 1080)
// RUN_BENCHMARK1(c3, k11x11, kPartiallyMasked, kErode, 1920, 1080)
// RUN_BENCHMARK0(c4, k11x11, kPartiallyMasked, kErode, 1920, 1080)

// RUN_BENCHMARK0(c1, k15x15, kPartiallyMasked, kErode, 1920, 1080)
// RUN_BENCHMARK1(c3, k15x15, kPartiallyMasked, kErode, 1920, 1080)
// RUN_BENCHMARK0(c4, k15x15, kPartiallyMasked, kErode, 1920, 1080)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, ksize, width, height, function)        \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_cuda, type, c1, ksize, kFullyMasked,       \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_x86_cuda, type, c3, ksize, kFullyMasked,   \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_cuda, type, c4, ksize, kFullyMasked,       \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_cuda, type, c1, ksize, kPartiallyMasked,   \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_x86_cuda, type, c3, ksize,                 \
                   kPartiallyMasked, function)->Args({width, height});         \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_cuda, type, c4, ksize, kPartiallyMasked,   \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, ksize, width, height, function)        \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, type, c1, ksize, kFullyMasked,          \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, type, c3, ksize, kFullyMasked,          \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, type, c4, ksize, kFullyMasked,          \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, type, c1, ksize, kPartiallyMasked,      \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, type, c3, ksize, kPartiallyMasked,      \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_cuda, type, c4, ksize, kPartiallyMasked,      \
                   function)->Args({width, height})->                          \
                   UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, k3x3, 640, 480, kDilate)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, k5x5, 640, 480, kDilate)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, k11x11, 1920, 1080, kDilate)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, k15x15, 1920, 1080, kDilate)
RUN_OPENCV_TYPE_FUNCTIONS(float, k3x3, 640, 480, kDilate)
RUN_OPENCV_TYPE_FUNCTIONS(float, k5x5, 640, 480, kDilate)
RUN_OPENCV_TYPE_FUNCTIONS(float, k11x11, 1920, 1080, kDilate)
RUN_OPENCV_TYPE_FUNCTIONS(float, k15x15, 1920, 1080, kDilate)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, k3x3, 640, 480, kDilate)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, k5x5, 640, 480, kDilate)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, k11x11, 1920, 1080, kDilate)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, k15x15, 1920, 1080, kDilate)
RUN_PPL_CV_TYPE_FUNCTIONS(float, k3x3, 640, 480, kDilate)
RUN_PPL_CV_TYPE_FUNCTIONS(float, k5x5, 640, 480, kDilate)
RUN_PPL_CV_TYPE_FUNCTIONS(float, k11x11, 1920, 1080, kDilate)
RUN_PPL_CV_TYPE_FUNCTIONS(float, k15x15, 1920, 1080, kDilate)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, k3x3, 640, 480, kErode)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, k5x5, 640, 480, kErode)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, k11x11, 1920, 1080, kErode)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, k15x15, 1920, 1080, kErode)
RUN_OPENCV_TYPE_FUNCTIONS(float, k3x3, 640, 480, kErode)
RUN_OPENCV_TYPE_FUNCTIONS(float, k5x5, 640, 480, kErode)
RUN_OPENCV_TYPE_FUNCTIONS(float, k11x11, 1920, 1080, kErode)
RUN_OPENCV_TYPE_FUNCTIONS(float, k15x15, 1920, 1080, kErode)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, k3x3, 640, 480, kErode)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, k5x5, 640, 480, kErode)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, k11x11, 1920, 1080, kErode)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, k15x15, 1920, 1080, kErode)
RUN_PPL_CV_TYPE_FUNCTIONS(float, k3x3, 640, 480, kErode)
RUN_PPL_CV_TYPE_FUNCTIONS(float, k5x5, 640, 480, kErode)
RUN_PPL_CV_TYPE_FUNCTIONS(float, k11x11, 1920, 1080, kErode)
RUN_PPL_CV_TYPE_FUNCTIONS(float, k15x15, 1920, 1080, kErode)
