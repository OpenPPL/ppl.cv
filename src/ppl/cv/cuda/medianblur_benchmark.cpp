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

#include "ppl/cv/cuda/medianblur.h"

#ifdef _MSC_VER
#include <time.h>
#else 
#include <sys/time.h>
#endif

#include "opencv2/imgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;
using namespace ppl::cv::debug;

template <typename T, int channels, int ksize>
void BM_MedianBlur_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    MedianBlur<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data, ksize, BORDER_REPLICATE);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      MedianBlur<T, channels>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
          (T*)gpu_dst.data, ksize, BORDER_REPLICATE);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, int ksize>
void BM_MedianBlur_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int iterations = 1000;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::Ptr<cv::cuda::Filter> median_filter =
      cv::cuda::createMedianFilter(gpu_src.type(), ksize, 128);
    median_filter->apply(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      cv::Ptr<cv::cuda::Filter> median_filter =
        cv::cuda::createMedianFilter(gpu_src.type(), ksize, 128);
      median_filter->apply(gpu_src, gpu_dst);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

template <typename T, int channels, int ksize>
void BM_MedianBlur_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));

  for (auto _ : state) {
    cv::medianBlur(src, dst, ksize);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(type, ksize, width, height)                             \
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86_cuda, type, c1, ksize)->           \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_cuda, type, c1, ksize)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86_cuda, type, c3, ksize)->           \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_cuda, type, c3, ksize)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86_cuda, type, c4, ksize)->           \
                   Args({width, height});                                      \
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_cuda, type, c4, ksize)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK0(uchar, 3, 640, 480)
// RUN_BENCHMARK0(float, 3, 640, 480)
// RUN_BENCHMARK0(uchar, 5, 640, 480)
// RUN_BENCHMARK0(float, 5, 640, 480)
// RUN_BENCHMARK0(uchar, 9, 640, 480)
// RUN_BENCHMARK0(uchar, 11, 640, 480)

// RUN_BENCHMARK0(uchar, 3, 1920, 1080)
// RUN_BENCHMARK0(float, 3, 1920, 1080)
// RUN_BENCHMARK0(uchar, 5, 1920, 1080)
// RUN_BENCHMARK0(float, 5, 1920, 1080)
// RUN_BENCHMARK0(uchar, 9, 1920, 1080)
// RUN_BENCHMARK0(uchar, 11, 1920, 1080)

#define RUN_BENCHMARK1(type, ksize, width, height)                             \
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_cuda, type, c1, ksize)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_cuda, type, c1, ksize)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(uchar, 3, 640, 480)
// RUN_BENCHMARK1(uchar, 5, 640, 480)
// RUN_BENCHMARK1(uchar, 7, 640, 480)
// RUN_BENCHMARK1(uchar, 9, 640, 480)
// RUN_BENCHMARK1(uchar, 11, 640, 480)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, ksize)                                 \
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86_cuda, type, c1, ksize)->           \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86_cuda, type, c3, ksize)->           \
                   Args({640, 480});                                           \
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_x86_cuda, type, c4, ksize)->           \
                   Args({640, 480});

#define RUN_PPL_CV_TYPE_FUNCTIONS(type, ksize)                                 \
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_cuda, type, c1, ksize)->                  \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_cuda, type, c3, ksize)->                  \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_cuda, type, c4, ksize)->                  \
                   Args({640, 480})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, 3)
RUN_OPENCV_TYPE_FUNCTIONS(float, 3)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 5)
RUN_OPENCV_TYPE_FUNCTIONS(float, 5)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 9)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 11)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 3)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 3)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 5)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 5)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 9)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 11)
