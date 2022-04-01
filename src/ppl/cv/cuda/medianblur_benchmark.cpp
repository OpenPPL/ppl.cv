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
#include "ppl/cv/cuda/use_memory_pool.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

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

  int iterations = 100;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (sizeof(T) == 1 && ksize > 7) {
    cudaEventRecord(start, 0);
    size_t volume = width * channels * (height + 255) / 256 * 272 *
                    sizeof(ushort);
    size_t ceiled_volume = ppl::cv::cuda::ceil1DVolume(volume);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "activateGpuMemoryPool() time: " << elapsed_time * 1000000
              << " ns" << std::endl;
  }

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::MedianBlur<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data, ksize, ppl::cv::BORDER_REPLICATE);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::MedianBlur<T, channels>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
          (T*)gpu_dst.data, ksize, ppl::cv::BORDER_REPLICATE);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  if (sizeof(T) == 1 && ksize > 7) {
    cudaEventRecord(start, 0);
    ppl::cv::cuda::shutDownGpuMemoryPool();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "shutDownGpuMemoryPool() time: " << elapsed_time * 1000000
              << " ns" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
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

  int iterations = 100;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::Ptr<cv::cuda::Filter> median_filter =
      cv::cuda::createMedianFilter(gpu_src.type(), ksize, 128);
    median_filter->apply(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::Ptr<cv::cuda::Filter> median_filter =
        cv::cuda::createMedianFilter(gpu_src.type(), ksize, 128);
      median_filter->apply(gpu_src, gpu_dst);
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
// RUN_BENCHMARK0(uchar, 15, 640, 480)
// RUN_BENCHMARK0(uchar, 25, 640, 480)

#define RUN_BENCHMARK1(type, ksize, width, height)                             \
BENCHMARK_TEMPLATE(BM_MedianBlur_opencv_cuda, type, c1, ksize)->               \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_MedianBlur_ppl_cuda, type, c1, ksize)->                  \
                   Args({width, height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK1(uchar, 3, 640, 480)
// RUN_BENCHMARK1(uchar, 5, 640, 480)
// RUN_BENCHMARK1(uchar, 9, 640, 480)
// RUN_BENCHMARK1(uchar, 15, 640, 480)
// RUN_BENCHMARK1(uchar, 25, 640, 480)
// RUN_BENCHMARK1(uchar, 43, 640, 480)

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
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 15)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, 25)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 3)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 3)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 5)
RUN_PPL_CV_TYPE_FUNCTIONS(float, 5)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 9)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 15)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, 25)
