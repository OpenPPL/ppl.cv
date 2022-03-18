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

#include "ppl/cv/cuda/sobel.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename Tsrc, typename Tdst, int channels, int ksize,
          ppl::cv::BorderType border_type>
void BM_Sobel_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int dxy = 1;
  float scale = 1.f;
  float delta = 0.f;

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Sobel<Tsrc, Tdst, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data,
        gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data, dxy, 0, ksize,
        scale, delta, border_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::Sobel<Tsrc, Tdst, channels>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data,
          gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data, dxy, 0, ksize,
          scale, delta, border_type);
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

template <typename Tsrc, typename Tdst, int channels, int ksize,
          ppl::cv::BorderType border_type>
void BM_Sobel_opencv_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  cv::BorderTypes border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_REPLICATE) {
    border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    border = cv::BORDER_REFLECT_101;
  }
  else {
  }

  int dxy = 1;
  float scale = 1.f;

  int iterations = 1000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::Ptr<cv::cuda::Filter> sobel =
      cv::cuda::createSobelFilter(gpu_src.type(), gpu_dst.type(), dxy, 0,
                                  ksize, scale, border);
      sobel->apply(gpu_src, gpu_dst);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::Ptr<cv::cuda::Filter> sobel =
        cv::cuda::createSobelFilter(gpu_src.type(), gpu_dst.type(), dxy, 0,
                                    ksize, scale, border);
        sobel->apply(gpu_src, gpu_dst);
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

template <typename Tsrc, typename Tdst, int channels, int ksize,
          ppl::cv::BorderType border_type>
void BM_Sobel_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));

  cv::BorderTypes border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_REPLICATE) {
    border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    border = cv::BORDER_REFLECT_101;
  }
  else {
  }

  int dxy = 1;
  float scale = 1.f;
  float delta = 0.f;

  for (auto _ : state) {
    cv::Sobel(src, dst, dst.depth(), dxy, 0, ksize, scale, delta, border);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK0(src_type, dst_type, ksize, border_type, width, height)  \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86_cuda, src_type, dst_type, c1, ksize,    \
                   border_type)->Args({width, height});                        \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c1, ksize,           \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86_cuda, src_type, dst_type, c3, ksize,    \
                   border_type)->Args({width, height});                        \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c3, ksize,           \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86_cuda, src_type, dst_type, c4, ksize,    \
                   border_type)->Args({width, height});                        \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c4, ksize,           \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);

// RUN_BENCHMARK0(uchar, uchar, 1, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 1, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 1, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 3, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 3, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 3, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 5, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 5, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 5, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 7, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 7, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, uchar, 7, ppl::cv::BORDER_REFLECT_101, 640, 480)

// RUN_BENCHMARK0(uchar, short, 1, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, short, 1, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, short, 1, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, short, 3, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, short, 3, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, short, 3, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, short, 5, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, short, 5, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, short, 5, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(uchar, short, 7, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(uchar, short, 7, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(uchar, short, 7, ppl::cv::BORDER_REFLECT_101, 640, 480)

// RUN_BENCHMARK0(float, float, 1, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, float, 1, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, float, 1, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(float, float, 3, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, float, 3, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, float, 3, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(float, float, 5, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, float, 5, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, float, 5, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK0(float, float, 7, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK0(float, float, 7, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK0(float, float, 7, ppl::cv::BORDER_REFLECT_101, 640, 480)

#define RUN_BENCHMARK1(src_type, dst_type, ksize, border_type, width, height)  \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_cuda, src_type, dst_type, c1, ksize,        \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c1, ksize,           \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_cuda, src_type, dst_type, c3, ksize,        \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c3, ksize,           \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_cuda, src_type, dst_type, c4, ksize,        \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c4, ksize,           \
                   border_type)->Args({width, height})->UseManualTime()->      \
                   Iterations(10);

// RUN_BENCHMARK1(uchar, uchar, 1, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 1, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 1, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 3, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 3, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 3, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 5, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 5, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 5, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 7, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 7, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, uchar, 7, ppl::cv::BORDER_REFLECT_101, 640, 480)

// RUN_BENCHMARK1(uchar, short, 1, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, short, 1, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, short, 1, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, short, 3, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, short, 3, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, short, 3, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, short, 5, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, short, 5, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, short, 5, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(uchar, short, 7, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(uchar, short, 7, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(uchar, short, 7, ppl::cv::BORDER_REFLECT_101, 640, 480)

// RUN_BENCHMARK1(float, float, 1, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(float, float, 1, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(float, float, 1, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(float, float, 3, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(float, float, 3, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(float, float, 3, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(float, float, 5, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(float, float, 5, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(float, float, 5, ppl::cv::BORDER_REFLECT_101, 640, 480)
// RUN_BENCHMARK1(float, float, 7, ppl::cv::BORDER_REPLICATE, 640, 480)
// RUN_BENCHMARK1(float, float, 7, ppl::cv::BORDER_REFLECT, 640, 480)
// RUN_BENCHMARK1(float, float, 7, ppl::cv::BORDER_REFLECT_101, 640, 480)

#define RUN_OPENCV_TYPE_FUNCTIONS(src_type, dst_type, ksize, border_type)      \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86_cuda, src_type, dst_type, c1, ksize,    \
                   border_type)->Args({640, 480});                             \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86_cuda, src_type, dst_type, c3, ksize,    \
                   border_type)->Args({640, 480});                             \
BENCHMARK_TEMPLATE(BM_Sobel_opencv_x86_cuda, src_type, dst_type, c4, ksize,    \
                   border_type)->Args({640, 480});

#define RUN_PPL_CV_TYPE_FUNCTIONS(src_type, dst_type, ksize, border_type)      \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c1, ksize,           \
                   border_type)->Args({640, 480})->UseManualTime()->           \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c3, ksize,           \
                   border_type)->Args({640, 480})->UseManualTime()->           \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Sobel_ppl_cuda, src_type, dst_type, c4, ksize,           \
                   border_type)->Args({640, 480})->UseManualTime()->           \
                   Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 3, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 3, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 3, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, short, 3, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, short, 3, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, short, 3, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REFLECT_101)

RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 7, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 7, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, uchar, 7, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, short, 7, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, short, 7, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, short, 7, ppl::cv::BORDER_REFLECT_101)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 7, ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 7, ppl::cv::BORDER_REFLECT)
RUN_OPENCV_TYPE_FUNCTIONS(float, float, 7, ppl::cv::BORDER_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 3, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 3, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 3, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, short, 3, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, short, 3, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, short, 3, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 3, ppl::cv::BORDER_REFLECT_101)

RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 7, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 7, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, uchar, 7, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, short, 7, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, short, 7, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(uchar, short, 7, ppl::cv::BORDER_REFLECT_101)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 7, ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 7, ppl::cv::BORDER_REFLECT)
RUN_PPL_CV_TYPE_FUNCTIONS(float, float, 7, ppl::cv::BORDER_REFLECT_101)
