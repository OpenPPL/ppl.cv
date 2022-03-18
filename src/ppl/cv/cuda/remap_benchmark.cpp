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

#include "ppl/cv/cuda/remap.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int channels, ppl::cv::InterpolationType inter_type,
          ppl::cv::BorderType border_type>
void BM_Remap_ppl_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src, map_x0, map_y0;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  map_x0 = createSourceImage(dst_height, dst_width,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  map_y0 = createSourceImage(dst_height, dst_width,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(dst_height, dst_width, src.type());
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int map_size = dst_height * dst_width * sizeof(float);
  float* map_x1 = (float*)malloc(map_size);
  float* map_y1 = (float*)malloc(map_size);
  float* gpu_map_x;
  float* gpu_map_y;
  cudaMalloc((void**)&gpu_map_x, map_size);
  cudaMalloc((void**)&gpu_map_y, map_size);
  copyMatToArray(map_x0, map_x1);
  copyMatToArray(map_y0, map_y1);
  cudaMemcpy(gpu_map_x, map_x1, map_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_map_y, map_y1, map_size, cudaMemcpyHostToDevice);

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::Remap<T, channels>(0, src.rows, src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, gpu_map_x, gpu_map_y,
        inter_type, border_type);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::Remap<T, channels>(0, src.rows, src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
          gpu_dst.step / sizeof(T), (T*)gpu_dst.data, gpu_map_x, gpu_map_y,
          inter_type, border_type);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  free(map_x1);
  free(map_y1);
  cudaFree(gpu_map_x);
  cudaFree(gpu_map_y);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename T, int channels, ppl::cv::InterpolationType inter_type,
          ppl::cv::BorderType border_type>
void BM_Remap_opencv_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src, map_x, map_y;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  map_x = createSourceImage(dst_height, dst_width,
                            CV_MAKETYPE(cv::DataType<float>::depth, 1));
  map_y = createSourceImage(dst_height, dst_width,
                            CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(dst_height, dst_width, src.type());
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_map_x(map_x);
  cv::cuda::GpuMat gpu_map_y(map_y);

  int cv_iterpolation;
  if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
    cv_iterpolation = cv::INTER_LINEAR;
  }
  else {
    cv_iterpolation = cv::INTER_NEAREST;
  }

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else {
  }

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    cv::cuda::remap(gpu_src, gpu_dst, gpu_map_x, gpu_map_y, cv_iterpolation,
                    cv_border);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      cv::cuda::remap(gpu_src, gpu_dst, gpu_map_x, gpu_map_y,
                      cv_iterpolation, cv_border);
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

template <typename T, int channels, ppl::cv::InterpolationType inter_type,
          ppl::cv::BorderType border_type>
void BM_Remap_opencv_x86_cuda(benchmark::State &state) {
  int src_width  = state.range(0);
  int src_height = state.range(1);
  int dst_width  = state.range(2);
  int dst_height = state.range(3);
  cv::Mat src, map_x, map_y;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  map_x = createSourceImage(dst_height, dst_width,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  map_y = createSourceImage(dst_height, dst_width,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(dst_height, dst_width, src.type());

  int cv_iterpolation;
  if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
    cv_iterpolation = cv::INTER_LINEAR;
  }
  else {
    cv_iterpolation = cv::INTER_NEAREST;
  }

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else {
  }

  for (auto _ : state) {
    cv::remap(src, dst, map_x, map_y, cv_iterpolation, cv_border);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(channels, inter_type, border_type, src_width, src_height,\
                      dst_width, dst_height)                                   \
BENCHMARK_TEMPLATE(BM_Remap_opencv_x86_cuda, uchar, channels, inter_type,      \
                   border_type)->Args({src_width, src_height, dst_width,       \
                   dst_height});                                               \
BENCHMARK_TEMPLATE(BM_Remap_opencv_cuda, uchar, channels, inter_type,          \
                   border_type)->Args({src_width, src_height, dst_width,       \
                   dst_height})->UseManualTime()->Iterations(10);              \
BENCHMARK_TEMPLATE(BM_Remap_ppl_cuda, uchar, channels, inter_type,             \
                   border_type)->Args({src_width, src_height, dst_width,       \
                   dst_height})->UseManualTime()->Iterations(10);              \
BENCHMARK_TEMPLATE(BM_Remap_opencv_x86_cuda, float, channels, inter_type,      \
                   border_type)->Args({src_width, src_height, dst_width,       \
                   dst_height});                                               \
BENCHMARK_TEMPLATE(BM_Remap_opencv_cuda, float, channels, inter_type,          \
                   border_type)->Args({src_width, src_height, dst_width,       \
                   dst_height})->UseManualTime()->Iterations(10);              \
BENCHMARK_TEMPLATE(BM_Remap_ppl_cuda, float, channels, inter_type,             \
                   border_type)->Args({src_width, src_height, dst_width,       \
                   dst_height})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT,
//               640, 480, 1280, 960)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE,
//               640, 480, 1280, 960)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT,
//               640, 480, 1280, 960)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT,
//               640, 480, 1280, 960)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE,
//               640, 480, 1280, 960)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT,
//               640, 480, 1280, 960)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_CONSTANT,
//               640, 480, 1280, 960)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_REPLICATE,
//               640, 480, 1280, 960)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT,
//               640, 480, 320, 240)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_LINEAR, ppl::cv::BORDER_TRANSPARENT,
//               640, 480, 1280, 960)

// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_CONSTANT, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_CONSTANT, 640, 480, 1280, 960)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_REPLICATE, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_REPLICATE, 640, 480, 1280, 960)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_TRANSPARENT, 640, 480, 320, 240)
// RUN_BENCHMARK(c1, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_TRANSPARENT, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_CONSTANT, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_CONSTANT, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_REPLICATE, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_REPLICATE, 640, 480, 1280, 960)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_TRANSPARENT, 640, 480, 320, 240)
// RUN_BENCHMARK(c3, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_TRANSPARENT, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_CONSTANT, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_CONSTANT, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_REPLICATE, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_REPLICATE, 640, 480, 1280, 960)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_TRANSPARENT, 640, 480, 320, 240)
// RUN_BENCHMARK(c4, ppl::cv::INTERPOLATION_NEAREST_POINT,
//               ppl::cv::BORDER_TRANSPARENT, 640, 480, 1280, 960)

#define RUN_OPENCV_TYPE_FUNCTIONS(type, inter_type, border_type)               \
BENCHMARK_TEMPLATE(BM_Remap_opencv_cuda, type, c1, inter_type, border_type)->  \
                   Args({640, 480, 320, 240})->UseManualTime()->               \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_opencv_cuda, type, c1, inter_type, border_type)->  \
                   Args({640, 480, 1280, 960})->UseManualTime()->              \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_opencv_cuda, type, c3, inter_type, border_type)->  \
                   Args({640, 480, 320, 240})->UseManualTime()->               \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_opencv_cuda, type, c3, inter_type, border_type)->  \
                   Args({640, 480, 1280, 960})->UseManualTime()->              \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_opencv_cuda, type, c4, inter_type, border_type)->  \
                   Args({640, 480, 320, 240})->UseManualTime()->               \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_opencv_cuda, type, c4, inter_type, border_type)->  \
                   Args({640, 480, 1280, 960})->UseManualTime()->              \
                   Iterations(10);

#define RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(type, inter_type, border_type)          \
BENCHMARK_TEMPLATE(BM_Remap_ppl_cuda, type, c1, inter_type, border_type)->     \
                   Args({640, 480, 320, 240})->UseManualTime()->               \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_ppl_cuda, type, c1, inter_type, border_type)->     \
                   Args({640, 480, 1280, 960})->UseManualTime()->              \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_ppl_cuda, type, c3, inter_type, border_type)->     \
                   Args({640, 480, 320, 240})->UseManualTime()->               \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_ppl_cuda, type, c3, inter_type, border_type)->     \
                   Args({640, 480, 1280, 960})->UseManualTime()->              \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_ppl_cuda, type, c4, inter_type, border_type)->     \
                   Args({640, 480, 320, 240})->UseManualTime()->               \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Remap_ppl_cuda, type, c4, inter_type, border_type)->     \
                   Args({640, 480, 1280, 960})->UseManualTime()->              \
                   Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_LINEAR,
                          ppl::cv::BORDER_CONSTANT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_LINEAR,
                          ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_LINEAR,
                          ppl::cv::BORDER_TRANSPARENT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_NEAREST_POINT,
                          ppl::cv::BORDER_CONSTANT)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_NEAREST_POINT,
                          ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_NEAREST_POINT,
                          ppl::cv::BORDER_TRANSPARENT)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_LINEAR,
                          ppl::cv::BORDER_CONSTANT)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_LINEAR,
                          ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_LINEAR,
                          ppl::cv::BORDER_TRANSPARENT)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_NEAREST_POINT,
                          ppl::cv::BORDER_CONSTANT)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_NEAREST_POINT,
                          ppl::cv::BORDER_REPLICATE)
RUN_OPENCV_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_NEAREST_POINT,
                          ppl::cv::BORDER_TRANSPARENT)

RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_LINEAR,
                               ppl::cv::BORDER_CONSTANT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_LINEAR,
                               ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_LINEAR,
                               ppl::cv::BORDER_TRANSPARENT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_NEAREST_POINT,
                               ppl::cv::BORDER_CONSTANT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_NEAREST_POINT,
                               ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(uchar, ppl::cv::INTERPOLATION_NEAREST_POINT,
                               ppl::cv::BORDER_TRANSPARENT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_LINEAR,
                               ppl::cv::BORDER_CONSTANT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_LINEAR,
                               ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_LINEAR,
                               ppl::cv::BORDER_TRANSPARENT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_NEAREST_POINT,
                               ppl::cv::BORDER_CONSTANT)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_NEAREST_POINT,
                               ppl::cv::BORDER_REPLICATE)
RUN_PPL_CV_CUDA_TYPE_FUNCTIONS(float, ppl::cv::INTERPOLATION_NEAREST_POINT,
                               ppl::cv::BORDER_TRANSPARENT)
