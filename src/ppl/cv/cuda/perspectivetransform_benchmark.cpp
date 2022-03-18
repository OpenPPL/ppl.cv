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

#include "ppl/cv/cuda/perspectivetransform.h"

#include "opencv2/core.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

template <typename T, int srcCns, int dstCns>
void BM_PerspectiveTransform_ppl_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, trans_coeffs0;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  trans_coeffs0 = createSourceImage(dstCns + 1, srcCns + 1,
                                    CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, dstCns));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  int coeff_size = (dstCns + 1) * (srcCns + 1) * sizeof(float);
  float* trans_coeff1 = (float*)malloc(coeff_size);
  copyMatToArray(trans_coeffs0, trans_coeff1);

  int iterations = 3000;
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::cuda::PerspectiveTransform<T, srcCns, dstCns>(0, gpu_src.rows,
        gpu_src.cols, gpu_src.step / sizeof(T), (T*)gpu_src.data,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, trans_coeff1);
  }
  cudaDeviceSynchronize();

  for (auto _ : state) {
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
      ppl::cv::cuda::PerspectiveTransform<T, srcCns, dstCns>(0, gpu_src.rows,
          gpu_src.cols, gpu_src.step / sizeof(T), (T*)gpu_src.data,
          gpu_dst.step / sizeof(T), (T*)gpu_dst.data, trans_coeff1);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    int time = elapsed_time * 1000 / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  free(trans_coeff1);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename T, int srcCns, int dstCns>
void BM_PerspectiveTransform_opencv_x86_cuda(benchmark::State &state) {
  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src, trans_coeffs;
  src = createSourceImage(height, width,
                          CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  trans_coeffs = createSourceImage(dstCns + 1, srcCns + 1,
                                   CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(height, width,
              CV_MAKETYPE(cv::DataType<T>::depth, dstCns));

  for (auto _ : state) {
      cv::perspectiveTransform(src, dst, trans_coeffs);
  }
  state.SetItemsProcessed(state.iterations() * 1);
}

#define RUN_BENCHMARK(srcCns, dstCns)                                          \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86_cuda, float, srcCns,     \
                   dstCns)->Args({320, 240});                                  \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_cuda, float, srcCns, dstCns)->  \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86_cuda, float, srcCns,     \
                   dstCns)->Args({640, 480});                                  \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_cuda, float, srcCns, dstCns)->  \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86_cuda, float, srcCns,     \
                   dstCns)->Args({1280, 720});                                 \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_cuda, float, srcCns, dstCns)->  \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86_cuda, float, srcCns,     \
                   dstCns)->Args({1920, 1080});                                \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_cuda, float, srcCns, dstCns)->  \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

// RUN_BENCHMARK(2, 2)
// RUN_BENCHMARK(2, 3)
// RUN_BENCHMARK(3, 2)
// RUN_BENCHMARK(3, 3)

#define RUN_OPENCV_TYPE_FUNCTIONS(srcCns, dstCns)                              \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86_cuda, float, srcCns,     \
                   dstCns)->Args({320, 240});                                  \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86_cuda, float, srcCns,     \
                   dstCns)->Args({640, 480});                                  \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86_cuda, float, srcCns,     \
                   dstCns)->Args({1280, 720});                                 \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_opencv_x86_cuda, float, srcCns,     \
                   dstCns)->Args({1920, 1080});

#define RUN_PPL_CV_TYPE_FUNCTIONS(srcCns, dstCns)                              \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_cuda, float, srcCns, dstCns)->  \
                   Args({320, 240})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_cuda, float, srcCns, dstCns)->  \
                   Args({640, 480})->UseManualTime()->Iterations(10);          \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_cuda, float, srcCns, dstCns)->  \
                   Args({1280, 720})->UseManualTime()->Iterations(10);         \
BENCHMARK_TEMPLATE(BM_PerspectiveTransform_ppl_cuda, float, srcCns, dstCns)->  \
                   Args({1920, 1080})->UseManualTime()->Iterations(10);

RUN_OPENCV_TYPE_FUNCTIONS(2, 2)
RUN_OPENCV_TYPE_FUNCTIONS(2, 3)
RUN_OPENCV_TYPE_FUNCTIONS(3, 2)
RUN_OPENCV_TYPE_FUNCTIONS(3, 3)

RUN_PPL_CV_TYPE_FUNCTIONS(2, 2)
RUN_PPL_CV_TYPE_FUNCTIONS(2, 3)
RUN_PPL_CV_TYPE_FUNCTIONS(3, 2)
RUN_PPL_CV_TYPE_FUNCTIONS(3, 3)
