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

#include "ppl/cv/ocl/dilate.h"
#include "ppl/cv/ocl/erode.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

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
void BM_Dilate_ppl_ocl(benchmark::State &state) {
  ppl::common::ocl::createSharedFrameChain(false);
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();

  int width  = state.range(0);
  int height = state.range(1);
  cv::Mat src;
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,
                          channels));
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int src_bytes = src.rows * src.step;
  int dst_bytes = dst.rows * dst.step;
  cl_int error_code = 0;
  cl_mem gpu_src = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  src_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_dst = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                  dst_bytes, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_TRUE, 0, src_bytes,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

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

  int iterations = 100;
  struct timeval start, end;

  // Warm up the GPU.
  for (int i = 0; i < iterations; i++) {
    ppl::cv::ocl::Dilate<T, channels>(queue, src.rows, src.cols,
        src.step / sizeof(T), gpu_src, ksize, ksize, mask,
        dst.step / sizeof(T), gpu_dst, ppl::cv::BORDER_REFLECT);
  }
  clFinish(queue);

  for (auto _ : state) {
    gettimeofday(&start, NULL);
    for (int i = 0; i < iterations; i++) {
      if (function == kDilate) {
        ppl::cv::ocl::Dilate<T, channels>(queue, src.rows, src.cols,
            src.step / sizeof(T), gpu_src, ksize, ksize, mask,
            dst.step / sizeof(T), gpu_dst, ppl::cv::BORDER_REFLECT);
      }
      else if (function == kErode) {
        ppl::cv::ocl::Erode<T, channels>(queue, src.rows, src.cols,
            src.step / sizeof(T), gpu_src, ksize, ksize, mask,
            dst.step / sizeof(T), gpu_dst, ppl::cv::BORDER_REFLECT);
      }
      else {
      }
    }
    clFinish(queue);
    gettimeofday(&end, NULL);
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;
    state.SetIterationTime(time * 1e-6);
  }
  state.SetItemsProcessed(state.iterations() * 1);

  free(mask);
  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
}

template <typename T, int channels, int ksize, Masks mask_type,
          Functions function>
void BM_Dilate_opencv_ocl(benchmark::State &state) {
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

#define RUN_BENCHMARK(ksize, mask_type, function, width, height)               \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, uchar, c1, ksize, mask_type,          \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, uchar, c1, ksize, mask_type, function)-> \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, uchar, c3, ksize, mask_type,          \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, uchar, c3, ksize, mask_type, function)-> \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, uchar, c4, ksize, mask_type,          \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, uchar, c4, ksize, mask_type, function)-> \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, float, c1, ksize, mask_type,          \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, float, c1, ksize, mask_type, function)-> \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, float, c3, ksize, mask_type,          \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, float, c3, ksize, mask_type, function)-> \
                   Args({width, height})->UseManualTime()->Iterations(10);     \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, float, c4, ksize, mask_type,          \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, float, c4, ksize, mask_type, function)-> \
                   Args({width, height})->UseManualTime()->Iterations(10);

RUN_BENCHMARK(k3x3, kFullyMasked, kDilate, 640, 480)
RUN_BENCHMARK(k5x5, kFullyMasked, kDilate, 640, 480)
RUN_BENCHMARK(k9x9, kFullyMasked, kDilate, 640, 480)
RUN_BENCHMARK(k15x15, kFullyMasked, kDilate, 640, 480)
RUN_BENCHMARK(k3x3, kFullyMasked, kDilate, 1920, 1080)
RUN_BENCHMARK(k5x5, kFullyMasked, kDilate, 1920, 1080)
RUN_BENCHMARK(k9x9, kFullyMasked, kDilate, 1920, 1080)
RUN_BENCHMARK(k15x15, kFullyMasked, kDilate, 1920, 1080)

RUN_BENCHMARK(k3x3, kPartiallyMasked, kDilate, 640, 480)
RUN_BENCHMARK(k5x5, kPartiallyMasked, kDilate, 640, 480)
RUN_BENCHMARK(k9x9, kPartiallyMasked, kDilate, 640, 480)
RUN_BENCHMARK(k15x15, kPartiallyMasked, kDilate, 640, 480)
RUN_BENCHMARK(k3x3, kPartiallyMasked, kDilate, 1920, 1080)
RUN_BENCHMARK(k5x5, kPartiallyMasked, kDilate, 1920, 1080)
RUN_BENCHMARK(k9x9, kPartiallyMasked, kDilate, 1920, 1080)
RUN_BENCHMARK(k15x15, kPartiallyMasked, kDilate, 1920, 1080)

RUN_BENCHMARK(k3x3, kFullyMasked, kErode, 640, 480)
RUN_BENCHMARK(k5x5, kFullyMasked, kErode, 640, 480)
RUN_BENCHMARK(k9x9, kFullyMasked, kErode, 640, 480)
RUN_BENCHMARK(k15x15, kFullyMasked, kErode, 640, 480)
RUN_BENCHMARK(k3x3, kFullyMasked, kErode, 1920, 1080)
RUN_BENCHMARK(k5x5, kFullyMasked, kErode, 1920, 1080)
RUN_BENCHMARK(k9x9, kFullyMasked, kErode, 1920, 1080)
RUN_BENCHMARK(k15x15, kFullyMasked, kErode, 1920, 1080)

RUN_BENCHMARK(k3x3, kPartiallyMasked, kErode, 640, 480)
RUN_BENCHMARK(k5x5, kPartiallyMasked, kErode, 640, 480)
RUN_BENCHMARK(k9x9, kPartiallyMasked, kErode, 640, 480)
RUN_BENCHMARK(k15x15, kPartiallyMasked, kErode, 640, 480)
RUN_BENCHMARK(k3x3, kPartiallyMasked, kErode, 1920, 1080)
RUN_BENCHMARK(k5x5, kPartiallyMasked, kErode, 1920, 1080)
RUN_BENCHMARK(k9x9, kPartiallyMasked, kErode, 1920, 1080)
RUN_BENCHMARK(k15x15, kPartiallyMasked, kErode, 1920, 1080)

#define RUN_OPENCV_FUNCTIONS(type, ksize, width, height, function)             \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, type, c1, ksize, kFullyMasked,        \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, type, c3, ksize, kFullyMasked,        \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, type, c4, ksize, kFullyMasked,        \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, type, c1, ksize, kPartiallyMasked,    \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, type, c3, ksize, kPartiallyMasked,    \
                   function)->Args({width, height});                           \
BENCHMARK_TEMPLATE(BM_Dilate_opencv_ocl, type, c4, ksize, kPartiallyMasked,    \
                   function)->Args({width, height});

#define RUN_PPL_CV_FUNCTIONS(type, ksize, width, height, function)             \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, type, c1, ksize, kFullyMasked,           \
                   function)->Args({width, height})->UseManualTime()->         \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, type, c3, ksize, kFullyMasked,           \
                   function)->Args({width, height})->UseManualTime()->         \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, type, c4, ksize, kFullyMasked,           \
                   function)->Args({width, height})->UseManualTime()->         \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, type, c1, ksize, kPartiallyMasked,       \
                   function)->Args({width, height})->UseManualTime()->         \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, type, c3, ksize, kPartiallyMasked,       \
                   function)->Args({width, height})->UseManualTime()->         \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_Dilate_ppl_ocl, type, c4, ksize, kPartiallyMasked,       \
                   function)->Args({width, height})->UseManualTime()->         \
                   Iterations(10);

// RUN_OPENCV_FUNCTIONS(uchar, k3x3, 640, 480, kDilate)
// RUN_OPENCV_FUNCTIONS(uchar, k5x5, 640, 480, kDilate)
// RUN_OPENCV_FUNCTIONS(uchar, k9x9, 640, 480, kDilate)
// RUN_OPENCV_FUNCTIONS(uchar, k15x15, 640, 480, kDilate)
// RUN_OPENCV_FUNCTIONS(float, k3x3, 640, 480, kDilate)
// RUN_OPENCV_FUNCTIONS(float, k5x5, 640, 480, kDilate)
// RUN_OPENCV_FUNCTIONS(float, k9x9, 640, 480, kDilate)
// RUN_OPENCV_FUNCTIONS(float, k15x15, 640, 480, kDilate)
// RUN_OPENCV_FUNCTIONS(uchar, k3x3, 1920, 1080, kDilate)
// RUN_OPENCV_FUNCTIONS(uchar, k5x5, 1920, 1080, kDilate)
// RUN_OPENCV_FUNCTIONS(uchar, k9x9, 1920, 1080, kDilate)
// RUN_OPENCV_FUNCTIONS(uchar, k15x15, 1920, 1080, kDilate)
// RUN_OPENCV_FUNCTIONS(float, k3x3, 1920, 1080, kDilate)
// RUN_OPENCV_FUNCTIONS(float, k5x5, 1920, 1080, kDilate)
// RUN_OPENCV_FUNCTIONS(float, k9x9, 1920, 1080, kDilate)
// RUN_OPENCV_FUNCTIONS(float, k15x15, 1920, 1080, kDilate)

// RUN_PPL_CV_FUNCTIONS(uchar, k3x3, 640, 480, kDilate)
// RUN_PPL_CV_FUNCTIONS(uchar, k5x5, 640, 480, kDilate)
// RUN_PPL_CV_FUNCTIONS(uchar, k9x9, 640, 480, kDilate)
// RUN_PPL_CV_FUNCTIONS(uchar, k15x15, 640, 480, kDilate)
// RUN_PPL_CV_FUNCTIONS(float, k3x3, 640, 480, kDilate)
// RUN_PPL_CV_FUNCTIONS(float, k5x5, 640, 480, kDilate)
// RUN_PPL_CV_FUNCTIONS(float, k9x9, 640, 480, kDilate)
// RUN_PPL_CV_FUNCTIONS(float, k15x15, 640, 480, kDilate)
// RUN_PPL_CV_FUNCTIONS(uchar, k3x3, 1920, 1080, kDilate)
// RUN_PPL_CV_FUNCTIONS(uchar, k5x5, 1920, 1080, kDilate)
// RUN_PPL_CV_FUNCTIONS(uchar, k9x9, 1920, 1080, kDilate)
// RUN_PPL_CV_FUNCTIONS(uchar, k15x15, 1920, 1080, kDilate)
// RUN_PPL_CV_FUNCTIONS(float, k3x3, 1920, 1080, kDilate)
// RUN_PPL_CV_FUNCTIONS(float, k5x5, 1920, 1080, kDilate)
// RUN_PPL_CV_FUNCTIONS(float, k9x9, 1920, 1080, kDilate)
// RUN_PPL_CV_FUNCTIONS(float, k15x15, 1920, 1080, kDilate)

// RUN_OPENCV_FUNCTIONS(uchar, k3x3, 640, 480, kErode)
// RUN_OPENCV_FUNCTIONS(uchar, k5x5, 640, 480, kErode)
// RUN_OPENCV_FUNCTIONS(uchar, k9x9, 640, 480, kErode)
// RUN_OPENCV_FUNCTIONS(uchar, k15x15, 640, 480, kErode)
// RUN_OPENCV_FUNCTIONS(float, k3x3, 640, 480, kErode)
// RUN_OPENCV_FUNCTIONS(float, k5x5, 640, 480, kErode)
// RUN_OPENCV_FUNCTIONS(float, k9x9, 640, 480, kErode)
// RUN_OPENCV_FUNCTIONS(float, k15x15, 640, 480, kErode)
// RUN_OPENCV_FUNCTIONS(uchar, k3x3, 1920, 1080, kErode)
// RUN_OPENCV_FUNCTIONS(uchar, k5x5, 1920, 1080, kErode)
// RUN_OPENCV_FUNCTIONS(uchar, k9x9, 1920, 1080, kErode)
// RUN_OPENCV_FUNCTIONS(uchar, k15x15, 1920, 1080, kErode)
// RUN_OPENCV_FUNCTIONS(float, k3x3, 1920, 1080, kErode)
// RUN_OPENCV_FUNCTIONS(float, k5x5, 1920, 1080, kErode)
// RUN_OPENCV_FUNCTIONS(float, k9x9, 1920, 1080, kErode)
// RUN_OPENCV_FUNCTIONS(float, k15x15, 1920, 1080, kErode)

// RUN_PPL_CV_FUNCTIONS(uchar, k3x3, 640, 480, kErode)
// RUN_PPL_CV_FUNCTIONS(uchar, k5x5, 640, 480, kErode)
// RUN_PPL_CV_FUNCTIONS(uchar, k9x9, 640, 480, kErode)
// RUN_PPL_CV_FUNCTIONS(uchar, k15x15, 640, 480, kErode)
// RUN_PPL_CV_FUNCTIONS(float, k3x3, 640, 480, kErode)
// RUN_PPL_CV_FUNCTIONS(float, k5x5, 640, 480, kErode)
// RUN_PPL_CV_FUNCTIONS(float, k9x9, 640, 480, kErode)
// RUN_PPL_CV_FUNCTIONS(float, k15x15, 640, 480, kErode)
// RUN_PPL_CV_FUNCTIONS(uchar, k3x3, 1920, 1080, kErode)
// RUN_PPL_CV_FUNCTIONS(uchar, k5x5, 1920, 1080, kErode)
// RUN_PPL_CV_FUNCTIONS(uchar, k9x9, 1920, 1080, kErode)
// RUN_PPL_CV_FUNCTIONS(uchar, k15x15, 1920, 1080, kErode)
// RUN_PPL_CV_FUNCTIONS(float, k3x3, 1920, 1080, kErode)
// RUN_PPL_CV_FUNCTIONS(float, k5x5, 1920, 1080, kErode)
// RUN_PPL_CV_FUNCTIONS(float, k9x9, 1920, 1080, kErode)
// RUN_PPL_CV_FUNCTIONS(float, k15x15, 1920, 1080, kErode)
