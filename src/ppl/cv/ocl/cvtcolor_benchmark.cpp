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

#include "ppl/cv/ocl/cvtcolor.h"

#include <time.h>
#include <sys/time.h>

#include "opencv2/imgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/common/ocl/pplopencl.h"
#include "ppl/cv/debug.h"
#include "utility/infrastructure.h"

using namespace ppl::cv::debug;

#define BENCHMARK_PPL_CV_OCL(Function)                                         \
template <typename T, int src_channels, int dst_channels>                      \
void BM_CvtColor ## Function ## _ppl_ocl(benchmark::State &state) {            \
  ppl::common::ocl::createSharedFrameChain(false);                             \
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();  \
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();\
                                                                               \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src;                                                                 \
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,   \
                          src_channels));                                      \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth,               \
              dst_channels));                                                  \
                                                                               \
  int src_bytes = src.rows * src.step;                                         \
  int dst_bytes = dst.rows * dst.step;                                         \
  cl_int error_code = 0;                                                       \
  cl_mem gpu_src = clCreateBuffer(context,                                     \
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,   \
                                  src_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_dst = clCreateBuffer(context,                                     \
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,   \
                                  dst_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes,    \
                                    src.data, 0, NULL, NULL);                  \
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);                               \
                                                                               \
  int iterations = 100;                                                        \
  struct timeval start, end;                                                   \
                                                                               \
  /* Warm up the GPU. */                                                       \
  for (int i = 0; i < iterations; i++) {                                       \
    ppl::cv::ocl::Function<T>(queue, src.rows, src.cols, src.step / sizeof(T), \
                              gpu_src, dst.step / sizeof(T), gpu_dst);         \
  }                                                                            \
  clFinish(queue);                                                             \
                                                                               \
  for (auto _ : state) {                                                       \
    gettimeofday(&start, NULL);                                                \
    for (int i = 0; i < iterations; i++) {                                     \
      ppl::cv::ocl::Function<T>(queue, src.rows, src.cols,                     \
                                src.step / sizeof(T), gpu_src,                 \
                                dst.step / sizeof(T), gpu_dst);                \
    }                                                                          \
    clFinish(queue);                                                           \
    gettimeofday(&end, NULL);                                                  \
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -                         \
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;        \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  clReleaseMemObject(gpu_src);                                                 \
  clReleaseMemObject(gpu_dst);                                                 \
}

#define BENCHMARK_OPENCV_CPU(Function)                                         \
template <typename T, int src_channels, int dst_channels>                      \
void BM_CvtColor ## Function ## _opencv_ocl(benchmark::State &state) {         \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src;                                                                 \
  src = createSourceImage(height, width, CV_MAKETYPE(cv::DataType<T>::depth,   \
                          src_channels));                                      \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth,               \
              dst_channels));                                                  \
                                                                               \
  for (auto _ : state) {                                                       \
    cv::cvtColor(src, dst, cv::COLOR_ ## Function);                            \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
}

#define BENCHMARK_2FUNCTIONS_DECLARATION(Function)                             \
BENCHMARK_PPL_CV_OCL(Function)                                                 \
BENCHMARK_OPENCV_CPU(Function)

#define RUN_2BENCHMARKS(Function, src_channels, dst_channels, width, height)   \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_ocl, uchar, src_channels,\
                   dst_channels)->Args({width, height});                       \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, uchar, src_channels,   \
                   dst_channels)->Args({width, height})->UseManualTime()->     \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_ocl, float, src_channels,\
                   dst_channels)->Args({width, height});                       \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, float, src_channels,   \
                   dst_channels)->Args({width, height})->UseManualTime()->     \
                   Iterations(10);

#define RUN_BENCHMARK_COMPARISON_2FUNCTIONS(Function, src_channels,            \
                                            dst_channels)                      \
BENCHMARK_2FUNCTIONS_DECLARATION(Function)                                     \
RUN_2BENCHMARKS(Function, src_channels, dst_channels, 320, 240)                \
RUN_2BENCHMARKS(Function, src_channels, dst_channels, 640, 480)                \
RUN_2BENCHMARKS(Function, src_channels, dst_channels, 1280, 720)               \
RUN_2BENCHMARKS(Function, src_channels, dst_channels, 1920, 1080)

/************************ LAB Comparison with OpenCV *************************/
enum LabFunctions {
  kBGR2LAB,
  kRGB2LAB,
  kLAB2BGR,
  kLAB2RGB,
};

#define BENCHMARK_OPENCV_CPU_LAB(Function)                                     \
template <typename T, int src_channels, int dst_channels>                      \
void BM_CvtColor ## Function ## _opencv_ocl(benchmark::State &state) {         \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src;                                                                 \
  src = createSourceImage(height, width,                                       \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth,               \
              dst_channels));                                                  \
                                                                               \
  LabFunctions ppl_function = k ## Function;                                   \
  cv::ColorConversionCodes cv_code;                                            \
  if (ppl_function == kBGR2LAB) {                                              \
    cv_code = cv::COLOR_BGR2Lab;                                               \
  }                                                                            \
  else if (ppl_function == kRGB2LAB) {                                         \
    cv_code = cv::COLOR_RGB2Lab;                                               \
  }                                                                            \
  else if (ppl_function == kLAB2BGR) {                                         \
    cv_code = cv::COLOR_Lab2BGR;                                               \
  }                                                                            \
  else if (ppl_function == kLAB2RGB) {                                         \
    cv_code = cv::COLOR_Lab2RGB;                                               \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  for (auto _ : state) {                                                       \
    cv::cvtColor(src, dst, cv_code);                                           \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
}

#define BENCHMARK_LAB_2FUNCTIONS_DECLARATION(Function)                         \
BENCHMARK_PPL_CV_OCL(Function)                                                 \
BENCHMARK_OPENCV_CPU_LAB(Function)

#define RUN_LAB_2BENCHMARKS(Function, src_channels, dst_channels, width,       \
                            height)                                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_ocl, uchar, src_channels,\
                   dst_channels)->Args({width, height});                       \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, uchar, src_channels,   \
                   dst_channels)->Args({width, height})->UseManualTime()->     \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_ocl, float, src_channels,\
                   dst_channels)->Args({width, height});                       \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, float, src_channels,   \
                   dst_channels)->Args({width, height})->UseManualTime()->     \
                   Iterations(10);

#define RUN_BENCHMARK_LAB_COMPARISON_2FUNCTIONS(Function, src_channels,        \
                                                dst_channels)                  \
BENCHMARK_LAB_2FUNCTIONS_DECLARATION(Function)                                 \
RUN_LAB_2BENCHMARKS(Function, src_channels, dst_channels, 320, 240)            \
RUN_LAB_2BENCHMARKS(Function, src_channels, dst_channels, 640, 480)            \
RUN_LAB_2BENCHMARKS(Function, src_channels, dst_channels, 1280, 720)           \
RUN_LAB_2BENCHMARKS(Function, src_channels, dst_channels, 1920, 1080)

/***************************** No comparison ********************************/

#define BENCHMARK_1FUNCTION_DECLARATION(Function)                              \
BENCHMARK_PPL_CV_OCL(Function)

#define RUN_PPL_CV_FUNCTIONS(Function, type, src_channels, dst_channels)       \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, type, src_channels,    \
                   dst_channels)->Args({320, 240})->UseManualTime()->          \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, type, src_channels,    \
                   dst_channels)->Args({640, 480})->UseManualTime()->          \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, type, src_channels,    \
                   dst_channels)->Args({1280, 720})->UseManualTime()->         \
                   Iterations(10);                                             \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, type, src_channels,    \
                   dst_channels)->Args({1920, 1080})->UseManualTime()->        \
                   Iterations(10);

#define RUN_BENCHMARK_BATCH_1FUNCTIONS(Function, src_channel, dst_channel)     \
BENCHMARK_1FUNCTION_DECLARATION(Function)                                      \
RUN_PPL_CV_FUNCTIONS(Function, uchar, src_channel, dst_channel)                \
RUN_PPL_CV_FUNCTIONS(Function, float, src_channel, dst_channel)

/************************ NV12/NV21 with comparison *************************/

enum NVXXFunctions {
  kNV122BGR,
  kNV122RGB,
  kNV122BGRA,
  kNV122RGBA,
  kNV212BGR,
  kNV212RGB,
  kNV212BGRA,
  kNV212RGBA,
};

#define BENCHMARK_PPL_CV_OCL_NVXX(Function)                                    \
template <typename T, int src_channels, int dst_channels>                      \
void BM_CvtColor ## Function ## _ppl_ocl(benchmark::State &state) {            \
  ppl::common::ocl::createSharedFrameChain(false);                             \
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();  \
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();\
                                                                               \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else if (dst_channels == 1) {                                                \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channels));                   \
                                                                               \
  int src_bytes = src.rows * src.step;                                         \
  int dst_bytes = dst.rows * dst.step;                                         \
  cl_int error_code = 0;                                                       \
  cl_mem gpu_src = clCreateBuffer(context,                                     \
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,   \
                                  src_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_dst = clCreateBuffer(context,                                     \
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,   \
                                  dst_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes,    \
                                    src.data, 0, NULL, NULL);                  \
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);                               \
                                                                               \
  int iterations = 100;                                                        \
  struct timeval start, end;                                                   \
                                                                               \
  /* Warm up the GPU. */                                                       \
  for (int i = 0; i < iterations; i++) {                                       \
    ppl::cv::ocl::Function<T>(queue, height, width, src.step / sizeof(T),      \
                              gpu_src, dst.step / sizeof(T), gpu_dst);         \
  }                                                                            \
  clFinish(queue);                                                             \
                                                                               \
  for (auto _ : state) {                                                       \
    gettimeofday(&start, NULL);                                                \
    for (int i = 0; i < iterations; i++) {                                     \
      ppl::cv::ocl::Function<T>(queue, height, width, src.step / sizeof(T),    \
                                gpu_src, dst.step / sizeof(T), gpu_dst);       \
    }                                                                          \
    clFinish(queue);                                                           \
    gettimeofday(&end, NULL);                                                  \
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -                         \
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;        \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  clReleaseMemObject(gpu_src);                                                 \
  clReleaseMemObject(gpu_dst);                                                 \
}

#define BENCHMARK_OPENCV_CPU_NVXX(Function)                                    \
template<typename T, int src_channels, int dst_channels>                       \
void BM_CvtColor ## Function ## _opencv_cpu_ocl(benchmark::State &state) {     \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channels));                   \
                                                                               \
  NVXXFunctions ppl_function = k ## Function;                                  \
  cv::ColorConversionCodes cv_code;                                            \
  if (ppl_function == kNV122BGR) {                                             \
    cv_code = cv::COLOR_YUV2BGR_NV12;                                          \
  }                                                                            \
  else if (ppl_function == kNV122RGB) {                                        \
    cv_code = cv::COLOR_YUV2RGB_NV12;                                          \
  }                                                                            \
  else if (ppl_function == kNV122BGRA) {                                       \
    cv_code = cv::COLOR_YUV2BGRA_NV12;                                         \
  }                                                                            \
  else if (ppl_function == kNV122RGBA) {                                       \
    cv_code = cv::COLOR_YUV2RGBA_NV12;                                         \
  }                                                                            \
  else if (ppl_function == kNV212BGR) {                                        \
    cv_code = cv::COLOR_YUV2BGR_NV21;                                          \
  }                                                                            \
  else if (ppl_function == kNV212RGB) {                                        \
    cv_code = cv::COLOR_YUV2RGB_NV21;                                          \
  }                                                                            \
  else if (ppl_function == kNV212BGRA) {                                       \
    cv_code = cv::COLOR_YUV2BGRA_NV21;                                         \
  }                                                                            \
  else if (ppl_function == kNV212RGBA) {                                       \
    cv_code = cv::COLOR_YUV2RGBA_NV21;                                         \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  for (auto _ : state) {                                                       \
    cv::cvtColor(src, dst, cv_code);                                           \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
}

#define BENCHMARK_NVXX_2FUNCTIONS_DECLARATION(Function)                        \
BENCHMARK_PPL_CV_OCL_NVXX(Function)                                            \
BENCHMARK_OPENCV_CPU_NVXX(Function)

#define BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, width, \
                                  height)                                      \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cpu_ocl, uchar,          \
                   src_channels, dst_channels)->Args({width, height});         \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, uchar,                 \
                   src_channels, dst_channels)->Args({width, height})->        \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(Function, src_channels,       \
                                                 dst_channels)                 \
BENCHMARK_NVXX_2FUNCTIONS_DECLARATION(Function)                                \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, 320, 240)      \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, 640, 480)      \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, 1280, 720)     \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, 1920, 1080)

/********************** NV12/NV21 without comparison ************************/

#define BENCHMARK_NVXX_1FUNCTION_DECLARATION(Function)                         \
BENCHMARK_PPL_CV_OCL_NVXX(Function)

#define BENCHMARK_NVXX_1FUNCTION(Function, src_channels, dst_channels, width,  \
                                 height)                                       \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, uchar, src_channels,   \
                   dst_channels)->Args({width, height})->UseManualTime()->     \
                   Iterations(10);

#define RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(Function, src_channels,             \
                                           dst_channels)                       \
BENCHMARK_NVXX_1FUNCTION_DECLARATION(Function)                                 \
BENCHMARK_NVXX_1FUNCTION(Function, src_channels, dst_channels, 320, 240)       \
BENCHMARK_NVXX_1FUNCTION(Function, src_channels, dst_channels, 640, 480)       \
BENCHMARK_NVXX_1FUNCTION(Function, src_channels, dst_channels, 1280, 720)      \
BENCHMARK_NVXX_1FUNCTION(Function, src_channels, dst_channels, 1920, 1080)

/****************** Descrete NV12/NV21 without comparison *******************/

#define BENCHMARK_PPL_CV_OCL_DESCRETE_NVXX(Function)                           \
template <typename T, int src_channels, int dst_channels>                      \
void BM_CvtColorDescrete ## Function ## _ppl_ocl(benchmark::State &state) {    \
  ppl::common::ocl::createSharedFrameChain(false);                             \
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();  \
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();\
                                                                               \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else if (dst_channels == 1) {                                                \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channels));                   \
                                                                               \
  int src_bytes = src.rows * src.step;                                         \
  int dst_bytes = dst.rows * dst.step;                                         \
  int uv_bytes  = (height >> 1) * width * sizeof(T);                           \
  cl_int error_code = 0;                                                       \
  cl_mem gpu_src = clCreateBuffer(context,                                     \
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,   \
                                  src_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_dst = clCreateBuffer(context,                                     \
                                  CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,   \
                                  dst_bytes, NULL, &error_code);               \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes,    \
                                    src.data, 0, NULL, NULL);                  \
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);                               \
  cl_mem gpu_uv = clCreateBuffer(context,                                      \
                                 CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY,   \
                                 uv_bytes, NULL, &error_code);                 \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  if (src_channels == 1) {                                                     \
    T* input = (T*)clEnqueueMapBuffer(queue, gpu_uv, CL_TRUE, CL_MAP_WRITE, 0, \
                                      uv_bytes, 0, NULL, NULL, &error_code);   \
    CHECK_ERROR(error_code, clEnqueueMapBuffer);                               \
    cv::Mat uv((height >> 1), width, CV_MAKETYPE(cv::DataType<T>::depth, 1),   \
                src.data + height * src.step, src.step);                       \
    copyMatToArray(uv, input);                                                 \
    error_code = clEnqueueUnmapMemObject(queue, gpu_uv, input, 0, NULL, NULL); \
  }                                                                            \
                                                                               \
  int iterations = 100;                                                        \
  struct timeval start, end;                                                   \
                                                                               \
  /* Warm up the GPU. */                                                       \
  for (int i = 0; i < iterations; i++) {                                       \
    if (src_channels == 1) {                                                   \
      ppl::cv::ocl::Function<T>(queue, height, width, src.step / sizeof(T),    \
                                gpu_src, width, gpu_uv, dst.step / sizeof(T),  \
                                gpu_dst);                                      \
    }                                                                          \
    else {                                                                     \
      ppl::cv::ocl::Function<T>(queue, height, width, src.step / sizeof(T),    \
                                gpu_src, dst.step / sizeof(T), gpu_dst,        \
                                width, gpu_uv);                                \
    }                                                                          \
  }                                                                            \
  clFinish(queue);                                                             \
                                                                               \
  for (auto _ : state) {                                                       \
    gettimeofday(&start, NULL);                                                \
    for (int i = 0; i < iterations; i++) {                                     \
      if (src_channels == 1) {                                                 \
        ppl::cv::ocl::Function<T>(queue, height, width, src.step / sizeof(T),  \
                                  gpu_src, width, gpu_uv, dst.step / sizeof(T),\
                                  gpu_dst);                                    \
      }                                                                        \
      else {                                                                   \
        ppl::cv::ocl::Function<T>(queue, height, width, src.step / sizeof(T),  \
                                  gpu_src, dst.step / sizeof(T), gpu_dst,      \
                                  width, gpu_uv);                              \
      }                                                                        \
    }                                                                          \
    clFinish(queue);                                                           \
    gettimeofday(&end, NULL);                                                  \
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -                         \
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;        \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  clReleaseMemObject(gpu_src);                                                 \
  clReleaseMemObject(gpu_dst);                                                 \
  clReleaseMemObject(gpu_uv);                                                  \
}

#define BENCHMARK_DESCRETE_NVXX_1FUNCTION_DECLARATION(Function)                \
BENCHMARK_PPL_CV_OCL_DESCRETE_NVXX(Function)

#define BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels,\
                                           width, height)                      \
BENCHMARK_TEMPLATE(BM_CvtColorDescrete ## Function ## _ppl_ocl, uchar,         \
                   src_channels, dst_channels)->Args({width, height})->        \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(Function, src_channels,    \
                                                    dst_channels)              \
BENCHMARK_DESCRETE_NVXX_1FUNCTION_DECLARATION(Function)                        \
BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels, 320,   \
                                  240)                                         \
BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels, 640,   \
                                  480)                                         \
BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels, 1280,  \
                                  720)                                         \
BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels, 1920,  \
                                  1080)

/************************ I420 Comparison with OpenCV *************************/

enum I420Functions {
  kBGR2I420,
  kRGB2I420,
  kBGRA2I420,
  kRGBA2I420,
  kI4202BGR,
  kI4202RGB,
  kI4202BGRA,
  kI4202RGBA,
  kYUV2GRAY,
};

#define BENCHMARK_OPENCV_CPU_I420(Function)                                    \
template<typename T, int src_channels, int dst_channels>                       \
void BM_CvtColor ## Function ## _opencv_cpu_ocl(benchmark::State &state) {     \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channels));                   \
                                                                               \
  I420Functions ppl_function = k ## Function;                                  \
  cv::ColorConversionCodes cv_code;                                            \
  if (ppl_function == kBGR2I420) {                                             \
    cv_code = cv::COLOR_BGR2YUV_I420;                                          \
  }                                                                            \
  else if (ppl_function == kRGB2I420) {                                        \
    cv_code = cv::COLOR_RGB2YUV_I420;                                          \
  }                                                                            \
  else if (ppl_function == kBGRA2I420) {                                       \
    cv_code = cv::COLOR_BGRA2YUV_I420;                                         \
  }                                                                            \
  else if (ppl_function == kRGBA2I420) {                                       \
    cv_code = cv::COLOR_RGBA2YUV_I420;                                         \
  }                                                                            \
  else if (ppl_function == kI4202BGR) {                                        \
    cv_code = cv::COLOR_YUV2BGR_I420;                                          \
  }                                                                            \
  else if (ppl_function == kI4202RGB) {                                        \
    cv_code = cv::COLOR_YUV2RGB_I420;                                          \
  }                                                                            \
  else if (ppl_function == kI4202BGRA) {                                       \
    cv_code = cv::COLOR_YUV2BGRA_I420;                                         \
  }                                                                            \
  else if (ppl_function == kI4202RGBA) {                                       \
    cv_code = cv::COLOR_YUV2RGBA_I420;                                         \
  }                                                                            \
  else if (ppl_function == kYUV2GRAY) {                                        \
    cv_code = cv::COLOR_YUV2GRAY_420;                                          \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  for (auto _ : state) {                                                       \
    cv::cvtColor(src, dst, cv_code);                                           \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
}

#define BENCHMARK_I420_2FUNCTIONS_DECLARATION(Function)                        \
BENCHMARK_PPL_CV_OCL_NVXX(Function)                                            \
BENCHMARK_OPENCV_CPU_I420(Function)

#define RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(Function, src_channels,       \
                                                 dst_channels)                 \
BENCHMARK_I420_2FUNCTIONS_DECLARATION(Function)                                \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, 320, 240)      \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, 640, 480)      \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, 1280, 720)     \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channels, dst_channels, 1920, 1080)

/****************** Descrete I420 without comparison *******************/

#define BENCHMARK_PPL_CV_OCL_DESCRETE_I420(Function)                           \
template <typename T, int src_channels, int dst_channels>                      \
void BM_CvtColorDescrete ## Function ## _ppl_ocl(benchmark::State &state) {    \
  ppl::common::ocl::createSharedFrameChain(false);                             \
  cl_context context = ppl::common::ocl::getSharedFrameChain()->getContext();  \
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();\
                                                                               \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channels == 1) {                                                     \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else if (dst_channels == 1) {                                                \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channels));  \
                                                                               \
  int src_size = src_height * width * src_channels * sizeof(T);                \
  int dst_size = dst_height * width * dst_channels * sizeof(T);                \
  int uv_size  = (height >> 1) * (width >> 1) * sizeof(T);                     \
  T* input  = (T*)malloc(src_size);                                            \
  T* output = (T*)malloc(dst_size);                                            \
  copyMatToArray(src, input);                                                  \
  cl_int error_code = 0;                                                       \
  cl_mem gpu_input = clCreateBuffer(context,                                   \
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,  \
                                    src_size, NULL, &error_code);              \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_output = clCreateBuffer(context,                                  \
                                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,\
                                     dst_size, NULL, &error_code);             \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  T* map = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE,     \
                                  0, src_size, 0, NULL, NULL, &error_code);    \
  CHECK_ERROR(error_code, clEnqueueMapBuffer);                                 \
  copyMatToArray(src, map);                                                    \
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, map, 0, NULL, NULL);  \
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);                            \
  cl_mem gpu_u = clCreateBuffer(context,                                       \
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,     \
                                uv_size, NULL, &error_code);                   \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  cl_mem gpu_v = clCreateBuffer(context,                                       \
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,     \
                                uv_size, NULL, &error_code);                   \
  CHECK_ERROR(error_code, clCreateBuffer);                                     \
  if (src_channels == 1) {                                                     \
    map = (T*)clEnqueueMapBuffer(queue, gpu_u, CL_TRUE, CL_MAP_WRITE, 0,       \
                                 uv_size, 0, NULL, NULL, &error_code);         \
    CHECK_ERROR(error_code, clEnqueueMapBuffer);                               \
    memcpy(map, input + height * width * sizeof(T), uv_size);                  \
    error_code = clEnqueueUnmapMemObject(queue, gpu_u, map, 0, NULL, NULL);    \
    map = (T*)clEnqueueMapBuffer(queue, gpu_v, CL_TRUE, CL_MAP_WRITE, 0,       \
                                 uv_size, 0, NULL, NULL, &error_code);         \
    CHECK_ERROR(error_code, clEnqueueMapBuffer);                               \
    memcpy(map, input + height * 5 / 4 * width * sizeof(T), uv_size);          \
    error_code = clEnqueueUnmapMemObject(queue, gpu_v, map, 0, NULL, NULL);    \
  }                                                                            \
                                                                               \
  int iterations = 100;                                                        \
  struct timeval start, end;                                                   \
                                                                               \
  /* Warm up the GPU. */                                                       \
  for (int i = 0; i < iterations; i++) {                                       \
    if (src_channels == 1) {                                                   \
      ppl::cv::ocl::Function<T>(queue, height, width, width, gpu_input,        \
                                width / 2, gpu_u, width / 2, gpu_v,            \
                                width * dst_channels, gpu_output);             \
    }                                                                          \
    else {                                                                     \
      ppl::cv::ocl::Function<T>(queue, height, width, width * src_channels,    \
                                gpu_input, width, gpu_output, width / 2,       \
                                gpu_u, width / 2, gpu_v);                      \
    }                                                                          \
  }                                                                            \
  clFinish(queue);                                                             \
                                                                               \
  for (auto _ : state) {                                                       \
    gettimeofday(&start, NULL);                                                \
    for (int i = 0; i < iterations; i++) {                                     \
      if (src_channels == 1) {                                                 \
        ppl::cv::ocl::Function<T>(queue, height, width, width, gpu_input,      \
                                  width / 2, gpu_u, width / 2, gpu_v,          \
                                  width * dst_channels, gpu_output);           \
      }                                                                        \
      else {                                                                   \
        ppl::cv::ocl::Function<T>(queue, height, width, width * src_channels,  \
                                  gpu_input, width, gpu_output, width / 2,     \
                                  gpu_u, width / 2, gpu_v);                    \
      }                                                                        \
    }                                                                          \
    clFinish(queue);                                                           \
    gettimeofday(&end, NULL);                                                  \
    int time = ((end.tv_sec * 1000000 + end.tv_usec) -                         \
                (start.tv_sec * 1000000 + start.tv_usec)) / iterations;        \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  free(input);                                                                 \
  free(output);                                                                \
  clReleaseMemObject(gpu_input);                                               \
  clReleaseMemObject(gpu_output);                                              \
  clReleaseMemObject(gpu_u);                                                   \
  clReleaseMemObject(gpu_v);                                                   \
}


#define BENCHMARK_DESCRETE_I420_1FUNCTION_DECLARATION(Function)                \
BENCHMARK_PPL_CV_OCL_DESCRETE_I420(Function)

#define RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(Function, src_channels,    \
                                                    dst_channels)              \
BENCHMARK_DESCRETE_I420_1FUNCTION_DECLARATION(Function)                        \
BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels, 320,   \
                                  240)                                         \
BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels, 640,   \
                                  480)                                         \
BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels, 1280,  \
                                  720)                                         \
BENCHMARK_DESCRETE_NVXX_1FUNCTION(Function, src_channels, dst_channels, 1920,  \
                                  1080)

/************************ YUV422 with comparison *************************/

enum YUV422Functions {
  kUYVY2BGR,
  kUYVY2GRAY,
  kYUYV2BGR,
  kYUYV2GRAY,
};

#define BENCHMARK_OPENCV_CPU_YUV422(Function)                                  \
template<typename T, int src_channels, int dst_channels>                       \
void BM_CvtColor ## Function ## _opencv_cpu_ocl(benchmark::State &state) {     \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src = createSourceImage(height, width,                               \
                  CV_MAKETYPE(cv::DataType<T>::depth, src_channels), 16, 235); \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth,               \
              dst_channels));                                                  \
                                                                               \
  YUV422Functions ppl_function = k ## Function;                                \
  cv::ColorConversionCodes cv_code;                                            \
  if (ppl_function == kUYVY2BGR) {                                             \
    cv_code = cv::COLOR_YUV2BGR_UYVY;                                          \
  }                                                                            \
  else if (ppl_function == kUYVY2GRAY) {                                       \
    cv_code = cv::COLOR_YUV2GRAY_UYVY;                                         \
  }                                                                            \
  else if (ppl_function == kYUYV2BGR) {                                        \
    cv_code = cv::COLOR_YUV2BGR_YUYV;                                          \
  }                                                                            \
  else if (ppl_function == kYUYV2GRAY) {                                       \
    cv_code = cv::COLOR_YUV2GRAY_YUYV;                                         \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
                                                                               \
  for (auto _ : state) {                                                       \
    cv::cvtColor(src, dst, cv_code);                                           \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
}

#define BENCHMARK_YUV422_2FUNCTIONS_DECLARATION(Function)                      \
BENCHMARK_PPL_CV_OCL(Function)                                                 \
BENCHMARK_OPENCV_CPU_YUV422(Function)

#define BENCHMARK_YUV422_2FUNCTIONS(Function, src_channels, dst_channels,      \
                                    width, height)                             \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cpu_ocl, uchar,          \
                   src_channels, dst_channels)->Args({width, height});         \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_ocl, uchar,                 \
                   src_channels, dst_channels)->Args({width, height})->        \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(Function, src_channels,     \
                                                   dst_channels)               \
BENCHMARK_YUV422_2FUNCTIONS_DECLARATION(Function)                              \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channels, dst_channels, 320, 240)    \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channels, dst_channels, 640, 480)    \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channels, dst_channels, 1280, 720)   \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channels, dst_channels, 1920, 1080)

// BGR(RBB) <-> BGRA(RGBA)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGR2BGRA, c3, c4)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGB2RGBA, c3, c4)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGRA2BGR, c4, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGBA2RGB, c4, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGR2RGBA, c3, c4)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGB2BGRA, c3, c4)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGRA2RGB, c4, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGBA2BGR, c4, c3)

// BGR <-> RGB
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGR2RGB, c4, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGB2BGR, c4, c3)

// BGRA <-> RGBA
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGRA2RGBA, c4, c4)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGBA2BGRA, c4, c4)

// BGR/RGB/BGRA/RGBA <-> Gray
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGR2GRAY, c3, c1)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGB2GRAY, c3, c1)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGRA2GRAY, c4, c1)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGBA2GRAY, c4, c1)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(GRAY2BGR, c1, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(GRAY2RGB, c1, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(GRAY2BGRA, c1, c4)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(GRAY2RGBA, c1, c4)

// BGR/RGB/BGRA/RGBA <-> YCrCb
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGR2YCrCb, c3, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGB2YCrCb, c3, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(BGRA2YCrCb, c4, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGBA2YCrCb, c4, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(YCrCb2BGR, c3, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(YCrCb2RGB, c3, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(YCrCb2BGRA, c3, c4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(YCrCb2RGBA, c3, c4)

// BGR/RGB/BGRA/RGBA <-> HSV
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(BGR2HSV, c3, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(RGB2HSV, c3, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(BGRA2HSV, c4, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGBA2HSV, c4, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(HSV2BGR, c3, c3)
RUN_BENCHMARK_COMPARISON_2FUNCTIONS(HSV2RGB, c3, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(HSV2BGRA, c3, c4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(HSV2RGBA, c3, c4)

// BGR/RGB/BGRA/RGBA <-> LAB
RUN_BENCHMARK_LAB_COMPARISON_2FUNCTIONS(BGR2LAB, c3, c3)
RUN_BENCHMARK_LAB_COMPARISON_2FUNCTIONS(RGB2LAB, c3, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(BGRA2LAB, c4, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGBA2LAB, c4, c3)
RUN_BENCHMARK_LAB_COMPARISON_2FUNCTIONS(LAB2BGR, c3, c3)
RUN_BENCHMARK_LAB_COMPARISON_2FUNCTIONS(LAB2RGB, c3, c3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(LAB2BGRA, 3, 4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(LAB2RGBA, 3, 4)

// BGR/RGB/BGRA/RGBA <-> NV12
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(BGR2NV12, 3, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(RGB2NV12, 3, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(BGRA2NV12, 4, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(RGBA2NV12, 4, 1)
RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV122BGR, 1, 3)
RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV122RGB, 1, 3)
RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV122BGRA, 1, 4)
RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV122RGBA, 1, 4)

RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(BGR2NV12, 3, 1)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(RGB2NV12, 3, 1)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(BGRA2NV12, 4, 1)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(RGBA2NV12, 4, 1)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(NV122BGR, 1, 3)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(NV122RGB, 1, 3)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(NV122BGRA, 1, 4)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(NV122RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> NV21
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(BGR2NV21, 3, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(RGB2NV21, 3, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(BGRA2NV21, 4, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(RGBA2NV21, 4, 1)
RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV212BGR, 1, 3)
RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV212RGB, 1, 3)
RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV212BGRA, 1, 4)
RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV212RGBA, 1, 4)

RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(BGR2NV21, 3, 1)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(RGB2NV21, 3, 1)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(BGRA2NV21, 4, 1)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(RGBA2NV21, 4, 1)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(NV212BGR, 1, 3)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(NV212RGB, 1, 3)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(NV212BGRA, 1, 4)
RUN_BENCHMARK_DESCRETE_NVXX_BATCH_1FUNCTION(NV212RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> I420
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(BGR2I420, 3, 1)
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(RGB2I420, 3, 1)
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(BGRA2I420, 4, 1)
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(RGBA2I420, 4, 1)
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(I4202BGR, 1, 3)
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(I4202RGB, 1, 3)
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(I4202BGRA, 1, 4)
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(I4202RGBA, 1, 4)

RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(BGR2I420, 3, 1)
RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(RGB2I420, 3, 1)
RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(BGRA2I420, 4, 1)
RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(RGBA2I420, 4, 1)
RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(I4202BGR, 1, 3)
RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(I4202RGB, 1, 3)
RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(I4202BGRA, 1, 4)
RUN_BENCHMARK_DESCRETE_I420_BATCH_1FUNCTION(I4202RGBA, 1, 4)

// YUV2GRAY
RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(YUV2GRAY, 1, 1)

// BGR/GRAY <-> UYVY
RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(UYVY2BGR, 2, 3)
RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(UYVY2GRAY, 2, 1)

// BGR/GRAY <-> YUYV
RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(YUYV2BGR, 2, 3)
RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(YUYV2GRAY, 2, 1)
