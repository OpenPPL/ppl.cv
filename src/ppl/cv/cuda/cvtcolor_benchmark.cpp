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

#include "ppl/cv/cuda/cvtcolor.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "benchmark/benchmark.h"

#include "ppl/cv/debug.h"
#include "utility/infrastructure.hpp"

using namespace ppl::cv::debug;

#define BENCHMARK_PPL_CV_CUDA(Function)                                        \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _ppl_cuda(benchmark::State &state) {           \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src;                                                                 \
  src = createSourceImage(height, width,                                       \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));\
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int iterations = 3000;                                                       \
  float elapsed_time;                                                          \
  cudaEvent_t start, stop;                                                     \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&stop);                                                      \
                                                                               \
  for (int i = 0; i < iterations; i++) {                                       \
    ppl::cv::cuda::Function<T>(0, gpu_src.rows, gpu_src.cols,                  \
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),  \
        (T*)gpu_dst.data);                                                     \
  }                                                                            \
  cudaDeviceSynchronize();                                                     \
                                                                               \
  for (auto _ : state) {                                                       \
    cudaEventRecord(start, 0);                                                 \
    for (int i = 0; i < iterations; i++) {                                     \
      ppl::cv::cuda::Function<T>(0, gpu_src.rows, gpu_src.cols,                \
          gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),\
          (T*)gpu_dst.data);                                                   \
    }                                                                          \
    cudaEventRecord(stop, 0);                                                  \
    cudaEventSynchronize(stop);                                                \
    cudaEventElapsedTime(&elapsed_time, start, stop);                          \
    int time = elapsed_time * 1000 / iterations;                               \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(stop);                                                      \
}

#define BENCHMARK_OPENCV_CUDA(Function)                                        \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _opencv_cuda(benchmark::State &state) {        \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src;                                                                 \
  src = createSourceImage(height, width,                                       \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));\
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int iterations = 3000;                                                       \
  float elapsed_time;                                                          \
  cudaEvent_t start, stop;                                                     \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&stop);                                                      \
                                                                               \
  for (int i = 0; i < iterations; i++) {                                       \
    cv::cuda::cvtColor(gpu_src, gpu_dst, cv::COLOR_ ## Function);              \
  }                                                                            \
  cudaDeviceSynchronize();                                                     \
                                                                               \
  for (auto _ : state) {                                                       \
    cudaEventRecord(start, 0);                                                 \
    for (int i = 0; i < iterations; i++) {                                     \
      cv::cuda::cvtColor(gpu_src, gpu_dst, cv::COLOR_ ## Function);            \
    }                                                                          \
    cudaEventRecord(stop, 0);                                                  \
    cudaEventSynchronize(stop);                                                \
    cudaEventElapsedTime(&elapsed_time, start, stop);                          \
    int time = elapsed_time * 1000 / iterations;                               \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(stop);                                                      \
}

#define BENCHMARK_OPENCV_X86(Function)                                         \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## opencv_x86_cuda(benchmark::State &state) {     \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src;                                                                 \
  src = createSourceImage(height, width,                                       \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));\
                                                                               \
  for (auto _ : state) {                                                       \
    cv::cvtColor(src, dst, cv::COLOR_ ## Function);                            \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
}

/************************** Comparison with OpenCV **************************/

#define BENCHMARK_2FUNCTIONS_DECLARATION(Function)                             \
BENCHMARK_OPENCV_CUDA(Function)                                                \
BENCHMARK_PPL_CV_CUDA(Function)

#define RUN_OPENCV_TYPE_FUNCTIONS(Function, type, src_channel, dst_channel)    \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, type, src_channel, \
                   dst_channel)->Args({320, 240})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, type, src_channel, \
                   dst_channel)->Args({321, 240})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, type, src_channel, \
                   dst_channel)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, type, src_channel, \
                   dst_channel)->Args({648, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, type, src_channel, \
                   dst_channel)->Args({1280, 720})->                           \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, type, src_channel, \
                   dst_channel)->Args({1293, 720})->                           \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, type, src_channel, \
                   dst_channel)->Args({1920, 1080})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, type, src_channel, \
                   dst_channel)->Args({1976, 1080})->                          \
                   UseManualTime()->Iterations(10);

#define RUN_PPL_CV_TYPE_FUNCTIONS(Function, type, src_channel, dst_channel)    \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({320, 240})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({321, 240})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({648, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({1280, 720})->                           \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({1293, 720})->                           \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({1920, 1080})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({1976, 1080})->                          \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_BATCH_2FUNCTIONS(Function, src_channel, dst_channel)     \
BENCHMARK_2FUNCTIONS_DECLARATION(Function)                                     \
RUN_OPENCV_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)           \
RUN_OPENCV_TYPE_FUNCTIONS(Function, float, src_channel, dst_channel)           \
RUN_PPL_CV_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)           \
RUN_PPL_CV_TYPE_FUNCTIONS(Function, float, src_channel, dst_channel)

#define BENCHMARK_3FUNCTIONS_DECLARATION(Function)                             \
BENCHMARK_OPENCV_X86(Function)                                                 \
BENCHMARK_OPENCV_CUDA(Function)                                                \
BENCHMARK_PPL_CV_CUDA(Function)

#define BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, width, height)\
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## opencv_x86_cuda, uchar,          \
                   src_channel, dst_channel)->Args({width, height});           \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, uchar,             \
                   src_channel, dst_channel)->Args({width, height})->          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, uchar,                \
                   src_channel, dst_channel)->Args({width, height})->          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## opencv_x86_cuda, float,          \
                   src_channel, dst_channel)->Args({width, height});           \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_cuda, float,             \
                   src_channel, dst_channel)->Args({width, height})->          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, float,                \
                   src_channel, dst_channel)->Args({width, height})->          \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_COMPARISON_3FUNCTIONS(Function, src_channel, dst_channel)\
BENCHMARK_3FUNCTIONS_DECLARATION(Function)                                     \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 320, 240)             \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 321, 240)             \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 640, 480)             \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 648, 480)             \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 1280, 720)            \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 1293, 720)            \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 1920, 1080)           \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 1976, 1080)

/************************ LAB Comparison with OpenCV *************************/

enum LabFunctions {
  kBGR2LAB,
  kRGB2LAB,
  kLAB2BGR,
  kLAB2RGB,
};

#define BENCHMARK_OPENCV_CUDA_LAB(Function)                                    \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _opencv_cuda(benchmark::State &state) {        \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src;                                                                 \
  src = createSourceImage(height, width,                                       \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));\
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
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
  int iterations = 3000;                                                       \
  float elapsed_time;                                                          \
  cudaEvent_t start, stop;                                                     \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&stop);                                                      \
                                                                               \
  for (int i = 0; i < iterations; i++) {                                       \
    cv::cuda::cvtColor(gpu_src, gpu_dst, cv_code);                             \
  }                                                                            \
  cudaDeviceSynchronize();                                                     \
                                                                               \
  for (auto _ : state) {                                                       \
    cudaEventRecord(start, 0);                                                 \
    for (int i = 0; i < iterations; i++) {                                     \
      cv::cuda::cvtColor(gpu_src, gpu_dst, cv_code);                           \
    }                                                                          \
    cudaEventRecord(stop, 0);                                                  \
    cudaEventSynchronize(stop);                                                \
    cudaEventElapsedTime(&elapsed_time, start, stop);                          \
    int time = elapsed_time * 1000 / iterations;                               \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(stop);                                                      \
}

#define BENCHMARK_OPENCV_X86_LAB(Function)                                     \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## opencv_x86_cuda(benchmark::State &state) {     \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src;                                                                 \
  src = createSourceImage(height, width,                                       \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));\
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
BENCHMARK_OPENCV_CUDA_LAB(Function)                                            \
BENCHMARK_PPL_CV_CUDA(Function)

#define RUN_BENCHMARK_LAB_BATCH_2FUNCTIONS(Function, src_channel, dst_channel) \
BENCHMARK_LAB_2FUNCTIONS_DECLARATION(Function)                                 \
RUN_OPENCV_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)           \
RUN_OPENCV_TYPE_FUNCTIONS(Function, float, src_channel, dst_channel)           \
RUN_PPL_CV_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)           \
RUN_PPL_CV_TYPE_FUNCTIONS(Function, float, src_channel, dst_channel)

#define BENCHMARK_LAB_3FUNCTIONS_DECLARATION(Function)                         \
BENCHMARK_OPENCV_X86_LAB(Function)                                             \
BENCHMARK_OPENCV_CUDA_LAB(Function)                                            \
BENCHMARK_PPL_CV_CUDA(Function)

#define RUN_BENCHMARK_LAB_COMPARISON_3FUNCTIONS(Function, src_channel,         \
                                                dst_channel)                   \
BENCHMARK_LAB_3FUNCTIONS_DECLARATION(Function)                                 \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 320, 240)             \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 321, 240)             \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 640, 480)             \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 648, 480)             \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 1280, 720)            \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 1293, 720)            \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 1920, 1080)           \
BENCHMARK_3FUNCTIONS(Function, src_channel, dst_channel, 1976, 1080)

/***************************** No comparison ********************************/

#define BENCHMARK_1FUNCTION_DECLARATION(Function)                              \
BENCHMARK_PPL_CV_CUDA(Function)

#define RUN_BENCHMARK_BATCH_1FUNCTIONS(Function, src_channel, dst_channel)     \
BENCHMARK_1FUNCTION_DECLARATION(Function)                                      \
RUN_PPL_CV_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)           \
RUN_PPL_CV_TYPE_FUNCTIONS(Function, float, src_channel, dst_channel)

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

#define BENCHMARK_PPL_CV_CUDA_NVXX(Function)                                   \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _ppl_cuda(benchmark::State &state) {           \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else if (dst_channel == 1) {                                                 \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channel));                    \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int iterations = 3000;                                                       \
  float elapsed_time;                                                          \
  cudaEvent_t start, stop;                                                     \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&stop);                                                      \
                                                                               \
  for (int i = 0; i < iterations; i++) {                                       \
    ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),     \
        (const T*)gpu_src.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);   \
  }                                                                            \
  cudaDeviceSynchronize();                                                     \
                                                                               \
  for (auto _ : state) {                                                       \
    cudaEventRecord(start, 0);                                                 \
    for (int i = 0; i < iterations; i++) {                                     \
      ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),   \
          (const T*)gpu_src.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data); \
    }                                                                          \
    cudaEventRecord(stop, 0);                                                  \
    cudaEventSynchronize(stop);                                                \
    cudaEventElapsedTime(&elapsed_time, start, stop);                          \
    int time = elapsed_time * 1000 / iterations;                               \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(stop);                                                      \
}

#define BENCHMARK_OPENCV_X86_NVXX(Function)                                    \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _opencv_x86_cuda(benchmark::State &state) {    \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channel));                    \
                                                                               \
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
BENCHMARK_PPL_CV_CUDA_NVXX(Function)                                           \
BENCHMARK_OPENCV_X86_NVXX(Function)

#define BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, width,   \
                                  height)                                      \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, uchar,         \
                   src_channel, dst_channel)->Args({width, height});           \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, uchar,                \
                   src_channel, dst_channel)->Args({width, height})->          \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(Function, src_channel,        \
                                                 dst_channel)                  \
BENCHMARK_NVXX_2FUNCTIONS_DECLARATION(Function)                                \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, 320, 240)        \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, 322, 240)        \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, 640, 480)        \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, 644, 480)        \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, 1280, 720)       \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, 1286, 720)       \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, 1920, 1080)      \
BENCHMARK_NVXX_2FUNCTIONS(Function, src_channel, dst_channel, 1928, 1080)

#define RUN_OPENCV_NVXX_TYPE_FUNCTIONS(Function, type, src_channel,            \
                                       dst_channel)                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, type,          \
                   src_channel, dst_channel)->Args({320, 240});                \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, type,          \
                   src_channel, dst_channel)->Args({322, 240});                \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, type,          \
                   src_channel, dst_channel)->Args({640, 480});                \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, type,          \
                   src_channel, dst_channel)->Args({644, 480});                \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, type,          \
                   src_channel, dst_channel)->Args({1280, 720});               \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, type,          \
                   src_channel, dst_channel)->Args({1286, 720});               \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, type,          \
                   src_channel, dst_channel)->Args({1920, 1080});              \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, type,          \
                   src_channel, dst_channel)->Args({1928, 1080});

#define RUN_PPL_CV_NVXX_TYPE_FUNCTIONS(Function, type, src_channel,            \
                                       dst_channel)                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({320, 240})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({322, 240})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({640, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({644, 480})->                            \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({1280, 720})->                           \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({1286, 720})->                           \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({1920, 1080})->                          \
                   UseManualTime()->Iterations(10);                            \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, type, src_channel,    \
                   dst_channel)->Args({1928, 1080})->                          \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(Function, src_channel,             \
                                            dst_channel)                       \
BENCHMARK_NVXX_2FUNCTIONS_DECLARATION(Function)                                \
RUN_OPENCV_NVXX_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)      \
RUN_PPL_CV_NVXX_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)


/********************** NV12/NV21 without comparison ************************/

#define BENCHMARK_NVXX_1FUNCTION_DECLARATION(Function)                         \
BENCHMARK_PPL_CV_CUDA_NVXX(Function)

#define BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, width,    \
                                 height)                                       \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, uchar, src_channel,   \
                   dst_channel)->Args({width, height})->                       \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(Function, src_channel, dst_channel) \
BENCHMARK_NVXX_1FUNCTION_DECLARATION(Function)                                 \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 320, 240)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 322, 240)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 640, 480)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 644, 480)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1280, 720)        \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1286, 720)        \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1920, 1080)       \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1928, 1080)

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

#define BENCHMARK_OPENCV_X86_I420(Function)                                    \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _opencv_x86_cuda(benchmark::State &state) {    \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height;                                                     \
  int dst_height = height;                                                     \
  cv::Mat src;                                                                 \
  if (src_channel == 1) {                                                      \
    src_height = height + (height >> 1);                                       \
  }                                                                            \
  else {                                                                       \
    dst_height = height + (height >> 1);                                       \
  }                                                                            \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channel));                    \
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
BENCHMARK_PPL_CV_CUDA_NVXX(Function)                                           \
BENCHMARK_OPENCV_X86_I420(Function)                                            \

#define BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, width,   \
                                  height)                                      \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, uchar,         \
                   src_channel, dst_channel)->Args({width, height});           \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, uchar,                \
                   src_channel, dst_channel)->Args({width, height})->          \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(Function, src_channel,        \
                                                 dst_channel)                  \
BENCHMARK_I420_2FUNCTIONS_DECLARATION(Function)                                \
BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, 320, 240)        \
BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, 322, 240)        \
BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, 640, 480)        \
BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, 644, 480)        \
BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, 1280, 720)       \
BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, 1286, 720)       \
BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, 1920, 1080)      \
BENCHMARK_I420_2FUNCTIONS(Function, src_channel, dst_channel, 1928, 1080)

#define RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(Function, src_channel, dst_channel)\
BENCHMARK_I420_2FUNCTIONS_DECLARATION(Function)                                \
RUN_OPENCV_NVXX_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)      \
RUN_PPL_CV_NVXX_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)

/************************* I420 without comparison **************************/

#define BENCHMARK_I420_1FUNCTION_DECLARATION(Function)                         \
BENCHMARK_PPL_CV_CUDA(Function)

#define BENCHMARK_I420_1FUNCTION(Function, src_channel, dst_channel, width,    \
                                 height)                                       \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, uchar, src_channel,   \
                   dst_channel)->Args({width, height})->                       \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_I420_1FUNCTION(Function, src_channel, dst_channel)       \
BENCHMARK_I420_1FUNCTION_DECLARATION(Function)                                 \
BENCHMARK_I420_1FUNCTION(Function, src_channel, dst_channel, 320, 240)         \
BENCHMARK_I420_1FUNCTION(Function, src_channel, dst_channel, 640, 480)         \
BENCHMARK_I420_1FUNCTION(Function, src_channel, dst_channel, 1280, 720)        \
BENCHMARK_I420_1FUNCTION(Function, src_channel, dst_channel, 1920, 1080)

/************************ YUV422 with comparison *************************/

enum YUV422Functions {
  kUYVY2BGR,
  kUYVY2GRAY,
  kYUYV2BGR,
  kYUYV2GRAY,
};

#define BENCHMARK_OPENCV_X86_YUV422(Function)                                  \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _opencv_x86_cuda(benchmark::State &state) {    \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  cv::Mat src = createSourceImage(height, width,                               \
                  CV_MAKETYPE(cv::DataType<T>::depth, src_channel), 16, 235);  \
  cv::Mat dst(height, width, CV_MAKETYPE(cv::DataType<T>::depth, dst_channel));\
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
BENCHMARK_PPL_CV_CUDA(Function)                                                \
BENCHMARK_OPENCV_X86_YUV422(Function)

#define RUN_BENCHMARK_YUV422_BATCH_2FUNCTIONS(Function, src_channel,           \
                                              dst_channel)                     \
BENCHMARK_YUV422_2FUNCTIONS_DECLARATION(Function)                              \
RUN_OPENCV_NVXX_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)      \
RUN_PPL_CV_NVXX_TYPE_FUNCTIONS(Function, uchar, src_channel, dst_channel)

#define BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, width, \
                                    height)                                    \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _opencv_x86_cuda, uchar,         \
                   src_channel, dst_channel)->Args({width, height});           \
BENCHMARK_TEMPLATE(BM_CvtColor ## Function ## _ppl_cuda, uchar,                \
                   src_channel, dst_channel)->Args({width, height})->          \
                   UseManualTime()->Iterations(10);

#define RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(Function, src_channel,      \
                                                   dst_channel)                \
BENCHMARK_YUV422_2FUNCTIONS_DECLARATION(Function)                              \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, 320, 240)      \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, 322, 240)      \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, 640, 480)      \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, 644, 480)      \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, 1280, 720)     \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, 1286, 720)     \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, 1920, 1080)    \
BENCHMARK_YUV422_2FUNCTIONS(Function, src_channel, dst_channel, 1928, 1080)

/******************* NV12/21 <-> I420 without comparison *******************/

#define BENCHMARK_PPL_CV_CUDA_NVXX_TO_I420(Function)                           \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _ppl_cuda(benchmark::State &state) {           \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height + (height >> 1);                                     \
  int dst_height = height + (height >> 1);                                     \
  cv::Mat src;                                                                 \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channel));                    \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int iterations = 3000;                                                       \
  float elapsed_time;                                                          \
  cudaEvent_t start, stop;                                                     \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&stop);                                                      \
                                                                               \
  for (int i = 0; i < iterations; i++) {                                       \
    ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),     \
        (const T*)gpu_src.data, gpu_src.step / sizeof(T),                      \
        (const T*)gpu_src.data + height * gpu_src.step,                        \
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data,                            \
        gpu_dst.step / sizeof(T) / 2, (T*)gpu_dst.data + height * gpu_dst.step,\
        gpu_dst.step / sizeof(T) / 2,                                          \
        (T*)gpu_dst.data + height * gpu_dst.step * 5 / 4);                     \
  }                                                                            \
  cudaDeviceSynchronize();                                                     \
                                                                               \
  for (auto _ : state) {                                                       \
    cudaEventRecord(start, 0);                                                 \
    for (int i = 0; i < iterations; i++) {                                     \
      ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),   \
          (const T*)gpu_src.data, gpu_src.step / sizeof(T),                    \
          (const T*)gpu_src.data + height * gpu_src.step,                      \
          gpu_dst.step / sizeof(T), (T*)gpu_dst.data,                          \
          gpu_dst.step / sizeof(T) / 2,                                        \
          (T*)gpu_dst.data + height * gpu_dst.step,                            \
          gpu_dst.step / sizeof(T) / 2,                                        \
          (T*)gpu_dst.data + height * gpu_dst.step * 5 / 4);                   \
    }                                                                          \
    cudaEventRecord(stop, 0);                                                  \
    cudaEventSynchronize(stop);                                                \
    cudaEventElapsedTime(&elapsed_time, start, stop);                          \
    int time = elapsed_time * 1000 / iterations;                               \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(stop);                                                      \
}

#define RUN_BENCHMARK_NVXX_TO_I420_BATCH_1FUNCTION(Function, src_channel,      \
                                                   dst_channel)                \
BENCHMARK_PPL_CV_CUDA_NVXX_TO_I420(Function)                                   \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 320, 240)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 322, 240)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 640, 480)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 644, 480)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1280, 720)        \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1286, 720)        \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1920, 1080)       \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1928, 1080)

#define BENCHMARK_PPL_CV_CUDA_I420_TO_NVXX(Function)                           \
template<typename T, int src_channel, int dst_channel>                         \
void BM_CvtColor ## Function ## _ppl_cuda(benchmark::State &state) {           \
  int width  = state.range(0);                                                 \
  int height = state.range(1);                                                 \
  int src_height = height + (height >> 1);                                     \
  int dst_height = height + (height >> 1);                                     \
  cv::Mat src;                                                                 \
  src = createSourceImage(src_height, width,                                   \
                          CV_MAKETYPE(cv::DataType<T>::depth, src_channel));   \
  cv::Mat dst(dst_height, width, CV_MAKETYPE(cv::DataType<T>::depth,           \
                                             dst_channel));                    \
  cv::cuda::GpuMat gpu_src(src);                                               \
  cv::cuda::GpuMat gpu_dst(dst);                                               \
                                                                               \
  int iterations = 3000;                                                       \
  float elapsed_time;                                                          \
  cudaEvent_t start, stop;                                                     \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&stop);                                                      \
                                                                               \
  for (int i = 0; i < iterations; i++) {                                       \
    ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),     \
          (const T*)gpu_src.data, gpu_src.step / sizeof(T) / 2,                \
          (const T*)gpu_src.data + height * gpu_src.step,                      \
          gpu_src.step / sizeof(T) / 2,                                        \
          (T*)gpu_src.data + height * gpu_src.step * 5 / 4,                    \
          gpu_dst.step / sizeof(T), (T*)gpu_dst.data,                          \
          gpu_dst.step / sizeof(T),                                            \
          (T*)gpu_dst.data + height * gpu_dst.step);                           \
  }                                                                            \
  cudaDeviceSynchronize();                                                     \
                                                                               \
  for (auto _ : state) {                                                       \
    cudaEventRecord(start, 0);                                                 \
    for (int i = 0; i < iterations; i++) {                                     \
      ppl::cv::cuda::Function<T>(0, height, width, gpu_src.step / sizeof(T),   \
          (const T*)gpu_src.data, gpu_src.step / sizeof(T) / 2,                \
          (const T*)gpu_src.data + height * gpu_src.step,                      \
          gpu_src.step / sizeof(T) / 2,                                        \
          (T*)gpu_src.data + height * gpu_src.step * 5 / 4,                    \
          gpu_dst.step / sizeof(T), (T*)gpu_dst.data,                          \
          gpu_dst.step / sizeof(T),                                            \
          (T*)gpu_dst.data + height * gpu_dst.step);                           \
    }                                                                          \
    cudaEventRecord(stop, 0);                                                  \
    cudaEventSynchronize(stop);                                                \
    cudaEventElapsedTime(&elapsed_time, start, stop);                          \
    int time = elapsed_time * 1000 / iterations;                               \
    state.SetIterationTime(time * 1e-6);                                       \
  }                                                                            \
  state.SetItemsProcessed(state.iterations() * 1);                             \
                                                                               \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(stop);                                                      \
}

#define RUN_BENCHMARK_I420_TO_NVXX_BATCH_1FUNCTION(Function, src_channel,      \
                                                   dst_channel)                \
BENCHMARK_PPL_CV_CUDA_I420_TO_NVXX(Function)                                   \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 320, 240)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 322, 240)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 640, 480)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 644, 480)         \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1280, 720)        \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1286, 720)        \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1920, 1080)       \
BENCHMARK_NVXX_1FUNCTION(Function, src_channel, dst_channel, 1928, 1080)

// BGR(RBB) <-> BGRA(RGBA)
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGR2BGRA, 3, 4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGB2RGBA, 3, 4)
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGRA2BGR, 4, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGBA2RGB, 4, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGR2RGBA, 3, 4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGB2BGRA, 3, 4)
RUN_BENCHMARK_BATCH_2FUNCTIONS(RGBA2BGR, 4, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(BGRA2RGB, 4, 3)

// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGR2BGRA, 3, 4)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGRA2BGR, 4, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGR2RGBA, 3, 4)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(RGBA2BGR, 4, 3)

// BGR <-> RGB
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGR2RGB, 3, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGB2BGR, 3, 3)

// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGR2RGB, 3, 3)

// BGRA <-> RGBA
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGRA2RGBA, 4, 4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGBA2BGRA, 4, 4)

// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGRA2RGBA, 4, 4)

// BGR/RGB/BGRA/RGBA <-> Gray
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGR2GRAY, 3, 1)
RUN_BENCHMARK_BATCH_2FUNCTIONS(RGB2GRAY, 3, 1)
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGRA2GRAY, 4, 1)
RUN_BENCHMARK_BATCH_2FUNCTIONS(RGBA2GRAY, 4, 1)
RUN_BENCHMARK_BATCH_2FUNCTIONS(GRAY2BGR, 1, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(GRAY2RGB, 1, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(GRAY2BGRA, 1, 4)
RUN_BENCHMARK_BATCH_2FUNCTIONS(GRAY2RGBA, 1, 4)

// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGR2GRAY, 3, 1)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(RGB2GRAY, 3, 1)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGRA2GRAY, 4, 1)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(RGBA2GRAY, 4, 1)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(GRAY2BGR, 1, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(GRAY2RGB, 1, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(GRAY2BGRA, 1, 4)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(GRAY2RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> YCrCb
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGR2YCrCb, 3, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(RGB2YCrCb, 3, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(BGRA2YCrCb, 4, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGBA2YCrCb, 4, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(YCrCb2BGR, 3, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(YCrCb2RGB, 3, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(YCrCb2BGRA, 3, 4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(YCrCb2RGBA, 3, 4)

// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGR2YCrCb, 3, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(RGB2YCrCb, 3, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(YCrCb2BGR, 3, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(YCrCb2RGB, 3, 3)

// BGR/RGB/BGRA/RGBA <-> HSV
RUN_BENCHMARK_BATCH_2FUNCTIONS(BGR2HSV, 3, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(RGB2HSV, 3, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(BGRA2HSV, 4, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGBA2HSV, 4, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(HSV2BGR, 3, 3)
RUN_BENCHMARK_BATCH_2FUNCTIONS(HSV2RGB, 3, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(HSV2BGRA, 3, 4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(HSV2RGBA, 3, 4)

// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(BGR2HSV, 3, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(RGB2HSV, 3, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(HSV2BGR, 3, 3)
// RUN_BENCHMARK_COMPARISON_3FUNCTIONS(HSV2RGB, 3, 3)

// BGR/RGB/BGRA/RGBA <-> LAB
RUN_BENCHMARK_LAB_BATCH_2FUNCTIONS(BGR2LAB, 3, 3)
RUN_BENCHMARK_LAB_BATCH_2FUNCTIONS(RGB2LAB, 3, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(BGRA2LAB, 4, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(RGBA2LAB, 4, 3)
RUN_BENCHMARK_LAB_BATCH_2FUNCTIONS(LAB2BGR, 3, 3)
RUN_BENCHMARK_LAB_BATCH_2FUNCTIONS(LAB2RGB, 3, 3)
RUN_BENCHMARK_BATCH_1FUNCTIONS(LAB2BGRA, 3, 4)
RUN_BENCHMARK_BATCH_1FUNCTIONS(LAB2RGBA, 3, 4)

// RUN_BENCHMARK_LAB_COMPARISON_3FUNCTIONS(BGR2LAB, 3, 3)
// RUN_BENCHMARK_LAB_COMPARISON_3FUNCTIONS(RGB2LAB, 3, 3)
// RUN_BENCHMARK_LAB_COMPARISON_3FUNCTIONS(LAB2BGR, 3, 3)
// RUN_BENCHMARK_LAB_COMPARISON_3FUNCTIONS(LAB2RGB, 3, 3)

// BGR/RGB/BGRA/RGBA <-> NV12
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(BGR2NV12, 3, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(RGB2NV12, 3, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(BGRA2NV12, 4, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(RGBA2NV12, 4, 1)
RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(NV122BGR, 1, 3)
RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(NV122RGB, 1, 3)
RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(NV122BGRA, 1, 4)
RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(NV122RGBA, 1, 4)

// RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV122BGR, 1, 3)
// RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV122RGB, 1, 3)
// RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV122BGRA, 1, 4)
// RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV122RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> NV21
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(BGR2NV21, 3, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(RGB2NV21, 3, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(BGRA2NV21, 4, 1)
RUN_BENCHMARK_NVXX_BATCH_1FUNCTION(RGBA2NV21, 4, 1)
RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(NV212BGR, 1, 3)
RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(NV212RGB, 1, 3)
RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(NV212BGRA, 1, 4)
RUN_BENCHMARK_NVXX_BATCH_2FUNCTIONS(NV212RGBA, 1, 4)

// RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV212BGR, 1, 3)
// RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV212RGB, 1, 3)
// RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV212BGRA, 1, 4)
// RUN_BENCHMARK_NVXX_COMPARISON_2FUNCTIONS(NV212RGBA, 1, 4)

// BGR/RGB/BGRA/RGBA <-> I420
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(BGR2I420, 3, 1)
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(RGB2I420, 3, 1)
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(BGRA2I420, 4, 1)
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(RGBA2I420, 4, 1)
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(I4202BGR, 1, 3)
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(I4202RGB, 1, 3)
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(I4202BGRA, 1, 4)
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(I4202RGBA, 1, 4)

// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(BGR2I420, 3, 1)
// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(RGB2I420, 3, 1)
// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(BGRA2I420, 4, 1)
// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(RGBA2I420, 4, 1)
// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(I4202BGR, 1, 3)
// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(I4202RGB, 1, 3)
// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(I4202BGRA, 1, 4)
// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(I4202RGBA, 1, 4)

// YUV2GRAY
RUN_BENCHMARK_I420_BATCH_2FUNCTIONS(YUV2GRAY, 1, 1)

// RUN_BENCHMARK_I420_COMPARISON_2FUNCTIONS(YUV2GRAY, 1, 1)

// BGR/GRAY <-> UYVY
RUN_BENCHMARK_YUV422_BATCH_2FUNCTIONS(UYVY2BGR, 2, 3)
RUN_BENCHMARK_YUV422_BATCH_2FUNCTIONS(UYVY2GRAY, 2, 1)

// RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(UYVY2BGR, 2, 3)
// RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(UYVY2GRAY, 2, 1)

// BGR/GRAY <-> YUYV
RUN_BENCHMARK_YUV422_BATCH_2FUNCTIONS(YUYV2BGR, 2, 3)
RUN_BENCHMARK_YUV422_BATCH_2FUNCTIONS(YUYV2GRAY, 2, 1)

// RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(YUYV2BGR, 2, 3)
// RUN_BENCHMARK_YUV422_COMPARISON_2FUNCTIONS(YUYV2GRAY, 2, 1)

// NV12/21 <-> I420
RUN_BENCHMARK_NVXX_TO_I420_BATCH_1FUNCTION(NV122I420, 1, 1)
RUN_BENCHMARK_NVXX_TO_I420_BATCH_1FUNCTION(NV212I420, 1, 1)

RUN_BENCHMARK_I420_TO_NVXX_BATCH_1FUNCTION(I4202NV12, 1, 1)
RUN_BENCHMARK_I420_TO_NVXX_BATCH_1FUNCTION(I4202NV21, 1, 1)
