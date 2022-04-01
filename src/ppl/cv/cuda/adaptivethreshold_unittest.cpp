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

#include "ppl/cv/cuda/adaptivethreshold.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MemoryPool, int, int, int, int, int,
                              ppl::cv::BorderType, cv::Size>;
inline std::string convertToStringThreshold(const Parameters& parameters) {
  std::ostringstream formatted;

  MemoryPool memory_pool = std::get<0>(parameters);
  if (memory_pool == kActivated) {
    formatted << "MemoryPoolUsed" << "_";
  }
  else {
    formatted << "MemoryPoolUnused" << "_";
  }

  int ksize = std::get<1>(parameters);
  formatted << "Ksize" << ksize << "_";

  int adaptive_method = std::get<2>(parameters);
  formatted << (adaptive_method == ppl::cv::ADAPTIVE_THRESH_MEAN_C ?
                                   "METHOD_MEAN" : "METHOD_GAUSSIAN") << "_";

  int threshold_type = std::get<3>(parameters);
  formatted << (threshold_type == ppl::cv::THRESH_BINARY ? "THRESH_BINARY" :
                                  "THRESH_BINARY_INV") << "_";

  int max_value = std::get<4>(parameters);
  formatted << "MaxValue" << max_value << "_";

  int int_delta = std::get<5>(parameters);
  formatted << "IntDelta" << int_delta << "_";

  ppl::cv::BorderType border_type = std::get<6>(parameters);
  if (border_type == ppl::cv::BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    formatted << "BORDER_REFLECT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    formatted << "BORDER_REFLECT_101" << "_";
  }
  else {  // border_type == ppl::cv::BORDER_DEFAULT
    formatted << "BORDER_DEFAULT" << "_";
  }

  cv::Size size = std::get<7>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaAdaptiveThresholdTest :
  public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaAdaptiveThresholdTest() {
    const Parameters& parameters = GetParam();
    memory_pool     = std::get<0>(parameters);
    ksize           = std::get<1>(parameters);
    adaptive_method = std::get<2>(parameters);
    threshold_type  = std::get<3>(parameters);
    max_value       = std::get<4>(parameters) / 10.f;
    delta           = std::get<5>(parameters) / 10.f;
    border_type     = std::get<6>(parameters);
    size            = std::get<7>(parameters);

    max_value -= 1.f;
  }

  ~PplCvCudaAdaptiveThresholdTest() {
  }

  bool apply();

 private:
  MemoryPool memory_pool;
  int ksize;
  int adaptive_method;
  int threshold_type;
  float max_value;
  float delta;
  ppl::cv::BorderType border_type;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaAdaptiveThresholdTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(T);
  T* input  = (T*)malloc(src_size);
  T* output = (T*)malloc(src_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, src_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  cv::AdaptiveThresholdTypes cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
  if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_MEAN_C) {
    cv_adaptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
  }
  else if (adaptive_method == ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C) {
    cv_adaptive_method = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
  }
  else {
  }
  cv::ThresholdTypes cv_threshold_type = cv::THRESH_BINARY;
  if (threshold_type == ppl::cv::THRESH_BINARY) {
    cv_threshold_type = cv::THRESH_BINARY;
  }
  else if (threshold_type == ppl::cv::THRESH_BINARY_INV) {
    cv_threshold_type = cv::THRESH_BINARY_INV;
  }
  cv::adaptiveThreshold(src, cv_dst, max_value, cv_adaptive_method,
                        cv_threshold_type, ksize, delta);

  if (memory_pool == kActivated) {
    size_t width = size.width * sizeof(float);
    size_t ceiled_volume = ppl::cv::cuda::ceil2DVolume(width, size.height);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
  }

  ppl::cv::cuda::AdaptiveThreshold(0, gpu_src.rows, gpu_src.cols, gpu_src.step,
      (uchar*)gpu_src.data, gpu_dst.step, (uchar*)gpu_dst.data, max_value,
      adaptive_method, threshold_type, ksize, delta, border_type);
  gpu_dst.download(dst);

  ppl::cv::cuda::AdaptiveThreshold(0, size.height, size.width,
      size.width * channels, gpu_input, size.width * channels, gpu_output,
      max_value, adaptive_method, threshold_type, ksize, delta, border_type);
  cudaMemcpy(output, gpu_output, src_size, cudaMemcpyDeviceToHost);

  if (memory_pool == kActivated) {
    ppl::cv::cuda::shutDownGpuMemoryPool();
  }

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_2F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);

  free(input);
  free(output);
  cudaFree(gpu_input);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaAdaptiveThresholdTest ## T ## channels =                        \
        PplCvCudaAdaptiveThresholdTest<T, channels>;                           \
TEST_P(PplCvCudaAdaptiveThresholdTest ## T ## channels, Standard) {            \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaAdaptiveThresholdTest ## T ## channels,                             \
  ::testing::Combine(                                                          \
    ::testing::Values(kActivated, kUnactivated),                               \
    ::testing::Values(3, 5, 13, 31, 43),                                       \
    ::testing::Values(ppl::cv::ADAPTIVE_THRESH_MEAN_C,                         \
                      ppl::cv::ADAPTIVE_THRESH_GAUSSIAN_C),                    \
    ::testing::Values(ppl::cv::THRESH_BINARY, ppl::cv::THRESH_BINARY_INV),     \
    ::testing::Values(0, 70, 1587, 3784),                                      \
    ::testing::Values(0, 70, 1587, 3784),                                      \
    ::testing::Values(ppl::cv::BORDER_REPLICATE),                              \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaAdaptiveThresholdTest ## T ## channels::ParamType>& info) {     \
    return convertToStringThreshold(info.param);                               \
  }                                                                            \
);

UNITTEST(uchar, 1)
