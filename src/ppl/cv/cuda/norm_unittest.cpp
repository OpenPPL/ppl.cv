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

#include "ppl/cv/cuda/norm.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MemoryPool, ppl::cv::NormTypes, MaskType,
                              cv::Size>;
inline std::string convertToStringNorm(const Parameters& parameters) {
  std::ostringstream formatted;

  MemoryPool memory_pool = std::get<0>(parameters);
  if (memory_pool == kActivated) {
    formatted << "MemoryPoolUsed" << "_";
  }
  else {
    formatted << "MemoryPoolUnused" << "_";
  }

  ppl::cv::NormTypes norm_type = std::get<1>(parameters);
  if (norm_type == ppl::cv::NORM_L1) {
    formatted << "NORM_L1" << "_";
  }
  else if (norm_type == ppl::cv::NORM_L2) {
    formatted << "NORM_L2" << "_";
  }
  else {
    formatted << "NORM_INF" << "_";
  }

  MaskType is_masked = std::get<2>(parameters);
  if (is_masked == kUnmasked) {
    formatted << "Unmasked" << "_";
  }
  else {
    formatted << "Masked" << "_";
  }

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaNormTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaNormTest() {
    const Parameters& parameters = GetParam();
    memory_pool = std::get<0>(parameters);
    norm_type   = std::get<1>(parameters);
    is_masked   = std::get<2>(parameters);
    size        = std::get<3>(parameters);
  }

  ~PplCvCudaNormTest() {
  }

  bool apply();

 private:
  MemoryPool memory_pool;
  ppl::cv::NormTypes norm_type;
  MaskType is_masked;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaNormTest<T, channels>::apply() {
  cv::Mat src, mask0;
  src  = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask0 = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask0(mask0);

  int src_size = size.height * size.width * channels * sizeof(T);
  int msk_size = size.height * size.width * sizeof(uchar);
  T* input = (T*)malloc(src_size);
  uchar* mask1 = (uchar*)malloc(msk_size);
  T* gpu_input;
  uchar* gpu_mask1;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_mask1, msk_size);
  copyMatToArray(src, input);
  copyMatToArray(mask0, mask1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mask1, mask1, msk_size, cudaMemcpyHostToDevice);

  cv::NormTypes cv_norm_type;
  if (norm_type == ppl::cv::NORM_INF) {
    cv_norm_type = cv::NORM_INF;
  }
  else if (norm_type == ppl::cv::NORM_L1) {
    cv_norm_type = cv::NORM_L1;
  }
  else {  // norm_type == ppl::cv::NORM_L2
    cv_norm_type = cv::NORM_L2;
  }

  if (memory_pool == kActivated) {
    size_t volume = 256 * sizeof(double);
    size_t ceiled_volume = ppl::cv::cuda::ceil1DVolume(volume);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
  }

  double result0, result1, result2;
  if (is_masked == kUnmasked) {
    result0 = cv::norm(src, cv_norm_type);
    ppl::cv::cuda::Norm<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, &result1, norm_type);
    ppl::cv::cuda::Norm<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input, &result2, norm_type);
  }
  else {
    result0 = cv::norm(src, cv_norm_type, mask0);
    ppl::cv::cuda::Norm<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, &result1, norm_type,
        gpu_mask0.step, gpu_mask0.data);
    ppl::cv::cuda::Norm<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input, &result2, norm_type, size.width,
        gpu_mask1);
  }

  if (memory_pool == kActivated) {
    ppl::cv::cuda::shutDownGpuMemoryPool();
  }

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    if (norm_type == ppl::cv::NORM_L1) {
      epsilon = EPSILON_1F;
    }
    else {
      epsilon = EPSILON_E4;
    }
  }

  bool identity0 = false;
  if (fabs(result0 - result1) < epsilon) {
    identity0 = true;
  }
  bool identity1 = false;
  if (fabs(result0 - result2) < epsilon) {
    identity1 = true;
  }

  free(input);
  free(mask1);
  cudaFree(gpu_input);
  cudaFree(gpu_mask1);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaNormTest ## T ## channels = PplCvCudaNormTest<T, channels>;     \
TEST_P(PplCvCudaNormTest ## T ## channels, Standard) {                         \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaNormTest ## T ## channels,           \
  ::testing::Combine(                                                          \
    ::testing::Values(kActivated, kUnactivated),                               \
    ::testing::Values(ppl::cv::NORM_INF, ppl::cv::NORM_L1, ppl::cv::NORM_L2),  \
    ::testing::Values(kUnmasked, kMasked),                                     \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaNormTest ## T ## channels::ParamType>& info) {                  \
    return convertToStringNorm(info.param);                                    \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
