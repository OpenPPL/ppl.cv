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

#include "ppl/cv/cuda/normalize.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MemoryPool, ppl::cv::NormTypes, MaskType, int,
                              int, cv::Size>;
inline std::string convertToStringNormalize(const Parameters& parameters) {
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
  else if (norm_type == ppl::cv::NORM_INF) {
    formatted << "NORM_INF" << "_";
  }
  else {  // norm_type == ppl::cv::NORM_MINMAX
    formatted << "NORM_MINMAX" << "_";
  }

  MaskType is_masked = std::get<2>(parameters);
  if (is_masked == kUnmasked) {
    formatted << "Unmasked" << "_";
  }
  else {
    formatted << "Masked" << "_";
  }

  int int_alpha = std::get<3>(parameters);
  formatted << "IntAlpha" << int_alpha << "_";

  int int_beta = std::get<4>(parameters);
  formatted << "IntBeta" << int_beta << "_";

  cv::Size size = std::get<5>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaNormalizeTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaNormalizeTest() {
    const Parameters& parameters = GetParam();
    memory_pool = std::get<0>(parameters);
    norm_type   = std::get<1>(parameters);
    is_masked   = std::get<2>(parameters);
    alpha       = std::get<3>(parameters) / 10.f;
    beta        = std::get<4>(parameters) / 10.f;
    size        = std::get<5>(parameters);
  }

  ~PplCvCudaNormalizeTest() {
  }

  bool apply();

 private:
  MemoryPool memory_pool;
  ppl::cv::NormTypes norm_type;
  MaskType is_masked;
  float alpha;
  float beta;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaNormalizeTest<T, channels>::apply() {
  cv::Mat src, cv_dst, mask0;
  src  = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask0 = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::Mat dst(src.rows, src.cols,
              CV_MAKETYPE(cv::DataType<float>::depth, channels));
  cv_dst = cv::Mat::zeros(src.rows, src.cols,
                          CV_MAKETYPE(cv::DataType<float>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask0(mask0);

  int src_size = size.height * size.width * channels * sizeof(T);
  int dst_size = size.height * size.width * channels * sizeof(float);
  int msk_size = size.height * size.width * sizeof(uchar);
  T* input = (T*)malloc(src_size);
  float* output = (float*)malloc(dst_size);
  uchar* mask1 = (uchar*)malloc(msk_size);
  T* gpu_input;
  float* gpu_output;
  uchar* gpu_mask1;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
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
  else if (norm_type == ppl::cv::NORM_L2) {
    cv_norm_type = cv::NORM_L2;
  }
  else {  // norm_type == ppl::cv::NORM_MINMAX
    cv_norm_type = cv::NORM_MINMAX;
  }

  if (memory_pool == kActivated) {
    size_t volume = 256 * 2 * sizeof(double);
    size_t ceiled_volume = ppl::cv::cuda::ceil1DVolume(volume);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
  }

  if (is_masked == kUnmasked) {
    cv::normalize(src, cv_dst, alpha, beta, cv_norm_type,
                  CV_MAT_DEPTH(dst.type()));
    ppl::cv::cuda::Normalize<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data,
        gpu_dst.step / sizeof(float), (float*)gpu_dst.data, alpha, beta,
        norm_type);
    ppl::cv::cuda::Normalize<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input, size.width * channels, gpu_output,
        alpha, beta, norm_type);
  }
  else {
    if (channels > 1 && norm_type == ppl::cv::NORM_MINMAX) {
      // cv::normalize() assertion fails.
      cudaMemset(gpu_dst.data, 0, gpu_dst.step * gpu_src.rows);
      cudaMemset(gpu_output, 0, dst_size);
    }
    else {
      cv::normalize(src, cv_dst, alpha, beta, cv_norm_type,
                    CV_MAT_DEPTH(dst.type()), mask0);
      ppl::cv::cuda::Normalize<T, channels>(0, gpu_src.rows, gpu_src.cols,
          gpu_src.step / sizeof(T), (T*)gpu_src.data,
          gpu_dst.step / sizeof(float), (float*)gpu_dst.data, alpha, beta,
          norm_type, gpu_mask0.step, (uchar*)gpu_mask0.data);
      ppl::cv::cuda::Normalize<T, channels>(0, size.height, size.width,
          size.width * channels, gpu_input, size.width * channels,
          gpu_output, alpha, beta, norm_type, size.width, gpu_mask1);
    }
  }
  gpu_dst.download(dst);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

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
  bool identity0 = checkMatricesIdentity<float>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<float>(cv_dst, output, epsilon);

  free(input);
  free(output);
  free(mask1);
  cudaFree(gpu_input);
  cudaFree(gpu_output);
  cudaFree(gpu_mask1);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaNormalizeTest ## T ## channels =                                \
        PplCvCudaNormalizeTest<T, channels>;                                   \
TEST_P(PplCvCudaNormalizeTest ## T ## channels, Standard) {                    \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaNormalizeTest ## T ## channels,      \
  ::testing::Combine(                                                          \
    ::testing::Values(kActivated, kUnactivated),                               \
    ::testing::Values(ppl::cv::NORM_INF, ppl::cv::NORM_L1, ppl::cv::NORM_L2,   \
                      ppl::cv::NORM_MINMAX),                                   \
    ::testing::Values(kUnmasked, kMasked),                                     \
    ::testing::Values(3, 10, 15),                                              \
    ::testing::Values(7, 19, 42),                                              \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaNormalizeTest ## T ## channels::ParamType>& info) {             \
    return convertToStringNormalize(info.param);                               \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
