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

#include "ppl/cv/cuda/bitwise.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MaskType, cv::Size>;
inline std::string convertToStringBitwise(const Parameters& parameters) {
  std::ostringstream formatted;

  MaskType is_masked = std::get<0>(parameters);
  if (is_masked == kUnmasked) {
    formatted << "Unmasked" << "_";
  }
  else {
    formatted << "Masked" << "_";
  }

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaBitwiseTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaBitwiseTest() {
    const Parameters& parameters = GetParam();
    is_masked = std::get<0>(parameters);
    size      = std::get<1>(parameters);
  }

  ~PplCvCudaBitwiseTest() {
  }

  bool apply();

 private:
  MaskType is_masked;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaBitwiseTest<T, channels>::apply() {
  cv::Mat src0, src1, dst, cv_dst, mask0;
  src0 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  src1 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  dst = cv::Mat::zeros(size.height, size.width,
                       CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask0 = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  dst.copyTo(cv_dst);
  cv::cuda::GpuMat gpu_src0(src0);
  cv::cuda::GpuMat gpu_src1(src1);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask0(mask0);

  int src_size = size.height * size.width * channels * sizeof(T);
  int mask_size = size.height * size.width * sizeof(uchar);
  T* input0 = (T*)malloc(src_size);
  T* input1 = (T*)malloc(src_size);
  T* output = (T*)malloc(src_size);
  uchar* mask1 = (uchar*)malloc(mask_size);
  T* gpu_input0;
  T* gpu_input1;
  T* gpu_output;
  uchar* gpu_mask1;
  cudaMalloc((void**)&gpu_input0, src_size);
  cudaMalloc((void**)&gpu_input1, src_size);
  cudaMalloc((void**)&gpu_output, src_size);
  cudaMalloc((void**)&gpu_mask1, mask_size);
  copyMatToArray(src0, input0);
  copyMatToArray(src1, input1);
  copyMatToArray(mask0, mask1);
  cudaMemcpy(gpu_input0, input0, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_input1, input1, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mask1, mask1, mask_size, cudaMemcpyHostToDevice);

  if (is_masked == kUnmasked) {
    cv::bitwise_and(src0, src1, cv_dst);
    ppl::cv::cuda::BitwiseAnd<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
        gpu_src0.step / sizeof(T), (T*)gpu_src0.data, gpu_src1.step / sizeof(T),
        (T*)gpu_src1.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);

    ppl::cv::cuda::BitwiseAnd<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input0, size.width * channels, gpu_input1,
        size.width * channels, gpu_output);
  }
  else {
    cv::bitwise_and(src0, src1, cv_dst, mask0);
    ppl::cv::cuda::BitwiseAnd<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
        gpu_src0.step / sizeof(T), (T*)gpu_src0.data, gpu_src1.step / sizeof(T),
        (T*)gpu_src1.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data,
        gpu_mask0.step / sizeof(uchar), (uchar*)gpu_mask0.data);

    ppl::cv::cuda::BitwiseAnd<T, channels>(0, size.height, size.width,
        size.width * channels, gpu_input0, size.width * channels, gpu_input1,
        size.width * channels, gpu_output, size.width, gpu_mask1);
  }
  gpu_dst.download(dst);
  cudaMemcpy(output, gpu_output, src_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);

  free(input0);
  free(input1);
  free(output);
  free(mask1);
  cudaFree(gpu_input0);
  cudaFree(gpu_input1);
  cudaFree(gpu_output);
  cudaFree(gpu_mask1);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaBitwiseTest ## T ## channels =                                  \
        PplCvCudaBitwiseTest<T, channels>;                                     \
TEST_P(PplCvCudaBitwiseTest ## T ## channels, Standard) {                      \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaBitwiseTest ## T ## channels,        \
  ::testing::Combine(                                                          \
    ::testing::Values(kUnmasked, kMasked),                                     \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1976, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaBitwiseTest ## T ## channels::ParamType>& info) {               \
    return convertToStringBitwise(info.param);                                 \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
