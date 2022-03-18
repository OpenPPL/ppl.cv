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

#include "ppl/cv/cuda/setvalue.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

enum SetValueFunctions {
  kUnmaskedSetTo,
  kMaskedSetTo,
  kOnes,
  kZeros,
};

using Parameters = std::tuple<SetValueFunctions, cv::Size>;
inline std::string convertToStringSetValue(const Parameters& parameters) {
  std::ostringstream formatted;

  SetValueFunctions function = std::get<0>(parameters);
  if (function == kUnmaskedSetTo) {
    formatted << "UnmaskedSetTo" << "_";
  }
  else if (function == kMaskedSetTo) {
    formatted << "MaskedSetTo" << "_";
  }
  else if (function == kOnes) {
    formatted << "Ones" << "_";
  }
  else {
    formatted << "Zeros" << "_";
  }

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int outChannels, int maskChannels>
class PplCvCudaSetValueTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaSetValueTest() {
    const Parameters& parameters = GetParam();
    function = std::get<0>(parameters);
    size     = std::get<1>(parameters);
  }

  ~PplCvCudaSetValueTest() {
  }

  bool apply();

 private:
  SetValueFunctions function;
  cv::Size size;
};

template <typename T, int outChannels, int maskChannels>
bool PplCvCudaSetValueTest<T, outChannels, maskChannels>::apply() {
  cv::Mat dst, mask0, cv_dst;
  dst = cv::Mat::zeros(size.height, size.width,
                       CV_MAKETYPE(cv::DataType<T>::depth, outChannels));
  mask0 = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth,
                            maskChannels));
  dst.copyTo(cv_dst);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_mask0(mask0);

  int dst_size = size.height * size.width * outChannels * sizeof(T);
  int mask_size = size.height * size.width * maskChannels * sizeof(uchar);
  T* output = (T*)malloc(dst_size);
  uchar* mask1 = (uchar*)malloc(mask_size);
  T* gpu_output;
  uchar* gpu_mask1;
  cudaMalloc((void**)&gpu_output, dst_size);
  cudaMalloc((void**)&gpu_mask1, mask_size);
  copyMatToArray(mask0, mask1);
  cudaMemcpy(gpu_output, output, dst_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mask1, mask1, mask_size, cudaMemcpyHostToDevice);

  int value = 5;
  if (function == kUnmaskedSetTo) {
    cv_dst.setTo(value);

    ppl::cv::cuda::SetTo<T, outChannels, maskChannels>(0, gpu_dst.rows,
        gpu_dst.cols, gpu_dst.step / sizeof(T), (T*)gpu_dst.data, value);

    ppl::cv::cuda::SetTo<T, outChannels, maskChannels>(0, size.height,
        size.width, size.width * outChannels, gpu_output, value);
  }
  else if (function == kMaskedSetTo) {
    cv_dst.setTo(value, mask0);

    ppl::cv::cuda::SetTo<T, outChannels, maskChannels>(0, gpu_dst.rows,
        gpu_dst.cols, gpu_dst.step / sizeof(T), (T*)gpu_dst.data, value,
        gpu_mask0.step, gpu_mask0.data);

    ppl::cv::cuda::SetTo<T, outChannels, maskChannels>(0, size.height,
        size.width, size.width * outChannels, gpu_output, value,
        size.width * maskChannels, gpu_mask1);
  }
  else if (function == kOnes) {
    cv_dst = cv::Mat::ones(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, outChannels));

    ppl::cv::cuda::Ones<T, outChannels>(0, gpu_dst.rows, gpu_dst.cols,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data);

    ppl::cv::cuda::Ones<T, outChannels>(0, size.height, size.width,
        size.width * outChannels, gpu_output);
  }
  else {
    cv_dst = cv::Mat::zeros(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<T>::depth, outChannels));

    ppl::cv::cuda::Zeros<T, outChannels>(0, gpu_dst.rows, gpu_dst.cols,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data);

    ppl::cv::cuda::Zeros<T, outChannels>(0, size.height, size.width,
        size.width * outChannels, gpu_output);
  }
  gpu_dst.download(dst);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);

  free(output);
  free(mask1);
  cudaFree(gpu_output);
  cudaFree(gpu_mask1);

  return (identity0 && identity1);
}

#define UNITTEST(T, outChannels, maskChannels)                                 \
using PplCvCudaSetValueTest ## T ## outChannels ## maskChannels =              \
        PplCvCudaSetValueTest<T, outChannels, maskChannels>;                   \
TEST_P(PplCvCudaSetValueTest ## T ## outChannels ## maskChannels, Standard) {  \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaSetValueTest ## T ## outChannels ## maskChannels,                   \
  ::testing::Combine(                                                          \
    ::testing::Values(kUnmaskedSetTo, kMaskedSetTo, kOnes, kZeros),            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaSetValueTest ## T ## outChannels ## maskChannels::ParamType>&   \
      info) {                                                                  \
    return convertToStringSetValue(info.param);                                \
  }                                                                            \
);

UNITTEST(uchar, 1, 1)
UNITTEST(uchar, 3, 3)
UNITTEST(uchar, 4, 4)
UNITTEST(uchar, 3, 1)
UNITTEST(uchar, 4, 1)
UNITTEST(float, 1, 1)
UNITTEST(float, 3, 3)
UNITTEST(float, 4, 4)
UNITTEST(float, 3, 1)
UNITTEST(float, 4, 1)
