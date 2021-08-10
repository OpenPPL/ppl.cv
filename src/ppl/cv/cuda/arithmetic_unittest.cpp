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

#include "ppl/cv/cuda/arithmetic.h"

#include <tuple>
#include <sstream>

#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;

enum ArithFunctions {
  kADD,
  kADDWEITHTED,
  kSUBTRACT,
  kMUL,
};

using Parameters = std::tuple<ArithFunctions, cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  ArithFunctions function = std::get<0>(parameters);
  if (function == kADD) {
    formatted << "Add" << "_";
  }
  else if (function == kADDWEITHTED) {
    formatted << "AddWeighted" << "_";
  }
  else if (function == kSUBTRACT) {
    formatted << "Subtract" << "_";
  }
  else if (function == kMUL) {
    formatted << "Mul" << "_";
  }
  else {
  }

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template<typename T, int channels>
class PplCvCudaArithmeticTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaArithmeticTest() {
    const Parameters& parameters = GetParam();
    function = std::get<0>(parameters);
    size     = std::get<1>(parameters);
  }

  ~PplCvCudaArithmeticTest() {
  }

  bool apply();

 private:
  ArithFunctions function;
  cv::Size size;
};

template<typename T, int channels>
bool PplCvCudaArithmeticTest<T, channels>::apply() {
  cv::Mat src0, src1, dst, cv_dst;
  src0 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  src1 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  dst  = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, channels));
  dst.copyTo(cv_dst);
  cv::cuda::GpuMat gpu_src0(src0);
  cv::cuda::GpuMat gpu_src1(src1);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(T);
  T* input0 = (T*)malloc(src_size);
  T* input1 = (T*)malloc(src_size);
  T* output = (T*)malloc(src_size);
  T* gpu_input0;
  T* gpu_input1;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input0, src_size);
  cudaMalloc((void**)&gpu_input1, src_size);
  cudaMalloc((void**)&gpu_output, src_size);
  copyMatToArray(src0, input0);
  copyMatToArray(src1, input1);
  cudaMemcpy(gpu_input0, input0, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_input1, input1, src_size, cudaMemcpyHostToDevice);

  bool check_result1 = false;
  if (function == kADD) {
    cv::add(src0, src1, cv_dst);
    Add<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
                     gpu_src0.step / sizeof(T), (T*)gpu_src0.data,
                     gpu_src1.step / sizeof(T), (T*)gpu_src1.data,
                     gpu_dst.step / sizeof(T), (T*)gpu_dst.data);

    check_result1 = true;
    Add<T, channels>(0, size.height, size.width,
                     size.width * channels, gpu_input0,
                     size.width * channels, gpu_input1,
                     size.width * channels, gpu_output);
  }
  else if (function == kADDWEITHTED) {
    float alpha = 0.1f;
    float beta  = 0.2f;
    float gamma = 0.3f;
    cv::addWeighted(src0, alpha, src1, beta, gamma, cv_dst);
    AddWeighted<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
        gpu_src0.step / sizeof(T), (T*)gpu_src0.data, alpha,
        gpu_src1.step / sizeof(T), (T*)gpu_src1.data, beta, gamma,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data);

    check_result1 = true;
    AddWeighted<T, channels>(0, size.height, size.width, size.width * channels,
        gpu_input0, alpha, size.width * channels, gpu_input1, beta, gamma,
        size.width * channels, gpu_output);
  }
  else if (function == kSUBTRACT) {
    cv::Scalar scalar;
    T scalars[4];
    for (int i = 0; i < channels; i++) {
      scalar[i]  = ((T*)(src1.data))[i];
      scalars[i] = ((T*)(src1.data))[i];
    }
    cv::subtract(src0, scalar, cv_dst);

    Subtract<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
                          gpu_src0.step / sizeof(T), (T*)gpu_src0.data,
                          (T*)scalars, gpu_dst.step / sizeof(T),
                          (T*)gpu_dst.data);

    check_result1 = true;
    Subtract<T, channels>(0, size.height, size.width, size.width * channels,
                          gpu_input0, (T*)scalars, size.width * channels,
                          gpu_output);
  }
  else if (function == kMUL) {
    cv::multiply(src0, src1, cv_dst);
    Mul<T, channels>(0, gpu_src0.rows, gpu_src0.cols,
                     gpu_src0.step / sizeof(T), (T*)gpu_src0.data,
                     gpu_src1.step / sizeof(T), (T*)gpu_src1.data,
                     gpu_dst.step / sizeof(T), (T*)gpu_dst.data);

    check_result1 = true;
    Mul<T, channels>(0, size.height, size.width,
                     size.width * channels, gpu_input0,
                     size.width * channels, gpu_input1,
                     size.width * channels, gpu_output);
  }
  else {
  }
  gpu_dst.download(dst);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

  bool identity1;
  if (check_result1) {
    cudaMemcpy(output, gpu_output, src_size, cudaMemcpyDeviceToHost);
    identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);
  }

  free(input0);
  free(input1);
  free(output);
  cudaFree(gpu_input0);
  cudaFree(gpu_input1);
  cudaFree(gpu_output);

  if (check_result1) {
    return (identity0 && identity1);
  }
  else {
    return identity0;
  }
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaArithmeticTest ## T ## channels =                               \
        PplCvCudaArithmeticTest<T, channels>;                                  \
TEST_P(PplCvCudaArithmeticTest ## T ## channels, Standard) {                   \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaArithmeticTest ## T ## channels,     \
  ::testing::Combine(                                                          \
    ::testing::Values(kADD, kADDWEITHTED, kSUBTRACT, kMUL),                    \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1976, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaArithmeticTest ## T ## channels::ParamType>& info) {            \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
