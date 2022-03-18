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

#include "ppl/cv/cuda/rotate.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, cv::Size>;
inline std::string convertToStringRotate(const Parameters& parameters) {
  std::ostringstream formatted;

  int degree = std::get<0>(parameters);
  formatted << "Degree" << degree << "_";

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaRotateTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaRotateTest() {
    const Parameters& parameters = GetParam();
    degree = std::get<0>(parameters);
    size   = std::get<1>(parameters);
  }

  ~PplCvCudaRotateTest() {
  }

  bool apply();

 private:
  int degree;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaRotateTest<T, channels>::apply() {
  int dst_height, dst_width;
  cv::RotateFlags cv_rotate_flag;
  if (degree == 90) {
    dst_height = size.width;
    dst_width  = size.height;
    cv_rotate_flag = cv::ROTATE_90_CLOCKWISE;
  }
  else if (degree == 180) {
    dst_height = size.height;
    dst_width  = size.width;
    cv_rotate_flag = cv::ROTATE_180;
  }
  else if (degree == 270) {
    dst_height = size.width;
    dst_width  = size.height;
    cv_rotate_flag = cv::ROTATE_90_COUNTERCLOCKWISE;
  }
  else {
    return false;
  }

  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(dst_height, dst_width,
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

  cv::rotate(src, cv_dst, cv_rotate_flag);

  ppl::cv::cuda::Rotate<T, channels>(0, size.height, size.width,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
      gpu_dst.step / sizeof(T), (T*)gpu_dst.data, degree);
  gpu_dst.download(dst);

  ppl::cv::cuda::Rotate<T, channels>(0, size.height, size.width,
      size.width * channels, gpu_input, dst_height, dst_width,
      dst_width * channels, gpu_output, degree);
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

  free(input);
  free(output);
  cudaFree(gpu_input);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaRotateTest ## T ## channels = PplCvCudaRotateTest<T, channels>; \
TEST_P(PplCvCudaRotateTest ## T ## channels, Standard) {                       \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaRotateTest ## T ## channels,         \
  ::testing::Combine(                                                          \
    ::testing::Values(90, 180, 270),                                           \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaRotateTest ## T ## channels::ParamType>& info) {                \
    return convertToStringRotate(info.param);                                  \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
