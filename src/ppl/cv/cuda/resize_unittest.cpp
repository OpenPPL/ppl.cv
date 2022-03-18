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

#include "ppl/cv/cuda/resize.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

enum Scaling {
  kHalfSize,
  kSmallerSize,
  kSameSize,
  kBiggerSize,
  kDoubleSize,
};

using Parameters = std::tuple<Scaling, ppl::cv::InterpolationType, cv::Size>;
inline std::string convertToStringResize(const Parameters& parameters) {
  std::ostringstream formatted;

  Scaling scale = std::get<0>(parameters);
  if (scale == kHalfSize) {
    formatted << "HalfSize" << "_";
  }
  else if (scale == kSmallerSize) {
    formatted << "SmallerSize" << "_";
  }
  else if (scale == kSameSize) {
    formatted << "SameSize" << "_";
  }
  else if (scale == kBiggerSize) {
    formatted << "BiggerSize" << "_";
  }
  else if (scale == kDoubleSize) {
    formatted << "DoubleSize" << "_";
  }
  else {
  }

  ppl::cv::InterpolationType inter_type = std::get<1>(parameters);
  if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
    formatted << "InterLinear" << "_";
  }
  else if (inter_type == ppl::cv::INTERPOLATION_NEAREST_POINT) {
    formatted << "InterNearest" << "_";
  }
  else if (inter_type == ppl::cv::INTERPOLATION_AREA) {
    formatted << "InterArea" << "_";
  }
  else {
  }

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaResizeTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaResizeTest() {
    const Parameters& parameters = GetParam();
    scale      = std::get<0>(parameters);
    inter_type = std::get<1>(parameters);
    size       = std::get<2>(parameters);
  }

  ~PplCvCudaResizeTest() {
  }

  bool apply();

 private:
  Scaling scale;
  ppl::cv::InterpolationType inter_type;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaResizeTest<T, channels>::apply() {
  float scale_coeff;
  if (scale == kHalfSize) {
    scale_coeff = 0.5f;
  }
  else if (scale == kSmallerSize) {
    scale_coeff = 0.7f;
  }
  else if (scale == kSameSize) {
    scale_coeff = 1.0f;
  }
  else if (scale == kBiggerSize) {
    scale_coeff = 1.4f;
  }
  else {  // scale == kDoubleSize
    scale_coeff = 2.0f;
  }
  int dst_height = size.height * scale_coeff;
  int dst_width  = size.width * scale_coeff;
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(dst_height, dst_width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(dst_height, dst_width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = src.rows * src.cols * channels * sizeof(T);
  int dst_size = dst.rows * dst.cols * channels * sizeof(T);
  T* input  = (T*)malloc(src_size);
  T* output = (T*)malloc(dst_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  int cv_iterpolation = cv::INTER_LINEAR;
  if (inter_type == ppl::cv::INTERPOLATION_NEAREST_POINT) {
    cv_iterpolation = cv::INTER_NEAREST;
  }
  else if (inter_type == ppl::cv::INTERPOLATION_AREA) {
    cv_iterpolation = cv::INTER_AREA;
  }
  else {
  }
  cv::resize(src, cv_dst, cv::Size(dst_width, dst_height), 0, 0,
             cv_iterpolation);

  ppl::cv::cuda::Resize<T, channels>(0, src.rows, src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
      gpu_dst.step / sizeof(T), (T*)gpu_dst.data, inter_type);
  gpu_dst.download(dst);

  ppl::cv::cuda::Resize<T, channels>(0, src.rows, src.cols, src.cols * channels,
      gpu_input, dst_height, dst_width, dst_width * channels, gpu_output,
      inter_type);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_2F;
  }
  else {
    epsilon = EPSILON_E2;
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
using PplCvCudaResizeTest ## T ## channels =                                   \
        PplCvCudaResizeTest<T, channels>;                                      \
TEST_P(PplCvCudaResizeTest ## T ## channels, Standard) {                       \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaResizeTest ## T ## channels,         \
  ::testing::Combine(                                                          \
    ::testing::Values(kHalfSize, kSmallerSize, kSameSize, kBiggerSize,         \
                      kDoubleSize),                                            \
    ::testing::Values(ppl::cv::INTERPOLATION_LINEAR,                           \
                      ppl::cv::INTERPOLATION_NEAREST_POINT,                    \
                      ppl::cv::INTERPOLATION_AREA),                            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
       PplCvCudaResizeTest ## T ## channels::ParamType>& info) {               \
    return convertToStringResize(info.param);                                  \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
