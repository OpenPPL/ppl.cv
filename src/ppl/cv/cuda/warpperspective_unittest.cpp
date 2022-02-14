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

#include "ppl/cv/cuda/warpperspective.h"

#include <tuple>
#include <sstream>

#include "opencv2/cudawarping.hpp"
#include "gtest/gtest.h"

#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;

enum Scaling {
  kHalfSize,
  kSameSize,
  kDoubleSize,
};

using Parameters = std::tuple<Scaling, InterpolationType, BorderType, cv::Size>;
inline std::string convertToStringWarpPerspect(const Parameters& parameters) {
  std::ostringstream formatted;

  Scaling scale = std::get<0>(parameters);
  if (scale == kHalfSize) {
    formatted << "HalfSize" << "_";
  }
  else if (scale == kSameSize) {
    formatted << "SameSize" << "_";
  }
  else if (scale == kDoubleSize) {
    formatted << "DoubleSize" << "_";
  }
  else {
  }

  InterpolationType inter_type = (InterpolationType)std::get<1>(parameters);
  if (inter_type == INTERPOLATION_LINEAR) {
    formatted << "InterLinear" << "_";
  }
  else if (inter_type == INTERPOLATION_NEAREST_POINT) {
    formatted << "InterNearest" << "_";
  }
  else {
  }

  BorderType border_type = (BorderType)std::get<2>(parameters);
  if (border_type == BORDER_CONSTANT) {
    formatted << "BORDER_CONSTANT" << "_";
  }
  else if (border_type == BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == BORDER_TRANSPARENT) {
    formatted << "BORDER_TRANSPARENT" << "_";
  }
  else {
  }

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaWarpPerspectiveTest :
        public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaWarpPerspectiveTest() {
    const Parameters& parameters = GetParam();
    scale       = std::get<0>(parameters);
    inter_type  = std::get<1>(parameters);
    border_type = std::get<2>(parameters);
    size        = std::get<3>(parameters);
  }

  ~PplCvCudaWarpPerspectiveTest() {
  }

  bool apply();

 private:
  Scaling scale;
  InterpolationType inter_type;
  BorderType border_type;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaWarpPerspectiveTest<T, channels>::apply() {
  float scale_coeff;
  if (scale == kHalfSize) {
    scale_coeff = 0.5f;
  }
  else if (scale == kDoubleSize) {
    scale_coeff = 2.0f;
  }
  else {
    scale_coeff = 1.0f;
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
  cv::cuda::GpuMat gpu_cv_dst(cv_dst);
  cv::Mat M = createSourceImage(3, 3, CV_32FC1);

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

  int cv_iterpolation;
  if (inter_type == INTERPOLATION_LINEAR) {
    cv_iterpolation = cv::INTER_LINEAR;
  }
  else {
    cv_iterpolation = cv::INTER_NEAREST;
  }

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_TRANSPARENT) {
    cv_border = cv::BORDER_TRANSPARENT;
  }
  else {
  }
  int border_value = 5;
  cv::cuda::warpPerspective(gpu_src, gpu_cv_dst, M, cv::Size(dst_width,
      dst_height), cv_iterpolation | cv::WARP_INVERSE_MAP, cv_border,
      cv::Scalar(border_value, border_value, border_value, border_value));
  gpu_cv_dst.download(cv_dst);

  WarpPerspective<T, channels>(0, src.rows, src.cols, gpu_src.step / sizeof(T),
      (T*)gpu_src.data, dst_height, dst_width, gpu_dst.step / sizeof(T),
      (T*)gpu_dst.data, (float*)M.data, inter_type, border_type, border_value);
  gpu_dst.download(dst);

  WarpPerspective<T, channels>(0, src.rows, src.cols, src.cols * channels,
      gpu_input, dst_height, dst_width, dst_width * channels, gpu_output,
      (float*)M.data, inter_type, border_type, border_value);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_2F;
  }
  else {
    if (channels == 1 || channels == 4) {
      epsilon = EPSILON_E2;
    }
    else {
      epsilon = EPSILON_E4;
    }
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
using PplCvCudaWarpPerspectiveTest ## T ## channels =                          \
        PplCvCudaWarpPerspectiveTest<T, channels>;                             \
TEST_P(PplCvCudaWarpPerspectiveTest ## T ## channels, Standard) {              \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaWarpPerspectiveTest ## T ## channels,\
  ::testing::Combine(                                                          \
    ::testing::Values(kHalfSize, kSameSize, kDoubleSize),                      \
    ::testing::Values(INTERPOLATION_LINEAR,                               \
                      INTERPOLATION_NEAREST_POINT),                       \
    ::testing::Values(BORDER_CONSTANT, BORDER_REPLICATE),            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
       PplCvCudaWarpPerspectiveTest ## T ## channels::ParamType>& info) {      \
    return convertToStringWarpPerspect(info.param);                            \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
