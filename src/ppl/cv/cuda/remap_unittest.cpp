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

#include "ppl/cv/cuda/remap.h"

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

using Parameters = std::tuple<Scaling, ppl::cv::InterpolationType,
                              ppl::cv::BorderType, cv::Size>;
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
  else {
  }

  ppl::cv::BorderType border_type = std::get<2>(parameters);
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    formatted << "BORDER_CONSTANT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
    formatted << "BORDER_TRANSPARENT" << "_";
  }
  else {  // border_type == ppl::cv::BORDER_DEFAULT
    formatted << "BORDER_DEFAULT" << "_";
  }

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaRemapTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaRemapTest() {
    const Parameters& parameters = GetParam();
    scale       = std::get<0>(parameters);
    inter_type  = std::get<1>(parameters);
    border_type = std::get<2>(parameters);
    size        = std::get<3>(parameters);
  }

  ~PplCvCudaRemapTest() {
  }

  bool apply();

 private:
  Scaling scale;
  ppl::cv::InterpolationType inter_type;
  ppl::cv::BorderType border_type;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaRemapTest<T, channels>::apply() {
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
  cv::Mat src, map_x0, map_y0;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  map_x0 = createSourceImage(dst_height, dst_width,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  map_y0 = createSourceImage(dst_height, dst_width,
                             CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(dst_height, dst_width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(dst_height, dst_width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = src.rows * src.cols * channels * sizeof(T);
  int map_size = dst_height * dst_width * sizeof(float);
  int dst_size = dst.rows * dst.cols * channels * sizeof(T);
  T* input = (T*)malloc(src_size);
  float* map_x1 = (float*)malloc(map_size);
  float* map_y1 = (float*)malloc(map_size);
  T* output = (T*)malloc(dst_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  float* gpu_map_x;
  float* gpu_map_y;
  cudaMalloc((void**)&gpu_map_x, map_size);
  cudaMalloc((void**)&gpu_map_y, map_size);
  copyMatToArray(src, input);
  copyMatToArray(map_x0, map_x1);
  copyMatToArray(map_y0, map_y1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_map_x, map_x1, map_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_map_y, map_y1, map_size, cudaMemcpyHostToDevice);

  int cv_iterpolation;
  if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
    cv_iterpolation = cv::INTER_LINEAR;
  }
  else {
    cv_iterpolation = cv::INTER_NEAREST;
  }

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_TRANSPARENT) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else {
  }
  cv::remap(src, cv_dst, map_x0, map_y0, cv_iterpolation, cv_border);

  ppl::cv::cuda::Remap<T, channels>(0, src.rows, src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, dst_height, dst_width,
      gpu_dst.step / sizeof(T), (T*)gpu_dst.data, gpu_map_x, gpu_map_y,
      inter_type, border_type);
  gpu_dst.download(dst);

  ppl::cv::cuda::Remap<T, channels>(0, src.rows, src.cols, src.cols * channels,
      gpu_input, dst_height, dst_width, dst_width * channels, gpu_output,
      gpu_map_x, gpu_map_y, inter_type, border_type);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (inter_type == ppl::cv::INTERPOLATION_LINEAR) {
    if (sizeof(T) == 1) {
      epsilon = 8.1f;
    }
    else {
      epsilon = EPSILON_E1;
    }
  }
  else {
    if (sizeof(T) == 1) {
      epsilon = EPSILON_1F;
    }
    else {
      epsilon = EPSILON_E5;
    }
  }
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);

  free(input);
  free(output);
  free(map_x1);
  free(map_y1);
  cudaFree(gpu_input);
  cudaFree(gpu_output);
  cudaFree(gpu_map_x);
  cudaFree(gpu_map_y);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaRemapTest ## T ## channels = PplCvCudaRemapTest<T, channels>;   \
TEST_P(PplCvCudaRemapTest ## T ## channels, Standard) {                        \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaRemapTest ## T ## channels,          \
  ::testing::Combine(                                                          \
    ::testing::Values(kHalfSize, kSmallerSize, kSameSize, kBiggerSize,         \
                      kDoubleSize),                                            \
    ::testing::Values(ppl::cv::INTERPOLATION_LINEAR,                           \
                      ppl::cv::INTERPOLATION_NEAREST_POINT),                   \
    ::testing::Values(ppl::cv::BORDER_CONSTANT, ppl::cv::BORDER_REPLICATE,     \
                      ppl::cv::BORDER_TRANSPARENT),                            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
       PplCvCudaRemapTest ## T ## channels::ParamType>& info) {                \
    return convertToStringResize(info.param);                                  \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
