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

#include "ppl/cv/cuda/copymakeborder.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, int, ppl::cv::BorderType, cv::Size>;
inline std::string convertToStringBorder(const Parameters& parameters) {
  std::ostringstream formatted;

  int top = std::get<0>(parameters);
  formatted << "TopBottom" << top << "_";

  int left = std::get<1>(parameters);
  formatted << "LeftRight" << left << "_";

  ppl::cv::BorderType border_type = std::get<2>(parameters);
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    formatted << "BORDER_CONSTANT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    formatted << "BORDER_REFLECT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_WRAP) {
    formatted << "BORDER_WRAP" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    formatted << "BORDER_REFLECT_101" << "_";
  }
  else {
  }

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaCopyMakeBorderTest :
  public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaCopyMakeBorderTest() {
    const Parameters& parameters = GetParam();
    top         = std::get<0>(parameters);
    left        = std::get<1>(parameters);
    border_type = std::get<2>(parameters);
    size        = std::get<3>(parameters);

    bottom = top;
    right  = left;
  }

  ~PplCvCudaCopyMakeBorderTest() {
  }

  bool apply();

 private:
  int top;
  int bottom;
  int left;
  int right;
  ppl::cv::BorderType border_type;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaCopyMakeBorderTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst((size.height + top + bottom), (size.width + left + right),
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst((size.height + top + bottom), (size.width + left + right),
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(T);
  int dst_size = (size.height + top + bottom) * (size.width + left + right) *
                 channels * sizeof(T);
  T* input  = (T*)malloc(src_size);
  T* output = (T*)malloc(dst_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    cv_border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_WRAP) {
    cv_border = cv::BORDER_WRAP;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    cv_border = cv::BORDER_REFLECT_101;
  }
  else {
  }
  cv::copyMakeBorder(src, cv_dst, top, bottom, left, right, cv_border);

  ppl::cv::cuda::CopyMakeBorder<T, channels>(0, src.rows, src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
      (T*)gpu_dst.data, top, bottom, left, right, border_type);
  gpu_dst.download(dst);

  ppl::cv::cuda::CopyMakeBorder<T, channels>(0, src.rows, src.cols,
      src.cols * channels, gpu_input, dst.cols * channels, gpu_output, top,
      bottom, left, right, border_type);
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

  free(input);
  free(output);
  cudaFree(gpu_input);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaCopyMakeBorderTest ## T ## channels =                           \
        PplCvCudaCopyMakeBorderTest<T, channels>;                              \
TEST_P(PplCvCudaCopyMakeBorderTest ## T ## channels, Standard) {               \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaCopyMakeBorderTest ## T ## channels, \
  ::testing::Combine(                                                          \
    ::testing::Values(0, 11, 17),                                              \
    ::testing::Values(0, 11, 17),                                              \
    ::testing::Values(ppl::cv::BORDER_CONSTANT, ppl::cv::BORDER_REPLICATE,     \
                      ppl::cv::BORDER_REFLECT, ppl::cv::BORDER_WRAP,           \
                      ppl::cv::BORDER_REFLECT_101),                            \
    ::testing::Values(cv::Size{11, 11}, cv::Size{25, 17},                      \
                      cv::Size{320, 240}, cv::Size{647, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1976, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaCopyMakeBorderTest ## T ## channels::ParamType>& info) {        \
    return convertToStringBorder(info.param);                                  \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
