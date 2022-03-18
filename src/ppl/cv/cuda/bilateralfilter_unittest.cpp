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

#include "ppl/cv/cuda/bilateralfilter.h"

#include <tuple>
#include <sstream>

#include "opencv2/cudaimgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, int, int, ppl::cv::BorderType, cv::Size>;
inline
std::string convertToStringBilateralFilter(const Parameters& parameters) {
  std::ostringstream formatted;

  int diameter = std::get<0>(parameters);
  formatted << "Diameter" << diameter << "_";

  int int_color = std::get<1>(parameters);
  formatted << "IntColor" << int_color << "_";

  int int_space = std::get<2>(parameters);
  formatted << "IntSpace" << int_space << "_";

  ppl::cv::BorderType border_type = std::get<3>(parameters);
  if (border_type == ppl::cv::BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    formatted << "BORDER_REFLECT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    formatted << "BORDER_REFLECT_101" << "_";
  }
  else {  // border_type == ppl::cv::BORDER_DEFAULT
    formatted << "BORDER_DEFAULT" << "_";
  }

  cv::Size size = std::get<4>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaBilateralFilterTest :
  public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaBilateralFilterTest() {
    const Parameters& parameters = GetParam();
    diameter    = std::get<0>(parameters);
    sigma_color = std::get<1>(parameters);
    sigma_space = std::get<2>(parameters);
    border_type = std::get<3>(parameters);
    size        = std::get<4>(parameters);

    diameter    -= 4;
    sigma_color -= 4;
    sigma_space -= 4;
  }

  ~PplCvCudaBilateralFilterTest() {
  }

  bool apply();

 private:
  int diameter;
  float sigma_color;
  float sigma_space;
  ppl::cv::BorderType border_type;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaBilateralFilterTest<T, channels>::apply() {
  if (diameter <= 0 && sigma_space <= 0) {
    return true;
  }

  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);
  cv::cuda::GpuMat gpu_cv_dst(cv_dst);

  int src_size = size.height * size.width * channels * sizeof(T);
  T* input  = (T*)malloc(src_size);
  T* output = (T*)malloc(src_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, src_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT) {
    cv_border = cv::BORDER_REFLECT;
  }
  else if (border_type == ppl::cv::BORDER_REFLECT_101) {
    cv_border = cv::BORDER_REFLECT_101;
  }
  else {
  }
  cv::cuda::bilateralFilter(gpu_src, gpu_cv_dst, diameter, sigma_color,
                            sigma_space, cv_border);
  gpu_cv_dst.download(cv_dst);

  ppl::cv::cuda::BilateralFilter<T, channels>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, diameter, sigma_color,
      sigma_space, gpu_dst.step / sizeof(T), (T*)gpu_dst.data, border_type);
  gpu_dst.download(dst);

  ppl::cv::cuda::BilateralFilter<T, channels>(0, size.height, size.width,
      size.width * channels, gpu_input, diameter, sigma_color, sigma_space,
      size.width * channels, gpu_output, border_type);
  cudaMemcpy(output, gpu_output, src_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E5;
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
using PplCvCudaBilateralFilterTest ## T ## channels =                          \
        PplCvCudaBilateralFilterTest<T, channels>;                             \
TEST_P(PplCvCudaBilateralFilterTest ## T ## channels, Standard) {              \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaBilateralFilterTest ## T ## channels,\
  ::testing::Combine(                                                          \
    ::testing::Values(1, 7, 12, 21, 35, 47),                                   \
    ::testing::Values(1, 27),                                                  \
    ::testing::Values(1, 14),                                                  \
    ::testing::Values(ppl::cv::BORDER_REPLICATE, ppl::cv::BORDER_REFLECT,      \
                      ppl::cv::BORDER_REFLECT_101),                            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaBilateralFilterTest ## T ## channels::ParamType>& info) {       \
    return convertToStringBilateralFilter(info.param);                         \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(float, 1)
UNITTEST(float, 3)
