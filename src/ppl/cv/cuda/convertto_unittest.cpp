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

#include "ppl/cv/cuda/convertto.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

#define BASE 50

using Parameters = std::tuple<int, int, cv::Size>;
inline std::string convertToStringConvertto(const Parameters& parameters) {
  std::ostringstream formatted;

  int int_alpha = std::get<0>(parameters);
  formatted << "IntAlpha" << int_alpha << "_";

  int int_beta = std::get<1>(parameters);
  formatted << "IntBeta" << int_beta << "_";

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename Tsrc, typename Tdst, int channels>
class PplCvCudaConvertToTest: public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaConvertToTest() {
    const Parameters& parameters = GetParam();
    alpha = (std::get<0>(parameters) - BASE) / 10.f;
    beta  = (std::get<1>(parameters) - BASE) / 10.f;
    size  = std::get<2>(parameters);
  }

  ~PplCvCudaConvertToTest() {
  }

  bool apply();

 private:
  float alpha;
  float beta;
  cv::Size size;
};

template <typename Tsrc, typename Tdst, int channels>
bool PplCvCudaConvertToTest<Tsrc, Tdst, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<Tsrc>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<Tdst>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(Tsrc);
  int dst_size = size.height * size.width * channels * sizeof(Tdst);
  Tsrc* input  = (Tsrc*)malloc(src_size);
  Tdst* output = (Tdst*)malloc(dst_size);
  Tsrc* gpu_input;
  Tdst* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  src.convertTo(cv_dst, cv_dst.type(), alpha, beta);

  ppl::cv::cuda::ConvertTo<Tsrc, Tdst, channels>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(Tsrc), (Tsrc*)gpu_src.data,
      gpu_dst.step / sizeof(Tdst), (Tdst*)gpu_dst.data, alpha, beta);
  gpu_dst.download(dst);

  ppl::cv::cuda::ConvertTo<Tsrc, Tdst, channels>(0, size.height, size.width,
      size.width * channels, gpu_input, size.width * channels, gpu_output,
      alpha, beta);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(Tdst) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0 = checkMatricesIdentity<Tdst>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<Tdst>(cv_dst, output, epsilon);

  free(input);
  free(output);
  cudaFree(gpu_input);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(Tsrc, Tdst, channels)                                         \
using PplCvCudaConvertToTest ## Tsrc ## To ## Tdst ## channels =               \
        PplCvCudaConvertToTest<Tsrc, Tdst, channels>;                          \
TEST_P(PplCvCudaConvertToTest ## Tsrc ## To ## Tdst ## channels, Standard) {   \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaConvertToTest ## Tsrc ## To ## Tdst ## channels,                    \
  ::testing::Combine(                                                          \
    ::testing::Values(37, 60, 65),                                             \
    ::testing::Values(13, 50, 89),                                             \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaConvertToTest ## Tsrc ## To ## Tdst ## channels::ParamType>&    \
        info) {                                                                \
    return convertToStringConvertto(info.param);                               \
  }                                                                            \
);

UNITTEST(uchar, uchar, 1)
UNITTEST(uchar, uchar, 3)
UNITTEST(uchar, uchar, 4)
UNITTEST(uchar, float, 1)
UNITTEST(uchar, float, 3)
UNITTEST(uchar, float, 4)
UNITTEST(float, uchar, 1)
UNITTEST(float, uchar, 3)
UNITTEST(float, uchar, 4)
UNITTEST(float, float, 1)
UNITTEST(float, float, 3)
UNITTEST(float, float, 4)
