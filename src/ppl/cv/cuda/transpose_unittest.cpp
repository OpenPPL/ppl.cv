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

#include "ppl/cv/cuda/transpose.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaTransposeTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaTransposeTest() {
    const Parameters& parameters = GetParam();
    size = std::get<0>(parameters);
  }

  ~PplCvCudaTransposeTest() {
  }

  bool apply();

 private:
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaTransposeTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.width, size.height,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.width, size.height,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(T);
  T* input = (T*)malloc(src_size);
  T* output = (T*)malloc(src_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, src_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  cv::transpose(src, cv_dst);
  ppl::cv::cuda::Transpose<T, channels>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.step / sizeof(T),
      (T*)gpu_dst.data);
  gpu_dst.download(dst);

  ppl::cv::cuda::Transpose<T, channels>(0, size.height, size.width,
      size.width * channels, gpu_input, size.height * channels, gpu_output);
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
using PplCvCudaTransposeTest ## T ## channels =                                \
        PplCvCudaTransposeTest<T, channels>;                                   \
TEST_P(PplCvCudaTransposeTest ## T ## channels, Standard) {                    \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaTransposeTest ## T ## channels,      \
  ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                    \
                    cv::Size{1283, 720}, cv::Size{1976, 1080},                 \
                    cv::Size{320, 240}, cv::Size{640, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1920, 1080}),                \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaTransposeTest ## T ## channels::ParamType>& info) {             \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
