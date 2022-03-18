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

#include "ppl/cv/cuda/split.h"

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
class PplCvCudaSplitTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaSplitTest() {
    const Parameters& parameters = GetParam();
    size = std::get<0>(parameters);
  }

  ~PplCvCudaSplitTest() {
  }

  bool apply();

 private:
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaSplitTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst0(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst1(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst2(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst3(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst0(size.height, size.width,
                  CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst1(size.height, size.width,
                  CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst2(size.height, size.width,
                  CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst3(size.height, size.width,
                  CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst0(dst0);
  cv::cuda::GpuMat gpu_dst1(dst1);
  cv::cuda::GpuMat gpu_dst2(dst2);
  cv::cuda::GpuMat gpu_dst3(dst3);

  int src_size = size.height * size.width * channels * sizeof(T);
  int dst_size = size.height * size.width * sizeof(T);
  T* input = (T*)malloc(src_size);
  T* output0 = (T*)malloc(dst_size);
  T* output1 = (T*)malloc(dst_size);
  T* output2 = (T*)malloc(dst_size);
  T* output3 = (T*)malloc(dst_size);
  T* gpu_input;
  T* gpu_output0;
  T* gpu_output1;
  T* gpu_output2;
  T* gpu_output3;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output0, dst_size);
  cudaMalloc((void**)&gpu_output1, dst_size);
  cudaMalloc((void**)&gpu_output2, dst_size);
  cudaMalloc((void**)&gpu_output3, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  if (channels == 3) {
    cv::Mat dsts[3] = {cv_dst0, cv_dst1, cv_dst2};
    cv::split(src, dsts);
    ppl::cv::cuda::Split3Channels<T>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst0.step / sizeof(T),
        (T*)gpu_dst0.data, (T*)gpu_dst1.data, (T*)gpu_dst2.data);
    ppl::cv::cuda::Split3Channels<T>(0, size.height, size.width,
        size.width * channels, gpu_input, size.width, gpu_output0, gpu_output1,
        gpu_output2);
  }
  else {  // channels == 4
    cv::Mat dsts[4] = {cv_dst0, cv_dst1, cv_dst2, cv_dst3};
    cv::split(src, dsts);
    ppl::cv::cuda::Split4Channels<T>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst0.step / sizeof(T),
        (T*)gpu_dst0.data, (T*)gpu_dst1.data, (T*)gpu_dst2.data,
        (T*)gpu_dst3.data);
    ppl::cv::cuda::Split4Channels<T>(0, size.height, size.width,
        size.width * channels, gpu_input, size.width, gpu_output0, gpu_output1,
        gpu_output2, gpu_output3);
  }
  gpu_dst0.download(dst0);
  gpu_dst1.download(dst1);
  gpu_dst2.download(dst2);
  cudaMemcpy(output0, gpu_output0, dst_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(output1, gpu_output1, dst_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(output2, gpu_output2, dst_size, cudaMemcpyDeviceToHost);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity0, identity1, identity2, identity3;
  bool identity4, identity5, identity6, identity7;
  identity0 = checkMatricesIdentity<T>(cv_dst0, dst0, epsilon);
  identity1 = checkMatricesIdentity<T>(cv_dst1, dst1, epsilon);
  identity2 = checkMatricesIdentity<T>(cv_dst2, dst2, epsilon);
  identity4 = checkMatArrayIdentity<T>(cv_dst0, output0, epsilon);
  identity5 = checkMatArrayIdentity<T>(cv_dst1, output1, epsilon);
  identity6 = checkMatArrayIdentity<T>(cv_dst2, output2, epsilon);
  if (channels == 4) {
    gpu_dst3.download(dst3);
    identity3 = checkMatricesIdentity<T>(cv_dst3, dst3, epsilon);
    cudaMemcpy(output3, gpu_output3, dst_size, cudaMemcpyDeviceToHost);
    identity7 = checkMatArrayIdentity<T>(cv_dst3, output3, epsilon);
  }

  free(input);
  free(output0);
  free(output1);
  free(output2);
  free(output3);
  cudaFree(gpu_input);
  cudaFree(gpu_output0);
  cudaFree(gpu_output1);
  cudaFree(gpu_output2);
  cudaFree(gpu_output3);

  if (channels == 3) {
    return (identity0 && identity1 && identity2 && identity4 && identity5 &&
            identity6);
  }
  else {
    return (identity0 && identity1 && identity2 && identity3 &&
            identity4 && identity5 && identity6 && identity7);
  }
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaSplitTest ## T ## channels =                                    \
        PplCvCudaSplitTest<T,  channels>;                                      \
TEST_P(PplCvCudaSplitTest ## T ## channels, Standard) {                        \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaSplitTest ## T ## channels,          \
  ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                    \
                    cv::Size{1283, 720}, cv::Size{1934, 1080},                 \
                    cv::Size{320, 240}, cv::Size{640, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1920, 1080}),                \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaSplitTest ## T ## channels::ParamType>& info) {                 \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar, 3)
UNITTEST(float, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 4)
