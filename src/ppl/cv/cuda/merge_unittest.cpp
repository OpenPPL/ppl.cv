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

#include "ppl/cv/cuda/merge.h"

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
class PplCvCudaMergeTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaMergeTest() {
    const Parameters& parameters = GetParam();
    size = std::get<0>(parameters);
  }

  ~PplCvCudaMergeTest() {
  }

  bool apply();

 private:
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaMergeTest<T, channels>::apply() {
  cv::Mat src0, src1, src2, src3;
  src0 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src1 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src2 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  src3 = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src0(src0);
  cv::cuda::GpuMat gpu_src1(src1);
  cv::cuda::GpuMat gpu_src2(src2);
  cv::cuda::GpuMat gpu_src3(src3);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * sizeof(T);
  int dst_size = size.height * size.width * channels * sizeof(T);
  T* input0  = (T*)malloc(src_size);
  T* input1  = (T*)malloc(src_size);
  T* input2  = (T*)malloc(src_size);
  T* input3  = (T*)malloc(src_size);
  T* output = (T*)malloc(dst_size);
  T* gpu_input0;
  T* gpu_input1;
  T* gpu_input2;
  T* gpu_input3;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input0, src_size);
  cudaMalloc((void**)&gpu_input1, src_size);
  cudaMalloc((void**)&gpu_input2, src_size);
  cudaMalloc((void**)&gpu_input3, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src0, input0);
  copyMatToArray(src1, input1);
  copyMatToArray(src2, input2);
  copyMatToArray(src3, input3);
  cudaMemcpy(gpu_input0, input0, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_input1, input1, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_input2, input2, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_input3, input3, src_size, cudaMemcpyHostToDevice);

  if (channels == 3) {
    cv::Mat srcs[3] = {src0, src1, src2};
    cv::merge(srcs, 3, cv_dst);
    ppl::cv::cuda::Merge3Channels<T>(0, gpu_src0.rows, gpu_src0.cols,
        gpu_src0.step / sizeof(T), (T*)gpu_src0.data, (T*)gpu_src1.data,
        (T*)gpu_src2.data, gpu_dst.step / sizeof(T), (T*)gpu_dst.data);
    ppl::cv::cuda::Merge3Channels<T>(0, size.height, size.width, size.width,
        gpu_input0, gpu_input1, gpu_input2, size.width * channels,
        gpu_output);
  }
  else {  // channels == 4
    cv::Mat srcs[4] = {src0, src1, src2, src3};
    cv::merge(srcs, 4, cv_dst);
    ppl::cv::cuda::Merge4Channels<T>(0, gpu_src0.rows, gpu_src0.cols,
        gpu_src0.step / sizeof(T), (T*)gpu_src0.data, (T*)gpu_src1.data,
        (T*)gpu_src2.data, (T*)gpu_src3.data, gpu_dst.step / sizeof(T),
        (T*)gpu_dst.data);
    ppl::cv::cuda::Merge4Channels<T>(0, size.height, size.width, size.width,
        gpu_input0, gpu_input1, gpu_input2, gpu_input3, size.width * channels,
        gpu_output);
  }
  gpu_dst.download(dst);
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

  free(input0);
  free(input1);
  free(input2);
  free(input3);
  free(output);
  cudaFree(gpu_input0);
  cudaFree(gpu_input1);
  cudaFree(gpu_input2);
  cudaFree(gpu_input3);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaMergeTest ## T ## channels =                                    \
        PplCvCudaMergeTest<T,  channels>;                                      \
TEST_P(PplCvCudaMergeTest ## T ## channels, Standard) {                        \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaMergeTest ## T ## channels,                                         \
  ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                    \
                    cv::Size{1283, 720}, cv::Size{1934, 1080},                 \
                    cv::Size{320, 240}, cv::Size{640, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1920, 1080}),                \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaMergeTest ## T ## channels::ParamType>& info) {                 \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar, 3)
UNITTEST(float, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 4)
