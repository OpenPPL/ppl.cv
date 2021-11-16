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

#include "ppl/cv/cuda/mean.h"

#include <tuple>
#include <sstream>

#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

#include "infrastructure.hpp"

using namespace ppl::cv;
using namespace ppl::cv::cuda;

enum MaskType {
  UNMASKED,
  MASKED,
};

using Parameters = std::tuple<MaskType, bool, cv::Size>;
inline std::string convertToStringMean(const Parameters& parameters) {
  std::ostringstream formatted;

  MaskType is_masked = std::get<0>(parameters);
  if (is_masked == UNMASKED) {
    formatted << "Unmasked" << "_";
  }
  else {
    formatted << "Masked" << "_";
  }

  bool channel_wise = std::get<1>(parameters);
  if (channel_wise) {
    formatted << "channel_wise" << "_";
  }
  else {
    formatted << "unchannel_wise" << "_";
  }

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaMeanTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaMeanTest() {
    const Parameters& parameters = GetParam();
    is_masked    = std::get<0>(parameters);
    channel_wise = std::get<1>(parameters);
    size         = std::get<2>(parameters);
  }

  ~PplCvCudaMeanTest() {
  }

  bool apply();

 private:
  MaskType is_masked;
  bool channel_wise;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaMeanTest<T, channels>::apply() {
  cv::Mat src, mask0;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask0 = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask0(mask0);
  int dst_size = channels * sizeof(float);
  float* dst = (float*)malloc(dst_size);
  float* gpu_dst;
  cudaMalloc((void**)&gpu_dst, dst_size);

  int src_size = size.height * size.width * channels * sizeof(T);
  int mask_size = size.height * size.width * sizeof(uchar);
  T* input = (T*)malloc(src_size);
  float* output = (float*)malloc(dst_size);
  uchar* mask1 = (uchar*)malloc(mask_size);
  T* gpu_input;
  float* gpu_output;
  uchar* gpu_mask1;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  cudaMalloc((void**)&gpu_mask1, mask_size);
  copyMatToArray(src, input);
  copyMatToArray(mask0, mask1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mask1, mask1, mask_size, cudaMemcpyHostToDevice);

  cv::Scalar mean_value;
  if (is_masked == UNMASKED) {
    mean_value = cv::mean(src);
    // Mean<T, channels>(0, gpu_src.rows, gpu_src.cols, gpu_src.step / sizeof(T),
    //                   (T*)gpu_src.data, gpu_dst, 0, nullptr, channel_wise);
    // cudaMemcpy(dst, gpu_dst, dst_size, cudaMemcpyDeviceToHost);

    Mean<T, channels>(0, size.height, size.width, size.width * channels,
                      gpu_input, gpu_output, 0, nullptr, channel_wise);
    cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);
  }
  else {
    mean_value = cv::mean(src, mask0);
    // Mean<T, channels>(0, gpu_src.rows, gpu_src.cols, gpu_src.step / sizeof(T),
    //                   (T*)gpu_src.data, gpu_dst, gpu_mask0.step / sizeof(uchar),
    //                   (uchar*)gpu_mask0.data, channel_wise);
    // cudaMemcpy(dst, gpu_dst, dst_size, cudaMemcpyDeviceToHost);

    Mean<T, channels>(0, size.height, size.width, size.width * channels,
                      gpu_input, gpu_output, size.width, gpu_mask1,
                      channel_wise);
    cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);
  }

  float epsilon = EPSILON_1F;
  bool identity = true;
  if (channel_wise) {
    for (int i = 0; i < channels; i++) {
      if (fabs(mean_value.val[i] - output[i]) > epsilon) {
        identity = false;
      }
    }
  }
  else {
    if (fabs(mean_value.val[0] - output[0]) > epsilon) {
      identity = false;
    }
  }

  free(dst);
  free(input);
  free(output);
  free(mask1);
  cudaFree(gpu_input);
  cudaFree(gpu_dst);
  cudaFree(gpu_output);
  cudaFree(gpu_mask1);

  return identity;
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaMeanTest ## T ## channels =  PplCvCudaMeanTest<T, channels>;    \
TEST_P(PplCvCudaMeanTest ## T ## channels, Standard) {                         \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaMeanTest ## T ## channels,           \
  ::testing::Combine(                                                          \
    ::testing::Values(UNMASKED, MASKED),                                       \
    ::testing::Values(true, false),                                            \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaMeanTest ## T ## channels::ParamType>& info) {                  \
    return convertToStringMean(info.param);                                    \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
