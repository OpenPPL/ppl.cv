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

#include "ppl/cv/cuda/calchist.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<MaskType, cv::Size>;
inline std::string convertToStringHist(const Parameters& parameters) {
  std::ostringstream formatted;

  MaskType is_masked = std::get<0>(parameters);
  if (is_masked == kUnmasked) {
    formatted << "Unmasked" << "_";
  }
  else {
    formatted << "Masked" << "_";
  }

  cv::Size size = std::get<1>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaCalcHistTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaCalcHistTest() {
    const Parameters& parameters = GetParam();
    is_masked = std::get<0>(parameters);
    size      = std::get<1>(parameters);
  }

  ~PplCvCudaCalcHistTest() {
  }

  bool apply();

 private:
  MaskType is_masked;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaCalcHistTest<T, channels>::apply() {
  cv::Mat src, mask0, dst, cv_dst0, cv_dst1;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  mask0 = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  dst = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));
  cv_dst0 = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask0(mask0);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * channels * sizeof(T);
  int mask_size = size.height * size.width * sizeof(uchar);
  int dst_size = 256 * sizeof(int);
  T* input = (T*)malloc(src_size);
  uchar* mask1 = (uchar*)malloc(mask_size);
  int* output = (int*)malloc(dst_size);
  T* gpu_input;
  uchar* gpu_mask1;
  int* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_mask1, mask_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  copyMatToArray(mask0, mask1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mask1, mask1, mask_size, cudaMemcpyHostToDevice);

  int channel[] = {0};
  int hist_size[] = {256};
  float data_range[2] = {0, 256};
  const float* ranges[1] = {data_range};

  if (is_masked == kUnmasked) {
    cv::calcHist(&src, 1, channel, cv::Mat(), cv_dst1, 1, hist_size, ranges,
                 true, false);
    ppl::cv::cuda::CalcHist<T>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, (int*)gpu_dst.data);
    gpu_dst.download(dst);

    ppl::cv::cuda::CalcHist<T>(0, size.height, size.width,
        size.width * channels, gpu_input, gpu_output);
    cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);
  }
  else {
    cv::calcHist(&src, 1, channel, mask0, cv_dst1, 1, hist_size, ranges, true,
                 false);
    ppl::cv::cuda::CalcHist<T>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, (int*)gpu_dst.data,
        gpu_mask0.step / sizeof(uchar), (uchar*)gpu_mask0.data);
    gpu_dst.download(dst);

    ppl::cv::cuda::CalcHist<T>(0, size.height, size.width,
        size.width * channels, gpu_input, gpu_output, size.width, gpu_mask1);
    cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);
  }
  cv::Mat temp_mat;
  cv_dst1.convertTo(temp_mat, CV_32S);
  cv_dst0 = temp_mat.reshape(0, 1);

  float epsilon = EPSILON_1F;
  bool identity0 = checkMatricesIdentity<int>(cv_dst0, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<int>(cv_dst0, output, epsilon);

  free(input);
  free(output);
  free(mask1);
  cudaFree(gpu_input);
  cudaFree(gpu_output);
  cudaFree(gpu_mask1);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaCalcHistTest ## T ## channels =                                 \
        PplCvCudaCalcHistTest<T, channels>;                                    \
TEST_P(PplCvCudaCalcHistTest ## T ## channels, Standard) {                     \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaCalcHistTest ## T ## channels,       \
  ::testing::Combine(                                                          \
    ::testing::Values(kUnmasked, kMasked),                                     \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaCalcHistTest ## T ## channels::ParamType>& info) {              \
    return convertToStringHist(info.param);                                    \
  }                                                                            \
);

UNITTEST(uchar, 1)
