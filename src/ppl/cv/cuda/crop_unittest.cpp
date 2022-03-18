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

#include "ppl/cv/cuda/crop.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<int, int, int, cv::Size>;
inline std::string convertToStringCrop(const Parameters& parameters) {
  std::ostringstream formatted;

  int left = std::get<0>(parameters);
  formatted << "Left" << left << "_";

  int top = std::get<1>(parameters);
  formatted << "Top" << top << "_";

  int int_scale = std::get<2>(parameters);
  formatted << "IntScale" << int_scale << "_";

  cv::Size size = std::get<3>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaCropTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaCropTest() {
    const Parameters& parameters = GetParam();
    left  = std::get<0>(parameters);
    top   = std::get<1>(parameters);
    scale = std::get<2>(parameters) / 10.f;
    size  = std::get<3>(parameters);
  }

  ~PplCvCudaCropTest() {
  }

  bool apply();

 private:
  int left;
  int top;
  float scale;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaCropTest<T, channels>::apply() {
  int src_height = size.height * 2;
  int src_width  = size.width * 2;
  cv::Mat src;
  src = createSourceImage(src_height, src_width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = src_height * src_width * channels * sizeof(T);
  int dst_size = size.height * size.width * channels * sizeof(T);
  T* input  = (T*)malloc(src_size);
  T* output = (T*)malloc(dst_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  cv::Rect roi(left, top, size.width, size.height);
  cv::Mat croppedImage = src(roi);
  croppedImage.copyTo(cv_dst);
  cv_dst = cv_dst * scale;

  ppl::cv::cuda::Crop<T, channels>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(T), (T*)gpu_src.data, gpu_dst.rows, gpu_dst.cols,
      gpu_dst.step / sizeof(T), (T*)gpu_dst.data, left, top, scale);
  gpu_dst.download(dst);

  ppl::cv::cuda::Crop<T, channels>(0, src_height, src_width,
      src_width * channels, gpu_input, size.height, size.width,
      size.width * channels, gpu_output, left, top, scale);
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
using PplCvCudaCropTest ## T ## channels = PplCvCudaCropTest<T, channels>;     \
TEST_P(PplCvCudaCropTest ## T ## channels, Standard) {                         \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaCropTest ## T ## channels,           \
  ::testing::Combine(                                                          \
    ::testing::Values(0, 11, 187),                                             \
    ::testing::Values(0, 11, 187),                                             \
    ::testing::Values(0, 10, 15),                                              \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaCropTest ## T ## channels::ParamType>& info) {                  \
    return convertToStringCrop(info.param);                                    \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
