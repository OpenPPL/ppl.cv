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

#include "ppl/cv/cuda/minmaxloc.h"

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

using Parameters = std::tuple<MaskType, cv::Size>;
inline std::string convertToString(const Parameters& parameters) {
  std::ostringstream formatted;

  MaskType is_masked = std::get<0>(parameters);
  if (is_masked == UNMASKED) {
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

template <typename T>
class PplCvCudaMinMaxLocTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaMinMaxLocTest() {
    const Parameters& parameters = GetParam();
    is_masked = std::get<0>(parameters);
    size      = std::get<1>(parameters);
  }

  ~PplCvCudaMinMaxLocTest() {
  }

  bool apply();

 private:
  MaskType is_masked;
  cv::Size size;
};

template <typename T>
bool PplCvCudaMinMaxLocTest<T>::apply() {
  cv::Mat src, mask;
  src  = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<T>::depth, 1));
  mask = createSourceImage(size.height, size.width,
                           CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_mask0(mask);

  int src_size = size.height * size.width * sizeof(T);
  int mask_size = size.height * size.width * sizeof(uchar);
  T* input = (T*)malloc(src_size);
  uchar* mask1 = (uchar*)malloc(mask_size);
  T* gpu_input;
  uchar* gpu_mask1;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_mask1, mask_size);
  copyMatToArray(src, input);
  copyMatToArray(mask, mask1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mask1, mask1, mask_size, cudaMemcpyHostToDevice);

  double min_value0, max_value0;
  cv::Point minLoc, maxLoc;
  T min_value1, min_value2;
  T max_value1, max_value2;
  int min_index_x, min_index_y, max_index_x, max_index_y;
  int min_index_x2, min_index_y2, max_index_x2, max_index_y2;
  if (is_masked == UNMASKED) {
    cv::minMaxLoc(src, &min_value0, &max_value0, &minLoc, &maxLoc);
    MinMaxLoc<T>(0, gpu_src.rows, gpu_src.cols, gpu_src.step / sizeof(T),
                 (T*)gpu_src.data, &min_value1, &max_value1, &min_index_x,
                 &min_index_y, &max_index_x, &max_index_y);
    MinMaxLoc<T>(0, size.height, size.width, size.width, gpu_input,
                 &min_value2, &max_value2, &min_index_x2, &min_index_y2,
                 &max_index_x2, &max_index_y2);
  }
  else {
    cv::minMaxLoc(src, &min_value0, &max_value0, &minLoc, &maxLoc, mask);
    MinMaxLoc<T>(0, gpu_src.rows, gpu_src.cols, gpu_src.step / sizeof(T),
                 (T*)gpu_src.data, &min_value1, &max_value1, &min_index_x,
                 &min_index_y, &max_index_x, &max_index_y,
                 gpu_mask0.step, gpu_mask0.data);
    MinMaxLoc<T>(0, size.height, size.width, size.width, gpu_input, &min_value2,
                 &max_value2, &min_index_x2, &min_index_y2, &max_index_x2,
                 &max_index_y2, size.width, gpu_mask1);
  }

  bool identity0 = false;
  if (fabs(min_value0 - min_value1) < EPSILON_E5 &&
      fabs(max_value0 - max_value1) < EPSILON_E5 &&
      minLoc.x == min_index_x && minLoc.y == min_index_y &&
      maxLoc.x == max_index_x && maxLoc.y == max_index_y) {
    identity0 = true;
  }

  bool identity1 = false;
  if (fabs(min_value0 - min_value2) < EPSILON_E5 &&
      fabs(max_value0 - max_value2) < EPSILON_E5 &&
      minLoc.x == min_index_x2 && minLoc.y == min_index_y2 &&
      maxLoc.x == max_index_x2 && maxLoc.y == max_index_y2) {
    identity1 = true;
  }

  free(input);
  free(mask1);
  cudaFree(gpu_input);
  cudaFree(gpu_mask1);

  return (identity0 && identity1);
}

#define UNITTEST(T)                                                            \
using PplCvCudaMinMaxLocTest ## T = PplCvCudaMinMaxLocTest<T>;                 \
TEST_P(PplCvCudaMinMaxLocTest ## T, Standard) {                                \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaMinMaxLocTest ## T,                  \
  ::testing::Combine(                                                          \
    ::testing::Values(UNMASKED, MASKED),                                       \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<PplCvCudaMinMaxLocTest ## T::ParamType>&     \
    info) {                                                                    \
    return convertToString(info.param);                                        \
  }                                                                            \
);

UNITTEST(uchar)
UNITTEST(float)
