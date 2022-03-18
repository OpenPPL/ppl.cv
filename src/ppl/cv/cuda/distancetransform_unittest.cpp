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

#include "ppl/cv/cuda/distancetransform.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<ppl::cv::DistTypes,
                              ppl::cv::DistanceTransformMasks, cv::Size>;
inline std::string convertToStringDistTransform(const Parameters& parameters) {
  std::ostringstream formatted;

  ppl::cv::DistTypes distance_type = std::get<0>(parameters);
  if (distance_type == ppl::cv::DIST_C) {
    formatted << "DIST_C" << "_";
  }
  else if (distance_type == ppl::cv::DIST_L1) {
    formatted << "DIST_L1" << "_";
  }
  else {
    formatted << "DIST_L2" << "_";
  }

  ppl::cv::DistanceTransformMasks mask_size = std::get<1>(parameters);
  if (mask_size == ppl::cv::DIST_MASK_3) {
    formatted << "MASK_3" << "_";
  }
  else if (mask_size == ppl::cv::DIST_MASK_5) {
    formatted << "MASK_5" << "_";
  }
  else {
    formatted << "MASK_PRECISE" << "_";
  }

  cv::Size size = std::get<2>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T>
class PplCvCudaDistanceTransformTest :
        public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaDistanceTransformTest() {
    const Parameters& parameters = GetParam();
    distance_type = std::get<0>(parameters);
    mask_size     = std::get<1>(parameters);
    size          = std::get<2>(parameters);
  }

  ~PplCvCudaDistanceTransformTest() {
  }

  bool apply();

 private:
  ppl::cv::DistTypes distance_type;
  ppl::cv::DistanceTransformMasks mask_size;
  cv::Size size;
};

template <typename T>
bool PplCvCudaDistanceTransformTest<T>::apply() {
  cv::Mat src;
  src = createBinaryImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  cv::Mat dst(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, 1));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * sizeof(uchar);
  int dst_size = size.height * size.width * sizeof(T);
  uchar* input = (uchar*)malloc(src_size);
  T* output = (T*)malloc(dst_size);
  uchar* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  if ((distance_type == ppl::cv::DIST_C || distance_type == ppl::cv::DIST_L1)
      && mask_size == ppl::cv::DIST_MASK_PRECISE) {
    mask_size = ppl::cv::DIST_MASK_3;
  }
  else if (distance_type == ppl::cv::DIST_L2) {
    mask_size = ppl::cv::DIST_MASK_PRECISE;
  }
  else {
  }

  cv::DistanceTypes cv_distance;
  if (distance_type == ppl::cv::DIST_C) {
    cv_distance = cv::DIST_C;
  }
  else if (distance_type == ppl::cv::DIST_L1) {
    cv_distance = cv::DIST_L1;
  }
  else {
    cv_distance = cv::DIST_L2;
  }
  cv::DistanceTransformMasks cv_mask;
  if (mask_size == ppl::cv::DIST_MASK_3) {
    cv_mask = cv::DIST_MASK_3;
  }
  else if (mask_size == ppl::cv::DIST_MASK_5) {
    cv_mask = cv::DIST_MASK_5;
  }
  else {
    cv_mask = cv::DIST_MASK_PRECISE;
  }
  cv::distanceTransform(src, cv_dst, cv_distance, cv_mask);

  ppl::cv::cuda::DistanceTransform<T>(0, gpu_src.rows, gpu_src.cols,
      gpu_src.step / sizeof(uchar), (uchar*)gpu_src.data,
      gpu_dst.step / sizeof(T), (T*)gpu_dst.data, distance_type, mask_size);
  gpu_dst.download(dst);

  ppl::cv::cuda::DistanceTransform<T>(0, size.height, size.width, size.width,
      gpu_input, size.width, gpu_output, distance_type, mask_size);
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

#define UNITTEST(T)                                                            \
using PplCvCudaDistanceTransformTest ## T = PplCvCudaDistanceTransformTest<T>; \
TEST_P(PplCvCudaDistanceTransformTest ## T, Standard) {                        \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaDistanceTransformTest ## T,          \
  ::testing::Combine(                                                          \
    ::testing::Values(ppl::cv::DIST_C, ppl::cv::DIST_L1, ppl::cv::DIST_L2),    \
    ::testing::Values(ppl::cv::DIST_MASK_3, ppl::cv::DIST_MASK_5,              \
                      ppl::cv::DIST_MASK_PRECISE),                             \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaDistanceTransformTest ## T::ParamType>& info) {                 \
    return convertToStringDistTransform(info.param);                           \
  }                                                                            \
);

UNITTEST(float)
