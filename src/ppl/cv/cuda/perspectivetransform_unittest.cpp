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

#include "ppl/cv/cuda/perspectivetransform.h"

#include <tuple>
#include <sstream>

#include "opencv2/core.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

using Parameters = std::tuple<cv::Size>;
inline std::string convertToStringPerspectiveTransform(const Parameters&
                                                       parameters) {
  std::ostringstream formatted;

  cv::Size size = std::get<0>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int srcCns, int dstCns>
class PplCvCudaPerspectiveTransformTest :
        public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaPerspectiveTransformTest() {
    const Parameters& parameters = GetParam();
    size = std::get<0>(parameters);
  }

  ~PplCvCudaPerspectiveTransformTest() {
  }

  bool apply();

 private:
  cv::Size size;
};

template <typename T, int srcCns, int dstCns>
bool PplCvCudaPerspectiveTransformTest<T, srcCns, dstCns>::apply() {
  cv::Mat src, trans_coeffs0;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, srcCns));
  trans_coeffs0 = createSourceImage(dstCns + 1, srcCns + 1,
                                    CV_MAKETYPE(cv::DataType<float>::depth, 1));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, dstCns));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, dstCns));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  int src_size = size.height * size.width * srcCns * sizeof(T);
  int dst_size = size.height * size.width * dstCns * sizeof(T);
  int coeff_size = (dstCns + 1) * (srcCns + 1) * sizeof(float);
  T* input  = (T*)malloc(src_size);
  T* output = (T*)malloc(dst_size);
  float* trans_coeff1 = (float*)malloc(coeff_size);
  T* gpu_input;
  T* gpu_output;
  cudaMalloc((void**)&gpu_input, src_size);
  cudaMalloc((void**)&gpu_output, dst_size);
  copyMatToArray(src, input);
  copyMatToArray(trans_coeffs0, trans_coeff1);
  cudaMemcpy(gpu_input, input, src_size, cudaMemcpyHostToDevice);

  cv::perspectiveTransform(src, cv_dst, trans_coeffs0);

  ppl::cv::cuda::PerspectiveTransform<T, srcCns, dstCns>(0, gpu_src.rows,
      gpu_src.cols, gpu_src.step / sizeof(T), (T*)gpu_src.data,
      gpu_dst.step / sizeof(T), (T*)gpu_dst.data, trans_coeff1);
  gpu_dst.download(dst);

  ppl::cv::cuda::PerspectiveTransform<T, srcCns, dstCns>(0, size.height,
      size.width, size.width * srcCns, gpu_input, size.width * dstCns,
      gpu_output, trans_coeff1);
  cudaMemcpy(output, gpu_output, dst_size, cudaMemcpyDeviceToHost);

  float epsilon = EPSILON_E4;
  bool identity0 = checkMatricesIdentity<T>(cv_dst, dst, epsilon);
  bool identity1 = checkMatArrayIdentity<T>(cv_dst, output, epsilon);

  free(input);
  free(output);
  free(trans_coeff1);
  cudaFree(gpu_input);
  cudaFree(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, srcCns, dstCns)                                            \
using PplCvCudaPerspectiveTransformTest ## srcCns ## dstCns =                  \
        PplCvCudaPerspectiveTransformTest<T, srcCns, dstCns>;                  \
TEST_P(PplCvCudaPerspectiveTransformTest ## srcCns ## dstCns, Standard) {      \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvCudaPerspectiveTransformTest ## srcCns ## dstCns,                       \
  ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                    \
                    cv::Size{1283, 720}, cv::Size{1934, 1080},                 \
                    cv::Size{320, 240}, cv::Size{640, 480},                    \
                    cv::Size{1280, 720}, cv::Size{1920, 1080}),                \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaPerspectiveTransformTest ## srcCns ## dstCns::ParamType>&       \
      info) {                                                                  \
    return convertToStringPerspectiveTransform(info.param);                    \
  }                                                                            \
);

UNITTEST(float, 2, 2)
UNITTEST(float, 2, 3)
UNITTEST(float, 3, 2)
UNITTEST(float, 3, 3)
