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

#include "ppl/cv/cuda/dilate.h"
#include "ppl/cv/cuda/erode.h"
#include "ppl/cv/cuda/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "utility/infrastructure.hpp"

enum Functions {
  kFullyMaskedDilate,
  kPartiallyMaskedDilate,
  kFullyMaskedErode,
  kPartiallyMaskedErode,
};

using Parameters = std::tuple<MemoryPool, Functions, ppl::cv::BorderType, int,
                              cv::Size>;
inline std::string convertToStringDilate(const Parameters& parameters) {
  std::ostringstream formatted;

  MemoryPool memory_pool = std::get<0>(parameters);
  if (memory_pool == kActivated) {
    formatted << "MemoryPoolUsed" << "_";
  }
  else {
    formatted << "MemoryPoolUnused" << "_";
  }

  Functions function = std::get<1>(parameters);
  if (function == kFullyMaskedDilate) {
    formatted << "FullyMaskedDilate" << "_";
  }
  else if (function == kPartiallyMaskedDilate) {
    formatted << "PartiallyMaskedDilate" << "_";
  }
  else if (function == kFullyMaskedErode) {
    formatted << "FullyMaskedErode" << "_";
  }
  else if (function == kPartiallyMaskedErode) {
    formatted << "PartiallyMaskedErode" << "_";
  }
  else {
  }

  ppl::cv::BorderType border_type = std::get<2>(parameters);
  if (border_type == ppl::cv::BORDER_DEFAULT) {
    formatted << "BORDER_DEFAULT" << "_";
  }
  else if (border_type == ppl::cv::BORDER_CONSTANT) {
    formatted << "BORDER_CONSTANT" << "_";
  }
  else {
  }

  int ksize = std::get<3>(parameters);
  formatted << "Ksize" << ksize << "_";

  cv::Size size = std::get<4>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvCudaDilateTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvCudaDilateTest() {
    const Parameters& parameters = GetParam();
    memory_pool = std::get<0>(parameters);
    function    = std::get<1>(parameters);
    border_type = std::get<2>(parameters);
    ksize       = std::get<3>(parameters);
    size        = std::get<4>(parameters);
  }

  ~PplCvCudaDilateTest() {
  }

  bool apply();

 private:
  MemoryPool memory_pool;
  Functions function;
  ppl::cv::BorderType border_type;
  int ksize;
  cv::Size size;
};

template <typename T, int channels>
bool PplCvCudaDilateTest<T, channels>::apply() {
  cv::Mat src;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::cuda::GpuMat gpu_src(src);
  cv::cuda::GpuMat gpu_dst(dst);

  cv::Size kSize(ksize, ksize);
  cv::Mat kernel0 = cv::getStructuringElement(cv::MORPH_RECT, kSize);
  cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, kSize);
  uchar* mask = (uchar*)malloc(ksize * ksize * sizeof(uchar));
  int index = 0;
  for (int i = 0; i < ksize; i++) {
    const uchar* data = kernel1.ptr<const uchar>(i);
    for (int j = 0; j < ksize; j++) {
      mask[index++] = data[j];
    }
  }

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == ppl::cv::BORDER_DEFAULT) {
    cv_border = cv::BORDER_DEFAULT;
  }
  else if (border_type == ppl::cv::BORDER_CONSTANT) {
    cv_border = cv::BORDER_CONSTANT;
  }
  else {
  }

  if (memory_pool == kActivated) {
    size_t width = size.width * channels * sizeof(T);
    size_t ceiled_volume = ppl::cv::cuda::ceil2DVolume(width, size.height);
    ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);
  }

  int constant_border;
  if (function == kFullyMaskedDilate) {
    constant_border = 253;
    cv::dilate(src, cv_dst, kernel0, cv::Point(-1, -1), 1, cv_border,
               constant_border);
    ppl::cv::cuda::Dilate<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, ksize, ksize, nullptr,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, border_type,
        constant_border);
  }
  else if (function == kPartiallyMaskedDilate) {
    constant_border = 253;
    cv::dilate(src, cv_dst, kernel1, cv::Point(-1, -1), 1, cv_border,
               constant_border);
    ppl::cv::cuda::Dilate<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, ksize, ksize, mask,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, border_type,
        constant_border);
  }
  else if (function == kFullyMaskedErode) {
    constant_border = 1;
    cv::erode(src, cv_dst, kernel0, cv::Point(-1, -1), 1, cv_border,
              constant_border);
    ppl::cv::cuda::Erode<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, ksize, ksize, nullptr,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, border_type,
        constant_border);
  }
  else {
    constant_border = 1;
    cv::erode(src, cv_dst, kernel1, cv::Point(-1, -1), 1, cv_border,
              constant_border);
    ppl::cv::cuda::Erode<T, channels>(0, gpu_src.rows, gpu_src.cols,
        gpu_src.step / sizeof(T), (T*)gpu_src.data, ksize, ksize, mask,
        gpu_dst.step / sizeof(T), (T*)gpu_dst.data, border_type,
        constant_border);
  }
  gpu_dst.download(dst);

  if (memory_pool == kActivated) {
    ppl::cv::cuda::shutDownGpuMemoryPool();
  }

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E6;
  }
  bool identity = checkMatricesIdentity<T>(cv_dst, dst, epsilon);

  free(mask);

  return identity;
}

#define UNITTEST(T, channels)                                                  \
using PplCvCudaDilateTest ## T ## channels = PplCvCudaDilateTest<T, channels>; \
TEST_P(PplCvCudaDilateTest ## T ## channels, Standard) {                       \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvCudaDilateTest ## T ## channels,         \
  ::testing::Combine(                                                          \
    ::testing::Values(kActivated, kUnactivated),                               \
    ::testing::Values(kFullyMaskedDilate, kPartiallyMaskedDilate,              \
                      kFullyMaskedErode, kPartiallyMaskedErode),               \
    ::testing::Values(ppl::cv::BORDER_DEFAULT, ppl::cv::BORDER_CONSTANT),      \
    ::testing::Values(1, 3, 5, 7, 11, 15),                                     \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1976, 1080},               \
                      cv::Size{320, 240}, cv::Size{640, 480},                  \
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvCudaDilateTest ## T ## channels::ParamType>& info) {                \
    return convertToStringDilate(info.param);                                  \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)
UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)
