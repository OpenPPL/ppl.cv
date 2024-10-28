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

#include "ppl/cv/ocl/boxfilter.h"
#include "ppl/cv/ocl/use_memory_pool.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"
#include "kerneltypes.h"

using Parameters = std::tuple<int, int, bool, BorderType, cv::Size>;
inline std::string boxfilterToString(const Parameters& parameters) {
  std::ostringstream formatted;

  int ksize_x = std::get<0>(parameters);
  formatted << "Ksize_x" << ksize_x << "_";

  int ksize_y = std::get<1>(parameters);
  formatted << "Ksize_y" << ksize_y << "_";

  bool normalize = std::get<2>(parameters);
  formatted << "Normalize" << normalize << "_";

  BorderType border_type = std::get<3>(parameters);
  if (border_type == BORDER_REPLICATE) {
    formatted << "BORDER_REPLICATE" << "_";
  }
  else if (border_type == BORDER_REFLECT) {
    formatted << "BORDER_REFLECT" << "_";
  }
  else if (border_type == BORDER_REFLECT_101) {
    formatted << "BORDER_REFLECT_101" << "_";
  }
  else {  // border_type == BORDER_DEFAULT
    formatted << "BORDER_DEFAULT" << "_";
  }

  cv::Size size = std::get<4>(parameters);
  formatted << size.width << "x";
  formatted << size.height;

  return formatted.str();
}

template <typename T, int channels>
class PplCvOclBoxFilterToTest: public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclBoxFilterToTest() {
    const Parameters& parameters = GetParam();
    ksize_x     = std::get<0>(parameters);
    ksize_y     = std::get<1>(parameters);
    normalize   = std::get<2>(parameters);
    border_type = std::get<3>(parameters);
    size        = std::get<4>(parameters);

    ppl::common::ocl::createSharedFrameChain(false);
    context = ppl::common::ocl::getSharedFrameChain()->getContext();
    queue   = ppl::common::ocl::getSharedFrameChain()->getQueue();

    bool status = ppl::common::ocl::initializeKernelBinariesManager(
                      ppl::common::ocl::BINARIES_RETRIEVE);
    if (status) {
      ppl::common::ocl::FrameChain* frame_chain =
          ppl::common::ocl::getSharedFrameChain();
      frame_chain->setCreatingProgramType(ppl::common::ocl::WITH_BINARIES);
    }
  }

  ~PplCvOclBoxFilterToTest() {
  }

  bool apply();

 private:
  int ksize_x;
  int ksize_y;
  bool normalize;
  BorderType border_type;
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

template <typename T, int channels>
bool PplCvOclBoxFilterToTest<T, channels>::apply() {
  cv::Mat src, kernel0;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat dst(size.height, size.width,
              CV_MAKETYPE(cv::DataType<T>::depth, channels));
  cv::Mat cv_dst(size.height, size.width,
                 CV_MAKETYPE(cv::DataType<T>::depth, channels));

  int src_bytes0 = src.rows * src.step;
  int dst_bytes0 = dst.rows * dst.step;

  cl_int error_code = 0;
  cl_mem gpu_src =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_dst =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes0,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int src_bytes1 = size.height * size.width * channels * sizeof(T);
  int dst_bytes1 = (size.height) * (size.width) * channels * sizeof(T);
  cl_mem gpu_input =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  cl_mem gpu_output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  size_t size_width = size.width * channels * sizeof(float);
  size_t ceiled_volume = ppl::cv::ocl::ceil2DVolume(size_width, size.height);
  ppl::cv::ocl::activateGpuMemoryPool(ceiled_volume);

  T* input = (T*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE, 0,
                                    src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  cv::BorderTypes cv_border = cv::BORDER_DEFAULT;
  if (border_type == BORDER_REPLICATE) {
    cv_border = cv::BORDER_REPLICATE;
  }
  else if (border_type == BORDER_REFLECT) {
    cv_border = cv::BORDER_REFLECT;
  }
  else if (border_type == BORDER_REFLECT_101) {
    cv_border = cv::BORDER_REFLECT_101;
  }
  else {
  }
  cv::boxFilter(src, cv_dst, cv_dst.depth(), cv::Size(ksize_x, ksize_y),
                cv::Point(-1, -1), normalize, cv_border);

  ppl::cv::ocl::BoxFilter<T, channels>(
      queue, src.rows, src.cols, src.step / sizeof(T), gpu_src, ksize_x,
      ksize_y, normalize, dst.step / sizeof(T), gpu_dst, border_type);
  ppl::cv::ocl::BoxFilter<T, channels>(
      queue, size.height, size.width, size.width * channels, gpu_input, ksize_x,
      ksize_y, normalize, size.width * channels, gpu_output, border_type);
  error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes0,
                                   dst.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueReadBuffer);
  T* output = (T*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ, 0,
                                     dst_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  float epsilon;
  if (sizeof(T) == 1) {
    epsilon = EPSILON_1F;
  }
  else {
    epsilon = EPSILON_E3;
  }

  bool identity0 = checkMatricesIdentity<T>(
      (const T*)cv_dst.data, cv_dst.rows, cv_dst.cols, cv_dst.channels(),
      cv_dst.step, (const T*)dst.data, dst.step, epsilon);
  bool identity1 = checkMatricesIdentity<T>(
      (const T*)cv_dst.data, cv_dst.rows, cv_dst.cols, cv_dst.channels(),
      cv_dst.step, output, dst.cols * channels * sizeof(T), epsilon);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  ppl::cv::ocl::shutDownGpuMemoryPool();
  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

#define UNITTEST(T, channels)                                                  \
using PplCvOclBoxFilterToTest ## T ## channels =                               \
        PplCvOclBoxFilterToTest<T, channels>;                                  \
TEST_P(PplCvOclBoxFilterToTest ## T ## channels, Standard) {                   \
  bool identity = this->apply();                                               \
  EXPECT_TRUE(identity);                                                       \
}                                                                              \
                                                                               \
INSTANTIATE_TEST_CASE_P(IsEqual,                                               \
  PplCvOclBoxFilterToTest ## T ## channels,                                    \
  ::testing::Combine(                                                          \
    ::testing::Values(1, 5),                                                   \
    ::testing::Values(1, 4),                                                   \
    ::testing::Values(true, false),                                            \
    ::testing::Values(BORDER_REPLICATE, BORDER_REFLECT,                        \
                      BORDER_REFLECT_101),                                     \
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  \
                      cv::Size{1283, 720}, cv::Size{1934, 1080})),             \
  [](const testing::TestParamInfo<                                             \
      PplCvOclBoxFilterToTest ## T ## channels::ParamType>&                    \
        info) {                                                                \
    return boxfilterToString(info.param);                                      \
  }                                                                            \
);

UNITTEST(uchar, 1)
UNITTEST(uchar, 3)
UNITTEST(uchar, 4)

UNITTEST(float, 1)
UNITTEST(float, 3)
UNITTEST(float, 4)