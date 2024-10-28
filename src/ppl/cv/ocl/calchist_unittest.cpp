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

#include "ppl/cv/ocl/calchist.h"

#include <tuple>
#include <sstream>

#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/infrastructure.h"

using Parameters = std::tuple<MaskType, cv::Size>;
inline std::string convertToStringCalchist(const Parameters& parameters) {
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

class PplCvOclCalchistTest : public ::testing::TestWithParam<Parameters> {
 public:
  PplCvOclCalchistTest() {
    const Parameters& parameters = GetParam();
    is_masked = std::get<0>(parameters);
    size      = std::get<1>(parameters);

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

  ~PplCvOclCalchistTest() {
    ppl::common::ocl::shutDownKernelBinariesManager(
        ppl::common::ocl::BINARIES_RETRIEVE);
  }

  bool apply();

 private:
  MaskType is_masked;
  cv::Size size;
  cl_context context;
  cl_command_queue queue;
};

bool PplCvOclCalchistTest::apply() {
  cv::Mat src, mask0, dst, cv_dst, cv_dst1;
  src = createSourceImage(size.height, size.width,
                          CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  mask0 = createSourceImage(size.height, size.width,
                            CV_MAKETYPE(cv::DataType<uchar>::depth, 1));
  dst = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));
  cv_dst = cv::Mat::zeros(1, 256, CV_MAKETYPE(cv::DataType<int>::depth, 1));

  int src_bytes0 = src.rows * src.step;
  int mask_bytes0 = mask0.rows * mask0.step;
  int dst_bytes0 = dst.rows * dst.step;

  cl_int error_code = 0;
  cl_mem gpu_src =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     src_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_mask =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     mask_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_dst =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     dst_bytes0, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_src, CL_FALSE, 0, src_bytes0,
                                    src.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);
  error_code = clEnqueueWriteBuffer(queue, gpu_mask, CL_FALSE, 0, mask_bytes0,
                                    mask0.data, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueWriteBuffer);

  int src_bytes1 = size.height * size.width * sizeof(uchar);
  int mask_bytes1 = size.height * size.width * sizeof(uchar);
  int dst_bytes1 = 256 * sizeof(int);

  cl_mem gpu_input =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     src_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_input_mask =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     mask_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem gpu_output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     dst_bytes1, NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);

  uchar* input =
      (uchar*)clEnqueueMapBuffer(queue, gpu_input, CL_TRUE, CL_MAP_WRITE, 0,
                                 src_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);
  uchar* input_mask =
      (uchar*)clEnqueueMapBuffer(queue, gpu_input_mask, CL_TRUE, CL_MAP_WRITE,
                                 0, mask_bytes1, 0, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, clEnqueueMapBuffer);

  copyMatToArray(src, input);
  error_code = clEnqueueUnmapMemObject(queue, gpu_input, input, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);
  copyMatToArray(mask0, input_mask);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_input_mask, input_mask, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  int channel[] = {0};
  int hist_size[] = {256};
  float data_range[2] = {0, 256};
  const float* ranges[1] = {data_range};

  int* output;

  if (is_masked == kUnmasked) {
    cv::calcHist(&src, 1, channel, cv::Mat(), cv_dst1, 1, hist_size, ranges,
                 true, false);
    ppl::cv::ocl::CalcHist(queue, src.rows, src.cols, src.step / sizeof(uchar),
                           gpu_src, gpu_dst, mask0.step / sizeof(uchar),
                           nullptr);
    error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes0,
                                     dst.data, 0, NULL, NULL);
    CHECK_ERROR(error_code, clEnqueueReadBuffer);

    ppl::cv::ocl::CalcHist(queue, size.height, size.width, size.width,
                           gpu_input, gpu_output, size.width, nullptr);
    output =
        (int*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ, 0,
                                 dst_bytes1, 0, NULL, NULL, &error_code);
    CHECK_ERROR(error_code, clEnqueueMapBuffer);
  }
  else {
    cv::calcHist(&src, 1, channel, mask0, cv_dst1, 1, hist_size, ranges, true,
                 false);

    ppl::cv::ocl::CalcHist(queue, src.rows, src.cols, src.step / sizeof(uchar),
                           gpu_src, gpu_dst, mask0.step / sizeof(uchar),
                           gpu_mask);
    error_code = clEnqueueReadBuffer(queue, gpu_dst, CL_TRUE, 0, dst_bytes0,
                                     dst.data, 0, NULL, NULL);
    CHECK_ERROR(error_code, clEnqueueReadBuffer);

    ppl::cv::ocl::CalcHist(queue, size.height, size.width, size.width,
                           gpu_input, gpu_output, size.width, gpu_input_mask);
    output =
        (int*)clEnqueueMapBuffer(queue, gpu_output, CL_TRUE, CL_MAP_READ, 0,
                                 dst_bytes1, 0, NULL, NULL, &error_code);
    CHECK_ERROR(error_code, clEnqueueMapBuffer);
  }

  cv::Mat temp_mat;
  cv_dst1.convertTo(temp_mat, CV_32S);
  cv_dst = temp_mat.reshape(0, 1);

  float epsilon;
  epsilon = 0.1f;

  bool identity0 = checkMatricesIdentity<int>(
      (const int*)cv_dst.data, cv_dst.rows, cv_dst.cols, cv_dst.channels(),
      cv_dst.step, (const int*)dst.data, dst.step, epsilon);
  bool identity1 = checkMatricesIdentity<int>(
      (const int*)cv_dst.data, cv_dst.rows, cv_dst.cols, cv_dst.channels(),
      cv_dst.step, output, 256 * sizeof(int), epsilon);
  error_code =
      clEnqueueUnmapMemObject(queue, gpu_output, output, 0, NULL, NULL);
  CHECK_ERROR(error_code, clEnqueueUnmapMemObject);

  clReleaseMemObject(gpu_src);
  clReleaseMemObject(gpu_mask);
  clReleaseMemObject(gpu_dst);
  clReleaseMemObject(gpu_input);
  clReleaseMemObject(gpu_input_mask);
  clReleaseMemObject(gpu_output);

  return (identity0 && identity1);
}

using PplCvOclCalchistTest = PplCvOclCalchistTest;       
TEST_P(PplCvOclCalchistTest, Standard) {                           
  bool identity = this->apply();                                                
  EXPECT_TRUE(identity);                                                        
}                                                                               
                                                                                
INSTANTIATE_TEST_CASE_P(IsEqual, PplCvOclCalchistTest,             
  ::testing::Combine(                                                           
    ::testing::Values(kUnmasked, kMasked),                                     
    ::testing::Values(cv::Size{321, 240}, cv::Size{642, 480},                  
                      cv::Size{1283, 720}, cv::Size{1934, 1080},               
                      cv::Size{320, 240}, cv::Size{640, 480},                  
                      cv::Size{1280, 720}, cv::Size{1920, 1080})),             
  [](const testing::TestParamInfo<                                              
      PplCvOclCalchistTest::ParamType>& info) {                    
    return convertToStringCalchist(info.param);                                     
  }                                                                             
);

