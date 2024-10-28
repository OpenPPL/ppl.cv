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

#include "ppl/cv/ocl/crop.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/crop.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode cropU8(const cl_mem src, int src_rows, int src_cols, int channels,
               int src_stride, cl_mem dst, int dst_rows, int dst_cols,
               int dst_stride, int left, int top, float scale,
               cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(src_rows >= dst_rows && src_cols >= dst_cols);
  PPL_ASSERT(left >= 0 && left < src_cols - dst_cols);
  PPL_ASSERT(top >= 0 && top < src_rows - dst_rows);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(uchar));
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, crop);
  int columns = dst_cols * channels;
  dst_cols = divideUp(columns, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};
  frame_chain->setCompileOptions("-D CROP_U8");
  runOclKernel(frame_chain, "cropU8Kernel", 2, global_size, local_size, src,
               src_stride, top, left * channels, scale, dst, dst_rows, columns,
               dst_stride);
  return RC_SUCCESS;
}

RetCode cropF32(const cl_mem src, int src_rows, int src_cols, int channels,
                int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                int dst_stride, int left, int top, float scale,
                cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(src_rows >= dst_rows && src_cols >= dst_cols);
  PPL_ASSERT(left >= 0 && left < src_cols - dst_cols);
  PPL_ASSERT(top >= 0 && top < src_rows - dst_rows);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(float));
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, crop);
  int columns = dst_cols * channels;
  dst_cols = divideUp(columns, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)dst_cols, (size_t)dst_rows};
  frame_chain->setCompileOptions("-D CROP_F32");
  runOclKernel(frame_chain, "cropF32Kernel", 2, global_size, local_size, src,
               src_stride, top, left * channels, scale, dst, dst_rows, columns,
               dst_stride);
  return RC_SUCCESS;
}

template <>
RetCode Crop<uchar, 1>(cl_command_queue queue,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const cl_mem inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       cl_mem outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      cropU8(inData, inHeight, inWidth, 1, inWidthStride, outData, outHeight,
             outWidth, outWidthStride, left, top, scale, queue);
  return code;
}

template <>
RetCode Crop<uchar, 3>(cl_command_queue queue,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const cl_mem inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       cl_mem outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      cropU8(inData, inHeight, inWidth, 3, inWidthStride, outData, outHeight,
             outWidth, outWidthStride, left, top, scale, queue);
  return code;
}

template <>
RetCode Crop<uchar, 4>(cl_command_queue queue,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const cl_mem inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       cl_mem outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      cropU8(inData, inHeight, inWidth, 4, inWidthStride, outData, outHeight,
             outWidth, outWidthStride, left, top, scale, queue);
  return code;
}

template <>
RetCode Crop<float, 1>(cl_command_queue queue,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const cl_mem inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       cl_mem outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      cropF32(inData, inHeight, inWidth, 1, inWidthStride, outData, outHeight,
              outWidth, outWidthStride, left, top, scale, queue);
  return code;
}

template <>
RetCode Crop<float, 3>(cl_command_queue queue,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const cl_mem inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       cl_mem outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      cropF32(inData, inHeight, inWidth, 3, inWidthStride, outData, outHeight,
              outWidth, outWidthStride, left, top, scale, queue);
  return code;
}

template <>
RetCode Crop<float, 4>(cl_command_queue queue,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const cl_mem inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       cl_mem outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      cropF32(inData, inHeight, inWidth, 4, inWidthStride, outData, outHeight,
              outWidth, outWidthStride, left, top, scale, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
