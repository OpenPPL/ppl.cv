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

#include "ppl/cv/ocl/merge.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/merge.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode merge3U8(const cl_mem src0, const cl_mem src1, const cl_mem src2,
                 int rows, int cols, int src_stride, cl_mem dst, int dst_stride,
                 cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * 3 * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, merge);

  int columns = cols;
  cols = columns;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};
  if (src_stride == columns * (int)sizeof(uchar) &&
      dst_stride == 3 * columns * (int)sizeof(uchar)) {
    columns *= rows;
    cols = columns;
    local_size[0] = 512;
    local_size[1] = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D MERGE3_U81D");
    runOclKernel(frame_chain, "merge3U8Kernel0", 2, global_size, local_size,
                 src0, src1, src2, columns, dst);
  }
  else {
    frame_chain->setCompileOptions("-D MERGE3_U82D");
    runOclKernel(frame_chain, "merge3U8Kernel1", 2, global_size, local_size,
                 src0, src1, src2, rows, columns, src_stride, dst, dst_stride);
  }
  return RC_SUCCESS;
}

RetCode merge3F32(const cl_mem src0, const cl_mem src1, const cl_mem src2,
                  int rows, int cols, int src_stride, cl_mem dst,
                  int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 3 * (int)sizeof(float));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, merge);

  int columns = cols;
  cols = columns;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};
  if (src_stride == columns * (int)sizeof(float) &&
      dst_stride == 3 * columns * (int)sizeof(float)) {
    columns *= rows;
    cols = columns;
    local_size[0] = 512;
    local_size[1] = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D MERGE3_F321D");
    runOclKernel(frame_chain, "merge3F32Kernel0", 2, global_size, local_size,
                 src0, src1, src2, columns, dst);
  }
  else {
    frame_chain->setCompileOptions("-D MERGE3_F322D");
    runOclKernel(frame_chain, "merge3F32Kernel1", 2, global_size, local_size,
                 src0, src1, src2, rows, columns, src_stride, dst, dst_stride);
  }
  return RC_SUCCESS;
}

RetCode merge4U8(const cl_mem src0, const cl_mem src1, const cl_mem src2,
                 const cl_mem src3, int rows, int cols, int src_stride,
                 cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(src3 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * 4 * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, merge);

  int columns = cols;
  cols = columns;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};
  if (src_stride == columns * (int)sizeof(uchar) &&
      dst_stride == 4 * columns * (int)sizeof(uchar)) {
    columns *= rows;
    cols = columns;
    local_size[0] = 512;
    local_size[1] = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D MERGE4_U81D");
    runOclKernel(frame_chain, "merge4U8Kernel0", 2, global_size, local_size,
                 src0, src1, src2, src3, columns, dst);
  }
  else {
    frame_chain->setCompileOptions("-D MERGE4_U82D");
    runOclKernel(frame_chain, "merge4U8Kernel1", 2, global_size, local_size,
                 src0, src1, src2, src3, rows, columns, src_stride, dst,
                 dst_stride);
  }
  return RC_SUCCESS;
}

RetCode merge4F32(const cl_mem src0, const cl_mem src1, const cl_mem src2,
                  const cl_mem src3, int rows, int cols, int src_stride,
                  cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(src3 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 4 * (int)sizeof(float));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, merge);
  
  int columns = cols;
  cols = columns;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)cols, (size_t)rows};
  if (src_stride == columns * (int)sizeof(float) &&
      dst_stride == 4 * columns * (int)sizeof(float)) {
    columns *= rows;
    cols = columns;
    local_size[0] = 512;
    local_size[1] = 1;
    global_size[0] = (size_t)roundUp(cols, 512, 9);
    global_size[1] = 1;
    frame_chain->setCompileOptions("-D MERGE4_F321D");
    runOclKernel(frame_chain, "merge4F32Kernel0", 2, global_size, local_size,
                 src0, src1, src2, src3, columns, dst);
  }
  else {
    frame_chain->setCompileOptions("-D MERGE4_F322D");
    runOclKernel(frame_chain, "merge4F32Kernel1", 2, global_size, local_size,
                 src0, src1, src2, src3, rows, columns, src_stride, dst,
                 dst_stride);
  }
  return RC_SUCCESS;
}

template <>
RetCode Merge3Channels<uchar>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData0,
                              const cl_mem inData1,
                              const cl_mem inData2,
                              int outWidthStride,
                              cl_mem outData) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = merge3U8(inData0, inData1, inData2, height, width,
                          inWidthStride, outData, outWidthStride, queue);
  return code;
}

template <>
RetCode Merge3Channels<float>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData0,
                              const cl_mem inData1,
                              const cl_mem inData2,
                              int outWidthStride,
                              cl_mem outData) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = merge3F32(inData0, inData1, inData2, height, width,
                          inWidthStride, outData, outWidthStride, queue);
  return code;
}

template <>
RetCode Merge4Channels<uchar>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData0,
                              const cl_mem inData1,
                              const cl_mem inData2,
                              const cl_mem inData3,
                              int outWidthStride,
                              cl_mem outData) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = merge4U8(inData0, inData1, inData2, inData3, height, width,
                          inWidthStride, outData, outWidthStride, queue);
  return code;
}

template <>
RetCode Merge4Channels<float>(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData0,
                              const cl_mem inData1,
                              const cl_mem inData2,
                              const cl_mem inData3,
                              int outWidthStride,
                              cl_mem outData) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = merge4F32(inData0, inData1, inData2, inData3, height, width,
                           inWidthStride, outData, outWidthStride, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
