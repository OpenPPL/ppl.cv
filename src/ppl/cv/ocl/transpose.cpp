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

#include "ppl/cv/ocl/transpose.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/transpose.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode transposeC1U8(const cl_mem src, int rows, int cols, int src_stride,
                      cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= rows * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, transpose);

  int global_cols, global_rows;
  global_cols = divideUp(cols, 4, 2);
  global_rows = divideUp(rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D TRANSPOSE_U8C1");
  runOclKernel(frame_chain, "transposeU8C1Kernel", 2, global_size, local_size,
               src, rows, cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode transposeC3U8(const cl_mem src, int rows, int cols, int src_stride,
                      cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar) * 3);
  PPL_ASSERT(dst_stride >= rows * (int)sizeof(uchar) * 3);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, transpose);

  int global_cols, global_rows;
  global_cols = cols;
  global_rows = divideUp(rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D TRANSPOSE_U8C3");
  runOclKernel(frame_chain, "transposeU8C3Kernel", 2, global_size, local_size,
               src, rows, cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode transposeC4U8(const cl_mem src, int rows, int cols, int src_stride,
                      cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar) * 4);
  PPL_ASSERT(dst_stride >= rows * (int)sizeof(uchar) * 4);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, transpose);

  int global_cols, global_rows;
  global_cols = cols;
  global_rows = divideUp(rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D TRANSPOSE_U8C4");
  runOclKernel(frame_chain, "transposeU8C4Kernel", 2, global_size, local_size,
               src, rows, cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode transposeC1F32(const cl_mem src, int rows, int cols, int src_stride,
                       cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= rows * (int)sizeof(float));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, transpose);

  int global_cols, global_rows;
  global_cols = divideUp(cols, 2, 1);
  global_rows = divideUp(rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D TRANSPOSE_F32C1");
  runOclKernel(frame_chain, "transposeF32C1Kernel", 2, global_size, local_size,
               src, rows, cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode transposeC3F32(const cl_mem src, int rows, int cols, int src_stride,
                       cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float) * 3);
  PPL_ASSERT(dst_stride >= rows * (int)sizeof(float) * 3);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, transpose);

  int global_cols, global_rows;
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D TRANSPOSE_F32C3");
  runOclKernel(frame_chain, "transposeF32C3Kernel", 2, global_size, local_size,
               src, rows, cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode transposeC4F32(const cl_mem src, int rows, int cols, int src_stride,
                       cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float) * 4);
  PPL_ASSERT(dst_stride >= rows * (int)sizeof(float) * 4);

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, transpose);

  int global_cols, global_rows;
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D TRANSPOSE_F32C4");
  runOclKernel(frame_chain, "transposeF32C4Kernel", 2, global_size, local_size,
               src, rows, cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

template <>
RetCode Transpose<uchar, 1>(cl_command_queue queue,
                            int height,
                            int width,
                            int inWidthStride,
                            const cl_mem inData,
                            int outWidthStride,
                            cl_mem outData) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = transposeC1U8(inData, height, width, inWidthStride, outData,
                               outWidthStride, queue);
  return code;
}

template <>
RetCode Transpose<uchar, 3>(cl_command_queue queue,
                            int height,
                            int width,
                            int inWidthStride,
                            const cl_mem inData,
                            int outWidthStride,
                            cl_mem outData) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = transposeC3U8(inData, height, width, inWidthStride, outData,
                               outWidthStride, queue);
  return code;
}

template <>
RetCode Transpose<uchar, 4>(cl_command_queue queue,
                            int height,
                            int width,
                            int inWidthStride,
                            const cl_mem inData,
                            int outWidthStride,
                            cl_mem outData) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = transposeC4U8(inData, height, width, inWidthStride, outData,
                               outWidthStride, queue);
  return code;
}

template <>
RetCode Transpose<float, 1>(cl_command_queue queue,
                            int height,
                            int width,
                            int inWidthStride,
                            const cl_mem inData,
                            int outWidthStride,
                            cl_mem outData) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = transposeC1F32(inData, height, width, inWidthStride, outData,
                                outWidthStride, queue);
  return code;
}

template <>
RetCode Transpose<float, 3>(cl_command_queue queue,
                            int height,
                            int width,
                            int inWidthStride,
                            const cl_mem inData,
                            int outWidthStride,
                            cl_mem outData) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = transposeC3F32(inData, height, width, inWidthStride, outData,
                                outWidthStride, queue);
  return code;
}

template <>
RetCode Transpose<float, 4>(cl_command_queue queue,
                            int height,
                            int width,
                            int inWidthStride,
                            const cl_mem inData,
                            int outWidthStride,
                            cl_mem outData) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = transposeC4F32(inData, height, width, inWidthStride, outData,
                                outWidthStride, queue);
  return code;
}

} // namespace ocl
} // namespace cv
} // namespace ppl
