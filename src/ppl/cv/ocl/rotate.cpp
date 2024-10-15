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

#include "ppl/cv/ocl/rotate.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/rotate.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode rotateC190U8(const cl_mem src, int src_rows, int src_cols,
                     int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                     int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = divideUp(src_cols, 4, 2);
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE90_U8C1");
  runOclKernel(frame_chain, "rotateC190U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC1180U8(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = divideUp(src_cols, 4, 2);
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE180_U8C1");
  runOclKernel(frame_chain, "rotateC1180U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC1270U8(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = divideUp(src_cols, 4, 2);
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE270_U8C1");
  runOclKernel(frame_chain, "rotateC1270U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC190F32(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = divideUp(src_cols, 2, 1);
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE90_F32C1");
  runOclKernel(frame_chain, "rotateC190F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC1180F32(const cl_mem src, int src_rows, int src_cols,
                       int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                       int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = divideUp(src_cols, 2, 1);
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE180_F32C1");
  runOclKernel(frame_chain, "rotateC1180F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC1270F32(const cl_mem src, int src_rows, int src_cols,
                       int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                       int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = divideUp(src_cols, 2, 1);
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE270_F32C1");
  runOclKernel(frame_chain, "rotateC1270F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC390U8(const cl_mem src, int src_rows, int src_cols,
                     int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                     int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE90_U8C3");
  runOclKernel(frame_chain, "rotateC390U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC3180U8(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE180_U8C3");
  runOclKernel(frame_chain, "rotateC3180U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC3270U8(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE270_U8C3");
  runOclKernel(frame_chain, "rotateC3270U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC390F32(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE90_F32C3");
  runOclKernel(frame_chain, "rotateC390F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC3180F32(const cl_mem src, int src_rows, int src_cols,
                       int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                       int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE180_F32C3");
  runOclKernel(frame_chain, "rotateC3180F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC3270F32(const cl_mem src, int src_rows, int src_cols,
                       int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                       int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE270_F32C3");
  runOclKernel(frame_chain, "rotateC3270F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC490U8(const cl_mem src, int src_rows, int src_cols,
                     int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                     int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE90_U8C4");
  runOclKernel(frame_chain, "rotateC490U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC4180U8(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE180_U8C4");
  runOclKernel(frame_chain, "rotateC4180U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC4270U8(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(uchar));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 4, 2);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE270_U8C4");
  runOclKernel(frame_chain, "rotateC4270U8Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC490F32(const cl_mem src, int src_rows, int src_cols,
                      int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                      int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE90_F32C4");
  runOclKernel(frame_chain, "rotateC490F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC4180F32(const cl_mem src, int src_rows, int src_cols,
                       int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                       int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE180_F32C4");
  runOclKernel(frame_chain, "rotateC4180F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

RetCode rotateC4270F32(const cl_mem src, int src_rows, int src_cols,
                       int src_stride, cl_mem dst, int dst_rows, int dst_cols,
                       int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(src_stride >= src_cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= src_rows * (int)sizeof(float));
  PPL_ASSERT((src_rows == dst_rows && src_cols == dst_cols) ||
             (src_rows == dst_cols && src_cols == dst_rows));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, rotate);

  int global_cols, global_rows;
  global_cols = src_cols;
  global_rows = divideUp(src_rows, 2, 1);
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)global_cols, (size_t)global_rows};
  frame_chain->setCompileOptions("-D ROTATE270_F32C4");
  runOclKernel(frame_chain, "rotateC4270F32Kernel", 2, global_size, local_size,
               src, src_rows, src_cols, src_stride, dst, dst_stride);
  return RC_SUCCESS;
}

template <>
RetCode Rotate<uchar, 1>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         int degree) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);
  RetCode code;
  if (degree == 90)
    code = rotateC190U8(inData, inHeight, inWidth, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, queue);
  else if (degree == 180)
    code = rotateC1180U8(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  else if (degree == 270)
    code = rotateC1270U8(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  return code;
}

template <>
RetCode Rotate<uchar, 3>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         int degree) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);
  RetCode code;
  if (degree == 90)
    code = rotateC390U8(inData, inHeight, inWidth, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, queue);
  else if (degree == 180)
    code = rotateC3180U8(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  else if (degree == 270)
    code = rotateC3270U8(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  return code;
}

template <>
RetCode Rotate<uchar, 4>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         int degree) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);
  RetCode code;
  if (degree == 90)
    code = rotateC490U8(inData, inHeight, inWidth, inWidthStride, outData,
                        outHeight, outWidth, outWidthStride, queue);
  else if (degree == 180)
    code = rotateC4180U8(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  else if (degree == 270)
    code = rotateC4270U8(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  return code;
}

template <>
RetCode Rotate<float, 1>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         int degree) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);
  RetCode code;
  if (degree == 90)
    code = rotateC190F32(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  else if (degree == 180)
    code = rotateC1180F32(inData, inHeight, inWidth, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, queue);
  else if (degree == 270)
    code = rotateC1270F32(inData, inHeight, inWidth, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, queue);
  return code;
}

template <>
RetCode Rotate<float, 3>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         int degree) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);
  RetCode code;
  if (degree == 90)
    code = rotateC390F32(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  else if (degree == 180)
    code = rotateC3180F32(inData, inHeight, inWidth, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, queue);
  else if (degree == 270)
    code = rotateC3270F32(inData, inHeight, inWidth, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, queue);
  return code;
}

template <>
RetCode Rotate<float, 4>(cl_command_queue queue,
                         int inHeight,
                         int inWidth,
                         int inWidthStride,
                         const cl_mem inData,
                         int outHeight,
                         int outWidth,
                         int outWidthStride,
                         cl_mem outData,
                         int degree) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  PPL_ASSERT(degree == 90 || degree == 180 || degree == 270);
  RetCode code;
  if (degree == 90)
    code = rotateC490F32(inData, inHeight, inWidth, inWidthStride, outData,
                         outHeight, outWidth, outWidthStride, queue);
  else if (degree == 180)
    code = rotateC4180F32(inData, inHeight, inWidth, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, queue);
  else if (degree == 270)
    code = rotateC4270F32(inData, inHeight, inWidth, inWidthStride, outData,
                          outHeight, outWidth, outWidthStride, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
