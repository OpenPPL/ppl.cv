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

#include "ppl/cv/ocl/cvtcolor.h"

// #include <iostream>  // debug

#include "ppl/common/ocl/oclcommon.h"
#include "utility/utility.hpp"

#include "kernels/cvtcolor.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

/***************** BGR/RBB/BGRA/RGBA <-> BGR/RBB/BGRA/RGBA ******************/

#define CVT_COLOR_BGR_INVOCATION(function, src_channels, dst_channels)         \
RetCode function ## _U8(const cl_mem src, int rows, int cols, int src_stride,  \
                    cl_mem dst, int dst_stride, cl_command_queue queue) {      \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));          \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain);                                             \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  if (src_stride == cols * src_channels * (int)sizeof(uchar) &&                \
      dst_stride == cols * dst_channels * (int)sizeof(uchar)) {                \
    int columns = cols * rows;                                                 \
    local_size[0]  = 512;                                                      \
    local_size[1]  = 1;                                                        \
    global_size[0] = (size_t)roundUp(columns, 512, 9);                         \
    global_size[1] = 1;                                                        \
    frame_chain->setCompileOptions("-D U8 -D " #function "_U8_1D");            \
    runOclKernel(frame_chain, #function "U8Kernel0", 2, global_size,           \
                 local_size, src, columns, dst);                               \
  }                                                                            \
  else {                                                                       \
    frame_chain->setCompileOptions("-D U8 -D " #function "_U8_2D");            \
    runOclKernel(frame_chain, #function "U8Kernel1", 2, global_size,           \
                 local_size, src, rows, cols, src_stride, dst, dst_stride);    \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
RetCode function ## _F32(const cl_mem src, int rows, int cols, int src_stride, \
                     cl_mem dst, int dst_stride, cl_command_queue queue) {     \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(float));          \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(float));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain);                                             \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  if (src_stride == cols * src_channels * (int)sizeof(float) &&                \
      dst_stride == cols * dst_channels * (int)sizeof(float)) {                \
    int columns = cols * rows;                                                 \
    local_size[0]  = 512;                                                      \
    local_size[1]  = 1;                                                        \
    global_size[0] = (size_t)roundUp(columns, 512, 9);                         \
    global_size[1] = 1;                                                        \
    frame_chain->setCompileOptions("-D F32 -D " #function "_F32_1D");          \
    runOclKernel(frame_chain, #function "F32Kernel0", 2, global_size,          \
                 local_size, src, columns, dst);                               \
  }                                                                            \
  else {                                                                       \
    frame_chain->setCompileOptions("-D F32 -D " #function "_F32_2D");          \
    runOclKernel(frame_chain, #function "F32Kernel1", 2, global_size,          \
                 local_size, src, rows, cols, src_stride, dst, dst_stride);    \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                        int height,                                            \
                        int width,                                             \
                        int inWidthStride,                                     \
                        const cl_mem inData,                                   \
                        int outWidthStride,                                    \
                        cl_mem outData) {                                      \
  RetCode code = function ## _U8(inData, height, width, inWidthStride,         \
                                  outData, outWidthStride, queue);             \
                                                                               \
  return code;                                                                 \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <float>(cl_command_queue queue,                               \
                        int height,                                            \
                        int width,                                             \
                        int inWidthStride,                                     \
                        const cl_mem inData,                                   \
                        int outWidthStride,                                    \
                        cl_mem outData) {                                      \
  inWidthStride  *= sizeof(float);                                             \
  outWidthStride *= sizeof(float);                                             \
  RetCode code = function ## _F32(inData, height, width, inWidthStride,        \
                                  outData, outWidthStride, queue);             \
                                                                               \
  return code;                                                                 \
}

// BGR(RBB) <-> BGRA(RGBA)
CVT_COLOR_BGR_INVOCATION(BGR2BGRA, 3, 4);
CVT_COLOR_BGR_INVOCATION(RGB2RGBA, 3, 4);
CVT_COLOR_BGR_INVOCATION(BGRA2BGR, 4, 3);
CVT_COLOR_BGR_INVOCATION(RGBA2RGB, 4, 3);
CVT_COLOR_BGR_INVOCATION(BGR2RGBA, 3, 4);
CVT_COLOR_BGR_INVOCATION(RGB2BGRA, 3, 4);
CVT_COLOR_BGR_INVOCATION(BGRA2RGB, 4, 3);
CVT_COLOR_BGR_INVOCATION(RGBA2BGR, 4, 3);

// BGR <-> RGB
CVT_COLOR_BGR_INVOCATION(BGR2RGB, 3, 3);
CVT_COLOR_BGR_INVOCATION(RGB2BGR, 3, 3);

// BGRA <-> RGBA
CVT_COLOR_BGR_INVOCATION(BGRA2RGBA, 4, 4);
CVT_COLOR_BGR_INVOCATION(RGBA2BGRA, 4, 4);

/*********************** BGR/RGB/BGRA/RGBA <-> Gray ************************/

#define CVT_COLOR_GRAY_INVOCATION(function, src_channels, dst_channels)        \
RetCode function ## _U8(const cl_mem src, int rows, int cols, int src_stride,  \
                    cl_mem dst, int dst_stride, cl_command_queue queue) {      \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));          \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain);                                             \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  if (src_stride == cols * src_channels * (int)sizeof(uchar) &&                \
      dst_stride == cols * dst_channels * (int)sizeof(uchar)) {                \
/*     int columns = cols * rows;                                                 \
    local_size[0]  = 512;                                                      \
    local_size[1]  = 1;                                                        \
    global_size[0] = (size_t)roundUp(divideUp(columns, 1), 512, 9);                         \
    global_size[1] = 1;                                                        \
    frame_chain->setCompileOptions("-D U8 -D " #function "_U8_1D");            \
    runOclKernel(frame_chain, #function "U8Kernel0", 2, global_size,           \
                 local_size, src, columns, dst);  */                              \
    int columns = cols * rows;                                                 \
    local_size[0]  = 512;                                                      \
    local_size[1]  = 1;                                                        \
    global_size[0] = (size_t)roundUp(columns, 512, 9);                         \
    global_size[1] = 1;                                                        \
    frame_chain->setCompileOptions("-D U8 -D " #function "_U8_1D");            \
    runOclKernel(frame_chain, #function "U8Kernel0", 2, global_size,           \
                 local_size, src, columns, dst);                               \
  }                                                                            \
  else {                                                                       \
    frame_chain->setCompileOptions("-D U8 -D " #function "_U8_2D");            \
    runOclKernel(frame_chain, #function "U8Kernel1", 2, global_size,           \
                 local_size, src, rows, cols, src_stride, dst, dst_stride);    \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
RetCode function ## _F32(const cl_mem src, int rows, int cols, int src_stride, \
                     cl_mem dst, int dst_stride, cl_command_queue queue) {     \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(float));          \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(float));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain);                                             \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  if (src_stride == cols * src_channels * (int)sizeof(float) &&                \
      dst_stride == cols * dst_channels * (int)sizeof(float)) {                \
    int columns = cols * rows;                                                 \
    local_size[0]  = 512;                                                      \
    local_size[1]  = 1;                                                        \
    global_size[0] = (size_t)roundUp(columns, 512, 9);                         \
    global_size[1] = 1;                                                        \
    frame_chain->setCompileOptions("-D F32 -D " #function "_F32_1D");          \
    runOclKernel(frame_chain, #function "F32Kernel0", 2, global_size,          \
                 local_size, src, columns, dst);                               \
  }                                                                            \
  else {                                                                       \
    frame_chain->setCompileOptions("-D F32 -D " #function "_F32_2D");          \
    runOclKernel(frame_chain, #function "F32Kernel1", 2, global_size,          \
                 local_size, src, rows, cols, src_stride, dst, dst_stride);    \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                        int height,                                            \
                        int width,                                             \
                        int inWidthStride,                                     \
                        const cl_mem inData,                                   \
                        int outWidthStride,                                    \
                        cl_mem outData) {                                      \
  RetCode code = function ## _U8(inData, height, width, inWidthStride,         \
                                  outData, outWidthStride, queue);             \
                                                                               \
  return code;                                                                 \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <float>(cl_command_queue queue,                               \
                        int height,                                            \
                        int width,                                             \
                        int inWidthStride,                                     \
                        const cl_mem inData,                                   \
                        int outWidthStride,                                    \
                        cl_mem outData) {                                      \
  inWidthStride  *= sizeof(float);                                             \
  outWidthStride *= sizeof(float);                                             \
  RetCode code = function ## _F32(inData, height, width, inWidthStride,        \
                                  outData, outWidthStride, queue);             \
                                                                               \
  return code;                                                                 \
}

CVT_COLOR_GRAY_INVOCATION(BGR2GRAY, 3, 1);
CVT_COLOR_GRAY_INVOCATION(RGB2GRAY, 3, 1);
CVT_COLOR_GRAY_INVOCATION(BGRA2GRAY, 4, 1);
CVT_COLOR_GRAY_INVOCATION(RGBA2GRAY, 4, 1);
CVT_COLOR_GRAY_INVOCATION(GRAY2BGR, 1, 3);
CVT_COLOR_GRAY_INVOCATION(GRAY2RGB, 1, 3);
CVT_COLOR_GRAY_INVOCATION(GRAY2BGRA, 1, 4);
CVT_COLOR_GRAY_INVOCATION(GRAY2RGBA, 1, 4);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
