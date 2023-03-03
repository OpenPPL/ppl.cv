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

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/cvtcolor.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

/************************** Common Conversions ****************************/

#define CVT_COLOR_COMMON_INVOCATION(function, src_channels, dst_channels)      \
RetCode function ## _U8(const cl_mem src, int rows, int cols, int src_stride,  \
                        cl_mem dst, int dst_stride, cl_command_queue queue) {  \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));          \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
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
    frame_chain->setCompileOptions("-D " #function "_U8_1D");                  \
    runOclKernel(frame_chain, #function "U8Kernel0", 2, global_size,           \
                 local_size, src, columns, dst);                               \
  }                                                                            \
  else {                                                                       \
    frame_chain->setCompileOptions("-D " #function "_U8_2D");                  \
    runOclKernel(frame_chain, #function "U8Kernel1", 2, global_size,           \
                 local_size, src, rows, cols, src_stride, dst, dst_stride);    \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
RetCode function ## _F32(const cl_mem src, int rows, int cols, int src_stride, \
                         cl_mem dst, int dst_stride, cl_command_queue queue) { \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(float));          \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(float));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
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
    frame_chain->setCompileOptions("-D " #function "_F32_1D");                 \
    runOclKernel(frame_chain, #function "F32Kernel0", 2, global_size,          \
                 local_size, src, columns, dst);                               \
  }                                                                            \
  else {                                                                       \
    frame_chain->setCompileOptions("-D " #function "_F32_2D");                 \
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
  RetCode code = function ## _U8(inData, height, width, inWidthStride, outData,\
                                 outWidthStride, queue);                       \
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
CVT_COLOR_COMMON_INVOCATION(BGR2BGRA, 3, 4);
CVT_COLOR_COMMON_INVOCATION(RGB2RGBA, 3, 4);
CVT_COLOR_COMMON_INVOCATION(BGRA2BGR, 4, 3);
CVT_COLOR_COMMON_INVOCATION(RGBA2RGB, 4, 3);
CVT_COLOR_COMMON_INVOCATION(BGR2RGBA, 3, 4);
CVT_COLOR_COMMON_INVOCATION(RGB2BGRA, 3, 4);
CVT_COLOR_COMMON_INVOCATION(BGRA2RGB, 4, 3);
CVT_COLOR_COMMON_INVOCATION(RGBA2BGR, 4, 3);

// BGR <-> RGB
CVT_COLOR_COMMON_INVOCATION(BGR2RGB, 3, 3);
CVT_COLOR_COMMON_INVOCATION(RGB2BGR, 3, 3);

// BGRA <-> RGBA
CVT_COLOR_COMMON_INVOCATION(BGRA2RGBA, 4, 4);
CVT_COLOR_COMMON_INVOCATION(RGBA2BGRA, 4, 4);

// BGR/RGB/BGRA/RGBA <-> Gray
CVT_COLOR_COMMON_INVOCATION(BGR2GRAY, 3, 1);
CVT_COLOR_COMMON_INVOCATION(RGB2GRAY, 3, 1);
CVT_COLOR_COMMON_INVOCATION(BGRA2GRAY, 4, 1);
CVT_COLOR_COMMON_INVOCATION(RGBA2GRAY, 4, 1);
CVT_COLOR_COMMON_INVOCATION(GRAY2BGR, 1, 3);
CVT_COLOR_COMMON_INVOCATION(GRAY2RGB, 1, 3);
CVT_COLOR_COMMON_INVOCATION(GRAY2BGRA, 1, 4);
CVT_COLOR_COMMON_INVOCATION(GRAY2RGBA, 1, 4);

// BGR/RGB/BGRA/RGBA <-> YCrCb
CVT_COLOR_COMMON_INVOCATION(BGR2YCrCb, 3, 3);
CVT_COLOR_COMMON_INVOCATION(RGB2YCrCb, 3, 3);
CVT_COLOR_COMMON_INVOCATION(BGRA2YCrCb, 4, 3);
CVT_COLOR_COMMON_INVOCATION(RGBA2YCrCb, 4, 3);
CVT_COLOR_COMMON_INVOCATION(YCrCb2BGR, 3, 3);
CVT_COLOR_COMMON_INVOCATION(YCrCb2RGB, 3, 3);
CVT_COLOR_COMMON_INVOCATION(YCrCb2BGRA, 3, 4);
CVT_COLOR_COMMON_INVOCATION(YCrCb2RGBA, 3, 4);

// BGR/RGB/BGRA/RGBA <-> HSV
CVT_COLOR_COMMON_INVOCATION(BGR2HSV, 3, 3);
CVT_COLOR_COMMON_INVOCATION(RGB2HSV, 3, 3);
CVT_COLOR_COMMON_INVOCATION(BGRA2HSV, 4, 3);
CVT_COLOR_COMMON_INVOCATION(RGBA2HSV, 4, 3);
CVT_COLOR_COMMON_INVOCATION(HSV2BGR, 3, 3);
CVT_COLOR_COMMON_INVOCATION(HSV2RGB, 3, 3);
CVT_COLOR_COMMON_INVOCATION(HSV2BGRA, 3, 4);
CVT_COLOR_COMMON_INVOCATION(HSV2RGBA, 3, 4);

// BGR/RGB/BGRA/RGBA <-> LAB
CVT_COLOR_COMMON_INVOCATION(BGR2LAB, 3, 3);
CVT_COLOR_COMMON_INVOCATION(RGB2LAB, 3, 3);
CVT_COLOR_COMMON_INVOCATION(BGRA2LAB, 4, 3);
CVT_COLOR_COMMON_INVOCATION(RGBA2LAB, 4, 3);
CVT_COLOR_COMMON_INVOCATION(LAB2BGR, 3, 3);
CVT_COLOR_COMMON_INVOCATION(LAB2RGB, 3, 3);
CVT_COLOR_COMMON_INVOCATION(LAB2BGRA, 3, 4);
CVT_COLOR_COMMON_INVOCATION(LAB2RGBA, 3, 4);

/********************** BGR/RGB/BGRA/RGBA <-> NV12/21 ***********************/

#define CVT_COLOR_NVXX_INVOCATION(function, src_channels, dst_channels)        \
RetCode function ## _U8(const cl_mem src, int rows, int cols, int src_stride,  \
                        cl_mem dst, int dst_stride, cl_command_queue queue) {  \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((rows & 0x1) == 0 && (cols & 0x1) == 0);                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));          \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  frame_chain->setCompileOptions("-D " #function "_U8_2D");                    \
  runOclKernel(frame_chain, #function "U8Kernel0", 2, global_size,             \
               local_size, src, rows, cols, src_stride, dst, dst_stride);      \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                         int height,                                           \
                         int width,                                            \
                         int inWidthStride,                                    \
                         const cl_mem inData,                                  \
                         int outWidthStride,                                   \
                         cl_mem outData) {                                     \
  RetCode code = function ## _U8(inData, height, width, inWidthStride, outData,\
                                 outWidthStride, queue);                       \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(function, src_channels)          \
RetCode function ## _U8(const cl_mem src, int rows, int cols, int src_stride,  \
                        cl_mem dst_y, int dst_y_stride, cl_mem dst_uv,         \
                        int dst_uv_stride, cl_command_queue queue) {           \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst_y != nullptr);                                                \
  PPL_ASSERT(dst_uv != nullptr);                                               \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((rows & 0x1) == 0 && (cols & 0x1) == 0);                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));          \
  PPL_ASSERT(dst_y_stride >= cols * (int)sizeof(uchar));                       \
  PPL_ASSERT(dst_uv_stride >= cols * (int)sizeof(uchar));                      \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  frame_chain->setCompileOptions("-D " #function "_DISCRETE_U8_2D");           \
  runOclKernel(frame_chain, #function "U8Kernel1", 2, global_size,             \
               local_size, src, rows, cols, src_stride, dst_y, dst_y_stride,   \
               dst_uv, dst_uv_stride);                                         \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                         int height,                                           \
                         int width,                                            \
                         int inWidthStride,                                    \
                         const cl_mem inData,                                  \
                         int outYStride,                                       \
                         cl_mem outY,                                          \
                         int outUVStride,                                      \
                         cl_mem outUV) {                                       \
  RetCode code = function ## _U8(inData, height, width, inWidthStride, outY,   \
                                 outYStride, outUV, outUVStride, queue);       \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(function, dst_channels)        \
RetCode function ## _U8(const cl_mem src_y, int rows, int cols,                \
                        int src_y_stride, const cl_mem src_uv,                 \
                        int src_uv_stride, cl_mem dst, int dst_stride,         \
                        cl_command_queue queue) {                              \
  PPL_ASSERT(src_y != nullptr);                                                \
  PPL_ASSERT(src_uv != nullptr);                                               \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((rows & 0x1) == 0 && (cols & 0x1) == 0);                          \
  PPL_ASSERT(src_y_stride >= cols * (int)sizeof(uchar));                       \
  PPL_ASSERT(src_uv_stride >= cols * (int)sizeof(uchar));                      \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  frame_chain->setCompileOptions("-D " #function "_DISCRETE_U8_2D");           \
  runOclKernel(frame_chain, #function "U8Kernel1", 2, global_size,             \
               local_size, src_y, rows, cols, src_y_stride, src_uv,            \
               src_uv_stride, dst, dst_stride);                                \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                         int height,                                           \
                         int width,                                            \
                         int inYStride,                                        \
                         const cl_mem inY,                                     \
                         int inUVStride,                                       \
                         const cl_mem inUV,                                    \
                         int outWidthStride,                                   \
                         cl_mem outData) {                                     \
  RetCode code = function ## _U8(inY, height, width, inYStride, inUV,          \
                                 inUVStride, outData, outWidthStride, queue);  \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_TO_DISCRETE_I420_INVOCATION(function, src_channels)          \
RetCode function ## _U8(const cl_mem src, int rows, int cols, int src_stride,  \
                        cl_mem dst_y, int dst_y_stride, cl_mem dst_u,          \
                        int dst_u_stride, cl_mem dst_v, int dst_v_stride,      \
                        cl_command_queue queue) {                              \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst_y != nullptr);                                                \
  PPL_ASSERT(dst_u != nullptr);                                                \
  PPL_ASSERT(dst_v != nullptr);                                                \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((rows & 0x1) == 0 && (cols & 0x1) == 0);                          \
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));          \
  PPL_ASSERT(dst_y_stride >= cols * (int)sizeof(uchar));                       \
  PPL_ASSERT(dst_u_stride >= cols / 2 * (int)sizeof(uchar));                   \
  PPL_ASSERT(dst_v_stride >= cols / 2 * (int)sizeof(uchar));                   \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  frame_chain->setCompileOptions("-D " #function "_DISCRETE_U8_2D");           \
  runOclKernel(frame_chain, #function "U8Kernel1", 2, global_size,             \
               local_size, src, rows, cols, src_stride, dst_y, dst_y_stride,   \
               dst_u, dst_u_stride, dst_v, dst_v_stride);                      \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                         int height,                                           \
                         int width,                                            \
                         int inWidthStride,                                    \
                         const cl_mem inData,                                  \
                         int outYStride,                                       \
                         cl_mem outY,                                          \
                         int outUStride,                                       \
                         cl_mem outU,                                          \
                         int outVStride,                                       \
                         cl_mem outV) {                                        \
  RetCode code = function ## _U8(inData, height, width, inWidthStride, outY,   \
                                 outYStride, outU, outUStride, outV,           \
                                 outVStride, queue);                           \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(function, dst_channels)        \
RetCode function ## _U8(const cl_mem src_y, int rows, int cols,                \
                        int src_y_stride, const cl_mem src_u, int src_u_stride,\
                        const cl_mem src_v, int src_v_stride, cl_mem dst,      \
                        int dst_stride, cl_command_queue queue) {              \
  PPL_ASSERT(src_y != nullptr);                                                \
  PPL_ASSERT(src_u != nullptr);                                                \
  PPL_ASSERT(src_v != nullptr);                                                \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((rows & 0x1) == 0 && (cols & 0x1) == 0);                          \
  PPL_ASSERT(src_y_stride >= cols * (int)sizeof(uchar));                       \
  PPL_ASSERT(src_u_stride >= cols / 2 * (int)sizeof(uchar));                   \
  PPL_ASSERT(src_v_stride >= cols / 2 * (int)sizeof(uchar));                   \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)cols, (size_t)rows};                         \
                                                                               \
  frame_chain->setCompileOptions("-D " #function "_DISCRETE_U8_2D");           \
  runOclKernel(frame_chain, #function "U8Kernel1", 2, global_size,             \
               local_size, src_y, rows, cols, src_y_stride, src_u,             \
               src_u_stride, src_v, src_v_stride, dst, dst_stride);            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                         int height,                                           \
                         int width,                                            \
                         int inYStride,                                        \
                         const cl_mem inY,                                     \
                         int inUStride,                                        \
                         const cl_mem inU,                                     \
                         int inVStride,                                        \
                         const cl_mem inV,                                     \
                         int outWidthStride,                                   \
                         cl_mem outData) {                                     \
  RetCode code = function ## _U8(inY, height, width, inYStride, inU, inUStride,\
                                 inV, inVStride, outData, outWidthStride,      \
                                 queue);                                       \
                                                                               \
  return code;                                                                 \
}

// BGR/RGB/BGRA/RGBA <-> NV12
CVT_COLOR_NVXX_INVOCATION(BGR2NV12, 3, 1)
CVT_COLOR_NVXX_INVOCATION(RGB2NV12, 3, 1)
CVT_COLOR_NVXX_INVOCATION(BGRA2NV12, 4, 1)
CVT_COLOR_NVXX_INVOCATION(RGBA2NV12, 4, 1)
CVT_COLOR_NVXX_INVOCATION(NV122BGR, 1, 3)
CVT_COLOR_NVXX_INVOCATION(NV122RGB, 1, 3)
CVT_COLOR_NVXX_INVOCATION(NV122BGRA, 1, 4)
CVT_COLOR_NVXX_INVOCATION(NV122RGBA, 1, 4)

CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(BGR2NV12, 3)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(RGB2NV12, 3)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(BGRA2NV12, 4)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(RGBA2NV12, 4)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV122BGR, 3)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV122RGB, 3)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV122BGRA, 4)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV122RGBA, 4)

// BGR/RGB/BGRA/RGBA <-> NV21
CVT_COLOR_NVXX_INVOCATION(BGR2NV21, 3, 1)
CVT_COLOR_NVXX_INVOCATION(RGB2NV21, 3, 1)
CVT_COLOR_NVXX_INVOCATION(BGRA2NV21, 4, 1)
CVT_COLOR_NVXX_INVOCATION(RGBA2NV21, 4, 1)
CVT_COLOR_NVXX_INVOCATION(NV212BGR, 1, 3)
CVT_COLOR_NVXX_INVOCATION(NV212RGB, 1, 3)
CVT_COLOR_NVXX_INVOCATION(NV212BGRA, 1, 4)
CVT_COLOR_NVXX_INVOCATION(NV212RGBA, 1, 4)

CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(BGR2NV21, 3)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(RGB2NV21, 3)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(BGRA2NV21, 4)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(RGBA2NV21, 4)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV212BGR, 3)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV212RGB, 3)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV212BGRA, 4)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV212RGBA, 4)

// BGR/RGB/BGRA/RGBA <-> I420
CVT_COLOR_NVXX_INVOCATION(BGR2I420, 3, 1)
CVT_COLOR_NVXX_INVOCATION(RGB2I420, 3, 1)
CVT_COLOR_NVXX_INVOCATION(BGRA2I420, 4, 1)
CVT_COLOR_NVXX_INVOCATION(RGBA2I420, 4, 1)
CVT_COLOR_NVXX_INVOCATION(I4202BGR, 1, 3)
CVT_COLOR_NVXX_INVOCATION(I4202RGB, 1, 3)
CVT_COLOR_NVXX_INVOCATION(I4202BGRA, 1, 4)
CVT_COLOR_NVXX_INVOCATION(I4202RGBA, 1, 4)

CVT_COLOR_TO_DISCRETE_I420_INVOCATION(BGR2I420, 3)
CVT_COLOR_TO_DISCRETE_I420_INVOCATION(RGB2I420, 3)
CVT_COLOR_TO_DISCRETE_I420_INVOCATION(BGRA2I420, 4)
CVT_COLOR_TO_DISCRETE_I420_INVOCATION(RGBA2I420, 4)
CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(I4202BGR, 3)
CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(I4202RGB, 3)
CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(I4202BGRA, 4)
CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(I4202RGBA, 4)

/***************************** YUV2 -> GRAY ******************************/

RetCode YUV2GRAY_U8(const cl_mem src, int rows, int cols, int src_stride,
                    cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT((rows & 0x1) == 0 && (cols & 0x1) == 0);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);

  int columns = divideUp(cols, 4, 2);
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};
  size_t global_size[] = {(size_t)columns, (size_t)rows};

  frame_chain->setCompileOptions("-D YUV2GRAY_U8_2D");
  runOclKernel(frame_chain, "YUV2GRAYU8Kernel", 2, global_size,
               local_size, src, rows, cols, src_stride, dst, dst_stride);

  return RC_SUCCESS;
}

template <>
RetCode YUV2GRAY<uchar>(cl_command_queue queue,
                        int height,
                        int width,
                        int inWidthStride,
                        const cl_mem inData,
                        int outWidthStride,
                        cl_mem outData) {
  RetCode code = YUV2GRAY_U8(inData, height, width, inWidthStride, outData,
                             outWidthStride, queue);

  return code;
}

/************************ BGR/GRAY <-> UYVY/YUYV *************************/

#define CVT_COLOR_FROM_YUV422_INVOCATION(function, src_channels, dst_channels) \
RetCode function ## _U8(const cl_mem src, int rows, int cols, int src_stride,  \
                        cl_mem dst, int dst_stride, cl_command_queue queue) {  \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((cols & 0x1) == 0);                                               \
  PPL_ASSERT(src_stride >= cols / 2 * src_channels * (int)sizeof(uchar));      \
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));          \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)divideUp(cols, 2, 1), (size_t)rows};         \
                                                                               \
  frame_chain->setCompileOptions("-D " #function "_U8_2D");                    \
  runOclKernel(frame_chain, #function "U8Kernel", 2, global_size,              \
               local_size, src, rows, cols, src_stride, dst, dst_stride);      \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                         int height,                                           \
                         int width,                                            \
                         int inWidthStride,                                    \
                         const cl_mem inData,                                  \
                         int outWidthStride,                                   \
                         cl_mem outData) {                                     \
  RetCode code = function ## _U8(inData, height, width, inWidthStride, outData,\
                                 outWidthStride, queue);                       \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_YUV422_TO_GRAY_INVOCATION(function)                          \
RetCode function ## _U8(const cl_mem src, int rows, int cols, int src_stride,  \
                        cl_mem dst, int dst_stride, cl_command_queue queue) {  \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((cols & 0x1) == 0);                                               \
  PPL_ASSERT(src_stride >= cols / 2 * (int)sizeof(uchar));                     \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));                         \
                                                                               \
  FrameChain* frame_chain = getSharedFrameChain();                             \
  frame_chain->setProjectName("cv");                                           \
  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);                                   \
                                                                               \
  size_t local_size[]  = {kBlockDimX0, kBlockDimY0};                           \
  size_t global_size[] = {(size_t)divideUp(cols, 2, 1), (size_t)rows};         \
                                                                               \
  frame_chain->setCompileOptions("-D " #function "_U8_2D");                    \
  runOclKernel(frame_chain, #function "U8Kernel", 2, global_size,              \
               local_size, src, rows, cols, src_stride, dst, dst_stride);      \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function <uchar>(cl_command_queue queue,                               \
                         int height,                                           \
                         int width,                                            \
                         int inWidthStride,                                    \
                         const cl_mem inData,                                  \
                         int outWidthStride,                                   \
                         cl_mem outData) {                                     \
  RetCode code = function ## _U8(inData, height, width, inWidthStride, outData,\
                                 outWidthStride, queue);                       \
                                                                               \
  return code;                                                                 \
}

// BGR/GRAY <-> UYVY
CVT_COLOR_FROM_YUV422_INVOCATION(UYVY2BGR, 4, 3)
CVT_COLOR_YUV422_TO_GRAY_INVOCATION(UYVY2GRAY)

// BGR/GRAY <-> YUYV
CVT_COLOR_FROM_YUV422_INVOCATION(YUYV2BGR, 4, 3)
CVT_COLOR_YUV422_TO_GRAY_INVOCATION(YUYV2GRAY)

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
