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

#include "ppl/cv/ocl/sepfilter2d.h"
#include "utility/use_memory_pool.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/sepfilter2d.cl"
#include "kerneltypes.h"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode sepfilter2dC1U8(const cl_mem src, int rows, int cols, int src_stride,
                        const cl_mem kernel_x, const cl_mem kernel_y, int ksize,
                        cl_mem dst, int dst_stride, float delta,
                        BorderType border_type, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101)
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);
  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  int is_symmetric = ksize & 1;
  ksize = ksize >> 1;
  global_cols = divideUp(cols, 2, 1);
  global_rows = divideUp(rows, 2, 1);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D SEPFILTER2DU8C1");
  cl_mem buffer;
  GpuMemoryBlock buffer_block;
  buffer_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols);
    buffer = buffer_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  runOclKernel(frame_chain, "sepfilter2dU8F32C1Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel_x, ksize, src_stride,
               buffer, rows * (int)sizeof(float), is_symmetric, 0.f,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 4, 2);
  global_rows = divideUp(rows, 4, 2);
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "sepfilter2dF32U8C1Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows,
               kernel_y, ksize, rows * (int)sizeof(float), dst, dst_stride,
               is_symmetric, delta, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
  }
  else {
    clReleaseMemObject(buffer);
  }
  return RC_SUCCESS;
}

RetCode sepfilter2dC3U8(const cl_mem src, int rows, int cols, int src_stride,
                        const cl_mem kernel_x, const cl_mem kernel_y, int ksize,
                        cl_mem dst, int dst_stride, float delta,
                        BorderType border_type, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar) * 3);
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar) * 3);
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101)
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);
  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  int is_symmetric = ksize & 1;
  ksize = ksize >> 1;
  cl_mem buffer;
  GpuMemoryBlock buffer_block;
  buffer_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols * 3);
    buffer = buffer_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols * 3, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D SEPFILTER2DU8C3");
  runOclKernel(frame_chain, "sepfilter2dU8F32C3Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel_x, ksize, src_stride,
               buffer, rows * (int)sizeof(float) * 3, is_symmetric, 0.f,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 4, 2);
  global_rows = rows;
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "sepfilter2dF32U8C3Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows,
               kernel_y, ksize, rows * (int)sizeof(float) * 3, dst, dst_stride,
               is_symmetric, delta, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
  }
  else {
    clReleaseMemObject(buffer);
  }
  return RC_SUCCESS;
}

RetCode sepfilter2dC4U8(const cl_mem src, int rows, int cols, int src_stride,
                        const cl_mem kernel_x, const cl_mem kernel_y, int ksize,
                        cl_mem dst, int dst_stride, float delta,
                        BorderType border_type, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar) * 4);
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar) * 4);
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101)
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);
  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  int is_symmetric = ksize & 1;
  ksize = ksize >> 1;
  cl_mem buffer;
  GpuMemoryBlock buffer_block;
  buffer_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols * 4);
    buffer = buffer_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols * 4, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D SEPFILTER2DU8C4");
  runOclKernel(frame_chain, "sepfilter2dU8F32C4Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel_x, ksize, src_stride,
               buffer, rows * (int)sizeof(float) * 4, is_symmetric, 0.f,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 4, 2);
  global_rows = rows;
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "sepfilter2dF32U8C4Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows,
               kernel_y, ksize, rows * (int)sizeof(float) * 4, dst, dst_stride,
               is_symmetric, delta, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
  }
  else {
    clReleaseMemObject(buffer);
  }
  return RC_SUCCESS;
}

RetCode sepfilter2dC1F32(const cl_mem src, int rows, int cols, int src_stride,
                         const cl_mem kernel_x, const cl_mem kernel_y,
                         int ksize, cl_mem dst, int dst_stride, float delta,
                         BorderType border_type, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101)
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);
  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  int is_symmetric = ksize & 1;
  ksize = ksize >> 1;
  global_cols = divideUp(cols, 2, 1);
  global_rows = divideUp(rows, 2, 1);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D SEPFILTER2DF32C1");
  cl_mem buffer;
  GpuMemoryBlock buffer_block;
  buffer_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols);
    buffer = buffer_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  runOclKernel(frame_chain, "sepfilter2dF32F32C1Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel_x, ksize, src_stride,
               buffer, rows * (int)sizeof(float), is_symmetric, 0.f,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 2, 1);
  global_rows = divideUp(rows, 2, 1);
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "sepfilter2dF32F32C1Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows,
               kernel_y, ksize, rows * (int)sizeof(float), dst, dst_stride,
               is_symmetric, delta, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
  }
  else {
    clReleaseMemObject(buffer);
  }
  return RC_SUCCESS;
}

RetCode sepfilter2dC3F32(const cl_mem src, int rows, int cols, int src_stride,
                         const cl_mem kernel_x, const cl_mem kernel_y,
                         int ksize, cl_mem dst, int dst_stride, float delta,
                         BorderType border_type, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float) * 3);
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float) * 3);
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101)
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);
  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  int is_symmetric = ksize & 1;
  ksize = ksize >> 1;
  cl_mem buffer;
  GpuMemoryBlock buffer_block;
  buffer_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols * 3);
    buffer = buffer_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols * 3, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D SEPFILTER2DF32C3");
  runOclKernel(frame_chain, "sepfilter2dF32F32C3Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel_x, ksize, src_stride,
               buffer, rows * (int)sizeof(float) * 3, is_symmetric, 0.f,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 1, 0);
  global_rows = rows;
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "sepfilter2dF32F32C3Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows,
               kernel_y, ksize, rows * (int)sizeof(float) * 3, dst, dst_stride,
               is_symmetric, delta, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
  }
  else {
    clReleaseMemObject(buffer);
  }
  return RC_SUCCESS;
}

RetCode sepfilter2dC4F32(const cl_mem src, int rows, int cols, int src_stride,
                         const cl_mem kernel_x, const cl_mem kernel_y,
                         int ksize, cl_mem dst, int dst_stride, float delta,
                         BorderType border_type, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float) * 4);
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float) * 4);
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101)
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);
  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  int is_symmetric = ksize & 1;
  ksize = ksize >> 1;
  cl_mem buffer;
  GpuMemoryBlock buffer_block;
  buffer_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols * 4);
    buffer = buffer_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols * 4, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D SEPFILTER2DF32C4");
  runOclKernel(frame_chain, "sepfilter2dF32F32C4Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel_x, ksize, src_stride,
               buffer, rows * (int)sizeof(float) * 4, is_symmetric, 0.f,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 1, 0);
  global_rows = rows;
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "sepfilter2dF32F32C4Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows,
               kernel_y, ksize, rows * (int)sizeof(float) * 4, dst, dst_stride,
               is_symmetric, delta, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
  }
  else {
    clReleaseMemObject(buffer);
  }
  return RC_SUCCESS;
}

template <>
RetCode SepFilter2D<uchar, uchar, 1>(cl_command_queue queue,
                                     int height,
                                     int width,
                                     int inWidthStride,
                                     const cl_mem inData,
                                     int ksize,
                                     const cl_mem kernelX,
                                     const cl_mem kernelY,
                                     int outWidthStride,
                                     cl_mem outData,
                                     float delta,
                                     BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = sepfilter2dC1U8(inData, height, width, inWidthStride, kernelX,
                                 kernelY, ksize, outData, outWidthStride, delta,
                                 border_type, queue);
  return code;
}

template <>
RetCode SepFilter2D<uchar, uchar, 3>(cl_command_queue queue,
                                     int height,
                                     int width,
                                     int inWidthStride,
                                     const cl_mem inData,
                                     int ksize,
                                     const cl_mem kernelX,
                                     const cl_mem kernelY,
                                     int outWidthStride,
                                     cl_mem outData,
                                     float delta,
                                     BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = sepfilter2dC3U8(inData, height, width, inWidthStride, kernelX,
                                 kernelY, ksize, outData, outWidthStride, delta,
                                 border_type, queue);
  return code;
}

template <>
RetCode SepFilter2D<uchar, uchar, 4>(cl_command_queue queue,
                                     int height,
                                     int width,
                                     int inWidthStride,
                                     const cl_mem inData,
                                     int ksize,
                                     const cl_mem kernelX,
                                     const cl_mem kernelY,
                                     int outWidthStride,
                                     cl_mem outData,
                                     float delta,
                                     BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code = sepfilter2dC4U8(inData, height, width, inWidthStride, kernelX,
                                 kernelY, ksize, outData, outWidthStride, delta,
                                 border_type, queue);
  return code;
}

template <>
RetCode SepFilter2D<float, float, 1>(cl_command_queue queue,
                                     int height,
                                     int width,
                                     int inWidthStride,
                                     const cl_mem inData,
                                     int ksize,
                                     const cl_mem kernelX,
                                     const cl_mem kernelY,
                                     int outWidthStride,
                                     cl_mem outData,
                                     float delta,
                                     BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sepfilter2dC1F32(inData, height, width, inWidthStride, kernelX,
                                  kernelY, ksize, outData, outWidthStride,
                                  delta, border_type, queue);
  return code;
}

template <>
RetCode SepFilter2D<float, float, 3>(cl_command_queue queue,
                                     int height,
                                     int width,
                                     int inWidthStride,
                                     const cl_mem inData,
                                     int ksize,
                                     const cl_mem kernelX,
                                     const cl_mem kernelY,
                                     int outWidthStride,
                                     cl_mem outData,
                                     float delta,
                                     BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sepfilter2dC3F32(inData, height, width, inWidthStride, kernelX,
                                  kernelY, ksize, outData, outWidthStride,
                                  delta, border_type, queue);
  return code;
}

template <>
RetCode SepFilter2D<float, float, 4>(cl_command_queue queue,
                                     int height,
                                     int width,
                                     int inWidthStride,
                                     const cl_mem inData,
                                     int ksize,
                                     const cl_mem kernelX,
                                     const cl_mem kernelY,
                                     int outWidthStride,
                                     cl_mem outData,
                                     float delta,
                                     BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sepfilter2dC4F32(inData, height, width, inWidthStride, kernelX,
                                  kernelY, ksize, outData, outWidthStride,
                                  delta, border_type, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
