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

#include "ppl/cv/ocl/gaussianblur.h"
#include "utility/use_memory_pool.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/gaussianblur.cl"
#include "kerneltypes.h"
#include <math.h>

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode gaussianblurC1U8(const cl_mem src, int rows, int cols, int src_stride,
                         int ksize, float sigma, cl_mem dst, int dst_stride,
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
  SET_PROGRAM_SOURCE(frame_chain, gaussianblur);

  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  global_cols = divideUp(cols, 2, 1);
  global_rows = divideUp(rows, 2, 1);
  frame_chain->setCompileOptions("-D GAUSSIANBLUR_U8C1");
  cl_mem buffer, kernel;
  GpuMemoryBlock buffer_block, kernel_block;
  buffer_block.offset = 0;
  kernel_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols);
    buffer = buffer_block.data;
    pplOclMalloc(kernel_block, ksize * (int)sizeof(float));
    kernel = kernel_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
    kernel = clCreateBuffer(frame_chain->getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                            ksize * (int)sizeof(float), NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_size[0] = (size_t)1;
  runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
               sigma, ksize, kernel, (int)kernel_block.offset);
  ksize = ksize >> 1;
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  runOclKernel(frame_chain, "gaussianblurU8F32C1Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel, (int)kernel_block.offset,
               ksize, src_stride, buffer, rows * (int)sizeof(float),
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 4, 2);
  global_rows = divideUp(rows, 4, 2);
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "gaussianblurF32U8C1Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows, kernel,
               (int)kernel_block.offset, ksize, rows * (int)sizeof(float), dst,
               dst_stride, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
    pplOclFree(kernel_block);
  }
  else {
    clReleaseMemObject(buffer);
    clReleaseMemObject(kernel);
  }
  return RC_SUCCESS;
}

RetCode gaussianblurC3U8(const cl_mem src, int rows, int cols, int src_stride,
                         int ksize, float sigma, cl_mem dst, int dst_stride,
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
  SET_PROGRAM_SOURCE(frame_chain, gaussianblur);

  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  cl_mem buffer, kernel;
  GpuMemoryBlock buffer_block, kernel_block;
  buffer_block.offset = 0;
  kernel_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols * 3);
    buffer = buffer_block.data;
    pplOclMalloc(kernel_block, ksize * (int)sizeof(float));
    kernel = kernel_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols * 3, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
    kernel = clCreateBuffer(frame_chain->getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                            ksize * (int)sizeof(float), NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_size[0] = (size_t)1;
  runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
               sigma, ksize, kernel, (int)kernel_block.offset);
  ksize = ksize >> 1;
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D GAUSSIANBLUR_U8C3");
  runOclKernel(frame_chain, "gaussianblurU8F32C3Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel, (int)kernel_block.offset,
               ksize, src_stride, buffer, rows * (int)sizeof(float) * 3,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 4, 2);
  global_rows = rows;
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "gaussianblurF32U8C3Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows, kernel,
               (int)kernel_block.offset, ksize, rows * (int)sizeof(float) * 3,
               dst, dst_stride, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
    pplOclFree(kernel_block);
  }
  else {
    clReleaseMemObject(buffer);
    clReleaseMemObject(kernel);
  }
  return RC_SUCCESS;
}

RetCode gaussianblurC4U8(const cl_mem src, int rows, int cols, int src_stride,
                         int ksize, float sigma, cl_mem dst, int dst_stride,
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
  SET_PROGRAM_SOURCE(frame_chain, gaussianblur);

  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  cl_mem buffer, kernel;
  GpuMemoryBlock buffer_block, kernel_block;
  buffer_block.offset = 0;
  kernel_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols * 4);
    buffer = buffer_block.data;
    pplOclMalloc(kernel_block, ksize * (int)sizeof(float));
    kernel = kernel_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols * 4, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
    kernel = clCreateBuffer(frame_chain->getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                            ksize * (int)sizeof(float), NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_size[0] = (size_t)1;
  runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
               sigma, ksize, kernel, (int)kernel_block.offset);
  ksize = ksize >> 1;
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D GAUSSIANBLUR_U8C4");
  runOclKernel(frame_chain, "gaussianblurU8F32C4Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel, (int)kernel_block.offset,
               ksize, src_stride, buffer, rows * (int)sizeof(float) * 4,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 4, 2);
  global_rows = rows;
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "gaussianblurF32U8C4Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows, kernel,
               (int)kernel_block.offset, ksize, rows * (int)sizeof(float) * 4,
               dst, dst_stride, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
    pplOclFree(kernel_block);
  }
  else {
    clReleaseMemObject(buffer);
    clReleaseMemObject(kernel);
  }
  return RC_SUCCESS;
}

RetCode gaussianblurC1F32(const cl_mem src, int rows, int cols, int src_stride,
                          int ksize, float sigma, cl_mem dst, int dst_stride,
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
  SET_PROGRAM_SOURCE(frame_chain, gaussianblur);

  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  global_cols = divideUp(cols, 2, 1);
  global_rows = divideUp(rows, 2, 1);
  frame_chain->setCompileOptions("-D GAUSSIANBLUR_F32C1");
  cl_mem buffer, kernel;
  GpuMemoryBlock buffer_block, kernel_block;
  buffer_block.offset = 0;
  kernel_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols);
    buffer = buffer_block.data;
    pplOclMalloc(kernel_block, ksize * (int)sizeof(float));
    kernel = kernel_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
    kernel = clCreateBuffer(frame_chain->getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                            ksize * (int)sizeof(float), NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_size[0] = (size_t)1;
  runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
               sigma, ksize, kernel, (int)kernel_block.offset);
  ksize = ksize >> 1;
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  runOclKernel(frame_chain, "gaussianblurF32F32C1Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel, (int)kernel_block.offset,
               ksize, src_stride, buffer, rows * (int)sizeof(float),
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 2, 1);
  global_rows = divideUp(rows, 2, 1);
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "gaussianblurF32F32C1Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows, kernel,
               (int)kernel_block.offset, ksize, rows * (int)sizeof(float), dst,
               dst_stride, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
    pplOclFree(kernel_block);
  }
  else {
    clReleaseMemObject(buffer);
    clReleaseMemObject(kernel);
  }
  return RC_SUCCESS;
}

RetCode gaussianblurC3F32(const cl_mem src, int rows, int cols, int src_stride,
                          int ksize, float sigma, cl_mem dst, int dst_stride,
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
  SET_PROGRAM_SOURCE(frame_chain, gaussianblur);

  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  cl_mem buffer, kernel;
  GpuMemoryBlock buffer_block, kernel_block;
  buffer_block.offset = 0;
  kernel_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols * 3);
    buffer = buffer_block.data;
    pplOclMalloc(kernel_block, ksize * (int)sizeof(float));
    kernel = kernel_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols * 3, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
    kernel = clCreateBuffer(frame_chain->getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                            ksize * (int)sizeof(float), NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_size[0] = (size_t)1;
  runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
               sigma, ksize, kernel, (int)kernel_block.offset);
  ksize = ksize >> 1;
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D GAUSSIANBLUR_F32C3");
  runOclKernel(frame_chain, "gaussianblurF32F32C3Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel, (int)kernel_block.offset,
               ksize, src_stride, buffer, rows * (int)sizeof(float) * 3,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 1, 0);
  global_rows = rows;
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "gaussianblurF32F32C3Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows, kernel,
               (int)kernel_block.offset, ksize, rows * (int)sizeof(float) * 3,
               dst, dst_stride, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
    pplOclFree(kernel_block);
  }
  else {
    clReleaseMemObject(buffer);
    clReleaseMemObject(kernel);
  }
  return RC_SUCCESS;
}

RetCode gaussianblurC4F32(const cl_mem src, int rows, int cols, int src_stride,
                          int ksize, float sigma, cl_mem dst, int dst_stride,
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
  SET_PROGRAM_SOURCE(frame_chain, gaussianblur);

  int global_cols, global_rows;
  size_t local_size[] = {kBlockDimX0, kBlockDimY0};
  size_t global_size[2];
  cl_mem buffer, kernel;
  GpuMemoryBlock buffer_block, kernel_block;
  buffer_block.offset = 0;
  kernel_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, rows * (int)sizeof(float) * cols * 4);
    buffer = buffer_block.data;
    pplOclMalloc(kernel_block, ksize * (int)sizeof(float));
    kernel = kernel_block.data;
  }
  else {
    cl_int error_code = 0;
    buffer = clCreateBuffer(
        frame_chain->getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
        rows * (int)sizeof(float) * cols * 4, NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
    kernel = clCreateBuffer(frame_chain->getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                            ksize * (int)sizeof(float), NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }
  global_size[0] = (size_t)1;
  runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
               sigma, ksize, kernel, (int)kernel_block.offset);
  ksize = ksize >> 1;
  global_cols = cols;
  global_rows = divideUp(rows, 1, 0);
  global_size[0] = (size_t)global_cols;
  global_size[1] = (size_t)global_rows;
  frame_chain->setCompileOptions("-D GAUSSIANBLUR_F32C4");
  runOclKernel(frame_chain, "gaussianblurF32F32C4Kernel", 2, global_size,
               local_size, src, 0, rows, cols, kernel, (int)kernel_block.offset,
               ksize, src_stride, buffer, rows * (int)sizeof(float) * 4,
               (int)buffer_block.offset, border_type);
  global_cols = divideUp(cols, 1, 0);
  global_rows = rows;
  global_size[0] = (size_t)global_rows;
  global_size[1] = (size_t)global_cols;
  runOclKernel(frame_chain, "gaussianblurF32F32C4Kernel", 2, global_size,
               local_size, buffer, (int)buffer_block.offset, cols, rows, kernel,
               (int)kernel_block.offset, ksize, rows * (int)sizeof(float) * 4,
               dst, dst_stride, 0, border_type);
  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
    pplOclFree(kernel_block);
  }
  else {
    clReleaseMemObject(buffer);
    clReleaseMemObject(kernel);
  }
  return RC_SUCCESS;
}

template <>
RetCode GaussianBlur<uchar, 1>(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               cl_mem outData,
                               BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      gaussianblurC1U8(inData, height, width, inWidthStride, ksize, sigma,
                       outData, outWidthStride, border_type, queue);
  return code;
}

template <>
RetCode GaussianBlur<uchar, 3>(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               cl_mem outData,
                               BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      gaussianblurC3U8(inData, height, width, inWidthStride, ksize, sigma,
                       outData, outWidthStride, border_type, queue);
  return code;
}

template <>
RetCode GaussianBlur<uchar, 4>(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               cl_mem outData,
                               BorderType border_type) {
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code =
      gaussianblurC4U8(inData, height, width, inWidthStride, ksize, sigma,
                       outData, outWidthStride, border_type, queue);
  return code;
}

template <>
RetCode GaussianBlur<float, 1>(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               cl_mem outData,
                               BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      gaussianblurC1F32(inData, height, width, inWidthStride, ksize, sigma,
                        outData, outWidthStride, border_type, queue);
  return code;
}

template <>
RetCode GaussianBlur<float, 3>(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               cl_mem outData,
                               BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      gaussianblurC3F32(inData, height, width, inWidthStride, ksize, sigma,
                        outData, outWidthStride, border_type, queue);
  return code;
}

template <>
RetCode GaussianBlur<float, 4>(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               cl_mem outData,
                               BorderType border_type) {
  inWidthStride *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code =
      gaussianblurC4F32(inData, height, width, inWidthStride, ksize, sigma,
                        outData, outWidthStride, border_type, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
