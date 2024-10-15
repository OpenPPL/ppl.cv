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

#include "ppl/cv/ocl/adaptivethreshold.h"
#include "utility/use_memory_pool.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/adaptivethreshold.cl"
#include "kerneltypes.h"
#include <cmath>

using namespace ppl::common;
using namespace ppl::common::ocl;
#define LARGE_MAX_KSIZE 256

namespace ppl {
namespace cv {
namespace ocl {

RetCode adaptivethreshold_meanC1U8(const cl_mem src, int rows, int cols,
                                   int src_stride, cl_mem dst, int dst_stride,
                                   float maxValue, int threshold_type,
                                   int ksize, float delta,
                                   BorderType border_type,
                                   cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(threshold_type == THRESH_BINARY ||
             threshold_type == THRESH_BINARY_INV);
  PPL_ASSERT((ksize & 1) == 1 && ksize > 1 && ksize < LARGE_MAX_KSIZE);
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101);
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, adaptivethreshold);
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
  int global_cols, global_rows;
  size_t global_size[2];
  if (border_type == BORDER_REPLICATE) {
    uchar setted_value = 0;
    if (maxValue < 255.f) {
      setted_value = rintf(maxValue);
    }
    else {
      setted_value = 255;
    }
    int int_delta = 0;
    if (threshold_type == THRESH_BINARY) {
      int_delta = std::ceil(delta);
    }
    else {
      int_delta = std::floor(delta);
    }
    float weight = 1.0f / (float)(ksize * ksize);
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};
    global_cols = divideUp(cols, 2, 1);
    global_rows = divideUp(rows, 2, 1);
    global_size[0] = (size_t)global_cols;
    global_size[1] = (size_t)global_rows;
    frame_chain->setCompileOptions(
        "-D ADAPTIVETHRESHOLD_interpolateReplicateBorderMEAN");
    runOclKernel(
        frame_chain,
        "adaptivethreshold_meanU8F32interpolateReplicateBorderC1Kernel", 2,
        global_size, local_size, src, src_stride, src, rows, cols, ksize >> 1,
        src_stride, buffer, rows * (int)sizeof(float), threshold_type, weight,
        int_delta, setted_value);
    global_cols = divideUp(cols, 4, 2);
    global_rows = divideUp(rows, 4, 2);
    global_size[0] = (size_t)global_rows;
    global_size[1] = (size_t)global_cols;
    runOclKernel(
        frame_chain,
        "adaptivethreshold_meanF32U8interpolateReplicateBorderC1Kernel", 2,
        global_size, local_size, src, src_stride, buffer, cols, rows,
        ksize >> 1, rows * (int)sizeof(float), dst, dst_stride, threshold_type,
        weight, int_delta, setted_value);
    if (memoryPoolUsed()) {
      pplOclFree(buffer_block);
    }
    else {
      clReleaseMemObject(buffer);
    }
  }
  else if (border_type == BORDER_REFLECT) {
    uchar setted_value = 0;
    if (maxValue < 255.f) {
      setted_value = rintf(maxValue);
    }
    else {
      setted_value = 255;
    }
    int int_delta = 0;
    if (threshold_type == THRESH_BINARY) {
      int_delta = std::ceil(delta);
    }
    else {
      int_delta = std::floor(delta);
    }
    float weight = 1.0f / (float)(ksize * ksize);
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};
    global_cols = divideUp(cols, 2, 1);
    global_rows = divideUp(rows, 2, 1);
    global_size[0] = (size_t)global_cols;
    global_size[1] = (size_t)global_rows;
    frame_chain->setCompileOptions(
        "-D ADAPTIVETHRESHOLD_interpolateReflectBorderMEAN");
    runOclKernel(frame_chain,
                 "adaptivethreshold_meanU8F32interpolateReflectBorderC1Kernel",
                 2, global_size, local_size, src, src_stride, src, rows, cols,
                 ksize >> 1, src_stride, buffer, rows * (int)sizeof(float),
                 threshold_type, weight, int_delta, setted_value);
    global_cols = divideUp(cols, 4, 2);
    global_rows = divideUp(rows, 4, 2);
    global_size[0] = (size_t)global_rows;
    global_size[1] = (size_t)global_cols;
    runOclKernel(frame_chain,
                 "adaptivethreshold_meanF32U8interpolateReflectBorderC1Kernel",
                 2, global_size, local_size, src, src_stride, buffer, cols,
                 rows, ksize >> 1, rows * (int)sizeof(float), dst, dst_stride,
                 threshold_type, weight, int_delta, setted_value);
    if (memoryPoolUsed()) {
      pplOclFree(buffer_block);
    }
    else {
      clReleaseMemObject(buffer);
    }
  }
  else if (border_type == BORDER_REFLECT_101) {
    uchar setted_value = 0;
    if (maxValue < 255.f) {
      setted_value = rintf(maxValue);
    }
    else {
      setted_value = 255;
    }
    int int_delta = 0;
    if (threshold_type == THRESH_BINARY) {
      int_delta = std::ceil(delta);
    }
    else {
      int_delta = std::floor(delta);
    }
    float weight = 1.0f / (float)(ksize * ksize);
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};
    global_cols = divideUp(cols, 2, 1);
    global_rows = divideUp(rows, 2, 1);
    global_size[0] = (size_t)global_cols;
    global_size[1] = (size_t)global_rows;
    frame_chain->setCompileOptions(
        "-D ADAPTIVETHRESHOLD_interpolateReflect101BorderMEAN");
    runOclKernel(
        frame_chain,
        "adaptivethreshold_meanU8F32interpolateReflect101BorderC1Kernel", 2,
        global_size, local_size, src, src_stride, src, rows, cols, ksize >> 1,
        src_stride, buffer, rows * (int)sizeof(float), threshold_type, weight,
        int_delta, setted_value);
    global_cols = divideUp(cols, 4, 2);
    global_rows = divideUp(rows, 4, 2);
    global_size[0] = (size_t)global_rows;
    global_size[1] = (size_t)global_cols;
    runOclKernel(
        frame_chain,
        "adaptivethreshold_meanF32U8interpolateReflect101BorderC1Kernel", 2,
        global_size, local_size, src, src_stride, buffer, cols, rows,
        ksize >> 1, rows * (int)sizeof(float), dst, dst_stride, threshold_type,
        weight, int_delta, setted_value);
    if (memoryPoolUsed()) {
      pplOclFree(buffer_block);
    }
    else {
      clReleaseMemObject(buffer);
    }
  }
  return RC_SUCCESS;
}

RetCode adaptivethreshold_gaussianblurC1U8(const cl_mem src, int rows, int cols,
                                           int src_stride, cl_mem dst,
                                           int dst_stride, float maxValue,
                                           int threshold_type, int ksize,
                                           float delta, BorderType border_type,
                                           cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(threshold_type == THRESH_BINARY ||
             threshold_type == THRESH_BINARY_INV);
  PPL_ASSERT((ksize & 1) == 1 && ksize > 1 && ksize < LARGE_MAX_KSIZE);
  PPL_ASSERT(border_type == BORDER_REPLICATE || border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101);
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, adaptivethreshold);
  cl_context context = frame_chain->getContext();
  cl_int error_code;
  cl_mem buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
      cols * (int)sizeof(float) * rows * (int)sizeof(float), NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  cl_mem kernel =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     ksize * (int)sizeof(float), NULL, &error_code);
  CHECK_ERROR(error_code, clCreateBuffer);
  int global_cols, global_rows;
  size_t global_size[2];
  if (border_type == BORDER_REPLICATE) {
    uchar setted_value = 0;
    if (maxValue < 255.f) {
      setted_value = rintf(maxValue);
    }
    else {
      setted_value = 255;
    }
    int int_delta = 0;
    if (threshold_type == THRESH_BINARY) {
      int_delta = std::ceil(delta);
    }
    else {
      int_delta = std::floor(delta);
    }
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
    frame_chain->setCompileOptions(
        "-D ADAPTIVETHRESHOLD_interpolateReplicateBorderGAUSSIANBLUE");
    runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
                 0, ksize, kernel, (int)kernel_block.offset);
    ksize = ksize >> 1;
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};
    global_cols = divideUp(cols, 2, 1);
    global_rows = divideUp(rows, 2, 1);
    global_size[0] = (size_t)global_cols;
    global_size[1] = (size_t)global_rows;
    runOclKernel(
        frame_chain,
        "adaptivethreshold_gaussianblurU8F32interpolateReplicateBorderC1Kernel",
        2, global_size, local_size, src, src_stride, src, rows, cols, kernel,
        (int)kernel_block.offset, ksize, src_stride, buffer,
        rows * (int)sizeof(float), threshold_type, int_delta, setted_value);
    global_cols = divideUp(cols, 4, 2);
    global_rows = divideUp(rows, 4, 2);
    global_size[0] = (size_t)global_rows;
    global_size[1] = (size_t)global_cols;
    runOclKernel(
        frame_chain,
        "adaptivethreshold_gaussianblurF32U8interpolateReplicateBorderC1Kernel",
        2, global_size, local_size, src, src_stride, buffer, cols, rows, kernel,
        (int)kernel_block.offset, ksize, rows * (int)sizeof(float), dst,
        dst_stride, threshold_type, int_delta, setted_value);
    if (memoryPoolUsed()) {
      pplOclFree(buffer_block);
      pplOclFree(kernel_block);
    }
    else {
      clReleaseMemObject(buffer);
      clReleaseMemObject(kernel);
    }
  }
  else if (border_type == BORDER_REFLECT) {
    uchar setted_value = 0;
    if (maxValue < 255.f) {
      setted_value = rintf(maxValue);
    }
    else {
      setted_value = 255;
    }
    int int_delta = 0;
    if (threshold_type == THRESH_BINARY) {
      int_delta = std::ceil(delta);
    }
    else {
      int_delta = std::floor(delta);
    }
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
    frame_chain->setCompileOptions(
        "-D ADAPTIVETHRESHOLD_interpolateReflectBorderGAUSSIANBLUE");
    runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
                 0, ksize, kernel, (int)kernel_block.offset);
    ksize = ksize >> 1;
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};
    global_cols = divideUp(cols, 2, 1);
    global_rows = divideUp(rows, 2, 1);
    global_size[0] = (size_t)global_cols;
    global_size[1] = (size_t)global_rows;
    runOclKernel(
        frame_chain,
        "adaptivethreshold_gaussianblurU8F32interpolateReflectBorderC1Kernel",
        2, global_size, local_size, src, src_stride, src, rows, cols, kernel,
        (int)kernel_block.offset, ksize, src_stride, buffer,
        rows * (int)sizeof(float), threshold_type, int_delta, setted_value);
    global_cols = divideUp(cols, 4, 2);
    global_rows = divideUp(rows, 4, 2);
    global_size[0] = (size_t)global_rows;
    global_size[1] = (size_t)global_cols;
    runOclKernel(
        frame_chain,
        "adaptivethreshold_gaussianblurF32U8interpolateReflectBorderC1Kernel",
        2, global_size, local_size, src, src_stride, buffer, cols, rows, kernel,
        (int)kernel_block.offset, ksize, rows * (int)sizeof(float), dst,
        dst_stride, threshold_type, int_delta, setted_value);
    if (memoryPoolUsed()) {
      pplOclFree(buffer_block);
      pplOclFree(kernel_block);
    }
    else {
      clReleaseMemObject(buffer);
      clReleaseMemObject(kernel);
    }
  }
  else if (border_type == BORDER_REFLECT_101) {
    uchar setted_value = 0;
    if (maxValue < 255.f) {
      setted_value = rintf(maxValue);
    }
    else {
      setted_value = 255;
    }
    int int_delta = 0;
    if (threshold_type == THRESH_BINARY) {
      int_delta = std::ceil(delta);
    }
    else {
      int_delta = std::floor(delta);
    }
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
    frame_chain->setCompileOptions(
        "-D ADAPTIVETHRESHOLD_interpolateReflect101BorderGAUSSIANBLUE");
    runOclKernel(frame_chain, "getGaussianKernel", 1, global_size, global_size,
                 0, ksize, kernel, (int)kernel_block.offset);
    ksize = ksize >> 1;
    size_t local_size[] = {kBlockDimX0, kBlockDimY0};
    global_cols = divideUp(cols, 2, 1);
    global_rows = divideUp(rows, 2, 1);
    global_size[0] = (size_t)global_cols;
    global_size[1] = (size_t)global_rows;
    runOclKernel(frame_chain,
                 "adaptivethreshold_"
                 "gaussianblurU8F32interpolateReflect101BorderC1Kernel",
                 2, global_size, local_size, src, src_stride, src, rows, cols,
                 kernel, (int)kernel_block.offset, ksize, src_stride, buffer,
                 rows * (int)sizeof(float), threshold_type, int_delta,
                 setted_value);
    global_cols = divideUp(cols, 4, 2);
    global_rows = divideUp(rows, 4, 2);
    global_size[0] = (size_t)global_rows;
    global_size[1] = (size_t)global_cols;
    runOclKernel(frame_chain,
                 "adaptivethreshold_"
                 "gaussianblurF32U8interpolateReflect101BorderC1Kernel",
                 2, global_size, local_size, src, src_stride, buffer, cols,
                 rows, kernel, (int)kernel_block.offset, ksize,
                 rows * (int)sizeof(float), dst, dst_stride, threshold_type,
                 int_delta, setted_value);
    if (memoryPoolUsed()) {
      pplOclFree(buffer_block);
      pplOclFree(kernel_block);
    }
    else {
      clReleaseMemObject(buffer);
      clReleaseMemObject(kernel);
    }
  }
  return RC_SUCCESS;
}

RetCode AdaptiveThreshold(cl_command_queue queue,
                          int height,
                          int width,
                          int inWidthStride,
                          const cl_mem inData,
                          int outWidthStride,
                          cl_mem outData,
                          float maxValue,
                          int adaptiveMethod,
                          int threshold_type,
                          int blockSize,
                          float delta,
                          BorderType border_type) {
  PPL_ASSERT(adaptiveMethod == ADAPTIVE_THRESH_MEAN_C ||
             adaptiveMethod == ADAPTIVE_THRESH_GAUSSIAN_C);
  inWidthStride *= sizeof(uchar);
  outWidthStride *= sizeof(uchar);
  RetCode code;
  if (adaptiveMethod == ADAPTIVE_THRESH_MEAN_C) {
    code = adaptivethreshold_meanC1U8(
        inData, height, width, inWidthStride, outData, outWidthStride, maxValue,
        threshold_type, blockSize, delta, border_type, queue);
    return code;
  }
  else {
    code = adaptivethreshold_gaussianblurC1U8(
        inData, height, width, inWidthStride, outData, outWidthStride, maxValue,
        threshold_type, blockSize, delta, border_type, queue);
  }
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
