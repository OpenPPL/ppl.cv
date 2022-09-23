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

#include "ppl/cv/ocl/abs.h"

#include "utility.hpp"
#include "ppl/common/ocl/kernel.h"
#include "ppl/common/ocl/framechain.h"

#include "abs.ocl"

using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

RetCode abs(const cl_mem src, int rows, int cols, int channels, int src_stride,
            cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(schar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(schar));

  int columns = cols * channels;
  cols = divideUp(columns, 4, 2);
  size_t block[] = {kBlockDimX0, kBlockDimY0};  // local_size, group
  size_t grid[]  = {(size_t)cols, (size_t)rows}; // global_size
  // struct EnqueueStatus queue_status;
  // queue_status.profiling = false;

  // detect gpu vendor?
  FrameChain frame_chain(queue);
  SET_PROGRAM_SOURCE(frame_chain); 
  compileOclKernels(frame_chain)

  if ((src_stride & 3) == 0 && (dst_stride & 3) == 0) {
    // checkNDrange(2, grid, block, frame_chain);
    runOclKernel(frame_chain, "absKernel0", 2, grid, block, src, rows, 
                 cols, src_stride, dst, dst_stride);
    // absKernel0<<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,
    //                                        dst_stride);
  }
  else if (src_stride == columns && dst_stride == columns) {  // grid[0] limit?
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    block[0] = 256;
    block[1] = 1;
    grid[0] = divideUp(cols, 256, 8);
    grid[1] = 1;
    // absKernel10<<<grid, block, 0, stream>>>(src, columns, dst);
    runOclKernel(frame_chain, "absKernel10", 2, grid, block, src, 
                 columns, dst);
  }
  else {
    runOclKernel(frame_chain, "absKernel10", 2, grid, block, src, rows,
                 cols, src_stride, dst, dst_stride);
    // absKernel11<<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,
                                            // dst_stride);
  }

  // cudaError_t code = cudaGetLastError();
  // if (code != cudaSuccess) {
  //   LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
  //   return RC_DEVICE_RUNTIME_ERROR;
  // }

  return RC_SUCCESS;
}

RetCode abs(const cl_mem src, int rows, int cols, int channels, int src_stride,
            cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  // dim3 block, grid;
  // block.x = kBlockDimX0;
  // block.y = kBlockDimY0;
  // grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  // grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);
  size_t block[] = {kBlockDimX0, kBlockDimY0};
  size_t grid[]  = {(size_t)divideUp(columns, 2, 1), (size_t)rows};
  // struct EnqueueStatus queue_status;
  // queue_status.profiling = false;

  FrameChain frame_chain(queue);
  // KernelDenpendency frame_chain(queue);
  SET_PROGRAM_SOURCE(frame_chain); 
  compileOclKernels(frame_chain)

  if ((src_stride & 7) == 0 && (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    runOclKernel(frame_chain, "absKernel0", 2, grid, block, src, rows,
                 cols, src_stride, dst, dst_stride);
    // absKernel0<<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,
    //                                        dst_stride);
  }
  else {
    runOclKernel(frame_chain, "absKernel1", 2, grid, block, src, rows,
                 columns, src_stride, dst, dst_stride);
    // absKernel1<<<grid, block, 0, stream>>>(src, rows, columns, src_stride, dst,
    //                                        dst_stride);
  }

  // cudaError_t code = cudaGetLastError();
  // if (code != cudaSuccess) {
  //   LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
  //   return RC_DEVICE_RUNTIME_ERROR;
  // }

  return RC_SUCCESS;
}

template <>
RetCode Abs<schar, 1>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = abs(inData, height, width, 1, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<schar, 3>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = abs(inData, height, width, 3, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<schar, 4>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  RetCode code = abs(inData, height, width, 4, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<float, 1>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = abs(inData, height, width, 1, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<float, 3>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = abs(inData, height, width, 3, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<float, 4>(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      int outWidthStride,
                      cl_mem outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = abs(inData, height, width, 4, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
