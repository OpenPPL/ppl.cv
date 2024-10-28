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

#include "ppl/cv/ocl/equalizehist.h"
#include "utility/use_memory_pool.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/equalizehist.cl"
#include <string.h>

using namespace ppl::common;
using namespace ppl::common::ocl;
using namespace ppl::cv::ocl;

namespace ppl {
namespace cv {
namespace ocl {

#define MAX_BLOCKS 128

RetCode equalizehist(const cl_mem src, int rows, int cols, int src_stride,
                     cl_mem dst, int dst_stride, cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, equalizehist);

  cl_mem hist;
  GpuMemoryBlock buffer_block;
  buffer_block.offset = 0;
  if (memoryPoolUsed()) {
    pplOclMalloc(buffer_block, 256 * sizeof(int));
    hist = buffer_block.data;
  }
  else {
    cl_int error_code = 0;
    hist = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_WRITE,
                          256 * sizeof(int), NULL, &error_code);
    CHECK_ERROR(error_code, clCreateBuffer);
  }

  int columns = cols;
  cols = divideUp(columns, 4, 2);
  size_t local_size[2];
  size_t global_size[2];
  global_size[0] = 256;

  if (src_stride == columns && dst_stride == columns) {
    frame_chain->setCompileOptions("-D EQUALIZEHIST_ALIGNED");

    runOclKernel(frame_chain, "equalizeHistKernel", 1, global_size, global_size,
                 hist, (int)buffer_block.offset);
    columns *= rows;
    local_size[0] = 256;
    local_size[1] = 1;
    global_size[0] =
        std::min((size_t)(MAX_BLOCKS * 256), (size_t)roundUp(columns, 256, 8));
    global_size[1] = 1;
    runOclKernel(frame_chain, "equalizeHistKernel0", 2, global_size, local_size,
                 src, columns, hist, (int)buffer_block.offset);
    runOclKernel(frame_chain, "equalizeHistKernel00", 2, global_size,
                 local_size, src, columns, dst, hist, (int)buffer_block.offset);
  }
  else {
    frame_chain->setCompileOptions("-D EQUALIZEHIST_UNALIGNED");
    runOclKernel(frame_chain, "equalizeHistKernel", 1, global_size, global_size,
                 hist, (int)buffer_block.offset);
    local_size[0] = kBlockDimX1;
    local_size[1] = kBlockDimY1;
    global_size[0] = roundUp(cols, kBlockDimX1, kBlockShiftX1);
    global_size[1] = roundUp(rows, kBlockDimY1, kBlockShiftY1);
    global_size[1] = std::min(
        (size_t)(MAX_BLOCKS * kBlockDimX1 * kBlockDimY1 / global_size[0]),
        global_size[1]);

    runOclKernel(frame_chain, "equalizeHistKernel1", 2, global_size, local_size,
                 src, src_stride, rows, columns, hist,
                 (int)buffer_block.offset);
    runOclKernel(frame_chain, "equalizeHistKernel11", 2, global_size,
                 local_size, src, src_stride, rows, columns, dst, dst_stride,
                 hist, (int)buffer_block.offset);
  }

  if (memoryPoolUsed()) {
    pplOclFree(buffer_block);
  }
  else {
    clReleaseMemObject(hist);
  }

  return RC_SUCCESS;
}

RetCode equalizeHist(cl_command_queue queue,
                     int height,
                     int width,
                     int inWidthStride,
                     const cl_mem inData,
                     int outWidthStride,
                     cl_mem outData) {
  RetCode code = equalizehist(inData, height, width, inWidthStride, outData,
                              outWidthStride, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
