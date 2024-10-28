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

#include "ppl/cv/ocl/calchist.h"

#include "ppl/common/ocl/pplopencl.h"
#include "utility/utility.hpp"

#include "kernels/calchist.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

#define MAX_BLOCKS 128

RetCode calchist(const cl_mem src, int rows, int cols,
              int src_stride, cl_mem dst,
              const cl_mem mask, int mask_stride,
              cl_command_queue queue) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  SET_PROGRAM_SOURCE(frame_chain, calchist);

  int columns = cols;
  cols = divideUp(columns, 4, 2);
  size_t local_size[2];
  size_t global_size[2];

  if (src_stride == columns && mask_stride == columns) {
    columns *= rows;
    local_size[0]  = 256;
    local_size[1]  = 1;
    global_size[0] = std::min((size_t)(MAX_BLOCKS * 256), (size_t)roundUp(cols * rows, 256, 8));
    global_size[1] = 1;

    if (mask == nullptr){
      frame_chain->setCompileOptions("-D CALCHIST_UNMAKED_ALIGHED");
      runOclKernel(frame_chain, "unmaskCalchistKernel0", 2, global_size, local_size, src,
                  columns, dst);
    } 
    else {
      frame_chain->setCompileOptions("-D CALCHIST_MAKED_ALIGHED");
      runOclKernel(frame_chain, "maskCalchistKernel0", 2, global_size, local_size, src,
                  mask, columns, dst);
    }
  }
  else {
    local_size[0]  = kBlockDimX1;
    local_size[1]  = kBlockDimY1;
    global_size[0] = roundUp(cols, kBlockDimX1, kBlockShiftX1);
    global_size[1] = roundUp(rows, kBlockDimY1, kBlockShiftY1);
    global_size[1] = std::min((size_t)(MAX_BLOCKS * kBlockDimX1 * kBlockDimY1 / global_size[0]), 
                         global_size[1]);

    if (mask == nullptr){
      frame_chain->setCompileOptions("-D CALCHIST_UNMAKED_UNALIGHED");
      runOclKernel(frame_chain, "unmaskCalchistKernel1", 2, global_size, local_size, src,
                  src_stride, rows, columns, dst);
    }
    else {
      frame_chain->setCompileOptions("-D CALCHIST_MAKED_UNALIGHED");
      runOclKernel(frame_chain, "maskCalchistKernel1", 2, global_size, local_size, src,
                  src_stride, mask, mask_stride, rows, columns, dst);
    }
  }
  return RC_SUCCESS;
}

RetCode CalcHist(cl_command_queue queue,
                      int height,
                      int width,
                      int inWidthStride,
                      const cl_mem inData,
                      cl_mem outHist,
                      int maskWidthStride,
                      const cl_mem mask) {
  RetCode code = calchist(inData, height, width, inWidthStride, 
                          outHist, mask, maskWidthStride, queue);
  return code;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
