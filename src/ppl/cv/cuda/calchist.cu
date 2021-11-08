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

#include "ppl/cv/cuda/calchist.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define BLOCKS_X 2
#define BLOCKS_Y 36

__global__
void calcHistKernel0(const uchar* src, int rows, int cols, int src_stride,
                     int* histogram) {
  __shared__ int local_histogram[256];

  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  
  int index = threadIdx.y * blockDim.x + threadIdx.x;
  local_histogram[index] = 0;
  __syncthreads();

  uchar value0, value1, value2, value3;
  
  int blocks_x = gridDim.x * blockDim.x * 4;
  int blocks_y = gridDim.y * blockDim.y;
  int index_x;
  while (element_y < rows) {
    const uchar* input = src + element_y * src_stride;
    index_x = element_x;
    while (index_x < cols) {
      if (index_x < cols - 3) {
        value0 = input[index_x];
        value1 = input[index_x + 1];
        value2 = input[index_x + 2];
        value3 = input[index_x + 3];
    
        atomicAdd(&local_histogram[value0], 1);
        atomicAdd(&local_histogram[value1], 1);
        atomicAdd(&local_histogram[value2], 1);
        atomicAdd(&local_histogram[value3], 1);
      }
      else {
        value0 = input[index_x];
        if (index_x < cols - 1) {
          value1 = input[index_x + 1];
        }
        if (index_x < cols - 2) {
          value2 = input[index_x + 2];
        }
    
        atomicAdd(&local_histogram[value0], 1);
        if (index_x < cols - 1) {
          atomicAdd(&local_histogram[value1], 1);
        }
        if (index_x < cols - 2) {
          atomicAdd(&local_histogram[value2], 1);
        }
      }
      index_x += blocks_x;
    }
    element_y += blocks_y;
  }
  
  __syncthreads();

  int count = local_histogram[index];
  if (count > 0) {
    atomicAdd(histogram + index, count);
  }
}

__global__
void calcHistKernel1(const uchar* src, int rows, int cols, int src_stride,
                     const uchar* mask, int mask_stride, int* histogram) {
  __shared__ int local_histogram[256];

  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  
  int index = threadIdx.y * blockDim.x + threadIdx.x;
  local_histogram[index] = 0;
  __syncthreads();

  uchar value0, value1, value2, value3;
  uchar mvalue0, mvalue1, mvalue2, mvalue3;
  
  int blocks_x = gridDim.x * blockDim.x * 4;
  int blocks_y = gridDim.y * blockDim.y;
  int index_x;
  while (element_y < rows) {
    const uchar* input = src + element_y * src_stride;
    const uchar* mask_start = mask + element_y * mask_stride;
    index_x = element_x;
    while (index_x < cols) {
      if (index_x < cols - 3) {
        value0 = input[index_x];
        value1 = input[index_x + 1];
        value2 = input[index_x + 2];
        value3 = input[index_x + 3];
  
        mvalue0 = mask_start[index_x];
        mvalue1 = mask_start[index_x + 1];
        mvalue2 = mask_start[index_x + 2];
        mvalue3 = mask_start[index_x + 3];
  
        if (mvalue0 > 0) {
          atomicAdd(&local_histogram[value0], 1);
        }
        if (mvalue1 > 0) {
          atomicAdd(&local_histogram[value1], 1);
        }
        if (mvalue2 > 0) {
          atomicAdd(&local_histogram[value2], 1);
        }
        if (mvalue3 > 0) {
          atomicAdd(&local_histogram[value3], 1);
        }
      }
      else {
        value0 = input[index_x];
        if (index_x < cols - 1) {
          value1 = input[index_x + 1];
        }
        if (index_x < cols - 2) {
          value2 = input[index_x + 2];
        }
  
        mvalue0 = mask_start[index_x];
        if (index_x < cols - 1) {
          mvalue1 = mask_start[index_x + 1];
        }
        if (index_x < cols - 2) {
          mvalue2 = mask_start[index_x + 2];
        }
    
        atomicAdd(&local_histogram[value0], 1);
        if (index_x < cols - 1 && mvalue1 > 0) {
          atomicAdd(&local_histogram[value1], 1);
        }
        if (index_x < cols - 2 && mvalue2 > 0) {
          atomicAdd(&local_histogram[value2], 1);
        }
      }
      index_x += blocks_x;
    }
    element_y += blocks_y;
  }
  __syncthreads();

  int count = local_histogram[index];
  if (count > 0) {
    atomicAdd(histogram + index, count);
  }
}

RetCode calcHist(const uchar* src, int rows, int cols, int src_stride, 
                 int* histogram, const uchar* mask, int mask_stride, 
                 cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(histogram != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  int count_x = kBlockDimX1 * BLOCKS_X * 4;
  int count_y = kBlockDimY1 * BLOCKS_Y;
  if (count_x < cols) {
    grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  }
  else {
    grid.x = BLOCKS_X;
  }
  if (count_y < rows) {
    grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);
  }
  else {
    grid.y = BLOCKS_Y;
  }

  if (mask == nullptr) {
    calcHistKernel0<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
                                                histogram);
  }
  else {
    calcHistKernel1<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
                                                mask, mask_stride, histogram);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode CalcHist<uchar>(cudaStream_t stream,
                        int height,
                        int width,
                        int inWidthStride,
                        const uchar* inData,
                        int* outHist,
                        int maskWidthStride,
                        const uchar* mask) {
  RetCode code = calcHist(inData, height, width, inWidthStride, outHist, 
                          mask, maskWidthStride, stream);
  return code;
}

}  // cuda
}  // cv
}  // ppl    