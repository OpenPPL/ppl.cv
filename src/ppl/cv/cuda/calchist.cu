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

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define MAX_BLOCKS 128

__global__
void unmaskedCalcHistKernel0(const uchar* src, int size, int* histogram) {
  __shared__ int local_histogram[256];

  int element_x = (blockIdx.x << 8) + threadIdx.x;
  int index_x = element_x << 2;
  int grid_offset = gridDim.x * 1024;

  int index = threadIdx.y * blockDim.x + threadIdx.x;
  local_histogram[index] = 0;
  __syncthreads();

  uint* input = (uint*)src;
  for (; index_x < size; index_x += grid_offset) {
    if (index_x < size) {
      uint value = input[element_x];
      if (index_x < size - 3) {
        atomicAdd(&local_histogram[(value >>  0) & 0xFFU], 1);
        atomicAdd(&local_histogram[(value >>  8) & 0xFFU], 1);
        atomicAdd(&local_histogram[(value >> 16) & 0xFFU], 1);
        atomicAdd(&local_histogram[(value >> 24) & 0xFFU], 1);
      }
      else {
        atomicAdd(&local_histogram[(value >>  0) & 0xFFU], 1);
        if (index_x < size - 1) {
          atomicAdd(&local_histogram[(value >>  8) & 0xFFU], 1);
        }
        if (index_x < size - 2) {
          atomicAdd(&local_histogram[(value >> 16) & 0xFFU], 1);
        }
        if (index_x < size - 3) {
          atomicAdd(&local_histogram[(value >> 24) & 0xFFU], 1);
        }
      }
    }
    input += (grid_offset >> 2);
  }
  __syncthreads();

  int count = local_histogram[index];
  if (count > 0) {
    atomicAdd(histogram + index, count);
  }
}

__global__
void unmaskedCalcHistKernel1(const uchar* src, int rows, int cols,
                             int src_stride, int* histogram) {
  __shared__ int local_histogram[256];

  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  int index_x = element_x << 2;

  int index = threadIdx.y * blockDim.x + threadIdx.x;
  local_histogram[index] = 0;
  __syncthreads();

  uint* input;
  for (; element_y < rows; element_y += gridDim.y * blockDim.y) {
    if (index_x < cols) {
      input = (uint*)((uchar*)src + element_y * src_stride);

      uint value = input[element_x];
      if (index_x < cols - 3) {
        atomicAdd(&local_histogram[(value >>  0) & 0xFFU], 1);
        atomicAdd(&local_histogram[(value >>  8) & 0xFFU], 1);
        atomicAdd(&local_histogram[(value >> 16) & 0xFFU], 1);
        atomicAdd(&local_histogram[(value >> 24) & 0xFFU], 1);
      }
      else {
        atomicAdd(&local_histogram[(value >>  0) & 0xFFU], 1);
        if (index_x < cols - 1) {
          atomicAdd(&local_histogram[(value >>  8) & 0xFFU], 1);
        }
        if (index_x < cols - 2) {
          atomicAdd(&local_histogram[(value >> 16) & 0xFFU], 1);
        }
        if (index_x < cols - 3) {
          atomicAdd(&local_histogram[(value >> 24) & 0xFFU], 1);
        }
      }
    }
  }
  __syncthreads();

  int count = local_histogram[index];
  if (count > 0) {
    atomicAdd(histogram + index, count);
  }
}

__global__
void unmaskedCalcHistKernel2(const uchar* src, int rows, int cols,
                             int src_stride, int* histogram) {
  __shared__ int local_histogram[256];

  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;

  int index = threadIdx.y * blockDim.x + threadIdx.x;
  local_histogram[index] = 0;
  __syncthreads();

  uchar* input;
  uchar value0, value1, value2, value3;

  for (; element_y < rows; element_y += gridDim.y * blockDim.y) {
    if (element_x < cols) {
      input = (uchar*)src + element_y * src_stride;

      if (element_x < cols - 3) {
        value0 = input[element_x];
        value1 = input[element_x + 1];
        value2 = input[element_x + 2];
        value3 = input[element_x + 3];

        atomicAdd(&local_histogram[value0], 1);
        atomicAdd(&local_histogram[value1], 1);
        atomicAdd(&local_histogram[value2], 1);
        atomicAdd(&local_histogram[value3], 1);
      }
      else {
        value0 = input[element_x];
        if (element_x < cols - 1) {
          value1 = input[element_x + 1];
        }
        if (element_x < cols - 2) {
          value2 = input[element_x + 2];
        }

        atomicAdd(&local_histogram[value0], 1);
        if (element_x < cols - 1) {
          atomicAdd(&local_histogram[value1], 1);
        }
        if (element_x < cols - 2) {
          atomicAdd(&local_histogram[value2], 1);
        }
      }
    }
  }
  __syncthreads();

  int count = local_histogram[index];
  if (count > 0) {
    atomicAdd(histogram + index, count);
  }
}

__global__
void maskedCalcHistKernel0(const uchar* src, int size, const uchar* mask,
                           int* histogram) {
  __shared__ int local_histogram[256];

  int element_x = (blockIdx.x << 8) + threadIdx.x;
  int index_x = element_x << 2;
  int grid_offset = gridDim.x * 1024;

  int index = threadIdx.y * blockDim.x + threadIdx.x;
  local_histogram[index] = 0;
  __syncthreads();

  uint* input = (uint*)src;
  uint* mask_start = (uint*)mask;
  for (; index_x < size; index_x += grid_offset) {
    if (index_x < size) {
      uint src_value  = input[element_x];
      uint mask_value = mask_start[element_x];

      if (index_x < size - 3) {
        if ((mask_value >>  0) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >>  0) & 0xFFU], 1);
        }
        if ((mask_value >>  8) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >>  8) & 0xFFU], 1);
        }
        if ((mask_value >> 16) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >> 16) & 0xFFU], 1);
        }
        if ((mask_value >> 24) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >> 24) & 0xFFU], 1);
        }
      }
      else {
        if ((mask_value >>  0) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >>  0) & 0xFFU], 1);
        }
        if ((mask_value >>  8) & 0xFFU && index_x < size - 1) {
          atomicAdd(&local_histogram[(src_value >>  8) & 0xFFU], 1);
        }
        if ((mask_value >> 16) & 0xFFU && index_x < size - 2) {
          atomicAdd(&local_histogram[(src_value >> 16) & 0xFFU], 1);
        }
        if ((mask_value >> 24) & 0xFFU && index_x < size - 3) {
          atomicAdd(&local_histogram[(src_value >> 24) & 0xFFU], 1);
        }
      }
    }
    input += (grid_offset >> 2);
    mask_start += (grid_offset >> 2);
  }
  __syncthreads();

  int count = local_histogram[index];
  if (count > 0) {
    atomicAdd(histogram + index, count);
  }
}

__global__
void maskedCalcHistKernel1(const uchar* src, int rows, int cols, int src_stride,
                           const uchar* mask, int mask_stride, int* histogram) {
  __shared__ int local_histogram[256];

  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  int index_x = element_x << 2;

  int index = threadIdx.y * blockDim.x + threadIdx.x;
  local_histogram[index] = 0;
  __syncthreads();

  uint* input;
  uint* mask_start;
  for (; element_y < rows; element_y += gridDim.y * blockDim.y) {
    if (index_x < cols) {
      input = (uint*)((uchar*)src + element_y * src_stride);
      mask_start = (uint*)((uchar*)mask + element_y * mask_stride);

      uint src_value  = input[element_x];
      uint mask_value = mask_start[element_x];

      if (index_x < cols - 3) {
        if ((mask_value >>  0) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >>  0) & 0xFFU], 1);
        }
        if ((mask_value >>  8) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >>  8) & 0xFFU], 1);
        }
        if ((mask_value >> 16) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >> 16) & 0xFFU], 1);
        }
        if ((mask_value >> 24) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >> 24) & 0xFFU], 1);
        }
      }
      else {
        if ((mask_value >>  0) & 0xFFU) {
          atomicAdd(&local_histogram[(src_value >>  0) & 0xFFU], 1);
        }
        if ((mask_value >>  8) & 0xFFU && index_x < cols - 1) {
          atomicAdd(&local_histogram[(src_value >>  8) & 0xFFU], 1);
        }
        if ((mask_value >> 16) & 0xFFU && index_x < cols - 2) {
          atomicAdd(&local_histogram[(src_value >> 16) & 0xFFU], 1);
        }
        if ((mask_value >> 24) & 0xFFU && index_x < cols - 3) {
          atomicAdd(&local_histogram[(src_value >> 24) & 0xFFU], 1);
        }
      }
    }
  }
  __syncthreads();

  int count = local_histogram[index];
  if (count > 0) {
    atomicAdd(histogram + index, count);
  }
}

__global__
void maskedCalcHistKernel2(const uchar* src, int rows, int cols, int src_stride,
                           const uchar* mask, int mask_stride, int* histogram) {
  __shared__ int local_histogram[256];

  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;

  int index = threadIdx.y * blockDim.x + threadIdx.x;
  local_histogram[index] = 0;
  __syncthreads();

  uchar* input;
  uchar* mask_start;
  uchar value0, value1, value2, value3;
  uchar mask_value0, mask_value1, mask_value2, mask_value3;

  for (; element_y < rows; element_y += gridDim.y * blockDim.y) {
    if (element_x < cols) {
      input = (uchar*)src + element_y * src_stride;
      mask_start = (uchar*)mask + element_y * mask_stride;

      if (element_x < cols - 3) {
        value0 = input[element_x];
        value1 = input[element_x + 1];
        value2 = input[element_x + 2];
        value3 = input[element_x + 3];

        mask_value0 = mask_start[element_x];
        mask_value1 = mask_start[element_x + 1];
        mask_value2 = mask_start[element_x + 2];
        mask_value3 = mask_start[element_x + 3];

        if (mask_value0 > 0) {
          atomicAdd(&local_histogram[value0], 1);
        }
        if (mask_value1 > 0) {
          atomicAdd(&local_histogram[value1], 1);
        }
        if (mask_value2 > 0) {
          atomicAdd(&local_histogram[value2], 1);
        }
        if (mask_value3 > 0) {
          atomicAdd(&local_histogram[value3], 1);
        }
      }
      else {
        value0 = input[element_x];
        if (element_x < cols - 1) {
          value1 = input[element_x + 1];
        }
        if (element_x < cols - 2) {
          value2 = input[element_x + 2];
        }

        mask_value0 = mask_start[element_x];
        if (element_x < cols - 1) {
          mask_value1 = mask_start[element_x + 1];
        }
        if (element_x < cols - 2) {
          mask_value2 = mask_start[element_x + 2];
        }

        if (mask_value0) {
          atomicAdd(&local_histogram[value0], 1);
        }
        if (element_x < cols - 1 && mask_value1 > 0) {
          atomicAdd(&local_histogram[value1], 1);
        }
        if (element_x < cols - 2 && mask_value2 > 0) {
          atomicAdd(&local_histogram[value2], 1);
        }
      }
    }
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
  if (src_stride == cols) {
    block.x = 256;
    block.y = 1;
    grid.x = divideUp(cols * rows, 256, 8);
    if (grid.x > MAX_BLOCKS) {
      grid.x = MAX_BLOCKS;
    }
    grid.y = 1;
  }
  else {
    int columns = divideUp(cols, 4, 2);
    block.x = kBlockDimX1;
    block.y = kBlockDimY1;
    grid.x = divideUp(columns, kBlockDimX1, kBlockShiftX1);
    grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);
    uint grid_y = MAX_BLOCKS / grid.x;
    grid.y = (grid_y < grid.y) ? grid_y : grid.y;
  }

  if (mask == nullptr) {
    if (src_stride == cols) {
      unmaskedCalcHistKernel0<<<grid, block, 0, stream>>>(src, rows * cols,
          histogram);
    }
    else if ((src_stride & 3) == 0) {
      unmaskedCalcHistKernel1<<<grid, block, 0, stream>>>(src, rows, cols,
          src_stride, histogram);
    }
    else {
      unmaskedCalcHistKernel2<<<grid, block, 0, stream>>>(src, rows, cols,
          src_stride, histogram);
    }
  }
  else {
    if (src_stride == cols && mask_stride == cols) {
      maskedCalcHistKernel0<<<grid, block, 0, stream>>>(src, rows * cols, mask,
          histogram);
    }
    else if ((src_stride & 3) == 0 && (mask_stride & 3) == 0) {
      maskedCalcHistKernel1<<<grid, block, 0, stream>>>(src, rows, cols,
          src_stride, mask, mask_stride, histogram);
    }
    else {
      maskedCalcHistKernel2<<<grid, block, 0, stream>>>(src, rows, cols,
          src_stride, mask, mask_stride, histogram);
    }
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