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

#include "ppl/cv/cuda/equalizehist.h"

#include "utility/utility.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

static __device__ uint block_count = 0;

#define MAX_BLOCKS 128

__global__
void calcHistKernel0(const uchar* src, int rows, int cols, int src_stride,
                     uint blocks, int elements, int hist_size, int* histogram) {
  __shared__ int local_histogram[256];
  __shared__ bool is_last_block_done;

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
  local_histogram[index] = 0;
  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    uint local_count = atomicInc(&block_count, blocks);
    is_last_block_done = (local_count == (blocks - 1));
    if (is_last_block_done) {
      int i = 0;
      while (!histogram[i]) ++i;
      float scale = (hist_size - 1.f) / (elements - histogram[i]);

      int sum = 0;
      for (local_histogram[i++] = 0; i < hist_size; ++i) {
        sum += histogram[i];
        local_histogram[i]= rintf(sum * scale);
      }
    }
  }
  __syncthreads();

  if (is_last_block_done) {
    histogram[index] = local_histogram[index];
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      block_count = 0;
    }
  }
}

__global__
void calcHistKernel1(const uchar* src, int rows, int cols, int src_stride,
                     uint blocks, int elements, int hist_size, int* histogram) {
  __shared__ int local_histogram[256];
  __shared__ bool is_last_block_done;

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
  local_histogram[index] = 0;
  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    uint local_count = atomicInc(&block_count, blocks);
    is_last_block_done = (local_count == (blocks - 1));
    if (is_last_block_done) {
      int i = 0;
      while (!histogram[i]) ++i;
      float scale = (hist_size - 1.f) / (elements - histogram[i]);

      int sum = 0;
      for (local_histogram[i++] = 0; i < hist_size; ++i) {
        sum += histogram[i];
        local_histogram[i]= rintf(sum * scale);
      }
    }
  }
  __syncthreads();

  if (is_last_block_done) {
    histogram[index] = local_histogram[index];
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      block_count = 0;
    }
  }
}

__global__
void equalizehistKernel(const uchar* src, int rows, int cols, int src_stride,
                        const int* histogram, uchar* dst, int dst_stride) {
  __shared__ int lookup_table[256];
  int index = threadIdx.y * blockDim.x + threadIdx.x;
  lookup_table[index] = histogram[index];
  __syncthreads();

  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  uchar* input = (uchar*)((uchar*)src + element_y * src_stride);
  uchar value0, value1, value2, value3;
  value0 = input[element_x];
  value1 = input[element_x + 1];
  value2 = input[element_x + 2];
  value3 = input[element_x + 3];

  uchar* output = (uchar*)((uchar*)dst + element_y * dst_stride);
  if (element_x < cols - 3) {
    output[element_x] = lookup_table[value0];
    output[element_x + 1] = lookup_table[value1];
    output[element_x + 2] = lookup_table[value2];
    output[element_x + 3] = lookup_table[value3];
  }
  else {
    output[element_x] = lookup_table[value0];
    if (element_x < cols - 1) {
      output[element_x + 1] = lookup_table[value1];
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = lookup_table[value2];
    }
  }
}

RetCode EqualizeHist(cudaStream_t stream, int rows, int cols, int src_stride,
                     const uchar* src, int dst_stride, uchar* dst) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);
  uint grid_y = MAX_BLOCKS / grid.x;
  grid.y = (grid_y < grid.y) ? grid_y : grid.y;

  int hist_size = 256;
  int* histogram;
  GpuMemoryBlock buffer_block;
  cudaError_t code;
  if (memoryPoolUsed()) {
    pplCudaMalloc(hist_size * sizeof(int), buffer_block);
    histogram = (int*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&histogram, hist_size * sizeof(int));
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }
  cudaMemset(histogram, 0, hist_size * sizeof(int));

  uint blocks = grid.y * grid.x;
  if ((src_stride & 3) == 0) {
    calcHistKernel0<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
        blocks, rows * cols, hist_size, histogram);
  }
  else {
    calcHistKernel1<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
        blocks, rows * cols, hist_size, histogram);
  }

  grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);
  equalizehistKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      histogram, dst, dst_stride);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(histogram);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  else {
    return RC_SUCCESS;
  }
}

}  // cuda
}  // cv
}  // ppl
