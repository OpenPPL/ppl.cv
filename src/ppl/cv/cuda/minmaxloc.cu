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

#include "ppl/cv/cuda/minmaxloc.h"

#include <cfloat>

#include "utility/utility.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define BLOCK_SIZE 128
#define BLOCK_SHIFT 7
#define MAX_BLOCKS 256
#define UCHAR_MIN 0

static __device__ uint block_count = 0;

template <typename T>
__DEVICE__
void checkMinMax1(T value, int index, int element_x, int element_y, T* min_vals,
                  T* max_vals, int* min_loc_xs, int* min_loc_ys,
                  int* max_loc_xs, int* max_loc_ys) {
  if (value < min_vals[index]) {
    min_vals[index]  = value;
    min_loc_xs[index] = element_x;
    min_loc_ys[index] = element_y;
  }

  if (value > max_vals[index]) {
    max_vals[index]  = value;
    max_loc_xs[index] = element_x;
    max_loc_ys[index] = element_y;
  }
}

template <typename T>
__DEVICE__
void checkMinMax2(int index0, int index1, T* min_vals, T* max_vals,
                  int* min_loc_xs, int* min_loc_ys, int* max_loc_xs,
                  int* max_loc_ys) {
  if (min_vals[index1] < min_vals[index0]) {
    min_vals[index0]  = min_vals[index1];
    min_loc_xs[index0] = min_loc_xs[index1];
    min_loc_ys[index0] = min_loc_ys[index1];
  }
  else if (min_vals[index1] == min_vals[index0]) {
    if (min_loc_ys[index1] < min_loc_ys[index0]) {
      min_loc_xs[index0] = min_loc_xs[index1];
      min_loc_ys[index0] = min_loc_ys[index1];
    }
    else if (min_loc_ys[index1] == min_loc_ys[index0]) {
      if (min_loc_xs[index1] < min_loc_xs[index0]) {
        min_loc_xs[index0] = min_loc_xs[index1];
      }
    }
    else {
    }
  }
  else {
  }

  if (max_vals[index1] > max_vals[index0]) {
    max_vals[index0]  = max_vals[index1];
    max_loc_xs[index0] = max_loc_xs[index1];
    max_loc_ys[index0] = max_loc_ys[index1];
  }
  else if (max_vals[index1] == max_vals[index0]) {
    if (max_loc_ys[index1] < max_loc_ys[index0]) {
      max_loc_xs[index0] = max_loc_xs[index1];
      max_loc_ys[index0] = max_loc_ys[index1];
    }
    else if (max_loc_ys[index1] == max_loc_ys[index0]) {
      if (max_loc_xs[index1] < max_loc_xs[index0]) {
        max_loc_xs[index0] = max_loc_xs[index1];
      }
    }
    else {
    }
  }
  else {
  }
}

template <typename T>
__DEVICE__
void checkMinMax3(int g_index, int sh_index, T* g_min_vals, T* g_max_vals,
                  int* g_min_loc_xs, int* g_min_loc_ys, int* g_max_loc_xs,
                  int* g_max_loc_ys, T* min_vals,  T* max_vals, int* min_loc_xs,
                  int* min_loc_ys, int* max_loc_xs, int* max_loc_ys) {
  if (g_min_vals[g_index] < min_vals[sh_index]) {
    min_vals[sh_index]  = g_min_vals[g_index];
    min_loc_xs[sh_index] = g_min_loc_xs[g_index];
    min_loc_ys[sh_index] = g_min_loc_ys[g_index];
  }
  else if (g_min_vals[g_index] == min_vals[sh_index]) {
    if (g_min_loc_ys[g_index] < min_loc_ys[sh_index]) {
      min_loc_xs[sh_index] = g_min_loc_xs[g_index];
      min_loc_ys[sh_index] = g_min_loc_ys[g_index];
    }
    else if (g_min_loc_ys[g_index] == min_loc_ys[sh_index]) {
      if (g_min_loc_xs[sh_index] < min_loc_xs[g_index]) {
        min_loc_xs[sh_index] = g_min_loc_xs[g_index];
      }
    }
  }
  else {
  }

  if (g_max_vals[g_index] > max_vals[sh_index]) {
    max_vals[sh_index]  = g_max_vals[g_index];
    max_loc_xs[sh_index] = g_max_loc_xs[g_index];
    max_loc_ys[sh_index] = g_max_loc_ys[g_index];
  }
  else if (g_max_vals[g_index] == max_vals[sh_index]) {
    if (g_max_loc_ys[g_index] < max_loc_ys[sh_index]) {
      max_loc_xs[sh_index] = g_max_loc_xs[g_index];
      max_loc_ys[sh_index] = g_max_loc_ys[g_index];
    }
    else if (g_max_loc_ys[g_index] == max_loc_ys[sh_index]) {
      if (g_max_loc_xs[sh_index] < max_loc_xs[g_index]) {
        max_loc_xs[sh_index] = g_max_loc_xs[g_index];
      }
    }
  }
  else {
  }
}

__global__
void minMaxLocKernel(const uchar* src, int rows, int cols, int src_stride,
                     const uchar* mask, int mask_stride, uint blocks,
                     int* buffer) {
  __shared__ uchar min_vals[BLOCK_SIZE];
  __shared__ uchar max_vals[BLOCK_SIZE];
  __shared__ int min_loc_xs[BLOCK_SIZE];
  __shared__ int min_loc_ys[BLOCK_SIZE];
  __shared__ int max_loc_xs[BLOCK_SIZE];
  __shared__ int max_loc_ys[BLOCK_SIZE];

  int threadIdx_x = threadIdx.x;
  int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  int element_y = blockIdx.y;

  min_vals[threadIdx_x]  = UCHAR_MAX;
  max_vals[threadIdx_x]  = UCHAR_MIN;
  min_loc_xs[threadIdx_x] = 0;
  min_loc_ys[threadIdx_x] = 0;
  max_loc_xs[threadIdx_x] = 0;
  max_loc_ys[threadIdx_x] = 0;

  uchar* input;
  uchar* mask_row;
  uchar value0, value1, value2, value3;
  uchar mask_value0, mask_value1, mask_value2, mask_value3;

  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      input = (uchar*)((uchar*)src + element_y * src_stride);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      if (mask == nullptr) {
        checkMinMax1(value0, threadIdx_x, element_x, element_y, min_vals,
                     max_vals, min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
        if (element_x < cols - 1) {
          checkMinMax1(value1, threadIdx_x, element_x + 1, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (element_x < cols - 2) {
          checkMinMax1(value2, threadIdx_x, element_x + 2, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (element_x < cols - 3) {
          checkMinMax1(value3, threadIdx_x, element_x + 3, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
      }
      else {
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mask_value0 = mask_row[element_x];
        mask_value1 = mask_row[element_x + 1];
        mask_value2 = mask_row[element_x + 2];
        mask_value3 = mask_row[element_x + 3];
        if (mask_value0 > 0) {
          checkMinMax1(value0, threadIdx_x, element_x, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (mask_value1 > 0 && element_x < cols - 1) {
          checkMinMax1(value1, threadIdx_x, element_x + 1, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (mask_value2 > 0 && element_x < cols - 2) {
          checkMinMax1(value2, threadIdx_x, element_x + 2, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (mask_value3 > 0 && element_x < cols - 3) {
          checkMinMax1(value3, threadIdx_x, element_x + 3, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
      }
    }
  }
  __syncthreads();

#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    checkMinMax2(threadIdx_x, threadIdx_x + 256, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    checkMinMax2(threadIdx_x, threadIdx_x + 128, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    checkMinMax2(threadIdx_x, threadIdx_x + 64, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    checkMinMax2(threadIdx_x, threadIdx_x + 32, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 16, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 8, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 4, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 2, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 1, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
  }

  __shared__ bool is_last_block_done;
  int block_size = blocks * sizeof(int);
  uchar* g_min_vals = (uchar*)buffer;
  uchar* g_max_vals = (uchar*)buffer + block_size;
  int* g_min_loc_xs = (int*)((uchar*)buffer + 2 * block_size);
  int* g_min_loc_ys = (int*)((uchar*)buffer + 3 * block_size);
  int* g_max_loc_xs = (int*)((uchar*)buffer + 4 * block_size);
  int* g_max_loc_ys = (int*)((uchar*)buffer + 5 * block_size);

  if (threadIdx_x == 0) {
    int offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_min_vals[offset]  = min_vals[0];
    g_max_vals[offset]  = max_vals[0];
    g_min_loc_xs[offset] = min_loc_xs[0];
    g_min_loc_ys[offset] = min_loc_ys[0];
    g_max_loc_xs[offset] = max_loc_xs[0];
    g_max_loc_ys[offset] = max_loc_ys[0];
    __threadfence();

    uint local_count = atomicInc(&block_count, blocks);
    is_last_block_done = (local_count == (blocks - 1));
  }
  __syncthreads();

  if (is_last_block_done) {
    min_vals[threadIdx_x]  = UCHAR_MAX;
    max_vals[threadIdx_x]  = UCHAR_MIN;
    min_loc_xs[threadIdx_x] = 0;
    min_loc_ys[threadIdx_x] = 0;
    max_loc_xs[threadIdx_x] = 0;
    max_loc_ys[threadIdx_x] = 0;

    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      checkMinMax3(element_x, threadIdx_x, g_min_vals, g_max_vals, g_min_loc_xs,
                   g_min_loc_ys, g_max_loc_xs, g_max_loc_ys, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      checkMinMax2(threadIdx_x, threadIdx_x + 256, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      checkMinMax2(threadIdx_x, threadIdx_x + 128, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      checkMinMax2(threadIdx_x, threadIdx_x + 64, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
    }
    __syncthreads();
#endif

    if (threadIdx_x < 32) {
      checkMinMax2(threadIdx_x, threadIdx_x + 32, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 16, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 8, min_vals, max_vals, min_loc_xs,
                   min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 4, min_vals, max_vals, min_loc_xs,
                   min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 2, min_vals, max_vals, min_loc_xs,
                   min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 1, min_vals, max_vals, min_loc_xs,
                   min_loc_ys, max_loc_xs, max_loc_ys);
    }

    if (threadIdx_x == 0) {
      buffer[1] = (int)min_vals[0];
      buffer[2] = (int)max_vals[0];
      buffer[3] = min_loc_xs[0];
      buffer[4] = min_loc_ys[0];
      buffer[5] = max_loc_xs[0];
      buffer[6] = max_loc_ys[0];

      block_count = 0;
    }
  }
}

__global__
void minMaxLocKernel(const float* src, int rows, int cols, int src_stride,
                     const uchar* mask, int mask_stride, uint blocks,
                     float* buffer) {
  __shared__ float min_vals[BLOCK_SIZE];
  __shared__ float max_vals[BLOCK_SIZE];
  __shared__ int min_loc_xs[BLOCK_SIZE];
  __shared__ int min_loc_ys[BLOCK_SIZE];
  __shared__ int max_loc_xs[BLOCK_SIZE];
  __shared__ int max_loc_ys[BLOCK_SIZE];

  int threadIdx_x = threadIdx.x;
  int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  int element_y = blockIdx.y;

  min_vals[threadIdx_x]  = FLT_MAX;
  max_vals[threadIdx_x]  = FLT_MIN;
  min_loc_xs[threadIdx_x] = 0;
  min_loc_ys[threadIdx_x] = 0;
  max_loc_xs[threadIdx_x] = 0;
  max_loc_ys[threadIdx_x] = 0;

  float* input;
  uchar* mask_row;
  float value0, value1, value2, value3;
  uchar mask_value0, mask_value1, mask_value2, mask_value3;

  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      input  = (float*)((uchar*)src + element_y * src_stride);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      if (mask == nullptr) {
        checkMinMax1(value0, threadIdx_x, element_x, element_y, min_vals,
                     max_vals, min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
        if (element_x < cols - 1) {
          checkMinMax1(value1, threadIdx_x, element_x + 1, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (element_x < cols - 2) {
          checkMinMax1(value2, threadIdx_x, element_x + 2, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (element_x < cols - 3) {
          checkMinMax1(value3, threadIdx_x, element_x + 3, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
      }
      else {
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mask_value0 = mask_row[element_x];
        mask_value1 = mask_row[element_x + 1];
        mask_value2 = mask_row[element_x + 2];
        mask_value3 = mask_row[element_x + 3];
        if (mask_value0 > 0) {
          checkMinMax1(value0, threadIdx_x, element_x, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (mask_value1 > 0 && element_x < cols - 1) {
          checkMinMax1(value1, threadIdx_x, element_x + 1, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (mask_value2 > 0 && element_x < cols - 2) {
          checkMinMax1(value2, threadIdx_x, element_x + 2, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
        if (mask_value3 > 0 && element_x < cols - 3) {
          checkMinMax1(value3, threadIdx_x, element_x + 3, element_y, min_vals,
                       max_vals, min_loc_xs, min_loc_ys, max_loc_xs,
                       max_loc_ys);
        }
      }
    }
  }
  __syncthreads();

#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    checkMinMax2(threadIdx_x, threadIdx_x + 256, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    checkMinMax2(threadIdx_x, threadIdx_x + 128, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    checkMinMax2(threadIdx_x, threadIdx_x + 64, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    checkMinMax2(threadIdx_x, threadIdx_x + 32, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 16, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 8, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 4, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 2, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
    checkMinMax2(threadIdx_x, threadIdx_x + 1, min_vals, max_vals, min_loc_xs,
                 min_loc_ys, max_loc_xs, max_loc_ys);
  }

  __shared__ bool is_last_block_done;
  int block_size = blocks * sizeof(float);
  float* g_min_vals = (float*)buffer;
  float* g_max_vals = (float*)((uchar*)buffer + block_size);
  int* g_min_loc_xs = (int*)((uchar*)buffer + 2 * block_size);
  int* g_min_loc_ys = (int*)((uchar*)buffer + 3 * block_size);
  int* g_max_loc_xs = (int*)((uchar*)buffer + 4 * block_size);
  int* g_max_loc_ys = (int*)((uchar*)buffer + 5 * block_size);

  if (threadIdx_x == 0) {
    int offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_min_vals[offset]  = min_vals[0];
    g_max_vals[offset]  = max_vals[0];
    g_min_loc_xs[offset] = min_loc_xs[0];
    g_min_loc_ys[offset] = min_loc_ys[0];
    g_max_loc_xs[offset] = max_loc_xs[0];
    g_max_loc_ys[offset] = max_loc_ys[0];
    __threadfence();

    uint local_count = atomicInc(&block_count, blocks);
    is_last_block_done = (local_count == (blocks - 1));
  }
  __syncthreads();

  if (is_last_block_done) {
    min_vals[threadIdx_x]  = FLT_MAX;
    max_vals[threadIdx_x]  = FLT_MIN;
    min_loc_xs[threadIdx_x] = 0;
    min_loc_ys[threadIdx_x] = 0;
    max_loc_xs[threadIdx_x] = 0;
    max_loc_ys[threadIdx_x] = 0;

    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      checkMinMax3(element_x, threadIdx_x, g_min_vals, g_max_vals, g_min_loc_xs,
                   g_min_loc_ys, g_max_loc_xs, g_max_loc_ys, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      checkMinMax2(threadIdx_x, threadIdx_x + 256, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      checkMinMax2(threadIdx_x, threadIdx_x + 128, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      checkMinMax2(threadIdx_x, threadIdx_x + 64, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
    }
    __syncthreads();
#endif

    if (threadIdx_x < 32) {
      checkMinMax2(threadIdx_x, threadIdx_x + 32, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 16, min_vals, max_vals,
                   min_loc_xs, min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 8, min_vals, max_vals, min_loc_xs,
                   min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 4, min_vals, max_vals, min_loc_xs,
                   min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 2, min_vals, max_vals, min_loc_xs,
                   min_loc_ys, max_loc_xs, max_loc_ys);
      checkMinMax2(threadIdx_x, threadIdx_x + 1, min_vals, max_vals, min_loc_xs,
                   min_loc_ys, max_loc_xs, max_loc_ys);
    }

    if (threadIdx_x == 0) {
      buffer[1] = min_vals[0];
      buffer[2] = max_vals[0];
      buffer[3] = (float)min_loc_xs[0];
      buffer[4] = (float)min_loc_ys[0];
      buffer[5] = (float)max_loc_xs[0];
      buffer[6] = (float)max_loc_ys[0];

      block_count = 0;
    }
  }
}

RetCode minMaxLoc(const uchar* src, int rows, int cols, int src_stride,
                  const uchar* mask, int mask_stride, uchar* min_val,
                  uchar* max_val, int* min_loc_x, int* min_loc_y,
                  int* max_loc_x, int* max_loc_y, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }
  PPL_ASSERT(min_val != nullptr);
  PPL_ASSERT(max_val != nullptr);
  PPL_ASSERT(min_loc_x != nullptr);
  PPL_ASSERT(min_loc_y != nullptr);
  PPL_ASSERT(max_loc_x != nullptr);
  PPL_ASSERT(max_loc_y != nullptr);

  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(divideUp(cols, 4, 2), BLOCK_SIZE, BLOCK_SHIFT);
  int grid_y = MAX_BLOCKS / grid.x;
  grid.y = (grid_y < rows) ? grid_y : rows;

  int blocks = grid.x * grid.y;
  int* buffer;
  size_t buffer_size = blocks * 6 * sizeof(int);
  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMalloc(buffer_size, buffer_block);
    buffer = (int*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&buffer, buffer_size);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  minMaxLocKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      mask, mask_stride, blocks, buffer);
  code = cudaGetLastError();
  if (code != cudaSuccess) {
    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(buffer);
    }
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  int results[7];
  code = cudaMemcpy(results, buffer, 7 * sizeof(int), cudaMemcpyDeviceToHost);
  if (code != cudaSuccess) {
    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(buffer);
    }
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  *min_val = (uchar)results[1];
  *max_val = (uchar)results[2];
  *min_loc_x = results[3];
  *min_loc_y = results[4];
  *max_loc_x = results[5];
  *max_loc_y = results[6];

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
  }

  return RC_SUCCESS;
}

RetCode minMaxLoc(const float* src, int rows, int cols, int src_stride,
                  const uchar* mask, int mask_stride, float* min_val,
                  float* max_val, int* min_loc_x, int* min_loc_y,
                  int* max_loc_x, int* max_loc_y, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }
  PPL_ASSERT(min_val != nullptr);
  PPL_ASSERT(max_val != nullptr);
  PPL_ASSERT(min_loc_x != nullptr);
  PPL_ASSERT(min_loc_y != nullptr);
  PPL_ASSERT(max_loc_x != nullptr);
  PPL_ASSERT(max_loc_y != nullptr);

  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(divideUp(cols, 4, 2), BLOCK_SIZE, BLOCK_SHIFT);
  int grid_y  = MAX_BLOCKS / grid.x;
  grid.y = (grid_y < rows) ? grid_y : rows;

  int blocks = grid.x * grid.y;
  float* buffer;
  size_t buffer_size = blocks * 6 * sizeof(float);
  GpuMemoryBlock buffer_block;
  cudaError_t code;
  if (memoryPoolUsed()) {
    pplCudaMalloc(buffer_size, buffer_block);
    buffer = (float*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&buffer, buffer_size);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  minMaxLocKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      mask, mask_stride, blocks, buffer);
  code = cudaGetLastError();
  if (code != cudaSuccess) {
    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(buffer);
    }
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  float results[7];
  code = cudaMemcpy(results, buffer, 7 * sizeof(float), cudaMemcpyDeviceToHost);
  if (code != cudaSuccess) {
    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(buffer);
    }
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  *min_val = results[1];
  *max_val = results[2];
  *min_loc_x = (int)results[3];
  *min_loc_y = (int)results[4];
  *max_loc_x = (int)results[5];
  *max_loc_y = (int)results[6];

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
  }

  return RC_SUCCESS;
}

template <>
RetCode MinMaxLoc<uchar>(cudaStream_t stream,
                         int height,
                         int width,
                         int inWidthStride,
                         const uchar* inData,
                         uchar* minVal,
                         uchar* maxVal,
                         int* minIdxX,
                         int* minIdxY,
                         int* maxIdxX,
                         int* maxIdxY,
                         int maskWidthStride,
                         const uchar* mask) {
  RetCode code = minMaxLoc(inData, height, width, inWidthStride, mask,
                           maskWidthStride, minVal, maxVal, minIdxX, minIdxY,
                           maxIdxX, maxIdxY, stream);

  return code;
}

template <>
RetCode MinMaxLoc<float>(cudaStream_t stream,
                         int height,
                         int width,
                         int inWidthStride,
                         const float* inData,
                         float* minVal,
                         float* maxVal,
                         int* minIdxX,
                         int* minIdxY,
                         int* maxIdxX,
                         int* maxIdxY,
                         int maskWidthStride,
                         const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = minMaxLoc(inData, height, width, inWidthStride, mask,
                           maskWidthStride, minVal, maxVal, minIdxX, minIdxY,
                           maxIdxX, maxIdxY, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
