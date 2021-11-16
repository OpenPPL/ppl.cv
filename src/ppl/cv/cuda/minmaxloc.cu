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

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define BLOCK_SIZE 128
#define BLOCK_SHIFT 7
#define MAX_BLOCKS 256
#define UCHAR_MIN 0

static __device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

__DEVICE__
void checkMinMax1(uchar value, int index, int element_x, int element_y,
                  uchar* minVals, uchar* maxVals, int* minIdxXs, int* minIdxYs,
                  int* maxIdxXs, int* maxIdxYs) {
  if (value < minVals[index]) {
    minVals[index]  = value;
    minIdxXs[index] = element_x;
    minIdxYs[index] = element_y;
  }

  if (value > maxVals[index]) {
    maxVals[index]  = value;
    maxIdxXs[index] = element_x;
    maxIdxYs[index] = element_y;
  }
}

__DEVICE__
void checkMinMax1(float value, int index, int element_x, int element_y,
                  float* minVals, float* maxVals, int* minIdxXs, int* minIdxYs,
                  int* maxIdxXs, int* maxIdxYs) {
  if (value < minVals[index]) {
    minVals[index]  = value;
    minIdxXs[index] = element_x;
    minIdxYs[index] = element_y;
  }

  if (value > maxVals[index]) {
    maxVals[index]  = value;
    maxIdxXs[index] = element_x;
    maxIdxYs[index] = element_y;
  }
}

__DEVICE__
void checkMinMax2(int index1, int index2, uchar* minVals, uchar* maxVals,
                  int* minIdxXs, int* minIdxYs, int* maxIdxXs, int* maxIdxYs) {
  if (minVals[index2] < minVals[index1]) {
    minVals[index1]  = minVals[index2];
    minIdxXs[index1] = minIdxXs[index2];
    minIdxYs[index1] = minIdxYs[index2];
  }
  else if (minVals[index2] == minVals[index1]) {
    if (minIdxYs[index2] < minIdxYs[index1]) {
      minIdxXs[index1] = minIdxXs[index2];
      minIdxYs[index1] = minIdxYs[index2];
    }
    else if (minIdxYs[index2] == minIdxYs[index1]) {
      if (minIdxXs[index2] < minIdxXs[index1]) {
        minIdxXs[index1] = minIdxXs[index2];
        minIdxYs[index1] = minIdxYs[index2];
      }
    }
    else {
    }
  }
  else {
  }

  if (maxVals[index2] > maxVals[index1]) {
    maxVals[index1]  = maxVals[index2];
    maxIdxXs[index1] = maxIdxXs[index2];
    maxIdxYs[index1] = maxIdxYs[index2];
  }
  else if (maxVals[index2] == maxVals[index1]) {
    if (maxIdxYs[index2] < maxIdxYs[index1]) {
      maxIdxXs[index1] = maxIdxXs[index2];
      maxIdxYs[index1] = maxIdxYs[index2];
    }
    else if (maxIdxYs[index2] == maxIdxYs[index1]) {
      if (maxIdxXs[index2] < maxIdxXs[index1]) {
        maxIdxXs[index1] = maxIdxXs[index2];
        maxIdxYs[index1] = maxIdxYs[index2];
      }
    }
    else {
    }
  }
  else {
  }
}

__DEVICE__
void checkMinMax2(int index1, int index2, float* minVals, float* maxVals,
                  int* minIdxXs, int* minIdxYs, int* maxIdxXs, int* maxIdxYs) {
  if (minVals[index2] < minVals[index1]) {
    minVals[index1]  = minVals[index2];
    minIdxXs[index1] = minIdxXs[index2];
    minIdxYs[index1] = minIdxYs[index2];
  }
  else if (minVals[index2] == minVals[index1]) {
    if (minIdxYs[index2] < minIdxYs[index1]) {
      minIdxXs[index1] = minIdxXs[index2];
      minIdxYs[index1] = minIdxYs[index2];
    }
    else if (minIdxYs[index2] == minIdxYs[index1]) {
      if (minIdxXs[index2] < minIdxXs[index1]) {
        minIdxXs[index1] = minIdxXs[index2];
        minIdxYs[index1] = minIdxYs[index2];
      }
    }
    else {
    }
  }
  else {
  }

  if (maxVals[index2] > maxVals[index1]) {
    maxVals[index1]  = maxVals[index2];
    maxIdxXs[index1] = maxIdxXs[index2];
    maxIdxYs[index1] = maxIdxYs[index2];
  }
  else if (maxVals[index2] == maxVals[index1]) {
    if (maxIdxYs[index2] < maxIdxYs[index1]) {
      maxIdxXs[index1] = maxIdxXs[index2];
      maxIdxYs[index1] = maxIdxYs[index2];
    }
    else if (maxIdxYs[index2] == maxIdxYs[index1]) {
      if (maxIdxXs[index2] < maxIdxXs[index1]) {
        maxIdxXs[index1] = maxIdxXs[index2];
        maxIdxYs[index1] = maxIdxYs[index2];
      }
    }
    else {
    }
  }
  else {
  }
}

__DEVICE__
void checkMinMax3(int gindex, int sindex, uchar* g_minVals, uchar* g_maxVals,
                  int* g_minIdxXs, int* g_minIdxYs, int* g_maxIdxXs,
                  int* g_maxIdxYs, uchar* minVals, uchar* maxVals,
                  int* minIdxXs, int* minIdxYs, int* maxIdxXs, int* maxIdxYs) {
  if (g_minVals[gindex] < minVals[sindex]) {
    minVals[sindex]  = g_minVals[gindex];
    minIdxXs[sindex] = g_minIdxXs[gindex];
    minIdxYs[sindex] = g_minIdxYs[gindex];
  }

  if (g_maxVals[gindex] > maxVals[sindex]) {
    maxVals[sindex]  = g_maxVals[gindex];
    maxIdxXs[sindex] = g_maxIdxXs[gindex];
    maxIdxYs[sindex] = g_maxIdxYs[gindex];
  }
}

__DEVICE__
void checkMinMax3(int gindex, int sindex, float* g_minVals, float* g_maxVals,
                  int* g_minIdxXs, int* g_minIdxYs, int* g_maxIdxXs,
                  int* g_maxIdxYs, float* minVals, float* maxVals,
                  int* minIdxXs, int* minIdxYs, int* maxIdxXs, int* maxIdxYs) {
  if (g_minVals[gindex] < minVals[sindex]) {
    minVals[sindex]  = g_minVals[gindex];
    minIdxXs[sindex] = g_minIdxXs[gindex];
    minIdxYs[sindex] = g_minIdxYs[gindex];
  }

  if (g_maxVals[gindex] > maxVals[sindex]) {
    maxVals[sindex]  = g_maxVals[gindex];
    maxIdxXs[sindex] = g_maxIdxXs[gindex];
    maxIdxYs[sindex] = g_maxIdxYs[gindex];
  }
}

__global__
void minMaxLocKernel(const uchar* src, int rows, int cols, int src_stride,
                     const uchar* mask, bool using_mask, int blocks,
                     uchar* g_minVals, uchar* g_maxVals, int* g_minIdxXs,
                     int* g_minIdxYs, int* g_maxIdxXs, int* g_maxIdxYs,
                     int* g_output) {
  __shared__ uchar minVals[BLOCK_SIZE];
  __shared__ uchar maxVals[BLOCK_SIZE];
  __shared__ int minIdxXs[BLOCK_SIZE];
  __shared__ int minIdxYs[BLOCK_SIZE];
  __shared__ int maxIdxXs[BLOCK_SIZE];
  __shared__ int maxIdxYs[BLOCK_SIZE];

  unsigned int threadIdx_x = threadIdx.x;
  unsigned int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  unsigned int element_y = blockIdx.y;

  minVals[threadIdx_x]  = UCHAR_MAX;
  maxVals[threadIdx_x]  = UCHAR_MIN;
  minIdxXs[threadIdx_x] = 0;
  minIdxYs[threadIdx_x] = 0;
  maxIdxXs[threadIdx_x] = 0;
  maxIdxYs[threadIdx_x] = 0;

  int offset;
  uchar* src_start;
  uchar* mask_start;
  uchar value0, value1, value2, value3;
  uchar mvalue0, mvalue1, mvalue2, mvalue3;

  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      offset = element_y * src_stride;
      src_start  = (uchar*)((uchar*)src + offset);
      value0 = src_start[element_x];
      value1 = src_start[element_x + 1];
      value2 = src_start[element_x + 2];
      value3 = src_start[element_x + 3];

      if (using_mask) {
        mask_start = (uchar*)((uchar*)mask + offset);
        mvalue0 = mask_start[element_x];
        mvalue1 = mask_start[element_x + 1];
        mvalue2 = mask_start[element_x + 2];
        mvalue3 = mask_start[element_x + 3];
        if (mvalue0 > 0) {
          checkMinMax1(value0, threadIdx_x, element_x, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (mvalue1 > 0 && element_x + 1 < cols) {
          checkMinMax1(value1, threadIdx_x, element_x + 1, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (mvalue2 > 0 && element_x + 2 < cols) {
          checkMinMax1(value2, threadIdx_x, element_x + 2, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (mvalue3 > 0 && element_x + 3 < cols) {
          checkMinMax1(value3, threadIdx_x, element_x + 3, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
      }
      else {
        checkMinMax1(value0, threadIdx_x, element_x, element_y, minVals,
                     maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        if (element_x + 1 < cols) {
          checkMinMax1(value1, threadIdx_x, element_x + 1, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (element_x + 2 < cols) {
          checkMinMax1(value2, threadIdx_x, element_x + 2, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (element_x + 3 < cols) {
          checkMinMax1(value3, threadIdx_x, element_x + 3, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
      }
    }
  }
  __syncthreads();

  // Do reduction in the shared memory.
#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    checkMinMax2(threadIdx_x, threadIdx_x + 256, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    checkMinMax2(threadIdx_x, threadIdx_x + 128, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    checkMinMax2(threadIdx_x, threadIdx_x + 64, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    checkMinMax2(threadIdx_x, threadIdx_x + 32, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 16, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 8, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 4, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 2, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 1, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
  }

  if (threadIdx_x == 0) {
    offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_minVals[offset]  = minVals[0];
    g_maxVals[offset]  = maxVals[0];
    g_minIdxXs[offset] = minIdxXs[0];
    g_minIdxYs[offset] = minIdxYs[0];
    g_maxIdxXs[offset] = maxIdxXs[0];
    g_maxIdxYs[offset] = maxIdxYs[0];
    __threadfence();

    unsigned int value = atomicInc(&count, blocks);
    isLastBlockDone = (value == (blocks - 1));
  }
  __syncthreads();

  // Do the final reduction in a thread block.
  if (isLastBlockDone) {
    minVals[threadIdx_x]  = UCHAR_MAX;
    maxVals[threadIdx_x]  = UCHAR_MIN;
    minIdxXs[threadIdx_x] = 0;
    minIdxYs[threadIdx_x] = 0;
    maxIdxXs[threadIdx_x] = 0;
    maxIdxYs[threadIdx_x] = 0;

    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      checkMinMax3(element_x, threadIdx_x, g_minVals, g_maxVals, g_minIdxXs,
                   g_minIdxYs, g_maxIdxXs, g_maxIdxYs, minVals, maxVals,
                   minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      checkMinMax2(threadIdx_x, threadIdx_x + 256, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      checkMinMax2(threadIdx_x, threadIdx_x + 128, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      checkMinMax2(threadIdx_x, threadIdx_x + 64, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
    }
    __syncthreads();
#endif

    if (threadIdx_x < 32) {
      checkMinMax2(threadIdx_x, threadIdx_x + 32, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 16, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 8, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 4, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 2, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 1, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
    }

    if (threadIdx_x == 0) {
      g_output[0] = (int)minVals[0];
      g_output[1] = (int)maxVals[0];
      g_output[2] = minIdxXs[0];
      g_output[3] = minIdxYs[0];
      g_output[4] = maxIdxXs[0];
      g_output[5] = maxIdxYs[0];
      count = 0;
    }
  }
}

__global__
void minMaxLocKernel(const float* src, int rows, int cols, int src_stride,
                     const uchar* mask, int mask_stride, bool using_mask,
                     int blocks, float* g_minVals, float* g_maxVals,
                     int* g_minIdxXs, int* g_minIdxYs, int* g_maxIdxXs,
                     int* g_maxIdxYs, float* g_output) {
  __shared__ float minVals[BLOCK_SIZE];
  __shared__ float maxVals[BLOCK_SIZE];
  __shared__ int minIdxXs[BLOCK_SIZE];
  __shared__ int minIdxYs[BLOCK_SIZE];
  __shared__ int maxIdxXs[BLOCK_SIZE];
  __shared__ int maxIdxYs[BLOCK_SIZE];

  unsigned int threadIdx_x = threadIdx.x;
  unsigned int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  unsigned int element_y = blockIdx.y;

  minVals[threadIdx_x]  = FLT_MAX;
  maxVals[threadIdx_x]  = FLT_MIN;
  minIdxXs[threadIdx_x] = 0;
  minIdxYs[threadIdx_x] = 0;
  maxIdxXs[threadIdx_x] = 0;
  maxIdxYs[threadIdx_x] = 0;

  int offset;
  float* src_start;
  uchar* mask_start;
  float value0, value1, value2, value3;
  uchar mvalue0, mvalue1, mvalue2, mvalue3;

  // Read data from the global memory and compare with data in shared memory.
  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      offset = element_y * src_stride;
      src_start  = (float*)((uchar*)src + offset);
      value0 = src_start[element_x];
      value1 = src_start[element_x + 1];
      value2 = src_start[element_x + 2];
      value3 = src_start[element_x + 3];

      if (using_mask) {
        mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
        mvalue0 = mask_start[element_x];
        mvalue1 = mask_start[element_x + 1];
        mvalue2 = mask_start[element_x + 2];
        mvalue3 = mask_start[element_x + 3];
        if (mvalue0 > 0) {
          checkMinMax1(value0, threadIdx_x, element_x, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (mvalue1 > 0 && element_x + 1 < cols) {
          checkMinMax1(value1, threadIdx_x, element_x + 1, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (mvalue2 > 0 && element_x + 2 < cols) {
          checkMinMax1(value2, threadIdx_x, element_x + 2, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (mvalue3 > 0 && element_x + 3 < cols) {
          checkMinMax1(value3, threadIdx_x, element_x + 3, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
      }
      else {
        checkMinMax1(value0, threadIdx_x, element_x, element_y, minVals,
                     maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        if (element_x + 1 < cols) {
          checkMinMax1(value1, threadIdx_x, element_x + 1, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (element_x + 2 < cols) {
          checkMinMax1(value2, threadIdx_x, element_x + 2, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
        if (element_x + 3 < cols) {
          checkMinMax1(value3, threadIdx_x, element_x + 3, element_y, minVals,
                       maxVals, minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
        }
      }
    }
  }
  __syncthreads();

  // Do reduction in the shared memory.
#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    checkMinMax2(threadIdx_x, threadIdx_x + 256, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    checkMinMax2(threadIdx_x, threadIdx_x + 128, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    checkMinMax2(threadIdx_x, threadIdx_x + 64, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    checkMinMax2(threadIdx_x, threadIdx_x + 32, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 16, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 8, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 4, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 2, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
    checkMinMax2(threadIdx_x, threadIdx_x + 1, minVals, maxVals, minIdxXs,
                 minIdxYs, maxIdxXs, maxIdxYs);
  }

  if (threadIdx_x == 0) {
    offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_minVals[offset]  = minVals[0];
    g_maxVals[offset]  = maxVals[0];
    g_minIdxXs[offset] = minIdxXs[0];
    g_minIdxYs[offset] = minIdxYs[0];
    g_maxIdxXs[offset] = maxIdxXs[0];
    g_maxIdxYs[offset] = maxIdxYs[0];
    __threadfence();

    unsigned int value = atomicInc(&count, blocks);
    isLastBlockDone = (value == (blocks - 1));
  }
  __syncthreads();

  // Do the final reduction in a thread block.
  if (isLastBlockDone) {
    minVals[threadIdx_x]  = FLT_MAX;
    maxVals[threadIdx_x]  = FLT_MIN;
    minIdxXs[threadIdx_x] = 0;
    minIdxYs[threadIdx_x] = 0;
    maxIdxXs[threadIdx_x] = 0;
    maxIdxYs[threadIdx_x] = 0;

    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      checkMinMax3(element_x, threadIdx_x, g_minVals, g_maxVals, g_minIdxXs,
                   g_minIdxYs, g_maxIdxXs, g_maxIdxYs, minVals, maxVals,
                   minIdxXs, minIdxYs, maxIdxXs, maxIdxYs);
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      checkMinMax2(threadIdx_x, threadIdx_x + 256, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      checkMinMax2(threadIdx_x, threadIdx_x + 128, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      checkMinMax2(threadIdx_x, threadIdx_x + 64, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
    }
    __syncthreads();
#endif

    if (threadIdx_x < 32) {
      checkMinMax2(threadIdx_x, threadIdx_x + 32, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 16, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 8, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 4, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 2, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
      checkMinMax2(threadIdx_x, threadIdx_x + 1, minVals, maxVals, minIdxXs,
                   minIdxYs, maxIdxXs, maxIdxYs);
    }

    if (threadIdx_x == 0) {
      g_output[0] = minVals[0];
      g_output[1] = maxVals[0];
      g_output[2] = (float)minIdxXs[0];
      g_output[3] = (float)minIdxYs[0];
      g_output[4] = (float)maxIdxXs[0];
      g_output[5] = (float)maxIdxYs[0];
      count = 0;
    }
  }
}

RetCode minMaxLoc(const uchar* src, int rows, int cols, int src_stride,
                  const uchar* mask, int mask_stride, uchar* minVal,
                  uchar* maxVal, int* minIdxX, int* minIdxY, int* maxIdxX,
                  int* maxIdxY, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(rows > 1 && cols > 1);
  PPL_ASSERT(src_stride  >= cols * (int)sizeof(uchar));
  PPL_ASSERT(mask_stride >= 0);
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }
  PPL_ASSERT(minVal != nullptr);
  PPL_ASSERT(maxVal != nullptr);
  PPL_ASSERT(minIdxX != nullptr);
  PPL_ASSERT(minIdxY != nullptr);
  PPL_ASSERT(maxIdxX != nullptr);
  PPL_ASSERT(maxIdxY != nullptr);

  // Each thread processes 4 consecutive elements.
  int col_threads = divideUp(cols, 4, 2);
  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(col_threads, BLOCK_SIZE, BLOCK_SHIFT);
  // run about MAX_BLOCKS thread blocks on a GPU.
  int gy  = MAX_BLOCKS / grid.x;
  grid.y  = (gy < rows) ? gy : rows;

  int blocks = grid.x * grid.y;
  uchar* g_minVals;
  uchar* g_maxVals;
  int* g_minIdxXs;
  int* g_minIdxYs;
  int* g_maxIdxXs;
  int* g_maxIdxYs;
  cudaMalloc(&g_minVals, blocks * sizeof(uchar));
  cudaMalloc(&g_maxVals, blocks * sizeof(uchar));
  cudaMalloc(&g_minIdxXs, blocks * sizeof(int));
  cudaMalloc(&g_minIdxYs, blocks * sizeof(int));
  cudaMalloc(&g_maxIdxXs, blocks * sizeof(int));
  cudaMalloc(&g_maxIdxYs, blocks * sizeof(int));

  int* g_output;
  cudaMalloc(&g_output, 6 * sizeof(int));

  bool using_mask;
  if (mask != nullptr) {
    using_mask = true;
  }
  else {
    using_mask = false;
  }

  minMaxLocKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      mask, using_mask, blocks, g_minVals, g_maxVals, g_minIdxXs, g_minIdxYs,
      g_maxIdxXs, g_maxIdxYs, g_output);

  int values[6];
  cudaMemcpy(values, g_output, 6 * sizeof(int), cudaMemcpyDeviceToHost);
  *minVal = (uchar)values[0];
  *maxVal = (uchar)values[1];
  *minIdxX = values[2];
  *minIdxY = values[3];
  *maxIdxX = values[4];
  *maxIdxY = values[5];

  cudaFree(g_minVals);
  cudaFree(g_maxVals);
  cudaFree(g_minIdxXs);
  cudaFree(g_minIdxYs);
  cudaFree(g_maxIdxXs);
  cudaFree(g_maxIdxYs);
  cudaFree(g_output);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode minMaxLoc(const float* src, int rows, int cols, int src_stride,
                  const uchar* mask, int mask_stride, float* minVal,
                  float* maxVal, int* minIdxX, int* minIdxY, int* maxIdxX,
                  int* maxIdxY, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(rows > 1 && cols > 1);
  PPL_ASSERT(src_stride  >= cols * (int)sizeof(float));
  PPL_ASSERT(mask_stride >= 0);
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }
  PPL_ASSERT(minVal != nullptr);
  PPL_ASSERT(maxVal != nullptr);
  PPL_ASSERT(minIdxX != nullptr);
  PPL_ASSERT(minIdxY != nullptr);
  PPL_ASSERT(maxIdxX != nullptr);
  PPL_ASSERT(maxIdxY != nullptr);

  // Each thread processes 4 consecutive elements.
  int col_threads = divideUp(cols, 4, 2);
  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(col_threads, BLOCK_SIZE, BLOCK_SHIFT);
  // run about MAX_BLOCKS thread blocks on a GPU.
  int gy  = MAX_BLOCKS / grid.x;
  grid.y  = (gy < rows) ? gy : rows;

  int blocks = grid.x * grid.y;
  float* g_minVals;
  float* g_maxVals;
  int* g_minIdxXs;
  int* g_minIdxYs;
  int* g_maxIdxXs;
  int* g_maxIdxYs;
  cudaMalloc(&g_minVals, blocks * sizeof(float));
  cudaMalloc(&g_maxVals, blocks * sizeof(float));
  cudaMalloc(&g_minIdxXs, blocks * sizeof(int));
  cudaMalloc(&g_minIdxYs, blocks * sizeof(int));
  cudaMalloc(&g_maxIdxXs, blocks * sizeof(int));
  cudaMalloc(&g_maxIdxYs, blocks * sizeof(int));

  float* g_output;
  cudaMalloc(&g_output, 6 * sizeof(float));

  bool using_mask;
  if (mask != nullptr) {
    using_mask = true;
    PPL_ASSERT(mask_stride > 0);
  }
  else {
    using_mask = false;
  }

  minMaxLocKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      mask, mask_stride, using_mask, blocks, g_minVals, g_maxVals, g_minIdxXs,
      g_minIdxYs, g_maxIdxXs, g_maxIdxYs, g_output);

  float values[6];
  cudaMemcpy(values, g_output, 6 * sizeof(float), cudaMemcpyDeviceToHost);
  *minVal = values[0];
  *maxVal = values[1];
  *minIdxX = (int)values[2];
  *minIdxY = (int)values[3];
  *maxIdxX = (int)values[4];
  *maxIdxY = (int)values[5];

  cudaFree(g_minVals);
  cudaFree(g_maxVals);
  cudaFree(g_minIdxXs);
  cudaFree(g_minIdxYs);
  cudaFree(g_maxIdxXs);
  cudaFree(g_maxIdxYs);
  cudaFree(g_output);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
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
