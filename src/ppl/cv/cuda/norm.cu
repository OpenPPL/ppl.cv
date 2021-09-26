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

#include "ppl/cv/cuda/norm.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

// BLOCK_SIZE must be 512, 256, 128, 64, 32.
#define BLOCK_SIZE 128
#define BLOCK_SHIFT 7
#define MAX_BLOCKS 256

static __device__ uint count = 0;
__shared__ bool isLastBlockDone;

__DEVICE__
void checkMax1(uchar* maxs, int index, uchar value) {
  if (maxs[index] < value) {
    maxs[index] = value;
  }
}

__DEVICE__
void checkMax1(float* maxs, int index, float value) {
  value = fabsf(value);
  if (maxs[index] < value) {
    maxs[index] = value;
  }
}

__DEVICE__
void checkMin1(uchar* mins, int index, float value) {
  if (mins[index] > value) {
    mins[index] = value;
  }
}

__DEVICE__
void checkMin1(float* mins, int index, float value) {
  value = fabsf(value);
  if (mins[index] > value) {
    mins[index] = value;
  }
}

__DEVICE__
void checkMax2(uchar* maxs, int index1, int index2) {
  if (maxs[index1] < maxs[index2]) {
    maxs[index1] = maxs[index2];
  }
}

__DEVICE__
void checkMax2(float* maxs, int index1, int index2) {
  if (maxs[index1] < maxs[index2]) {
    maxs[index1] = maxs[index2];
  }
}

__DEVICE__
void checkMin2(uchar* mins, int index1, int index2) {
  if (mins[index1] > mins[index2]) {
    mins[index1] = mins[index2];
  }
}

__DEVICE__
void checkMin2(float* mins, int index1, int index2) {
  if (mins[index1] > mins[index2]) {
    mins[index1] = mins[index2];
  }
}

__DEVICE__
void checkMax3(uchar* maxs, int index, uchar value) {
  if (maxs[index] < value) {
    maxs[index] = value;
  }
}

__DEVICE__
void checkMax3(float* maxs, int index, float value) {
  if (maxs[index] < value) {
    maxs[index] = value;
  }
}

__DEVICE__
void checkMin3(float* mins, int index, float value) {
  if (mins[index] > value) {
    mins[index] = value;
  }
}

template <typename Tsrc, typename Tdst>
__global__
void normLinfKernel(const Tsrc* src, int rows, int cols, int channels,
                    int src_stride, const uchar* mask, int mask_stride,
                    bool using_mask, int blocks, Tdst* g_norms) {
  __shared__ Tsrc partial_norms[BLOCK_SIZE];

  unsigned int threadIdx_x = threadIdx.x;
  unsigned int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  unsigned int element_y = blockIdx.y;

  partial_norms[threadIdx_x] = 0;

  int offset;
  Tsrc* input;
  Tsrc value0, value1, value2, value3;

  // Reads data from the global memory and reduces in shared memory.
  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      offset = element_y * src_stride;
      input  = (Tsrc*)((uchar*)src + offset);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      if (using_mask) {
        uchar* mask_row;
        uchar mvalue0, mvalue1, mvalue2, mvalue3;
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mvalue0 = mask_row[element_x / channels];
        mvalue1 = mask_row[(element_x + 1) / channels];
        mvalue2 = mask_row[(element_x + 2) / channels];
        mvalue3 = mask_row[(element_x + 3) / channels];
        if (mvalue0 > 0) {
          checkMax1(partial_norms, threadIdx_x, value0);
        }
        if (mvalue1 > 0 && element_x + 1 < cols) {
          checkMax1(partial_norms, threadIdx_x, value1);
        }
        if (mvalue2 > 0 && element_x + 2 < cols) {
        checkMax1(partial_norms, threadIdx_x, value2);
        }
        if (mvalue3 > 0 && element_x + 3 < cols) {
          checkMax1(partial_norms, threadIdx_x, value3);
        }
      }
      else {
        checkMax1(partial_norms, threadIdx_x, value0);
        if (element_x + 1 < cols) {
          checkMax1(partial_norms, threadIdx_x, value1);
        }
        if (element_x + 2 < cols) {
        checkMax1(partial_norms, threadIdx_x, value2);
        }
        if (element_x + 3 < cols) {
          checkMax1(partial_norms, threadIdx_x, value3);
        }
      }
    }
  }
  __syncthreads();

  // Does reduction in the shared memory.
#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 256);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 128);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 64);
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 32);
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 16);
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 8);
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 4);
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 2);
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    checkMax2(partial_norms, threadIdx_x, threadIdx_x + 1);
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_norms[offset] = partial_norms[0];
    __threadfence();

    unsigned int value = atomicInc(&count, blocks);
    isLastBlockDone = (value == (blocks - 1));
  }
  __syncthreads();

  // Does the final reduction in a thread block.
  if (isLastBlockDone) {
    partial_norms[threadIdx_x] = 0;

    Tsrc reduced_norm;
    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      reduced_norm = g_norms[element_x];
      checkMax3(partial_norms, threadIdx_x, reduced_norm);
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 256);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 128);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 64);
    }
    __syncthreads();
#endif

    if (threadIdx_x < 32) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 32);
    }
    __syncthreads();
    if (threadIdx_x < 16) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 16);
    }
    __syncthreads();
    if (threadIdx_x < 8) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 8);
    }
    __syncthreads();
    if (threadIdx_x < 4) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 4);
    }
    __syncthreads();
    if (threadIdx_x < 2) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 2);
    }
    __syncthreads();
    if (threadIdx_x < 1) {
      checkMax2(partial_norms, threadIdx_x, threadIdx_x + 1);
    }
    __syncthreads();

    if (threadIdx_x == 0) {
      g_norms[0] = partial_norms[0];
      count = 0;
    }
  }
}



__global__
void normMinMaxKernel(const uchar* src, int rows, int cols, int channels,
                      int src_stride, const uchar* mask, int mask_stride,
                      bool using_mask, int blocks, long* g_norms)
{
  __shared__ uchar partial_maxs[BLOCK_SIZE];
  __shared__ uchar partial_mins[BLOCK_SIZE];

  unsigned int threadIdx_x = threadIdx.x;
  unsigned int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  unsigned int element_y = blockIdx.y;

  partial_maxs[threadIdx_x] = 0;
  partial_mins[threadIdx_x] = 255;

  int offset;
  uchar* input;
  uchar* mask_row;
  uchar value0, value1, value2, value3;
  uchar mvalue0, mvalue1, mvalue2, mvalue3;

  // Reads data from the global memory and reduces in shared memory.
  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      offset = element_y * src_stride;
      input  = (uchar*)((uchar*)src + offset);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      if (using_mask) {
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mvalue0 = mask_row[element_x / channels];
        mvalue1 = mask_row[(element_x + 1) / channels];
        mvalue2 = mask_row[(element_x + 2) / channels];
        mvalue3 = mask_row[(element_x + 3) / channels];
        if (mvalue0 > 0) {
          checkMax1(partial_maxs, threadIdx_x, value0);
          checkMin1(partial_mins, threadIdx_x, value0);
        }
        if (mvalue1 > 0 && element_x + 1 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value1);
          checkMin1(partial_mins, threadIdx_x, value1);
        }
        if (mvalue2 > 0 && element_x + 2 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value2);
          checkMin1(partial_mins, threadIdx_x, value2);
        }
        if (mvalue3 > 0 && element_x + 3 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value3);
          checkMin1(partial_mins, threadIdx_x, value3);
        }
      }
      else {
        checkMax1(partial_maxs, threadIdx_x, value0);
        checkMin1(partial_mins, threadIdx_x, value0);
        if (element_x + 1 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value1);
          checkMin1(partial_mins, threadIdx_x, value1);
        }
        if (element_x + 2 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value2);
          checkMin1(partial_mins, threadIdx_x, value2);
        }
        if (element_x + 3 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value3);
          checkMin1(partial_mins, threadIdx_x, value3);
        }
      }
    }
  }
  __syncthreads();

  // Does reduction in the shared memory.
#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 256);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 256);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 128);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 128);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 64);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 64);
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 32);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 32);
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 16);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 16);
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 8);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 8);
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 4);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 4);
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 2);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 2);
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 1);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 1);
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_norms[offset] = partial_maxs[0];
    g_norms[offset + blocks] = partial_mins[0];
    __threadfence();

    unsigned int value = atomicInc(&count, blocks);
    isLastBlockDone = (value == (blocks - 1));
  }
  __syncthreads();

  // Does the final reduction in a thread block.
  if (isLastBlockDone) {
    partial_maxs[threadIdx_x] = 0;
    partial_mins[threadIdx_x] = 255;

    uchar reduced_max, reduced_min;
    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      reduced_max = g_norms[element_x];
      reduced_min = g_norms[element_x + blocks];
      checkMax1(partial_maxs, threadIdx_x, reduced_max);
      checkMin1(partial_mins, threadIdx_x, reduced_min);
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 256);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 256);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 128);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 128);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 64);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 64);
    }
    __syncthreads();
    #endif

    if (threadIdx_x < 32) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 32);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 32);
    }
    __syncthreads();
    if (threadIdx_x < 16) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 16);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 16);
    }
    __syncthreads();
    if (threadIdx_x < 8) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 8);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 8);
    }
    __syncthreads();
    if (threadIdx_x < 4) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 4);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 4);
    }
    __syncthreads();
    if (threadIdx_x < 2) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 2);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 2);
    }
    __syncthreads();
    if (threadIdx_x < 1) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 1);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 1);
    }
    __syncthreads();

    if (threadIdx_x == 0) {
      g_norms[0] = partial_maxs[0];
      g_norms[1] = partial_mins[0];
      count = 0;
    }
  }
}

__global__
void normMinMaxKernel(const float* src, int rows, int cols, int channels,
                      int src_stride, const uchar* mask, int mask_stride,
                      bool using_mask, int blocks, double* g_norms)
{
  __shared__ float partial_maxs[BLOCK_SIZE];
  __shared__ float partial_mins[BLOCK_SIZE];

  unsigned int threadIdx_x = threadIdx.x;
  unsigned int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  unsigned int element_y = blockIdx.y;

  partial_maxs[threadIdx_x] = 0.f;  // FLT_MIN?
  partial_mins[threadIdx_x] = 1.f;  // FLT_MAX?

  int offset;
  float* input;
  uchar* mask_row;
  float value0, value1, value2, value3;
  uchar mvalue0, mvalue1, mvalue2, mvalue3;

  // Reads data from the global memory and reduces in shared memory.
  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      offset = element_y * src_stride;
      input  = (float*)((uchar*)src + offset);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      if (using_mask) {
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mvalue0 = mask_row[element_x / channels];
        mvalue1 = mask_row[(element_x + 1) / channels];
        mvalue2 = mask_row[(element_x + 2) / channels];
        mvalue3 = mask_row[(element_x + 3) / channels];
        if (mvalue0 > 0) {
          checkMax1(partial_maxs, threadIdx_x, value0);
          checkMin1(partial_mins, threadIdx_x, value0);
        }
        if (mvalue1 > 0 && element_x + 1 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value1);
          checkMin1(partial_mins, threadIdx_x, value1);
        }
        if (mvalue2 > 0 && element_x + 2 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value2);
          checkMin1(partial_mins, threadIdx_x, value2);
        }
        if (mvalue3 > 0 && element_x + 3 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value3);
          checkMin1(partial_mins, threadIdx_x, value3);
        }
      }
      else {
        checkMax1(partial_maxs, threadIdx_x, value0);
        checkMin1(partial_mins, threadIdx_x, value0);
        if (element_x + 1 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value1);
          checkMin1(partial_mins, threadIdx_x, value1);
        }
        if (element_x + 2 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value2);
          checkMin1(partial_mins, threadIdx_x, value2);
        }
        if (element_x + 3 < cols) {
          checkMax1(partial_maxs, threadIdx_x, value3);
          checkMin1(partial_mins, threadIdx_x, value3);
        }
      }
    }
  }
  __syncthreads();

  // Does reduction in the shared memory.
#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 256);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 256);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 128);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 128);
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 64);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 64);
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 32);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 32);
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 16);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 16);
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 8);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 8);
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 4);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 4);
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 2);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 2);
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 1);
    checkMin2(partial_mins, threadIdx_x, threadIdx_x + 1);
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_norms[offset] = partial_maxs[0];
    g_norms[offset + blocks] = partial_mins[0];
    __threadfence();

    unsigned int value = atomicInc(&count, blocks);
    isLastBlockDone = (value == (blocks - 1));
  }
  __syncthreads();

  // Does the final reduction in a thread block.
  if (isLastBlockDone) {
    partial_maxs[threadIdx_x] = 0.f;  // FLT_MIN?
    partial_mins[threadIdx_x] = 1.f;  // FLT_MAX?

    float reduced_max, reduced_min;
    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      reduced_max = g_norms[element_x];
      reduced_min = g_norms[element_x + blocks];
      checkMax3(partial_maxs, threadIdx_x, reduced_max);
      checkMin3(partial_mins, threadIdx_x, reduced_min);
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 256);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 256);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 128);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 128);
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 64);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 64);
    }
    __syncthreads();
    #endif

    if (threadIdx_x < 32) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 32);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 32);
    }
    __syncthreads();
    if (threadIdx_x < 16) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 16);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 16);
    }
    __syncthreads();
    if (threadIdx_x < 8) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 8);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 8);
    }
    __syncthreads();
    if (threadIdx_x < 4) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 4);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 4);
    }
    __syncthreads();
    if (threadIdx_x < 2) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 2);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 2);
    }
    __syncthreads();
    if (threadIdx_x < 1) {
      checkMax2(partial_maxs, threadIdx_x, threadIdx_x + 1);
      checkMin2(partial_mins, threadIdx_x, threadIdx_x + 1);
    }
    __syncthreads();

    if (threadIdx_x == 0) {
      g_norms[0] = partial_maxs[0];
      g_norms[1] = partial_mins[0];
      count = 0;
    }
  }
}

template <typename Tsrc, typename Tsum, typename Tdst>
__global__
void normL1Kernel(const Tsrc* src, int rows, int cols, int channels,
                  int src_stride, const uchar* mask, int mask_stride,
                  bool using_mask, int blocks, Tdst* g_norms) {
  __shared__ Tsum partial_norms[BLOCK_SIZE];

  unsigned int threadIdx_x = threadIdx.x;
  unsigned int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  unsigned int element_y = blockIdx.y;

  partial_norms[threadIdx_x] = 0;

  int offset;
  Tsrc* input;
  Tsrc value0, value1, value2, value3;

  // Reads data from the global memory and reduces in shared memory.
  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      offset = element_y * src_stride;
      input  = (Tsrc*)((uchar*)src + offset);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      if (using_mask) {
        uchar* mask_row;
        uchar mvalue0, mvalue1, mvalue2, mvalue3;
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mvalue0 = mask_row[element_x / channels];
        mvalue1 = mask_row[(element_x + 1) / channels];
        mvalue2 = mask_row[(element_x + 2) / channels];
        mvalue3 = mask_row[(element_x + 3) / channels];
        if (mvalue0 > 0) {
          partial_norms[threadIdx_x] += value0;
        }
        if (mvalue1 > 0 && element_x + 1 < cols) {
          partial_norms[threadIdx_x] += value1;
        }
        if (mvalue2 > 0 && element_x + 2 < cols) {
        partial_norms[threadIdx_x] += value2;
        }
        if (mvalue3 > 0 && element_x + 3 < cols) {
          partial_norms[threadIdx_x] += value3;
        }
      }
      else {
        partial_norms[threadIdx_x] += value0;
        if (element_x + 1 < cols) {
          partial_norms[threadIdx_x] += value1;
        }
        if (element_x + 2 < cols) {
          partial_norms[threadIdx_x] += value2;
        }
        if (element_x + 3 < cols) {
          partial_norms[threadIdx_x] += value3;
        }
      }
    }
  }
  __syncthreads();

  // Does reduction in the shared memory.
#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 256];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 128];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 64];
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 32];
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 16];
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 8];
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 4];
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 2];
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 1];
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_norms[offset] = partial_norms[0];
    __threadfence();

    unsigned int value = atomicInc(&count, blocks);
    isLastBlockDone = (value == (blocks - 1));
  }
  __syncthreads();

  // Does the final reduction in a thread block.
  if (isLastBlockDone) {
    partial_norms[threadIdx_x] = 0;

    Tsum reduced_norm;
    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      reduced_norm = g_norms[element_x];
      partial_norms[threadIdx_x] += reduced_norm;
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 256];
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 128];
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 64];
    }
    __syncthreads();
#endif

    if (threadIdx_x < 32) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 32];
    }
    __syncthreads();
    if (threadIdx_x < 16) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 16];
    }
    __syncthreads();
    if (threadIdx_x < 8) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 8];
    }
    __syncthreads();
    if (threadIdx_x < 4) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 4];
    }
    __syncthreads();
    if (threadIdx_x < 2) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 2];
    }
    __syncthreads();
    if (threadIdx_x < 1) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 1];
    }
    __syncthreads();

    if (threadIdx_x == 0) {
      g_norms[0] = partial_norms[0];
      count = 0;
    }
  }
}

template <typename Tsrc, typename Tsum, typename Tdst>
__global__
void normL2Kernel(const Tsrc* src, int rows, int cols, int channels,
                  int src_stride, const uchar* mask, int mask_stride,
                  bool using_mask, int blocks, Tdst* g_norms) {
  __shared__ Tsum partial_norms[BLOCK_SIZE];

  int threadIdx_x = threadIdx.x;
  int element_x = ((blockIdx.x << BLOCK_SHIFT) + threadIdx_x) << 2;
  int element_y = blockIdx.y;

  partial_norms[threadIdx_x] = 0;

  int offset;
  Tsrc* input;
  Tsrc value0, value1, value2, value3;

  // Reads data from the global memory and reduces in shared memory.
  for (; element_y < rows; element_y += gridDim.y) {
    if (element_x < cols) {
      offset = element_y * src_stride;
      input  = (Tsrc*)((uchar*)src + offset);
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];

      if (using_mask) {
        uchar* mask_row;
        uchar mvalue0, mvalue1, mvalue2, mvalue3;
        mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
        mvalue0 = mask_row[element_x / channels];
        mvalue1 = mask_row[(element_x + 1) / channels];
        mvalue2 = mask_row[(element_x + 2) / channels];
        mvalue3 = mask_row[(element_x + 3) / channels];
        if (mvalue0 > 0) {
          partial_norms[threadIdx_x] += value0 * value0;
        }
        if (mvalue1 > 0 && element_x + 1 < cols) {
          partial_norms[threadIdx_x] += value1 * value1;
        }
        if (mvalue2 > 0 && element_x + 2 < cols) {
          partial_norms[threadIdx_x] += value2 * value2;
        }
        if (mvalue3 > 0 && element_x + 3 < cols) {
          partial_norms[threadIdx_x] += value3 * value3;
        }
      }
      else {
        partial_norms[threadIdx_x] += value0 * value0;
        if (element_x + 1 < cols) {
          partial_norms[threadIdx_x] += value1 * value1;
        }
        if (element_x + 2 < cols) {
          partial_norms[threadIdx_x] += value2 * value2;
        }
        if (element_x + 3 < cols) {
          partial_norms[threadIdx_x] += value3 * value3;
        }
      }
    }
  }
  __syncthreads();

  // Does reduction in the shared memory.
#if BLOCK_SIZE == 512
  if (threadIdx_x < 256) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 256];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 256
  if (threadIdx_x < 128) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 128];
  }
  __syncthreads();
#endif

#if BLOCK_SIZE >= 128
  if (threadIdx_x < 64) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 64];
  }
  __syncthreads();
#endif

  if (threadIdx_x < 32) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 32];
  }
  __syncthreads();
  if (threadIdx_x < 16) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 16];
  }
  __syncthreads();
  if (threadIdx_x < 8) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 8];
  }
  __syncthreads();
  if (threadIdx_x < 4) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 4];
  }
  __syncthreads();
  if (threadIdx_x < 2) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 2];
  }
  __syncthreads();
  if (threadIdx_x < 1) {
    partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 1];
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    offset = gridDim.x * blockIdx.y + blockIdx.x;
    g_norms[offset] = partial_norms[0];
    __threadfence();

    unsigned int value = atomicInc(&count, blocks);
    isLastBlockDone = (value == (blocks - 1));
  }
  __syncthreads();

  // Does the final reduction in a thread block.
  if (isLastBlockDone) {
    partial_norms[threadIdx_x] = 0;

    Tsum reduced_norm;
    for (element_x = threadIdx_x; element_x < blocks; element_x += BLOCK_SIZE) {
      reduced_norm = g_norms[element_x];
      partial_norms[threadIdx_x] += reduced_norm;
    }
    __syncthreads();

#if BLOCK_SIZE == 512
    if (threadIdx_x < 256) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 256];
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 256
    if (threadIdx_x < 128) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 128];
    }
    __syncthreads();
#endif

#if BLOCK_SIZE >= 128
    if (threadIdx_x < 64) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 64];
    }
    __syncthreads();
#endif

    if (threadIdx_x < 32) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 32];
    }
    __syncthreads();
    if (threadIdx_x < 16) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 16];
    }
    __syncthreads();
    if (threadIdx_x < 8) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 8];
    }
    __syncthreads();
    if (threadIdx_x < 4) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 4];
    }
    __syncthreads();
    if (threadIdx_x < 2) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 2];
    }
    __syncthreads();
    if (threadIdx_x < 1) {
      partial_norms[threadIdx_x] += partial_norms[threadIdx_x + 1];
    }
    __syncthreads();

    if (threadIdx_x == 0) {
      g_norms[0] = partial_norms[0];
      count = 0;
    }
  }
}

RetCode norm(const uchar* src, int rows, int cols, int channels, int src_stride,
             NormTypes norm_type, const uchar* mask, int mask_stride,
             double* norm_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(norm_value != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(norm_type == NORM_INF || norm_type == NORM_L1 ||
             norm_type == NORM_L2);
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  cols *= channels;
  int grid_y, columns = divideUp(cols, 4, 2);
  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(columns, BLOCK_SIZE, BLOCK_SHIFT);
  // run about MAX_BLOCKS thread blocks on a GPU.
  grid_y  = MAX_BLOCKS / grid.x;
  grid.y  = (grid_y < rows) ? grid_y : rows;

  int blocks = grid.x * grid.y;
  long* g_norms;
  cudaError_t code = cudaSuccess;
  code = cudaMalloc(&g_norms, blocks * sizeof(long));
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }

  bool using_mask;
  if (mask != nullptr) {
    using_mask = true;
  }
  else {
    using_mask = false;
  }

  if (norm_type == NORM_INF) {
    normLinfKernel<uchar, long><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, mask, mask_stride, using_mask, blocks, g_norms);
  }
  else if (norm_type == NORM_L1) {
    normL1Kernel<uchar, uint, long><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, mask, mask_stride, using_mask, blocks, g_norms);
  }
  else {  // norm_type == NORM_L2
    normL2Kernel<uchar, long, long><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, mask, mask_stride, using_mask, blocks, g_norms);
  }

  long value;
  code = cudaMemcpy(&value, g_norms, sizeof(long), cudaMemcpyDeviceToHost);
  if (code != cudaSuccess) {
    cudaFree(g_norms);
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  cudaFree(g_norms);

  if (norm_type == NORM_L2) {
    *norm_value = sqrt(value);
  }
  else {
    *norm_value = value;
  }

  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode norm(const float* src, int rows, int cols, int channels, int src_stride,
             NormTypes norm_type, const uchar* mask, int mask_stride,
             double* norm_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(norm_value != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(norm_type == NORM_INF || norm_type == NORM_L1 ||
             norm_type == NORM_L2);
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * (int)sizeof(uchar));
  }

  cols *= channels;
  int grid_y, columns = divideUp(cols, 4, 2);
  dim3 block, grid;
  block.x = BLOCK_SIZE;
  block.y = 1;
  grid.x  = divideUp(columns, BLOCK_SIZE, BLOCK_SHIFT);
  // run about MAX_BLOCKS thread blocks on a GPU.
  grid_y  = MAX_BLOCKS / grid.x;
  grid.y  = (grid_y < rows) ? grid_y : rows;

  int blocks = grid.x * grid.y;
  double* g_norms;
  cudaError_t code = cudaSuccess;
  code = cudaMalloc(&g_norms, blocks * sizeof(double));
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  bool using_mask;
  if (mask != nullptr) {
    using_mask = true;
  }
  else {
    using_mask = false;
  }

  if (norm_type == NORM_INF) {
    normLinfKernel<float, double><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, mask, mask_stride, using_mask, blocks, g_norms);
  }
  else if (norm_type == NORM_L1) {
    normL1Kernel<float, float, double><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, using_mask, blocks,
        g_norms);
  }
  else {  // norm_type == NORM_L2
    normL2Kernel<float, float, double><<<grid, block, 0, stream>>>(src, rows,
        cols, channels, src_stride, mask, mask_stride, using_mask, blocks,
        g_norms);
  }

  double value;
  code = cudaMemcpy(&value, g_norms, sizeof(double), cudaMemcpyDeviceToHost);
  if (code != cudaSuccess) {
    cudaFree(g_norms);
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }
  cudaFree(g_norms);

  if (norm_type == NORM_L2) {
    *norm_value = sqrt(value);
  }
  else {
    *norm_value = value;
  }

  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Norm<uchar, 1>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       double* normValue,
                       NormTypes normTpye,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = norm(inData, inHeight, inWidth, 1, inWidthStride, normTpye,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<uchar, 3>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       double* normValue,
                       NormTypes normTpye,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = norm(inData, inHeight, inWidth, 3, inWidthStride, normTpye,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<uchar, 4>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       double* normValue,
                       NormTypes normTpye,
                       int maskWidthStride,
                       const uchar* mask) {
  RetCode code = norm(inData, inHeight, inWidth, 4, inWidthStride, normTpye,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<float, 1>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       double* normValue,
                       NormTypes normTpye,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = norm(inData, inHeight, inWidth, 1, inWidthStride, normTpye,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<float, 3>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       double* normValue,
                       NormTypes normTpye,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = norm(inData, inHeight, inWidth, 3, inWidthStride, normTpye,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

template <>
RetCode Norm<float, 4>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       double* normValue,
                       NormTypes normTpye,
                       int maskWidthStride,
                       const uchar* mask) {
  inWidthStride *= sizeof(float);
  RetCode code = norm(inData, inHeight, inWidth, 4, inWidthStride, normTpye,
                      mask, maskWidthStride, normValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
