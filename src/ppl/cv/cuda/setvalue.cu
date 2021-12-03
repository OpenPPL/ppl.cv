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

#include "ppl/cv/cuda/setvalue.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/******************************* SetTo() *******************************/

__global__
void setToKernel1(uchar* dst, int rows, int cols, int dst_stride,
                  const uchar value, const uchar* mask, int mask_stride,
                  bool using_mask) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  uchar* output = (uchar*)((uchar*)dst + offset);
  uchar value0 = 0, value1 = 0, value2 = 0, value3 = 0;

  if (using_mask) {
    uchar* mask_start;
    uchar mvalue0, mvalue1, mvalue2, mvalue3;
    mask_start = (uchar*)((uchar*)mask + offset);
    mvalue0 = mask_start[element_x];
    mvalue1 = mask_start[element_x + 1];
    mvalue2 = mask_start[element_x + 2];
    mvalue3 = mask_start[element_x + 3];
    if (mvalue0 > 0) {
      value0 = value;
    }
    if (mvalue1 > 0) {
      value1 = value;
    }
    if (mvalue2 > 0) {
      value2 = value;
    }
    if (mvalue3 > 0) {
      value3 = value;
    }
  }
  else {
    value0 = value;
    value1 = value;
    value2 = value;
    value3 = value;
  }

  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
  output[element_x + 3] = value3;
}

__global__
void setToKernel2(uchar* dst, int rows, int cols, int dst_stride,
                  const uchar value, const uchar* mask, int mask_stride,
                  bool using_mask) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  uchar* output = (uchar*)((uchar*)dst + offset);
  uchar value0 = 0, value1 = 0, value2 = 0, value3 = 0;

  if (blockIdx.x < gridDim.x - 1) {
    if (using_mask) {
      uchar* mask_start;
      uchar mvalue0, mvalue1, mvalue2, mvalue3;
      mask_start = (uchar*)((uchar*)mask + offset);
      mvalue0 = mask_start[element_x];
      mvalue1 = mask_start[element_x + 1];
      mvalue2 = mask_start[element_x + 2];
      mvalue3 = mask_start[element_x + 3];
      if (mvalue0 > 0) {
        value0 = value;
      }
      if (mvalue1 > 0) {
        value1 = value;
      }
      if (mvalue2 > 0) {
        value2 = value;
      }
      if (mvalue3 > 0) {
        value3 = value;
      }
    }
    else {
      value0 = value;
      value1 = value;
      value2 = value;
      value3 = value;
    }

    output[element_x]     = value0;
    output[element_x + 1] = value1;
    output[element_x + 2] = value2;
    output[element_x + 3] = value3;
  }
  else {
    if (using_mask) {
      uchar* mask_start;
      uchar mvalue0, mvalue1, mvalue2, mvalue3;
      mask_start = (uchar*)((uchar*)mask + offset);
      mvalue0 = mask_start[element_x];
      if (element_x < cols - 1) {
        mvalue1 = mask_start[element_x + 1];
      }
      if (element_x < cols - 2) {
        mvalue2 = mask_start[element_x + 2];
      }
      if (element_x < cols - 3) {
        mvalue3 = mask_start[element_x + 3];
      }
      if (mvalue0 > 0) {
        value0 = value;
      }
      if (element_x < cols - 1) {
        if (mvalue1 > 0) {
          value1 = value;
        }
      }
      if (element_x < cols - 2) {
        if (mvalue2 > 0) {
          value2 = value;
        }
      }
      if (element_x < cols - 3) {
        if (mvalue3 > 0) {
          value3 = value;
        }
      }
    }
    else {
      value0 = value;
      if (element_x < cols - 1) {
        value1 = value;
      }
      if (element_x < cols - 2) {
        value2 = value;
      }
      if (element_x < cols - 3) {
        value3 = value;
      }
    }

    output[element_x] = value0;
    if (element_x < cols - 1) {
      output[element_x + 1] = value1;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = value2;
    }
    if (element_x < cols - 3) {
      output[element_x + 3] = value3;
    }
  }
}

__global__
void setToKernel1(uchar* dst, int rows, int cols, int dst_channels,
                  int dst_stride, const uchar value, const uchar* mask,
                  int mask_stride, bool using_mask) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  uchar* output = (uchar*)((uchar*)dst + offset);
  uchar value0 = 0, value1 = 0, value2 = 0, value3 = 0;

  if (using_mask) {
    uchar* mask_start;
    uchar mvalue0, mvalue1, mvalue2, mvalue3;
    mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
    mvalue0 = mask_start[element_x / dst_channels];
    mvalue1 = mask_start[(element_x + 1) / dst_channels];
    mvalue2 = mask_start[(element_x + 2) / dst_channels];
    mvalue3 = mask_start[(element_x + 3) / dst_channels];
    if (mvalue0 > 0) {
      value0 = value;
    }
    if (mvalue1 > 0) {
      value1 = value;
    }
    if (mvalue2 > 0) {
      value2 = value;
    }
    if (mvalue3 > 0) {
      value3 = value;
    }
  }
  else {
    value0 = value;
    value1 = value;
    value2 = value;
    value3 = value;
  }

  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
  output[element_x + 3] = value3;
}

__global__
void setToKernel2(uchar* dst, int rows, int cols, int dst_channels,
                  int dst_stride, const uchar value, const uchar* mask,
                  int mask_stride, bool using_mask) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  uchar* output = (uchar*)((uchar*)dst + offset);
  uchar value0 = 0, value1 = 0, value2 = 0, value3 = 0;

  if (blockIdx.x < gridDim.x - 1) {
    if (using_mask) {
      uchar* mask_start;
      uchar mvalue0, mvalue1, mvalue2, mvalue3;
      mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
      mvalue0 = mask_start[element_x / dst_channels];
      mvalue1 = mask_start[(element_x + 1) / dst_channels];
      mvalue2 = mask_start[(element_x + 2) / dst_channels];
      mvalue3 = mask_start[(element_x + 3) / dst_channels];
      if (mvalue0 > 0) {
        value0 = value;
      }
      if (mvalue1 > 0) {
        value1 = value;
      }
      if (mvalue2 > 0) {
        value2 = value;
      }
      if (mvalue3 > 0) {
        value3 = value;
      }
    }
    else {
      value0 = value;
      value1 = value;
      value2 = value;
      value3 = value;
    }

    output[element_x]     = value0;
    output[element_x + 1] = value1;
    output[element_x + 2] = value2;
    output[element_x + 3] = value3;
  }
  else {
    if (using_mask) {
      uchar* mask_start;
      uchar mvalue0, mvalue1, mvalue2, mvalue3;
      mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
      mvalue0 = mask_start[element_x / dst_channels];
      if (element_x < cols - 1) {
        mvalue1 = mask_start[(element_x + 1) / dst_channels];
      }
      if (element_x < cols - 2) {
        mvalue2 = mask_start[(element_x + 2) / dst_channels];
      }
      if (element_x < cols - 3) {
        mvalue3 = mask_start[(element_x + 3) / dst_channels];
      }
      if (mvalue0 > 0) {
        value0 = value;
      }
      if (element_x < cols - 1) {
        if (mvalue1 > 0) {
          value1 = value;
        }
      }
      if (element_x < cols - 2) {
        if (mvalue2 > 0) {
          value2 = value;
        }
      }
      if (element_x < cols - 3) {
        if (mvalue3 > 0) {
          value3 = value;
        }
      }
    }
    else {
      value0 = value;
      if (element_x < cols - 1) {
        value1 = value;
      }
      if (element_x < cols - 2) {
        value2 = value;
      }
      if (element_x < cols - 3) {
        value3 = value;
      }
    }

    output[element_x] = value0;
    if (element_x < cols - 1) {
      output[element_x + 1] = value1;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = value2;
    }
    if (element_x < cols - 3) {
      output[element_x + 3] = value3;
    }
  }
}

__global__
void setToKernel1(float* dst, int rows, int cols, int dst_stride,
                  const float value, const uchar* mask, int mask_stride,
                  bool using_mask) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  float* output = (float*)((uchar*)dst + offset);
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;

  if (using_mask) {
    uchar* mask_start;
    uchar mvalue0, mvalue1, mvalue2, mvalue3;
    mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
    mvalue0 = mask_start[element_x];
    mvalue1 = mask_start[element_x + 1];
    mvalue2 = mask_start[element_x + 2];
    mvalue3 = mask_start[element_x + 3];
    if (mvalue0 > 0) {
      value0 = value;
    }
    if (mvalue1 > 0) {
      value1 = value;
    }
    if (mvalue2 > 0) {
      value2 = value;
    }
    if (mvalue3 > 0) {
      value3 = value;
    }
  }
  else {
    value0 = value;
    value1 = value;
    value2 = value;
    value3 = value;
  }

  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
  output[element_x + 3] = value3;
}

__global__
void setToKernel2(float* dst, int rows, int cols, int dst_stride,
                  const float value, const uchar* mask, int mask_stride,
                  bool using_mask) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  float* output = (float*)((uchar*)dst + offset);
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;

  if (blockIdx.x < gridDim.x - 1) {
    if (using_mask) {
      uchar* mask_start;
      uchar mvalue0, mvalue1, mvalue2, mvalue3;
      mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
      mvalue0 = mask_start[element_x];
      mvalue1 = mask_start[element_x + 1];
      mvalue2 = mask_start[element_x + 2];
      mvalue3 = mask_start[element_x + 3];
      if (mvalue0 > 0) {
        value0 = value;
      }
      if (mvalue1 > 0) {
        value1 = value;
      }
      if (mvalue2 > 0) {
        value2 = value;
      }
      if (mvalue3 > 0) {
        value3 = value;
      }
    }
    else {
      value0 = value;
      value1 = value;
      value2 = value;
      value3 = value;
    }

    output[element_x]     = value0;
    output[element_x + 1] = value1;
    output[element_x + 2] = value2;
    output[element_x + 3] = value3;
  }
  else {
    if (using_mask) {
      uchar* mask_start;
      uchar mvalue0, mvalue1, mvalue2, mvalue3;
      mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
      mvalue0 = mask_start[element_x];
      if (element_x < cols - 1) {
        mvalue1 = mask_start[element_x + 1];
      }
      if (element_x < cols - 2) {
        mvalue2 = mask_start[element_x + 2];
      }
      if (element_x < cols - 3) {
        mvalue3 = mask_start[element_x + 3];
      }
      if (mvalue0 > 0) {
        value0 = value;
      }
      if (element_x < cols - 1) {
        if (mvalue1 > 0) {
          value1 = value;
        }
      }
      if (element_x < cols - 2) {
        if (mvalue2 > 0) {
          value2 = value;
        }
      }
      if (element_x < cols - 3) {
        if (mvalue3 > 0) {
          value3 = value;
        }
      }
    }
    else {
      value0 = value;
      if (element_x < cols - 1) {
        value1 = value;
      }
      if (element_x < cols - 2) {
        value2 = value;
      }
      if (element_x < cols - 3) {
        value3 = value;
      }
    }

    output[element_x] = value0;
    if (element_x < cols - 1) {
      output[element_x + 1] = value1;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = value2;
    }
    if (element_x < cols - 3) {
      output[element_x + 3] = value3;
    }
  }
}

__global__
void setToKernel1(float* dst, int rows, int cols, int dst_channels,
                  int dst_stride, const float value, const uchar* mask,
                  int mask_stride, bool using_mask) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  float* output = (float*)((uchar*)dst + offset);
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;

  if (using_mask) {
    uchar* mask_start;
    uchar mvalue0, mvalue1, mvalue2, mvalue3;
    mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
    mvalue0 = mask_start[element_x / dst_channels];
    mvalue1 = mask_start[(element_x + 1) / dst_channels];
    mvalue2 = mask_start[(element_x + 2) / dst_channels];
    mvalue3 = mask_start[(element_x + 3) / dst_channels];
    if (mvalue0 > 0) {
      value0 = value;
    }
    if (mvalue1 > 0) {
      value1 = value;
    }
    if (mvalue2 > 0) {
      value2 = value;
    }
    if (mvalue3 > 0) {
      value3 = value;
    }
  }
  else {
    value0 = value;
    value1 = value;
    value2 = value;
    value3 = value;
  }

  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
  output[element_x + 3] = value3;
}

__global__
void setToKernel2(float* dst, int rows, int cols, int dst_channels,
                  int dst_stride, const float value, const uchar* mask,
                  int mask_stride, bool using_mask) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  float* output = (float*)((uchar*)dst + offset);
  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;

  if (blockIdx.x < gridDim.x - 1) {
    if (using_mask) {
      uchar* mask_start;
      uchar mvalue0, mvalue1, mvalue2, mvalue3;
      mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
      mvalue0 = mask_start[element_x / dst_channels];
      mvalue1 = mask_start[(element_x + 1) / dst_channels];
      mvalue2 = mask_start[(element_x + 2) / dst_channels];
      mvalue3 = mask_start[(element_x + 3) / dst_channels];
      if (mvalue0 > 0) {
        value0 = value;
      }
      if (mvalue1 > 0) {
        value1 = value;
      }
      if (mvalue2 > 0) {
        value2 = value;
      }
      if (mvalue3 > 0) {
        value3 = value;
      }
    }
    else {
      value0 = value;
      value1 = value;
      value2 = value;
      value3 = value;
    }

    output[element_x]     = value0;
    output[element_x + 1] = value1;
    output[element_x + 2] = value2;
    output[element_x + 3] = value3;
  }
  else {
    if (using_mask) {
      uchar* mask_start;
      uchar mvalue0, mvalue1, mvalue2, mvalue3;
      mask_start = (uchar*)((uchar*)mask + element_y * mask_stride);
      mvalue0 = mask_start[element_x / dst_channels];
      if (element_x < cols - 1) {
        mvalue1 = mask_start[(element_x + 1) / dst_channels];
      }
      if (element_x < cols - 2) {
        mvalue2 = mask_start[(element_x + 2) / dst_channels];
      }
      if (element_x < cols - 3) {
        mvalue3 = mask_start[(element_x + 3) / dst_channels];
      }
      if (mvalue0 > 0) {
        value0 = value;
      }
      if (element_x < cols - 1) {
        if (mvalue1 > 0) {
          value1 = value;
        }
      }
      if (element_x < cols - 2) {
        if (mvalue2 > 0) {
          value2 = value;
        }
      }
      if (element_x < cols - 3) {
        if (mvalue3 > 0) {
          value3 = value;
        }
      }
    }
    else {
      value0 = value;
      if (element_x < cols - 1) {
        value1 = value;
      }
      if (element_x < cols - 2) {
        value2 = value;
      }
      if (element_x < cols - 3) {
        value3 = value;
      }
    }

    output[element_x] = value0;
    if (element_x < cols - 1) {
      output[element_x + 1] = value1;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = value2;
    }
    if (element_x < cols - 3) {
      output[element_x + 3] = value3;
    }
  }
}

RetCode setTo(uchar* dst, int rows, int cols, int dst_channels, int dst_stride,
              const uchar* mask, int mask_stride, int mask_channels,
              const uchar value, cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(dst_channels == 1 || dst_channels == 3 || dst_channels == 4);
  PPL_ASSERT(mask_channels == 1 || mask_channels == 3 || mask_channels == 4);
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * mask_channels * (int)sizeof(uchar));
  }

  int columns = cols * dst_channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  bool using_mask;
  if (mask != nullptr) {
    using_mask = true;
  }
  else {
    using_mask = false;
  }

  int padded_stride = roundUp(cols, 4, 2) * dst_channels * sizeof(uchar);
  if (dst_channels == mask_channels) {
    if (dst_stride >= padded_stride) {
      setToKernel1<<<grid, block, 0, stream>>>(dst, rows, columns, dst_stride,
                                               value, mask, mask_stride,
                                               using_mask);
    }
    else {
      setToKernel2<<<grid, block, 0, stream>>>(dst, rows, columns, dst_stride,
                                               value, mask, mask_stride,
                                               using_mask);
    }
  }
  else {
    if (mask_channels == 1) {
      if (dst_stride >= padded_stride) {
        setToKernel1<<<grid, block, 0, stream>>>(dst, rows, columns,
                                                 dst_channels, dst_stride,
                                                 value, mask, mask_stride,
                                                 using_mask);
      }
      else {
        setToKernel2<<<grid, block, 0, stream>>>(dst, rows, columns,
                                                 dst_channels, dst_stride,
                                                 value, mask, mask_stride,
                                                 using_mask);
      }
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode setTo(float* dst, int rows, int cols, int dst_channels, int dst_stride,
              const uchar* mask, int mask_stride, int mask_channels,
              const float value, cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(dst_channels == 1 || dst_channels == 3 || dst_channels == 4);
  PPL_ASSERT(mask_channels == 1 || mask_channels == 3 || mask_channels == 4);
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(float));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * mask_channels * (int)sizeof(uchar));
  }

  int columns = cols * dst_channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  bool using_mask;
  if (mask != nullptr) {
    using_mask = true;
  }
  else {
    using_mask = false;
  }

  int padded_stride = roundUp(cols, 4, 2) * dst_channels * sizeof(float);
  if (dst_channels == mask_channels) {
    if (dst_stride >= padded_stride) {
      setToKernel1<<<grid, block, 0, stream>>>(dst, rows, columns, dst_stride,
                                               value, mask, mask_stride,
                                               using_mask);
    }
    else {
      setToKernel2<<<grid, block, 0, stream>>>(dst, rows, columns, dst_stride,
                                               value, mask, mask_stride,
                                               using_mask);
    }
  }
  else {
    if (mask_channels == 1) {
      if (dst_stride >= padded_stride) {
        setToKernel1<<<grid, block, 0, stream>>>(dst, rows, columns,
                                                 dst_channels, dst_stride,
                                                 value, mask, mask_stride,
                                                 using_mask);
      }
      else {
        setToKernel2<<<grid, block, 0, stream>>>(dst, rows, columns,
                                                 dst_channels, dst_stride,
                                                 value, mask, mask_stride,
                                                 using_mask);
      }
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
RetCode SetTo<uchar, 1, 1>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           uchar *outData,
                           const uchar value,
                           int maskWidthStride,
                           const uchar *mask) {
  RetCode code = setTo(outData, outHeight, outWidth, 1, outWidthStride, mask,
                       maskWidthStride, 1, value, stream);

  return code;
}

template <>
RetCode SetTo<uchar, 3, 3>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           uchar *outData,
                           const uchar value,
                           int maskWidthStride,
                           const uchar *mask) {
  RetCode code = setTo(outData, outHeight, outWidth, 3, outWidthStride, mask,
                       maskWidthStride, 3, value, stream);

  return code;
}

template <>
RetCode SetTo<uchar, 4, 4>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           uchar *outData,
                           const uchar value,
                           int maskWidthStride,
                           const uchar *mask) {
  RetCode code = setTo(outData, outHeight, outWidth, 4, outWidthStride, mask,
                       maskWidthStride, 4, value, stream);

  return code;
}

template <>
RetCode SetTo<uchar, 3, 1>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           uchar *outData,
                           const uchar value,
                           int maskWidthStride,
                           const uchar *mask) {
  RetCode code = setTo(outData, outHeight, outWidth, 3, outWidthStride, mask,
                       maskWidthStride, 1, value, stream);

  return code;
}

template <>
RetCode SetTo<uchar, 4, 1>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           uchar *outData,
                           const uchar value,
                           int maskWidthStride,
                           const uchar *mask) {
  RetCode code = setTo(outData, outHeight, outWidth, 4, outWidthStride, mask,
                       maskWidthStride, 1, value, stream);

  return code;
}

template <>
RetCode SetTo<float, 1, 1>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           float *outData,
                           const float value,
                           int maskWidthStride,
                           const uchar *mask) {
  outWidthStride *= sizeof(float);
  RetCode code = setTo(outData, outHeight, outWidth, 1, outWidthStride, mask,
                       maskWidthStride, 1, value, stream);

  return code;
}

template <>
RetCode SetTo<float, 3, 3>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           float *outData,
                           const float value,
                           int maskWidthStride,
                           const uchar *mask) {
  outWidthStride *= sizeof(float);
  RetCode code = setTo(outData, outHeight, outWidth, 3, outWidthStride, mask,
                       maskWidthStride, 3, value, stream);

  return code;
}

template <>
RetCode SetTo<float, 4, 4>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           float *outData,
                           const float value,
                           int maskWidthStride,
                           const uchar *mask) {
  outWidthStride *= sizeof(float);
  RetCode code = setTo(outData, outHeight, outWidth, 4, outWidthStride, mask,
                       maskWidthStride, 4, value, stream);

  return code;
}

template <>
RetCode SetTo<float, 3, 1>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           float *outData,
                           const float value,
                           int maskWidthStride,
                           const uchar *mask) {
  outWidthStride *= sizeof(float);
  RetCode code = setTo(outData, outHeight, outWidth, 3, outWidthStride, mask,
                       maskWidthStride, 1, value, stream);

  return code;
}

template <>
RetCode SetTo<float, 4, 1>(cudaStream_t stream,
                           int outHeight,
                           int outWidth,
                           int outWidthStride,
                           float *outData,
                           const float value,
                           int maskWidthStride,
                           const uchar *mask) {
  outWidthStride *= sizeof(float);
  RetCode code = setTo(outData, outHeight, outWidth, 4, outWidthStride, mask,
                       maskWidthStride, 1, value, stream);

  return code;
}

/******************************* Ones() *******************************/

__global__
void onesKernel(uchar* dst, int rows, int cols, int channels, int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  uchar* output = dst + offset;

  if (channels == 1) {
    if (element_x < cols - 3) {
      output[element_x]     = 1;
      output[element_x + 1] = 1;
      output[element_x + 2] = 1;
      output[element_x + 3] = 1;
    }
    else {
      output[element_x] = 1;
      if (element_x < cols - 1) {
        output[element_x + 1] = 1;
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = 1;
      }
    }
  }
  else if (channels == 3) {
    if (element_x < cols - 3) {
      int remainder = (element_x >> 2) % 3;
      if (remainder == 0) {
        output[element_x]     = 1;
        output[element_x + 1] = 0;
        output[element_x + 2] = 0;
        output[element_x + 3] = 1;
      }
      else if (remainder == 1) {
        output[element_x]     = 0;
        output[element_x + 1] = 0;
        output[element_x + 2] = 1;
        output[element_x + 3] = 0;
      }
      else {  // remainder == 2
        output[element_x]     = 0;
        output[element_x + 1] = 1;
        output[element_x + 2] = 0;
        output[element_x + 3] = 0;
      }
    }
    else {
      int remainder = (element_x >> 2) % 3;
      if (remainder == 0) {
        output[element_x] = 1;
        if (element_x < cols - 1) {
          output[element_x + 1] = 0;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = 0;
        }
      }
      else if (remainder == 1) {
        output[element_x] = 0;
        if (element_x < cols - 1) {
          output[element_x + 1] = 0;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = 1;
        }
      }
      else {  // remainder == 2
        output[element_x] = 0;
        if (element_x < cols - 1) {
          output[element_x + 1] = 1;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = 0;
        }
      }
    }
  }
  else {  // channels == 4
    if (element_x < cols - 3) {
      output[element_x]     = 1;
      output[element_x + 1] = 0;
      output[element_x + 2] = 0;
      output[element_x + 3] = 0;
    }
    else {
      output[element_x] = 1;
      if (element_x < cols - 1) {
        output[element_x + 1] = 0;
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = 0;
      }
    }
  }
}

__global__
void onesKernel(float* dst, int rows, int cols, int channels, int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  float* output  = (float*)((uchar*)dst + offset);

  if (channels == 1) {
    if (element_x < cols - 1) {
      output[element_x]     = 1.f;
      output[element_x + 1] = 1.f;
    }
    else {
      output[element_x] = 1.f;
    }
  }
  else if (channels == 3) {
    if (element_x < cols - 1) {
      int remainder = (element_x >> 1) % 3;
      if (remainder == 0) {
        output[element_x]     = 1.f;
        output[element_x + 1] = 0.f;
      }
      else if (remainder == 1) {
        output[element_x]     = 0.f;
        output[element_x + 1] = 1.f;
      }
      else {  // remainder == 2
        output[element_x]     = 0.f;
        output[element_x + 1] = 0.f;
      }
    }
    else {
      int remainder = (element_x >> 1) % 3;
      if (remainder == 0) {
        output[element_x] = 1.f;
      }
      else if (remainder == 1) {
        output[element_x] = 0.f;
      }
      else {  // remainder == 2
        output[element_x] = 0.f;
      }
    }
  }
  else {  // channels == 4
    if (element_x < cols - 1) {
      int remainder = element_x & 3;
      if (remainder == 0) {
        output[element_x]     = 1.f;
        output[element_x + 1] = 0.f;
      }
      else {
        output[element_x]     = 0.f;
        output[element_x + 1] = 0.f;
      }
    }
    else {
      int remainder = element_x & 3;
      if (remainder == 0) {
        output[element_x] = 1.f;
      }
      else {
        output[element_x] = 0.f;
      }
    }
  }
}

RetCode ones(uchar* dst, int rows, int cols, int channels, int dst_stride,
             cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  onesKernel<<<grid, block, 0, stream>>>(dst, rows, columns, channels,
                                         dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode ones(float* dst, int rows, int cols, int channels, int dst_stride,
             cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  onesKernel<<<grid, block, 0, stream>>>(dst, rows, columns, channels,
                                         dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Ones<uchar, 1>(cudaStream_t stream,
                       int height,
                       int width,
                       int outWidthStride,
                       uchar* outData) {
  RetCode code = ones(outData, height, width, 1, outWidthStride, stream);

  return code;
}

template <>
RetCode Ones<uchar, 3>(cudaStream_t stream,
                       int height,
                       int width,
                       int outWidthStride,
                       uchar* outData) {
  RetCode code = ones(outData, height, width, 3, outWidthStride, stream);

  return code;
}

template <>
RetCode Ones<uchar, 4>(cudaStream_t stream,
                       int height,
                       int width,
                       int outWidthStride,
                       uchar* outData) {
  RetCode code = ones(outData, height, width, 4, outWidthStride, stream);

  return code;
}

template <>
RetCode Ones<float, 1>(cudaStream_t stream,
                       int height,
                       int width,
                       int outWidthStride,
                       float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = ones(outData, height, width, 1, outWidthStride, stream);

  return code;
}

template <>
RetCode Ones<float, 3>(cudaStream_t stream,
                       int height,
                       int width,
                       int outWidthStride,
                       float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = ones(outData, height, width, 3, outWidthStride, stream);

  return code;
}

template <>
RetCode Ones<float, 4>(cudaStream_t stream,
                       int height,
                       int width,
                       int outWidthStride,
                       float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = ones(outData, height, width, 4, outWidthStride, stream);

  return code;
}

/******************************* Zeros() *******************************/

RetCode zeros(uchar* dst, int rows, int cols, int channels, int dst_stride,
              cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  cudaError_t code;
  code = cudaMemset2DAsync(dst, dst_stride, 0, cols * channels * sizeof(uchar),
                           rows, stream);

  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode zeros(float* dst, int rows, int cols, int channels, int dst_stride,
              cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  cudaError_t code;
  code = cudaMemset2DAsync(dst, dst_stride, 0, cols * channels * sizeof(float),
                           rows, stream);

  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Zeros<uchar, 1>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        uchar* outData) {
  RetCode code = zeros(outData, height, width, 1, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<uchar, 3>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        uchar* outData) {
  RetCode code = zeros(outData, height, width, 3, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<uchar, 4>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        uchar* outData) {
  RetCode code = zeros(outData, height, width, 4, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<float, 1>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = zeros(outData, height, width, 1, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<float, 3>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = zeros(outData, height, width, 3, outWidthStride, stream);

  return code;
}

template <>
RetCode Zeros<float, 4>(cudaStream_t stream,
                        int height,
                        int width,
                        int outWidthStride,
                        float* outData) {
  outWidthStride *= sizeof(float);
  RetCode code = zeros(outData, height, width, 4, outWidthStride, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
