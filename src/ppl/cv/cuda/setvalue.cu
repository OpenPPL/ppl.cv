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

__global__
void ones_kernel1(uchar* dst, int rows, int cols, int channels, int dst_stride)
{
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  uchar* output = dst + offset;

  if (channels == 1) {
    output[element_x]     = 1;
    output[element_x + 1] = 1;
    output[element_x + 2] = 1;
    output[element_x + 3] = 1;
  }
  else if (channels == 3) {
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
  else {  // channels == 4
    output[element_x]     = 1;
    output[element_x + 1] = 0;
    output[element_x + 2] = 0;
    output[element_x + 3] = 0;
  }
}

__global__
void ones_kernel2(uchar* dst, int rows, int cols, int channels, int dst_stride)
{
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  uchar* output = dst + offset;

  if (channels == 1) {
    if (blockIdx.x < gridDim.x - 1) {
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
      if (element_x < cols - 3) {
        output[element_x + 3] = 1;
      }
    }
  }
  else if (channels == 3) {
    if (blockIdx.x < gridDim.x - 1) {
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
        if (element_x < cols - 3) {
          output[element_x + 3] = 1;
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
        if (element_x < cols - 3) {
          output[element_x + 3] = 0;
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
        if (element_x < cols - 3) {
          output[element_x + 3] = 0;
        }
      }
    }
  }
  else {  // channels == 4
    if (blockIdx.x < gridDim.x - 1) {
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
      if (element_x < cols - 3) {
        output[element_x + 3] = 0;
      }
    }
  }
}

__global__
void ones_kernel1(float* dst, int rows, int cols, int channels, int dst_stride)
{
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  float* output  = (float*)((uchar*)dst + offset);

  if (channels == 1) {
    output[element_x]     = 1.f;
    output[element_x + 1] = 1.f;
    output[element_x + 2] = 1.f;
    output[element_x + 3] = 1.f;
  }
  else if (channels == 3) {
    int remainder = (element_x >> 2) % 3;
    if (remainder == 0) {
      output[element_x]     = 1.f;
      output[element_x + 1] = 0.f;
      output[element_x + 2] = 0.f;
      output[element_x + 3] = 1.f;
    }
    else if (remainder == 1) {
      output[element_x]     = 0.f;
      output[element_x + 1] = 0.f;
      output[element_x + 2] = 1.f;
      output[element_x + 3] = 0.f;
    }
    else {  // remainder == 2
      output[element_x]     = 0.f;
      output[element_x + 1] = 1.f;
      output[element_x + 2] = 0.f;
      output[element_x + 3] = 0.f;
    }
  }
  else {  // channels == 4
    output[element_x]     = 1.f;
    output[element_x + 1] = 0.f;
    output[element_x + 2] = 0.f;
    output[element_x + 3] = 0.f;
  }
}

__global__
void ones_kernel2(float* dst, int rows, int cols, int channels, int dst_stride)
{
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * dst_stride;
  float* output  = (float*)((uchar*)dst + offset);

  if (channels == 1) {
    if (blockIdx.x < gridDim.x - 1) {
      output[element_x]     = 1.f;
      output[element_x + 1] = 1.f;
      output[element_x + 2] = 1.f;
      output[element_x + 3] = 1.f;
    }
    else {
      output[element_x] = 1.f;
      if (element_x < cols - 1) {
        output[element_x + 1] = 1.f;
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = 1.f;
      }
      if (element_x < cols - 3) {
        output[element_x + 3] = 1.f;
      }
    }
  }
  else if (channels == 3) {
    if (blockIdx.x < gridDim.x - 1) {
      int remainder = (element_x >> 2) % 3;
      if (remainder == 0) {
        output[element_x]     = 1.f;
        output[element_x + 1] = 0.f;
        output[element_x + 2] = 0.f;
        output[element_x + 3] = 1.f;
      }
      else if (remainder == 1) {
        output[element_x]     = 0.f;
        output[element_x + 1] = 0.f;
        output[element_x + 2] = 1.f;
        output[element_x + 3] = 0.f;
      }
      else {  // remainder == 2
        output[element_x]     = 0.f;
        output[element_x + 1] = 1.f;
        output[element_x + 2] = 0.f;
        output[element_x + 3] = 0.f;
      }
    }
    else {
      int remainder = (element_x >> 2) % 3;
      if (remainder == 0) {
        output[element_x] = 1.f;
        if (element_x < cols - 1) {
          output[element_x + 1] = 0.f;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = 0.f;
        }
        if (element_x < cols - 3) {
          output[element_x + 3] = 1.f;
        }
      }
      else if (remainder == 1) {
        output[element_x] = 0.f;
        if (element_x < cols - 1) {
          output[element_x + 1] = 0.f;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = 1.f;
        }
        if (element_x < cols - 3) {
          output[element_x + 3] = 0.f;
        }
      }
      else {  // remainder == 2
        output[element_x] = 0.f;
        if (element_x < cols - 1) {
          output[element_x + 1] = 1.f;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = 0.f;
        }
        if (element_x < cols - 3) {
          output[element_x + 3] = 0.f;
        }
      }
    }
  }
  else {  // channels == 4
    if (blockIdx.x < gridDim.x - 1) {
      output[element_x]     = 1.f;
      output[element_x + 1] = 0.f;
      output[element_x + 2] = 0.f;
      output[element_x + 3] = 0.f;
    }
    else {
      output[element_x] = 1.f;
      if (element_x < cols - 1) {
        output[element_x + 1] = 0.f;
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = 0.f;
      }
      if (element_x < cols - 3) {
        output[element_x + 3] = 0.f;
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

  int padded_stride = roundUp(cols, 4, 2) * channels * sizeof(uchar);
  if (dst_stride >= padded_stride) {
    ones_kernel1<<<grid, block, 0, stream>>>(dst, rows, columns, channels,
                                             dst_stride);
  }
  else {
    ones_kernel2<<<grid, block, 0, stream>>>(dst, rows, columns, channels,
                                             dst_stride);
  }

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
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  int padded_stride = roundUp(cols, 4, 2) * channels * sizeof(float);
  if (dst_stride >= padded_stride) {
    ones_kernel1<<<grid, block, 0, stream>>>(dst, rows, columns, channels,
                                             dst_stride);
  }
  else {
    ones_kernel2<<<grid, block, 0, stream>>>(dst, rows, columns, channels,
                                             dst_stride);
  }

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

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
