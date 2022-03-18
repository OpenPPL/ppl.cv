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

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/******************************* SetTo() *******************************/

template <typename Tn, typename T>
__DEVICE__
void setVector(Tn &result, T value);

template <>
__DEVICE__
void setVector(uchar3 &result, uchar value) {
  result.x = value;
  result.y = value;
  result.z = value;
}

template <>
__DEVICE__
void setVector(uchar4 &result, uchar value) {
  result.x = value;
  result.y = value;
  result.z = value;
  result.w = value;
}

template <>
__DEVICE__
void setVector(float3 &result, float value) {
  result.x = value;
  result.y = value;
  result.z = value;
}

template <>
__DEVICE__
void setVector(float4 &result, float value) {
  result.x = value;
  result.y = value;
  result.z = value;
  result.w = value;
}

__global__
void setToKernel0(const uchar* mask, int rows, int cols, int mask_stride,
                  const uchar value, uchar* dst, int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar* mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
  uchar* output = (uchar*)((uchar*)dst + element_y * dst_stride);
  uchar mask_value0, mask_value1, mask_value2, mask_value3;
  uchar value0 = 0, value1 = 0, value2 = 0, value3 = 0;

  if (element_x < cols - 3) {
    mask_value0 = mask_row[element_x];
    mask_value1 = mask_row[element_x + 1];
    mask_value2 = mask_row[element_x + 2];
    mask_value3 = mask_row[element_x + 3];
    if (mask_value0 > 0) {
      value0 = value;
    }
    if (mask_value1 > 0) {
      value1 = value;
    }
    if (mask_value2 > 0) {
      value2 = value;
    }
    if (mask_value3 > 0) {
      value3 = value;
    }

    output[element_x]     = value0;
    output[element_x + 1] = value1;
    output[element_x + 2] = value2;
    output[element_x + 3] = value3;
  }
  else {
    mask_value0 = mask_row[element_x];
    if (element_x < cols - 1) {
      mask_value1 = mask_row[element_x + 1];
    }
    if (element_x < cols - 2) {
      mask_value2 = mask_row[element_x + 2];
    }

    if (mask_value0 > 0) {
      value0 = value;
    }
    if (element_x < cols - 1) {
      if (mask_value1 > 0) {
        value1 = value;
      }
    }
    if (element_x < cols - 2) {
      if (mask_value2 > 0) {
        value2 = value;
      }
    }

    output[element_x] = value0;
    if (element_x < cols - 1) {
      output[element_x + 1] = value1;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = value2;
    }
  }
}

__global__
void setToKernel0(const float value, float* dst, int rows, int cols,
                  int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  float value0 = 0.f, value1 = 0.f;

  if (element_x < cols - 1) {
    value0 = value;
    value1 = value;

    output[element_x]     = value0;
    output[element_x + 1] = value1;
  }
  else {
    value0 = value;

    output[element_x] = value0;
  }
}

__global__
void setToKernel0(const uchar* mask, int rows, int cols, int mask_stride,
                  const float value, float* dst, int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar* mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  uchar mask_value0, mask_value1;
  float value0 = 0.f, value1 = 0.f;

  if (element_x < cols - 1) {
    mask_value0 = mask_row[element_x];
    mask_value1 = mask_row[element_x + 1];
    if (mask_value0 > 0) {
      value0 = value;
    }
    if (mask_value1 > 0) {
      value1 = value;
    }

    output[element_x]     = value0;
    output[element_x + 1] = value1;
  }
  else {
    mask_value0 = mask_row[element_x];
    if (mask_value0 > 0) {
      value0 = value;
    }

    output[element_x] = value0;
  }
}

template <typename T, typename Tn>
__global__
void setToKernel1(const uchar* mask, int rows, int cols, int mask_stride,
                  const T value, T* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  Tn result;
  setVector(result, (T)0);

  uchar* mask_row = (uchar*)((uchar*)mask + element_y * mask_stride);
  uchar mask_value;
  mask_value = mask_row[element_x];
  if (mask_value > 0) {
    setVector(result, value);
  }

  Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = result;
}

RetCode setTo(uchar* dst, int rows, int cols, int dst_channels, int dst_stride,
              const uchar value, const uchar* mask, int mask_channels,
              int mask_stride, cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(dst_channels == 1 || dst_channels == 3 || dst_channels == 4);
  PPL_ASSERT(mask_channels == 1 || mask_channels == 3 || mask_channels == 4);
  PPL_ASSERT(dst_stride >= cols * dst_channels * (int)sizeof(uchar));
  if (mask != nullptr) {
    PPL_ASSERT(mask_stride >= cols * mask_channels * (int)sizeof(uchar));
  }

  cudaError_t code;
  if (mask == nullptr) {
    size_t width_bytes = cols * dst_channels * sizeof(uchar);
    if (dst_stride == cols * dst_channels * (int)sizeof(uchar)) {
      code = cudaMemset(dst, value, rows * width_bytes);
    }
    else {
      code = cudaMemset2D(dst, dst_stride, value, width_bytes, rows);
    }
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }

    return RC_SUCCESS;
  }

  int columns = cols * dst_channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if (dst_channels == mask_channels) {
    setToKernel0<<<grid, block, 0, stream>>>(mask, rows, columns, mask_stride,
                                             value, dst, dst_stride);
  }
  else {
    if (mask_channels == 1) {
      grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);
      if (dst_channels == 3) {
        setToKernel1<uchar, uchar3><<<grid, block, 0, stream>>>(mask, rows,
            cols, mask_stride, value, dst, dst_stride);
      }
      else {
        setToKernel1<uchar, uchar4><<<grid, block, 0, stream>>>(mask, rows,
            cols, mask_stride, value, dst, dst_stride);
      }
    }
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode setTo(float* dst, int rows, int cols, int dst_channels, int dst_stride,
              const float value, const uchar* mask, int mask_channels,
              int mask_stride, cudaStream_t stream) {
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
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if (mask == nullptr) {
    setToKernel0<<<grid, block, 0, stream>>>(value, dst, rows, columns,
                                             dst_stride);
  }
  else {
    if (dst_channels == mask_channels) {
      setToKernel0<<<grid, block, 0, stream>>>(mask, rows, columns, mask_stride,
                                               value, dst, dst_stride);
    }
    else {
      if (mask_channels == 1) {
        grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);
        if (dst_channels == 3) {
          setToKernel1<float, float3><<<grid, block, 0, stream>>>(mask, rows,
              cols, mask_stride, value, dst, dst_stride);
        }
        else {
          setToKernel1<float, float4><<<grid, block, 0, stream>>>(mask, rows,
              cols, mask_stride, value, dst, dst_stride);
        }
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
  RetCode code = setTo(outData, outHeight, outWidth, 1, outWidthStride, value,
                       mask, 1, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 3, outWidthStride, value,
                       mask, 3, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 4, outWidthStride, value,
                       mask, 4, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 3, outWidthStride, value,
                       mask, 1, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 4, outWidthStride, value,
                       mask, 1, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 1, outWidthStride, value,
                       mask, 1, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 3, outWidthStride, value,
                       mask, 3, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 4, outWidthStride, value,
                       mask, 4, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 3, outWidthStride, value,
                       mask, 1, maskWidthStride, stream);

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
  RetCode code = setTo(outData, outHeight, outWidth, 4, outWidthStride, value,
                       mask, 1, maskWidthStride, stream);

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
  code = cudaMemset2D(dst, dst_stride, 0, cols * channels * sizeof(uchar),
                      rows);

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
  code = cudaMemset2D(dst, dst_stride, 0, cols * channels * sizeof(float),
                      rows);

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
