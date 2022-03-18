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

#include "ppl/cv/cuda/abs.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

__DEVICE__
schar abs_device(const schar& src) {
  if (src == -128) {
    return 127;
  }
  else {
    return abs((int)src);
  }
}

__DEVICE__
float abs_device(const float& src) {
  if (src >= 0) {
    return src;
  }
  else {
    return (0 - src);
  }
}

__global__
void absKernel0(const schar* src, int rows, int cols, int src_stride,
                schar* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const char4* input = (char4*)(src + element_y * src_stride);
  char4 input_value = input[element_x];

  char4 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);
  output_value.z = abs_device(input_value.z);
  output_value.w = abs_device(input_value.w);

  char4* output = (char4*)(dst + element_y * dst_stride);
  output[element_x] = output_value;
}

__global__
void absKernel10(const schar* src, int cols, schar* dst) {
  int element_x = (blockIdx.x << 8) + threadIdx.x;
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  const char4* input = (char4*)src;
  char4 input_value, output_value;
  input_value = input[element_x];

  if (index_x < cols - 3) {
    output_value.x = abs_device(input_value.x);
    output_value.y = abs_device(input_value.y);
    output_value.z = abs_device(input_value.z);
    output_value.w = abs_device(input_value.w);

    char4* output = (char4*)dst;
    output[element_x] = output_value;
  }
  else {
    output_value.x = abs_device(input_value.x);
    if (index_x < cols - 1) {
      output_value.y = abs_device(input_value.y);
    }
    if ((index_x < cols - 2)) {
      output_value.z = abs_device(input_value.z);
    }

    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      dst[index_x + 2] = output_value.z;
    }
  }
}

__global__
void absKernel11(const schar* src, int rows, int cols, int src_stride,
                 schar* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  const schar* input = src + element_y * src_stride;
  schar* output = dst + element_y * dst_stride;

  schar input_value0, input_value1, input_value2, input_value3;
  schar output_value0, output_value1, output_value2, output_value3;

  if (blockIdx.x < gridDim.x - 1) {
    input_value0 = input[index_x];
    input_value1 = input[index_x + 1];
    input_value2 = input[index_x + 2];
    input_value3 = input[index_x + 3];

    output_value0 = abs_device(input_value0);
    output_value1 = abs_device(input_value1);
    output_value2 = abs_device(input_value2);
    output_value3 = abs_device(input_value3);

    output[index_x]     = output_value0;
    output[index_x + 1] = output_value1;
    output[index_x + 2] = output_value2;
    output[index_x + 3] = output_value3;
  }
  else {
    input_value0 = input[index_x];
    if (index_x < cols - 1) {
      input_value1 = input[index_x + 1];
    }
    if ((index_x < cols - 2)) {
      input_value2 = input[index_x + 2];
    }

    output_value0 = abs_device(input_value0);
    if (index_x < cols - 1) {
      output_value1 = abs_device(input_value1);
    }
    if ((index_x < cols - 2)) {
      output_value2 = abs_device(input_value2);
    }

    output[index_x] = output_value0;
    if (index_x < cols - 1) {
      output[index_x + 1] = output_value1;
    }
    if ((index_x < cols - 2)) {
      output[index_x + 2] = output_value2;
    }
  }
}

__global__
void absKernel0(const float* src, int rows, int cols, int src_stride,
                float* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float2* input = (float2*)((uchar*)src + element_y * src_stride);
  float2 input_value = input[element_x];

  float2 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);

  float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = output_value;
}

__global__
void absKernel1(const float* src, int rows, int cols, int src_stride,
                float* dst, int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float* input = (float*)((uchar*)src + element_y * src_stride);
  float* output = (float*)((uchar*)dst + element_y * dst_stride);

  float input_value0, input_value1;
  float output_value0, output_value1;

  if (blockIdx.x < gridDim.x - 1) {
    input_value0 = input[element_x];
    input_value1 = input[element_x + 1];

    output_value0 = abs_device(input_value0);
    output_value1 = abs_device(input_value1);

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    input_value0 = input[element_x];
    if (element_x != cols - 1) {
      input_value1 = input[element_x + 1];
    }

    output_value0 = abs_device(input_value0);
    if (element_x != cols - 1) {
      output_value1 = abs_device(input_value1);
    }

    output[element_x] = output_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = output_value1;
    }
  }
}

RetCode abs(const schar* src, int rows, int cols, int channels, int src_stride,
            schar* dst, int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(schar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(schar));

  int columns = cols * channels;
  cols = divideUp(columns, 4, 2);
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src_stride & 3) == 0 && (dst_stride & 3) == 0) {
    absKernel0<<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,
                                           dst_stride);
  }
  else if (src_stride == columns && dst_stride == columns) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    block.x = 256;
    block.y = 1;
    grid.x = divideUp(cols, 256, 8);
    grid.y = 1;
    absKernel10<<<grid, block, 0, stream>>>(src, columns, dst);
  }
  else {
    absKernel11<<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,
                                            dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode abs(const float* src, int rows, int cols, int channels, int src_stride,
            float* dst, int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src_stride & 7) == 0 && (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    absKernel0<<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,
                                           dst_stride);
  }
  else {
    absKernel1<<<grid, block, 0, stream>>>(src, rows, columns, src_stride, dst,
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
RetCode Abs<schar, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride,
                      const schar* inData,
                      int outWidthStride,
                      schar* outData) {
  RetCode code = abs(inData, height, width, 1, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<schar, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride,
                      const schar* inData,
                      int outWidthStride,
                      schar* outData) {
  RetCode code = abs(inData, height, width, 3, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<schar, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride,
                      const schar* inData,
                      int outWidthStride,
                      schar* outData) {
  RetCode code = abs(inData, height, width, 4, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<float, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride,
                      const float* inData,
                      int outWidthStride,
                      float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = abs(inData, height, width, 1, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<float, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride,
                      const float* inData,
                      int outWidthStride,
                      float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = abs(inData, height, width, 3, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

template <>
RetCode Abs<float, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride,
                      const float* inData,
                      int outWidthStride,
                      float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = abs(inData, height, width, 4, inWidthStride, outData,
                     outWidthStride, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
