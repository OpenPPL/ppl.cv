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

#include "ppl/cv/cuda/guidedfilter.h"

#include <cfloat>

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/**************************** function declaration **************************/

RetCode convertTo(const uchar* src, int rows, int cols, int channels,
                  int src_stride, float* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream);

RetCode convertTo(const float* src, int rows, int cols, int channels,
                  int src_stride, uchar* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream);

/********************************* add() ***********************************/

__global__
void addKernel(const float* src0, int rows, int cols, int src0_stride,
               int src0_offset, const float* src1, int src1_stride,
               int src1_offset, float* dst, int dst_stride, int dst_offset) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_offset + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_offset + offset);
  float* output = (float*)((uchar*)dst + dst_offset + element_y * dst_stride);

  if (element_x < cols - 1) {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    output_value0 = input_value00 + input_value10;
    output_value1 = input_value01 + input_value11;

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    float input_value0, input_value1, output_value;
    input_value0 = input0[element_x];
    input_value1 = input1[element_x];

    output_value = input_value0 + input_value1;

    output[element_x] = output_value;
  }
}

RetCode add(const float* src0, int rows, int cols, int channels,
            int src0_offset, const float* src1, int src1_offset, float* dst,
            int dst_stride, int dst_offset, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src0_offset >= 0);
  PPL_ASSERT(src1_offset >= 0);
  PPL_ASSERT(dst_offset >= 0);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  addKernel<<<grid, block, 0, stream>>>(src0, rows, columns, dst_stride,
      src0_offset, src1, dst_stride, src1_offset, dst, dst_stride, dst_offset);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/******************************* addScalar() *******************************/

__global__
void addScalarKernel(float* dst, int rows, int columns, int stride,
                     int dst_offset, float value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= columns) {
    return;
  }

  int offset = element_y * stride;
  float* output = (float*)((uchar*)dst + dst_offset + offset);
  float result = output[element_x];
  result += value;

  output[element_x] = result;
}

RetCode addScalar(float* dst, int rows, int cols, int channels, int stride,
                  int dst_offset, float value, cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_offset >= 0);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(columns, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  addScalarKernel<<<grid, block, 0, stream>>>(dst, rows, columns, stride,
                                              dst_offset, value);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/****************************** multiply() ********************************/

__global__
void multiplyKernel(const float* src0, int rows, int cols, int src0_stride,
                    int src0_offset, const float* src1, int src1_stride,
                    int src1_offset, float* dst, int dst_stride, int dst_offset,
                    float scale) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_offset + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_offset + offset);
  float* output = (float*)((uchar*)dst + dst_offset + element_y * dst_stride);

  if (element_x < cols - 1) {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    if (scale == 1.f) {
      output_value0 = input_value00 * input_value10;
      output_value1 = input_value01 * input_value11;
    }
    else {
      output_value0 = input_value00 * input_value10 * scale;
      output_value1 = input_value01 * input_value11 * scale;
    }

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    float input_value0, input_value1, output_value;
    input_value0 = input0[element_x];
    input_value1 = input1[element_x];

    if (scale == 1.f) {
      output_value = input_value0 * input_value1;
    }
    else {
      output_value = input_value0 * input_value1 * scale;
    }

    output[element_x] = output_value;
  }
}

RetCode multiply(const float* src0, int rows, int cols, int channels,
                 int src0_stride, int src0_offset, const float* src1,
                 int src1_stride, int src1_offset, float* dst, int dst_stride,
                 int dst_offset, float scale, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src0_offset >= 0);
  PPL_ASSERT(src1_offset >= 0);
  PPL_ASSERT(dst_offset >= 0);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  multiplyKernel<<<grid, block, 0, stream>>>(src0, rows, columns, src0_stride,
      src0_offset, src1, src1_stride, src1_offset, dst, dst_stride, dst_offset,
      scale);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/******************************* divide() ********************************/

__global__
void divideKernel(const float* src0, int rows, int cols, int src0_stride,
                  int src0_offset, const float* src1, int src1_stride,
                  int src1_offset, float* dst, int dst_stride, int dst_offset,
                  float scale) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src0_stride;
  const float* input0 = (float*)((uchar*)src0 + src0_offset + offset);
  const float* input1 = (float*)((uchar*)src1 + src1_offset + offset);
  float* output = (float*)((uchar*)dst + dst_offset + element_y * dst_stride);

  if (element_x < cols - 1) {
    float input_value00, input_value01;
    float input_value10, input_value11;
    float output_value0, output_value1;

    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    if (scale == 1.f) {
      output_value0 = input_value10 == 0 ? 0 : input_value00 / input_value10;
      output_value1 = input_value11 == 0 ? 0 : input_value01 / input_value11;
    }
    else {
      output_value0 = input_value10 == 0 ? 0 :
                      scale * input_value00 / input_value10;
      output_value1 = input_value11 == 0 ? 0 :
                      scale * input_value01 / input_value11;
    }

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    float input_value0, input_value1, output_value;
    input_value0 = input0[element_x];
    input_value1 = input1[element_x];

    if (scale == 1.f) {
      output_value = input_value1 == 0 ? 0 : input_value0 / input_value1;
    }
    else {
      output_value = input_value1 == 0 ? 0 :
                     scale * input_value0 / input_value1;
    }

    output[element_x] = output_value;
  }
}

RetCode divide(const float* src0, int rows, int cols, int channels,
               int src0_stride, int src0_offset, const float* src1,
               int src1_stride, int src1_offset, float* dst, int dst_stride,
               int dst_offset, float scale, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src0_offset >= 0);
  PPL_ASSERT(src1_offset >= 0);
  PPL_ASSERT(dst_offset >= 0);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  divideKernel<<<grid, block, 0, stream>>>(src0, rows, columns, src0_stride,
      src0_offset, src1, src1_stride, src1_offset, dst, dst_stride, dst_offset,
      scale);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/******************************* subtract() ********************************/

__global__
void subtractKernel(const float* src0, int rows, int cols, int stride,
                    int src0_offset, const float* src1, int src1_offset,
                    float* dst, int dst_offset) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * stride;
  float* input0 = (float*)((uchar*)src0 + src0_offset + offset);
  float* input1 = (float*)((uchar*)src1 + src1_offset + offset);
  float* output = (float*)((uchar*)dst + dst_offset + offset);

  float value0 = input0[element_x];
  float value1 = input1[element_x];
  float result = value0 - value1;

  output[element_x] = result;
}

RetCode subtract(const float* src0, int rows, int cols, int channels,
                 int stride, int src0_offset, const float* src1,
                 int src1_offset, float* dst, int dst_offset,
                 cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src0_offset >= 0);
  PPL_ASSERT(src1_offset >= 0);
  PPL_ASSERT(dst_offset >= 0);

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(columns, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  subtractKernel<<<grid, block, 0, stream>>>(src0, rows, columns, stride,
      src0_offset, src1, src1_offset, dst, dst_offset);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/******************************* boxFilter() ********************************/

#define RADIUS 8
#define SMALL_KSIZE RADIUS * 2 + 1

template <typename Tsrc, typename Tdst, typename BorderInterpolation>
__global__
void rowColC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                    int src_offset, int radius_x, int radius_y,
                    bool is_x_symmetric, bool is_y_symmetric, bool normalize,
                    float weight, Tdst* dst, int dst_stride, int dst_offset,
                    BorderInterpolation interpolation) {
  __shared__ float data[kDimY0 * 3][(kDimX0 << 2)];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  int bottom = element_x - radius_x;
  int top    = element_x + radius_x;
  if (!is_x_symmetric) {
    top -= 1;
  }

  int data_index, row_index;
  Tsrc* input;
  float4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius_x >> (kShiftX0 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius_x) >> (kShiftX0 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  if (threadIdx.y < radius_y && element_x < cols) {
    row_index = interpolation(rows, radius_y, element_y - radius_y);
    input = (Tsrc*)((uchar*)src + src_offset + row_index * src_stride);
    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        sum += value;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius_x, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius_x, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius_x, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius_x, i + 3);
        value.w = input[data_index];
        sum += value;
      }
    }
    data_index = threadIdx.x << 2;
    data[threadIdx.y][data_index] = sum.x;
    data[threadIdx.y][data_index + 1] = sum.y;
    data[threadIdx.y][data_index + 2] = sum.z;
    data[threadIdx.y][data_index + 3] = sum.w;
  }

  if (element_y < rows && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    input = (Tsrc*)((uchar*)src + src_offset + element_y * src_stride);

    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        sum += value;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius_x, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius_x, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius_x, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius_x, i + 3);
        value.w = input[data_index];
        sum += value;
      }
    }
    data_index = threadIdx.x << 2;
    data[radius_y + threadIdx.y][data_index] = sum.x;
    data[radius_y + threadIdx.y][data_index + 1] = sum.y;
    data[radius_y + threadIdx.y][data_index + 2] = sum.z;
    data[radius_y + threadIdx.y][data_index + 3] = sum.w;
  }

  if (threadIdx.y < radius_y && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    if (blockIdx.y != gridDim.y - 1) {
      row_index = interpolation(rows, radius_y,
                                ((blockIdx.y + 1) << kShiftY0) + threadIdx.y);
    }
    else {
      row_index = interpolation(rows, radius_y, rows + threadIdx.y);
    }
    input = (Tsrc*)((uchar*)src + src_offset + row_index * src_stride);

    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        sum += value;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
        data_index = interpolation(cols, radius_x, i);
        value.x = input[data_index];
        data_index = interpolation(cols, radius_x, i + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius_x, i + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius_x, i + 3);
        value.w = input[data_index];
        sum += value;
      }
    }

    data_index = threadIdx.x << 2;
    if (blockIdx.y != gridDim.y - 1) {
      row_index = radius_y + kDimY0 + threadIdx.y;
    }
    else {
      row_index = radius_y + (rows - (blockIdx.y << kShiftY0)) + threadIdx.y;
    }
    data[row_index][data_index] = sum.x;
    data[row_index][data_index + 1] = sum.y;
    data[row_index][data_index + 2] = sum.z;
    data[row_index][data_index + 3] = sum.w;
  }
  __syncthreads();

  if (element_y < rows && element_x < cols) {
    top = (radius_y << 1) + 1;
    if (!is_y_symmetric) {
      top -= 1;
    }
    sum = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int i = 0; i < top; i++) {
      data_index = threadIdx.x << 2;
      value.x = data[i + threadIdx.y][data_index];
      value.y = data[i + threadIdx.y][data_index + 1];
      value.z = data[i + threadIdx.y][data_index + 2];
      value.w = data[i + threadIdx.y][data_index + 3];
      sum += value;
    }

    if (normalize) {
      sum.x *= weight;
      sum.y *= weight;
      sum.z *= weight;
      sum.w *= weight;
    }

    Tdst* output = (Tdst*)((uchar*)dst + dst_offset + element_y * dst_stride);
    if (sizeof(Tdst) == 1) {
      if (element_x < cols - 3) {
        output[element_x]     = saturateCast(sum.x);
        output[element_x + 1] = saturateCast(sum.y);
        output[element_x + 2] = saturateCast(sum.z);
        output[element_x + 3] = saturateCast(sum.w);
      }
      else {
        output[element_x] = saturateCast(sum.x);
        if (element_x < cols - 1) {
          output[element_x + 1] = saturateCast(sum.y);
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = saturateCast(sum.z);
        }
      }
    }
    else {
      if (element_x < cols - 3) {
        output[element_x]     = sum.x;
        output[element_x + 1] = sum.y;
        output[element_x + 2] = sum.z;
        output[element_x + 3] = sum.w;
      }
      else {
        output[element_x] = sum.x;
        if (element_x < cols - 1) {
          output[element_x + 1] = sum.y;
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = sum.z;
        }
      }
    }
  }
}

template <typename Tsrc, typename Tsrc4, typename BorderInterpolation>
__global__
void rowBatch4Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                     int src_offset, int radius_x, bool is_x_symmetric,
                     float* dst, int dst_stride,
                     BorderInterpolation interpolation) {
  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius_x;
  int top_x    = element_x + radius_x;
  if (!is_x_symmetric) {
    top_x -= 1;
  }

  int data_index;
  Tsrc* input;
  Tsrc4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius_x >> (kBlockShiftX1 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius_x) >> (kBlockShiftX1 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (Tsrc*)((uchar*)src + src_offset + element_y * src_stride);
  if (isnt_border_block) {
    for (int i = origin_x; i <= top_x; i++) {
      value.x = input[i];
      value.y = input[i + 1];
      value.z = input[i + 2];
      value.w = input[i + 3];
      sum += value;
    }
  }
  else {
    for (int i = origin_x; i <= top_x; i++) {
      data_index = interpolation(cols, radius_x, i);
      value.x = input[data_index];
      data_index = interpolation(cols, radius_x, i + 1);
      value.y = input[data_index];
      data_index = interpolation(cols, radius_x, i + 2);
      value.z = input[data_index];
      data_index = interpolation(cols, radius_x, i + 3);
      value.w = input[data_index];
      sum += value;
    }
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  if (element_x < cols - 3) {
    output[element_x]     = sum.x;
    output[element_x + 1] = sum.y;
    output[element_x + 2] = sum.z;
    output[element_x + 3] = sum.w;
  }
  else {
    output[element_x] = sum.x;
    if (element_x < cols - 1) {
      output[element_x + 1] = sum.y;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = sum.z;
    }
  }
}

template <typename Tdst, typename BorderInterpolation>
__global__
void colSharedKernel(const float* src, int rows, int cols4, int cols,
                     int src_stride, int radius_y, bool is_y_symmetric,
                     bool normalize, float weight, Tdst* dst, int dst_stride,
                     int dst_offset, BorderInterpolation interpolation) {
  __shared__ float4 data[kDimY0 * 3][kDimX0];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;
  if (element_x >= cols4) {
    return;
  }

  float4* input;
  float4 value;
  int index;
  int ksize_y = (radius_y << 1) + 1;
  if (!is_y_symmetric) {
    ksize_y -= 1;
  }

  if (threadIdx.y < radius_y) {
    if (blockIdx.y == 0) {
      index = interpolation(rows, radius_y, element_y - radius_y);
    }
    else {
      index = element_y - radius_y;
    }
    input = (float4*)((uchar*)src + index * src_stride);
    value = input[element_x];
    data[threadIdx.y][threadIdx.x] = value;
  }

  if (element_y < rows) {
    input = (float4*)((uchar*)src + element_y * src_stride);
    value = input[element_x];
    data[radius_y + threadIdx.y][threadIdx.x] = value;
  }

  if (threadIdx.y < radius_y) {
    index = (rows - radius_y) >> kShiftY0;
    if (blockIdx.y >= index) {
      if (blockIdx.y != gridDim.y - 1) {
        index = interpolation(rows, radius_y, element_y + kDimY0);
        input = (float4*)((uchar*)src + index * src_stride);
        value = input[element_x];
        data[radius_y + kDimY0 + threadIdx.y][threadIdx.x] = value;
      }
      else {
        index = interpolation(rows, radius_y, rows + threadIdx.y);
        input = (float4*)((uchar*)src + index * src_stride);
        value = input[element_x];
        index = rows - (blockIdx.y << kShiftY0);
        data[radius_y + index + threadIdx.y][threadIdx.x] = value;
      }
    }
    else {
      index = element_y + kDimY0;
      input = (float4*)((uchar*)src + index * src_stride);
      value = input[element_x];
      data[radius_y + kDimY0 + threadIdx.y][threadIdx.x] = value;
    }
  }
  __syncthreads();

  if (element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (index = 0; index < ksize_y; index++) {
    sum += data[threadIdx.y + index][threadIdx.x];
  }

  if (normalize) {
    sum.x *= weight;
    sum.y *= weight;
    sum.z *= weight;
    sum.w *= weight;
  }

  Tdst* output = (Tdst*)((uchar*)dst + dst_offset + element_y * dst_stride);
  index = element_x << 2;
  if (element_x < cols4 - 1) {
    if (sizeof(Tdst) == 1) {
      output[index] = saturateCast(sum.x);
      output[index + 1] = saturateCast(sum.y);
      output[index + 2] = saturateCast(sum.z);
      output[index + 3] = saturateCast(sum.w);
    }
    else {
      output[index] = sum.x;
      output[index + 1] = sum.y;
      output[index + 2] = sum.z;
      output[index + 3] = sum.w;
    }
  }
  else {
    if (sizeof(Tdst) == 1) {
      output[index] = saturateCast(sum.x);
      if (index < cols - 1) {
        output[index + 1] = saturateCast(sum.y);
      }
      if (index < cols - 2) {
        output[index + 2] = saturateCast(sum.z);
      }
      if (index < cols - 3) {
        output[index + 3] = saturateCast(sum.w);
      }
    }
    else {
      output[index] = sum.x;
      if (index < cols - 1) {
        output[index + 1] = sum.y;
      }
      if (index < cols - 2) {
        output[index + 2] = sum.z;
      }
      if (index < cols - 3) {
        output[index + 3] = sum.w;
      }
    }
  }
}

template <typename Tdst, typename BorderInterpolation>
__global__
void colBatch4Kernel(const float* src, int rows, int cols, int src_stride,
                     int radius_y, bool is_y_symmetric, bool normalize,
                     float weight, Tdst* dst, int dst_stride, int dst_offset,
                     BorderInterpolation interpolation) {
  __shared__ Tdst data[kBlockDimY1][kBlockDimX1 << 2];

  int element_x = (blockIdx.x << (kBlockShiftX1 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_y = element_y - radius_y;
  int top_y    = element_y + radius_y;
  if (!is_y_symmetric) {
    top_y -= 1;
  }

  int data_index;
  float* input;
  float value;
  float sum = 0.f;

  bool isnt_border_block = true;
  data_index = radius_y >> kBlockShiftY1;
  if (blockIdx.y <= data_index) isnt_border_block = false;
  data_index = (rows - radius_y) >> kBlockShiftY1;
  if (blockIdx.y >= data_index) isnt_border_block = false;

  if (isnt_border_block) {
    for (int i = origin_y; i <= top_y; i++) {
      input = (float*)((uchar*)src + i * src_stride);
      value = input[element_x];
      sum += value;
    }
  }
  else {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius_y, i);
      input = (float*)((uchar*)src + data_index * src_stride);
      value = input[element_x];
      sum += value;
    }
  }

  if (normalize) {
    sum *= weight;
  }

  if (sizeof(Tdst) == 1) {
    data[threadIdx.y][threadIdx.x] = saturateCast(sum);
  }
  __syncthreads();

  Tdst* output = (Tdst*)((uchar*)dst + dst_offset + element_y * dst_stride);
  if (sizeof(Tdst) == 1) {
    if (threadIdx.x < kBlockDimX1) {
      element_x = (((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2);
      data_index = threadIdx.x << 2;
      if (element_x < cols - 3) {
        output[element_x]     = data[threadIdx.y][data_index];
        output[element_x + 1] = data[threadIdx.y][data_index + 1];
        output[element_x + 2] = data[threadIdx.y][data_index + 2];
        output[element_x + 3] = data[threadIdx.y][data_index + 3];
      }
      else if (element_x < cols) {
        output[element_x] = data[threadIdx.y][data_index];
        if (element_x < cols - 1) {
          output[element_x + 1] = data[threadIdx.y][data_index + 1];
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = data[threadIdx.y][data_index + 2];
        }
      }
      else {
      }
    }
  }
  else {
    output[element_x] = sum;
  }
}

#define RUN_CHANNEL1_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
rowColC1Kernel<Tsrc, Tdst, Interpolation><<<grid, block, 0, stream>>>(src,     \
    rows, cols, src_stride, src_offset, radius_x, radius_y, is_x_symmetric,    \
    is_y_symmetric, normalize, weight, dst, dst_stride, dst_offset,            \
    interpolation);

#define RUN_KERNELS(Tsrc, Tdst, Interpolation)                                 \
Interpolation interpolation;                                                   \
rowBatch4Kernel<Tsrc, Tsrc ## 4, Interpolation><<<grid, block, 0, stream>>>(   \
    src, rows, cols, src_stride, src_offset, radius_x, is_x_symmetric, buffer, \
    pitch, interpolation);                                                     \
if (ksize_x <= 33 && ksize_y <= 33) {                                          \
  colSharedKernel<Tdst, Interpolation><<<grid1, block1, 0, stream>>>(buffer,   \
      rows, columns4, columns, pitch, radius_y, is_y_symmetric, normalize,     \
      weight, dst, dst_stride, dst_offset, interpolation);                     \
}                                                                              \
else {                                                                         \
  colBatch4Kernel<Tdst, Interpolation><<<grid2, block2, 0, stream>>>(buffer,   \
      rows, columns, pitch, radius_y, is_y_symmetric, normalize, weight, dst,  \
      dst_stride, dst_offset, interpolation);                                  \
}

RetCode boxFilter(const float* src, int rows, int cols, int channels,
                  int src_stride, int src_offset, int ksize_x, int ksize_y,
                  bool normalize, float* dst, int dst_stride, int dst_offset,
                  BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src_offset >= 0);
  PPL_ASSERT(dst_offset >= 0);
  PPL_ASSERT(ksize_x > 0);
  PPL_ASSERT(ksize_y > 0);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize_x == 1 && ksize_y == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, rows * src_stride,
                             cudaMemcpyDeviceToDevice, stream);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  int radius_x = ksize_x >> 1;
  int radius_y = ksize_y >> 1;
  bool is_x_symmetric = ksize_x & 1;
  bool is_y_symmetric = ksize_y & 1;
  float weight = 1.f / (ksize_x * ksize_y);

  if (ksize_x <= SMALL_KSIZE && ksize_y <= SMALL_KSIZE && channels == 1) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNEL1_SMALL_KERNELS(float, float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNEL1_SMALL_KERNELS(float, float, ReflectBorder);
    }
    else {
      RUN_CHANNEL1_SMALL_KERNELS(float, float, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block1, grid1;
  block1.x = kDimX0;
  block1.y = kDimY0;
  int columns = cols * channels;
  int columns4 = divideUp(columns, 4, 2);
  grid1.x = divideUp(columns4, kDimX0, kShiftX0);
  grid1.y = divideUp(rows, kDimY0, kShiftY0);

  dim3 block2, grid2;
  block2.x = (kBlockDimX1 << 2);
  block2.y = kBlockDimY1;
  grid2.x  = divideUp(columns, (kBlockDimX1 << 2), (kBlockShiftX1 + 2));
  grid2.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  float* buffer;
  size_t pitch;
  code = cudaMallocPitch(&buffer, &pitch, cols * channels * sizeof(float),
                         rows);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }

  if (border_type == BORDER_REPLICATE) {
    RUN_KERNELS(float, float, ReplicateBorder);
  }
  else if (border_type == BORDER_REFLECT) {
    RUN_KERNELS(float, float, ReflectBorder);
  }
  else {
    RUN_KERNELS(float, float, Reflect101Border);
  }

  cudaFree(buffer);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** splitChannels() ******************************/

__global__
void split3ChannelsKernel(const float* src, int rows, int cols, int src_stride,
                          float* dst, int dst_stride, int dst0_offset,
                          int dst1_offset, int dst2_offset) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int input_x = element_x * 3;
  float* input = (float*)((uchar*)src + element_y * src_stride);
  float value0 = input[input_x];
  float value1 = input[input_x + 1];
  float value2 = input[input_x + 2];

  int offset = element_y * dst_stride;
  float* output0 = (float*)((uchar*)dst + dst0_offset + offset);
  float* output1 = (float*)((uchar*)dst + dst1_offset + offset);
  float* output2 = (float*)((uchar*)dst + dst2_offset + offset);
  output0[element_x] = value0;
  output1[element_x] = value1;
  output2[element_x] = value2;
}

RetCode split3Channels(const float* src, int rows, int cols, int src_stride,
                       float* dst, int dst_stride, int dst0_offset,
                       int dst1_offset, int dst2_offset, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 3 * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst0_offset >= 0);
  PPL_ASSERT(dst1_offset >= 0);
  PPL_ASSERT(dst2_offset >= 0);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  split3ChannelsKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      dst, dst_stride, dst0_offset, dst1_offset, dst2_offset);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

__global__
void split4ChannelsKernel(const float* src, int rows, int cols, int src_stride,
                          float* dst, int dst_stride, int dst0_offset,
                          int dst1_offset, int dst2_offset, int dst3_offset) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int input_x = element_x << 2;
  float* input = (float*)((uchar*)src + element_y * src_stride);
  float value0 = input[input_x];
  float value1 = input[input_x + 1];
  float value2 = input[input_x + 2];
  float value3 = input[input_x + 3];

  int offset = element_y * dst_stride;
  float* output0 = (float*)((uchar*)dst + dst0_offset + offset);
  float* output1 = (float*)((uchar*)dst + dst1_offset + offset);
  float* output2 = (float*)((uchar*)dst + dst2_offset + offset);
  float* output3 = (float*)((uchar*)dst + dst3_offset + offset);
  output0[element_x] = value0;
  output1[element_x] = value1;
  output2[element_x] = value2;
  output3[element_x] = value3;
}

RetCode split4Channels(const float* src, int rows, int cols, int src_stride,
                       float* dst, int dst_stride, int dst0_offset,
                       int dst1_offset, int dst2_offset, int dst3_offset,
                       cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 4 * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst0_offset >= 0);
  PPL_ASSERT(dst1_offset >= 0);
  PPL_ASSERT(dst2_offset >= 0);
  PPL_ASSERT(dst3_offset >= 0);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  split4ChannelsKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      dst, dst_stride, dst0_offset, dst1_offset, dst2_offset, dst3_offset);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** mergeChannels() ******************************/

__global__
void merge3ChannelsKernel(const float* src, int rows, int cols, int src_stride,
                          int src0_offset, int src1_offset, int src2_offset,
                          float* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src_stride;
  float* input0 = (float*)((uchar*)src + src0_offset + offset);
  float* input1 = (float*)((uchar*)src + src1_offset + offset);
  float* input2 = (float*)((uchar*)src + src2_offset + offset);
  float value0  = input0[element_x];
  float value1  = input1[element_x];
  float value2  = input2[element_x];

  element_x = element_x * 3;
  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
}

RetCode merge3Channels(const float* src, int rows, int cols, int src_stride,
                       int src0_offset, int src1_offset, int src2_offset,
                       float* dst, int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 3 * (int)sizeof(float));
  PPL_ASSERT(src0_offset >= 0);
  PPL_ASSERT(src1_offset >= 0);
  PPL_ASSERT(src2_offset >= 0);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  merge3ChannelsKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      src0_offset, src1_offset, src2_offset, dst, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

__global__
void merge4ChannelsKernel(const float* src, int rows, int cols, int src_stride,
                          int src0_offset, int src1_offset, int src2_offset,
                          int src3_offset, float* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  int offset = element_y * src_stride;
  float* input0 = (float*)((uchar*)src + src0_offset + offset);
  float* input1 = (float*)((uchar*)src + src1_offset + offset);
  float* input2 = (float*)((uchar*)src + src2_offset + offset);
  float* input3 = (float*)((uchar*)src + src3_offset + offset);

  float value0  = input0[element_x];
  float value1  = input1[element_x];
  float value2  = input2[element_x];
  float value3  = input3[element_x];

  element_x = element_x << 2;
  dst_stride >>= 2;
  float* output = dst + element_y * dst_stride;
  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
  output[element_x + 3] = value3;
}

RetCode merge4Channels(const float* src, int rows, int cols, int src_stride,
                       int src0_offset, int src1_offset, int src2_offset,
                       int src3_offset, float* dst, int dst_stride,
                       cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst  != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 4 * (int)sizeof(float));
  PPL_ASSERT(src0_offset >= 0);
  PPL_ASSERT(src1_offset >= 0);
  PPL_ASSERT(src2_offset >= 0);
  PPL_ASSERT(src3_offset >= 0);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  merge4ChannelsKernel<<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
      src0_offset, src1_offset, src2_offset, src3_offset, dst, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** guidedFilter() ******************************/

/*
 * guide image: 1 channel.
 * input image: 1 channel.
 * output image: 1 channel.
 */
void guidedFilter_1to1(const float* src, int src_rows, int src_cols,
                       int src_stride, const float* guide, int guide_stride,
                       float* dst, int dst_stride, int radius, double eps,
                       BorderType border_type, cudaStream_t stream) {
  float* buffer;
  size_t pitch;
  cudaMallocPitch(&buffer, &pitch, src_cols * sizeof(float), src_rows * 8);

  int offset = pitch * src_rows;
  float* II = buffer;
  float* IP = buffer;
  float* meanI = buffer;
  float* meanP = buffer;
  float* meanII = buffer;
  float* meanIP = buffer;
  float* varI = buffer;
  float* covIP = buffer;
  int IP_offset     = offset;
  int meanI_offset  = offset * 2;
  int meanP_offset  = offset * 3;
  int meanII_offset = offset * 4;
  int meanIP_offset = offset * 5;
  int varI_offset   = offset * 6;
  int covIP_offset  = offset * 7;

  multiply(guide, src_rows, src_cols, 1, guide_stride, 0, guide, guide_stride,
           0, II, pitch, 0, 1.f, stream);
  multiply(guide, src_rows, src_cols, 1, guide_stride, 0, src, src_stride, 0,
           IP, pitch, IP_offset, 1.f, stream);

  int side_length = (radius << 1) + 1;
  boxFilter(guide, src_rows, src_cols, 1, src_stride, 0, side_length,
            side_length, true, meanI, pitch, meanI_offset, border_type, stream);
  boxFilter(src, src_rows, src_cols, 1, src_stride, 0, side_length, side_length,
            true, meanP, pitch, meanP_offset, border_type, stream);
  boxFilter(II, src_rows, src_cols, 1, src_stride, 0, side_length, side_length,
            true, meanII, pitch, meanII_offset, border_type, stream);
  boxFilter(IP, src_rows, src_cols, 1, src_stride, IP_offset, side_length,
            side_length, true, meanIP, pitch, meanIP_offset, border_type,
            stream);

  float* meanII_mul = II;
  float* meanIP_mul = IP;
  multiply(meanI, src_rows, src_cols, 1, pitch, meanI_offset, meanI, pitch,
           meanI_offset,  meanII_mul, pitch, 0, 1.f, stream);
  multiply(meanI, src_rows, src_cols, 1, pitch, meanI_offset, meanP, pitch,
           meanP_offset, meanIP_mul, pitch, IP_offset, 1.f, stream);
  subtract(meanII, src_rows, src_cols, 1, pitch, meanII_offset, meanII_mul, 0,
           varI, varI_offset, stream);
  subtract(meanIP, src_rows, src_cols, 1, pitch, meanIP_offset, meanIP_mul,
           IP_offset, covIP, covIP_offset, stream);

  float* a = meanII;
  float* b = meanIP;
  float* aMeanI = covIP;
  addScalar(varI, src_rows, src_cols, 1, pitch, varI_offset, eps, stream);
  divide(covIP, src_rows, src_cols, 1, pitch, covIP_offset, varI, pitch,
         varI_offset, a, pitch, meanII_offset, 1.f, stream);
  multiply(a, src_rows, src_cols, 1, pitch, meanII_offset, meanI, pitch,
           meanI_offset, aMeanI, pitch, covIP_offset, 1.f, stream);
  subtract(meanP, src_rows, src_cols, 1, pitch, meanP_offset, aMeanI,
           covIP_offset, b, meanIP_offset, stream);

  float* meanA = II;
  float* meanB = IP;
  boxFilter(a, src_rows, src_cols, 1, src_stride, meanII_offset, side_length,
            side_length, true, meanA, pitch, 0, border_type, stream);
  boxFilter(b, src_rows, src_cols, 1, src_stride, meanIP_offset, side_length,
            side_length, true, meanB, pitch, IP_offset, border_type, stream);

  float* meanAI = meanI;
  multiply(meanA, src_rows, src_cols, 1, pitch, 0, guide, guide_stride, 0,
           meanAI, pitch, meanI_offset, 1.f, stream);
  add(meanAI, src_rows, src_cols, 1, meanI_offset, meanB, IP_offset, dst,
      dst_stride, 0, stream);

  cudaFree(buffer);
}

void guidedFilter_1to1(const float* src, int src_rows, int src_cols,
                       int src_stride, int src_offset, const float* guide,
                       int guide_stride, float* dst, int dst_stride,
                       int dst_offset, int radius, double eps,
                       BorderType border_type, cudaStream_t stream) {
  float* buffer;
  size_t pitch;
  cudaMallocPitch(&buffer, &pitch, src_cols * sizeof(float), src_rows * 8);

  int offset = pitch * src_rows;
  float* II = buffer;
  float* IP = buffer;
  float* meanI = buffer;
  float* meanP = buffer;
  float* meanII = buffer;
  float* meanIP = buffer;
  float* varI = buffer;
  float* covIP = buffer;
  int IP_offset     = offset;
  int meanI_offset  = offset * 2;
  int meanP_offset  = offset * 3;
  int meanII_offset = offset * 4;
  int meanIP_offset = offset * 5;
  int varI_offset   = offset * 6;
  int covIP_offset  = offset * 7;

  multiply(guide, src_rows, src_cols, 1, guide_stride, 0, guide, guide_stride,
           0, II, pitch, 0, 1.f, stream);
  multiply(guide, src_rows, src_cols, 1, guide_stride, 0, src, src_stride,
           src_offset, IP, pitch, IP_offset, 1.f, stream);

  int side_length = (radius << 1) + 1;
  boxFilter(guide, src_rows, src_cols, 1, src_stride, 0, side_length,
            side_length, true, meanI, pitch, meanI_offset, border_type, stream);
  boxFilter(src, src_rows, src_cols, 1, src_stride, src_offset, side_length,
            side_length, true, meanP, pitch, meanP_offset, border_type, stream);
  boxFilter(II, src_rows, src_cols, 1, src_stride, 0, side_length, side_length,
            true, meanII, pitch, meanII_offset, border_type, stream);
  boxFilter(IP, src_rows, src_cols, 1, src_stride, IP_offset, side_length,
            side_length, true, meanIP, pitch, meanIP_offset, border_type,
            stream);

  float* meanII_mul = II;
  float* meanIP_mul = IP;
  multiply(meanI, src_rows, src_cols, 1, pitch, meanI_offset, meanI, pitch,
           meanI_offset, meanII_mul, pitch, 0, 1.f, stream);
  multiply(meanI, src_rows, src_cols, 1, pitch, meanI_offset, meanP, pitch,
           meanP_offset, meanIP_mul, pitch, IP_offset, 1.f, stream);
  subtract(meanII, src_rows, src_cols, 1, pitch, meanII_offset, meanII_mul, 0,
           varI, varI_offset, stream);
  subtract(meanIP, src_rows, src_cols, 1, pitch, meanIP_offset, meanIP_mul,
           IP_offset, covIP, covIP_offset, stream);

  float* a = meanII;
  float* b = meanIP;
  float* aMeanI = covIP;
  addScalar(varI, src_rows, src_cols, 1, pitch, varI_offset, eps, stream);
  divide(covIP, src_rows, src_cols, 1, pitch, covIP_offset, varI, pitch,
         varI_offset, a, pitch, meanII_offset, 1.f, stream);
  multiply(a, src_rows, src_cols, 1, pitch, meanII_offset, meanI, pitch,
           meanI_offset, aMeanI, pitch, covIP_offset, 1.f, stream);
  subtract(meanP, src_rows, src_cols, 1, pitch, meanP_offset, aMeanI,
           covIP_offset, b, meanIP_offset, stream);

  float* meanA = II;
  float* meanB = IP;
  boxFilter(a, src_rows, src_cols, 1, src_stride, meanII_offset, side_length,
            side_length, true, meanA, pitch, 0, border_type, stream);
  boxFilter(b, src_rows, src_cols, 1, src_stride, meanIP_offset, side_length,
            side_length, true, meanB, pitch, IP_offset, border_type, stream);

  float* meanAI = meanI;
  multiply(meanA, src_rows, src_cols, 1, pitch, 0, guide, guide_stride, 0,
           meanAI, pitch, meanI_offset, 1.f, stream);
  add(meanAI, src_rows, src_cols, 1, meanI_offset, meanB, IP_offset, dst,
      dst_stride, dst_offset, stream);

  cudaFree(buffer);
}

void filtering(const float* src, int rows, int cols, int src_channels,
               int src_stride, const float* guide, int guide_channels,
               int guide_stride, float* dst, int dst_stride, int radius,
               float eps, BorderType border_type, cudaStream_t stream) {
  if (guide_channels == 1) {
    if (src_channels == 1) {
      guidedFilter_1to1(src, rows, cols, src_stride, guide, guide_stride,
                        dst, dst_stride, radius, eps, border_type, stream);
    }
    else if (src_channels == 3) {  // src_channels == 3
      float* buffer;
      size_t pitch;
      cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows * 6);

      int offset = pitch * rows;
      float* src0 = buffer;
      float* src1 = buffer;
      float* src2 = buffer;
      float* dst0 = buffer;
      float* dst1 = buffer;
      float* dst2 = buffer;
      size_t src0_offset = 0;
      size_t src1_offset = offset;
      size_t src2_offset = offset * 2;
      size_t dst0_offset = offset * 3;
      size_t dst1_offset = offset * 4;
      size_t dst2_offset = offset * 5;

      split3Channels(src, rows, cols, src_stride, buffer, pitch, 0, src1_offset,
                     src2_offset, stream);
      guidedFilter_1to1(src0, rows, cols, pitch, src0_offset, guide,
                        guide_stride, dst0, pitch, dst0_offset, radius, eps,
                        border_type, stream);
      guidedFilter_1to1(src1, rows, cols, pitch, src1_offset, guide,
                        guide_stride, dst1, pitch, dst1_offset, radius, eps,
                        border_type, stream);
      guidedFilter_1to1(src2, rows, cols, pitch, src2_offset, guide,
                        guide_stride, dst2, pitch, dst2_offset, radius, eps,
                        border_type, stream);
      merge3Channels(buffer, rows, cols, pitch, dst0_offset, dst1_offset,
                     dst2_offset, dst, dst_stride, stream);

      cudaFree(buffer);
    }
    else {  // src_channels == 4
      float* buffer;
      size_t pitch;
      cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows * 8);

      int offset = pitch * rows;
      float* src0 = buffer;
      float* src1 = buffer;
      float* src2 = buffer;
      float* src3 = buffer;
      float* dst0 = buffer;
      float* dst1 = buffer;
      float* dst2 = buffer;
      float* dst3 = buffer;
      size_t src0_offset = 0;
      size_t src1_offset = offset;
      size_t src2_offset = offset * 2;
      size_t src3_offset = offset * 3;
      size_t dst0_offset = offset * 4;
      size_t dst1_offset = offset * 5;
      size_t dst2_offset = offset * 6;
      size_t dst3_offset = offset * 7;

      split4Channels(src, rows, cols, src_stride, buffer, pitch, 0, src1_offset,
                     src2_offset, src3_offset, stream);
      guidedFilter_1to1(src0, rows, cols, pitch, src0_offset, guide,
                        guide_stride, dst0, pitch, dst0_offset, radius, eps,
                        border_type, stream);
      guidedFilter_1to1(src1, rows, cols, pitch, src1_offset, guide,
                        guide_stride, dst1, pitch, dst1_offset, radius, eps,
                        border_type, stream);
      guidedFilter_1to1(src2, rows, cols, pitch, src2_offset, guide,
                        guide_stride, dst2, pitch, dst2_offset, radius, eps,
                        border_type, stream);
      guidedFilter_1to1(src3, rows, cols, pitch, src3_offset, guide,
                        guide_stride, dst3, pitch, dst3_offset, radius, eps,
                        border_type, stream);
      merge4Channels(buffer, rows, cols, pitch, dst0_offset, dst1_offset,
                     dst2_offset, dst3_offset, dst, dst_stride, stream);

      cudaFree(buffer);
    }
  }
  else {  // guide_channels == 3
    if (src_channels == 1) {
    }
    else if (src_channels == 3) { // src_channels == 3
    }
    else { // src_channels == 4
    }
  }
}

RetCode guidedFilter(const uchar* src, int rows, int cols, int src_channels,
                     int src_stride, const uchar* guide, int guide_channels,
                     int guide_stride, uchar* dst, int dst_stride, int radius,
                     float eps, BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(guide != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));
  PPL_ASSERT(guide_stride >= cols * guide_channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * src_channels * (int)sizeof(uchar));
  PPL_ASSERT(guide_channels == 1);
  PPL_ASSERT(src_channels == 1 || src_channels == 3 || src_channels == 4);
  PPL_ASSERT(radius > 0);
  PPL_ASSERT(eps > 0.0);
  PPL_ASSERT(border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_REFLECT);

  float* fguide;
  float* fsrc;
  float* fdst;
  size_t fguide_stride, fsrc_stride, fdst_stride;
  cudaMallocPitch(&fguide, &fguide_stride,
                  cols * guide_channels * sizeof(float), rows);
  cudaMallocPitch(&fsrc, &fsrc_stride, cols * src_channels * sizeof(float),
                  rows);
  cudaMallocPitch(&fdst, &fdst_stride, cols * src_channels * sizeof(float),
                  rows);

  convertTo(guide, rows, cols, guide_channels, guide_stride, fguide,
            fguide_stride, 1, 0.0, stream);
  convertTo(src, rows, cols, src_channels, src_stride, fsrc, fsrc_stride,
            1, 0.0, stream);

  filtering(fsrc, rows, cols, src_channels, fsrc_stride, fguide, guide_channels,
            fguide_stride, fdst, fdst_stride, radius, eps, border_type, stream);

  convertTo(fdst, rows, cols, src_channels, fdst_stride, dst, dst_stride,
            1, 0.0, stream);

  cudaFree(fguide);
  cudaFree(fsrc);
  cudaFree(fdst);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode guidedFilter(const float* src, int rows, int cols, int src_channels,
                     int src_stride, const float* guide, int guide_channels,
                     int guide_stride, float* dst, int dst_stride, int radius,
                     float eps, BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(guide != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(float));
  PPL_ASSERT(guide_stride >= cols * guide_channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * src_channels * (int)sizeof(float));
  PPL_ASSERT(guide_channels == 1);
  PPL_ASSERT(src_channels == 1 || src_channels == 3 || src_channels == 4);
  PPL_ASSERT(radius > 0);
  PPL_ASSERT(eps > 0.0);
  PPL_ASSERT(border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_REFLECT);

  filtering(src, rows, cols, src_channels, src_stride, guide, guide_channels,
            guide_stride, dst, dst_stride, radius, eps, border_type, stream);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode GuidedFilter<uchar, 1, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(inData, height, width, 1, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<uchar, 3, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(inData, height, width, 3, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<uchar, 4, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(inData, height, width, 4, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 1, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(inData, height, width, 1, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 3, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(inData, height, width, 3, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 4, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(inData, height, width, 4, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
