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

#include "sepfilter2d.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename Tsrc, typename Tsrc4, typename BorderInterpolation>
__global__
void rowBatch4Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                     const float* kernel, int radius, float* dst,
                     int dst_stride, BorderInterpolation interpolation)
{
  int element_x = (((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2);
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int top_x    = element_x + radius;

  int data_index, kernel_index = 0;
  Tsrc* input;
  Tsrc4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius >> (kBlockShiftX1 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kBlockShiftX1 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (Tsrc*)((uchar*)src + element_y * src_stride);
  if (isnt_border_block) {
    for (int i = origin_x; i <= top_x; i++) {
      value.x = input[i];
      value.y = input[i + 1];
      value.z = input[i + 2];
      value.w = input[i + 3];
      mulAdd(sum, value, kernel[kernel_index]);
      kernel_index++;
    }
  }
  else {
    for (int i = origin_x; i <= top_x; i++) {
      data_index = interpolation(cols, radius, i);
      value.x = input[data_index];
      data_index = interpolation(cols, radius, i + 1);
      value.y = input[data_index];
      data_index = interpolation(cols, radius, i + 2);
      value.z = input[data_index];
      data_index = interpolation(cols, radius, i + 3);
      value.w = input[data_index];
      mulAdd(sum, value, kernel[kernel_index]);
      kernel_index++;
    }
  }

  float* output = (float*)((uchar*)dst + element_y * dst_stride);
  if (element_x < cols - 4) {
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
    if (element_x < cols - 3) {
      output[element_x + 3] = sum.w;
    }
  }
}

template <typename Tsrc, typename Tsrc4, typename Tdst4,
          typename BorderInterpolation>
__global__
void rowBatch2Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                     const float* kernel, int radius, float* dst,
                     int dst_stride, BorderInterpolation interpolation)
{
  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int top_x    = element_x + radius;

  int data_index, kernel_index = 0;
  Tsrc4* input;
  Tsrc4 value0;
  Tsrc4 value1;
  float4 sum0 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 sum1 = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius >> (kBlockShiftX1 + 1);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kBlockShiftX1 + 1);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (Tsrc4*)((uchar*)src + element_y * src_stride);
  if (isnt_border_block) {
    for (int i = origin_x; i <= top_x; i++) {
      value0 = input[i];
      value1 = input[i + 1];
      mulAdd(sum0, value0, kernel[kernel_index]);
      mulAdd(sum1, value1, kernel[kernel_index]);
      kernel_index++;
    }
  }
  else {
    for (int i = origin_x; i <= top_x; i++) {
      data_index = interpolation(cols, radius, i);
      value0 = input[data_index];
      data_index = interpolation(cols, radius, i + 1);
      value1 = input[data_index];
      mulAdd(sum0, value0, kernel[kernel_index]);
      mulAdd(sum1, value1, kernel[kernel_index]);
      kernel_index++;
    }
  }

  Tdst4* output = (Tdst4*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<Tdst4, float4>(sum0);
  if (element_x < cols - 1) {
    output[element_x + 1] = saturate_cast_vector<Tdst4, float4>(sum1);
  }
}

template <typename Tsrc, typename Tsrcn, typename Tdstn,
          typename BorderInterpolation>
__global__
void rowSharedKernel(const Tsrc* src, int rows, int cols, int src_stride,
                     const float* kernel, int radius, float* dst,
                     int dst_stride, BorderInterpolation interpolation)
{
  __shared__ Tsrcn data[kBlockDimY1][(kBlockDimX1 << 1)];

  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;

  Tsrcn* input = (Tsrcn*)((uchar*)src + element_y * src_stride);
  Tsrcn value;
  int index0, index1;

  if (threadIdx.x < radius) {
    if (blockIdx.x == 0) {
      index0 = interpolation(cols, radius, element_x - radius);
    }
    else {
      index0 = element_x - radius;
    }
    value = input[index0];
    data[threadIdx.y][threadIdx.x] = value;
  }

  if (element_x < cols) {
    value = input[element_x];
    data[threadIdx.y][radius + threadIdx.x] = value;
  }

  if (threadIdx.x < radius) {
    index0 = (cols - radius) >> kBlockShiftX1;
    if (blockIdx.x >= index0) {
      if (blockIdx.x != gridDim.x - 1) {
        index0 = interpolation(cols, radius, element_x + kBlockDimX1);
        value = input[index0];
        data[threadIdx.y][radius + kBlockDimX1 + threadIdx.x] = value;
      }
      else {
        index0 = interpolation(cols, radius, cols + threadIdx.x);
        value = input[index0];
        index1 = cols - (blockIdx.x << kBlockShiftX1);
        data[threadIdx.y][radius + index1 + threadIdx.x] = value;
      }
    }
    else {
      index0 = element_x + kBlockDimX1;
      value = input[index0];
      data[threadIdx.y][radius + kBlockDimX1 + threadIdx.x] = value;
    }
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (index1 = 0; index1 < (radius << 1) + 1; index1++) {
    mulAdd(sum, data[threadIdx.y][threadIdx.x + index1], kernel[index1]);
  }

  Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<Tdstn, float4>(sum);
}

template <typename Tsrc, typename Tsrcn, typename Tdstn,
          typename BorderInterpolation>
__global__
void rowFilterKernel(const Tsrc* src, int rows, int cols, int src_stride,
                     const float* kernel, int radius, float* dst,
                     int dst_stride, BorderInterpolation interpolation)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int top_x    = element_x + radius;

  int data_index, kernel_index = 0;
  Tsrcn* input;
  Tsrcn value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius >> kBlockShiftX1;
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> kBlockShiftX1;
  if (blockIdx.x >= data_index) isnt_border_block = false;

  input = (Tsrcn*)((uchar*)src + element_y * src_stride);
  if (isnt_border_block) {
    for (int i = origin_x; i <= top_x; i++) {
      value = input[i];
      mulAdd(sum, value, kernel[kernel_index]);
      kernel_index++;
    }
  }
  else {
    for (int i = origin_x; i <= top_x; i++) {
      data_index = interpolation(cols, radius, i);
      value = input[data_index];
      mulAdd(sum, value, kernel[kernel_index]);
      kernel_index++;
    }
  }

  Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<Tdstn, float4>(sum);
}

template <typename Tdst, typename BorderInterpolation>
__global__
void colSharedKernel(const float* src, int rows, int cols4, int cols,
                     int src_stride, const float* kernel, int radius, Tdst* dst,
                     int dst_stride, float delta,
                     BorderInterpolation interpolation)
{
  __shared__ float4 data[kDimY0 * 3][kDimX0];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  float4* input;
  float4 value;
  int index0, index1;

  if (threadIdx.y < radius) {
    if (blockIdx.y == 0) {
      index0 = interpolation(rows, radius, element_y - radius);
    }
    else {
      index0 = element_y - radius;
    }
    input = (float4*)((uchar*)src + index0 * src_stride);
    value = input[element_x];
    data[threadIdx.y][threadIdx.x] = value;
  }

  if (element_y < rows) {
    input = (float4*)((uchar*)src + element_y * src_stride);
    value = input[element_x];
    data[radius + threadIdx.y][threadIdx.x] = value;
  }

  if (threadIdx.y < radius) {
    index0 = (rows - radius) >> kShiftY0;
    if (blockIdx.y >= index0) {
      if (blockIdx.y != gridDim.y - 1) {
        index0 = interpolation(rows, radius, element_y + kDimY0);
        input = (float4*)((uchar*)src + index0 * src_stride);
        value = input[element_x];
        data[radius + kDimY0 + threadIdx.y][threadIdx.x] = value;
      }
      else {
        index0 = interpolation(rows, radius, rows + threadIdx.y);
        input = (float4*)((uchar*)src + index0 * src_stride);
        value = input[element_x];
        index1 = rows - (blockIdx.y << kShiftY0);
        data[radius + index1 + threadIdx.y][threadIdx.x] = value;
      }
    }
    else {
      index0 = element_y + kDimY0;
      input = (float4*)((uchar*)src + index0 * src_stride);
      value = input[element_x];
      data[radius + kDimY0 + threadIdx.y][threadIdx.x] = value;
    }
  }
  __syncthreads();

  if (element_x >= cols4 || element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (index1 = 0; index1 < (radius << 1) + 1; index1++) {
    mulAdd(sum, data[threadIdx.y + index1][threadIdx.x], kernel[index1]);
  }

  if (delta != 0.f) {
    sum.x += delta;
    sum.y += delta;
    sum.z += delta;
    sum.w += delta;
  }

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  index0 = element_x << 2;
  if (element_x < cols4 - 1) {
    if (sizeof(Tdst) == 1) {
      output[index0] = saturate_cast(sum.x);
      output[index0 + 1] = saturate_cast(sum.y);
      output[index0 + 2] = saturate_cast(sum.z);
      output[index0 + 3] = saturate_cast(sum.w);
    }
    else {
      output[index0] = sum.x;
      output[index0 + 1] = sum.y;
      output[index0 + 2] = sum.z;
      output[index0 + 3] = sum.w;
    }
  }
  else {
    if (sizeof(Tdst) == 1) {
      output[index0] = saturate_cast(sum.x);
      if (index0 < cols - 1) {
        output[index0 + 1] = saturate_cast(sum.y);
      }
      if (index0 < cols - 2) {
        output[index0 + 2] = saturate_cast(sum.z);
      }
      if (index0 < cols - 3) {
        output[index0 + 3] = saturate_cast(sum.w);
      }
    }
    else {
      output[index0] = sum.x;
      if (index0 < cols - 1) {
        output[index0 + 1] = sum.y;
      }
      if (index0 < cols - 2) {
        output[index0 + 2] = sum.z;
      }
      if (index0 < cols - 3) {
        output[index0 + 3] = sum.w;
      }
    }
  }
}

template <typename Tsrcn, typename Tdst, typename Tdstn,
          typename BorderInterpolation>
__global__
void colFilterKernel(const float* src, int rows, int cols, int src_stride,
                     const float* kernel, int radius, Tdst* dst, int dst_stride,
                     float delta, BorderInterpolation interpolation)
{
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_y = element_y - radius;
  int top_y    = element_y + radius;

  int data_index, kernel_index = 0;
  Tsrcn* input;
  Tsrcn value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int i = origin_y; i <= top_y; i++) {
    data_index = interpolation(rows, radius, i);
    input = (Tsrcn*)((uchar*)src + data_index * src_stride);
    value = input[element_x];
    mulAdd(sum, value, kernel[kernel_index]);
    kernel_index++;
  }

  if (delta != 0.f) {
    sum.x += delta;
    sum.y += delta;
    sum.z += delta;
    sum.w += delta;
  }

  Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<Tdstn, float4>(sum);
}

RetCode sepfilter2D(const uchar* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, uchar* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(kernel_x != nullptr);
  PPL_ASSERT(kernel_y != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid1;
  grid1.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid1.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid2;
  grid2.x = divideUp(divideUp(cols, 2, 1), kBlockDimX1, kBlockShiftX1);
  grid2.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block3, grid3;
  block3.x = kDimX0;
  block3.y = kDimY0;
  int columns = cols * channels;
  int columns4 = divideUp(columns, 4, 2);
  grid3.x = divideUp(columns4, kDimX0, kShiftX0);
  grid3.y = divideUp(rows, kDimY0, kShiftY0);

  unsigned int radius = ksize >> 1;
  float* buffer;
  size_t pitch;
  cudaMallocPitch(&buffer, &pitch, cols * channels * sizeof(float), rows);

  if (border_type == BORDER_TYPE_REPLICATE) {
    ReplicateBorder interpolation;
    if (channels == 1) {
      rowBatch4Kernel<uchar, uchar4, ReplicateBorder><<<grid1,
          block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
          buffer, pitch, interpolation);
      if (ksize <= 33) {
        colSharedKernel<uchar, ReplicateBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        colFilterKernel<float, uchar, uchar, ReplicateBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else if (channels == 3) {
      if (ksize <= 33) {
        rowSharedKernel<uchar, uchar3, float3, ReplicateBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<uchar, ReplicateBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowFilterKernel<uchar, uchar3, float3, ReplicateBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius, buffer,
            pitch, interpolation);
        colFilterKernel<float3, uchar, uchar3, ReplicateBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else {
      if (ksize <= 33) {
        rowSharedKernel<uchar, uchar4, float4, ReplicateBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<uchar, ReplicateBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowBatch2Kernel<uchar, uchar4, float4, ReplicateBorder><<<grid2,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colFilterKernel<float4, uchar, uchar4, ReplicateBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    ReflectBorder interpolation;
    if (channels == 1) {
      rowBatch4Kernel<uchar, uchar4, ReflectBorder><<<grid1,
          block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
          buffer, pitch, interpolation);
      if (ksize <= 33) {
        colSharedKernel<uchar, ReflectBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        colFilterKernel<float, uchar, uchar, ReflectBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else if (channels == 3) {
      if (ksize <= 33) {
        rowSharedKernel<uchar, uchar3, float3, ReflectBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<uchar, ReflectBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowFilterKernel<uchar, uchar3, float3, ReflectBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius, buffer,
            pitch, interpolation);
        colFilterKernel<float3, uchar, uchar3, ReflectBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else {
      if (ksize <= 33) {
        rowSharedKernel<uchar, uchar4, float4, ReflectBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<uchar, ReflectBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowBatch2Kernel<uchar, uchar4, float4, ReflectBorder><<<grid2,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colFilterKernel<float4, uchar, uchar4, ReflectBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
  }
  else {
    Reflect101Border interpolation;
    if (channels == 1) {
      rowBatch4Kernel<uchar, uchar4, Reflect101Border><<<grid1,
          block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
          buffer, pitch, interpolation);
      if (ksize <= 33) {
        colSharedKernel<uchar, Reflect101Border><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        colFilterKernel<float, uchar, uchar, Reflect101Border><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else if (channels == 3) {
      if (ksize <= 33) {
        rowSharedKernel<uchar, uchar3, float3, Reflect101Border><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<uchar, Reflect101Border><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowFilterKernel<uchar, uchar3, float3, Reflect101Border><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius, buffer,
            pitch, interpolation);
        colFilterKernel<float3, uchar, uchar3, Reflect101Border><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else {
      if (ksize <= 33) {
        rowSharedKernel<uchar, uchar4, float4, Reflect101Border><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<uchar, Reflect101Border><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowBatch2Kernel<uchar, uchar4, float4, Reflect101Border><<<grid2,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colFilterKernel<float4, uchar, uchar4, Reflect101Border><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
  }

  cudaFree(buffer);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode sepfilter2D(const float* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, float* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(kernel_x != nullptr);
  PPL_ASSERT(kernel_y != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid1;
  grid1.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid1.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid2;
  grid2.x = divideUp(divideUp(cols, 2, 1), kBlockDimX1, kBlockShiftX1);
  grid2.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block3, grid3;
  block3.x = kDimX0;
  block3.y = kDimY0;
  int columns = cols * channels;
  int columns4 = divideUp(columns, 4, 2);
  grid3.x = divideUp(columns4, kDimX0, kShiftX0);
  grid3.y = divideUp(rows, kDimY0, kShiftY0);

  unsigned int radius = ksize >> 1;
  float* buffer;
  size_t pitch;
  cudaMallocPitch(&buffer, &pitch, cols * channels * sizeof(float), rows);

  if (border_type == BORDER_TYPE_REPLICATE) {
    ReplicateBorder interpolation;
    if (channels == 1) {
      rowBatch4Kernel<float, float4, ReplicateBorder><<<grid1,
          block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
          buffer, pitch, interpolation);
      if (ksize <= 33) {
        colSharedKernel<float, ReplicateBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        colFilterKernel<float, float, float, ReplicateBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else if (channels == 3) {
      if (ksize <= 33) {
        rowSharedKernel<float, float3, float3, ReplicateBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<float, ReplicateBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowFilterKernel<float, float3, float3, ReplicateBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius, buffer,
            pitch, interpolation);
        colFilterKernel<float3, float, float3, ReplicateBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else {
      if (ksize <= 33) {
        rowSharedKernel<float, float4, float4, ReplicateBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<float, ReplicateBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowBatch2Kernel<float, float4, float4, ReplicateBorder><<<grid2,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colFilterKernel<float4, float, float4, ReplicateBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    ReflectBorder interpolation;
    if (channels == 1) {
      rowBatch4Kernel<float, float4, ReflectBorder><<<grid1,
          block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
          buffer, pitch, interpolation);
      if (ksize <= 33) {
        colSharedKernel<float, ReflectBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        colFilterKernel<float, float, float, ReflectBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else if (channels == 3) {
      if (ksize <= 33) {
        rowSharedKernel<float, float3, float3, ReflectBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<float, ReflectBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowFilterKernel<float, float3, float3, ReflectBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius, buffer,
            pitch, interpolation);
        colFilterKernel<float3, float, float3, ReflectBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else {
      if (ksize <= 33) {
        rowSharedKernel<float, float4, float4, ReflectBorder><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<float, ReflectBorder><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowBatch2Kernel<float, float4, float4, ReflectBorder><<<grid2,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colFilterKernel<float4, float, float4, ReflectBorder><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
  }
  else {
    Reflect101Border interpolation;
    if (channels == 1) {
      rowBatch4Kernel<float, float4, Reflect101Border><<<grid1,
          block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
          buffer, pitch, interpolation);
      if (ksize <= 33) {
        colSharedKernel<float, Reflect101Border><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        colFilterKernel<float, float, float, Reflect101Border><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else if (channels == 3) {
      if (ksize <= 33) {
        rowSharedKernel<float, float3, float3, Reflect101Border><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<float, Reflect101Border><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowFilterKernel<float, float3, float3, Reflect101Border><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius, buffer,
            pitch, interpolation);
        colFilterKernel<float3, float, float3, Reflect101Border><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
    else {
      if (ksize <= 33) {
        rowSharedKernel<float, float4, float4, Reflect101Border><<<grid,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colSharedKernel<float, Reflect101Border><<<grid3, block3, 0, stream>>>(
            buffer, rows, columns4, columns, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
      else {
        rowBatch2Kernel<float, float4, float4, Reflect101Border><<<grid2,
            block, 0, stream>>>(src, rows, cols, src_stride, kernel_x, radius,
            buffer, pitch, interpolation);
        colFilterKernel<float4, float, float4, Reflect101Border><<<grid,
            block, 0, stream>>>(buffer, rows, cols, pitch, kernel_y, radius, dst,
            dst_stride, delta, interpolation);
      }
    }
  }

  cudaFree(buffer);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode SepFilter2D<uchar, 1>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const uchar* inData,
                              int ksize,
                              const float* kernelX,
                              const float* kernelY,
                              int outWidthStride,
                              uchar* outData,
                              float delta,
                              BorderType border_type) {
  RetCode code = sepfilter2D(inData, height, width, 1, inWidthStride, kernelX,
                             kernelY, ksize, outData, outWidthStride, delta,
                             border_type, stream);

  return code;
}

template <>
RetCode SepFilter2D<uchar, 3>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const uchar* inData,
                              int ksize,
                              const float* kernelX,
                              const float* kernelY,
                              int outWidthStride,
                              uchar* outData,
                              float delta,
                              BorderType border_type) {
  RetCode code = sepfilter2D(inData, height, width, 3, inWidthStride, kernelX,
                             kernelY, ksize, outData, outWidthStride, delta,
                             border_type, stream);

  return code;
}

template <>
RetCode SepFilter2D<uchar, 4>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const uchar* inData,
                              int ksize,
                              const float* kernelX,
                              const float* kernelY,
                              int outWidthStride,
                              uchar* outData,
                              float delta,
                              BorderType border_type) {
  RetCode code = sepfilter2D(inData, height, width, 4, inWidthStride, kernelX,
                             kernelY, ksize, outData, outWidthStride, delta,
                             border_type, stream);

  return code;
}

template <>
RetCode SepFilter2D<float, 1>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const float* inData,
                              int ksize,
                              const float* kernelX,
                              const float* kernelY,
                              int outWidthStride,
                              float* outData,
                              float delta,
                              BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sepfilter2D(inData, height, width, 1, inWidthStride, kernelX,
                             kernelY, ksize, outData, outWidthStride, delta,
                             border_type, stream);

  return code;
}

template <>
RetCode SepFilter2D<float, 3>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const float* inData,
                              int ksize,
                              const float* kernelX,
                              const float* kernelY,
                              int outWidthStride,
                              float* outData,
                              float delta,
                              BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sepfilter2D(inData, height, width, 3, inWidthStride, kernelX,
                             kernelY, ksize, outData, outWidthStride, delta,
                             border_type, stream);

  return code;
}

template <>
RetCode SepFilter2D<float, 4>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const float* inData,
                              int ksize,
                              const float* kernelX,
                              const float* kernelY,
                              int outWidthStride,
                              float* outData,
                              float delta,
                              BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sepfilter2D(inData, height, width, 4, inWidthStride, kernelX,
                             kernelY, ksize, outData, outWidthStride, delta,
                             border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
