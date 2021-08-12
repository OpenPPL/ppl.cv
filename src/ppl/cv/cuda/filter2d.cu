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

#include "filter2d.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename Tsrc, typename Tdst, typename BorderInterpolation>
__global__
void filter2DC1SharedKernel(const Tsrc* src, int rows, int cols, int src_stride,
                            const float* kernel, int radius, Tdst* dst,
                            int dst_stride, float delta,
                            BorderInterpolation interpolation) {
  __shared__ Tsrc data[kBlockDimY1][(kBlockDimX1 << 3)];

  int element_x = (((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2);
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;

  int origin_y = element_y - radius;
  int top_y    = element_y + radius;

  int index, row_index, prev_index = -2, kernel_index = 0;
  Tsrc* input;
  Tsrc value0, value1, value2, value3;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int i = origin_y; i <= top_y; i++) {
    if (element_y < rows) {
      row_index = interpolation(rows, radius, i);
      if (row_index != prev_index) {
        input = (Tsrc*)((uchar*)src + row_index * src_stride);
        if (threadIdx.x < radius) {
          if (blockIdx.x == 0) {
            index = interpolation(cols, radius, threadIdx.x - radius);
          }
          else {
            index = (blockIdx.x << (kBlockShiftX1 + 2)) + threadIdx.x - radius;
          }
          value0 = input[index];
          data[threadIdx.y][threadIdx.x] = value0;
        }

        if (element_x < cols) {
          value0 = input[element_x];
          value1 = input[element_x + 1];
          value2 = input[element_x + 2];
          value3 = input[element_x + 3];
          index = radius + (threadIdx.x << 2);
          data[threadIdx.y][index] = value0;
          data[threadIdx.y][index + 1] = value1;
          data[threadIdx.y][index + 2] = value2;
          data[threadIdx.y][index + 3] = value3;
        }

        if (threadIdx.x < radius) {
          index = (cols - radius) >> (kBlockShiftX1 + 2);
          if (blockIdx.x >= index) {
            if (blockIdx.x != gridDim.x - 1) {
              index = ((blockIdx.x + 1) << (kBlockShiftX1 + 2)) + threadIdx.x;
              index = interpolation(cols, radius, index);
              value0 = input[index];
              index = radius + (kBlockDimX1 << 2) + threadIdx.x;
              data[threadIdx.y][index] = value0;
            }
            else {
              index = interpolation(cols, radius, cols + threadIdx.x);
              value0 = input[index];
              index = cols - (blockIdx.x << (kBlockShiftX1 + 2));
              index += (radius + threadIdx.x);
              data[threadIdx.y][index] = value0;
            }
          }
          else {
            index = ((blockIdx.x + 1) << (kBlockShiftX1 + 2)) + threadIdx.x;
            value0 = input[index];
            index = radius + (kBlockDimX1 << 2) + threadIdx.x;
            data[threadIdx.y][index] = value0;
          }
        }

        prev_index = row_index;
      }
    }
    __syncthreads();

    if (element_x < cols && element_y < rows) {
      for (index = 0; index < (radius << 1) + 1; index++) {
        mulAdd(sum.x, data[threadIdx.y][(threadIdx.x << 2) + index],
               kernel[kernel_index]);
        mulAdd(sum.y, data[threadIdx.y][(threadIdx.x << 2) + index + 1],
               kernel[kernel_index]);
        mulAdd(sum.z, data[threadIdx.y][(threadIdx.x << 2) + index + 2],
               kernel[kernel_index]);
        mulAdd(sum.w, data[threadIdx.y][(threadIdx.x << 2) + index + 3],
               kernel[kernel_index]);
        kernel_index++;
      }
    }
  }

  if (element_x < cols && element_y < rows) {
    if (delta != 0.f) {
      sum.x += delta;
      sum.y += delta;
      sum.z += delta;
      sum.w += delta;
    }

    Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
    if (sizeof(Tsrc) == 1) {
      if (element_x < cols - 4) {
        output[element_x]     = saturate_cast(sum.x);
        output[element_x + 1] = saturate_cast(sum.y);
        output[element_x + 2] = saturate_cast(sum.z);
        output[element_x + 3] = saturate_cast(sum.w);
      }
      else {
        output[element_x] = saturate_cast(sum.x);
        if (element_x < cols - 1) {
          output[element_x + 1] = saturate_cast(sum.y);
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = saturate_cast(sum.z);
        }
        if (element_x < cols - 3) {
          output[element_x + 3] = saturate_cast(sum.w);
        }
      }
    }
    else {
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
  }
}

template <typename Tsrc, typename Tsrc4, typename Tdst, typename Tdst4,
          typename BorderInterpolation>
__global__
void filter2DC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                      const float* kernel, int radius, Tdst* dst,
                      int dst_stride, float delta,
                      BorderInterpolation interpolation) {
  int element_x, element_y;
  if (sizeof(Tsrc) == 1) {
    element_x = (((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2);
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2);
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int data_index, kernel_index = 0;
  Tsrc* input;
  Tsrc4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  if (sizeof(Tsrc) == 1) {
    data_index = radius >> (kBlockShiftX0 + 2);
    if (blockIdx.x <= data_index) isnt_border_block = false;
    data_index = (cols - radius) >> (kBlockShiftX0 + 2);
    if (blockIdx.x >= data_index) isnt_border_block = false;
  }
  else {
    data_index = radius >> (kBlockShiftX1 + 2);
    if (blockIdx.x <= data_index) isnt_border_block = false;
    data_index = (cols - radius) >> (kBlockShiftX1 + 2);
    if (blockIdx.x >= data_index) isnt_border_block = false;
  }

  if (isnt_border_block) {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input = (Tsrc*)((uchar*)src + data_index * src_stride);
      for (int j = origin_x; j <= top_x; j++) {
        value.x = input[j];
        value.y = input[j + 1];
        value.z = input[j + 2];
        value.w = input[j + 3];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
  }
  else {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input = (Tsrc*)((uchar*)src + data_index * src_stride);
      for (int j = origin_x; j <= top_x; j++) {
        data_index = interpolation(cols, radius, j);
        value.x = input[data_index];
        data_index = interpolation(cols, radius, j + 1);
        value.y = input[data_index];
        data_index = interpolation(cols, radius, j + 2);
        value.z = input[data_index];
        data_index = interpolation(cols, radius, j + 3);
        value.w = input[data_index];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
  }

  if (delta != 0.f) {
    sum.x += delta;
    sum.y += delta;
    sum.z += delta;
    sum.w += delta;
  }

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  if (sizeof(Tsrc) == 1) {
    if (element_x < cols - 4) {
      output[element_x]     = saturate_cast(sum.x);
      output[element_x + 1] = saturate_cast(sum.y);
      output[element_x + 2] = saturate_cast(sum.z);
      output[element_x + 3] = saturate_cast(sum.w);
    }
    else {
      output[element_x] = saturate_cast(sum.x);
      if (element_x < cols - 1) {
        output[element_x + 1] = saturate_cast(sum.y);
      }
      if (element_x < cols - 2) {
        output[element_x + 2] = saturate_cast(sum.z);
      }
      if (element_x < cols - 3) {
        output[element_x + 3] = saturate_cast(sum.w);
      }
    }
  }
  else {
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
}

template <typename Tsrc, typename Tsrcn, typename Tdst, typename Tdstn,
          typename BorderInterpolation>
__global__
void filter2DSharedKernel(const Tsrc* src, int rows, int cols, int src_stride,
                          const float* kernel, int radius, Tdst* dst,
                          int dst_stride, float delta,
                          BorderInterpolation interpolation) {
  __shared__ Tsrcn data[kBlockDimY1][(kBlockDimX1 << 1)];

  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;

  int origin_y = element_y - radius;
  int top_y    = element_y + radius;
  int index, row_index, prev_index = -2, kernel_index = 0;
  Tsrcn* input;
  Tsrcn value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int i = origin_y; i <= top_y; i++) {
    if (element_y < rows) {
      row_index = interpolation(rows, radius, i);
      if (row_index != prev_index) {
        input = (Tsrcn*)((uchar*)src + row_index * src_stride);
        if (threadIdx.x < radius) {
          if (blockIdx.x == 0) {
            index = interpolation(cols, radius, element_x - radius);
          }
          else {
            index = element_x - radius;
          }
          value = input[index];
          data[threadIdx.y][threadIdx.x] = value;
        }

        if (element_x < cols) {
          value = input[element_x];
          data[threadIdx.y][radius + threadIdx.x] = value;
        }

        if (threadIdx.x < radius) {
          index = (cols - radius) >> kBlockShiftX1;
          if (blockIdx.x >= index) {
            if (blockIdx.x != gridDim.x - 1) {
              index = interpolation(cols, radius, element_x + kBlockDimX1);
              value = input[index];
              data[threadIdx.y][radius + kBlockDimX1 + threadIdx.x] = value;
            }
            else {
              index = interpolation(cols, radius, cols + threadIdx.x);
              value = input[index];
              index = cols - (blockIdx.x << kBlockShiftX1);
              data[threadIdx.y][radius + index + threadIdx.x] = value;
            }
          }
          else {
            index = element_x + kBlockDimX1;
            value = input[index];
            data[threadIdx.y][radius + kBlockDimX1 + threadIdx.x] = value;
          }
        }

        prev_index = row_index;
      }
    }
    __syncthreads();

    if (element_x < cols && element_y < rows) {
      for (index = 0; index < (radius << 1) + 1; index++) {
        mulAdd(sum, data[threadIdx.y][threadIdx.x + index],
               kernel[kernel_index]);
        kernel_index++;
      }
    }
  }

  if (element_x < cols && element_y < rows) {
    if (delta != 0.f) {
      sum.x += delta;
      sum.y += delta;
      sum.z += delta;
      sum.w += delta;
    }

    Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = saturate_cast_vector<Tdstn, float4>(sum);
  }
}

template <typename Tsrc, typename Tsrc4, typename Tdst, typename Tdst4,
          typename BorderInterpolation>
__global__
void filter2DKernel(const Tsrc* src, int rows, int cols, int src_stride,
                    const float* kernel, int radius, Tdst* dst, int dst_stride,
                    float delta, BorderInterpolation interpolation) {
  int element_x, element_y;
  if (sizeof(Tsrc) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int origin_x = element_x - radius;
  int origin_y = element_y - radius;
  int top_x    = element_x + radius;
  int top_y    = element_y + radius;

  int data_index, kernel_index = 0;
  Tsrc4* input;
  Tsrc4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  if (sizeof(Tsrc) == 1) {
    data_index = radius >> kBlockShiftX0;
    if (blockIdx.x <= data_index) isnt_border_block = false;
    data_index = (cols - radius) >> kBlockShiftX0;
    if (blockIdx.x >= data_index) isnt_border_block = false;
  }
  else {
    data_index = radius >> kBlockShiftX1;
    if (blockIdx.x <= data_index) isnt_border_block = false;
    data_index = (cols - radius) >> kBlockShiftX1;
    if (blockIdx.x >= data_index) isnt_border_block = false;
  }

  if (isnt_border_block) {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input = (Tsrc4*)((uchar*)src + data_index * src_stride);
      for (int j = origin_x; j <= top_x; j++) {
        value = input[j];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
  }
  else {
    for (int i = origin_y; i <= top_y; i++) {
      data_index = interpolation(rows, radius, i);
      input = (Tsrc4*)((uchar*)src + data_index * src_stride);
      for (int j = origin_x; j <= top_x; j++) {
        data_index = interpolation(cols, radius, j);
        value = input[data_index];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
  }

  if (delta != 0.f) {
    sum.x += delta;
    sum.y += delta;
    sum.z += delta;
    sum.w += delta;
  }

  Tdst4* output = (Tdst4*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<Tdst4, float4>(sum);
}

#define RUN_KERNELS(Interpolation, T, grid_x)                                  \
Interpolation interpolation;                                                   \
if (channels == 1) {                                                           \
  if (ksize <= 33) {                                                           \
    filter2DC1SharedKernel<T, T, Interpolation><<<grid2, block2, 0, stream>>>( \
        src, rows, cols, src_stride, kernel, radius, dst, dst_stride, delta,   \
        interpolation);                                                        \
  }                                                                            \
  else {                                                                       \
    grid.x = grid_x;                                                           \
    filter2DC1Kernel<T, T ## 4, T, T ## 4, Interpolation><<<grid, block, 0,    \
        stream>>>(src, rows, cols, src_stride, kernel, radius, dst, dst_stride,\
        delta, interpolation);                                                 \
  }                                                                            \
}                                                                              \
else if (channels == 3) {                                                      \
  if (ksize <= 33) {                                                           \
    filter2DSharedKernel<T, T ## 3, T, T ## 3, Interpolation><<<grid1, block1, \
        0, stream>>>(src, rows, cols, src_stride, kernel, radius, dst,         \
        dst_stride, delta, interpolation);                                     \
  }                                                                            \
  else {                                                                       \
    filter2DKernel<T, T ## 3, T, T ## 3, Interpolation><<<grid, block, 0,      \
        stream>>>(src, rows, cols, src_stride, kernel, radius, dst, dst_stride,\
        delta, interpolation);                                                 \
  }                                                                            \
}                                                                              \
else {                                                                         \
  if (ksize <= 33) {                                                           \
    filter2DSharedKernel<T, T ## 4, T, T ## 4, Interpolation><<<grid1, block1, \
        0, stream>>>(src, rows, cols, src_stride, kernel, radius, dst,         \
        dst_stride, delta, interpolation);                                     \
  }                                                                            \
  else {                                                                       \
    filter2DKernel<T, T ## 4, T, T ## 4, Interpolation><<<grid, block, 0,      \
        stream>>>(src, rows, cols, src_stride, kernel, radius, dst, dst_stride,\
        delta, interpolation);                                                 \
  }                                                                            \
}

RetCode filter2D(const uchar* src, int rows, int cols, int channels,
                 int src_stride, const float* kernel, int ksize, uchar* dst,
                 int dst_stride, float delta, BorderType border_type,
                 cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(kernel != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  dim3 block1, grid1;
  block1.x = kBlockDimX1;
  block1.y = kBlockDimY1;
  grid1.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid1.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block2, grid2;
  block2.x = kBlockDimX1;
  block2.y = kBlockDimY1;
  grid2.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid2.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  unsigned int radius = ksize >> 1;
  int grid_x = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);

  if (border_type == BORDER_TYPE_REPLICATE) {
    RUN_KERNELS(ReplicateBorder, uchar, grid_x);
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    RUN_KERNELS(ReflectBorder, uchar, grid_x);
  }
  else {
    RUN_KERNELS(Reflect101Border, uchar, grid_x);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode filter2D(const float* src, int rows, int cols, int channels,
                 int src_stride, const float* kernel, int ksize, float* dst,
                 int dst_stride, float delta, BorderType border_type,
                 cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(kernel != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block1 = block, grid1 = grid;

  dim3 block2, grid2;
  block2.x = kBlockDimX1;
  block2.y = kBlockDimY1;
  grid2.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid2.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  unsigned int radius = ksize >> 1;
  int grid_x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);

  if (border_type == BORDER_TYPE_REPLICATE) {
    RUN_KERNELS(ReplicateBorder, float, grid_x);
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    RUN_KERNELS(ReflectBorder, float, grid_x);
  }
  else {
    RUN_KERNELS(Reflect101Border, float, grid_x);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Filter2D<uchar, 1>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int ksize,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           float delta,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 1, inWidthStride, kernel,
                          ksize, outData, outWidthStride, delta, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<uchar, 3>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int ksize,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           float delta,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 3, inWidthStride, kernel,
                          ksize, outData, outWidthStride, delta, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<uchar, 4>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int ksize,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           float delta,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 4, inWidthStride, kernel,
                          ksize, outData, outWidthStride, delta, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<float, 1>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int ksize,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 1, inWidthStride, kernel,
                          ksize, outData, outWidthStride, delta, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<float, 3>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int ksize,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 3, inWidthStride, kernel,
                          ksize, outData, outWidthStride, delta, border_type,
                          stream);

  return code;
}

template <>
RetCode Filter2D<float, 4>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int ksize,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 4, inWidthStride, kernel,
                          ksize, outData, outWidthStride, delta, border_type,
                          stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
