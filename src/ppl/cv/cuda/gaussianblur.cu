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

#include "ppl/cv/cuda/gaussianblur.h"

#include <cmath>

#include "utility/utility.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define SMALL_SIZE 7
#define RADIUS 8
#define SMALL_KSIZE RADIUS * 2 + 1
#define SMALL_MAX_KSIZE 32

RetCode sepfilter2D(const uchar* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, uchar* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream);

RetCode sepfilter2D(const float* src, int rows, int cols, int channels,
                    int src_stride, const float* kernel_x,
                    const float* kernel_y, int ksize, float* dst,
                    int dst_stride, float delta, BorderType border_type,
                    cudaStream_t stream);

__global__
void getGaussianKernel(float sigma, int ksize, float* coefficients) {
  float value = sigma > 0 ? sigma : ((ksize - 1) * 0.5f - 1) * 0.3f + 0.8f;
  float scale_2x = -0.5f / (value * value);
  float sum = 0.f;

  int i;
  float x;
  for (i = 0; i < ksize; i++) {
    x = i - (ksize - 1) * 0.5f;
    value = std::exp(scale_2x * x * x);
    coefficients[i] = value;
    sum +=value;
  }

  sum = 1.f / sum;
  for (i = 0; i < ksize; i++) {
    coefficients[i] *= sum;
  }
}

__DEVICE__
void createGaussianKernel(float* coefficients, float sigma, int ksize) {
  bool fixed_kernel = false;
  if ((ksize & 1) == 1 && ksize <= SMALL_SIZE && sigma <= 0) {
    if (ksize == 1) {
      coefficients[0] = 1.f;
    }
    else if (ksize == 3) {
      coefficients[0] = 0.25f;
      coefficients[1] = 0.5f;
      coefficients[2] = 0.25f;
    }
    else if (ksize == 5) {
      coefficients[0] = 0.0625f;
      coefficients[1] = 0.25f;
      coefficients[2] = 0.375f;
      coefficients[3] = 0.25f;
      coefficients[4] = 0.0625f;
    }
    else {
      coefficients[0] = 0.03125f;
      coefficients[1] = 0.109375f;
      coefficients[2] = 0.21875f;
      coefficients[3] = 0.28125f;
      coefficients[4] = 0.21875f;
      coefficients[5] = 0.109375f;
      coefficients[6] = 0.03125f;
    }
    fixed_kernel = true;
  }

  float value = sigma > 0 ? sigma : ((ksize - 1) * 0.5f - 1) * 0.3f + 0.8f;
  float scale_2x = -0.5f / (value * value);
  float sum = 0.f;

  int i;
  float x;
  for (i = 0; i < ksize; i++) {
    x = i - (ksize - 1) * 0.5f;
    value = fixed_kernel ? coefficients[i] : std::exp(scale_2x * x * x);
    if (!fixed_kernel) {
      coefficients[i] = value;
    }
    sum += value;
  }

  sum = 1.f / sum;
  for (i = 0; i < ksize; i++) {
    coefficients[i] *= sum;
  }
}

template <typename Tsrc, typename Tdst, typename BorderInterpolation>
__global__
void rowColC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                    float sigma, int ksize, Tdst* dst, int dst_stride,
                    BorderInterpolation interpolation) {
  __shared__ float data[kDimY0 * 3][(kDimX0 << 2)];
  __shared__ float kernel[SMALL_MAX_KSIZE];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    createGaussianKernel(kernel, sigma, ksize);
  }
  __syncthreads();

  int radius = ksize >> 1;
  int bottom = element_x - radius;
  int top    = element_x + radius;

  int data_index, row_index, kernel_index = 0;
  Tsrc* input;
  float4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  bool isnt_border_block = true;
  data_index = radius >> (kShiftX0 + 2);
  if (blockIdx.x <= data_index) isnt_border_block = false;
  data_index = (cols - radius) >> (kShiftX0 + 2);
  if (blockIdx.x >= data_index) isnt_border_block = false;

  if (threadIdx.y < radius && element_x < cols) {
    row_index = interpolation(rows, radius, element_y - radius);
    input = (Tsrc*)((uchar*)src + row_index * src_stride);
    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
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
    data_index = threadIdx.x << 2;
    data[threadIdx.y][data_index] = sum.x;
    data[threadIdx.y][data_index + 1] = sum.y;
    data[threadIdx.y][data_index + 2] = sum.z;
    data[threadIdx.y][data_index + 3] = sum.w;
  }

  if (element_y < rows && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    input = (Tsrc*)((uchar*)src + element_y * src_stride);
    kernel_index = 0;

    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
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
    data_index = threadIdx.x << 2;
    data[radius + threadIdx.y][data_index] = sum.x;
    data[radius + threadIdx.y][data_index + 1] = sum.y;
    data[radius + threadIdx.y][data_index + 2] = sum.z;
    data[radius + threadIdx.y][data_index + 3] = sum.w;
  }

  if (threadIdx.y < radius && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    if (blockIdx.y != gridDim.y - 1) {
      row_index = interpolation(rows, radius,
                                ((blockIdx.y + 1) << kShiftY0) + threadIdx.y);
    }
    else {
      row_index = interpolation(rows, radius, rows + threadIdx.y);
    }
    input = (Tsrc*)((uchar*)src + row_index * src_stride);
    kernel_index = 0;

    if (isnt_border_block) {
      for (int i = bottom; i <= top; i++) {
        value.x = input[i];
        value.y = input[i + 1];
        value.z = input[i + 2];
        value.w = input[i + 3];
        mulAdd(sum, value, kernel[kernel_index]);
        kernel_index++;
      }
    }
    else {
      for (int i = bottom; i <= top; i++) {
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

    data_index = threadIdx.x << 2;
    if (blockIdx.y != gridDim.y - 1) {
      row_index = radius + kDimY0 + threadIdx.y;
    }
    else {
      row_index = radius + (rows - (blockIdx.y << kShiftY0)) + threadIdx.y;
    }
    data[row_index][data_index] = sum.x;
    data[row_index][data_index + 1] = sum.y;
    data[row_index][data_index + 2] = sum.z;
    data[row_index][data_index + 3] = sum.w;
  }
  __syncthreads();

  if (element_y < rows && element_x < cols) {
    top = (radius << 1) + 1;
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    kernel_index = 0;

    for (int i = 0; i < top; i++) {
      data_index = threadIdx.x << 2;
      value.x = data[i + threadIdx.y][data_index];
      value.y = data[i + threadIdx.y][data_index + 1];
      value.z = data[i + threadIdx.y][data_index + 2];
      value.w = data[i + threadIdx.y][data_index + 3];
      mulAdd(sum, value, kernel[kernel_index]);
      kernel_index++;
    }

    Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
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

template <typename Tsrc, typename Tsrcn, typename Tbufn, typename Tdst,
          typename Tdstn, typename BorderInterpolation>
__global__
void rowColCnKernel(const Tsrc* src, int rows, int cols, int src_stride,
                    float sigma, int ksize, Tdst* dst, int dst_stride,
                    BorderInterpolation interpolation) {
  __shared__ Tsrcn row_data[kDimY0 + RADIUS * 2][kDimX0 + RADIUS * 2];
  __shared__ Tbufn col_data[kDimY0 + RADIUS * 2][kDimX0];
  __shared__ float kernel[SMALL_MAX_KSIZE];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    createGaussianKernel(kernel, sigma, ksize);
  }
  __syncthreads();

  int index, y_index, row_index;
  int radius = ksize >> 1;
  Tsrcn* input;
  float4 sum;

  y_index   = threadIdx.y;
  row_index = element_y - radius;
  while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
         row_index < rows + radius) {
    index = interpolation(rows, radius, row_index);
    input = (Tsrcn*)((uchar*)src + index * src_stride);

    int x_index   = threadIdx.x;
    int col_index = element_x - radius;
    while (col_index < (int)(((blockIdx.x + 1) << kShiftX0) + radius) &&
           col_index < cols + radius) {
      index = interpolation(cols, radius, col_index);
      row_data[y_index][x_index] = input[index];
      x_index   += kDimX0;
      col_index += kDimX0;
    }

    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  y_index   = threadIdx.y;
  row_index = element_y - radius;
  while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
         row_index < rows + radius && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (index = 0; index < ksize; index++) {
      mulAdd(sum, row_data[y_index][threadIdx.x + index], kernel[index]);
    }

    col_data[y_index][threadIdx.x] = transform<Tbufn>(sum);
    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  if (element_y < rows && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (index = 0; index < ksize; index++) {
      mulAdd(sum, col_data[threadIdx.y + index][threadIdx.x], kernel[index]);
    }

    Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = saturateCastVector<Tdstn, float4>(sum);
  }
}

template <typename Tsrc, typename Tsrcn, typename Tdstn,
          typename BorderInterpolation>
__global__
void rowSharedKernel(const Tsrc* src, int rows, int cols, int src_stride,
                     float sigma, int ksize, float* dst, int dst_stride,
                     BorderInterpolation interpolation) {
  __shared__ Tsrcn data[kBlockDimY1][(kBlockDimX1 << 1)];
  __shared__ float kernel[SMALL_MAX_KSIZE];

  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows) {
    return;
  }

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    createGaussianKernel(kernel, sigma, ksize);
  }
  __syncthreads();

  Tsrcn* input = (Tsrcn*)((uchar*)src + element_y * src_stride);
  Tsrcn value;
  int index;
  int radius = ksize >> 1;

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
  __syncthreads();

  if (element_x >= cols) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (index = 0; index < ksize; index++) {
    mulAdd(sum, data[threadIdx.y][threadIdx.x + index], kernel[index]);
  }

  Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<Tdstn, float4>(sum);
}

template <typename Tdst, typename BorderInterpolation>
__global__
void colSharedKernel(const float* src, int rows, int cols4, int cols,
                     int src_stride, float sigma, int ksize, Tdst* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  __shared__ float4 data[kDimY0 * 3][kDimX0];
  __shared__ float kernel[SMALL_MAX_KSIZE];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;
  if (element_x >= cols4) {
    return;
  }

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    createGaussianKernel(kernel, sigma, ksize);
  }
  __syncthreads();

  float4* input;
  float4 value;
  int index;
  int radius = ksize >> 1;

  if (threadIdx.y < radius) {
    if (blockIdx.y == 0) {
      index = interpolation(rows, radius, element_y - radius);
    }
    else {
      index = element_y - radius;
    }
    input = (float4*)((uchar*)src + index * src_stride);
    value = input[element_x];
    data[threadIdx.y][threadIdx.x] = value;
  }

  if (element_y < rows) {
    input = (float4*)((uchar*)src + element_y * src_stride);
    value = input[element_x];
    data[radius + threadIdx.y][threadIdx.x] = value;
  }

  if (threadIdx.y < radius) {
    index = (rows - radius) >> kShiftY0;
    if (blockIdx.y >= index) {
      if (blockIdx.y != gridDim.y - 1) {
        index = interpolation(rows, radius, element_y + kDimY0);
        input = (float4*)((uchar*)src + index * src_stride);
        value = input[element_x];
        data[radius + kDimY0 + threadIdx.y][threadIdx.x] = value;
      }
      else {
        index = interpolation(rows, radius, rows + threadIdx.y);
        input = (float4*)((uchar*)src + index * src_stride);
        value = input[element_x];
        index = rows - (blockIdx.y << kShiftY0);
        data[radius + index + threadIdx.y][threadIdx.x] = value;
      }
    }
    else {
      index = element_y + kDimY0;
      input = (float4*)((uchar*)src + index * src_stride);
      value = input[element_x];
      data[radius + kDimY0 + threadIdx.y][threadIdx.x] = value;
    }
  }
  __syncthreads();

  if (element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (index = 0; index < ksize; index++) {
    mulAdd(sum, data[threadIdx.y + index][threadIdx.x], kernel[index]);
  }

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
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

#define RUN_CHANNEL1_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
rowColC1Kernel<Tsrc, Tdst, Interpolation><<<grid, block, 0, stream>>>(src,     \
    rows, cols, src_stride, sigma, ksize, dst, dst_stride, interpolation);

#define RUN_CHANNELN_SMALL_KERNELS0(Tsrc, Tdst, Interpolation)                 \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  rowColCnKernel<Tsrc, Tsrc ## 3, float ## 3, Tdst, Tdst ## 3, Interpolation>  \
      <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, sigma, ksize,  \
      dst, dst_stride, interpolation);                                         \
}                                                                              \
else {                                                                         \
  rowColCnKernel<Tsrc, Tsrc ## 4, float ## 4, Tdst, Tdst ## 4, Interpolation>  \
      <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, sigma, ksize,  \
      dst, dst_stride, interpolation);                                         \
}

#define RUN_CHANNELN_SMALL_KERNELS1(Tsrc, Tdst, Interpolation)                 \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  rowSharedKernel<Tsrc, Tsrc ## 3, float ## 3, Interpolation><<<grid, block,   \
      0, stream>>>(src, rows, cols, src_stride, sigma, ksize, buffer, pitch,   \
      interpolation);                                                          \
  colSharedKernel<Tdst, Interpolation><<<grid1, block1, 0, stream>>>(buffer,   \
      rows, columns4, columns, pitch, sigma, ksize, dst, dst_stride,           \
      interpolation);                                                          \
}                                                                              \
else {                                                                         \
  rowSharedKernel<Tsrc, Tsrc ## 4, float ## 4, Interpolation><<<grid, block,   \
      0, stream>>>(src, rows, cols, src_stride, sigma, ksize, buffer, pitch,   \
      interpolation);                                                          \
  colSharedKernel<Tdst, Interpolation><<<grid1, block1, 0, stream>>>(buffer,   \
      rows, columns4, columns, pitch, sigma, ksize, dst, dst_stride,           \
      interpolation);                                                          \
}

RetCode gaussianblur(const uchar* src, int rows, int cols, int channels,
                     int src_stride, int ksize, float sigma, uchar* dst,
                     int dst_stride, BorderType border_type,
                     cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, rows * src_stride, cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  if (ksize < SMALL_MAX_KSIZE && channels == 1) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, uchar, ReflectBorder);
    }
    else {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, uchar, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  if (ksize <= SMALL_KSIZE && (channels == 3 || channels == 4)) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(cols, kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNELN_SMALL_KERNELS0(uchar, uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNELN_SMALL_KERNELS0(uchar, uchar, ReflectBorder);
    }
    else {
      RUN_CHANNELN_SMALL_KERNELS0(uchar, uchar, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  size_t kernel_size = ksize * sizeof(float);
  float* gpu_kernel;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMalloc(kernel_size, buffer_block);
    gpu_kernel = (float*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&gpu_kernel, kernel_size);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  getGaussianKernel<<<1, 1>>>(sigma, ksize, gpu_kernel);
  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  RetCode return_code = RC_SUCCESS;
  return_code = sepfilter2D(src, rows, cols, channels, src_stride, gpu_kernel,
                            gpu_kernel, ksize, dst, dst_stride, 0, border_type,
                            stream);

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(gpu_kernel);
  }

  return return_code;
}

RetCode gaussianblur(const float* src, int rows, int cols, int channels,
                     int src_stride, int ksize, float sigma, float* dst,
                     int dst_stride, BorderType border_type,
                     cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize > 0);
  PPL_ASSERT((ksize & 1) == 1);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize == 1 && src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpy(dst, src, rows * src_stride, cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  if (ksize < SMALL_MAX_KSIZE && channels == 1) {
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

  if (ksize <= SMALL_KSIZE && (channels == 3 || channels == 4)) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(cols, kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNELN_SMALL_KERNELS0(float, float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNELN_SMALL_KERNELS0(float, float, ReflectBorder);
    }
    else {
      RUN_CHANNELN_SMALL_KERNELS0(float, float, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  if (ksize < SMALL_MAX_KSIZE && (channels == 3 || channels == 4)) {
    dim3 block, grid;
    block.x = kBlockDimX1;
    block.y = kBlockDimY1;
    grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
    grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

    dim3 block1, grid1;
    block1.x = kDimX0;
    block1.y = kDimY0;
    int columns = cols * channels;
    int columns4 = divideUp(columns, 4, 2);
    grid1.x = divideUp(columns4, kDimX0, kShiftX0);
    grid1.y = divideUp(rows, kDimY0, kShiftY0);

    float* buffer;
    size_t pitch;

    GpuMemoryBlock buffer_block;
    if (memoryPoolUsed()) {
      pplCudaMallocPitch(cols * channels * sizeof(float), rows, buffer_block);
      buffer = (float*)(buffer_block.data);
      pitch  = buffer_block.pitch;
    }
    else {
      code = cudaMallocPitch(&buffer, &pitch, cols * channels * sizeof(float),
                             rows);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNELN_SMALL_KERNELS1(float, float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNELN_SMALL_KERNELS1(float, float, ReflectBorder);
    }
    else {
      RUN_CHANNELN_SMALL_KERNELS1(float, float, Reflect101Border);
    }
    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block);
    }
    else {
      cudaFree(buffer);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
    else {
      return RC_SUCCESS;
    }
  }

  size_t kernel_size = ksize * sizeof(float);
  float* gpu_kernel;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMalloc(kernel_size, buffer_block);
    gpu_kernel = (float*)(buffer_block.data);
  }
  else {
    code = cudaMalloc(&gpu_kernel, kernel_size);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  getGaussianKernel<<<1, 1>>>(sigma, ksize, gpu_kernel);
  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  RetCode return_code = RC_SUCCESS;
  return_code = sepfilter2D(src, rows, cols, channels, src_stride, gpu_kernel,
                            gpu_kernel, ksize, dst, dst_stride, 0, border_type,
                            stream);

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(gpu_kernel);
  }

  return return_code;
}

template <>
RetCode GaussianBlur<uchar, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               uchar* outData,
                               BorderType border_type) {
  RetCode code = gaussianblur(inData, height, width, 1, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<uchar, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               uchar* outData,
                               BorderType border_type) {
  RetCode code = gaussianblur(inData, height, width, 3, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<uchar, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               uchar* outData,
                               BorderType border_type) {
  RetCode code = gaussianblur(inData, height, width, 4, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<float, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               float* outData,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = gaussianblur(inData, height, width, 1, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<float, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               float* outData,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = gaussianblur(inData, height, width, 3, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

template <>
RetCode GaussianBlur<float, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int ksize,
                               float sigma,
                               int outWidthStride,
                               float* outData,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = gaussianblur(inData, height, width, 4, inWidthStride, ksize,
                              sigma, outData, outWidthStride, border_type,
                              stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
