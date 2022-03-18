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

#include "ppl/cv/cuda/sobel.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define MAX_KSIZE 32
#define SCHARR_SIZE0 -1
#define SCHARR_SIZE1 3
#define RADIUS 8
#define SMALL_KSIZE RADIUS * 2 + 1

__DEVICE__
void createScharrKernels(float* kernel_x, float* kernel_y, int dx, int dy,
                         float scale) {
  if (dx < 0 || dy < 0 || dx + dy != 1) {
    return;
  }

  int ksize = 3;
  for (int k = 0; k < 2; k++) {
    float* kernel = k == 0 ? kernel_x : kernel_y;
    int order = k == 0 ? dx : dy;

    if (order == 0) {
      kernel[0] = 3, kernel[1] = 10, kernel[2] = 3;
    }
    if (order == 1) {
      kernel[0] = -1, kernel[1] = 0, kernel[2] = 1;
    }

    float scale0 = 1.f;
    if (k == 1) {
      scale0 *= scale;
    }
    for (int i = 0; i < ksize; i++) {
      kernel[i] *= scale0;
    }
  }
}

__DEVICE__
void createSobelKernels(float* kernel_x, float* kernel_y, int dx, int dy,
                        int ksize, float scale) {
  if (ksize > 31 || (ksize & 1) == 0) {
    return;
  }

  int i, j, ksize_x = ksize, ksize_y = ksize;
  if (ksize_x == 1 && dx > 0) {
    ksize_x = 3;
  }
  if (ksize_y == 1 && dy > 0) {
    ksize_y = 3;
  }

  for (int k = 0; k < 2; k++) {
    float* kernel = k == 0 ? kernel_x : kernel_y;
    int order = k == 0 ? dx : dy;
    int size = k == 0 ? ksize_x : ksize_y;

    if (size <= order) return;

    if (size == 1) {
      kernel[0] = 0, kernel[1] = 1, kernel[2] = 0;
    }
    else if (size == 3) {
      if (order == 0) {
        kernel[0] = 1, kernel[1] = 2, kernel[2] = 1;
      }
      else if (order == 1) {
        kernel[0] = -1, kernel[1] = 0, kernel[2] = 1;
      }
      else {
        kernel[0] = 1, kernel[1] = -2, kernel[2] = 1;
      }
    }
    else {
      int old_value, new_value;
      kernel[0] = 1;
      for (i = 0; i < size; i++) {
        kernel[i + 1] = 0;
      }

      for (i = 0; i < size - order - 1; i++) {
        old_value = kernel[0];
        for (j = 1; j <= size; j++) {
          new_value = kernel[j] + kernel[j - 1];
          kernel[j - 1] = old_value;
          old_value = new_value;
        }
      }

      for (i = 0; i < order; i++) {
        old_value = -kernel[0];
        for (j = 1; j <= size; j++) {
          new_value = kernel[j - 1] - kernel[j];
          kernel[j - 1] = old_value;
          old_value = new_value;
        }
      }
    }

    float scale0 = 1.f;
    if (k == 1) {
      scale0 *= scale;
    }
    for (i = 0; i < max(ksize_x, ksize_y); i++) {
      kernel[i] *= scale0;
    }
  }
}

template <typename Tsrc, typename Tdst, typename BorderInterpolation>
__global__
void rowColC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                    int dx, int dy, int ksize, Tdst* dst, int dst_stride,
                    float scale, float delta,
                    BorderInterpolation interpolation) {
  __shared__ float data[kDimY0 * 3][(kDimX0 << 2)];
  __shared__ float kernel_x[MAX_KSIZE];
  __shared__ float kernel_y[MAX_KSIZE];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    if (ksize == SCHARR_SIZE0) {
      createScharrKernels(kernel_x, kernel_y, dx, dy, scale);
    }
    else {
      createSobelKernels(kernel_x, kernel_y, dx, dy, ksize, scale);
    }
  }
  __syncthreads();

  if (ksize == SCHARR_SIZE0 || ksize == 1) {
    ksize = SCHARR_SIZE1;
  }
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
        mulAdd(sum, value, kernel_x[kernel_index]);
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
        mulAdd(sum, value, kernel_x[kernel_index]);
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
        mulAdd(sum, value, kernel_x[kernel_index]);
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
        mulAdd(sum, value, kernel_x[kernel_index]);
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
        mulAdd(sum, value, kernel_x[kernel_index]);
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
        mulAdd(sum, value, kernel_x[kernel_index]);
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
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    kernel_index = 0;

    for (int i = 0; i < ksize; i++) {
      data_index = threadIdx.x << 2;
      value.x = data[i + threadIdx.y][data_index];
      value.y = data[i + threadIdx.y][data_index + 1];
      value.z = data[i + threadIdx.y][data_index + 2];
      value.w = data[i + threadIdx.y][data_index + 3];
      mulAdd(sum, value, kernel_y[kernel_index]);
      kernel_index++;
    }

    if (delta != 0.f) {
      sum.x += delta;
      sum.y += delta;
      sum.z += delta;
      sum.w += delta;
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
    else if (sizeof(Tdst) == 2) {
      if (element_x < cols - 3) {
        output[element_x]     = saturateCastF2S(sum.x);
        output[element_x + 1] = saturateCastF2S(sum.y);
        output[element_x + 2] = saturateCastF2S(sum.z);
        output[element_x + 3] = saturateCastF2S(sum.w);
      }
      else {
        output[element_x] = saturateCastF2S(sum.x);
        if (element_x < cols - 1) {
          output[element_x + 1] = saturateCastF2S(sum.y);
        }
        if (element_x < cols - 2) {
          output[element_x + 2] = saturateCastF2S(sum.z);
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
                    int dx, int dy, int ksize, Tdst* dst, int dst_stride,
                    float scale, float delta,
                    BorderInterpolation interpolation) {
  __shared__ Tsrcn row_data[kDimY0 + RADIUS * 2][kDimX0 + RADIUS * 2];
  __shared__ Tbufn col_data[kDimY0 + RADIUS * 2][kDimX0];
  __shared__ float kernel_x[MAX_KSIZE];
  __shared__ float kernel_y[MAX_KSIZE];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    if (ksize == SCHARR_SIZE0) {
      createScharrKernels(kernel_x, kernel_y, dx, dy, scale);
    }
    else {
      createSobelKernels(kernel_x, kernel_y, dx, dy, ksize, scale);
    }
  }
  __syncthreads();

  if (ksize == SCHARR_SIZE0 || ksize == 1) {
    ksize = SCHARR_SIZE1;
  }
  int radius = ksize >> 1;
  int index, y_index, row_index;
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
      mulAdd(sum, row_data[y_index][threadIdx.x + index], kernel_x[index]);
    }

    col_data[y_index][threadIdx.x] = transform<Tbufn>(sum);
    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  if (element_y < rows && element_x < cols) {
    sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (index = 0; index < ksize; index++) {
      mulAdd(sum, col_data[threadIdx.y + index][threadIdx.x], kernel_y[index]);
    }

    if (delta != 0.f) {
      sum.x += delta;
      sum.y += delta;
      sum.z += delta;
      sum.w += delta;
    }

    Tdstn* output = (Tdstn*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = saturateCastVector<Tdstn, float4>(sum);
  }
}

#define RUN_CHANNEL1_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
rowColC1Kernel<Tsrc, Tdst, Interpolation><<<grid, block, 0, stream>>>(src,     \
    rows, cols, src_stride, dx, dy, ksize, dst, dst_stride, scale, delta,      \
    interpolation);

#define RUN_CHANNELN_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  rowColCnKernel<Tsrc, Tsrc ## 3, float ## 3, Tdst, Tdst ## 3, Interpolation>  \
      <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dx, dy, ksize, \
      dst, dst_stride, scale, delta, interpolation);                           \
}                                                                              \
else {                                                                         \
  rowColCnKernel<Tsrc, Tsrc ## 4, float ## 4, Tdst, Tdst ## 4, Interpolation>  \
      <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dx, dy, ksize, \
      dst, dst_stride, scale, delta, interpolation);                           \
}

RetCode sobel(const uchar* src, int rows, int cols, int channels,
              int src_stride, int dx, int dy, int ksize, uchar* dst,
              int dst_stride, float scale, float delta, BorderType border_type,
              cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dx == 0 || dx == 1 || dx == 2 || dx == 3);
  PPL_ASSERT(dy == 0 || dy == 1 || dy == 2 || dy == 3);
  PPL_ASSERT(!(dx > 0 && dy > 0));
  PPL_ASSERT(ksize == SCHARR_SIZE0 || ksize == 1 || ksize == 3 || ksize == 5 ||
             ksize == 7);
  PPL_ASSERT(!(ksize == SCHARR_SIZE0 && (dx > 1 || dy > 1)));
  PPL_ASSERT(!(ksize == 1 && (dx > 2 || dy > 2)));
  PPL_ASSERT(!(ksize == 3 && (dx > 2 || dy > 2)));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize < MAX_KSIZE && channels == 1) {
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
      RUN_CHANNELN_SMALL_KERNELS(uchar, uchar, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNELN_SMALL_KERNELS(uchar, uchar, ReflectBorder);
    }
    else {
      RUN_CHANNELN_SMALL_KERNELS(uchar, uchar, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  return RC_OTHER_ERROR;
}

RetCode sobel(const uchar* src, int rows, int cols, int channels,
              int src_stride, int dx, int dy, int ksize, short* dst,
              int dst_stride, float scale, float delta, BorderType border_type,
              cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(short));
  PPL_ASSERT(dx == 0 || dx == 1 || dx == 2 || dx == 3);
  PPL_ASSERT(dy == 0 || dy == 1 || dy == 2 || dy == 3);
  PPL_ASSERT(!(dx > 0 && dy > 0));
  PPL_ASSERT(ksize == -1 || ksize == 1 || ksize == 3 || ksize == 5 ||
             ksize == 7);
  PPL_ASSERT(!(ksize == -1 && (dx > 1 || dy > 1)));
  PPL_ASSERT(!(ksize == 1 && (dx > 2 || dy > 2)));
  PPL_ASSERT(!(ksize == 3 && (dx > 2 || dy > 2)));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize < MAX_KSIZE && channels == 1) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_REPLICATE) {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, short, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, short, ReflectBorder);
    }
    else {
      RUN_CHANNEL1_SMALL_KERNELS(uchar, short, Reflect101Border);
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
      RUN_CHANNELN_SMALL_KERNELS(uchar, short, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNELN_SMALL_KERNELS(uchar, short, ReflectBorder);
    }
    else {
      RUN_CHANNELN_SMALL_KERNELS(uchar, short, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  return RC_OTHER_ERROR;
}

RetCode sobel(const float* src, int rows, int cols, int channels,
              int src_stride, int dx, int dy, int ksize, float* dst,
              int dst_stride, float scale, float delta, BorderType border_type,
              cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dx == 0 || dx == 1 || dx == 2 || dx == 3);
  PPL_ASSERT(dy == 0 || dy == 1 || dy == 2 || dy == 3);
  PPL_ASSERT(!(dx > 0 && dy > 0));
  PPL_ASSERT(ksize == -1 || ksize == 1 || ksize == 3 || ksize == 5 ||
             ksize == 7);
  PPL_ASSERT(!(ksize == -1 && (dx > 1 || dy > 1)));
  PPL_ASSERT(!(ksize == 1 && (dx > 2 || dy > 2)));
  PPL_ASSERT(!(ksize == 3 && (dx > 2 || dy > 2)));
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize < MAX_KSIZE && channels == 1) {
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
      RUN_CHANNELN_SMALL_KERNELS(float, float, ReplicateBorder);
    }
    else if (border_type == BORDER_REFLECT) {
      RUN_CHANNELN_SMALL_KERNELS(float, float, ReflectBorder);
    }
    else {
      RUN_CHANNELN_SMALL_KERNELS(float, float, Reflect101Border);
    }

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
  }

  return RC_OTHER_ERROR;
}

template <>
RetCode Sobel<uchar, uchar, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               uchar* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  RetCode code = sobel(inData, height, width, 1, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, uchar, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               uchar* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  RetCode code = sobel(inData, height, width, 3, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, uchar, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               uchar* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  RetCode code = sobel(inData, height, width, 4, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, short, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               short* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = sobel(inData, height, width, 1, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, short, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               short* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = sobel(inData, height, width, 3, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<uchar, short, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const uchar* inData,
                               int outWidthStride,
                               short* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = sobel(inData, height, width, 4, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<float, float, 1>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int outWidthStride,
                               float* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sobel(inData, height, width, 1, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<float, float, 3>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int outWidthStride,
                               float* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sobel(inData, height, width, 3, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

template <>
RetCode Sobel<float, float, 4>(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const float* inData,
                               int outWidthStride,
                               float* outData,
                               int dx,
                               int dy,
                               int ksize,
                               float scale,
                               float delta,
                               BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = sobel(inData, height, width, 4, inWidthStride, dx, dy, ksize,
                       outData, outWidthStride, scale, delta, border_type,
                       stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
