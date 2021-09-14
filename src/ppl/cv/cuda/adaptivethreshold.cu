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

#include "ppl/cv/cuda/boxfilter.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define RADIUS 16
#define SMALL_KSIZE RADIUS * 2 + 1

template <typename Tsrc, typename Tdst, typename BorderInterpolation>
__global__
void rowColC1Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                    int radius_x, int radius_y,  bool is_x_symmetric,
                    bool is_y_symmetric, bool normalize, float weight,
                    Tdst* dst, int dst_stride,
                    BorderInterpolation interpolation) {
  __shared__ float data[kDimY0 * 3][(kDimX0 << 2)];

  int element_x = (((blockIdx.x << kShiftX0) + threadIdx.x) << 2);
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
    input = (Tsrc*)((uchar*)src + row_index * src_stride);
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
    input = (Tsrc*)((uchar*)src + element_y * src_stride);

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
    input = (Tsrc*)((uchar*)src + row_index * src_stride);

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

    Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
    if (sizeof(Tdst) == 1) {
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

template <typename Tsrc, typename Tsrc4, typename BorderInterpolation>
__global__
void rowBatch4Kernel(const Tsrc* src, int rows, int cols, int src_stride,
                     int radius_x, bool is_x_symmetric, float* dst,
                     int dst_stride, BorderInterpolation interpolation) {
  int element_x = (((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2);
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

  input = (Tsrc*)((uchar*)src + element_y * src_stride);
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

template <typename Tdst, typename BorderInterpolation>
__global__
void colBatch4Kernel(const float* src, int rows, int cols, int src_stride,
                     int radius_y, bool is_y_symmetric, bool normalize,
                     float weight, Tdst* dst, int dst_stride,
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
    data[threadIdx.y][threadIdx.x] = saturate_cast(sum);
  }
  __syncthreads();

  Tdst* output = (Tdst*)((uchar*)dst + element_y * dst_stride);
  if (sizeof(Tdst) == 1) {
    if (threadIdx.x < kBlockDimX1) {
      element_x = (((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2);
      data_index = threadIdx.x << 2;
      if (element_x <= cols - 4) {
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
        if (element_x < cols - 3) {
          output[element_x + 3] = data[threadIdx.y][data_index + 3];
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

#define RUN_CHANNEL1_SMALL_KERNELS(Interpolation, Tsrc, Tdst)                  \
Interpolation interpolation;                                                   \
rowColC1Kernel<Tsrc, Tdst, Interpolation><<<grid, block, 0, stream>>>(src,     \
    rows, cols, src_stride, radius_x, radius_y, is_x_symmetric, is_y_symmetric,\
    normalize, weight, dst, dst_stride, interpolation);

#define RUN_KERNELS(Interpolation, Tsrc, Tdst)                                 \
Interpolation interpolation;                                                   \
rowBatch4Kernel<Tsrc, Tsrc ## 4, Interpolation><<<grid1, block, 0, stream    \
    >>>(src, rows, cols, src_stride, radius_x, is_x_symmetric, buffer,       \
    pitch, interpolation);                                                   \
colBatch4Kernel<Tdst, Interpolation><<<grid4, block4, 0, stream>>>(buffer, \
    rows, columns, pitch, radius_y, is_y_symmetric, normalize, weight,     \
    dst, dst_stride, interpolation);

RetCode
AdaptiveThreshold(cudaStream_t stream, int rows, int cols, int src_stride,
                  const uchar* src, int dst_stride, uchar* dst, float max_value,
                  int adaptive_method, int threshold_type, int ksize,
                  float delta, BorderType border_type) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(max_value != 0);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(adaptive_method == ADAPTIVE_THRESH_MEAN_C ||
             adaptive_method == ADAPTIVE_THRESH_GAUSSIAN_C);
  PPL_ASSERT(threshold_type == THRESH_BINARY ||
             threshold_type == THRESH_BINARY_INV);
  PPL_ASSERT((ksize & 1) == 1 && ksize > 1);
  PPL_ASSERT(border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_REFLECT ||
             border_type == BORDER_TYPE_REFLECT_101 ||
             border_type == BORDER_TYPE_DEFAULT);

  int channels = 1;
  int ksize_x = ksize;
  int ksize_y = ksize;
  bool normalize = true;

  uchar setted_value = 0;
  if (max_value <= 0) {
    // Zeros<uchar, 1>(stream, rows, cols, dst_stride, dst);
    return RC_SUCCESS;
  }
  else if (max_value < 255.f) {
    setted_value = (uchar)max_value;
  }
  else {
    setted_value = 255;
  }

  cudaError_t code = cudaSuccess;

  int radius_x = ksize_x >> 1;
  int radius_y = ksize_y >> 1;
  bool is_x_symmetric = ksize_x & 1;
  bool is_y_symmetric = ksize_y & 1;
  float weight = 1.0 / (ksize_x * ksize_y);

  if (ksize <= SMALL_KSIZE) {
    dim3 block, grid;
    block.x = kDimX0;
    block.y = kDimY0;
    grid.x = divideUp(divideUp(cols, 4, 2), kDimX0, kShiftX0);
    grid.y = divideUp(rows, kDimY0, kShiftY0);

    if (border_type == BORDER_TYPE_REPLICATE) {
      RUN_CHANNEL1_SMALL_KERNELS(ReplicateBorder, uchar, uchar);
    }
    else if (border_type == BORDER_TYPE_REFLECT) {
      RUN_CHANNEL1_SMALL_KERNELS(ReflectBorder, uchar, uchar);
    }
    else {
      RUN_CHANNEL1_SMALL_KERNELS(Reflect101Border, uchar, uchar);
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
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 grid1;
  grid1.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid1.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  dim3 block4, grid4;
  block4.x = (kBlockDimX1 << 2);
  block4.y = kBlockDimY1;
  int columns = cols * channels;
  grid4.x  = divideUp(columns, (kBlockDimX1 << 2), (kBlockShiftX1 + 2));
  grid4.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  float* buffer;
  size_t pitch;
  code = cudaMallocPitch(&buffer, &pitch, cols * channels * sizeof(float),
                         rows);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_MEMORY_ERROR;
  }

  if (border_type == BORDER_TYPE_REPLICATE) {
    RUN_KERNELS(ReplicateBorder, uchar, uchar);
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    RUN_KERNELS(ReflectBorder, uchar, uchar);
  }
  else {
    RUN_KERNELS(Reflect101Border, uchar, uchar);
  }

  cudaFree(buffer);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

}  // cuda
}  // cv
}  // ppl
