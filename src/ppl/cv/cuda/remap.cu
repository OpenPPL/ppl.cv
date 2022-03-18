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

#include "ppl/cv/cuda/remap.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

__DEVICE__
void clipVector(const float4 &value, uchar3 &result) {
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);
}

__DEVICE__
void clipVector(const float4 &value, uchar4 &result) {
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);
  result.w = saturateCast(value.w);
}

__DEVICE__
void clipVector(const float4 &value, float3 &result) {
  result.x = value.x;
  result.y = value.y;
  result.z = value.z;
}

__DEVICE__
void clipVector(const float4 &value, float4 &result) {
  result.x = value.x;
  result.y = value.y;
  result.z = value.z;
  result.w = value.w;
}

template <typename T>
__global__
void remapLinearC1Kernel(const T* src, int src_rows, int src_cols,
                         int src_stride, const float* map_x, const float* map_y,
                         T* dst, int dst_rows, int dst_cols, int dst_stride,
                         BorderType border_type, T border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int dst_xy = element_y * dst_cols + element_x;
  float float_x = map_x[dst_xy];
  float float_y = map_y[dst_xy];
  int int_x = (int)(float_x);
  int int_y = (int)(float_y);
  float fractional_x = (float)(float_x - int_x);
  float fractional_y = (float)(float_y - int_y);

  float4 fractionals;
  float4 coefficients;
  fractionals.x = 1.0f - fractional_y;
  fractionals.y = fractional_y;
  fractionals.z = 1.0f - fractional_x;
  fractionals.w = fractional_x;
  coefficients.x = fractionals.x * fractionals.z;
  coefficients.y = fractionals.x * fractionals.w;
  coefficients.z = fractionals.y * fractionals.z;
  coefficients.w = fractionals.y * fractionals.w;

  T* input;
  T value0, value1;
  float sum = 0.f;

  if (border_type == BORDER_CONSTANT) {
    bool flag0 = int_x >= 0 && int_x < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag1 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag2 = int_x >= 0 && int_x < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;
    bool flag3 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;

    input = (T*)((uchar*)src + int_y * src_stride);
    value0 = input[int_x];
    value1 = input[int_x + 1];
    sum += (flag0 ? value0 : border_value) * coefficients.x;
    sum += (flag1 ? value1 : border_value) * coefficients.y;

    input = (T*)((uchar*)src + (int_y + 1) * src_stride);
    value0 = input[int_x];
    value1 = input[int_x + 1];
    sum += (flag2 ? value0 : border_value) * coefficients.z;
    sum += (flag3 ? value1 : border_value) * coefficients.w;

    T* output = (T*)((uchar*)dst + element_y * dst_stride);
    if (sizeof(T) == 1) {
      output[element_x] = saturateCast(sum);
    }
    else {
      output[element_x] = sum;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    int int_x1 = int_x + 1;
    int int_y1 = int_y + 1;
    int_x  = clip(int_x, 0, src_cols - 1);
    int_x1 = clip(int_x1, 0, src_cols - 1);
    int_y  = clip(int_y, 0, src_rows - 1);
    int_y1 = clip(int_y1, 0, src_rows - 1);

    input = (T*)((uchar*)src + int_y * src_stride);
    value0 = input[int_x];
    value1 = input[int_x1];
    sum += value0 * coefficients.x;
    sum += value1 * coefficients.y;

    input = (T*)((uchar*)src + int_y1 * src_stride);
    value0 = input[int_x];
    value1 = input[int_x1];
    sum += value0 * coefficients.z;
    sum += value1 * coefficients.w;

    T* output = (T*)((uchar*)dst + element_y * dst_stride);
    if (sizeof(T) == 1) {
      output[element_x] = saturateCast(sum);
    }
    else {
      output[element_x] = sum;
    }
  }
  else if (border_type == BORDER_TRANSPARENT) {
    bool flag0 = int_x >= 0 && int_x < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag1 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag2 = int_x >= 0 && int_x < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;
    bool flag3 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;
    if (flag0 && flag1 && flag2 && flag3) {
      input = (T*)((uchar*)src + int_y * src_stride);
      value0 = input[int_x];
      value1 = input[int_x + 1];
      sum += value0 * coefficients.x;
      sum += value1 * coefficients.y;

      input = (T*)((uchar*)src + (int_y + 1) * src_stride);
      value0 = input[int_x];
      value1 = input[int_x + 1];
      sum += value0 * coefficients.z;
      sum += value1 * coefficients.w;

      T* output = (T*)((uchar*)dst + element_y * dst_stride);
      if (sizeof(T) == 1) {
        output[element_x] = saturateCast(sum);
      }
      else {
        output[element_x] = sum;
      }
    }
  }
  else {
  }
}

template <typename T, typename Tn>
__global__
void remapLinearCnKernel(const T* src, int src_rows, int src_cols,
                         int src_stride, const float* map_x, const float* map_y,
                         T* dst, int dst_rows, int dst_cols, int dst_stride,
                         BorderType border_type, T border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int dst_xy = element_y * dst_cols + element_x;
  float float_x = map_x[dst_xy];
  float float_y = map_y[dst_xy];
  int int_x = (int)(float_x);
  int int_y = (int)(float_y);
  float fractional_x = (float)(float_x - int_x);
  float fractional_y = (float)(float_y - int_y);

  float4 fractionals;
  float4 coefficients;
  fractionals.x = 1.0f - fractional_y;
  fractionals.y = fractional_y;
  fractionals.z = 1.0f - fractional_x;
  fractionals.w = fractional_x;
  coefficients.x = fractionals.x * fractionals.z;
  coefficients.y = fractionals.x * fractionals.w;
  coefficients.z = fractionals.y * fractionals.z;
  coefficients.w = fractionals.y * fractionals.w;

  Tn* input;
  Tn value0, value1;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

  if (border_type == BORDER_CONSTANT) {
    bool flag0 = int_x >= 0 && int_x < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag1 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag2 = int_x >= 0 && int_x < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;
    bool flag3 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;

    input = (Tn*)((uchar*)src + int_y * src_stride);
    value0 = input[int_x];
    value1 = input[int_x + 1];

    if (flag0) {
      fractionals = value0 * coefficients.x;
      sum += fractionals;
    }
    else {
      sum += border_value * coefficients.x;
    }

    if (flag1) {
      fractionals = value1 * coefficients.y;
      sum += fractionals;
    }
    else {
      sum += border_value * coefficients.y;
    }

    input = (Tn*)((uchar*)src + (int_y + 1) * src_stride);
    value0 = input[int_x];
    value1 = input[int_x + 1];

    if (flag2) {
      fractionals = value0 * coefficients.z;
      sum += fractionals;
    }
    else {
      sum += border_value * coefficients.z;
    }

    if (flag3) {
      fractionals = value1 * coefficients.w;
      sum += fractionals;
    }
    else {
      sum += border_value * coefficients.w;
    }

    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
    clipVector(sum, value0);
    output[element_x] = value0;
  }
  else if (border_type == BORDER_REPLICATE) {
    int int_x1 = int_x + 1;
    int int_y1 = int_y + 1;
    int_x  = clip(int_x, 0, src_cols - 1);
    int_x1 = clip(int_x1, 0, src_cols - 1);
    int_y  = clip(int_y, 0, src_rows - 1);
    int_y1 = clip(int_y1, 0, src_rows - 1);

    input = (Tn*)((uchar*)src + int_y * src_stride);
    value0 = input[int_x];
    value1 = input[int_x1];

    fractionals = value0 * coefficients.x;
    sum += fractionals;
    fractionals = value1 * coefficients.y;
    sum += fractionals;

    input = (Tn*)((uchar*)src + int_y1 * src_stride);
    value0 = input[int_x];
    value1 = input[int_x1];

    fractionals = value0 * coefficients.z;
    sum += fractionals;
    fractionals = value1 * coefficients.w;
    sum += fractionals;

    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
    clipVector(sum, value0);
    output[element_x] = value0;
  }
  else if (border_type == BORDER_TRANSPARENT) {
    bool flag0 = int_x >= 0 && int_x < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag1 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag2 = int_x >= 0 && int_x < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;
    bool flag3 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;
    if (flag0 && flag1 && flag2 && flag3) {
      input = (Tn*)((uchar*)src + int_y * src_stride);
      value0 = input[int_x];
      value1 = input[int_x + 1];

      fractionals = value0 * coefficients.x;
      sum += fractionals;
      fractionals = value1 * coefficients.y;
      sum += fractionals;

      input = (Tn*)((uchar*)src + (int_y + 1) * src_stride);
      value0 = input[int_x];
      value1 = input[int_x + 1];

      fractionals = value0 * coefficients.z;
      sum += fractionals;
      fractionals = value1 * coefficients.w;
      sum += fractionals;

      Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
      clipVector(sum, value0);
      output[element_x] = value0;
    }
  }
  else {
  }
}

template <typename T>
__global__
void remapNPC1Kernel(const T* src, int src_rows, int src_cols, int src_stride,
                     const float* map_x, const float* map_y, T* dst,
                     int dst_rows, int dst_cols, int dst_stride,
                     BorderType border_type, T border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int dst_xy = element_y * dst_cols + element_x;
  float float_x = map_x[dst_xy];
  float float_y = map_y[dst_xy];
  int int_x = __float2int_rn(float_x);
  int int_y = __float2int_rn(float_y);

  if (border_type == BORDER_CONSTANT) {
    T* input  = (T*)((uchar*)src + int_y * src_stride);
    T* output = (T*)((uchar*)dst + element_y * dst_stride);
    T value;

    if (int_x >= 0 && int_x < src_cols && int_y >= 0 && int_y < src_rows) {
      value = input[int_x];
      output[element_x] = value;
    }
    else {
      output[element_x] = border_value;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    int_x = clip(int_x, 0, src_cols - 1);
    int_y = clip(int_y, 0, src_rows - 1);

    T* input  = (T*)((uchar*)src + int_y * src_stride);
    T* output = (T*)((uchar*)dst + element_y * dst_stride);
    T value = input[int_x];
    output[element_x] = value;
  }
  else if (border_type == BORDER_TRANSPARENT) {
    if (int_x >= 0 && int_x < src_cols && int_y >= 0 && int_y < src_rows) {
      T* input  = (T*)((uchar*)src + int_y * src_stride);
      T* output = (T*)((uchar*)dst + element_y * dst_stride);
      T value = input[int_x];
      output[element_x] = value;
    }
  }
  else {
  }
}

template <typename T, typename Tn>
__global__
void remapNPCnKernel(const T* src, int src_rows, int src_cols, int src_stride,
                     const float* map_x, const float* map_y, T* dst,
                     int dst_rows, int dst_cols, int dst_stride,
                     BorderType border_type, T border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int dst_xy = element_y * dst_cols + element_x;
  float float_x = map_x[dst_xy];
  float float_y = map_y[dst_xy];
  int int_x = __float2int_rn(float_x);
  int int_y = __float2int_rn(float_y);

  if (border_type == BORDER_CONSTANT) {
    Tn* input  = (Tn*)((uchar*)src + int_y * src_stride);
    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
    Tn value;

    if (int_x >= 0 && int_x < src_cols && int_y >= 0 && int_y < src_rows) {
      value = input[int_x];
      output[element_x] = value;
    }
    else {
      float4 border = make_float4(0.f, 0.f, 0.f, 0.f);
      clipVector(border, value);
      output[element_x] = value;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    int_x = clip(int_x, 0, src_cols - 1);
    int_y = clip(int_y, 0, src_rows - 1);

    Tn* input  = (Tn*)((uchar*)src + int_y * src_stride);
    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
    Tn value = input[int_x];
    output[element_x] = value;
  }
  else if (border_type == BORDER_TRANSPARENT) {
    if (int_x >= 0 && int_x < src_cols && int_y >= 0 && int_y < src_rows) {
      Tn* input  = (Tn*)((uchar*)src + int_y * src_stride);
      Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
      Tn value = input[int_x];
      output[element_x] = value;
    }
  }
  else {
  }
}

RetCode remap(const uchar* src, int src_rows, int src_cols, int channels,
              int src_stride, const float* map_x, const float* map_y,
              uchar* dst, int dst_rows, int dst_cols, int dst_stride,
              InterpolationType interpolation, BorderType border_type,
              uchar border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(map_x != nullptr);
  PPL_ASSERT(map_y != nullptr);
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  if (interpolation == INTERPOLATION_LINEAR) {
    if (channels == 1) {
      remapLinearC1Kernel<uchar><<<grid, block, 0, stream>>>(src, src_rows,
          src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else if (channels == 3) {
      remapLinearCnKernel<uchar, uchar3><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else {  // channels == 4
      remapLinearCnKernel<uchar, uchar4><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      remapNPC1Kernel<uchar><<<grid, block, 0, stream>>>(src, src_rows,
          src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else if (channels == 3) {
      remapNPCnKernel<uchar, uchar3><<<grid, block, 0, stream>>>(src, src_rows,
          src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else {  // channels == 4
      remapNPCnKernel<uchar, uchar4><<<grid, block, 0, stream>>>(src, src_rows,
          src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
  }
  else {
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode remap(const float* src, int src_rows, int src_cols, int channels,
              int src_stride, const float* map_x, const float* map_y,
              float* dst, int dst_rows, int dst_cols, int dst_stride,
              InterpolationType interpolation, BorderType border_type,
              float border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(float));
  PPL_ASSERT(map_x != nullptr);
  PPL_ASSERT(map_y != nullptr);
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  if (interpolation == INTERPOLATION_LINEAR) {
    if (channels == 1) {
      remapLinearC1Kernel<float><<<grid, block, 0, stream>>>(src, src_rows,
          src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else if (channels == 3) {
      remapLinearCnKernel<float, float3><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else {  // channels == 4
      remapLinearCnKernel<float, float4><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      remapNPC1Kernel<float><<<grid, block, 0, stream>>>(src, src_rows,
          src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else if (channels == 3) {
      remapNPCnKernel<float, float3><<<grid, block, 0, stream>>>(src, src_rows,
          src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else {  // channels == 4
      remapNPCnKernel<float, float4><<<grid, block, 0, stream>>>(src, src_rows,
          src_cols, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
  }
  else {
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Remap<uchar, 1>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const uchar* inData,
                        int outHeight,
                        int outWidth,
                        int outWidthStride,
                        uchar* outData,
                        const float* mapX,
                        const float* mapY,
                        InterpolationType interpolation,
                        BorderType border_type,
                        uchar borderValue) {
  RetCode code = remap(inData, inHeight, inWidth, 1, inWidthStride, mapX, mapY,
                       outData, outHeight, outWidth, outWidthStride,
                       interpolation, border_type, borderValue, stream);

  return code;
}

template <>
RetCode Remap<uchar, 3>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const uchar* inData,
                        int outHeight,
                        int outWidth,
                        int outWidthStride,
                        uchar* outData,
                        const float* mapX,
                        const float* mapY,
                        InterpolationType interpolation,
                        BorderType border_type,
                        uchar borderValue) {
  RetCode code = remap(inData, inHeight, inWidth, 3, inWidthStride, mapX, mapY,
                       outData, outHeight, outWidth, outWidthStride,
                       interpolation, border_type, borderValue, stream);

  return code;
}

template <>
RetCode Remap<uchar, 4>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const uchar* inData,
                        int outHeight,
                        int outWidth,
                        int outWidthStride,
                        uchar* outData,
                        const float* mapX,
                        const float* mapY,
                        InterpolationType interpolation,
                        BorderType border_type,
                        uchar borderValue) {
  RetCode code = remap(inData, inHeight, inWidth, 4, inWidthStride, mapX, mapY,
                       outData, outHeight, outWidth, outWidthStride,
                       interpolation, border_type, borderValue, stream);

  return code;
}

template <>
RetCode Remap<float, 1>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const float* inData,
                        int outHeight,
                        int outWidth,
                        int outWidthStride,
                        float* outData,
                        const float* mapX,
                        const float* mapY,
                        InterpolationType interpolation,
                        BorderType border_type,
                        float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = remap(inData, inHeight, inWidth, 1, inWidthStride, mapX, mapY,
                       outData, outHeight, outWidth, outWidthStride,
                       interpolation, border_type, borderValue, stream);

  return code;
}

template <>
RetCode Remap<float, 3>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const float* inData,
                        int outHeight,
                        int outWidth,
                        int outWidthStride,
                        float* outData,
                        const float* mapX,
                        const float* mapY,
                        InterpolationType interpolation,
                        BorderType border_type,
                        float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = remap(inData, inHeight, inWidth, 3, inWidthStride, mapX, mapY,
                       outData, outHeight, outWidth, outWidthStride,
                       interpolation, border_type, borderValue, stream);

  return code;
}

template <>
RetCode Remap<float, 4>(cudaStream_t stream,
                        int inHeight,
                        int inWidth,
                        int inWidthStride,
                        const float* inData,
                        int outHeight,
                        int outWidth,
                        int outWidthStride,
                        float* outData,
                        const float* mapX,
                        const float* mapY,
                        InterpolationType interpolation,
                        BorderType border_type,
                        float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = remap(inData, inHeight, inWidth, 4, inWidthStride, mapX, mapY,
                       outData, outHeight, outWidth, outWidthStride,
                       interpolation, border_type, borderValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
