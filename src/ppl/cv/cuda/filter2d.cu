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

/* __DEVICE__
void mulAdd(float4 &result, uchar &value0, float value1) {
  result.x += value0 * value1;
}

__DEVICE__
void mulAdd(float4 &result, float &value0, float value1) {
  result.x += value0 * value1;
}

__DEVICE__
void mulAdd(float4 &result, uchar3 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
}

__DEVICE__
void mulAdd(float4 &result, float3 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
}

__DEVICE__
void mulAdd(float4 &result, uchar4 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
  result.w += value0.w * value1;
}

__DEVICE__
void mulAdd(float4 &result, float4 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
  result.w += value0.w * value1;
} */

template <typename Tsrc, typename Tsrc4, typename Tdst, typename Tdst4,
          typename BorderInterpolation>
__global__
void filter2DC1Kernel(const Tsrc* src, int rows, int cols, int channels,
                      int src_stride, Tdst* dst, int dst_stride,
                      const float* kernel, int radius, float delta,
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

template <typename Tsrc, typename Tsrc4, typename Tdst, typename Tdst4,
          typename BorderInterpolation>
__global__
void filter2DKernel(const Tsrc* src, int rows, int cols, int channels,
                    int src_stride, Tdst* dst, int dst_stride,
                    const float* kernel, int radius, float delta,
                    BorderInterpolation interpolation) {
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

RetCode filter2D(const uchar* src, int rows, int cols, int channels,
                 int src_stride, uchar* dst, int dst_stride,
                 const float* kernel, int ksize, float delta,
                 BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(kernel != nullptr);
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

  unsigned int radius = ksize >> 1;

  if (border_type == BORDER_TYPE_REPLICATE) {
    ReplicateBorder interpolation;
    if (channels == 1) {
      grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);
      filter2DC1Kernel<uchar, uchar4, uchar, uchar4, ReplicateBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else if (channels == 3) {
      filter2DKernel<uchar, uchar3, uchar, uchar3, ReplicateBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else {
      filter2DKernel<uchar, uchar4, uchar, uchar4, ReplicateBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    ReflectBorder interpolation;
    if (channels == 1) {
      grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);
      filter2DC1Kernel<uchar, uchar4, uchar, uchar4, ReflectBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else if (channels == 3) {
      filter2DKernel<uchar, uchar3, uchar, uchar3, ReflectBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else {
      filter2DKernel<uchar, uchar4, uchar, uchar4, ReflectBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
  }
  else {
    Reflect101Border interpolation;
    if (channels == 1) {
      grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);
      filter2DC1Kernel<uchar, uchar4, uchar, uchar4, Reflect101Border><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else if (channels == 3) {
      filter2DKernel<uchar, uchar3, uchar, uchar3, Reflect101Border><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else {
      filter2DKernel<uchar, uchar4, uchar, uchar4, Reflect101Border><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode filter2D(const float* src, int rows, int cols, int channels,
                 int src_stride, float* dst, int dst_stride,
                 const float* kernel, int ksize, float delta,
                 BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(kernel != nullptr);
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

  unsigned int radius = ksize >> 1;

  if (border_type == BORDER_TYPE_REPLICATE) {
    ReplicateBorder interpolation;
    if (channels == 1) {
      grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
      filter2DC1Kernel<float, float4, float, float4, ReplicateBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else if (channels == 3) {
      filter2DKernel<float, float3, float, float3, ReplicateBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else {
      filter2DKernel<float, float4, float, float4, ReplicateBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
  }
  else if (border_type == BORDER_TYPE_REFLECT) {
    ReflectBorder interpolation;
    if (channels == 1) {
      grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
      filter2DC1Kernel<float, float4, float, float4, ReflectBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else if (channels == 3) {
      filter2DKernel<float, float3, float, float3, ReflectBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else {
      filter2DKernel<float, float4, float, float4, ReflectBorder><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
  }
  else {
    Reflect101Border interpolation;
    if (channels == 1) {
      grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);
      filter2DC1Kernel<float, float4, float, float4, Reflect101Border><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else if (channels == 3) {
      filter2DKernel<float, float3, float, float3, Reflect101Border><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
    }
    else {
      filter2DKernel<float, float4, float, float4, Reflect101Border><<<grid,
          block, 0, stream>>>(src, rows, cols, channels, src_stride, dst,
          dst_stride, kernel, radius, delta, interpolation);
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
RetCode Filter2D<uchar, 1>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           float delta,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 1, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, delta,
                          border_type, stream);

  return code;
}

template <>
RetCode Filter2D<uchar, 3>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           float delta,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 3, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, delta,
                          border_type, stream);

  return code;
}

template <>
RetCode Filter2D<uchar, 4>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           uchar* outData,
                           float delta,
                           BorderType border_type) {
  RetCode code = filter2D(inData, height, width, 4, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, delta,
                          border_type, stream);

  return code;
}

template <>
RetCode Filter2D<float, 1>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 1, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, delta,
                          border_type, stream);

  return code;
}

template <>
RetCode Filter2D<float, 3>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 3, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, delta,
                          border_type, stream);

  return code;
}

template <>
RetCode Filter2D<float, 4>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           int kernel_len,
                           const float* kernel,
                           int outWidthStride,
                           float* outData,
                           float delta,
                           BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = filter2D(inData, height, width, 4, inWidthStride, outData,
                          outWidthStride, kernel, kernel_len, delta,
                          border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
