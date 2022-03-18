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

#include "ppl/cv/cuda/cvtcolor.h"
#include "cvtcolor_compute.hpp"
#include "cvtcolor_memory.hpp"

#include "utility/utility.hpp"

namespace ppl {
namespace cv {
namespace cuda {

// BGR(RBB) <-> BGRA(RGBA)
CVT_COLOR_2VECTORS_INVOCATION(BGR2BGRA, uchar3, uchar4, float3, float4)
CVT_COLOR_2VECTORS_INVOCATION(RGB2RGBA, uchar3, uchar4, float3, float4)
CVT_COLOR_1VECTORS_INVOCATION(BGRA2BGR, uchar4, uchar3, float4, float3)
CVT_COLOR_1VECTORS_INVOCATION(RGBA2RGB, uchar4, uchar3, float4, float3)
CVT_COLOR_2VECTORS_INVOCATION(BGR2RGBA, uchar3, uchar4, float3, float4)
CVT_COLOR_2VECTORS_INVOCATION(RGB2BGRA, uchar3, uchar4, float3, float4)
CVT_COLOR_1VECTORS_INVOCATION(RGBA2BGR, uchar4, uchar3, float4, float3)
CVT_COLOR_1VECTORS_INVOCATION(BGRA2RGB, uchar4, uchar3, float4, float3)

// BGR <-> RGB
CVT_COLOR_1VECTORS_INVOCATION(BGR2RGB, uchar3, uchar3, float3, float3)
CVT_COLOR_1VECTORS_INVOCATION(RGB2BGR, uchar3, uchar3, float3, float3)

// BGRA <-> RGBA
CVT_COLOR_2VECTORS_INVOCATION(BGRA2RGBA, uchar4, uchar4, float4, float4)
CVT_COLOR_2VECTORS_INVOCATION(RGBA2BGRA, uchar4, uchar4, float4, float4)

// BGR/RGB/BGRA/RGBA <-> Gray
CVT_COLOR_2VECTORS_INVOCATION(BGR2GRAY, uchar3, uchar, float3, float)
CVT_COLOR_2VECTORS_INVOCATION(RGB2GRAY, uchar3, uchar, float3, float)
CVT_COLOR_2VECTORS_INVOCATION(BGRA2GRAY, uchar4, uchar, float4, float)
CVT_COLOR_2VECTORS_INVOCATION(RGBA2GRAY, uchar4, uchar, float4, float)
CVT_COLOR_1VECTORS_INVOCATION(GRAY2BGR, uchar, uchar3, float, float3)
CVT_COLOR_1VECTORS_INVOCATION(GRAY2RGB, uchar, uchar3, float, float3)
CVT_COLOR_2VECTORS_INVOCATION(GRAY2BGRA, uchar, uchar4, float, float4)
CVT_COLOR_2VECTORS_INVOCATION(GRAY2RGBA, uchar, uchar4, float, float4)

// BGR/RGB/BGRA/RGBA <-> YCrCb
CVT_COLOR_1VECTORS_INVOCATION(BGR2YCrCb, uchar3, uchar3, float3, float3)
CVT_COLOR_1VECTORS_INVOCATION(RGB2YCrCb, uchar3, uchar3, float3, float3)
CVT_COLOR_1VECTORS_INVOCATION(BGRA2YCrCb, uchar4, uchar3, float4, float3)
CVT_COLOR_1VECTORS_INVOCATION(RGBA2YCrCb, uchar4, uchar3, float4, float3)
CVT_COLOR_1VECTORS_INVOCATION(YCrCb2BGR, uchar3, uchar3, float3, float3)
CVT_COLOR_1VECTORS_INVOCATION(YCrCb2RGB, uchar3, uchar3, float3, float3)
CVT_COLOR_2VECTORS_INVOCATION(YCrCb2BGRA, uchar3, uchar4, float3, float4)
CVT_COLOR_2VECTORS_INVOCATION(YCrCb2RGBA, uchar3, uchar4, float3, float4)

// BGR/RGB/BGRA/RGBA <-> HSV
CVT_COLOR_2VECTORS_INVOCATION(BGR2HSV, uchar3, uchar3, float3, float3)
CVT_COLOR_2VECTORS_INVOCATION(RGB2HSV, uchar3, uchar3, float3, float3)
CVT_COLOR_2VECTORS_INVOCATION(BGRA2HSV, uchar4, uchar3, float4, float3)
CVT_COLOR_2VECTORS_INVOCATION(RGBA2HSV, uchar4, uchar3, float4, float3)
CVT_COLOR_1VECTORS_INVOCATION(HSV2BGR, uchar3, uchar3, float3, float3)
CVT_COLOR_1VECTORS_INVOCATION(HSV2RGB, uchar3, uchar3, float3, float3)
CVT_COLOR_2VECTORS_INVOCATION(HSV2BGRA, uchar3, uchar4, float3, float4)
CVT_COLOR_2VECTORS_INVOCATION(HSV2RGBA, uchar3, uchar4, float3, float4)

// BGR/RGB/BGRA/RGBA <-> LAB
CVT_COLOR_2VECTORS_INVOCATION(BGR2LAB, uchar3, uchar3, float3, float3)
CVT_COLOR_2VECTORS_INVOCATION(RGB2LAB, uchar3, uchar3, float3, float3)
CVT_COLOR_2VECTORS_INVOCATION(BGRA2LAB, uchar4, uchar3, float4, float3)
CVT_COLOR_2VECTORS_INVOCATION(RGBA2LAB, uchar4, uchar3, float4, float3)
CVT_COLOR_1VECTORS_INVOCATION(LAB2BGR, uchar3, uchar3, float3, float3)
CVT_COLOR_1VECTORS_INVOCATION(LAB2RGB, uchar3, uchar3, float3, float3)
CVT_COLOR_1VECTORS_INVOCATION(LAB2BGRA, uchar3, uchar4, float3, float4)
CVT_COLOR_1VECTORS_INVOCATION(LAB2RGBA, uchar3, uchar4, float3, float4)

// BGR/RGB/BGRA/RGBA <-> NV12
CVT_COLOR_TO_NVXX_INVOCATION(BGR2NV12, uchar3)
CVT_COLOR_TO_NVXX_INVOCATION(RGB2NV12, uchar3)
CVT_COLOR_TO_NVXX_INVOCATION(BGRA2NV12, uchar4)
CVT_COLOR_TO_NVXX_INVOCATION(RGBA2NV12, uchar4)
CVT_COLOR_FROM_NVXX_INVOCATION(NV122BGR, uchar3)
CVT_COLOR_FROM_NVXX_INVOCATION(NV122RGB, uchar3)
CVT_COLOR_FROM_NVXX_INVOCATION(NV122BGRA, uchar4)
CVT_COLOR_FROM_NVXX_INVOCATION(NV122RGBA, uchar4)

CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(BGR2NV12, uchar3)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(RGB2NV12, uchar3)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(BGRA2NV12, uchar4)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(RGBA2NV12, uchar4)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV122BGR, uchar3)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV122RGB, uchar3)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV122BGRA, uchar4)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV122RGBA, uchar4)

// BGR/RGB/BGRA/RGBA <-> NV21
CVT_COLOR_TO_NVXX_INVOCATION(BGR2NV21, uchar3)
CVT_COLOR_TO_NVXX_INVOCATION(RGB2NV21, uchar3)
CVT_COLOR_TO_NVXX_INVOCATION(BGRA2NV21, uchar4)
CVT_COLOR_TO_NVXX_INVOCATION(RGBA2NV21, uchar4)
CVT_COLOR_FROM_NVXX_INVOCATION(NV212BGR, uchar3)
CVT_COLOR_FROM_NVXX_INVOCATION(NV212RGB, uchar3)
CVT_COLOR_FROM_NVXX_INVOCATION(NV212BGRA, uchar4)
CVT_COLOR_FROM_NVXX_INVOCATION(NV212RGBA, uchar4)

CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(BGR2NV21, uchar3)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(RGB2NV21, uchar3)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(BGRA2NV21, uchar4)
CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(RGBA2NV21, uchar4)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV212BGR, uchar3)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV212RGB, uchar3)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV212BGRA, uchar4)
CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(NV212RGBA, uchar4)

// BGR/RGB/BGRA/RGBA <-> I420
CVT_COLOR_TO_I420_INVOCATION(BGR2I420, uchar3)
CVT_COLOR_TO_I420_INVOCATION(RGB2I420, uchar3)
CVT_COLOR_TO_I420_INVOCATION(BGRA2I420, uchar4)
CVT_COLOR_TO_I420_INVOCATION(RGBA2I420, uchar4)
CVT_COLOR_FROM_I420_INVOCATION(I4202BGR, uchar3)
CVT_COLOR_FROM_I420_INVOCATION(I4202RGB, uchar3)
CVT_COLOR_FROM_I420_INVOCATION(I4202BGRA, uchar4)
CVT_COLOR_FROM_I420_INVOCATION(I4202RGBA, uchar4)

CVT_COLOR_TO_DISCRETE_I420_INVOCATION(BGR2I420, uchar3)
CVT_COLOR_TO_DISCRETE_I420_INVOCATION(RGB2I420, uchar3)
CVT_COLOR_TO_DISCRETE_I420_INVOCATION(BGRA2I420, uchar4)
CVT_COLOR_TO_DISCRETE_I420_INVOCATION(RGBA2I420, uchar4)
CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(I4202BGR, uchar3)
CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(I4202RGB, uchar3)
CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(I4202BGRA, uchar4)
CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(I4202RGBA, uchar4)

// BGR/GRAY <-> UYVY
CVT_COLOR_FROM_YUV422_UCHAR_INVOCATION(UYVY2BGR, uchar4, uchar3)
CVT_COLOR_YUV422_TO_GRAY_UCHAR_INVOCATION(UYVY2GRAY, uchar2)

// BGR/GRAY <-> YUYV
CVT_COLOR_FROM_YUV422_UCHAR_INVOCATION(YUYV2BGR, uchar4, uchar3)
CVT_COLOR_YUV422_TO_GRAY_UCHAR_INVOCATION(YUYV2GRAY, uchar2)

/******************* definition of YUV2 -> GRAY ********************/

__global__
void cvtColorYUV2GRAYKernel0(const uchar* src, int rows, int cols,
                             int src_stride, uchar* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const uchar4* input = (uchar4*)(src + element_y * src_stride);
  uchar4 value = input[element_x];

  uchar4* output = (uchar4*)(dst + element_y * dst_stride);
  output[element_x] = value;
}

__global__
void cvtColorYUV2GRAYKernel1(const uchar* src, int rows, int cols,
                             int src_stride, uchar* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  const uchar* input = src + element_y * src_stride;
  uchar* output = dst + element_y * dst_stride;
  uchar value0, value1, value2, value3;

  if (index_x < cols - 3) {
    value0 = input[index_x];
    value1 = input[index_x + 1];
    value2 = input[index_x + 2];
    value3 = input[index_x + 3];

    output[index_x]     = value0;
    output[index_x + 1] = value1;
    output[index_x + 2] = value2;
    output[index_x + 3] = value3;
  }
  else {
    value0 = input[index_x];
    if (index_x < cols - 1) {
      value1 = input[index_x + 1];
    }
    if (index_x < cols - 2) {
      value2 = input[index_x + 2];
    }

    output[index_x] = value0;
    if (index_x < cols - 1) {
      output[index_x + 1] = value1;
    }
    if (index_x < cols - 2) {
      output[index_x + 2] = value2;
    }
  }
}

RetCode YUV2GRAY(const uchar* src, int rows, int cols, int src_stride,
                 uchar* dst, int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  int columns = divideUp(cols, 4, 2);
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x = divideUp(columns, kBlockDimX0, kBlockShiftX0);
  grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  int padded_stride = roundUp(cols, 4, 2) * sizeof(uchar);
  if (dst_stride >= padded_stride) {
    cvtColorYUV2GRAYKernel0<<<grid, block, 0, stream>>>(src, rows, columns,
        src_stride, dst, dst_stride);
  }
  else {
    cvtColorYUV2GRAYKernel1<<<grid, block, 0, stream>>>(src, rows, cols,
        src_stride, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode YUV2GRAY<uchar>(cudaStream_t stream,
                        int rows,
                        int cols,
                        int src_stride,
                        const uchar* src,
                        int dst_stride,
                        uchar* dst) {
  RetCode code = YUV2GRAY(src, rows, cols, src_stride, dst, dst_stride, stream);

  return code;
}

/******************* definition of NV12/21 <-> I420 ********************/

__global__
void cvtColorNV2I420Kernel(const uchar* src_y, int rows, int cols,
                           int src_y_stride, const uchar* src_uv,
                           int src_uv_stride, uchar* dst_y, int dst_y_stride,
                           uchar* dst_u, int dst_u_stride, uchar* dst_v,
                           int dst_v_stride, bool is_NV12) {
  int index_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int index_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int element_x = index_x << 1;
  int element_y = index_y << 1;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar* input_y0  = (uchar*)src_y + element_y * src_y_stride;
  uchar* input_y1  = input_y0 + src_y_stride;
  uchar2* input_uv = (uchar2*)((uchar*)src_uv + index_y * src_uv_stride);
  uchar* output_y0  = (uchar*)dst_y + element_y * dst_y_stride;
  uchar* output_y1  = output_y0 + dst_y_stride;
  uchar* output_u = (uchar*)((uchar*)dst_u + index_y * dst_u_stride);
  uchar* output_v = (uchar*)((uchar*)dst_v + index_y * dst_v_stride);

  uchar value_y00, value_y01, value_y10, value_y11;
  value_y00 = input_y0[element_x];
  value_y01 = input_y0[element_x + 1];
  value_y10 = input_y1[element_x];
  value_y11 = input_y1[element_x + 1];
  uchar2 value_uv = input_uv[index_x];

  output_y0[element_x] = value_y00;
  output_y0[element_x + 1] = value_y01;
  output_y1[element_x] = value_y10;
  output_y1[element_x + 1] = value_y11;

  if (is_NV12) {
    output_u[index_x] = value_uv.x;
    output_v[index_x] = value_uv.y;
  }
  else {
    output_u[index_x] = value_uv.y;
    output_v[index_x] = value_uv.x;
  }
}

RetCode NV122I420(const uchar* src_y, int rows, int cols, int src_y_stride,
                  const uchar* src_uv, int src_uv_stride, uchar* dst_y,
                  int dst_y_stride, uchar* dst_u, int dst_u_stride,
                  uchar* dst_v, int dst_v_stride, cudaStream_t stream) {
  PPL_ASSERT(src_y  != nullptr);
  PPL_ASSERT(src_uv != nullptr);
  PPL_ASSERT(dst_u  != nullptr);
  PPL_ASSERT(dst_v  != nullptr);
  PPL_ASSERT(dst_y  != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);
  PPL_ASSERT(src_y_stride  >= cols * (int)sizeof(uchar));
  PPL_ASSERT(src_uv_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_y_stride  >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_u_stride  >= cols / 2 * (int)sizeof(uchar));
  PPL_ASSERT(dst_v_stride  >= cols / 2 * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(cols, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(divideUp(rows, 2, 1), kBlockDimY0, kBlockShiftY0);

  cvtColorNV2I420Kernel<<<grid, block, 0, stream>>>(src_y, rows, cols,
      src_y_stride, src_uv, src_uv_stride, dst_y, dst_y_stride, dst_u,
      dst_u_stride, dst_v, dst_v_stride, true);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode NV122I420<uchar>(cudaStream_t stream,
                         int rows,
                         int cols,
                         int src_y_stride,
                         const uchar* src_y,
                         int src_uv_stride,
                         const uchar* src_uv,
                         int dst_y_stride,
                         uchar* dst_y,
                         int dst_u_stride,
                         uchar* dst_u,
                         int dst_v_stride,
                         uchar* dst_v) {
  RetCode code = NV122I420(src_y, rows, cols, src_y_stride, src_uv,
                           src_uv_stride, dst_y, dst_y_stride, dst_u,
                           dst_u_stride, dst_v, dst_v_stride, stream);

  return code;
}


RetCode NV212I420(const uchar* src_y, int rows, int cols, int src_y_stride,
                  const uchar* src_uv, int src_uv_stride, uchar* dst_y,
                  int dst_y_stride, uchar* dst_u, int dst_u_stride,
                  uchar* dst_v, int dst_v_stride, cudaStream_t stream) {
  PPL_ASSERT(src_y  != nullptr);
  PPL_ASSERT(src_uv != nullptr);
  PPL_ASSERT(dst_u  != nullptr);
  PPL_ASSERT(dst_v  != nullptr);
  PPL_ASSERT(dst_y  != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);
  PPL_ASSERT(src_y_stride  >= cols * (int)sizeof(uchar));
  PPL_ASSERT(src_uv_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_y_stride  >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_u_stride  >= cols / 2 * (int)sizeof(uchar));
  PPL_ASSERT(dst_v_stride  >= cols / 2 * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(cols, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(divideUp(rows, 2, 1), kBlockDimY0, kBlockShiftY0);

  cvtColorNV2I420Kernel<<<grid, block, 0, stream>>>(src_y, rows, cols,
      src_y_stride, src_uv, src_uv_stride, dst_y, dst_y_stride, dst_u,
      dst_u_stride, dst_v, dst_v_stride, false);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode NV212I420<uchar>(cudaStream_t stream,
                         int rows,
                         int cols,
                         int src_y_stride,
                         const uchar* src_y,
                         int src_uv_stride,
                         const uchar* src_uv,
                         int dst_y_stride,
                         uchar* dst_y,
                         int dst_u_stride,
                         uchar* dst_u,
                         int dst_v_stride,
                         uchar* dst_v) {
  RetCode code = NV212I420(src_y, rows, cols, src_y_stride, src_uv,
                           src_uv_stride, dst_y, dst_y_stride, dst_u,
                           dst_u_stride, dst_v, dst_v_stride, stream);

  return code;
}

__global__
void cvtColorI4202NVKernel(const uchar* src_y, int rows, int cols,
                           int src_y_stride, const uchar* src_u,
                           int src_u_stride, const uchar* src_v,
                           int src_v_stride, uchar* dst_y, int dst_y_stride,
                           uchar* dst_uv, int dst_uv_stride, bool is_NV12) {
  int index_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int index_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int element_x = index_x << 1;
  int element_y = index_y << 1;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar* input_y0  = (uchar*)src_y + element_y * src_y_stride;
  uchar* input_y1  = input_y0 + src_y_stride;
  uchar* input_u = (uchar*)((uchar*)src_u + index_y * src_u_stride);
  uchar* input_v = (uchar*)((uchar*)src_v + index_y * src_v_stride);
  uchar* output_y0  = (uchar*)dst_y + element_y * dst_y_stride;
  uchar* output_y1  = output_y0 + dst_y_stride;
  uchar2* output_uv = (uchar2*)((uchar*)dst_uv + index_y * dst_uv_stride);

  uchar value_y00, value_y01, value_y10, value_y11;
  value_y00 = input_y0[element_x];
  value_y01 = input_y0[element_x + 1];
  value_y10 = input_y1[element_x];
  value_y11 = input_y1[element_x + 1];
  uchar value_u = input_u[index_x];
  uchar value_v = input_v[index_x];

  output_y0[element_x] = value_y00;
  output_y0[element_x + 1] = value_y01;
  output_y1[element_x] = value_y10;
  output_y1[element_x + 1] = value_y11;

  if (is_NV12) {
    output_uv[index_x] = make_uchar2(value_u, value_v);
  }
  else {
    output_uv[index_x] = make_uchar2(value_v, value_u);
  }
}

RetCode I4202NV12(const uchar* src_y, int rows, int cols, int src_y_stride,
                  const uchar* src_u, int src_u_stride, const uchar* src_v,
                  int src_v_stride, uchar* dst_y, int dst_y_stride,
                  uchar* dst_uv, int dst_uv_stride, cudaStream_t stream) {
  PPL_ASSERT(src_y  != nullptr);
  PPL_ASSERT(src_u  != nullptr);
  PPL_ASSERT(src_v  != nullptr);
  PPL_ASSERT(dst_y  != nullptr);
  PPL_ASSERT(dst_uv != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);
  PPL_ASSERT(src_y_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(src_u_stride >= cols / 2 * (int)sizeof(uchar));
  PPL_ASSERT(src_v_stride >= cols / 2 * (int)sizeof(uchar));
  PPL_ASSERT(dst_y_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_uv_stride >= cols * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(cols, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(divideUp(rows, 2, 1), kBlockDimY0, kBlockShiftY0);

  cvtColorI4202NVKernel<<<grid, block, 0, stream>>>(src_y, rows, cols,
      src_y_stride, src_u, src_u_stride, src_v, src_v_stride, dst_y,
      dst_y_stride, dst_uv, dst_uv_stride, true);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode I4202NV12<uchar>(cudaStream_t stream,
                         int rows,
                         int cols,
                         int src_y_stride,
                         const uchar* src_y,
                         int src_u_stride,
                         const uchar* src_u,
                         int src_v_stride,
                         const uchar* src_v,
                         int dst_y_stride,
                         uchar* dst_y,
                         int dst_uv_stride,
                         uchar* dst_uv) {
  RetCode code = I4202NV12(src_y, rows, cols, src_y_stride, src_u, src_u_stride,
                           src_v, src_v_stride, dst_y, dst_y_stride, dst_uv,
                           dst_uv_stride, stream);

  return code;
}

RetCode I4202NV21(const uchar* src_y, int rows, int cols, int src_y_stride,
                  const uchar* src_u, int src_u_stride, const uchar* src_v,
                  int src_v_stride, uchar* dst_y, int dst_y_stride,
                  uchar* dst_uv, int dst_uv_stride, cudaStream_t stream) {
  PPL_ASSERT(src_y  != nullptr);
  PPL_ASSERT(src_u  != nullptr);
  PPL_ASSERT(src_v  != nullptr);
  PPL_ASSERT(dst_y  != nullptr);
  PPL_ASSERT(dst_uv != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);
  PPL_ASSERT(src_y_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(src_u_stride >= cols / 2 * (int)sizeof(uchar));
  PPL_ASSERT(src_v_stride >= cols / 2 * (int)sizeof(uchar));
  PPL_ASSERT(dst_y_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_uv_stride >= cols * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(cols, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(divideUp(rows, 2, 1), kBlockDimY0, kBlockShiftY0);

  cvtColorI4202NVKernel<<<grid, block, 0, stream>>>(src_y, rows, cols,
      src_y_stride, src_u, src_u_stride, src_v, src_v_stride, dst_y,
      dst_y_stride, dst_uv, dst_uv_stride, false);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode I4202NV21<uchar>(cudaStream_t stream,
                         int rows,
                         int cols,
                         int src_y_stride,
                         const uchar* src_y,
                         int src_u_stride,
                         const uchar* src_u,
                         int src_v_stride,
                         const uchar* src_v,
                         int dst_y_stride,
                         uchar* dst_y,
                         int dst_uv_stride,
                         uchar* dst_uv) {
  RetCode code = I4202NV21(src_y, rows, cols, src_y_stride, src_u, src_u_stride,
                           src_v, src_v_stride, dst_y, dst_y_stride, dst_uv,
                           dst_uv_stride, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
