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

#ifndef _ST_HPC_PPL_CV_CUDA_CVTCOLOR_MEMORY_HPP_
#define _ST_HPC_PPL_CV_CUDA_CVTCOLOR_MEMORY_HPP_

#include "cuda_runtime.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/************************ 1-vector kernel template ************************/

template <typename T0, typename T1, typename T2, typename Conversion>
__global__
void cvtColor1VecKernel(const T0* src, int rows, int cols, int src_stride,
                        T0* dst, int dst_stride, Conversion conversion) {
  int element_x, element_y;
  if (sizeof(T0) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T1* input  = (T1*)((uchar*)src + element_y * src_stride);
  T2* output = (T2*)((uchar*)dst + element_y * dst_stride);

  T2 result = conversion(input[element_x]);
  output[element_x] = result;
}

/************************ 2-vector kernel template ************************/

template <typename T0, typename T1, typename T2, typename Conversion>
__global__
void cvtColor2VecKernel0(const T0* src, int rows, int cols, int src_stride,
                         T0* dst, int dst_stride, Conversion conversion) {
  int element_x, element_y;
  if (sizeof(T0) == 1) {
    element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 1;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T1* input = (T1*)((uchar*)src + element_y * src_stride);
  T2* output = (T2*)((uchar*)dst + element_y * dst_stride);

  T1 in_value0, in_value1;
  in_value0 = input[element_x];
  in_value1 = input[element_x + 1];

  T2 out_value0, out_value1;
  out_value0 = conversion(in_value0);
  out_value1 = conversion(in_value1);

  output[element_x]     = out_value0;
  output[element_x + 1] = out_value1;
}

template <typename T0, typename T1, typename T2, typename Conversion>
__global__
void cvtColor2VecKernel1(const T0* src, int rows, int cols, int src_stride,
                         T0* dst, int dst_stride, Conversion conversion) {
  int element_x, element_y;
  if (sizeof(T0) == 1) {
    element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 1;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T1* input = (T1*)((uchar*)src + element_y * src_stride);
  T2* output = (T2*)((uchar*)dst + element_y * dst_stride);

  if (blockIdx.x < gridDim.x - 1) {
    T1 in_value0, in_value1;
    in_value0 = input[element_x];
    in_value1 = input[element_x + 1];

    T2 out_value0, out_value1;
    out_value0 = conversion(in_value0);
    out_value1 = conversion(in_value1);

    output[element_x]     = out_value0;
    output[element_x + 1] = out_value1;
  }
  else {
    T1 in_value0, in_value1;
    in_value0 = input[element_x];
    if (element_x != cols - 1) {
      in_value1 = input[element_x + 1];
    }

    T2 out_value0, out_value1;
    out_value0 = conversion(in_value0);
    if (element_x != cols - 1) {
      out_value1 = conversion(in_value1);
    }

    output[element_x] = out_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = out_value1;
    }
  }
}

/************************ 4-vector kernel template ************************/

template <typename T0, typename T1, typename T2, typename Conversion>
__global__
void cvtColor4VecKernel0(const T0* src, int rows, int cols, int src_stride,
                         T0* dst, int dst_stride, Conversion conversion) {
  int element_x, element_y;
  if (sizeof(T0) == 1) {
    element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T1* input = (T1*)((uchar*)src + element_y * src_stride);
  T2* output = (T2*)((uchar*)dst + element_y * dst_stride);

  T1 in_value0, in_value1, in_value2, in_value3;
  in_value0 = input[element_x];
  in_value1 = input[element_x + 1];
  in_value2 = input[element_x + 2];
  in_value3 = input[element_x + 3];

  T2 out_value0, out_value1, out_value2, out_value3;
  out_value0 = conversion(in_value0);
  out_value1 = conversion(in_value1);
  out_value2 = conversion(in_value2);
  out_value3 = conversion(in_value3);

  output[element_x]     = out_value0;
  output[element_x + 1] = out_value1;
  output[element_x + 2] = out_value2;
  output[element_x + 3] = out_value3;
}

template <typename T0, typename T1, typename T2, typename Conversion>
__global__
void cvtColor4VecKernel1(const T0* src, int rows, int cols, int src_stride,
                         T0* dst, int dst_stride, Conversion conversion) {
  int element_x, element_y;
  if (sizeof(T0) == 1) {
    element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 2;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T1* input = (T1*)((uchar*)src + element_y * src_stride);
  T2* output = (T2*)((uchar*)dst + element_y * dst_stride);

  if (blockIdx.x < gridDim.x - 1) {
    T1 in_value0, in_value1, in_value2, in_value3;
    in_value0 = input[element_x];
    in_value1 = input[element_x + 1];
    in_value2 = input[element_x + 2];
    in_value3 = input[element_x + 3];

    T2 out_value0, out_value1, out_value2, out_value3;
    out_value0 = conversion(in_value0);
    out_value1 = conversion(in_value1);
    out_value2 = conversion(in_value2);
    out_value3 = conversion(in_value3);

    output[element_x]     = out_value0;
    output[element_x + 1] = out_value1;
    output[element_x + 2] = out_value2;
    output[element_x + 3] = out_value3;
  }
  else {
    T1 in_value0, in_value1, in_value2, in_value3;
    in_value0 = input[element_x];
    if (element_x < cols - 1) {
      in_value1 = input[element_x + 1];
    }
    if (element_x < cols - 2) {
      in_value2 = input[element_x + 2];
    }
    if (element_x < cols - 3) {
      in_value3 = input[element_x + 3];
    }

    T2 out_value0, out_value1, out_value2, out_value3;
    out_value0 = conversion(in_value0);
    if (element_x < cols - 1) {
      out_value1 = conversion(in_value1);
    }
    if (element_x < cols - 2) {
      out_value2 = conversion(in_value2);
    }
    if (element_x < cols - 3) {
      out_value3 = conversion(in_value3);
    }

    output[element_x] = out_value0;
    if (element_x < cols - 1) {
      output[element_x + 1] = out_value1;
    }
    if (element_x < cols - 2) {
      output[element_x + 2] = out_value2;
    }
    if (element_x < cols - 3) {
      output[element_x + 3] = out_value3;
    }
  }
}

/*************************** invocation template ***************************/

#define CVT_COLOR_VECTORS_INVOCATION(function, base_type, from_type, to_type,  \
                                     vectors)                                  \
RetCode function(const base_type* src, int rows, int cols, int src_stride,     \
                 base_type* dst, int dst_stride, cudaStream_t stream) {        \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(src_stride >= cols * (int)sizeof(from_type));                     \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(to_type));                       \
                                                                               \
  int padded_stride;                                                           \
  function ## Compute conversion;                                              \
                                                                               \
  dim3 block, grid;                                                            \
  if (sizeof(base_type) == 1) {                                                \
    block.x = kBlockDimX0;                                                     \
    block.y = kBlockDimY0;                                                     \
  }                                                                            \
  else {                                                                       \
    block.x = kBlockDimX1;                                                     \
    block.y = kBlockDimY1;                                                     \
  }                                                                            \
                                                                               \
  if (vectors == 4) {                                                          \
    if (sizeof(base_type) == 1) {                                              \
      grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);     \
      grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);                     \
    }                                                                          \
    else {                                                                     \
      grid.x = divideUp(divideUp(cols, 4, 2), kBlockDimX1, kBlockShiftX1);     \
      grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);                     \
    }                                                                          \
                                                                               \
    padded_stride = roundUp(cols, 4, 2) * sizeof(to_type);                     \
    if (dst_stride >= padded_stride) {                                         \
      cvtColor4VecKernel0<base_type, from_type, to_type, function ## Compute>  \
          <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,       \
                                       dst_stride, conversion);                \
    }                                                                          \
    else {                                                                     \
      cvtColor4VecKernel1<base_type, from_type, to_type, function ## Compute>  \
          <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,       \
                                       dst_stride, conversion);                \
    }                                                                          \
  }                                                                            \
  else if (vectors == 2) {                                                     \
    if (sizeof(base_type) == 1) {                                              \
      grid.x = divideUp(divideUp(cols, 2, 1), kBlockDimX0, kBlockShiftX0);     \
      grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);                     \
    }                                                                          \
    else {                                                                     \
      grid.x = divideUp(divideUp(cols, 2, 1), kBlockDimX1, kBlockShiftX1);     \
      grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);                     \
    }                                                                          \
                                                                               \
    padded_stride = roundUp(cols, 2, 1) * sizeof(to_type);                     \
    if (dst_stride >= padded_stride) {                                         \
      cvtColor2VecKernel0<base_type, from_type, to_type, function ## Compute>  \
          <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,       \
                                       dst_stride, conversion);                \
    }                                                                          \
    else {                                                                     \
      cvtColor2VecKernel1<base_type, from_type, to_type, function ## Compute>  \
          <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, dst,       \
                                       dst_stride, conversion);                \
    }                                                                          \
  }                                                                            \
  else {                                                                       \
    if (sizeof(base_type) == 1) {                                              \
      grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);                     \
      grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);                     \
    }                                                                          \
    else {                                                                     \
      grid.x = divideUp(cols, kBlockDimX1, kBlockShiftX1);                     \
      grid.y = divideUp(rows, kBlockDimY1, kBlockShiftY1);                     \
    }                                                                          \
                                                                               \
    cvtColor1VecKernel<base_type, from_type, to_type, function ## Compute><<<  \
        grid, block, 0, stream>>>(src, rows, cols, src_stride, dst, dst_stride,\
                                  conversion);                                 \
  }                                                                            \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<base_type>(cudaStream_t stream, int rows, int cols,           \
                            int src_stride, const base_type* src,              \
                            int dst_stride, base_type* dst) {                  \
  if (sizeof(base_type) == 4) {                                                \
    src_stride *= sizeof(float);                                               \
    dst_stride *= sizeof(float);                                               \
  }                                                                            \
                                                                               \
  RetCode code = function(src, rows, cols, src_stride, dst, dst_stride,        \
                          stream);                                             \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_1VECTORS_INVOCATION(function, from_type0, to_type0,          \
                                      from_type1, to_type1)                    \
CVT_COLOR_VECTORS_INVOCATION(function, uchar, from_type0, to_type0, 1)         \
CVT_COLOR_VECTORS_INVOCATION(function, float, from_type1, to_type1, 1)

#define CVT_COLOR_2VECTORS_INVOCATION(function, from_type0, to_type0,          \
                                      from_type1, to_type1)                    \
CVT_COLOR_VECTORS_INVOCATION(function, uchar, from_type0, to_type0, 2)         \
CVT_COLOR_VECTORS_INVOCATION(function, float, from_type1, to_type1, 1)

#define CVT_COLOR_4VECTORS_INVOCATION(function, from_type0, to_type0,          \
                                      from_type1, to_type1)                    \
CVT_COLOR_VECTORS_INVOCATION(function, uchar, from_type0, to_type0, 4)         \
CVT_COLOR_VECTORS_INVOCATION(function, float, from_type1, to_type1, 1)

#define CVT_COLOR_UCHAR_2VECTORS_INVOCATION(function, from_type, to_type)      \
CVT_COLOR_VECTORS_INVOCATION(function, uchar, from_type, to_type, 2)

/******************* NV12/21 conversion template ********************/

template <typename T, typename Conversion>
__global__
void cvtColorToNxKernel(const uchar* src, int rows, int cols, int src_stride,
                        uchar* dst, int dst_stride, Conversion conversion) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T* input = (T*)((uchar*)src + element_y * src_stride);
  uchar* output_y  = (uchar*)dst + element_y * dst_stride;
  uchar* output_uv = (uchar*)dst + (rows + (element_y >> 1)) * dst_stride;

  T value = input[element_x];
  uchar3 result;
  result = conversion(value, element_y, element_x);
  output_y[element_x] = result.x;
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
    output_uv[element_x]     = result.y;
    output_uv[element_x + 1] = result.z;
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_uv[element_x]     = result.y;
      output_uv[element_x + 1] = result.z;
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_uv[element_x]     = result.y;
      output_uv[element_x + 1] = result.z;
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_uv[element_x]     = result.y;
      output_uv[element_x + 1] = result.z;
    }
  }
}

template <typename T, typename Conversion>
__global__
void cvtColorToNxKernel(const uchar* src, int rows, int cols, int src_stride,
                        uchar* dst_y, int dst_y_stride, uchar* dst_uv,
                        int dst_uv_stride, Conversion conversion) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T* input = (T*)((uchar*)src + element_y * src_stride);
  uchar* output_y  = (uchar*)dst_y + element_y * dst_y_stride;
  uchar* output_uv = (uchar*)dst_uv + (element_y >> 1) * dst_uv_stride;

  T value = input[element_x];
  uchar3 result;
  result = conversion(value, element_y, element_x);
  output_y[element_x] = result.x;
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
    output_uv[element_x]     = result.y;
    output_uv[element_x + 1] = result.z;
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_uv[element_x]     = result.y;
      output_uv[element_x + 1] = result.z;
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_uv[element_x]     = result.y;
      output_uv[element_x + 1] = result.z;
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_uv[element_x]     = result.y;
      output_uv[element_x + 1] = result.z;
    }
  }
}

template <typename T, typename Conversion>
__global__
void cvtColorFromNxKernel(const uchar* src, int rows, int cols, int src_stride,
                          uchar* dst, int dst_stride, Conversion conversion) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar* input_y   = (uchar*)src + element_y * src_stride;
  uchar2* input_uv = (uchar2*)((uchar*)src +
                      (rows + (element_y >> 1)) * src_stride);
  T* output = (T*)((uchar*)dst + element_y * dst_stride);

  uchar value_y = input_y[element_x];
  uchar2 value_uv = input_uv[element_x >> 1];
  T result;
  result = conversion(value_y, value_uv);
  output[element_x] = result;

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_uv);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_uv);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_uv);
    output[element_x] = result;
  }
}

template <typename T, typename Conversion>
__global__
void cvtColorFromNxKernel(const uchar* src_y, int rows, int cols,
                          int src_y_stride, const uchar* src_uv,
                          int src_uv_stride, uchar* dst, int dst_stride,
                          Conversion conversion) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar* input_y   = (uchar*)src_y + element_y * src_y_stride;
  uchar2* input_uv = (uchar2*)((uchar*)src_uv +
                      (element_y >> 1) * src_uv_stride);
  T* output = (T*)((uchar*)dst + element_y * dst_stride);

  uchar value_y = input_y[element_x];
  uchar2 value_uv = input_uv[element_x >> 1];
  T result;
  result = conversion(value_y, value_uv);
  output[element_x] = result;

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_uv);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_uv);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_uv = input_uv[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_uv);
    output[element_x] = result;
  }
}

#define CVT_COLOR_TO_NVXX_INVOCATION(function, from_type)                      \
RetCode function(const uchar* src, int rows, int cols, int src_stride,         \
                 uchar* dst, int dst_stride, cudaStream_t stream) {            \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);                                  \
  PPL_ASSERT(src_stride >= cols * (int)sizeof(from_type));                     \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));                         \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorToNxKernel<from_type, function ## Compute><<<grid, block, 0, stream  \
      >>>(src, rows, cols, src_stride, dst, dst_stride, conversion);           \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_stride, const uchar* src, int dst_stride,      \
                        uchar* dst) {                                          \
  RetCode code = function(src, rows, cols, src_stride, dst, dst_stride,        \
                          stream);                                             \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_FROM_NVXX_INVOCATION(function, to_type)                      \
RetCode function(const uchar* src, int rows, int cols, int src_stride,         \
                 uchar* dst, int dst_stride, cudaStream_t stream) {            \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);                                  \
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));                         \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(to_type));                       \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorFromNxKernel<to_type, function ## Compute><<<grid, block, 0, stream  \
      >>>(src, rows, cols, src_stride, dst, dst_stride, conversion);           \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_stride, const uchar* src, int dst_stride,      \
                        uchar* dst) {                                          \
  RetCode code = function(src, rows, cols, src_stride, dst, dst_stride,        \
                          stream);                                             \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_TO_DISCRETE_NVXX_INVOCATION(function, from_type)             \
RetCode function(const uchar* src, int rows, int cols, int src_stride,         \
                 uchar* dst_y, int dst_y_stride, uchar* dst_uv,                \
                 int dst_uv_stride, cudaStream_t stream) {                     \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst_y != nullptr);                                                \
  PPL_ASSERT(dst_uv != nullptr);                                               \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);                                  \
  PPL_ASSERT(src_stride >= cols * (int)sizeof(from_type));                     \
  PPL_ASSERT(dst_y_stride >= cols * (int)sizeof(uchar));                       \
  PPL_ASSERT(dst_uv_stride >= cols * (int)sizeof(uchar));                      \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorToNxKernel<from_type, function ## Compute><<<grid, block, 0, stream  \
      >>>(src, rows, cols, src_stride, dst_y, dst_y_stride, dst_uv,            \
          dst_uv_stride, conversion);                                          \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_stride, const uchar* src, int dst_y_stride,    \
                        uchar* dst_y, int dst_uv_stride, uchar* dst_uv) {      \
  RetCode code = function(src, rows, cols, src_stride, dst_y, dst_y_stride,    \
                          dst_uv, dst_uv_stride, stream);                      \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_FROM_DISCRETE_NVXX_INVOCATION(function, to_type)             \
RetCode function(const uchar* src_y, int rows, int cols, int src_y_stride,     \
                 const uchar* src_uv, int src_uv_stride, uchar* dst,           \
                 int dst_stride, cudaStream_t stream) {                        \
  PPL_ASSERT(src_y != nullptr);                                                \
  PPL_ASSERT(src_uv != nullptr);                                               \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);                                  \
  PPL_ASSERT(src_y_stride >= cols * (int)sizeof(uchar));                       \
  PPL_ASSERT(src_uv_stride >= cols * (int)sizeof(uchar));                      \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(to_type));                       \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorFromNxKernel<to_type, function ## Compute><<<grid, block, 0, stream  \
      >>>(src_y, rows, cols, src_y_stride, src_uv, src_uv_stride, dst,         \
          dst_stride, conversion);                                             \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_y_stride, const uchar* src_y,                  \
                        int src_uv_stride, const uchar* src_uv, int dst_stride,\
                        uchar* dst) {                                          \
  RetCode code = function(src_y, rows, cols, src_y_stride, src_uv,             \
                          src_uv_stride, dst, dst_stride, stream);             \
                                                                               \
  return code;                                                                 \
}

/******************* I420 conversion template ********************/

template <typename T, typename Conversion>
__global__
void cvtColorToI420Kernel(const uchar* src, int rows, int cols, int src_stride,
                          uchar* dst, int dst_stride, Conversion conversion) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T* input = (T*)((uchar*)src + element_y * src_stride);
  uchar* output_y = (uchar*)dst + element_y * dst_stride;
  uchar* output_u = (uchar*)dst + (rows + (element_y >> 2)) * dst_stride;
  uchar* output_v = output_u + (rows >> 2) * dst_stride;

  int half_cols = cols >> 1;
  T value = input[element_x];
  uchar3 result;
  result = conversion(value, element_y, element_x);
  output_y[element_x] = result.x;
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
    if (((element_y >> 1) + 1) & 1) {
      output_u[element_x >> 1] = result.y;
      output_v[element_x >> 1] = result.z;
    } else {
      output_u[(element_x >> 1) + half_cols] = result.y;
      output_v[(element_x >> 1) + half_cols] = result.z;
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      if (((element_y >> 1) + 1) & 1) {
        output_u[element_x >> 1] = result.y;
        output_v[element_x >> 1] = result.z;
      } else {
        output_u[(element_x >> 1) + half_cols] = result.y;
        output_v[(element_x >> 1) + half_cols] = result.z;
      }
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      if (((element_y >> 1) + 1) & 1) {
        output_u[element_x >> 1] = result.y;
        output_v[element_x >> 1] = result.z;
      } else {
        output_u[(element_x >> 1) + half_cols] = result.y;
        output_v[(element_x >> 1) + half_cols] = result.z;
      }
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      if (((element_y >> 1) + 1) & 1) {
        output_u[element_x >> 1] = result.y;
        output_v[element_x >> 1] = result.z;
      } else {
        output_u[(element_x >> 1) + half_cols] = result.y;
        output_v[(element_x >> 1) + half_cols] = result.z;
      }
    }
  }
}

template <typename T, typename Conversion>
__global__
void cvtColorToI420Kernel(const uchar* src, int rows, int cols, int src_stride,
                          uchar* dst_y, int dst_y_stride, uchar* dst_u,
                          int dst_u_stride, uchar* dst_v, int dst_v_stride,
                          Conversion conversion) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T* input = (T*)((uchar*)src + element_y * src_stride);
  uchar* output_y = (uchar*)dst_y + element_y * dst_y_stride;
  uchar* output_u = (uchar*)dst_u + (element_y >> 1) * dst_u_stride;
  uchar* output_v = (uchar*)dst_v + (element_y >> 1) * dst_v_stride;

  T value = input[element_x];
  uchar3 result;
  result = conversion(value, element_y, element_x);
  output_y[element_x] = result.x;
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
    output_u[element_x >> 1] = result.y;
    output_v[element_x >> 1] = result.z;
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_u[element_x >> 1] = result.y;
      output_v[element_x >> 1] = result.z;
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_u[element_x >> 1] = result.y;
      output_v[element_x >> 1] = result.z;
    }
  }

  element_x += kBlockDimX0;
  value = input[element_x];
  if (element_x < cols) {
    result = conversion(value, element_y, element_x);
    output_y[element_x] = result.x;
    if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {
      output_u[element_x >> 1] = result.y;
      output_v[element_x >> 1] = result.z;
    }
  }
}

template <typename T, typename Conversion>
__global__
void cvtColorFromI420Kernel(const uchar* src, int rows, int cols,
                            int src_stride, uchar* dst, int dst_stride,
                            Conversion conversion) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar* input_y = (uchar*)src + element_y * src_stride;
  uchar* input_u = (uchar*)src + (rows + (element_y >> 2)) * src_stride;
  uchar* input_v = input_u + (rows >> 2) * src_stride;
  T* output = (T*)((uchar*)dst + element_y * dst_stride);

  T result;
  int uv_index = element_x >> 1;
  if ((element_y >> 1) & 1) {
    uv_index += (cols >> 1);
  }
  uchar value_y = input_y[element_x];
  uchar value_u = input_u[uv_index];
  uchar value_v = input_v[uv_index];
  result = conversion(value_y, value_u, value_v);
  output[element_x] = result;

  element_x += kBlockDimX0;
  if (element_x < cols) {
    uv_index = element_x >> 1;
    if ((element_y >> 1) & 1) {
      uv_index += (cols >> 1);
    }
    value_y = input_y[element_x];
    value_u = input_u[uv_index];
    value_v = input_v[uv_index];
    result = conversion(value_y, value_u, value_v);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  if (element_x < cols) {
    uv_index = element_x >> 1;
    if ((element_y >> 1) & 1) {
      uv_index += (cols >> 1);
    }
    value_y = input_y[element_x];
    value_u = input_u[uv_index];
    value_v = input_v[uv_index];
    result = conversion(value_y, value_u, value_v);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  if (element_x < cols) {
    uv_index = element_x >> 1;
    if ((element_y >> 1) & 1) {
      uv_index += (cols >> 1);
    }
    value_y = input_y[element_x];
    value_u = input_u[uv_index];
    value_v = input_v[uv_index];
    result = conversion(value_y, value_u, value_v);
    output[element_x] = result;
  }
}

template <typename T, typename Conversion>
__global__
void cvtColorFromI420Kernel(const uchar* src_y, int rows, int cols,
                            int src_y_stride, const uchar* src_u,
                            int src_u_stride, const uchar* src_v,
                            int src_v_stride, uchar* dst, int dst_stride,
                            Conversion conversion) {
  int element_x = (blockIdx.x << (kBlockShiftX0 + 2)) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  uchar* input_y = (uchar*)src_y + element_y * src_y_stride;
  uchar* input_u = (uchar*)((uchar*)src_u + (element_y >> 1) * src_u_stride);
  uchar* input_v = (uchar*)((uchar*)src_v + (element_y >> 1) * src_v_stride);
  T* output = (T*)((uchar*)dst + element_y * dst_stride);

  T result;
  uchar value_y = input_y[element_x];
  uchar value_u = input_u[element_x >> 1];
  uchar value_v = input_v[element_x >> 1];
  result = conversion(value_y, value_u, value_v);
  output[element_x] = result;

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_u = input_u[element_x >> 1];
  value_v = input_v[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_u, value_v);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_u = input_u[element_x >> 1];
  value_v = input_v[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_u, value_v);
    output[element_x] = result;
  }

  element_x += kBlockDimX0;
  value_y = input_y[element_x];
  value_u = input_u[element_x >> 1];
  value_v = input_v[element_x >> 1];
  if (element_x < cols) {
    result = conversion(value_y, value_u, value_v);
    output[element_x] = result;
  }
}

#define CVT_COLOR_TO_I420_INVOCATION(function, from_type)                      \
RetCode function(const uchar* src, int rows, int cols, int src_stride,         \
                 uchar* dst, int dst_stride, cudaStream_t stream) {            \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);                                  \
  PPL_ASSERT(src_stride >= cols * (int)sizeof(from_type));                     \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));                         \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorToI420Kernel<from_type, function ## Compute><<<grid, block, 0, stream\
      >>>(src, rows, cols, src_stride, dst, dst_stride, conversion);           \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_stride, const uchar* src, int dst_stride,      \
                        uchar* dst) {                                          \
  RetCode code = function(src, rows, cols, src_stride, dst, dst_stride,        \
                          stream);                                             \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_FROM_I420_INVOCATION(function, to_type)                      \
RetCode function(const uchar* src, int rows, int cols, int src_stride,         \
                 uchar* dst, int dst_stride, cudaStream_t stream) {            \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);                                  \
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));                         \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(to_type));                       \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorFromI420Kernel<to_type, function ## Compute><<<grid, block, 0, stream\
      >>>(src, rows, cols, src_stride, dst, dst_stride, conversion);           \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_stride, const uchar* src, int dst_stride,      \
                        uchar* dst) {                                          \
  RetCode code = function(src, rows, cols, src_stride, dst, dst_stride,        \
                          stream);                                             \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_TO_DISCRETE_I420_INVOCATION(function, from_type)             \
RetCode function(const uchar* src, int rows, int cols, int src_stride,         \
                 uchar* dst_y, int dst_y_stride, uchar* dst_u,                 \
                 int dst_u_stride, uchar* dst_v, int dst_v_stride,             \
                 cudaStream_t stream) {                                        \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst_y != nullptr);                                                \
  PPL_ASSERT(dst_u != nullptr);                                                \
  PPL_ASSERT(dst_v != nullptr);                                                \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);                                  \
  PPL_ASSERT(src_stride >= cols * (int)sizeof(from_type));                     \
  PPL_ASSERT(dst_y_stride >= cols * (int)sizeof(uchar));                       \
  PPL_ASSERT(dst_u_stride >= cols / 2 * (int)sizeof(uchar));                   \
  PPL_ASSERT(dst_v_stride >= cols / 2 * (int)sizeof(uchar));                   \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorToI420Kernel<from_type, function ## Compute><<<grid, block, 0, stream\
      >>>(src, rows, cols, src_stride, dst_y, dst_y_stride, dst_u,             \
          dst_u_stride, dst_v, dst_v_stride, conversion);                      \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_stride, const uchar* src, int dst_y_stride,    \
                        uchar* dst_y, int dst_u_stride, uchar* dst_u,          \
                        int dst_v_stride, uchar* dst_v) {                      \
  RetCode code = function(src, rows, cols, src_stride, dst_y, dst_y_stride,    \
                          dst_u, dst_u_stride, dst_v, dst_v_stride, stream);   \
                                                                               \
  return code;                                                                 \
}

#define CVT_COLOR_FROM_DISCRETE_I420_INVOCATION(function, to_type)             \
RetCode function(const uchar* src_y, int rows, int cols, int src_y_stride,     \
                 const uchar* src_u, int src_u_stride, const uchar* src_v,     \
                 int src_v_stride, uchar* dst, int dst_stride,                 \
                 cudaStream_t stream)  {                                       \
  PPL_ASSERT(src_y != nullptr);                                                \
  PPL_ASSERT(src_u != nullptr);                                                \
  PPL_ASSERT(src_v != nullptr);                                                \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT(rows % 1 == 0 && cols % 1 == 0);                                  \
  PPL_ASSERT(src_y_stride >= cols * (int)sizeof(uchar));                       \
  PPL_ASSERT(src_u_stride >= cols / 2 * (int)sizeof(uchar));                   \
  PPL_ASSERT(src_v_stride >= cols / 2 * (int)sizeof(uchar));                   \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(to_type));                       \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 4, 2), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorFromI420Kernel<to_type, function ## Compute><<<grid, block, 0, stream\
      >>>(src_y, rows, cols, src_y_stride, src_u, src_u_stride, src_v,         \
          src_v_stride, dst, dst_stride, conversion);                          \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_y_stride, const uchar* src_y, int src_u_stride,\
                        const uchar* src_u, int src_v_stride,                  \
                        const uchar* src_v, int dst_stride, uchar* dst) {      \
  RetCode code = function(src_y, rows, cols, src_y_stride, src_u, src_u_stride,\
                          src_v, src_v_stride, dst, dst_stride, stream);       \
                                                                               \
  return code;                                                                 \
}

/******************* UYVY/YUYV conversion template ********************/

template <typename T0, typename T1, typename Conversion>
__global__
void cvtColorYuyvKernel0(const uchar* src, int rows, int cols, int src_stride,
                         uchar* dst, int dst_stride, Conversion conversion) {
  int index_x   = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_x = index_x << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  T0* input  = (T0*)((uchar*)src + element_y * src_stride);
  T1* output = (T1*)((uchar*)dst + element_y * dst_stride);

  T0 in_value0, in_value1;
  in_value0 = input[element_x];
  in_value1 = input[element_x + 1];

  T1 result = conversion(in_value0, in_value1);
  output[index_x] = result;
}

template <typename T0, typename T1, typename Conversion0, typename Conversion1>
__global__
void cvtColorYuyvKernel1(const uchar* src, int rows, int cols, int src_stride,
                         uchar* dst, int dst_stride, Conversion0 conversion0,
                         Conversion1 conversion1) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  T0* input  = (T0*)((uchar*)src + element_y * src_stride);
  T1* output = (T1*)((uchar*)dst + element_y * dst_stride);

  T0 value = input[element_x];
  T1 result0, result1;
  result0 = conversion0(value);
  result1 = conversion1(value);

  output[index_x]     = result0;
  output[index_x + 1] = result1;
}

#define CVT_COLOR_FROM_YUV422_UCHAR_INVOCATION(function, from_type, to_type)   \
RetCode function(const uchar* src, int rows, int cols, int src_stride,         \
                 uchar* dst, int dst_stride, cudaStream_t stream) {            \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((cols & 0x1) == 0);                                               \
  PPL_ASSERT(src_stride >= cols / 2 * (int)sizeof(from_type));                 \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(to_type));                       \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 2, 1), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute0 conversion0;                                            \
  function ## Compute1 conversion1;                                            \
  cvtColorYuyvKernel1<from_type, to_type, function ## Compute0,                \
      function ## Compute1><<<grid, block, 0, stream>>>(src, rows, cols,       \
      src_stride, dst, dst_stride, conversion0, conversion1);                  \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_stride, const uchar* src, int dst_stride,      \
                        uchar* dst) {                                          \
  RetCode code = function(src, rows, cols, src_stride, dst, dst_stride,        \
                          stream);                                             \
  return code;                                                                 \
}

#define CVT_COLOR_YUV422_TO_GRAY_UCHAR_INVOCATION(function, type)              \
RetCode function(const uchar* src, int rows, int cols, int src_stride,         \
                 uchar* dst, int dst_stride, cudaStream_t stream) {            \
  PPL_ASSERT(src != nullptr);                                                  \
  PPL_ASSERT(dst != nullptr);                                                  \
  PPL_ASSERT(rows >= 1 && cols >= 1);                                          \
  PPL_ASSERT((cols & 0x1) == 0);                                               \
  PPL_ASSERT(src_stride >= cols * 2 * (int)sizeof(uchar));                     \
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));                         \
                                                                               \
  dim3 block, grid;                                                            \
  block.x = kBlockDimX0;                                                       \
  block.y = kBlockDimY0;                                                       \
  grid.x  = divideUp(divideUp(cols, 2, 1), kBlockDimX0, kBlockShiftX0);        \
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);                        \
                                                                               \
  function ## Compute conversion;                                              \
  cvtColorYuyvKernel0<type, type, function ## Compute><<<grid, block, 0, stream\
      >>>(src, rows, cols, src_stride, dst, dst_stride, conversion);           \
                                                                               \
  cudaError_t code = cudaGetLastError();                                       \
  if (code != cudaSuccess) {                                                   \
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);                  \
    return RC_DEVICE_RUNTIME_ERROR;                                            \
  }                                                                            \
                                                                               \
  return RC_SUCCESS;                                                           \
}                                                                              \
                                                                               \
template <>                                                                    \
RetCode function<uchar>(cudaStream_t stream, int rows, int cols,               \
                        int src_stride, const uchar* src, int dst_stride,      \
                        uchar* dst) {                                          \
  RetCode code = function(src, rows, cols, src_stride, dst, dst_stride,        \
                          stream);                                             \
                                                                               \
  return code;                                                                 \
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_CVTCOLOR_MEMORY_HPP_
