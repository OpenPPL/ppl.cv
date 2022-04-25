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

#include "ppl/cv/cuda/laplacian.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

#define RADIUS 8

template <typename Tsrc, typename Tdst, typename BorderInterpolation>
__global__
void laplacianC1SharedKernel(const Tsrc* src, int rows, int cols,
                             int src_stride, int ksize, Tdst* dst,
                             int dst_stride, float scale, float delta,
                             BorderInterpolation interpolation) {
  __shared__ Tsrc data[kDimY0 + RADIUS * 2][(kDimX0 << 2) + RADIUS * 2];
  __shared__ float kernel[25];

  int element_x = ((blockIdx.x << kShiftX0) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  int index, y_index, row_index, col_index;
  int radius = ksize >> 1;
  if (ksize == 1) {
    radius = 1;
  }
  Tsrc* input;
  Tsrc value0, value1, value2, value3;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (ksize == 1) {
      kernel[0] = 0;
      kernel[1] = 1;
      kernel[2] = 0;
      kernel[3] = 1;
      kernel[4] = -4;
      kernel[5] = 1;
      kernel[6] = 0;
      kernel[7] = 1;
      kernel[8] = 0;
    }
    else if (ksize == 3) {
      kernel[0] = 2;
      kernel[1] = 0;
      kernel[2] = 2;
      kernel[3] = 0;
      kernel[4] = -8;
      kernel[5] = 0;
      kernel[6] = 2;
      kernel[7] = 0;
      kernel[8] = 2;
    }
    else if (ksize == 5) {
      kernel[0] = 2;
      kernel[1] = 4;
      kernel[2] = 4;
      kernel[3] = 4;
      kernel[4] = 2;
      kernel[5] = 4;
      kernel[6] = 0;
      kernel[7] = -8;
      kernel[8] = 0;
      kernel[9] = 4;
      kernel[10] = 4;
      kernel[11] = -8;
      kernel[12] = -24;
      kernel[13] = -8;
      kernel[14] = 4;
      kernel[15] = 4;
      kernel[16] = 0;
      kernel[17] = -8;
      kernel[18] = 0;
      kernel[19] = 4;
      kernel[20] = 2;
      kernel[21] = 4;
      kernel[22] = 4;
      kernel[23] = 4;
      kernel[24] = 2;
    }
    else {
    }
  }

  y_index   = threadIdx.y;
  row_index = element_y - radius;
  while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
         row_index < rows + radius) {
    index = interpolation(rows, radius, row_index);
    input = (Tsrc*)((uchar*)src + index * src_stride);

    if (threadIdx.x < radius) {
      if (blockIdx.x == 0) {
        index = interpolation(cols, radius, threadIdx.x - radius);
      }
      else {
        index = (blockIdx.x << (kShiftX0 + 2)) + threadIdx.x - radius;
      }
      value0 = input[index];
      data[y_index][threadIdx.x] = value0;
    }

    if (element_x < cols) {
      value0 = input[element_x];
      value1 = input[element_x + 1];
      value2 = input[element_x + 2];
      value3 = input[element_x + 3];
      index = radius + (threadIdx.x << 2);
      data[y_index][index] = value0;
      data[y_index][index + 1] = value1;
      data[y_index][index + 2] = value2;
      data[y_index][index + 3] = value3;
    }

    if (threadIdx.x < radius) {
      index = (cols - radius) >> (kShiftX0 + 2);
      if (blockIdx.x >= index) {
        if (blockIdx.x != gridDim.x - 1) {
          index = ((blockIdx.x + 1) << (kShiftX0 + 2)) + threadIdx.x;
          index = interpolation(cols, radius, index);
          value0 = input[index];
          index = radius + (kDimX0 << 2) + threadIdx.x;
          data[y_index][index] = value0;
        }
        else {
          index = interpolation(cols, radius, cols + threadIdx.x);
          value0 = input[index];
          index = cols - (blockIdx.x << (kShiftX0 + 2));
          index += (radius + threadIdx.x);
          data[y_index][index] = value0;
        }
      }
      else {
        index = ((blockIdx.x + 1) << (kShiftX0 + 2)) + threadIdx.x;
        value0 = input[index];
        index = radius + (kDimX0 << 2) + threadIdx.x;
        data[y_index][index] = value0;
      }
    }

    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  y_index = threadIdx.y;
  index = 0;
  if (ksize == 1) ksize = 3;
  for (row_index = 0; row_index < ksize; row_index++) {
    for (col_index = 0; col_index < ksize; col_index++) {
      mulAdd(sum.x, data[y_index][(threadIdx.x << 2) + col_index],
             kernel[index]);
      mulAdd(sum.y, data[y_index][(threadIdx.x << 2) + col_index + 1],
             kernel[index]);
      mulAdd(sum.z, data[y_index][(threadIdx.x << 2) + col_index + 2],
             kernel[index]);
      mulAdd(sum.w, data[y_index][(threadIdx.x << 2) + col_index + 3],
             kernel[index]);
      index++;
    }
    y_index++;
  }

  if (scale != 1.f) {
    sum.x *= scale;
    sum.y *= scale;
    sum.z *= scale;
    sum.w *= scale;
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

template <typename Tsrc, typename Tsrcn, typename Tdst, typename Tdstn,
          typename BorderInterpolation>
__global__
void laplacianCnSharedKernel0(const Tsrc* src, int rows, int cols,
                              int src_stride, int ksize, Tdst* dst,
                              int dst_stride, float scale, float delta,
                              BorderInterpolation interpolation) {
  __shared__ Tsrcn data[kDimY0 + RADIUS * 2][kDimX0 + RADIUS * 2];
  __shared__ float kernel[25];

  int element_x = (blockIdx.x << kShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kShiftY0) + threadIdx.y;

  int index, y_index, row_index, col_index;
  int radius = ksize >> 1;
  if (ksize == 1) {
    radius = 1;
  }
  Tsrcn* input;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (ksize == 1) {
      kernel[0] = 0;
      kernel[1] = 1;
      kernel[2] = 0;
      kernel[3] = 1;
      kernel[4] = -4;
      kernel[5] = 1;
      kernel[6] = 0;
      kernel[7] = 1;
      kernel[8] = 0;
    }
    else if (ksize == 3) {
      kernel[0] = 2;
      kernel[1] = 0;
      kernel[2] = 2;
      kernel[3] = 0;
      kernel[4] = -8;
      kernel[5] = 0;
      kernel[6] = 2;
      kernel[7] = 0;
      kernel[8] = 2;
    }
    else if (ksize == 5) {
      kernel[0] = 2;
      kernel[1] = 4;
      kernel[2] = 4;
      kernel[3] = 4;
      kernel[4] = 2;
      kernel[5] = 4;
      kernel[6] = 0;
      kernel[7] = -8;
      kernel[8] = 0;
      kernel[9] = 4;
      kernel[10] = 4;
      kernel[11] = -8;
      kernel[12] = -24;
      kernel[13] = -8;
      kernel[14] = 4;
      kernel[15] = 4;
      kernel[16] = 0;
      kernel[17] = -8;
      kernel[18] = 0;
      kernel[19] = 4;
      kernel[20] = 2;
      kernel[21] = 4;
      kernel[22] = 4;
      kernel[23] = 4;
      kernel[24] = 2;
    }
    else {
    }
  }

  y_index   = threadIdx.y;
  row_index = element_y - radius;
  while (row_index < (int)(((blockIdx.y + 1) << kShiftY0) + radius) &&
         row_index < rows + radius) {
    index = interpolation(rows, radius, row_index);
    input = (Tsrcn*)((uchar*)src + index * src_stride);

    int x_index = threadIdx.x;
    col_index = element_x - radius;
    while (col_index < (int)(((blockIdx.x + 1) << kShiftX0) + radius) &&
           col_index < cols + radius) {
      index = interpolation(cols, radius, col_index);
      data[y_index][x_index] = input[index];
      x_index   += kDimX0;
      col_index += kDimX0;
    }

    y_index   += kDimY0;
    row_index += kDimY0;
  }
  __syncthreads();

  if (element_x >= cols || element_y >= rows) {
    return;
  }

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  y_index = threadIdx.y;
  index = 0;
  if (ksize == 1) ksize = 3;
  for (row_index = 0; row_index < ksize; row_index++) {
    for (col_index = 0; col_index < ksize; col_index++) {
      mulAdd(sum, data[y_index][threadIdx.x + col_index], kernel[index++]);
    }
    y_index++;
  }

  if (scale != 1.f) {
    sum.x *= scale;
    sum.y *= scale;
    sum.z *= scale;
    sum.w *= scale;
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

#define RUN_CHANNEL1_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
laplacianC1SharedKernel<Tsrc, Tdst, Interpolation><<<grid, block, 0, stream>>>(\
    src, rows, cols, src_stride, ksize, dst, dst_stride, scale, delta,         \
    interpolation);

#define RUN_CHANNELN_SMALL_KERNELS(Tsrc, Tdst, Interpolation)                  \
Interpolation interpolation;                                                   \
if (channels == 3) {                                                           \
  laplacianCnSharedKernel0<Tsrc, Tsrc ## 3, Tdst, Tdst ## 3, Interpolation>    \
      <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, ksize, dst,    \
      dst_stride, scale, delta, interpolation);                                \
}                                                                              \
else {                                                                         \
  laplacianCnSharedKernel0<Tsrc, Tsrc ## 4, Tdst, Tdst ## 4, Interpolation>    \
      <<<grid, block, 0, stream>>>(src, rows, cols, src_stride, ksize, dst,    \
      dst_stride, scale, delta, interpolation);                                \
}

RetCode laplacian(const uchar* src, int rows, int cols, int channels,
                  int src_stride, int ksize, uchar* dst, int dst_stride,
                  float scale, float delta, BorderType border_type,
                  cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(ksize ==1 || ksize == 3 || ksize == 5);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize <= 5 && channels == 1) {
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

RetCode laplacian(const uchar* src, int rows, int cols, int channels,
                  int src_stride, int ksize, short* dst, int dst_stride,
                  float scale, float delta, BorderType border_type,
                  cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(short));
  PPL_ASSERT(ksize ==1 || ksize == 3 || ksize == 5);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize <= 5 && channels == 1) {
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

RetCode laplacian(const float* src, int rows, int cols, int channels,
                  int src_stride, int ksize, float* dst, int dst_stride,
                  float scale, float delta, BorderType border_type,
                  cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(ksize ==1 || ksize == 3 || ksize == 5);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

  cudaError_t code;
  if (ksize <= 5 && channels == 1) {
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

template <>
RetCode Laplacian<uchar, uchar, 1>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  RetCode code = laplacian(inData, height, width, 1, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

template <>
RetCode Laplacian<uchar, uchar, 3>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  RetCode code = laplacian(inData, height, width, 3, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

template <>
RetCode Laplacian<uchar, uchar, 4>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   uchar* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  RetCode code = laplacian(inData, height, width, 4, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

template <>
RetCode Laplacian<uchar, short, 1>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   short* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = laplacian(inData, height, width, 1, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

template <>
RetCode Laplacian<uchar, short, 3>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   short* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = laplacian(inData, height, width, 3, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

template <>
RetCode Laplacian<uchar, short, 4>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outWidthStride,
                                   short* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  outWidthStride *= sizeof(short);
  RetCode code = laplacian(inData, height, width, 4, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

template <>
RetCode Laplacian<float, float, 1>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   float* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = laplacian(inData, height, width, 1, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

template <>
RetCode Laplacian<float, float, 3>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   float* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = laplacian(inData, height, width, 3, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

template <>
RetCode Laplacian<float, float, 4>(cudaStream_t stream,
                                   int height,
                                   int width,
                                   int inWidthStride,
                                   const float* inData,
                                   int outWidthStride,
                                   float* outData,
                                   int ksize,
                                   float scale,
                                   float delta,
                                   BorderType border_type) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = laplacian(inData, height, width, 4, inWidthStride, ksize,
                           outData, outWidthStride, scale, delta, border_type,
                           stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
