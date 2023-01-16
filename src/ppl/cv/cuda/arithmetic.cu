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

#include "ppl/cv/cuda/arithmetic.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/******************************* add operation *******************************/

__global__
void addKernel0(const uchar* src0, int rows, int cols, int src0_stride,
                const uchar* src1, int src1_stride, uchar* dst,
                int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const uchar4* input0 = (uchar4*)(src0 + element_y * src0_stride);
  const uchar4* input1 = (uchar4*)(src1 + element_y * src1_stride);
  uchar4 input_value0 = input0[element_x];
  uchar4 input_value1 = input1[element_x];

  uchar4 output_value;
  output_value.x = saturateCast((int)input_value0.x + input_value1.x);
  output_value.y = saturateCast((int)input_value0.y + input_value1.y);
  output_value.z = saturateCast((int)input_value0.z + input_value1.z);
  output_value.w = saturateCast((int)input_value0.w + input_value1.w);

  uchar4* output = (uchar4*)(dst + element_y * dst_stride);
  output[element_x] = output_value;
}

__global__
void addKernel10(const uchar* src0, int cols, const uchar* src1, uchar* dst) {
  int element_x = (blockIdx.x << 8) + threadIdx.x;
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  const uchar4* input0 = (uchar4*)src0;
  const uchar4* input1 = (uchar4*)src1;

  uchar4 input_value0, input_value1, output_value;
  input_value0 = input0[element_x];
  input_value1 = input1[element_x];

  if (index_x < cols - 3) {
    output_value.x = saturateCast((int)input_value0.x + input_value1.x);
    output_value.y = saturateCast((int)input_value0.y + input_value1.y);
    output_value.z = saturateCast((int)input_value0.z + input_value1.z);
    output_value.w = saturateCast((int)input_value0.w + input_value1.w);

    uchar4* output = (uchar4*)dst;
    output[element_x] = output_value;
  }
  else {
    output_value.x = saturateCast((int)input_value0.x + input_value1.x);
    if (index_x < cols - 1) {
      output_value.y = saturateCast((int)input_value0.y + input_value1.y);
    }
    if (index_x < cols - 2) {
      output_value.z = saturateCast((int)input_value0.z + input_value1.z);
    }

    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if (index_x < cols - 2) {
      dst[index_x + 2] = output_value.z;
    }
  }
}

__global__
void addKernel11(const uchar* src0, int rows, int cols, int src0_stride,
                 const uchar* src1, int src1_stride, uchar* dst,
                 int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  const uchar* input0 = src0 + element_y * src0_stride;
  const uchar* input1 = src1 + element_y * src1_stride;
  uchar* output = dst + element_y * dst_stride;

  uchar input_value00, input_value01, input_value02, input_value03;
  uchar input_value10, input_value11, input_value12, input_value13;
  int output_value0, output_value1, output_value2, output_value3;

  if (blockIdx.x < gridDim.x - 1) {
    input_value00 = input0[index_x];
    input_value01 = input0[index_x + 1];
    input_value02 = input0[index_x + 2];
    input_value03 = input0[index_x + 3];

    input_value10 = input1[index_x];
    input_value11 = input1[index_x + 1];
    input_value12 = input1[index_x + 2];
    input_value13 = input1[index_x + 3];

    output_value0 = input_value00 + input_value10;
    output_value1 = input_value01 + input_value11;
    output_value2 = input_value02 + input_value12;
    output_value3 = input_value03 + input_value13;

    output[index_x]     = saturateCast(output_value0);
    output[index_x + 1] = saturateCast(output_value1);
    output[index_x + 2] = saturateCast(output_value2);
    output[index_x + 3] = saturateCast(output_value3);
  }
  else {
    input_value00 = input0[index_x];
    if (index_x < cols - 1) {
      input_value01 = input0[index_x + 1];
    }
    if (index_x < cols - 2) {
      input_value02 = input0[index_x + 2];
    }

    input_value10 = input1[index_x];
    if (index_x < cols - 1) {
      input_value11 = input1[index_x + 1];
    }
    if (index_x < cols - 2) {
      input_value12 = input1[index_x + 2];
    }

    output_value0 = input_value00 + input_value10;
    if (index_x < cols - 1) {
      output_value1 = input_value01 + input_value11;
    }
    if (index_x < cols - 2) {
      output_value2 = input_value02 + input_value12;
    }

    output[index_x] = saturateCast(output_value0);
    if (index_x < cols - 1) {
      output[index_x + 1] = saturateCast(output_value1);
    }
    if (index_x < cols - 2) {
      output[index_x + 2] = saturateCast(output_value2);
    }
  }
}

__global__
void addKernel0(const float* src0, int rows, int cols, int src0_stride,
                const float* src1, int src1_stride, float* dst,
                int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float2* input0 = (float2*)((uchar*)src0 + element_y * src0_stride);
  const float2* input1 = (float2*)((uchar*)src1 + element_y * src1_stride);
  float2 input_value0 = input0[element_x];
  float2 input_value1 = input1[element_x];

  float2 output_value;
  output_value.x = input_value0.x + input_value1.x;
  output_value.y = input_value0.y + input_value1.y;

  float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = output_value;
}

__global__
void addKernel1(const float* src0, int rows, int cols, int src0_stride,
                const float* src1, int src1_stride, float* dst,
                int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float* input0 = (float*)((uchar*)src0 + element_y * src0_stride);
  const float* input1 = (float*)((uchar*)src1 + element_y * src1_stride);
  float* output  = (float*)((uchar*)dst + element_y * dst_stride);

  float input_value00, input_value01;
  float input_value10, input_value11;
  float output_value0, output_value1;

  if (blockIdx.x < gridDim.x - 1) {
    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    output_value0 = input_value00 + input_value10;
    output_value1 = input_value01 + input_value11;

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    output_value0 = input_value00 + input_value10;
    if (element_x != cols - 1) {
      output_value1 = input_value01 + input_value11;
    }

    output[element_x] = output_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = output_value1;
    }
  }
}

RetCode add(const uchar* src0, int rows, int cols, int channels,
            int src0_stride, const uchar* src1, int src1_stride, uchar* dst,
            int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(uchar));

  int columns = cols * channels;
  cols = divideUp(columns, 4, 2);
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x = divideUp(cols, kBlockDimX0, kBlockShiftX0);
  grid.y = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 3) == 0 && (src1_stride & 3) == 0 &&
      (dst_stride & 3) == 0) {
    addKernel0<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride, src1,
                                           src1_stride, dst, dst_stride);
  }
  else if (src0_stride == columns && src1_stride == columns &&
           dst_stride == columns) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    block.x = 256;
    block.y = 1;
    grid.x = divideUp(cols, 256, 8);
    grid.y = 1;
    addKernel10<<<grid, block, 0, stream>>>(src0, columns, src1, dst);
  }
  else {
    addKernel11<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride, src1,
                                            src1_stride, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode add(const float* src0, int rows, int cols, int channels,
            int src0_stride, const float* src1, int src1_stride, float* dst,
            int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 7) == 0 && (src1_stride & 7) == 0 &&
      (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    addKernel0<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride,
                                           src1, src1_stride, dst, dst_stride);
  }
  else {
    addKernel1<<<grid, block, 0, stream>>>(src0, rows, columns, src0_stride,
                                           src1, src1_stride, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Add<uchar, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData) {
  RetCode code = add(inData0, height, width, 1, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Add<uchar, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData) {
  RetCode code = add(inData0, height, width, 3, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Add<uchar, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData) {
  RetCode code = add(inData0, height, width, 4, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Add<float, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = add(inData0, height, width, 1, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Add<float, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = add(inData0, height, width, 3, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Add<float, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = add(inData0, height, width, 4, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

/*************************** addWeighted operation ***************************/

template <typename T0, typename T1>
__global__
void addWeightedKernel0(const T0* src0, int rows, int cols, int src0_stride,
                        float alpha, const T0* src1, int src1_stride,
                        float beta, float gamma, T0* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const T1* input0 = (T1*)((uchar*)src0 + element_y * src0_stride);
  const T1* input1 = (T1*)((uchar*)src1 + element_y * src1_stride);
  T1* output = (T1*)((uchar*)dst + element_y * dst_stride);

  T1 input_value0 = input0[element_x];
  T1 input_value1 = input1[element_x];
  float2 output_value0 = make_float2(gamma, gamma);

  output_value0.x += input_value0.x * alpha;
  output_value0.y += input_value0.y * alpha;

  output_value0.x += input_value1.x * beta;
  output_value0.y += input_value1.y * beta;

  output[element_x] = saturateCastVector<T1, float2>(output_value0);
}

template <typename T>
__global__
void addWeightedKernel1(const T* src0, int rows, int cols, int src0_stride,
                        float alpha, const T* src1, int src1_stride,
                        float beta, float gamma, T* dst, int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const T* input0 = (T*)((uchar*)src0 + element_y * src0_stride);
  const T* input1 = (T*)((uchar*)src1 + element_y * src1_stride);
  T* output = (T*)((uchar*)dst + element_y * dst_stride);

  T input_value00, input_value01;
  T input_value10, input_value11;
  float output_value0 = gamma;
  float output_value1 = gamma;

  if (blockIdx.x < gridDim.x - 1) {
    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    output_value0 += input_value00 * alpha;
    output_value1 += input_value01 * alpha;

    output_value0 += input_value10 * beta;
    output_value1 += input_value11 * beta;

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    output_value0 += input_value00 * alpha;
    output_value0 += input_value10 * beta;

    if (element_x != cols - 1) {
      output_value1 += input_value01 * alpha;
      output_value1 += input_value11 * beta;
    }

    if (sizeof(T) == 1) {
      output[element_x] = saturateCast(output_value0);
      if (element_x != cols - 1) {
        output[element_x + 1] = saturateCast(output_value1);
      }
    }
    else {
      output[element_x] = output_value0;
      if (element_x != cols - 1) {
        output[element_x + 1] = output_value1;
      }
    }
  }
}

RetCode addWeighted(const uchar* src0, int rows, int cols, int channels,
                    int src0_stride, float alpha, const uchar* src1,
                    int src1_stride, float beta, float gamma, uchar* dst,
                    int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(uchar));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 1) == 0 && (src1_stride & 1) == 0 &&
      (dst_stride & 1) == 0) {
    cols = divideUp(columns, 2, 1);
    addWeightedKernel0<uchar, uchar2><<<grid, block, 0, stream>>>(src0, rows,
        cols, src0_stride, alpha, src1, src1_stride, beta, gamma, dst,
        dst_stride);
  }
  else {
    addWeightedKernel1<uchar><<<grid, block, 0, stream>>>(src0, rows, columns,
        src0_stride, alpha, src1, src1_stride, beta, gamma, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode addWeighted(const float* src0, int rows, int cols, int channels,
                    int src0_stride, float alpha, const float* src1,
                    int src1_stride, float beta, float gamma, float* dst,
                    int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 7) == 0 && (src1_stride & 7) == 0 &&
      (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    addWeightedKernel0<float, float2><<<grid, block, 0, stream>>>(src0, rows,
        cols, src0_stride, alpha, src1, src1_stride, beta, gamma, dst,
        dst_stride);
  }
  else {
    addWeightedKernel1<float><<<grid, block, 0, stream>>>(src0, rows, columns,
        src0_stride, alpha, src1, src1_stride, beta, gamma, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode AddWeighted<uchar, 1>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride0,
                              const uchar* inData0,
                              float alpha,
                              int inWidthStride1,
                              const uchar* inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              uchar* outData) {
  RetCode code = addWeighted(inData0, height, width, 1, inWidthStride0, alpha,
                             inData1, inWidthStride1, beta, gamma, outData,
                             outWidthStride, stream);

  return code;
}

template <>
RetCode AddWeighted<uchar, 3>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride0,
                              const uchar* inData0,
                              float alpha,
                              int inWidthStride1,
                              const uchar* inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              uchar* outData) {
  RetCode code = addWeighted(inData0, height, width, 3, inWidthStride0, alpha,
                             inData1, inWidthStride1, beta, gamma, outData,
                             outWidthStride, stream);

  return code;
}

template <>
RetCode AddWeighted<uchar, 4>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride0,
                              const uchar* inData0,
                              float alpha,
                              int inWidthStride1,
                              const uchar* inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              uchar* outData) {
  RetCode code = addWeighted(inData0, height, width, 4, inWidthStride0, alpha,
                             inData1, inWidthStride1, beta, gamma, outData,
                             outWidthStride, stream);

  return code;
}

template <>
RetCode AddWeighted<float, 1>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride0,
                              const float* inData0,
                              float alpha,
                              int inWidthStride1,
                              const float* inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addWeighted(inData0, height, width, 1, inWidthStride0, alpha,
                             inData1, inWidthStride1, beta, gamma, outData,
                             outWidthStride, stream);

  return code;
}

template <>
RetCode AddWeighted<float, 3>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride0,
                              const float* inData0,
                              float alpha,
                              int inWidthStride1,
                              const float* inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addWeighted(inData0, height, width, 3, inWidthStride0, alpha,
                             inData1, inWidthStride1, beta, gamma, outData,
                             outWidthStride, stream);

  return code;
}

template <>
RetCode AddWeighted<float, 4>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride0,
                              const float* inData0,
                              float alpha,
                              int inWidthStride1,
                              const float* inData1,
                              float beta,
                              float gamma,
                              int outWidthStride,
                              float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = addWeighted(inData0, height, width, 4, inWidthStride0, alpha,
                             inData1, inWidthStride1, beta, gamma, outData,
                             outWidthStride, stream);

  return code;
}

/**************************** subtract operation *****************************/

template <typename T0, typename T1>
__global__
void subtractKernel0(const T0* src, int rows, int cols, int channels,
                     int src_stride, T0 scalar0, T0 scalar1, T0 scalar2,
                     T0 scalar3, T0* dst, int dst_stride) {
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

  const T1* input = (T1*)((uchar*)src + element_y * src_stride);
  T1* output = (T1*)((uchar*)dst + element_y * dst_stride);

  T1 input_value = input[element_x];
  T0 value0, value1, value2, value3;
  T1 output_value;

  if (channels == 1) {
    value0 = scalar0;
    value1 = scalar0;
    value2 = scalar0;
    value3 = scalar0;
  }
  else if (channels == 3) {
    int value = (element_x << 2) % 3;
    if (value == 0) {
      value0 = scalar0;
      value1 = scalar1;
      value2 = scalar2;
      value3 = scalar0;
    }
    else if (value == 1) {
      value0 = scalar1;
      value1 = scalar2;
      value2 = scalar0;
      value3 = scalar1;
    }
    else {
      value0 = scalar2;
      value1 = scalar0;
      value2 = scalar1;
      value3 = scalar2;
    }
  }
  else {  // channel === 4
    value0 = scalar0;
    value1 = scalar1;
    value2 = scalar2;
    value3 = scalar3;
  }

  if (sizeof(T0) == 1) {
    output_value.x = saturateCast(input_value.x - value0);
    output_value.y = saturateCast(input_value.y - value1);
    output_value.z = saturateCast(input_value.z - value2);
    output_value.w = saturateCast(input_value.w - value3);
  }
  else {
    output_value.x = input_value.x - value0;
    output_value.y = input_value.y - value1;
    output_value.z = input_value.z - value2;
    output_value.w = input_value.w - value3;
  }

  output[element_x] = output_value;
}

template <typename T0, typename T1>
__global__
void subtractKernel10(const T0* src, int cols, int channels, T0 scalar0,
                      T0 scalar1, T0 scalar2, T0 scalar3, T0* dst) {
  int element_x = (blockIdx.x << 8) + threadIdx.x;
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  const T1* input = (T1*)((uchar*)src);
  T1 input_value = input[element_x];
  T0 value0, value1, value2, value3;
  T1 output_value;

  if (channels == 1) {
    value0 = scalar0;
    value1 = scalar0;
    value2 = scalar0;
    value3 = scalar0;
  }
  else if (channels == 3) {
    int value = (element_x << 2) % 3;
    if (value == 0) {
      value0 = scalar0;
      value1 = scalar1;
      value2 = scalar2;
      value3 = scalar0;
    }
    else if (value == 1) {
      value0 = scalar1;
      value1 = scalar2;
      value2 = scalar0;
      value3 = scalar1;
    }
    else {
      value0 = scalar2;
      value1 = scalar0;
      value2 = scalar1;
      value3 = scalar2;
    }
  }
  else {  // channel === 4
    value0 = scalar0;
    value1 = scalar1;
    value2 = scalar2;
    value3 = scalar3;
  }

  if (index_x < cols - 3) {
    if (sizeof(T0) == 1) {
      output_value.x = saturateCast(input_value.x - value0);
      output_value.y = saturateCast(input_value.y - value1);
      output_value.z = saturateCast(input_value.z - value2);
      output_value.w = saturateCast(input_value.w - value3);
    }
    else {
      output_value.x = input_value.x - value0;
      output_value.y = input_value.y - value1;
      output_value.z = input_value.z - value2;
      output_value.w = input_value.w - value3;
    }

    T1* output = (T1*)((uchar*)dst);
    output[element_x] = output_value;
  }
  else {
    if (sizeof(T0) == 1) {
      output_value.x = saturateCast(input_value.x - value0);
      if (index_x < cols - 1) {
        output_value.y = saturateCast(input_value.y - value1);
      }
      if (index_x < cols - 2) {
        output_value.z = saturateCast(input_value.z - value2);
      }
    }
    else {
      output_value.x = input_value.x - value0;
      if (index_x < cols - 1) {
        output_value.y = input_value.y - value1;
      }
      if (index_x < cols - 2) {
        output_value.z = input_value.z - value2;
      }
    }

    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if (index_x < cols - 2) {
      dst[index_x + 2] = output_value.z;
    }
  }
}

template <typename T>
__global__
void subtractKernel11(const T* src, int rows, int cols, int channels,
                      int src_stride, T scalar0, T scalar1, T scalar2,
                      T scalar3, T* dst, int dst_stride) {
  int element_x, element_y;
  if (sizeof(T) == 1) {
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

  const T* input = (T*)((uchar*)src + element_y * src_stride);
  T* output = (T*)((uchar*)dst + element_y * dst_stride);

  T input_value0, input_value1;
  T value0, value1;
  T output_value0, output_value1;

  if (channels == 1) {
    value0 = scalar0;
    value1 = scalar0;
  }
  else if (channels == 3) {
    int value = element_x % 3;
    if (value == 0) {
      value0 = scalar0;
      value1 = scalar1;
    }
    else if (value == 1) {
      value0 = scalar1;
      value1 = scalar2;
    }
    else {
      value0 = scalar2;
      value1 = scalar0;
    }
  }
  else {  // channel === 4
    int value = element_x & 3;
    if (value == 0) {
      value0 = scalar0;
      value1 = scalar1;
    }
    else if (value == 1) {
      value0 = scalar1;
      value1 = scalar2;
    }
    else if (value == 2) {
      value0 = scalar2;
      value1 = scalar3;
    }
    else {
      value0 = scalar3;
      value1 = scalar0;
    }
  }

  if (blockIdx.x < gridDim.x - 1) {
    input_value0 = input[element_x];
    input_value1 = input[element_x + 1];

    if (sizeof(T) == 1) {
      output_value0 = saturateCast(input_value0 - value0);
      output_value1 = saturateCast(input_value1 - value1);
    }
    else {
      output_value0 = input_value0 - value0;
      output_value1 = input_value1 - value1;
    }

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    input_value0 = input[element_x];
    if (element_x != cols - 1) {
      input_value1 = input[element_x + 1];
    }

    if (element_x != cols - 1) {
      if (sizeof(T) == 1) {
        output_value0 = saturateCast(input_value0 - value0);
        output_value1 = saturateCast(input_value1 - value1);
      }
      else {
        output_value0 = input_value0 - value0;
        output_value1 = input_value1 - value1;
      }
    }
    else {
      if (sizeof(T) == 1) {
        output_value0 = saturateCast(input_value0 - value0);
      }
      else {
        output_value0 = input_value0 - value0;
      }
    }

    output[element_x] = output_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = output_value1;
    }
  }
}

RetCode subtract(const uchar* src, int rows, int cols, int channels,
                 int src_stride, const uchar* scalar, uchar* dst,
                 int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(scalar != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(uchar));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  uchar value0 = 0, value1 = 0, value2 = 0, value3 = 0;
  if (channels == 1) {
    value0 = scalar[0];
  }
  else if (channels == 3) {
    value0 = scalar[0];
    value1 = scalar[1];
    value2 = scalar[2];
  }
  else {  // channels == 4
    value0 = scalar[0];
    value1 = scalar[1];
    value2 = scalar[2];
    value3 = scalar[3];
  }

  if ((src_stride & 3) == 0 && (dst_stride & 3) == 0) {
    cols = divideUp(columns, 4, 2);
    subtractKernel0<uchar, uchar4><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, value0, value1, value2, value3, dst, dst_stride);
  }
  else if (src_stride == columns && dst_stride == columns) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    block.x = 256;
    block.y = 1;
    grid.x = divideUp(cols, 256, 8);
    grid.y = 1;
    subtractKernel10<uchar, uchar4><<<grid, block, 0, stream>>>(src, columns,
        channels, value0, value1, value2, value3, dst);
  }
  else {
    grid.x = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
    subtractKernel11<uchar><<<grid, block, 0, stream>>>(src, rows, columns,
        channels, src_stride, value0, value1, value2, value3, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode subtract(const float* src, int rows, int cols, int channels,
                 int src_stride, const float* scalar, float* dst,
                 int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(scalar != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  float value0 = 0.f, value1 = 0.f, value2 = 0.f, value3 = 0.f;
  if (channels == 1) {
    value0 = scalar[0];
  }
  else if (channels == 3) {
    value0 = scalar[0];
    value1 = scalar[1];
    value2 = scalar[2];
  }
  else {  // channels == 4
    value0 = scalar[0];
    value1 = scalar[1];
    value2 = scalar[2];
    value3 = scalar[3];
  }

  if ((src_stride & 15) == 0 && (dst_stride & 15) == 0) {
    cols = divideUp(columns, 4, 2);
    subtractKernel0<float, float4><<<grid, block, 0, stream>>>(src, rows, cols,
        channels, src_stride, value0, value1, value2, value3, dst, dst_stride);
  }
  else if (src_stride == columns * (int)sizeof(float) &&
           dst_stride == columns * (int)sizeof(float)) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    block.x = 256;
    block.y = 1;
    grid.x = divideUp(cols, 256, 8);
    grid.y = 1;
    subtractKernel10<float, float4><<<grid, block, 0, stream>>>(src, columns,
      channels, value0, value1, value2, value3, dst);
  }
  else {
    grid.x = divideUp(divideUp(columns, 2, 1), kBlockDimX1, kBlockShiftX1);
    subtractKernel11<float><<<grid, block, 0, stream>>>(src, rows, columns,
        channels, src_stride, value0, value1, value2, value3, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Subtract<uchar, 1>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           const uchar* scalar,
                           int outWidthStride,
                           uchar* outData) {
  RetCode code = subtract(inData, height, width, 1, inWidthStride, scalar,
                          outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Subtract<uchar, 3>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           const uchar* scalar,
                           int outWidthStride,
                           uchar* outData) {
  RetCode code = subtract(inData, height, width, 3, inWidthStride, scalar,
                          outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Subtract<uchar, 4>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const uchar* inData,
                           const uchar* scalar,
                           int outWidthStride,
                           uchar* outData) {
  RetCode code = subtract(inData, height, width, 4, inWidthStride, scalar,
                          outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Subtract<float, 1>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           const float* scalar,
                           int outWidthStride,
                           float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = subtract(inData, height, width, 1, inWidthStride, scalar,
                          outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Subtract<float, 3>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           const float* scalar,
                           int outWidthStride,
                           float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = subtract(inData, height, width, 3, inWidthStride, scalar,
                          outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Subtract<float, 4>(cudaStream_t stream,
                           int height,
                           int width,
                           int inWidthStride,
                           const float* inData,
                           const float* scalar,
                           int outWidthStride,
                           float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = subtract(inData, height, width, 4, inWidthStride, scalar,
                          outData, outWidthStride, stream);

  return code;
}

/**************************** multiply operation *****************************/

__global__
void multiplyKernel0(const uchar* src0, int rows, int cols, int src0_stride,
                     const uchar* src1, int src1_stride, uchar* dst,
                     int dst_stride, float scale) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const uchar4* input0 = (uchar4*)(src0 + element_y * src0_stride);
  const uchar4* input1 = (uchar4*)(src1 + element_y * src1_stride);
  uchar4 input_value0 = input0[element_x];
  uchar4 input_value1 = input1[element_x];

  float4 output_value;
  if (scale == 1.f) {
    output_value.x = input_value0.x * input_value1.x;
    output_value.y = input_value0.y * input_value1.y;
    output_value.z = input_value0.z * input_value1.z;
    output_value.w = input_value0.w * input_value1.w;
  }
  else {
    output_value.x = input_value0.x * input_value1.x * scale;
    output_value.y = input_value0.y * input_value1.y * scale;
    output_value.z = input_value0.z * input_value1.z * scale;
    output_value.w = input_value0.w * input_value1.w * scale;
  }

  uchar4* output = (uchar4*)(dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<uchar4, float4>(output_value);
}

__global__
void multiplyKernel10(const uchar* src0, int cols, const uchar* src1,
                      uchar* dst, float scale) {
  int element_x = (blockIdx.x << 8) + threadIdx.x;
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  const uchar4* input0 = (uchar4*)src0;
  const uchar4* input1 = (uchar4*)src1;
  uchar4 input_value0 = input0[element_x];
  uchar4 input_value1 = input1[element_x];

  float4 output_value;
  if (index_x < cols - 3) {
    if (scale == 1.f) {
      output_value.x = input_value0.x * input_value1.x;
      output_value.y = input_value0.y * input_value1.y;
      output_value.z = input_value0.z * input_value1.z;
      output_value.w = input_value0.w * input_value1.w;
    }
    else {
      output_value.x = input_value0.x * input_value1.x * scale;
      output_value.y = input_value0.y * input_value1.y * scale;
      output_value.z = input_value0.z * input_value1.z * scale;
      output_value.w = input_value0.w * input_value1.w * scale;
    }

    uchar4* output = (uchar4*)dst;
    output[element_x] = saturateCastVector<uchar4, float4>(output_value);
  }
  else {
    if (scale == 1.f) {
      output_value.x = input_value0.x * input_value1.x;
      if (index_x < cols - 1) {
        output_value.y = input_value0.y * input_value1.y;
      }
      if (index_x < cols - 2) {
        output_value.z = input_value0.z * input_value1.z;
      }
    }
    else {
      output_value.x = input_value0.x * input_value1.x * scale;
      if (index_x < cols - 1) {
        output_value.y = input_value0.y * input_value1.y * scale;
      }
      if (index_x < cols - 2) {
        output_value.z = input_value0.z * input_value1.z * scale;
      }
    }

    dst[index_x] = saturateCast(output_value.x);
    if (index_x < cols - 1) {
      dst[index_x + 1] = saturateCast(output_value.y);
    }
    if (index_x < cols - 2) {
      dst[index_x + 2] = saturateCast(output_value.z);
    }
  }
}

template <typename T>
__global__
void multiplyKernel11(const T* src0, int rows, int cols, int src0_stride,
                      const T* src1, int src1_stride, T* dst, int dst_stride,
                      float scale) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const T* input0 = (T*)((uchar*)src0 + element_y * src0_stride);
  const T* input1 = (T*)((uchar*)src1 + element_y * src1_stride);
  T* output  = (T*)((uchar*)dst + element_y * dst_stride);

  T input_value00, input_value01;
  T input_value10, input_value11;
  float output_value0, output_value1;

  if (blockIdx.x < gridDim.x - 1) {
    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    if (scale == 1.f) {
      output_value0 = input_value00 * input_value10;
      output_value1 = input_value01 * input_value11;
    }
    else {
      output_value0 = input_value00 * input_value10 * scale;
      output_value1 = input_value01 * input_value11 * scale;
    }

    if (sizeof(T) == 1) {
      output[element_x]     = saturateCast(output_value0);
      output[element_x + 1] = saturateCast(output_value1);
    }
    else {
      output[element_x]     = output_value0;
      output[element_x + 1] = output_value1;
    }
  }
  else {
    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    if (scale == 1.f) {
      output_value0 = input_value00 * input_value10;
      if (element_x != cols - 1) {
        output_value1 = input_value01 * input_value11;
      }
    }
    else {
      output_value0 = input_value00 * input_value10 * scale;
      if (element_x != cols - 1) {
        output_value1 = input_value01 * input_value11 * scale;
      }
    }

    if (sizeof(T) == 1) {
      output[element_x] = saturateCast(output_value0);
      if (element_x != cols - 1) {
        output[element_x + 1] = saturateCast(output_value1);
      }
    }
    else {
      output[element_x] = output_value0;
      if (element_x != cols - 1) {
        output[element_x + 1] = output_value1;
      }
    }
  }
}

__global__
void multiplyKernel0(const float* src0, int rows, int cols, int src0_stride,
                     const float* src1, int src1_stride, float* dst,
                     int dst_stride, float scale) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float2* input0 = (float2*)((uchar*)src0 + element_y * src0_stride);
  const float2* input1 = (float2*)((uchar*)src1 + element_y * src1_stride);
  float2 input_value0 = input0[element_x];
  float2 input_value1 = input1[element_x];

  float2 output_value;
  if (scale == 1.f) {
    output_value.x = input_value0.x * input_value1.x;
    output_value.y = input_value0.y * input_value1.y;
  }
  else {
    output_value.x = input_value0.x * input_value1.x * scale;
    output_value.y = input_value0.y * input_value1.y * scale;
  }

  float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = output_value;
}

RetCode multiply(const uchar* src0, int rows, int cols, int channels,
                 int src0_stride, const uchar* src1, int src1_stride,
                 uchar* dst, int dst_stride, float scale, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(uchar));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 3) == 0 && (src1_stride & 3) == 0 &&
      (dst_stride & 3) == 0) {
    cols = divideUp(columns, 4, 2);
    multiplyKernel0<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride,
        src1, src1_stride, dst, dst_stride, scale);
  }
  else if (src0_stride == columns && src1_stride == columns &&
           dst_stride == columns) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    block.x = 256;
    block.y = 1;
    grid.x = divideUp(cols, 256, 8);
    grid.y = 1;
    multiplyKernel10<<<grid, block, 0, stream>>>(src0, columns, src1, dst,
                                                 scale);
  }
  else {
    grid.x = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
    multiplyKernel11<uchar><<<grid, block, 0, stream>>>(src0, rows, columns,
        src0_stride, src1, src1_stride, dst, dst_stride, scale);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode multiply(const float* src0, int rows, int cols, int channels,
                 int src0_stride, const float* src1, int src1_stride,
                 float* dst, int dst_stride, float scale, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 7) == 0 && (src1_stride & 7) == 0 &&
      (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    multiplyKernel0<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride,
        src1, src1_stride, dst, dst_stride, scale);
  }
  else {
    multiplyKernel11<float><<<grid, block, 0, stream>>>(src0, rows, columns,
        src0_stride, src1, src1_stride, dst, dst_stride, scale);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Mul<uchar, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData,
                      float scale) {
  RetCode code = multiply(inData0, height, width, 1, inWidthStride0, inData1,
                          inWidthStride1, outData, outWidthStride, scale,
                          stream);

  return code;
}

template <>
RetCode Mul<uchar, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData,
                      float scale) {
  RetCode code = multiply(inData0, height, width, 3, inWidthStride0, inData1,
                          inWidthStride1, outData, outWidthStride, scale,
                          stream);

  return code;
}

template <>
RetCode Mul<uchar, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData,
                      float scale) {
  RetCode code = multiply(inData0, height, width, 4, inWidthStride0, inData1,
                          inWidthStride1, outData, outWidthStride, scale,
                          stream);

  return code;
}

template <>
RetCode Mul<float, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData,
                      float scale) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = multiply(inData0, height, width, 1, inWidthStride0, inData1,
                          inWidthStride1, outData, outWidthStride, scale,
                          stream);

  return code;
}

template <>
RetCode Mul<float, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData,
                      float scale) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = multiply(inData0, height, width, 3, inWidthStride0, inData1,
                          inWidthStride1, outData, outWidthStride, scale,
                          stream);

  return code;
}

template <>
RetCode Mul<float, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData,
                      float scale) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = multiply(inData0, height, width, 4, inWidthStride0, inData1,
                          inWidthStride1, outData, outWidthStride, scale,
                          stream);

  return code;
}

/***************************** divide operation ******************************/

__global__
void divideKernel0(const uchar* src0, int rows, int cols, int src0_stride,
                   const uchar* src1, int src1_stride, uchar* dst,
                   int dst_stride, float scale) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const uchar4* input0 = (uchar4*)(src0 + element_y * src0_stride);
  const uchar4* input1 = (uchar4*)(src1 + element_y * src1_stride);
  uchar4 input_value0 = input0[element_x];
  uchar4 input_value1 = input1[element_x];

  float4 output_value;
  if (scale == 1.f) {
    output_value.x = input_value1.x == 0 ? 0 : input_value0.x / input_value1.x;
    output_value.y = input_value1.y == 0 ? 0 : input_value0.y / input_value1.y;
    output_value.z = input_value1.z == 0 ? 0 : input_value0.z / input_value1.z;
    output_value.w = input_value1.w == 0 ? 0 : input_value0.w / input_value1.w;
  }
  else {
    output_value.x = input_value1.x == 0 ? 0 :
                       scale * input_value0.x / input_value1.x;
    output_value.y = input_value1.y == 0 ? 0 :
                       scale * input_value0.y / input_value1.y;
    output_value.z = input_value1.z == 0 ? 0 :
                       scale * input_value0.z / input_value1.z;
    output_value.w = input_value1.w == 0 ? 0 :
                       scale * input_value0.w / input_value1.w;
  }

  uchar4* output = (uchar4*)(dst + element_y * dst_stride);
  output[element_x] = saturateCastVector<uchar4, float4>(output_value);
}

__global__
void divideKernel10(const uchar* src0, int cols, const uchar* src1, uchar* dst,
                    float scale) {
  int element_x = (blockIdx.x << 8) + threadIdx.x;
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  const uchar4* input0 = (uchar4*)src0;
  const uchar4* input1 = (uchar4*)src1;
  uchar4 input_value0 = input0[element_x];
  uchar4 input_value1 = input1[element_x];

  float4 output_value;
  if (index_x < cols - 3) {
    if (scale == 1.f) {
      output_value.x = input_value1.x == 0 ? 0 :
                         input_value0.x / input_value1.x;
      output_value.y = input_value1.y == 0 ? 0 :
                         input_value0.y / input_value1.y;
      output_value.z = input_value1.z == 0 ? 0 :
                         input_value0.z / input_value1.z;
      output_value.w = input_value1.w == 0 ? 0 :
                         input_value0.w / input_value1.w;
    }
    else {
      output_value.x = input_value1.x == 0 ? 0 :
                         scale * input_value0.x / input_value1.x;
      output_value.y = input_value1.y == 0 ? 0 :
                         scale * input_value0.y / input_value1.y;
      output_value.z = input_value1.z == 0 ? 0 :
                         scale * input_value0.z / input_value1.z;
      output_value.w = input_value1.w == 0 ? 0 :
                         scale * input_value0.w / input_value1.w;
    }

    uchar4* output = (uchar4*)dst;
    output[element_x] = saturateCastVector<uchar4, float4>(output_value);
  }
  else {
    if (scale == 1.f) {
      output_value.x = input_value1.x == 0 ? 0 :
                         input_value0.x / input_value1.x;
      if (index_x < cols - 1) {
        output_value.y = input_value1.y == 0 ? 0 :
                           input_value0.y / input_value1.y;
      }
      if (index_x < cols - 2) {
        output_value.z = input_value1.z == 0 ? 0 :
                           input_value0.z / input_value1.z;
      }
    }
    else {
      output_value.x = input_value1.x == 0 ? 0 :
                         scale * input_value0.x / input_value1.x;
      if (index_x < cols - 1) {
        output_value.y = input_value1.y == 0 ? 0 :
                           scale * input_value0.y / input_value1.y;
      }
      if (index_x < cols - 2) {
        output_value.z = input_value1.z == 0 ? 0 :
                           scale * input_value0.z / input_value1.z;
      }
    }

    dst[index_x] = saturateCast(output_value.x);
    if (index_x < cols - 1) {
      dst[index_x + 1] = saturateCast(output_value.y);
    }
    if (index_x < cols - 2) {
      dst[index_x + 2] = saturateCast(output_value.z);
    }
  }
}

template <typename T>
__global__
void divideKernel11(const T* src0, int rows, int cols, int src0_stride,
                    const T* src1, int src1_stride, T* dst, int dst_stride,
                    float scale) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const T* input0 = (T*)((uchar*)src0 + element_y * src0_stride);
  const T* input1 = (T*)((uchar*)src1 + element_y * src1_stride);
  T* output = (T*)((uchar*)dst + element_y * dst_stride);

  T input_value00, input_value01;
  T input_value10, input_value11;
  float output_value0, output_value1;

  if (blockIdx.x < gridDim.x - 1) {
    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    if (scale == 1.f) {
      output_value0 = input_value10 == 0 ? 0 : input_value00 / input_value10;
      output_value1 = input_value11 == 0 ? 0 : input_value01 / input_value11;
    }
    else {
      output_value0 = input_value10 == 0 ? 0 :
                        scale * input_value00 / input_value10;
      output_value1 = input_value11 == 0 ? 0 :
                        scale * input_value01 / input_value11;
    }

    if (sizeof(T) == 1) {
      output[element_x]     = saturateCast(output_value0);
      output[element_x + 1] = saturateCast(output_value1);
    }
    else {
      output[element_x]     = output_value0;
      output[element_x + 1] = output_value1;
    }
  }
  else {
    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    if (scale == 1.f) {
      output_value0 = input_value10 == 0 ? 0 : input_value00 / input_value10;
      if (element_x != cols - 1) {
        output_value1 = input_value11 == 0 ? 0 : input_value01 / input_value11;
      }
    }
    else {
      output_value0 = input_value10 == 0 ? 0 :
                        scale * input_value00 / input_value10;
      if (element_x != cols - 1) {
        output_value1 = input_value11 == 0 ? 0 :
                          scale * input_value01 / input_value11;
      }
    }

    if (sizeof(T) == 1) {
      output[element_x] = saturateCast(output_value0);
      if (element_x != cols - 1) {
        output[element_x + 1] = saturateCast(output_value1);
      }
    }
    else {
      output[element_x] = output_value0;
      if (element_x != cols - 1) {
        output[element_x + 1] = output_value1;
      }
    }
  }
}

__global__
void divideKernel0(const float* src0, int rows, int cols, int src0_stride,
                   const float* src1, int src1_stride, float* dst,
                   int dst_stride, float scale) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float2* input0 = (float2*)((uchar*)src0 + element_y * src0_stride);
  const float2* input1 = (float2*)((uchar*)src1 + element_y * src1_stride);
  float2 input_value0 = input0[element_x];
  float2 input_value1 = input1[element_x];

  float2 output_value;
  if (scale == 1.f) {
    output_value.x = input_value1.x == 0 ? 0 : input_value0.x / input_value1.x;
    output_value.y = input_value1.y == 0 ? 0 : input_value0.y / input_value1.y;
  }
  else {
    output_value.x = input_value1.x == 0 ? 0 :
                       scale * input_value0.x / input_value1.x;
    output_value.y = input_value1.y == 0 ? 0 :
                       scale * input_value0.y / input_value1.y;
  }

  float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = output_value;
}

RetCode divide(const uchar* src0, int rows, int cols, int channels,
               int src0_stride, const uchar* src1, int src1_stride,
               uchar* dst, int dst_stride, float scale, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(uchar));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 3) == 0 && (src1_stride & 3) == 0 &&
      (dst_stride & 3) == 0) {
    cols = divideUp(columns, 4, 2);
    divideKernel0<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride,
        src1, src1_stride, dst, dst_stride, scale);
  }
  else if (src0_stride == columns && src1_stride == columns &&
           dst_stride == columns) {
    columns *= rows;
    cols = divideUp(columns, 4, 2);
    block.x = 256;
    block.y = 1;
    grid.x = divideUp(cols, 256, 8);
    grid.y = 1;
    divideKernel10<<<grid, block, 0, stream>>>(src0, columns, src1, dst,
                                                 scale);
  }
  else {
    grid.x = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
    divideKernel11<uchar><<<grid, block, 0, stream>>>(src0, rows, columns,
        src0_stride, src1, src1_stride, dst, dst_stride, scale);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode divide(const float* src0, int rows, int cols, int channels,
               int src0_stride, const float* src1, int src1_stride,
               float* dst, int dst_stride, float scale, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 7) == 0 && (src1_stride & 7) == 0 &&
      (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    divideKernel0<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride,
        src1, src1_stride, dst, dst_stride, scale);
  }
  else {
    divideKernel11<float><<<grid, block, 0, stream>>>(src0, rows, columns,
        src0_stride, src1, src1_stride, dst, dst_stride, scale);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Div<uchar, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData,
                      float scale) {
  RetCode code = divide(inData0, height, width, 1, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, scale, stream);

  return code;
}

template <>
RetCode Div<uchar, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData,
                      float scale) {
  RetCode code = divide(inData0, height, width, 3, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, scale, stream);

  return code;
}

template <>
RetCode Div<uchar, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const uchar* inData0,
                      int inWidthStride1,
                      const uchar* inData1,
                      int outWidthStride,
                      uchar* outData,
                      float scale) {
  RetCode code = divide(inData0, height, width, 4, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, scale, stream);

  return code;
}

template <>
RetCode Div<float, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData,
                      float scale) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = divide(inData0, height, width, 1, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, scale, stream);

  return code;
}

template <>
RetCode Div<float, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData,
                      float scale) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = divide(inData0, height, width, 3, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, scale, stream);

  return code;
}

template <>
RetCode Div<float, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData,
                      float scale) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = divide(inData0, height, width, 4, inWidthStride0, inData1,
                        inWidthStride1, outData, outWidthStride, scale, stream);

  return code;
}

/******************** multiplication-addition operation ********************/

__global__
void mlaKernel0(const float* src0, int rows, int cols, int src0_stride,
                const float* src1, int src1_stride, float* dst,
                int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float2* input0 = (float2*)((uchar*)src0 + element_y * src0_stride);
  const float2* input1 = (float2*)((uchar*)src1 + element_y * src1_stride);
  float2 input_value0 = input0[element_x];
  float2 input_value1 = input1[element_x];

  float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
  float2 output_value = output[element_x];

  output_value.x += input_value0.x * input_value1.x;
  output_value.y += input_value0.y * input_value1.y;

  output[element_x] = output_value;
}

__global__
void mlaKernel1(const float* src0, int rows, int cols, int src0_stride,
                const float* src1, int src1_stride, float* dst,
                int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float* input0 = (float*)((uchar*)src0 + element_y * src0_stride);
  const float* input1 = (float*)((uchar*)src1 + element_y * src1_stride);
  float* output = (float*)((uchar*)dst + element_y * dst_stride);

  float input_value00, input_value01;
  float input_value10, input_value11;
  float output_value0, output_value1;

  if (blockIdx.x < gridDim.x - 1) {
    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    output_value0 = output[element_x];
    output_value1 = output[element_x + 1];

    output_value0 += input_value00 * input_value10;
    output_value1 += input_value01 * input_value11;

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    output_value0 = output[element_x];
    if (element_x != cols - 1) {
      output_value1 = output[element_x + 1];
    }

    output_value0 += input_value00 * input_value10;
    output_value1 += input_value01 * input_value11;

    output[element_x] = output_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = output_value1;
    }
  }
}

RetCode mla(const float* src0, int rows, int cols, int channels,
            int src0_stride, const float* src1, int src1_stride, float* dst,
            int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 7) == 0 && (src1_stride & 7) == 0 &&
      (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    mlaKernel0<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride,
        src1, src1_stride, dst, dst_stride);
  }
  else {
    mlaKernel1<<<grid, block, 0, stream>>>(src0, rows, columns, src0_stride,
        src1, src1_stride, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Mla<float, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = mla(inData0, height, width, 1, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Mla<float, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = mla(inData0, height, width, 3, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Mla<float, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = mla(inData0, height, width, 4, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

/******************** multiplication-subtraction operation ********************/

__global__
void mlsKernel0(const float* src0, int rows, int cols, int src0_stride,
                const float* src1, int src1_stride, float* dst,
                int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float2* input0 = (float2*)((uchar*)src0 + element_y * src0_stride);
  const float2* input1 = (float2*)((uchar*)src1 + element_y * src1_stride);
  float2 input_value0 = input0[element_x];
  float2 input_value1 = input1[element_x];

  float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
  float2 output_value = output[element_x];

  output_value.x -= input_value0.x * input_value1.x;
  output_value.y -= input_value0.y * input_value1.y;

  output[element_x] = output_value;
}

__global__
void mlsKernel1(const float* src0, int rows, int cols, int src0_stride,
                const float* src1, int src1_stride, float* dst,
                int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX0) + threadIdx.x) << 1;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float* input0 = (float*)((uchar*)src0 + element_y * src0_stride);
  const float* input1 = (float*)((uchar*)src1 + element_y * src1_stride);
  float* output = (float*)((uchar*)dst + element_y * dst_stride);

  float input_value00, input_value01;
  float input_value10, input_value11;
  float output_value0, output_value1;

  if (blockIdx.x < gridDim.x - 1) {
    input_value00 = input0[element_x];
    input_value01 = input0[element_x + 1];

    input_value10 = input1[element_x];
    input_value11 = input1[element_x + 1];

    output_value0 = output[element_x];
    output_value1 = output[element_x + 1];

    output_value0 -= input_value00 * input_value10;
    output_value1 -= input_value01 * input_value11;

    output[element_x]     = output_value0;
    output[element_x + 1] = output_value1;
  }
  else {
    input_value00 = input0[element_x];
    if (element_x != cols - 1) {
      input_value01 = input0[element_x + 1];
    }

    input_value10 = input1[element_x];
    if (element_x != cols - 1) {
      input_value11 = input1[element_x + 1];
    }

    output_value0 = output[element_x];
    if (element_x != cols - 1) {
      output_value1 = output[element_x + 1];
    }

    output_value0 -= input_value00 * input_value10;
    output_value1 -= input_value01 * input_value11;

    output[element_x] = output_value0;
    if (element_x != cols - 1) {
      output[element_x + 1] = output_value1;
    }
  }
}

RetCode mls(const float* src0, int rows, int cols, int channels,
            int src0_stride, const float* src1, int src1_stride, float* dst,
            int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride  >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(divideUp(columns, 2, 1), kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(rows, kBlockDimY0, kBlockShiftY0);

  if ((src0_stride & 7) == 0 && (src1_stride & 7) == 0 &&
      (dst_stride & 7) == 0) {
    cols = divideUp(columns, 2, 1);
    mlsKernel0<<<grid, block, 0, stream>>>(src0, rows, cols, src0_stride,
        src1, src1_stride, dst, dst_stride);
  }
  else {
    mlsKernel1<<<grid, block, 0, stream>>>(src0, rows, columns, src0_stride,
        src1, src1_stride, dst, dst_stride);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Mls<float, 1>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = mls(inData0, height, width, 1, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Mls<float, 3>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = mls(inData0, height, width, 3, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Mls<float, 4>(cudaStream_t stream,
                      int height,
                      int width,
                      int inWidthStride0,
                      const float* inData0,
                      int inWidthStride1,
                      const float* inData1,
                      int outWidthStride,
                      float* outData) {
  inWidthStride0 *= sizeof(float);
  inWidthStride1 *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = mls(inData0, height, width, 4, inWidthStride0, inData1,
                     inWidthStride1, outData, outWidthStride, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
