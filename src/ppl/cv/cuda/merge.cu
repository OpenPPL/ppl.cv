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

#include "ppl/cv/cuda/merge.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename T>
__global__
void merge3Kernel(const T* src0, const T* src1, const T* src2, int rows,
                  int cols, int src_stride, T* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int offset = element_y * src_stride;
  T* input0 = (T*)((uchar*)src0 + offset);
  T* input1 = (T*)((uchar*)src1 + offset);
  T* input2 = (T*)((uchar*)src2 + offset);
  T value0  = input0[element_x];
  T value1  = input1[element_x];
  T value2  = input2[element_x];

  element_x *= 3;
  T* output = (T*)((uchar*)dst + element_y * dst_stride);
  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
}

RetCode merge3Channels(const uchar* src0, const uchar* src1, const uchar* src2,
                       int rows, int cols, int src_stride, uchar* dst,
                       int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(dst  != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * 3 * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  merge3Kernel<uchar><<<grid, block, 0, stream>>>(src0, src1, src2, rows, cols,
                                                  src_stride, dst, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode merge3Channels(const float* src0, const float* src1, const float* src2,
                       int rows, int cols, int src_stride, float* dst,
                       int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(dst  != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 3 * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  merge3Kernel<float><<<grid, block, 0, stream>>>(src0, src1, src2, rows, cols,
                                                  src_stride, dst, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Merge3Channels<uchar>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const uchar* inData0,
                              const uchar* inData1,
                              const uchar* inData2,
                              int outWidthStride,
                              uchar* outData) {
  RetCode code = merge3Channels(inData0, inData1, inData2, height, width,
                                inWidthStride, outData, outWidthStride, stream);

  return code;
}

template <>
RetCode Merge3Channels<float>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const float* inData0,
                              const float* inData1,
                              const float* inData2,
                              int outWidthStride,
                              float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = merge3Channels(inData0, inData1, inData2, height, width,
                                inWidthStride, outData, outWidthStride, stream);

  return code;
}

template <typename T>
__global__
void merge4Kernel(const T* src0, const T* src1, const T* src2, const T* src3,
                  int rows, int cols, int src_stride, T* dst, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int offset = element_y * src_stride;
  T* input0 = (T*)((uchar*)src0 + offset);
  T* input1 = (T*)((uchar*)src1 + offset);
  T* input2 = (T*)((uchar*)src2 + offset);
  T* input3 = (T*)((uchar*)src3 + offset);
  T value0  = input0[element_x];
  T value1  = input1[element_x];
  T value2  = input2[element_x];
  T value3  = input3[element_x];

  element_x = (element_x << 2);
  T* output = (T*)((uchar*)dst + element_y * dst_stride);
  output[element_x]     = value0;
  output[element_x + 1] = value1;
  output[element_x + 2] = value2;
  output[element_x + 3] = value3;
}

RetCode merge4Channels(const uchar* src0, const uchar* src1, const uchar* src2,
                       const uchar* src3, int rows, int cols, int src_stride,
                       uchar* dst, int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(src3 != nullptr);
  PPL_ASSERT(dst  != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * 4 * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  merge4Kernel<uchar><<<grid, block, 0, stream>>>(src0, src1, src2, src3, rows,
                                                  cols, src_stride, dst,
                                                  dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode merge4Channels(const float* src0, const float* src1, const float* src2,
                       const float* src3, int rows, int cols, int src_stride,
                       float* dst, int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(src2 != nullptr);
  PPL_ASSERT(src3 != nullptr);
  PPL_ASSERT(dst  != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * 4 * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  merge4Kernel<float><<<grid, block, 0, stream>>>(src0, src1, src2, src3, rows,
                                                  cols, src_stride, dst,
                                                  dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Merge4Channels<uchar>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const uchar* inData0,
                              const uchar* inData1,
                              const uchar* inData2,
                              const uchar* inData3,
                              int outWidthStride,
                              uchar* outData) {
  RetCode code = merge4Channels(inData0, inData1, inData2, inData3, height,
                                width, inWidthStride, outData, outWidthStride,
                                stream);

  return code;
}

template <>
RetCode Merge4Channels<float>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const float* inData0,
                              const float* inData1,
                              const float* inData2,
                              const float* inData3,
                              int outWidthStride,
                              float* outData) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = merge4Channels(inData0, inData1, inData2, inData3, height,
                                width, inWidthStride, outData, outWidthStride,
                                stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl