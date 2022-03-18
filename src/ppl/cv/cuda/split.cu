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

#include "ppl/cv/cuda/split.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename T>
__global__
void split3Kernel(const T* src, int rows, int cols, int src_stride, T* dst0,
                  T* dst1, T* dst2, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int index = element_x * 3;
  T* input = (T*)((uchar*)src + element_y * src_stride);
  T value0, value1, value2;
  value0 = input[index];
  value1 = input[index + 1];
  value2 = input[index + 2];

  int offset = element_y * dst_stride;
  T* output0 = (T*)((uchar*)dst0 + offset);
  T* output1 = (T*)((uchar*)dst1 + offset);
  T* output2 = (T*)((uchar*)dst2 + offset);
  output0[element_x] = value0;
  output1[element_x] = value1;
  output2[element_x] = value2;
}

RetCode split3Channels(const uchar* src, int rows, int cols, int src_stride,
                       uchar* dst0, uchar* dst1, uchar* dst2, int dst_stride,
                       cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst0 != nullptr);
  PPL_ASSERT(dst1 != nullptr);
  PPL_ASSERT(dst2 != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 3 * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  split3Kernel<uchar><<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
                                                  dst0, dst1, dst2, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode split3Channels(const float* src, int rows, int cols, int src_stride,
                       float* dst0, float* dst1, float* dst2, int dst_stride,
                       cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst0 != nullptr);
  PPL_ASSERT(dst1 != nullptr);
  PPL_ASSERT(dst2 != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 3 * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  split3Kernel<float><<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
                                                  dst0, dst1, dst2, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Split3Channels<uchar>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const uchar* inData,
                              int outWidthStride,
                              uchar* outData0,
                              uchar* outData1,
                              uchar* outData2) {
  RetCode code = split3Channels(inData, height, width, inWidthStride, outData0,
                                outData1, outData2, outWidthStride, stream);

  return code;
}

template <>
RetCode Split3Channels<float>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const float* inData,
                              int outWidthStride,
                              float* outData0,
                              float* outData1,
                              float* outData2) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = split3Channels(inData, height, width, inWidthStride, outData0,
                                outData1, outData2, outWidthStride, stream);

  return code;
}

template <typename T>
__global__
void split4Kernel(const T* src, int rows, int cols, int src_stride, T* dst0,
                  T* dst1, T* dst2, T* dst3, int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= cols || element_y >= rows) {
    return;
  }

  int index = element_x << 2;
  T* input = (T*)((uchar*)src + element_y * src_stride);
  T value0, value1, value2, value3;
  value0 = input[index];
  value1 = input[index + 1];
  value2 = input[index + 2];
  value3 = input[index + 3];

  int offset = element_y * dst_stride;
  T* output0 = (T*)((uchar*)dst0 + offset);
  T* output1 = (T*)((uchar*)dst1 + offset);
  T* output2 = (T*)((uchar*)dst2 + offset);
  T* output3 = (T*)((uchar*)dst3 + offset);
  output0[element_x] = value0;
  output1[element_x] = value1;
  output2[element_x] = value2;
  output3[element_x] = value3;
}

RetCode split4Channels(const uchar* src, int rows, int cols, int src_stride,
                       uchar* dst0, uchar* dst1, uchar* dst2, uchar* dst3,
                       int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst0 != nullptr);
  PPL_ASSERT(dst1 != nullptr);
  PPL_ASSERT(dst2 != nullptr);
  PPL_ASSERT(dst3 != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 4 * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(uchar));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  split4Kernel<uchar><<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
                                                  dst0, dst1, dst2, dst3,
                                                  dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode split4Channels(const float* src, int rows, int cols, int src_stride,
                       float* dst0, float* dst1, float* dst2, float* dst3,
                       int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst0 != nullptr);
  PPL_ASSERT(dst1 != nullptr);
  PPL_ASSERT(dst2 != nullptr);
  PPL_ASSERT(dst3 != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * 4 * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * (int)sizeof(float));

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  split4Kernel<float><<<grid, block, 0, stream>>>(src, rows, cols, src_stride,
                                                  dst0, dst1, dst2, dst3,
                                                  dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode Split4Channels<uchar>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const uchar* inData,
                              int outWidthStride,
                              uchar* outData0,
                              uchar* outData1,
                              uchar* outData2,
                              uchar* outData3) {
  RetCode code = split4Channels(inData, height, width, inWidthStride, outData0,
                                outData1, outData2, outData3, outWidthStride,
                                stream);

  return code;
}

template <>
RetCode Split4Channels<float>(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const float* inData,
                              int outWidthStride,
                              float* outData0,
                              float* outData1,
                              float* outData2,
                              float* outData3) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = split4Channels(inData, height, width, inWidthStride, outData0,
                                outData1, outData2, outData3, outWidthStride,
                                stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
