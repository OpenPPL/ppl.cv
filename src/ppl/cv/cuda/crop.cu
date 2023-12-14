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

#include "ppl/cv/cuda/crop.h"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename T>
__global__
void cropKernel(const T* src, int src_stride, const int top, const int left,
                const float scale, T* dst, int dst_rows, int dst_cols,
                int dst_stride) {
  int element_x = ((blockIdx.x << kBlockShiftX1) + threadIdx.x) << 2;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  T* input = (T*)((uchar*)src + (top + element_y) * src_stride);
  T value0, value1, value2, value3;
  value0 = input[left + element_x];
  value1 = input[left + element_x + 1];
  value2 = input[left + element_x + 2];
  value3 = input[left + element_x + 3];

  if (scale != 1.f) {
    float fvalue0, fvalue1, fvalue2, fvalue3;
    fvalue0 = value0 * scale;
    fvalue1 = value1 * scale;
    fvalue2 = value2 * scale;
    fvalue3 = value3 * scale;

    if (sizeof(T) == 1) {
      value0 = saturateCast(fvalue0);
      value1 = saturateCast(fvalue1);
      value2 = saturateCast(fvalue2);
      value3 = saturateCast(fvalue3);
    }
    else {
      value0 = fvalue0;
      value1 = fvalue1;
      value2 = fvalue2;
      value3 = fvalue3;
    }
  }

  T* output = (T*)((uchar*)dst + element_y * dst_stride);
  if (element_x < dst_cols - 3) {
    output[element_x] = value0;
    output[element_x + 1] = value1;
    output[element_x + 2] = value2;
    output[element_x + 3] = value3;
  }
  else {
    output[element_x] = value0;
    if (element_x < dst_cols - 1) {
      output[element_x + 1] = value1;
    }
    if (element_x < dst_cols - 2) {
      output[element_x + 2] = value2;
    }
  }
}

RetCode crop(const uchar* src, int src_rows, int src_cols, int channels,
             int src_stride, uchar* dst, int dst_rows, int dst_cols,
             int dst_stride, const int left, const int top, const float scale,
             cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(src_rows >= dst_rows && src_cols >= dst_cols);
  PPL_ASSERT(left >= 0 && left < src_cols);
  PPL_ASSERT(top >= 0 && top < src_rows);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(uchar));

  int columns = dst_cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  cudaError_t code;
  if (scale == 1.f) {
    uchar* src_start = (uchar*)src + top * src_stride +
                       left * channels * sizeof(uchar);
    code = cudaMemcpy2DAsync(dst, dst_stride, src_start, src_stride,
                        dst_cols * channels * sizeof(uchar), dst_rows,
                        cudaMemcpyDeviceToDevice, stream);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }
  else {
    cropKernel<uchar><<<grid, block, 0, stream>>>(src, src_stride, top,
        left * channels, scale, dst, dst_rows, columns, dst_stride);
    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }
  }

  return RC_SUCCESS;
}

RetCode crop(const float* src, int src_rows, int src_cols, int channels,
             int src_stride, float* dst, int dst_rows, int dst_cols,
             int dst_stride, const int left, const int top, const float scale,
             cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(src_rows >= dst_rows && src_cols >= dst_cols);
  PPL_ASSERT(left >= 0 && left < src_cols);
  PPL_ASSERT(top >= 0 && top < src_rows);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(float));

  int columns = dst_cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(divideUp(columns, 4, 2), kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  cudaError_t code;
  if (scale == 1.f) {
    float* src_start = (float*)((uchar*)src + top * src_stride +
                       left * channels * sizeof(float));
    code = cudaMemcpy2DAsync(dst, dst_stride, src_start, src_stride,
                        dst_cols * channels * sizeof(float), dst_rows,
                        cudaMemcpyDeviceToDevice, stream);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }
  else {
    cropKernel<float><<<grid, block, 0, stream>>>(src, src_stride, top,
        left * channels, scale, dst, dst_rows, columns, dst_stride);

    code = cudaGetLastError();
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_RUNTIME_ERROR;
    }
  }

  return RC_SUCCESS;
}

template <>
RetCode Crop<uchar, 1>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       uchar* outData,
                       const int left,
                       const int top,
                       const float scale) {
  RetCode code = crop(inData, inHeight, inWidth, 1, inWidthStride, outData,
                      outHeight, outWidth, outWidthStride, left, top, scale,
                      stream);

  return code;
}

template <>
RetCode Crop<uchar, 3>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       uchar* outData,
                       const int left,
                       const int top,
                       const float scale) {
  RetCode code = crop(inData, inHeight, inWidth, 3, inWidthStride, outData,
                      outHeight, outWidth, outWidthStride, left, top, scale,
                      stream);

  return code;
}

template <>
RetCode Crop<uchar, 4>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const uchar* inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       uchar* outData,
                       const int left,
                       const int top,
                       const float scale) {
  RetCode code = crop(inData, inHeight, inWidth, 4, inWidthStride, outData,
                      outHeight, outWidth, outWidthStride, left, top, scale,
                      stream);

  return code;
}

template <>
RetCode Crop<float, 1>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       float* outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = crop(inData, inHeight, inWidth, 1, inWidthStride, outData,
                      outHeight, outWidth, outWidthStride, left, top, scale,
                      stream);

  return code;
}

template <>
RetCode Crop<float, 3>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       float* outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = crop(inData, inHeight, inWidth, 3, inWidthStride, outData,
                      outHeight, outWidth, outWidthStride, left, top, scale,
                      stream);

  return code;
}

template <>
RetCode Crop<float, 4>(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const float* inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       float* outData,
                       const int left,
                       const int top,
                       const float scale) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = crop(inData, inHeight, inWidth, 4, inWidthStride, outData,
                      outHeight, outWidth, outWidthStride, left, top, scale,
                      stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
