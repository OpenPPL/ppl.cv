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

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename T>
__global__
void remapLinearKernel(const T* src, int src_rows, int src_cols,
                       int channels, int src_stride, const float* map_x,
                       const float* map_y, T* dst, int dst_rows,
                       int dst_cols, int dst_stride, BorderType border_type,
                       T border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int dst_xy = element_y * dst_cols + element_x;
  int dst_index = element_y * dst_stride + element_x * channels;
  float float_x = map_x[dst_xy];
  float float_y = map_y[dst_xy];
  int int_x = (int)(float_x);
  int int_y = (int)(float_y);
  float fractional_x = (float)(float_x - int_x);
  float fractional_y = (float)(float_y - int_y);

  float value0, value1, value2, value3;
  float tab[4];
  float tab_y[2], tab_x[2];
  tab_y[0] = 1.0f - fractional_y;
  tab_y[1] = fractional_y;
  tab_x[0] = 1.0f - fractional_x;
  tab_x[1] = fractional_x;
  tab[0] = tab_y[0] * tab_x[0];
  tab[1] = tab_y[0] * tab_x[1];
  tab[2] = tab_y[1] * tab_x[0];
  tab[3] = tab_y[1] * tab_x[1];

  if (border_type == BORDER_CONSTANT) {
    bool flag0 = int_x >= 0 && int_x < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag1 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y >= 0 &&
                 int_y < src_rows;
    bool flag2 = int_x >= 0 && int_x < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;
    bool flag3 = int_x + 1 >= 0 && int_x + 1 < src_cols && int_y + 1 >= 0 &&
                 int_y + 1 < src_rows;
    int position0 = int_y * src_stride + int_x * channels;
    int position1 = (int_y + 1) * src_stride + int_x * channels;
    for (int i = 0; i < channels; i++) {
      value0 = flag0 ? src[position0 + i] : border_value;
      value1 = flag1 ? src[position0 + channels + i] : border_value;
      value2 = flag2 ? src[position1 + i] : border_value;
      value3 = flag3 ? src[position1 + channels + i] : border_value;
      float sum = value0 * tab[0] + value1 * tab[1] + value2 * tab[2] +
                  value3 * tab[3];
      if (sizeof(T) == 1) {
        dst[dst_index + i] = saturateCast(sum);
      }
      else {
        dst[dst_index + i] = sum;
      }
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    int int_x1 = int_x + 1;
    int int_y1 = int_y + 1;
    int_x  = clip(int_x, 0, src_cols - 1);
    int_x1 = clip(int_x1, 0, src_cols - 1);
    int_y  = clip(int_y, 0, src_rows - 1);
    int_y1 = clip(int_y1, 0, src_rows - 1);
    const T* src0 = src + int_y * src_stride + int_x * channels;
    const T* src1 = src + int_y * src_stride + int_x1 * channels;
    const T* src2 = src + int_y1 * src_stride + int_x * channels;
    const T* src3 = src + int_y1 * src_stride + int_x1 * channels;
    for (int i = 0; i < channels; ++i) {
      float sum = src0[i] * tab[0] + src1[i] * tab[1] + src2[i] * tab[2] +
                  src3[i] * tab[3];
      if (sizeof(T) == 1) {
        dst[dst_index + i] = saturateCast(sum);
      }
      else {
        dst[dst_index + i] = sum;
      }
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
      int position0 = (int_y * src_stride + int_x * channels);
      int position1 = ((int_y + 1) * src_stride + int_x * channels);
      for (int i = 0; i < channels; i++) {
        value0 = src[position0 + i];
        value1 = src[position0 + channels + i];
        value2 = src[position1 + i];
        value3 = src[position1 + channels +i];
        float sum = value0 * tab[0] + value1 * tab[1] + value2 * tab[2] +
                    value3 * tab[3];
        if (sizeof(T) == 1) {
          dst[dst_index + i] = saturateCast(sum);
        }
        else {
          dst[dst_index + i] = sum;
        }
      }
    }
  }
  else {
  }
}

template <typename T>
__global__
void remapNPKernel(const T* src, int src_rows, int src_cols, int channels,
                   int src_stride, const float* map_x, const float* map_y,
                   T* dst, int dst_rows, int dst_cols, int dst_stride,
                   BorderType border_type, T border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int dst_xy = element_y * dst_cols + element_x;
  int dst_index = element_y * dst_stride + element_x * channels;
  float float_x = map_x[dst_xy];
  float float_y = map_y[dst_xy];
  int int_x = __float2int_rn(float_x);
  int int_y = __float2int_rn(float_y);

  if (border_type == BORDER_CONSTANT) {
    int src_index = int_y * src_stride + int_x * channels;
    if (int_x >= 0 && int_x < src_cols && int_y >= 0 && int_y < src_rows) {
      for (int i = 0; i < channels; i++) {
        dst[dst_index + i] = src[src_index + i];
      }
    }
    else {
      for (int i = 0; i < channels; i++) {
        dst[dst_index + i] = border_value;
      }
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    int_x = clip(int_x, 0, src_cols - 1);
    int_y = clip(int_y, 0, src_rows - 1);
    int src_index = int_y * src_stride + int_x * channels;
    for (int i = 0; i < channels; ++i) {
      dst[dst_index + i] = src[src_index + i];
    }
  }
  else if (border_type == BORDER_TRANSPARENT) {
    if (int_x >= 0 && int_x < src_cols && int_y >= 0 && int_y < src_rows) {
      int src_index = int_y * src_stride + int_x * channels;
      for (int i = 0; i < channels; i++) {
        dst[dst_index + i] = src[src_index + i];
      }
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
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
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
    remapLinearKernel<uchar><<<grid, block, 0, stream>>>(src, src_rows,
        src_cols, channels, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
        dst_stride, border_type, border_value);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    remapNPKernel<uchar><<<grid, block, 0, stream>>>(src, src_rows, src_cols,
        channels, src_stride, map_x, map_y, dst, dst_rows, dst_cols, dst_stride,
        border_type, border_value);
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
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
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
    remapLinearKernel<float><<<grid, block, 0, stream>>>(src, src_rows,
        src_cols, channels, src_stride, map_x, map_y, dst, dst_rows, dst_cols,
        dst_stride, border_type, border_value);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    remapNPKernel<float><<<grid, block, 0, stream>>>(src, src_rows, src_cols,
        channels, src_stride, map_x, map_y, dst, dst_rows, dst_cols, dst_stride,
        border_type, border_value);
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
  RetCode code = remap(inData, inHeight, inWidth, 4, inWidthStride, mapX, mapY,
                       outData, outHeight, outWidth, outWidthStride,
                       interpolation, border_type, borderValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
