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

#include "ppl/cv/cuda/warpperspective.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

__global__
void warpPerspectiveLinearKernel(const uchar* src, int src_rows, int src_cols,
                                 int channels, int src_stride, float coeffe0,
                                 float coeffe1, float coeffe2, float coeffe3,
                                 float coeffe4, float coeffe5, float coeffe6,
                                 float coeffe7, float coeffe8, uchar* dst,
                                 int dst_rows, int dst_cols, int dst_stride,
                                 BorderType border_type, uchar border_value) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float src_x  = coeffe0 * element_x + coeffe1 * element_y + coeffe2;
  float src_y  = coeffe3 * element_x + coeffe4 * element_y + coeffe5;
  float weight = coeffe6 * element_x + coeffe7 * element_y + coeffe8;
  src_x /= weight;
  src_y /= weight;

  int src_x0 = __float2int_rd(src_x);
  int src_y0 = __float2int_rd(src_y);
  int src_x1 = src_x0 + 1;
  int src_y1 = src_y0 + 1;

  if (border_type == BORDER_CONSTANT ||
      border_type == BORDER_TRANSPARENT) {
    bool flag0 = src_y0 >= 0 && src_y0 < src_rows && src_x0 >= 0 &&
                 src_x0 < src_cols;
    bool flag1 = src_y0 >= 0 && src_y0 < src_rows && src_x1 >= 0 &&
                 src_x1 < src_cols;
    bool flag2 = src_y1 >= 0 && src_y1 < src_rows && src_x0 >= 0 &&
                 src_x0 < src_cols;
    bool flag3 = src_y1 >= 0 && src_y1 < src_rows && src_x1 >= 0 &&
                 src_x1 < src_cols;

    if ((border_type == BORDER_TRANSPARENT) &&
        ((!flag0) || (!flag1) || (!flag2) || (!flag3))) {
      return;
    }

    if (channels == 1) {
      uchar* input = (uchar*)(src + src_y0 * src_stride);
      uchar src_value0 = flag0 ? input[src_x0] : border_value;
      uchar src_value1 = flag1 ? input[src_x1] : border_value;
      float value0 = (src_x1 - src_x) * (src_y1 - src_y) * src_value0;
      float value1 = (src_x - src_x0) * (src_y1 - src_y) * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      input = (uchar*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value;
      src_value1 = flag3 ? input[src_x1] : border_value;
      value0 = (src_x1 - src_x) * (src_y - src_y0) * src_value0;
      value1 = (src_x - src_x0) * (src_y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      uchar* output = (uchar*)(dst + element_y * dst_stride);
      output[element_x] = saturateCast(sum);
    }
    else if (channels == 3) {
      uchar3 border_value1 = make_uchar3(border_value, border_value,
                                         border_value);
      uchar3* input = (uchar3*)(src + src_y0 * src_stride);
      uchar3 src_value0 = flag0 ? input[src_x0] : border_value1;
      uchar3 src_value1 = flag1 ? input[src_x1] : border_value1;
      float3 value0 = (src_x1 - src_x) * (src_y1 - src_y) * src_value0;
      float3 value1 = (src_x - src_x0) * (src_y1 - src_y) * src_value1;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (uchar3*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value1;
      src_value1 = flag3 ? input[src_x1] : border_value1;
      value0 = (src_x1 - src_x) * (src_y - src_y0) * src_value0;
      value1 = (src_x - src_x0) * (src_y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      uchar3* output = (uchar3*)(dst + element_y * dst_stride);
      if (src_x > src_cols - 1 || src_y > src_rows - 1) {
        output[element_x] = border_value1;  // align with npp.
      }
      else {
        output[element_x] = saturateCastVector<uchar3, float3>(sum);
      }
    }
    else {
      uchar4 border_value1 = make_uchar4(border_value, border_value,
                                         border_value, border_value);
      uchar4* input = (uchar4*)(src + src_y0 * src_stride);
      uchar4 src_value0 = flag0 ? input[src_x0] : border_value1;
      uchar4 src_value1 = flag1 ? input[src_x1] : border_value1;
      float4 value0 = (src_x1 - src_x) * (src_y1 - src_y) * src_value0;
      float4 value1 = (src_x - src_x0) * (src_y1 - src_y) * src_value1;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (uchar4*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value1;
      src_value1 = flag3 ? input[src_x1] : border_value1;
      value0 = (src_x1 - src_x) * (src_y - src_y0) * src_value0;
      value1 = (src_x - src_x0) * (src_y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      uchar4* output = (uchar4*)(dst + element_y * dst_stride);
      output[element_x] = saturateCastVector<uchar4, float4>(sum);
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    float diff_x0 = src_x - src_x0;
    float diff_x1 = src_x1 - src_x;
    float diff_y0 = src_y - src_y0;
    float diff_y1 = src_y1 - src_y;

    src_x0 = clip(src_x0, 0, src_cols - 1);
    src_y0 = clip(src_y0, 0, src_rows - 1);
    src_x1 = clip(src_x1, 0, src_cols - 1);
    src_y1 = clip(src_y1, 0, src_rows - 1);

    if (channels == 1) {
      uchar* input = (uchar*)(src + src_y0 * src_stride);
      uchar src_value0 = input[src_x0];
      uchar src_value1 = input[src_x1];
      float value0 = diff_x1 * diff_y1 * src_value0;
      float value1 = diff_x0 * diff_y1 * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      input = (uchar*)(src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      uchar* output = (uchar*)(dst + element_y * dst_stride);
      output[element_x] = saturateCast(sum);
    }
    else if (channels == 3) {
      uchar3* input = (uchar3*)(src + src_y0 * src_stride);
      uchar3 src_value0 = input[src_x0];
      uchar3 src_value1 = input[src_x1];
      float3 value0 = diff_x1 * diff_y1 * src_value0;
      float3 value1 = diff_x0 * diff_y1 * src_value1;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (uchar3*)(src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      uchar3* output = (uchar3*)(dst + element_y * dst_stride);
      output[element_x] = saturateCastVector<uchar3, float3>(sum);
    }
    else {
      uchar4* input = (uchar4*)(src + src_y0 * src_stride);
      uchar4 src_value0 = input[src_x0];
      uchar4 src_value1 = input[src_x1];
      float4 value0 = diff_x1 * diff_y1 * src_value0;
      float4 value1 = diff_x0 * diff_y1 * src_value1;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (uchar4*)(src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      uchar4* output = (uchar4*)(dst + element_y * dst_stride);
      output[element_x] = saturateCastVector<uchar4, float4>(sum);
    }
  }
  else {
  }
}

__global__
void warpPerspectiveLinearKernel(const float* src, int src_rows, int src_cols,
                                 int channels, int src_stride, float coeffe0,
                                 float coeffe1, float coeffe2, float coeffe3,
                                 float coeffe4, float coeffe5, float coeffe6,
                                 float coeffe7, float coeffe8, float* dst,
                                 int dst_rows, int dst_cols, int dst_stride,
                                 BorderType border_type, float border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float src_x  = coeffe0 * element_x + coeffe1 * element_y + coeffe2;
  float src_y  = coeffe3 * element_x + coeffe4 * element_y + coeffe5;
  float weight = coeffe6 * element_x + coeffe7 * element_y + coeffe8;
  src_x /= weight;
  src_y /= weight;

  int src_x0 = __float2int_rd(src_x);
  int src_y0 = __float2int_rd(src_y);
  int src_x1 = src_x0 + 1;
  int src_y1 = src_y0 + 1;

  if (border_type == BORDER_CONSTANT ||
      border_type == BORDER_TRANSPARENT) {
    bool flag0 = src_y0 >= 0 && src_y0 < src_rows && src_x0 >= 0 &&
                 src_x0 < src_cols;
    bool flag1 = src_y0 >= 0 && src_y0 < src_rows && src_x1 >= 0 &&
                 src_x1 < src_cols;
    bool flag2 = src_y1 >= 0 && src_y1 < src_rows && src_x0 >= 0 &&
                 src_x0 < src_cols;
    bool flag3 = src_y1 >= 0 && src_y1 < src_rows && src_x1 >= 0 &&
                 src_x1 < src_cols;

    if ((border_type == BORDER_TRANSPARENT) &&
        ((!flag0) || (!flag1) || (!flag2) || (!flag3))) {
      return;
    }

    if (channels == 1) {
      float* input = (float*)(src + src_y0 * src_stride);
      float src_value0 = flag0 ? input[src_x0] : border_value;
      float src_value1 = flag1 ? input[src_x1] : border_value;
      float value0 = (src_x1 - src_x) * (src_y1 - src_y) * src_value0;
      float value1 = (src_x - src_x0) * (src_y1 - src_y) * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      input = (float*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value;
      src_value1 = flag3 ? input[src_x1] : border_value;
      value0 = (src_x1 - src_x) * (src_y - src_y0) * src_value0;
      value1 = (src_x - src_x0) * (src_y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      float* output = (float*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else if (channels == 3) {
      float3 border_value1 = make_float3(border_value, border_value,
                                         border_value);
      float3* input = (float3*)(src + src_y0 * src_stride);
      float3 src_value0 = flag0 ? input[src_x0] : border_value1;
      float3 src_value1 = flag1 ? input[src_x1] : border_value1;
      float3 value0 = (src_x1 - src_x) * (src_y1 - src_y) * src_value0;
      float3 value1 = (src_x - src_x0) * (src_y1 - src_y) * src_value1;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (float3*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value1;
      src_value1 = flag3 ? input[src_x1] : border_value1;
      value0 = (src_x1 - src_x) * (src_y - src_y0) * src_value0;
      value1 = (src_x - src_x0) * (src_y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      float3* output = (float3*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else {
      float4 border_value1 = make_float4(border_value, border_value,
                                         border_value, border_value);
      float4* input = (float4*)(src + src_y0 * src_stride);
      float4 src_value0 = flag0 ? input[src_x0] : border_value1;
      float4 src_value1 = flag1 ? input[src_x1] : border_value1;
      float4 value0 = (src_x1 - src_x) * (src_y1 - src_y) * src_value0;
      float4 value1 = (src_x - src_x0) * (src_y1 - src_y) * src_value1;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (float4*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value1;
      src_value1 = flag3 ? input[src_x1] : border_value1;
      value0 = (src_x1 - src_x) * (src_y - src_y0) * src_value0;
      value1 = (src_x - src_x0) * (src_y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      float4* output = (float4*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    float diff_x0 = src_x - src_x0;
    float diff_x1 = src_x1 - src_x;
    float diff_y0 = src_y - src_y0;
    float diff_y1 = src_y1 - src_y;

    src_x0 = clip(src_x0, 0, src_cols - 1);
    src_y0 = clip(src_y0, 0, src_rows - 1);
    src_x1 = clip(src_x1, 0, src_cols - 1);
    src_y1 = clip(src_y1, 0, src_rows - 1);

    if (channels == 1) {
      float* input = (float*)(src + src_y0 * src_stride);
      float src_value0 = input[src_x0];
      float src_value1 = input[src_x1];
      float value0 = diff_x1 * diff_y1 * src_value0;
      float value1 = diff_x0 * diff_y1 * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      input = (float*)(src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      float* output = (float*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else if (channels == 3) {
      float3* input = (float3*)(src + src_y0 * src_stride);
      float3 src_value0 = input[src_x0];
      float3 src_value1 = input[src_x1];
      float3 value0 = diff_x1 * diff_y1 * src_value0;
      float3 value1 = diff_x0 * diff_y1 * src_value1;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (float3*)(src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      float3* output = (float3*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else {
      float4* input = (float4*)(src + src_y0 * src_stride);
      float4 src_value0 = input[src_x0];
      float4 src_value1 = input[src_x1];
      float4 value0 = diff_x1 * diff_y1 * src_value0;
      float4 value1 = diff_x0 * diff_y1 * src_value1;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (float4*)(src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      float4* output = (float4*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
  }
  else {
  }
}

template <typename T, typename Tn>
__global__
void warpPerspectiveNPKernel(const T* src, int src_rows, int src_cols,
                             int channels, int src_stride, float coeffe0,
                             float coeffe1, float coeffe2, float coeffe3,
                             float coeffe4, float coeffe5, float coeffe6,
                             float coeffe7, float coeffe8, T* dst, int dst_rows,
                             int dst_cols, int dst_stride,
                             BorderType border_type, Tn border_value) {
  int element_x, element_y;
  if (sizeof(T) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float src_x_float = coeffe0 * element_x + coeffe1 * element_y + coeffe2;
  float src_y_float = coeffe3 * element_x + coeffe4 * element_y + coeffe5;
  float weight      = coeffe6 * element_x + coeffe7 * element_y + coeffe8;
  src_x_float /= weight;
  src_y_float /= weight;

  int src_x = src_x_float;
  int src_y = src_y_float;

  if (border_type == BORDER_CONSTANT ||
    border_type == BORDER_TRANSPARENT) {
    Tn* output = (Tn*)(dst + element_y * dst_stride);

    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      Tn* input  = (Tn*)(src + src_y * src_stride);
      output[element_x] = input[src_x];
    }
    else {
      output[element_x] = border_value;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = clip(src_x, 0, src_cols - 1);
    src_y = clip(src_y, 0, src_rows - 1);

    Tn* input  = (Tn*)(src + src_y * src_stride);
    Tn* output = (Tn*)(dst + element_y * dst_stride);
    output[element_x] = input[src_x];
  }
  else {
  }
}

RetCode warpPerspective(const uchar* src, int src_rows, int src_cols,
                        int channels, int src_stride,
                        const float* affine_matrix, uchar* dst, int dst_rows,
                        int dst_cols, int dst_stride,
                        InterpolationType interpolation, BorderType border_type,
                        uchar border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src != dst);
  PPL_ASSERT(affine_matrix != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(dst_cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(dst_rows, kBlockDimY0, kBlockShiftY0);

  if (interpolation == INTERPOLATION_LINEAR) {
    warpPerspectiveLinearKernel<<<grid, block, 0, stream>>>(src, src_rows,
        src_cols, channels, src_stride, affine_matrix[0], affine_matrix[1],
        affine_matrix[2], affine_matrix[3], affine_matrix[4], affine_matrix[5],
        affine_matrix[6], affine_matrix[7], affine_matrix[8], dst, dst_rows,
        dst_cols, dst_stride, border_type, border_value);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      warpPerspectiveNPKernel<uchar, uchar><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, affine_matrix[0],
          affine_matrix[1], affine_matrix[2], affine_matrix[3],
          affine_matrix[4], affine_matrix[5], affine_matrix[6],
          affine_matrix[7], affine_matrix[8], dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else if (channels == 3) {
      uchar3 border_value1 = make_uchar3(border_value, border_value,
                                         border_value);
      warpPerspectiveNPKernel<uchar, uchar3><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, affine_matrix[0],
          affine_matrix[1], affine_matrix[2], affine_matrix[3],
          affine_matrix[4], affine_matrix[5], affine_matrix[6],
          affine_matrix[7], affine_matrix[8], dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value1);
    }
    else {
      uchar4 border_value1 = make_uchar4(border_value, border_value,
                                         border_value, border_value);
      warpPerspectiveNPKernel<uchar, uchar4><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, affine_matrix[0],
          affine_matrix[1], affine_matrix[2], affine_matrix[3],
          affine_matrix[4], affine_matrix[5], affine_matrix[6],
          affine_matrix[7], affine_matrix[8], dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value1);
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

RetCode warpPerspective(const float* src, int src_rows, int src_cols,
                        int channels, int src_stride,
                        const float* affine_matrix, float* dst, int dst_rows,
                        int dst_cols, int dst_stride,
                        InterpolationType interpolation, BorderType border_type,
                        float border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src != dst);
  PPL_ASSERT(affine_matrix != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
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
    warpPerspectiveLinearKernel<<<grid, block, 0, stream>>>(src, src_rows,
        src_cols, channels, src_stride, affine_matrix[0], affine_matrix[1],
        affine_matrix[2], affine_matrix[3], affine_matrix[4], affine_matrix[5],
        affine_matrix[6], affine_matrix[7], affine_matrix[8], dst, dst_rows,
        dst_cols, dst_stride, border_type, border_value);
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      warpPerspectiveNPKernel<float, float><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, affine_matrix[0],
          affine_matrix[1], affine_matrix[2], affine_matrix[3],
          affine_matrix[4], affine_matrix[5], affine_matrix[6],
          affine_matrix[7], affine_matrix[8], dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value);
    }
    else if (channels == 3) {
      float3 border_value1 = make_float3(border_value, border_value,
                                         border_value);
      warpPerspectiveNPKernel<float, float3><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, affine_matrix[0],
          affine_matrix[1], affine_matrix[2], affine_matrix[3],
          affine_matrix[4], affine_matrix[5], affine_matrix[6],
          affine_matrix[7], affine_matrix[8], dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value1);
    }
    else {
      float4 border_value1 = make_float4(border_value, border_value,
                                         border_value, border_value);
      warpPerspectiveNPKernel<float, float4><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, affine_matrix[0],
          affine_matrix[1], affine_matrix[2], affine_matrix[3],
          affine_matrix[4], affine_matrix[5], affine_matrix[6],
          affine_matrix[7], affine_matrix[8], dst, dst_rows, dst_cols,
          dst_stride, border_type, border_value1);
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
RetCode WarpPerspective<uchar, 1>(cudaStream_t stream,
                                  int inHeight,
                                  int inWidth,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  uchar* outData,
                                  const float* affineMatrix,
                                  InterpolationType interpolation,
                                  BorderType borderType,
                                  uchar borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 1, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<uchar, 3>(cudaStream_t stream,
                                  int inHeight,
                                  int inWidth,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  uchar* outData,
                                  const float* affineMatrix,
                                  InterpolationType interpolation,
                                  BorderType borderType,
                                  uchar borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 3, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<uchar, 4>(cudaStream_t stream,
                                  int inHeight,
                                  int inWidth,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  uchar* outData,
                                  const float* affineMatrix,
                                  InterpolationType interpolation,
                                  BorderType borderType,
                                  uchar borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 4, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<float, 1>(cudaStream_t stream,
                                  int inHeight,
                                  int inWidth,
                                  int inWidthStride,
                                  const float* inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  float* outData,
                                  const float* affineMatrix,
                                  InterpolationType interpolation,
                                  BorderType borderType,
                                  float borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 1, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<float, 3>(cudaStream_t stream,
                                  int inHeight,
                                  int inWidth,
                                  int inWidthStride,
                                  const float* inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  float* outData,
                                  const float* affineMatrix,
                                  InterpolationType interpolation,
                                  BorderType borderType,
                                  float borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 3, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<float, 4>(cudaStream_t stream,
                                  int inHeight,
                                  int inWidth,
                                  int inWidthStride,
                                  const float* inData,
                                  int outHeight,
                                  int outWidth,
                                  int outWidthStride,
                                  float* outData,
                                  const float* affineMatrix,
                                  InterpolationType interpolation,
                                  BorderType borderType,
                                  float borderValue) {
  RetCode code = warpPerspective(inData, inHeight, inWidth, 4, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
