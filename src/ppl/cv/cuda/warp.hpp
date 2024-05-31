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

#include "utility/utility.hpp"

namespace ppl {
namespace cv {
namespace cuda {

#define TEXTURE_ALIGNMENT 32

static cudaTextureObject_t uchar_c1_tex = 0;
static cudaTextureObject_t uchar_c4_tex = 0;
static cudaTextureObject_t float_c1_tex = 0;
static cudaTextureObject_t float_c4_tex = 0;

template <typename Transform>
__global__
void warpLinearTexKernel(const uchar* src, int src_rows, int src_cols,
                         int channels, int src_stride, Transform transform,
                         uchar* dst, int dst_rows, int dst_cols, int dst_stride,
                         BorderType border_type, uchar border_value) {
  static cudaTextureObject_t uchar_c1_tex = 0;
  static cudaTextureObject_t uchar_c4_tex = 0;
  static cudaTextureObject_t float_c1_tex = 0;
  static cudaTextureObject_t float_c4_tex = 0;
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float2 src_xy = transform.calculateCoordinates(element_x, element_y);
  if (border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) {
    int src_x0 = __float2int_rd(src_xy.x);
    int src_y0 = __float2int_rd(src_xy.y);
    int src_x1 = src_x0 + 1;
    int src_y1 = src_y0 + 1;

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
      if (flag0 && flag1 && flag2 && flag3) {
        float value = tex2D<float>(uchar_c1_tex, src_xy.x + 0.5f, src_xy.y + 0.5f);
        value *= 255.0f;

        uchar* output = (uchar*)(dst + element_y * dst_stride);
        output[element_x] = saturateCast(value);
      }
      else {
        uchar* input = (uchar*)(src + src_y0 * src_stride);
        uchar src_value0 = flag0 ? input[src_x0] : border_value;
        uchar src_value1 = flag1 ? input[src_x1] : border_value;
        float value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
        float value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
        float sum = 0.f;
        sum += value0;
        sum += value1;

        input = (uchar*)(src + src_y1 * src_stride);
        src_value0 = flag2 ? input[src_x0] : border_value;
        src_value1 = flag3 ? input[src_x1] : border_value;
        value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
        value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
        sum += value0;
        sum += value1;

        uchar* output = (uchar*)(dst + element_y * dst_stride);
        output[element_x] = saturateCast(sum);
      }
    }
    else {  // channels == 4
      if (flag0 && flag1 && flag2 && flag3) {
        float4 value = tex2D<float4>(uchar_c4_tex, src_xy.x + 0.5f, src_xy.y + 0.5f);
        value.x *= 255.0f;
        value.y *= 255.0f;
        value.z *= 255.0f;
        value.w *= 255.0f;

        uchar4* output = (uchar4*)(dst + element_y * dst_stride);
        output[element_x] = saturateCastVector<uchar4, float4>(value);
      }
      else {
        uchar4 border_value1 = make_uchar4(border_value, border_value,
                                           border_value, border_value);
        uchar4* input = (uchar4*)(src + src_y0 * src_stride);
        uchar4 src_value0 = flag0 ? input[src_x0] : border_value1;
        uchar4 src_value1 = flag1 ? input[src_x1] : border_value1;
        float4 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
        float4 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        sum += value0;
        sum += value1;

        input = (uchar4*)(src + src_y1 * src_stride);
        src_value0 = flag2 ? input[src_x0] : border_value1;
        src_value1 = flag3 ? input[src_x1] : border_value1;
        value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
        value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
        sum += value0;
        sum += value1;

        uchar4* output = (uchar4*)(dst + element_y * dst_stride);
        output[element_x] = saturateCastVector<uchar4, float4>(sum);
      }
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    if (channels == 1) {
      float value = tex2D<float>(uchar_c1_tex, src_xy.x + 0.5f, src_xy.y + 0.5f);
      value *= 255.0f;

      uchar* output = (uchar*)(dst + element_y * dst_stride);
      output[element_x] = saturateCast(value);
    }
    else {  // channels == 4
      float4 value = tex2D<float4>(uchar_c4_tex, src_xy.x + 0.5f, src_xy.y + 0.5f);
      value.x *= 255.0f;
      value.y *= 255.0f;
      value.z *= 255.0f;
      value.w *= 255.0f;

      uchar4* output = (uchar4*)(dst + element_y * dst_stride);
      output[element_x] = saturateCastVector<uchar4, float4>(value);
    }
  }
  else {
  }
}

template <typename Transform>
__global__
void warpLinearTexKernel(const float* src, int src_rows, int src_cols,
                         int channels, int src_stride, Transform transform,
                         float* dst, int dst_rows, int dst_cols, int dst_stride,
                         BorderType border_type, float border_value) {
  static cudaTextureObject_t uchar_c1_tex = 0;
  static cudaTextureObject_t uchar_c4_tex = 0;
  static cudaTextureObject_t float_c1_tex = 0;
  static cudaTextureObject_t float_c4_tex = 0;
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float2 src_xy = transform.calculateCoordinates(element_x, element_y);
  if (border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) {
    int src_x0 = __float2int_rd(src_xy.x);
    int src_y0 = __float2int_rd(src_xy.y);
    int src_x1 = src_x0 + 1;
    int src_y1 = src_y0 + 1;

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
      if (flag0 && flag1 && flag2 && flag3) {
        float value = tex2D<float>(float_c1_tex, src_xy.x + 0.5f, src_xy.y + 0.5f);

        float* output = (float*)((uchar*)dst + element_y * dst_stride);
        output[element_x] = value;
      }
      else {
        float* input = (float*)((uchar*)src + src_y0 * src_stride);
        float src_value0 = flag0 ? input[src_x0] : border_value;
        float src_value1 = flag1 ? input[src_x1] : border_value;
        float value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
        float value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
        float sum = 0.f;
        sum += value0;
        sum += value1;

        input = (float*)((uchar*)src + src_y1 * src_stride);
        src_value0 = flag2 ? input[src_x0] : border_value;
        src_value1 = flag3 ? input[src_x1] : border_value;
        value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
        value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
        sum += value0;
        sum += value1;

        float* output = (float*)((uchar*)dst + element_y * dst_stride);
        output[element_x] = sum;
      }
    }
    else {  // channels == 4
      if (flag0 && flag1 && flag2 && flag3) {
        float4 value = tex2D<float4>(float_c4_tex, src_xy.x + 0.5f, src_xy.y + 0.5f);

        float4* output = (float4*)((uchar*)dst + element_y * dst_stride);
        output[element_x] = value;
      }
      else {
        float4 border_value1 = make_float4(border_value, border_value,
                                           border_value, border_value);
        float4* input = (float4*)((uchar*)src + src_y0 * src_stride);
        float4 src_value0 = flag0 ? input[src_x0] : border_value1;
        float4 src_value1 = flag1 ? input[src_x1] : border_value1;
        float4 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
        float4 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        sum += value0;
        sum += value1;

        input = (float4*)((uchar*)src + src_y1 * src_stride);
        src_value0 = flag2 ? input[src_x0] : border_value1;
        src_value1 = flag3 ? input[src_x1] : border_value1;
        value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
        value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
        sum += value0;
        sum += value1;

        float4* output = (float4*)((uchar*)dst + element_y * dst_stride);
        output[element_x] = sum;
      }
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    if (channels == 1) {
      float value = tex2D<float>(float_c1_tex, src_xy.x + 0.5f, src_xy.y + 0.5f);

      float* output = (float*)((uchar*)dst + element_y * dst_stride);
      output[element_x] = value;
    }
    else {  // channels == 4
      float4 value = tex2D<float4>(float_c4_tex, src_xy.x + 0.5f, src_xy.y + 0.5f);

      float4* output = (float4*)((uchar*)dst + element_y * dst_stride);
      output[element_x] = value;
    }
  }
  else {
  }
}

template <typename Transform>
__global__
void warpLinearKernel(const uchar* src, int src_rows, int src_cols,
                      int channels, int src_stride, Transform transform,
                      uchar* dst, int dst_rows, int dst_cols, int dst_stride,
                      BorderType border_type, uchar border_value) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float2 src_xy = transform.calculateCoordinates(element_x, element_y);
  int src_x0 = __float2int_rd(src_xy.x);
  int src_y0 = __float2int_rd(src_xy.y);
  int src_x1 = src_x0 + 1;
  int src_y1 = src_y0 + 1;

  if (border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) {
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
      float value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      input = (uchar*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value;
      src_value1 = flag3 ? input[src_x1] : border_value;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
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
      float3 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float3 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (uchar3*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value1;
      src_value1 = flag3 ? input[src_x1] : border_value1;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      uchar3* output = (uchar3*)(dst + element_y * dst_stride);
      if (src_xy.x > src_cols - 1 || src_xy.y > src_rows - 1) {
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
      float4 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float4 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (uchar4*)(src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value1;
      src_value1 = flag3 ? input[src_x1] : border_value1;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      uchar4* output = (uchar4*)(dst + element_y * dst_stride);
      output[element_x] = saturateCastVector<uchar4, float4>(sum);
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    float diff_x0 = src_xy.x - src_x0;
    float diff_x1 = src_x1 - src_xy.x;
    float diff_y0 = src_xy.y - src_y0;
    float diff_y1 = src_y1 - src_xy.y;

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

template <typename Transform>
__global__
void warpLinearKernel(const float* src, int src_rows, int src_cols,
                      int channels, int src_stride, Transform transform,
                      float* dst, int dst_rows, int dst_cols, int dst_stride,
                      BorderType border_type, float border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float2 src_xy = transform.calculateCoordinates(element_x, element_y);
  int src_x0 = __float2int_rd(src_xy.x);
  int src_y0 = __float2int_rd(src_xy.y);
  int src_x1 = src_x0 + 1;
  int src_y1 = src_y0 + 1;

  if (border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) {
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
      float* input = (float*)((uchar*)src + src_y0 * src_stride);
      float src_value0 = flag0 ? input[src_x0] : border_value;
      float src_value1 = flag1 ? input[src_x1] : border_value;
      float value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      input = (float*)((uchar*)src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value;
      src_value1 = flag3 ? input[src_x1] : border_value;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      float* output = (float*)((uchar*)dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else if (channels == 3) {
      float3 border_value1 = make_float3(border_value, border_value,
                                         border_value);
      float3* input = (float3*)((uchar*)src + src_y0 * src_stride);
      float3 src_value0 = flag0 ? input[src_x0] : border_value1;
      float3 src_value1 = flag1 ? input[src_x1] : border_value1;
      float3 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float3 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (float3*)((uchar*)src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value1;
      src_value1 = flag3 ? input[src_x1] : border_value1;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      float3* output = (float3*)((uchar*)dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else {
      float4 border_value1 = make_float4(border_value, border_value,
                                         border_value, border_value);
      float4* input = (float4*)((uchar*)src + src_y0 * src_stride);
      float4 src_value0 = flag0 ? input[src_x0] : border_value1;
      float4 src_value1 = flag1 ? input[src_x1] : border_value1;
      float4 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float4 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (float4*)((uchar*)src + src_y1 * src_stride);
      src_value0 = flag2 ? input[src_x0] : border_value1;
      src_value1 = flag3 ? input[src_x1] : border_value1;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      float4* output = (float4*)((uchar*)dst + element_y * dst_stride);
      output[element_x] = sum;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    float diff_x0 = src_xy.x - src_x0;
    float diff_x1 = src_x1 - src_xy.x;
    float diff_y0 = src_xy.y - src_y0;
    float diff_y1 = src_y1 - src_xy.y;

    src_x0 = clip(src_x0, 0, src_cols - 1);
    src_y0 = clip(src_y0, 0, src_rows - 1);
    src_x1 = clip(src_x1, 0, src_cols - 1);
    src_y1 = clip(src_y1, 0, src_rows - 1);

    if (channels == 1) {
      float* input = (float*)((uchar*)src + src_y0 * src_stride);
      float src_value0 = input[src_x0];
      float src_value1 = input[src_x1];
      float value0 = diff_x1 * diff_y1 * src_value0;
      float value1 = diff_x0 * diff_y1 * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      input = (float*)((uchar*)src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      float* output = (float*)((uchar*)dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else if (channels == 3) {
      float3* input = (float3*)((uchar*)src + src_y0 * src_stride);
      float3 src_value0 = input[src_x0];
      float3 src_value1 = input[src_x1];
      float3 value0 = diff_x1 * diff_y1 * src_value0;
      float3 value1 = diff_x0 * diff_y1 * src_value1;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (float3*)((uchar*)src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      float3* output = (float3*)((uchar*)dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else {
      float4* input = (float4*)((uchar*)src + src_y0 * src_stride);
      float4 src_value0 = input[src_x0];
      float4 src_value1 = input[src_x1];
      float4 value0 = diff_x1 * diff_y1 * src_value0;
      float4 value1 = diff_x0 * diff_y1 * src_value1;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      input = (float4*)((uchar*)src + src_y1 * src_stride);
      src_value0 = input[src_x0];
      src_value1 = input[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      float4* output = (float4*)((uchar*)dst + element_y * dst_stride);
      output[element_x] = sum;
    }
  }
  else {
  }
}

template <typename T, typename Tn, typename Transform>
__global__
void warpNearestKernel(const T* src, int src_rows, int src_cols, int channels,
                       int src_stride, Transform transform, T* dst,
                       int dst_rows, int dst_cols, int dst_stride,
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
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float2 src_xy = transform.calculateCoordinates(element_x, element_y);
  int src_x = src_xy.x;
  int src_y = src_xy.y;

  if (border_type == BORDER_CONSTANT) {
    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);

    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      Tn* input = (Tn*)((uchar*)src + src_y * src_stride);
      output[element_x] = input[src_x];
    }
    else {
      output[element_x] = border_value;
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = clip(src_x, 0, src_cols - 1);
    src_y = clip(src_y, 0, src_rows - 1);

    Tn* input  = (Tn*)((uchar*)src + src_y * src_stride);
    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = input[src_x];
  }
  else if (border_type == BORDER_TRANSPARENT) {
    Tn* output = (Tn*)((uchar*)dst + element_y * dst_stride);

    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      Tn* input = (Tn*)((uchar*)src + src_y * src_stride);
      output[element_x] = input[src_x];
    }
  }
  else {
  }
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
