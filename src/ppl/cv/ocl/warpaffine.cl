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

#include "kerneltypes.h"

#define AB_BITS 10
#define AB_SCALE 1024
#define ROUND_DELTA 512

#if defined(WARPAFFINE_LINEAR_U8) || defined(ALL_KERNELS)
__kernel
void warpaffineLinearU8Kernel(global const uchar* src, int src_rows,
         int src_cols, int channels, int src_stride, float coeff0, float coeff1,
         float coeff2, float coeff3, float coeff4, float coeff5,
         global uchar* dst, int dst_rows, int dst_cols, int dst_stride,
         enum BorderType border_type, uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float2 src_xy;
  src_xy.x = coeff0 * element_x + coeff1 * element_y + coeff2;
  src_xy.y = coeff3 * element_x + coeff4 * element_y + coeff5;
  int src_x0 = floor(src_xy.x);
  int src_y0 = floor(src_xy.y);
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
      global uchar* data = src + src_y0 * src_stride;
      uchar src_value0 = flag0 ? data[src_x0] : border_value;
      uchar src_value1 = flag1 ? data[src_x1] : border_value;
      float value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      data = src + src_y1 * src_stride;
      src_value0 = flag2 ? data[src_x0] : border_value;
      src_value1 = flag3 ? data[src_x1] : border_value;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      data = dst + element_y * dst_stride;
      data[element_x] = convert_uchar_sat(sum);
    }
    else if (channels == 3) {
      uchar3 border_value1 = (uchar3)(border_value, border_value, border_value);
      global uchar* data = src + src_y0 * src_stride;
      uchar3 src_value0 = flag0 ? vload3(src_x0, data) : border_value1;
      uchar3 src_value1 = flag1 ? vload3(src_x1, data) : border_value1;
      float3 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) *
                      convert_float3(src_value0);
      float3 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) *
                      convert_float3(src_value1);
      float3 sum = (float3)(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      data = src + src_y1 * src_stride;
      src_value0 = flag2 ? vload3(src_x0, data) : border_value1;
      src_value1 = flag3 ? vload3(src_x1, data) : border_value1;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) *
               convert_float3(src_value0);
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) *
               convert_float3(src_value1);
      sum += value0;
      sum += value1;

      data = dst + element_y * dst_stride;
      uchar3 result = convert_uchar3_sat(sum);
      vstore3(result, element_x, data);
    }
    else {
      uchar4 border_value1 = (uchar4)(border_value, border_value, border_value,
                                      border_value);
      global uchar* data = src + src_y0 * src_stride;
      uchar4 src_value0 = flag0 ? vload4(src_x0, data) : border_value1;
      uchar4 src_value1 = flag1 ? vload4(src_x1, data) : border_value1;
      float4 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) *
                      convert_float4(src_value0);
      float4 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) *
                      convert_float4(src_value1);
      float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      data = src + src_y1 * src_stride;
      src_value0 = flag2 ? vload4(src_x0, data) : border_value1;
      src_value1 = flag3 ? vload4(src_x1, data) : border_value1;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) *
               convert_float4(src_value0);
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) *
               convert_float4(src_value1);
      sum += value0;
      sum += value1;

      data = dst + element_y * dst_stride;
      uchar4 result = convert_uchar4_sat(sum);
      vstore4(result, element_x, data);
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    float diff_x0 = src_xy.x - src_x0;
    float diff_x1 = src_x1 - src_xy.x;
    float diff_y0 = src_xy.y - src_y0;
    float diff_y1 = src_y1 - src_xy.y;

    src_x0 = clamp(src_x0, 0, src_cols - 1);
    src_y0 = clamp(src_y0, 0, src_rows - 1);
    src_x1 = clamp(src_x1, 0, src_cols - 1);
    src_y1 = clamp(src_y1, 0, src_rows - 1);

    if (channels == 1) {
      global uchar* data = src + src_y0 * src_stride;
      uchar src_value0 = data[src_x0];
      uchar src_value1 = data[src_x1];
      float value0 = diff_x1 * diff_y1 * src_value0;
      float value1 = diff_x0 * diff_y1 * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      data = src + src_y1 * src_stride;
      src_value0 = data[src_x0];
      src_value1 = data[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      data = dst + element_y * dst_stride;
      data[element_x] = convert_uchar_sat(sum);
    }
    else if (channels == 3) {
      global uchar* data = src + src_y0 * src_stride;
      uchar3 src_value0 = vload3(src_x0, data);
      uchar3 src_value1 = vload3(src_x1, data);
      float3 value0 = diff_x1 * diff_y1 * convert_float3(src_value0);
      float3 value1 = diff_x0 * diff_y1 * convert_float3(src_value1);
      float3 sum = (float3)(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      data = src + src_y1 * src_stride;
      src_value0 = vload3(src_x0, data);
      src_value1 = vload3(src_x1, data);
      value0 = diff_x1 * diff_y0 * convert_float3(src_value0);
      value1 = diff_x0 * diff_y0 * convert_float3(src_value1);
      sum += value0;
      sum += value1;

      data = dst + element_y * dst_stride;
      uchar3 result = convert_uchar3_sat(sum);
      vstore3(result, element_x, data);
    }
    else {
      global uchar* data = src + src_y0 * src_stride;
      uchar4 src_value0 = vload4(src_x0, data);
      uchar4 src_value1 = vload4(src_x1, data);
      float4 value0 = diff_x1 * diff_y1 * convert_float4(src_value0);
      float4 value1 = diff_x0 * diff_y1 * convert_float4(src_value1);
      float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      data = src + src_y1 * src_stride;
      src_value0 = vload4(src_x0, data);
      src_value1 = vload4(src_x1, data);
      value0 = diff_x1 * diff_y0 * convert_float4(src_value0);
      value1 = diff_x0 * diff_y0 * convert_float4(src_value1);
      sum += value0;
      sum += value1;

      data = dst + element_y * dst_stride;
      uchar4 result = convert_uchar4_sat(sum);
      vstore4(result, element_x, data);
    }
  }
  else {
  }
}
#endif

#if defined(WARPAFFINE_LINEAR_F32) || defined(ALL_KERNELS)
__kernel
void warpaffineLinearF32Kernel(global const float* src, int src_rows,
         int src_cols, int channels, int src_stride, float coeff0, float coeff1,
         float coeff2, float coeff3, float coeff4, float coeff5,
         global float* dst, int dst_rows, int dst_cols, int dst_stride,
         enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float2 src_xy;
  src_xy.x = coeff0 * element_x + coeff1 * element_y + coeff2;
  src_xy.y = coeff3 * element_x + coeff4 * element_y + coeff5;
  int src_x0 = floor(src_xy.x);
  int src_y0 = floor(src_xy.y);
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
      global float* data = (global float*)((global uchar*)src +
                                            src_y0 * src_stride);
      float src_value0 = flag0 ? data[src_x0] : border_value;
      float src_value1 = flag1 ? data[src_x1] : border_value;
      float value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)src + src_y1 * src_stride);
      src_value0 = flag2 ? data[src_x0] : border_value;
      src_value1 = flag3 ? data[src_x1] : border_value;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      data[element_x] = sum;
    }
    else if (channels == 3) {
      float3 border_value1 = (float3)(border_value, border_value, border_value);
      global float* data = (global float*)((global uchar*)src +
                                            src_y0 * src_stride);
      float3 src_value0 = flag0 ? vload3(src_x0, data) : border_value1;
      float3 src_value1 = flag1 ? vload3(src_x1, data) : border_value1;
      float3 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float3 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float3 sum = (float3)(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)src + src_y1 * src_stride);
      src_value0 = flag2 ? vload3(src_x0, data) : border_value1;
      src_value1 = flag3 ? vload3(src_x1, data) : border_value1;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      vstore3(sum, element_x, data);
    }
    else {
      float4 border_value1 = (float4)(border_value, border_value, border_value,
                                      border_value);
      global float* data = (global float*)((global uchar*)src +
                                            src_y0 * src_stride);
      float4 src_value0 = flag0 ? vload4(src_x0, data) : border_value1;
      float4 src_value1 = flag1 ? vload4(src_x1, data) : border_value1;
      float4 value0 = (src_x1 - src_xy.x) * (src_y1 - src_xy.y) * src_value0;
      float4 value1 = (src_xy.x - src_x0) * (src_y1 - src_xy.y) * src_value1;
      float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)src + src_y1 * src_stride);
      src_value0 = flag2 ? vload4(src_x0, data) : border_value1;
      src_value1 = flag3 ? vload4(src_x1, data) : border_value1;
      value0 = (src_x1 - src_xy.x) * (src_xy.y - src_y0) * src_value0;
      value1 = (src_xy.x - src_x0) * (src_xy.y - src_y0) * src_value1;
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      vstore4(sum, element_x, data);
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    float diff_x0 = src_xy.x - src_x0;
    float diff_x1 = src_x1 - src_xy.x;
    float diff_y0 = src_xy.y - src_y0;
    float diff_y1 = src_y1 - src_xy.y;

    src_x0 = clamp(src_x0, 0, src_cols - 1);
    src_y0 = clamp(src_y0, 0, src_rows - 1);
    src_x1 = clamp(src_x1, 0, src_cols - 1);
    src_y1 = clamp(src_y1, 0, src_rows - 1);

    if (channels == 1) {
      global float* data = (global float*)((global uchar*)src +
                                            src_y0 * src_stride);
      float src_value0 = data[src_x0];
      float src_value1 = data[src_x1];
      float value0 = diff_x1 * diff_y1 * src_value0;
      float value1 = diff_x0 * diff_y1 * src_value1;
      float sum = 0.f;
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)src + src_y1 * src_stride);
      src_value0 = data[src_x0];
      src_value1 = data[src_x1];
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      data[element_x] = sum;
    }
    else if (channels == 3) {
      global float* data = (global float*)((global uchar*)src +
                                            src_y0 * src_stride);
      float3 src_value0 = vload3(src_x0, data);
      float3 src_value1 = vload3(src_x1, data);
      float3 value0 = diff_x1 * diff_y1 * src_value0;
      float3 value1 = diff_x0 * diff_y1 * src_value1;
      float3 sum = (float3)(0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)src + src_y1 * src_stride);
      src_value0 = vload3(src_x0, data);
      src_value1 = vload3(src_x1, data);
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      vstore3(sum, element_x, data);
    }
    else {
      global float* data = (global float*)((global uchar*)src +
                                            src_y0 * src_stride);
      float4 src_value0 = vload4(src_x0, data);
      float4 src_value1 = vload4(src_x1, data);
      float4 value0 = diff_x1 * diff_y1 * src_value0;
      float4 value1 = diff_x0 * diff_y1 * src_value1;
      float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)src + src_y1 * src_stride);
      src_value0 = vload4(src_x0, data);
      src_value1 = vload4(src_x1, data);
      value0 = diff_x1 * diff_y0 * src_value0;
      value1 = diff_x0 * diff_y0 * src_value1;
      sum += value0;
      sum += value1;

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      vstore4(sum, element_x, data);
    }
  }
  else {
  }
}
#endif

#if defined(WARPAFFINE_NP_U8) || defined(ALL_KERNELS)
__kernel
void warpaffineNPU8Kernel(global const uchar* src, int src_rows, int src_cols,
         int channels, int src_stride, float coeff0, float coeff1, float coeff2,
         float coeff3, float coeff4, float coeff5, global uchar* dst,
         int dst_rows, int dst_cols, int dst_stride,
         enum BorderType border_type, uchar border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

/*   float2 src_xy;
  src_xy.x = coeff0 * element_x + coeff1 * element_y + coeff2;
  src_xy.y = coeff3 * element_x + coeff4 * element_y + coeff5;
  int src_x = floor(src_xy.x);
  int src_y = floor(src_xy.y); */

  int base_x = convert_int_sat_rte((coeff1 * element_y + coeff2) * AB_SCALE) +
               ROUND_DELTA;
  int base_y = convert_int_sat_rte((coeff4 * element_y + coeff5) * AB_SCALE) +
               ROUND_DELTA;
  int src_x = (convert_int_sat_rte(coeff0 * element_x * AB_SCALE) + base_x) >>
              AB_BITS;
  int src_y = (convert_int_sat_rte(coeff3 * element_x * AB_SCALE) + base_y) >>
              AB_BITS;

  if (border_type == BORDER_CONSTANT) {
    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      global uchar* data = src + src_y * src_stride;
      if (channels == 1) {
        uchar value = data[src_x];

        data = dst + element_y * dst_stride;
        data[element_x] = value;
      }
      else if (channels == 3) {
        uchar3 value = vload3(src_x, data);

        data = dst + element_y * dst_stride;
        vstore3(value, element_x, data);
      }
      else {  // channels == 4
        uchar4 value = vload4(src_x, data);

        data = dst + element_y * dst_stride;
        vstore4(value, element_x, data);
      }
    }
    else {
      global uchar* data = dst + element_y * dst_stride;
      if (channels == 1) {
        data[element_x] = border_value;
      }
      else if (channels == 3) {
        uchar3 value = (uchar3)(border_value, border_value, border_value);
        vstore3(value, element_x, data);
      }
      else {  // channels == 4
        uchar4 value = (uchar4)(border_value, border_value, border_value,
                                border_value);
        vstore4(value, element_x, data);
      }
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = clamp(src_x, 0, src_cols - 1);
    src_y = clamp(src_y, 0, src_rows - 1);

    global uchar* data = src + src_y * src_stride;
    if (channels == 1) {
      uchar value = data[src_x];

      data = dst + element_y * dst_stride;
      data[element_x] = value;
    }
    else if (channels == 3) {
      uchar3 value = vload3(src_x, data);

      data = dst + element_y * dst_stride;
      vstore3(value, element_x, data);
    }
    else {  // channels == 4
      uchar4 value = vload4(src_x, data);

      data = dst + element_y * dst_stride;
      vstore4(value, element_x, data);
    }
  }
  else if (border_type == BORDER_TRANSPARENT) {
    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      global uchar* data = src + src_y * src_stride;
      if (channels == 1) {
        uchar value = data[src_x];

        data = dst + element_y * dst_stride;
        data[element_x] = value;
      }
      else if (channels == 3) {
        uchar3 value = vload3(src_x, data);

        data = dst + element_y * dst_stride;
        vstore3(value, element_x, data);
      }
      else {  // channels == 4
        uchar4 value = vload4(src_x, data);

        data = dst + element_y * dst_stride;
        vstore4(value, element_x, data);
      }
    }
  }
  else {
  }
}
#endif

#if defined(WARPAFFINE_NP_F32) || defined(ALL_KERNELS)
__kernel
void warpaffineNPF32Kernel(global const float* src, int src_rows, int src_cols,
         int channels, int src_stride, float coeff0, float coeff1, float coeff2,
         float coeff3, float coeff4, float coeff5, global float* dst,
         int dst_rows, int dst_cols, int dst_stride,
         enum BorderType border_type, float border_value) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

/*   float2 src_xy;
  src_xy.x = coeff0 * element_x + coeff1 * element_y + coeff2;
  src_xy.y = coeff3 * element_x + coeff4 * element_y + coeff5;
  int src_x = floor(src_xy.x);
  int src_y = floor(src_xy.y); */

  int base_x = convert_int_sat_rte((coeff1 * element_y + coeff2) * AB_SCALE) +
               ROUND_DELTA;
  int base_y = convert_int_sat_rte((coeff4 * element_y + coeff5) * AB_SCALE) +
               ROUND_DELTA;
  int src_x = (convert_int_sat_rte(coeff0 * element_x * AB_SCALE) + base_x) >>
              AB_BITS;
  int src_y = (convert_int_sat_rte(coeff3 * element_x * AB_SCALE) + base_y) >>
              AB_BITS;

  if (border_type == BORDER_CONSTANT) {
    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      global float* data = (global float*)((global uchar*)src +
                                           src_y * src_stride);
      if (channels == 1) {
        float value = data[src_x];

        data = (global float*)((global uchar*)dst + element_y * dst_stride);
        data[element_x] = value;
      }
      else if (channels == 3) {
        float3 value = vload3(src_x, data);

        data = (global float*)((global uchar*)dst + element_y * dst_stride);
        vstore3(value, element_x, data);
      }
      else {  // channels == 4
        float4 value = vload4(src_x, data);

        data = (global float*)((global uchar*)dst + element_y * dst_stride);
        vstore4(value, element_x, data);
      }
    }
    else {
      global float* data = (global float*)((global uchar*)dst +
                                           element_y * dst_stride);
      if (channels == 1) {
        data[element_x] = border_value;
      }
      else if (channels == 3) {
        float3 value = (float3)(border_value, border_value, border_value);
        vstore3(value, element_x, data);
      }
      else {  // channels == 4
        float4 value = (float4)(border_value, border_value, border_value,
                                border_value);
        vstore4(value, element_x, data);
      }
    }
  }
  else if (border_type == BORDER_REPLICATE) {
    src_x = clamp(src_x, 0, src_cols - 1);
    src_y = clamp(src_y, 0, src_rows - 1);

    global float* data = (global float*)((global uchar*)src +
                                         src_y * src_stride);
    if (channels == 1) {
      float value = data[src_x];

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      data[element_x] = value;
    }
    else if (channels == 3) {
      float3 value = vload3(src_x, data);

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      vstore3(value, element_x, data);
    }
    else {  // channels == 4
      float4 value = vload4(src_x, data);

      data = (global float*)((global uchar*)dst + element_y * dst_stride);
      vstore4(value, element_x, data);
    }
  }
  else if (border_type == BORDER_TRANSPARENT) {
    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      global float* data = (global float*)((global uchar*)src +
                                           src_y * src_stride);
      if (channels == 1) {
        float value = data[src_x];

        data = (global float*)((global uchar*)dst + element_y * dst_stride);
        data[element_x] = value;
      }
      else if (channels == 3) {
        float3 value = vload3(src_x, data);

        data = (global float*)((global uchar*)dst + element_y * dst_stride);
        vstore3(value, element_x, data);
      }
      else {  // channels == 4
        float4 value = vload4(src_x, data);

        data = (global float*)((global uchar*)dst + element_y * dst_stride);
        vstore4(value, element_x, data);
      }
    }
  }
  else {
  }
}
#endif
