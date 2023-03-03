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

#define INCREASE(x, l) ((x + 1) >= (l) ? (x) : ((x) + 1))
#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

#if defined(RESIZE_LINEAR_U8) || defined(ALL_KERNELS)
__kernel
void resizeLinearU8Kernel(global const uchar* src, int src_rows, int src_cols,
                          int channels, int src_stride, global uchar* dst,
                          int dst_rows, int dst_cols, int dst_stride,
                          float col_scale, float row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_y = ((element_y + 0.5f) * row_scale - 0.5f);
  float float_x = ((element_x + 0.5f) * col_scale - 0.5f);
  int int_y0 = floor(float_y);
  int int_x0 = floor(float_x);
  float_y -= int_y0;
  float_x -= int_x0;
  if (int_y0 < 0) {
    int_y0  = 0;
    float_y = 0;
  }
  if (int_x0 < 0) {
    int_x0  = 0;
    float_x = 0;
  }
  if (int_y0 >= src_rows) {
    int_y0  = src_rows - 1;
    float_y = 0;
  }
  if (int_x0 >= src_cols) {
    int_x0  = src_cols - 1;
    float_x = 0;
  }

  int int_y1 = INCREASE(int_y0, src_rows);
  int buf_y[2];
  float_y = float_y * INTER_RESIZE_COEF_SCALE;
  buf_y[0] = rint(INTER_RESIZE_COEF_SCALE - float_y);
  buf_y[1] = rint(float_y);

  int int_x1 = INCREASE(int_x0, src_cols);
  int buf_x[2];
  float_x = float_x * INTER_RESIZE_COEF_SCALE;
  buf_x[0] = rint(INTER_RESIZE_COEF_SCALE - rint(float_x));
  buf_x[1] = rint(float_x);

  if (channels == 1) {
    int index = int_y0 * src_stride;
    uchar src0 = src[index + int_x0];
    uchar src1 = src[index + int_x1];
    int value0 = buf_y[0] * buf_x[0] * src0;
    int value1 = buf_y[0] * buf_x[1] * src1;
    int sum = 0;
    sum += value0 + value1;

    index = int_y1 * src_stride;
    src0 = src[index + int_x0];
    src1 = src[index + int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0 + value1;
    uchar result = convert_uchar((sum + (1 << (CAST_BITS - 1))) >> CAST_BITS);

    global uchar* output = dst + element_y * dst_stride;
    output[element_x] = result;
  }
  else if (channels == 3) {
    int index = int_y0 * src_stride;
    int3 src0 = convert_int3(vload3(int_x0, src + index));
    int3 src1 = convert_int3(vload3(int_x1, src + index));
    int3 value0 = buf_y[0] * buf_x[0] * src0;
    int3 value1 = buf_y[0] * buf_x[1] * src1;
    int3 sum = (int3)(0, 0, 0);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = convert_int3(vload3(int_x0, src + index));
    src1 = convert_int3(vload3(int_x1, src + index));
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;
    uchar3 result = convert_uchar3((sum + (1 << (CAST_BITS - 1))) >> CAST_BITS);

    global uchar* output = dst + element_y * dst_stride;
    vstore3(result, element_x, output);
  }
  else {  // channels == 4
    int index = int_y0 * src_stride;
    int4 src0 = convert_int4(vload4(int_x0, src + index));
    int4 src1 = convert_int4(vload4(int_x1, src + index));
    int4 value0 = buf_y[0] * buf_x[0] * src0;
    int4 value1 = buf_y[0] * buf_x[1] * src1;
    int4 sum = (int4)(0, 0, 0, 0);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = convert_int4(vload4(int_x0, src + index));
    src1 = convert_int4(vload4(int_x1, src + index));
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;
    uchar4 result = convert_uchar4((sum + (1 << (CAST_BITS - 1))) >> CAST_BITS);

    global uchar* output = dst + element_y * dst_stride;
    vstore4(result, element_x, output);
  }
}
#endif

#if defined(RESIZE_LINEAR_F32) || defined(ALL_KERNELS)
__kernel
void resizeLinearF32Kernel(global const float* src, int src_rows, int src_cols,
                           int channels, int src_stride, global float* dst,
                           int dst_rows, int dst_cols, int dst_stride,
                           float col_scale, float row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_x = ((element_x + 0.5f) * col_scale - 0.5f);
  float float_y = ((element_y + 0.5f) * row_scale - 0.5f);
  int int_x0 = floor(float_x);
  int int_y0 = floor(float_y);
  float_x -= int_x0;
  float_y -= int_y0;
  if (int_y0 < 0) {
    int_y0  = 0;
    float_y = 0;
  }
  if (int_x0 < 0) {
    int_x0  = 0;
    float_x = 0;
  }
  if (int_y0 >= src_rows) {
    int_y0  = src_rows - 1;
    float_y = 0;
  }
  if (int_x0 >= src_cols) {
    int_x0  = src_cols - 1;
    float_x = 0;
  }

  int int_y1 = INCREASE(int_y0, src_rows);
  float buf_y[2];
  buf_y[0] = 1.f - float_y;
  buf_y[1] = 1.f - buf_y[0];

  int int_x1 = INCREASE(int_x0, src_cols);
  float buf_x[2];
  buf_x[0] = 1.f - float_x;
  buf_x[1] = 1.f - buf_x[0];

  if (channels == 1) {
    int index = int_y0 * src_stride;
    float src0 = src[index + int_x0];
    float src1 = src[index + int_x1];
    float value0 = buf_y[0] * buf_x[0] * src0;
    float value1 = buf_y[0] * buf_x[1] * src1;
    float sum = 0.f;
    sum += value0 + value1;

    index = int_y1 * src_stride;
    src0 = src[index + int_x0];
    src1 = src[index + int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0 + value1;

    global float* output = dst + element_y * dst_stride;
    output[element_x] = sum;
  }
  else if (channels == 3) {
    int index = int_y0 * src_stride;
    float3 src0 = vload3(int_x0, src + index);
    float3 src1 = vload3(int_x1, src + index);
    float3 value0 = buf_y[0] * buf_x[0] * src0;
    float3 value1 = buf_y[0] * buf_x[1] * src1;
    float3 sum = (float3)(0.f, 0.f, 0.f);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = vload3(int_x0, src + index);
    src1 = vload3(int_x1, src + index);
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;

    global float* output = dst + element_y * dst_stride;
    vstore3(sum, element_x, output);
  }
  else {  // channels == 4
    int index = int_y0 * src_stride;
    float4 src0 = vload4(int_x0, src + index);
    float4 src1 = vload4(int_x1, src + index);
    float4 value0 = buf_y[0] * buf_x[0] * src0;
    float4 value1 = buf_y[0] * buf_x[1] * src1;
    float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = vload4(int_x0, src + index);
    src1 = vload4(int_x1, src + index);
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;

    global float* output = dst + element_y * dst_stride;
    vstore4(sum, element_x, output);
  }
}
#endif

#if defined(RESIZE_NP_U8) || defined(ALL_KERNELS)
__kernel
void resizeNPU8Kernel(global const uchar* src, int src_rows, int src_cols,
                      int channels, int src_stride, global uchar* dst,
                      int dst_rows, int dst_cols, int dst_stride,
                      float col_scale, float row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int int_y = element_y * row_scale;
  int_y = min(int_y, src_rows - 1);
  int int_x = element_x * col_scale;
  int_x = min(int_x, src_cols - 1);

  global uchar* data = src + int_y * src_stride;
  if (channels == 1) {
    uchar value = data[int_x];

    data = dst + element_y * dst_stride;
    data[element_x] = value;
  }
  else if (channels == 3) {
    uchar3 value = vload3(int_x, data);

    data = dst + element_y * dst_stride;
    vstore3(value, element_x, data);
  }
  else {  // channels == 4
    uchar4 value = vload4(int_x, data);

    data = dst + element_y * dst_stride;
    vstore4(value, element_x, data);
  }
}
#endif

#if defined(RESIZE_NP_F32) || defined(ALL_KERNELS)
__kernel
void resizeNPF32Kernel(global const float* src, int src_rows, int src_cols,
                       int channels, int src_stride, global float* dst,
                       int dst_rows, int dst_cols, int dst_stride,
                       float col_scale, float row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int int_y = element_y * row_scale;
  int_y = min(int_y, src_rows - 1);
  int int_x = element_x * col_scale;
  int_x = min(int_x, src_cols - 1);

  global float* data = src + int_y * src_stride;
  if (channels == 1) {
    float value = data[int_x];

    data = dst + element_y * dst_stride;
    data[element_x] = value;
  }
  else if (channels == 3) {
    float3 value = vload3(int_x, data);

    data = dst + element_y * dst_stride;
    vstore3(value, element_x, data);
  }
  else {  // channels == 4
    float4 value = vload4(int_x, data);

    data = dst + element_y * dst_stride;
    vstore4(value, element_x, data);
  }
}
#endif

#if defined(RESIZE_AREA0_U8) || defined(ALL_KERNELS)
__kernel
void resizeAreaU8Kernel0(global const uchar* src, int src_rows, int src_cols,
                         int channels, int src_stride, global uchar* dst,
                         int dst_rows, int dst_cols, int dst_stride,
                         float col_scale, float row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  float area = (x_end - x_start) * (y_end - y_start);

  global uchar* data;
  if (channels == 1) {
    float sum = 0.f;
    data = src + y_start * src_stride;
    for (int i = y_start; i < y_end; ++i) {
      for (int j = x_start; j < x_end; ++j) {
        sum += data[j];
      }
      data += src_stride;
    }
    sum /= area;

    data = dst + element_y * dst_stride;
    data[element_x] = convert_uchar_sat(sum);
  }
  else if (channels == 3) {
    float3 sum = (float3)(0.f, 0.f, 0.f);
    data = src + y_start * src_stride;
    for (int i = y_start; i < y_end; ++i) {
      for (int j = x_start; j < x_end; ++j) {
        sum += convert_float3(vload3(j, data));
      }
      data += src_stride;
    }
    sum /= area;
    uchar3 result = convert_uchar3_sat(sum);

    data = dst + element_y * dst_stride;
    vstore3(result, element_x, data);
  }
  else {  // channels == 4
    float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
    data = src + y_start * src_stride;
    for (int i = y_start; i < y_end; ++i) {
      for (int j = x_start; j < x_end; ++j) {
        sum += convert_float4(vload4(j, data));
      }
      data += src_stride;
    }
    sum /= area;
    uchar4 result = convert_uchar4_sat(sum);

    data = dst + element_y * dst_stride;
    vstore4(result, element_x, data);
  }
}
#endif

#if defined(RESIZE_AREA0_F32) || defined(ALL_KERNELS)
__kernel
void resizeAreaF32Kernel0(global const float* src, int src_rows, int src_cols,
                          int channels, int src_stride, global float* dst,
                          int dst_rows, int dst_cols, int dst_stride,
                          float col_scale, float row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  float area = (x_end - x_start) * (y_end - y_start);

  global float* data;
  if (channels == 1) {
    float sum = 0.f;
    data = src + y_start * src_stride;
    for (int i = y_start; i < y_end; ++i) {
      for (int j = x_start; j < x_end; ++j) {
        sum += data[j];
      }
      data += src_stride;
    }
    sum /= area;

    data = dst + element_y * dst_stride;
    data[element_x] = sum;
  }
  else if (channels == 3) {
    float3 sum = (float3)(0.f, 0.f, 0.f);
    data = src + y_start * src_stride;
    for (int i = y_start; i < y_end; ++i) {
      for (int j = x_start; j < x_end; ++j) {
        sum += vload3(j, data);
      }
      data += src_stride;
    }
    sum /= area;

    data = dst + element_y * dst_stride;
    vstore3(sum, element_x, data);
  }
  else {  // channels == 4
    float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
    data = src + y_start * src_stride;
    for (int i = y_start; i < y_end; ++i) {
      for (int j = x_start; j < x_end; ++j) {
        sum += vload4(j, data);
      }
      data += src_stride;
    }
    sum /= area;

    data = dst + element_y * dst_stride;
    vstore4(sum, element_x, data);
  }
}
#endif

#if defined(RESIZE_AREA1_U8) || defined(ALL_KERNELS)
__kernel
void resizeAreaU8Kernel1(global const uchar* src, int src_rows, int src_cols,
                         int channels, int src_stride, global uchar* dst,
                         int dst_rows, int dst_cols, int dst_stride,
                         float col_scale, float row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_y0 = element_y * row_scale;
  float float_y1 = float_y0 + row_scale;
  int int_y0 = ceil(float_y0);
  int int_y1 = floor(float_y1);

  float float_x0 = element_x * col_scale;
  float float_x1 = float_x0 + col_scale;
  int int_x0 = ceil(float_x0);
  int int_x1 = floor(float_x1);

  if (channels == 1) {
    global uchar* data;
    float coeff0, coeff1;
    float sum = 0.f;
    float area = fmin(col_scale, src_cols - float_x0) *
                 fmin(row_scale, src_rows - float_y0);

    coeff0 = int_y0 - float_y0;
    coeff1 = int_x0 - float_x0;
    if (coeff0 > 1e-3) {
      data = src + (int_y0 - 1) * src_stride;
      if (coeff1 > 1e-3) {
        sum += coeff0 * coeff1 * data[int_x0 - 1];
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += coeff0 * data[dx];
      }

      if (float_x1 - int_x1 > 1e-3) {
        sum += coeff0 * (float_x1 - int_x1) * data[int_x1];
      }
    }

    data = src + int_y0 * src_stride;
    for (int dy = int_y0; dy < int_y1; ++dy) {
      if (coeff1 > 1e-3) {
        sum += coeff1 * data[int_x0 - 1];
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += data[dx];
      }

      if (float_x1 - int_x1 > 1e-3) {
        sum += (float_x1 - int_x1) * data[int_x1];
      }
      data += src_stride;
    }

    coeff0 = float_y1 - int_y1;
    if (coeff0 > 1e-3) {
      if (coeff1 > 1e-3) {
        sum += coeff0 * coeff1 * data[int_x0 - 1];
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += coeff0 * data[dx];
      }

      if (float_x1 - int_x1 > 1e-3) {
        sum += coeff0 * (float_x1 - int_x1) * data[int_x1];
      }
    }
    sum = sum / area;

    data = dst + element_y * dst_stride;
    data[element_x] = convert_uchar_sat(sum);
  }
  else if (channels == 3) {
    global uchar* data;
    float coeff0, coeff1;
    float3 value;
    float3 sum = (float3)(0.f, 0.f, 0.f);
    float area = fmin(col_scale, src_cols - float_x0) *
                 fmin(row_scale, src_rows - float_y0);

    coeff0 = int_y0 - float_y0;
    coeff1 = int_x0 - float_x0;
    if (coeff0 > 1e-3) {
      data = src + (int_y0 - 1) * src_stride;
      if (coeff1 > 1e-3) {
        value = coeff0 * coeff1 * convert_float3(vload3(int_x0 - 1, data));
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        value = coeff0 * convert_float3(vload3(dx, data));
        sum += value;
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = coeff0 * (float_x1 - int_x1) *
                convert_float3(vload3(int_x1, data));
        sum += value;
      }
    }

    data = src + int_y0 * src_stride;
    for (int dy = int_y0; dy < int_y1; ++dy) {
      if (coeff1 > 1e-3) {
        value = coeff1 * convert_float3(vload3(int_x0 - 1, data));
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += convert_float3(vload3(dx, data));
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = (float_x1 - int_x1) * convert_float3(vload3(int_x1, data));
        sum += value;
      }
      data += src_stride;
    }

    coeff0 = float_y1 - int_y1;
    if (coeff0 > 1e-3) {
      if (coeff1 > 1e-3) {
        value = coeff0 * coeff1 * convert_float3(vload3(int_x0 - 1, data));
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        value = coeff0 * convert_float3(vload3(dx, data));
        sum += value;
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = coeff0 * (float_x1 - int_x1) *
                convert_float3(vload3(int_x1, data));
        sum += value;
      }
    }
    sum /= area;
    uchar3 result = convert_uchar3_sat(sum);

    data = dst + element_y * dst_stride;
    vstore3(result, element_x, data);
  }
  else {  // channels == 4
    global uchar* data;
    float coeff0, coeff1;
    float4 value;
    float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
    float area = fmin(col_scale, src_cols - float_x0) *
                 fmin(row_scale, src_rows - float_y0);

    coeff0 = int_y0 - float_y0;
    coeff1 = int_x0 - float_x0;
    if (coeff0 > 1e-3) {
      data = src + (int_y0 - 1) * src_stride;
      if (coeff1 > 1e-3) {
        value = coeff0 * coeff1 * convert_float4(vload4(int_x0 - 1, data));
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        value = coeff0 * convert_float4(vload4(dx, data));
        sum += value;
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = coeff0 * (float_x1 - int_x1) *
                convert_float4(vload4(int_x1, data));
        sum += value;
      }
    }

    data = src + int_y0 * src_stride;
    for (int dy = int_y0; dy < int_y1; ++dy) {
      if (coeff1 > 1e-3) {
        value = coeff1 * convert_float4(vload4(int_x0 - 1, data));
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += convert_float4(vload4(dx, data));
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = (float_x1 - int_x1) * convert_float4(vload4(int_x1, data));
        sum += value;
      }
      data += src_stride;
    }

    coeff0 = float_y1 - int_y1;
    if (coeff0 > 1e-3) {
      if (coeff1 > 1e-3) {
        value = coeff0 * coeff1 * convert_float4(vload4(int_x0 - 1, data));
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        value = coeff0 * convert_float4(vload4(dx, data));
        sum += value;
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = coeff0 * (float_x1 - int_x1) *
                convert_float4(vload4(int_x1, data));
        sum += value;
      }
    }
    sum /= area;
    uchar4 result = convert_uchar4_sat(sum);

    data = dst + element_y * dst_stride;
    vstore4(result, element_x, data);
  }
}
#endif

#if defined(RESIZE_AREA1_F32) || defined(ALL_KERNELS)
__kernel
void resizeAreaF32Kernel1(global const float* src, int src_rows, int src_cols,
                          int channels, int src_stride, global float* dst,
                          int dst_rows, int dst_cols, int dst_stride,
                          float col_scale, float row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  float float_y0 = element_y * row_scale;
  float float_y1 = float_y0 + row_scale;
  int int_y0 = ceil(float_y0);
  int int_y1 = floor(float_y1);

  float float_x0 = element_x * col_scale;
  float float_x1 = float_x0 + col_scale;
  int int_x0 = ceil(float_x0);
  int int_x1 = floor(float_x1);

  if (channels == 1) {
    global float* data;
    float coeff0, coeff1;
    float sum = 0.f;
    float area = fmin(col_scale, src_cols - float_x0) *
                 fmin(row_scale, src_rows - float_y0);

    coeff0 = int_y0 - float_y0;
    coeff1 = int_x0 - float_x0;
    if (coeff0 > 1e-3) {
      data = src + (int_y0 - 1) * src_stride;
      if (coeff1 > 1e-3) {
        sum += coeff0 * coeff1 * data[int_x0 - 1];
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += coeff0 * data[dx];
      }

      if (float_x1 - int_x1 > 1e-3) {
        sum += coeff0 * (float_x1 - int_x1) * data[int_x1];
      }
    }

    data = src + int_y0 * src_stride;
    for (int dy = int_y0; dy < int_y1; ++dy) {
      if (coeff1 > 1e-3) {
        sum += coeff1 * data[int_x0 - 1];
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += data[dx];
      }

      if (float_x1 - int_x1 > 1e-3) {
        sum += (float_x1 - int_x1) * data[int_x1];
      }
      data += src_stride;
    }

    coeff0 = float_y1 - int_y1;
    if (coeff0 > 1e-3) {
      if (coeff1 > 1e-3) {
        sum += coeff0 * coeff1 * data[int_x0 - 1];
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += coeff0 * data[dx];
      }

      if (float_x1 - int_x1 > 1e-3) {
        sum += coeff0 * (float_x1 - int_x1) * data[int_x1];
      }
    }
    sum = sum / area;

    data = dst + element_y * dst_stride;
    data[element_x] = sum;
  }
  else if (channels == 3) {
    global float* data;
    float coeff0, coeff1;
    float3 value;
    float3 sum = (float3)(0.f, 0.f, 0.f);
    float area = fmin(col_scale, src_cols - float_x0) *
                 fmin(row_scale, src_rows - float_y0);

    coeff0 = int_y0 - float_y0;
    coeff1 = int_x0 - float_x0;
    if (coeff0 > 1e-3) {
      data = src + (int_y0 - 1) * src_stride;
      if (coeff1 > 1e-3) {
        value = coeff0 * coeff1 * vload3(int_x0 - 1, data);
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        value = coeff0 * vload3(dx, data);
        sum += value;
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = coeff0 * (float_x1 - int_x1) * vload3(int_x1, data);
        sum += value;
      }
    }

    data = src + int_y0 * src_stride;
    for (int dy = int_y0; dy < int_y1; ++dy) {
      if (coeff1 > 1e-3) {
        value = coeff1 * vload3(int_x0 - 1, data);
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += vload3(dx, data);
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = (float_x1 - int_x1) * vload3(int_x1, data);
        sum += value;
      }
      data += src_stride;
    }

    coeff0 = float_y1 - int_y1;
    if (coeff0 > 1e-3) {
      if (coeff1 > 1e-3) {
        value = coeff0 * coeff1 * vload3(int_x0 - 1, data);
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        value = coeff0 * vload3(dx, data);
        sum += value;
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = coeff0 * (float_x1 - int_x1) * vload3(int_x1, data);
        sum += value;
      }
    }
    sum /= area;

    data = dst + element_y * dst_stride;
    vstore3(sum, element_x, data);
  }
  else {  // channels == 4
    global float* data;
    float coeff0, coeff1;
    float4 value;
    float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
    float area = fmin(col_scale, src_cols - float_x0) *
                 fmin(row_scale, src_rows - float_y0);

    coeff0 = int_y0 - float_y0;
    coeff1 = int_x0 - float_x0;
    if (coeff0 > 1e-3) {
      data = src + (int_y0 - 1) * src_stride;
      if (coeff1 > 1e-3) {
        value = coeff0 * coeff1 * vload4(int_x0 - 1, data);
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        value = coeff0 * vload4(dx, data);
        sum += value;
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = coeff0 * (float_x1 - int_x1) * vload4(int_x1, data);
        sum += value;
      }
    }

    data = src + int_y0 * src_stride;
    for (int dy = int_y0; dy < int_y1; ++dy) {
      if (coeff1 > 1e-3) {
        value = coeff1 * vload4(int_x0 - 1, data);
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        sum += vload4(dx, data);
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = (float_x1 - int_x1) * vload4(int_x1, data);
        sum += value;
      }
      data += src_stride;
    }

    coeff0 = float_y1 - int_y1;
    if (coeff0 > 1e-3) {
      if (coeff1 > 1e-3) {
        value = coeff0 * coeff1 * vload4(int_x0 - 1, data);
        sum += value;
      }

      for (int dx = int_x0; dx < int_x1; ++dx) {
        value = coeff0 * vload4(dx, data);
        sum += value;
      }

      if (float_x1 - int_x1 > 1e-3) {
        value = coeff0 * (float_x1 - int_x1) * vload4(int_x1, data);
        sum += value;
      }
    }
    sum /= area;

    data = dst + element_y * dst_stride;
    vstore4(sum, element_x, data);
  }
}
#endif

#if defined(RESIZE_AREA2_U8) || defined(ALL_KERNELS)
__kernel
void resizeAreaU8Kernel2(global const uchar* src, int src_rows, int src_cols,
                         int channels, int src_stride, global uchar* dst,
                         int dst_rows, int dst_cols, int dst_stride,
                         float col_scale, float row_scale, float inv_col_scale,
                         float inv_row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int int_y0 = floor(element_y * row_scale);
  int int_x0 = floor(element_x * col_scale);
  float float_y = element_y + 1 - (int_y0 + 1) * inv_row_scale;
  float float_x = element_x + 1 - (int_x0 + 1) * inv_col_scale;
  float_y = float_y <= 0 ? 0.f : float_y - floor(float_y);
  float_x = float_x <= 0 ? 0.f : float_x - floor(float_x);
  if (int_y0 < 0) {
    int_y0  = 0;
    float_y = 0;
  }
  if (int_x0 < 0) {
    int_x0  = 0;
    float_x = 0;
  }
  if (int_y0 >= src_rows) {
    int_y0  = src_rows - 1;
    float_y = 0;
  }
  if (int_x0 >= src_cols) {
    int_x0  = src_cols - 1;
    float_x = 0;
  }

  int int_y1 = INCREASE(int_y0, src_rows);
  int buf_y[2];
  float_y = float_y * INTER_RESIZE_COEF_SCALE;
  buf_y[0] = rint(INTER_RESIZE_COEF_SCALE - float_y);
  buf_y[1] = rint(float_y);

  int int_x1 = INCREASE(int_x0, src_cols);
  int buf_x[2];
  float_x = float_x * INTER_RESIZE_COEF_SCALE;
  buf_x[0] = rint(INTER_RESIZE_COEF_SCALE - rint(float_x));
  buf_x[1] = rint(float_x);

  if (channels == 1) {
    int index = int_y0 * src_stride;
    uchar src0 = src[index + int_x0];
    uchar src1 = src[index + int_x1];
    int value0 = buf_y[0] * buf_x[0] * src0;
    int value1 = buf_y[0] * buf_x[1] * src1;
    int sum = 0;
    sum += value0 + value1;

    index = int_y1 * src_stride;
    src0 = src[index + int_x0];
    src1 = src[index + int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0 + value1;
    uchar result = convert_uchar((sum + (1 << (CAST_BITS - 1))) >> CAST_BITS);

    global uchar* output = dst + element_y * dst_stride;
    output[element_x] = result;
  }
  else if (channels == 3) {
    int index = int_y0 * src_stride;
    int3 src0 = convert_int3(vload3(int_x0, src + index));
    int3 src1 = convert_int3(vload3(int_x1, src + index));
    int3 value0 = buf_y[0] * buf_x[0] * src0;
    int3 value1 = buf_y[0] * buf_x[1] * src1;
    int3 sum = (int3)(0, 0, 0);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = convert_int3(vload3(int_x0, src + index));
    src1 = convert_int3(vload3(int_x1, src + index));
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;
    uchar3 result = convert_uchar3((sum + (1 << (CAST_BITS - 1))) >> CAST_BITS);

    global uchar* output = dst + element_y * dst_stride;
    vstore3(result, element_x, output);
  }
  else {  // channels == 4
    int index = int_y0 * src_stride;
    int4 src0 = convert_int4(vload4(int_x0, src + index));
    int4 src1 = convert_int4(vload4(int_x1, src + index));
    int4 value0 = buf_y[0] * buf_x[0] * src0;
    int4 value1 = buf_y[0] * buf_x[1] * src1;
    int4 sum = (int4)(0, 0, 0, 0);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = convert_int4(vload4(int_x0, src + index));
    src1 = convert_int4(vload4(int_x1, src + index));
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;
    uchar4 result = convert_uchar4((sum + (1 << (CAST_BITS - 1))) >> CAST_BITS);

    global uchar* output = dst + element_y * dst_stride;
    vstore4(result, element_x, output);
  }
}
#endif

#if defined(RESIZE_AREA2_F32) || defined(ALL_KERNELS)
__kernel
void resizeAreaF32Kernel2(global const float* src, int src_rows, int src_cols,
                          int channels, int src_stride, global float* dst,
                          int dst_rows, int dst_cols, int dst_stride,
                          float col_scale, float row_scale, float inv_col_scale,
                          float inv_row_scale) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_x >= dst_cols || element_y >= dst_rows) {
    return;
  }

  int int_y0 = floor(element_y * row_scale);
  int int_x0 = floor(element_x * col_scale);
  float float_y = element_y + 1 - (int_y0 + 1) * inv_row_scale;
  float float_x = element_x + 1 - (int_x0 + 1) * inv_col_scale;
  float_y = float_y <= 0 ? 0.f : float_y - floor(float_y);
  float_x = float_x <= 0 ? 0.f : float_x - floor(float_x);
  if (int_y0 < 0) {
    int_y0  = 0;
    float_y = 0;
  }
  if (int_x0 < 0) {
    int_x0  = 0;
    float_x = 0;
  }
  if (int_y0 >= src_rows) {
    int_y0  = src_rows - 1;
    float_y = 0;
  }
  if (int_x0 >= src_cols) {
    int_x0  = src_cols - 1;
    float_x = 0;
  }

  int int_y1 = INCREASE(int_y0,src_rows);
  float buf_y[2];
  buf_y[0] = 1.f - float_y;
  buf_y[1] = 1.f - buf_y[0];

  int int_x1 = INCREASE(int_x0,src_cols);
  float buf_x[2];
  buf_x[0] = 1.f - float_x;
  buf_x[1] = 1.f - buf_x[0];

  if (channels == 1) {
    int index = int_y0 * src_stride;
    float src0 = src[index + int_x0];
    float src1 = src[index + int_x1];
    float value0 = buf_y[0] * buf_x[0] * src0;
    float value1 = buf_y[0] * buf_x[1] * src1;
    float sum = 0.f;
    sum += value0 + value1;

    index = int_y1 * src_stride;
    src0 = src[index + int_x0];
    src1 = src[index + int_x1];
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0 + value1;

    global float* output = dst + element_y * dst_stride;
    output[element_x] = sum;
  }
  else if (channels == 3) {
    int index = int_y0 * src_stride;
    float3 src0 = vload3(int_x0, src + index);
    float3 src1 = vload3(int_x1, src + index);
    float3 value0 = buf_y[0] * buf_x[0] * src0;
    float3 value1 = buf_y[0] * buf_x[1] * src1;
    float3 sum = (float3)(0.f, 0.f, 0.f);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = vload3(int_x0, src + index);
    src1 = vload3(int_x1, src + index);
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;

    global float* output = dst + element_y * dst_stride;
    vstore3(sum, element_x, output);
  }
  else {  // channels == 4
    int index = int_y0 * src_stride;
    float4 src0 = vload4(int_x0, src + index);
    float4 src1 = vload4(int_x1, src + index);
    float4 value0 = buf_y[0] * buf_x[0] * src0;
    float4 value1 = buf_y[0] * buf_x[1] * src1;
    float4 sum = (float4)(0.f, 0.f, 0.f, 0.f);
    sum += value0;
    sum += value1;

    index = int_y1 * src_stride;
    src0 = vload4(int_x0, src + index);
    src1 = vload4(int_x1, src + index);
    value0 = buf_y[1] * buf_x[0] * src0;
    value1 = buf_y[1] * buf_x[1] * src1;
    sum += value0;
    sum += value1;

    global float* output = dst + element_y * dst_stride;
    vstore4(sum, element_x, output);
  }
}
#endif
