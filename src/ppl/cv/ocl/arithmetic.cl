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

/******************************* add operation *******************************/

#if defined(ADD_U81D) || defined(ALL_KERNELS)
__kernel
void addU8Kernel0(global const uchar* src0, int cols, global const uchar* src1,
                  global uchar* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  uchar4 input_value0 = vload4(element_x, src0);
  uchar4 input_value1 = vload4(element_x, src1);
  uchar4 output_value = add_sat(input_value0, input_value1);

  if (index_x < cols - 3) {
    vstore4(output_value, element_x, dst);
  }
  else {
    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      dst[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(ADD_U82D) || defined(ALL_KERNELS)
__kernel
void addU8Kernel1(global const uchar* src0, int rows, int cols, int src0_stride,
                  global const uchar* src1, int src1_stride, global uchar* dst,
                  int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global uchar* data = src0 + mul24(element_y, src0_stride);
  uchar4 input_value0 = vload4(element_x, data);
  data = src1 + mul24(element_y, src1_stride);
  uchar4 input_value1 = vload4(element_x, data);
  uchar4 output_value = add_sat(input_value0, input_value1);

  data = dst + mul24(element_y, dst_stride);
  if (index_x < cols - 3) {
    vstore4(output_value, element_x, data);
  }
  else {
    data[index_x] = output_value.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(ADD_F32ALIGNED) || defined(ALL_KERNELS)
__kernel
void addF32Kernel0(global const float* src0, int rows, int cols,
                   int src0_stride, global const float* src1, int src1_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 input_value0 = vload2(element_x, data);
  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  float2 input_value1 = vload2(element_x, data);
  float2 output_value = input_value0 + input_value1;

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(output_value, element_x, data);
}
#endif

#if defined(ADD_F32UNALIGNED) || defined(ALL_KERNELS)
__kernel
void addF32Kernel1(global const float* src0, int rows, int cols,
                   int src0_stride, global const float* src1, int src1_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 input_value0 = vload2(element_x, data);
  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  float2 input_value1 = vload2(element_x, data);
  float2 output_value = input_value0 + input_value1;

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  if (index_x < cols - 1) {
    vstore2(output_value, element_x, data);
  }
  else {
    data[index_x] = output_value.x;
  }
}
#endif

/*************************** addWeighted operation ***************************/

#if defined(ADDWEIGHTED_U81D) || defined(ALL_KERNELS)
__kernel
void addWeightedU8Kernel0(global const uchar* src0, int cols, float alpha,
                          global const uchar* src1, float beta, float gamma,
                          global uchar* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  float4 sum = (float4)gamma;
  uchar4 value = vload4(element_x, src0);
  sum += convert_float4(value) * alpha;
  value = vload4(element_x, src1);
  sum += convert_float4(value) * beta;
  value = convert_uchar4_sat(sum);

  if (index_x < cols - 3) {
    vstore4(value, element_x, dst);
  }
  else {
    dst[index_x] = value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = value.y;
    }
    if ((index_x < cols - 2)) {
      dst[index_x + 2] = value.z;
    }
  }
}
#endif

#if defined(ADDWEIGHTED_U82D) || defined(ALL_KERNELS)
__kernel
void addWeightedU8Kernel1(global const uchar* src0, int rows, int cols,
                          int src0_stride, float alpha,
                          global const uchar* src1, int src1_stride, float beta,
                          float gamma, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  float4 sum = (float4)gamma;
  global uchar* data = src0 + mul24(element_y, src0_stride);
  uchar4 value = vload4(element_x, data);
  sum += convert_float4(value) * alpha;

  data = src1 + mul24(element_y, src1_stride);
  value = vload4(element_x, data);
  sum += convert_float4(value) * beta;
  value = convert_uchar4_sat(sum);

  data = dst + mul24(element_y, dst_stride);
  if (index_x < cols - 3) {
    vstore4(value, element_x, data);
  }
  else {
    data[index_x] = value.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = value.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = value.z;
    }
  }
}
#endif

#if defined(ADDWEIGHTED_F32ALIGNED) || defined(ALL_KERNELS)
__kernel
void addWeightedF32Kernel0(global const float* src0, int rows, int cols,
                           int src0_stride, float alpha,
                           global const float* src1, int src1_stride,
                           float beta, float gamma, global float* dst,
                           int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  float2 sum = (float2)gamma;
  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 value = vload2(element_x, data);
  sum += value * alpha;

  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  value = vload2(element_x, data);
  sum += value * beta;

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(sum, element_x, data);
}
#endif

#if defined(ADDWEIGHTED_F32UNALIGNED) || defined(ALL_KERNELS)
__kernel
void addWeightedF32Kernel1(global const float* src0, int rows, int cols,
                           int src0_stride, float alpha,
                           global const float* src1, int src1_stride,
                           float beta, float gamma, global float* dst,
                           int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  float2 sum = (float2)gamma;
  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 value = vload2(element_x, data);
  sum += value * alpha;

  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  value = vload2(element_x, data);
  sum += value * beta;

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  if (index_x < cols - 1) {
    vstore2(sum, element_x, data);
  }
  else {
    data[index_x] = sum.x;
  }
}
#endif

/**************************** subtract operation ****************************/

#if defined(SUBTRACT_U81D) || defined(ALL_KERNELS)
__kernel
void subtractU8Kernel0(global const uchar* src0, int cols,
                       global const uchar* src1, global uchar* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  uchar4 input_value0 = vload4(element_x, src0);
  uchar4 input_value1 = vload4(element_x, src1);
  uchar4 output_value = sub_sat(input_value0, input_value1);

  if (index_x < cols - 3) {
    vstore4(output_value, element_x, dst);
  }
  else {
    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      dst[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(SUBTRACT_U82D) || defined(ALL_KERNELS)
__kernel
void subtractU8Kernel1(global const uchar* src0, int rows, int cols,
                       int src0_stride, global const uchar* src1,
                       int src1_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global uchar* data = src0 + mul24(element_y, src0_stride);
  uchar4 input_value0 = vload4(element_x, data);
  data = src1 + mul24(element_y, src1_stride);
  uchar4 input_value1 = vload4(element_x, data);
  uchar4 output_value = sub_sat(input_value0, input_value1);

  data = dst + mul24(element_y, dst_stride);
  if (index_x < cols - 3) {
    vstore4(output_value, element_x, data);
  }
  else {
    data[index_x] = output_value.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(SUBTRACT_F32ALIGNED) || defined(ALL_KERNELS)
__kernel
void subtractF32Kernel0(global const float* src0, int rows, int cols,
                        int src0_stride, global const float* src1,
                        int src1_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 input_value0 = vload2(element_x, data);
  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  float2 input_value1 = vload2(element_x, data);
  float2 output_value = input_value0 - input_value1;

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(output_value, element_x, data);
}
#endif

#if defined(SUBTRACT_F32UNALIGNED) || defined(ALL_KERNELS)
__kernel
void subtractF32Kernel1(global const float* src0, int rows, int cols,
                        int src0_stride, global const float* src1,
                        int src1_stride, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 input_value0 = vload2(element_x, data);
  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  float2 input_value1 = vload2(element_x, data);
  float2 output_value = input_value0 - input_value1;

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  if (index_x < cols - 1) {
    vstore2(output_value, element_x, data);
  }
  else {
    data[index_x] = output_value.x;
  }
}
#endif

/**************************** multiply operation *****************************/

#if defined(MUL_U81D) || defined(ALL_KERNELS)
__kernel
void mulU8Kernel0(global const uchar* src0, int cols, global const uchar* src1,
                  float scale, global uchar* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  uchar4 input_value0 = vload4(element_x, src0);
  uchar4 input_value1 = vload4(element_x, src1);
  uchar4 output_value;
  if (scale == 1.f) {
    output_value = convert_uchar4_sat(convert_ushort4(input_value0) *
                                      convert_ushort4(input_value1));
  }
  else {
    float4 sum = convert_float4(input_value0) * scale;
    sum = sum * convert_float4(input_value1);
    output_value = convert_uchar4_sat(sum);
  }

  if (index_x < cols - 3) {
    vstore4(output_value, element_x, dst);
  }
  else {
    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      dst[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(MUL_U82D) || defined(ALL_KERNELS)
__kernel
void mulU8Kernel1(global const uchar* src0, int rows, int cols, int src0_stride,
                  global const uchar* src1, int src1_stride, float scale,
                  global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global uchar* data = src0 + mul24(element_y, src0_stride);
  uchar4 input_value0 = vload4(element_x, data);
  data = src1 + mul24(element_y, src1_stride);
  uchar4 input_value1 = vload4(element_x, data);
  uchar4 output_value;
  if (scale == 1.f) {
    output_value = convert_uchar4_sat(convert_ushort4(input_value0) *
                                      convert_ushort4(input_value1));
  }
  else {
    float4 sum = convert_float4(input_value0) * scale;
    sum = sum * convert_float4(input_value1);
    output_value = convert_uchar4_sat(sum);
  }

  data = dst + mul24(element_y, dst_stride);
  if (index_x < cols - 3) {
    vstore4(output_value, element_x, data);
  }
  else {
    data[index_x] = output_value.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(MUL_F32ALIGNED) || defined(ALL_KERNELS)
__kernel
void mulF32Kernel0(global const float* src0, int rows, int cols,
                   int src0_stride, global const float* src1, int src1_stride,
                   float scale, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 input_value0 = vload2(element_x, data);
  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  float2 input_value1 = vload2(element_x, data);
  float2 output_value = input_value0 * input_value1;
  output_value = output_value * scale;

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(output_value, element_x, data);
}
#endif

#if defined(MUL_F32UNALIGNED) || defined(ALL_KERNELS)
__kernel
void mulF32Kernel1(global const float* src0, int rows, int cols,
                   int src0_stride, global const float* src1, int src1_stride,
                   float scale, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 input_value0 = vload2(element_x, data);
  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  float2 input_value1 = vload2(element_x, data);
  float2 output_value = input_value0 * input_value1;
  output_value = output_value * scale;

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  if (index_x < cols - 1) {
    vstore2(output_value, element_x, data);
  }
  else {
    data[index_x] = output_value.x;
  }
}
#endif

/**************************** divide operation *****************************/

#if defined(DIV_U81D) || defined(ALL_KERNELS)
__kernel
void divU8Kernel0(global const uchar* src0, int cols, global const uchar* src1,
                  float scale, global uchar* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  float4 input_value0 = convert_float4(vload4(element_x, src0));
  float4 input_value1 = convert_float4(vload4(element_x, src1));
  if (scale != 1.f) {
    input_value0 *= scale;
  }
  float4 value = input_value1 == 0.f ? 0.f :
                 native_divide(input_value0, input_value1);
  uchar4 output_value = convert_uchar4_sat_rte(value);

  if (index_x < cols - 3) {
    vstore4(output_value, element_x, dst);
  }
  else {
    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      dst[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(DIV_U82D) || defined(ALL_KERNELS)
__kernel
void divU8Kernel1(global const uchar* src0, int rows, int cols, int src0_stride,
                  global const uchar* src1, int src1_stride, float scale,
                  global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global uchar* data = src0 + mul24(element_y, src0_stride);
  float4 input_value0 = convert_float4(vload4(element_x, data));
  data = src1 + mul24(element_y, src1_stride);
  float4 input_value1 = convert_float4(vload4(element_x, data));
  if (scale != 1.f) {
    input_value0 *= scale;
  }
  float4 value = input_value1 == 0.f ? 0.f :
                 native_divide(input_value0, input_value1);
  uchar4 output_value = convert_uchar4_sat_rte(value);

  data = dst + mul24(element_y, dst_stride);
  if (index_x < cols - 3) {
    vstore4(output_value, element_x, data);
  }
  else {
    data[index_x] = output_value.x;
    if (index_x < cols - 1) {
      data[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      data[index_x + 2] = output_value.z;
    }
  }
}
#endif

#if defined(DIV_F32ALIGNED) || defined(ALL_KERNELS)
__kernel
void divF32Kernel0(global const float* src0, int rows, int cols,
                   int src0_stride, global const float* src1, int src1_stride,
                   float scale, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 input_value0 = vload2(element_x, data);
  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  float2 input_value1 = vload2(element_x, data);
  if (scale != 1.f) {
    input_value0 *= scale;
  }
  float2 output_value = input_value1 == 0.f ? 0.f :
                        native_divide(input_value0, input_value1);

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  vstore2(output_value, element_x, data);
}
#endif

#if defined(DIV_F32UNALIGNED) || defined(ALL_KERNELS)
__kernel
void divF32Kernel1(global const float* src0, int rows, int cols,
                   int src0_stride, global const float* src1, int src1_stride,
                   float scale, global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global float* data = (global float*)((global uchar*)src0 +
                        mul24(element_y, src0_stride));
  float2 input_value0 = vload2(element_x, data);
  data = (global float*)((global uchar*)src1 + mul24(element_y, src1_stride));
  float2 input_value1 = vload2(element_x, data);
  if (scale != 1.f) {
    input_value0 *= scale;
  }
  float2 output_value = input_value1 == 0.f ? (float2)0.f :
                        native_divide(input_value0, input_value1);

  data = (global float*)((global uchar*)dst + mul24(element_y, dst_stride));
  if (index_x < cols - 1) {
    vstore2(output_value, element_x, data);
  }
  else {
    data[index_x] = output_value.x;
  }
}
#endif
