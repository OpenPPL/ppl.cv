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

/************************** BGR(RBB) <-> BGRA(RGBA) *************************/

#if defined(BGR2BGRA_U8_1D) || defined(BGR2BGRA_F32_1D) ||                     \
    defined(RGB2RGBA_U8_1D) || defined(RGB2RGBA_F32_1D) || defined(ALL_KERNELS)
#define BGR2BGRATYPE_1D(Function, base_type, T, T3, T4, alpha)                 \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T3 input_value = vload3(element_x, src);                                     \
  T4 value = (T4)(input_value.x, input_value.y, input_value.z, alpha);         \
                                                                               \
  vstore4(value, element_x, dst);                                              \
}
#endif

#if defined(BGR2BGRA_U8_2D) || defined(BGR2BGRA_F32_2D) ||                     \
    defined(RGB2RGBA_U8_2D) || defined(RGB2RGBA_F32_2D) || defined(ALL_KERNELS)
#define BGR2BGRATYPE_2D(Function, base_type, T, T3, T4, alpha)                 \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T3 input_value = vload3(element_x, data);                                    \
  T4 value = (T4)(input_value.x, input_value.y, input_value.z, alpha);         \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore4(value, element_x, data);                                             \
}
#endif

#if defined(BGR2BGRA_U8_1D) || defined(ALL_KERNELS)
BGR2BGRATYPE_1D(BGR2BGRA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2BGRA_U8_2D) || defined(ALL_KERNELS)
BGR2BGRATYPE_2D(BGR2BGRA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2BGRA_F32_1D) || defined(ALL_KERNELS)
BGR2BGRATYPE_1D(BGR2BGRA, F32, float, float3, float4, 1.f)
#endif

#if defined(BGR2BGRA_F32_2D) || defined(ALL_KERNELS)
BGR2BGRATYPE_2D(BGR2BGRA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGB2RGBA_U8_1D) || defined(ALL_KERNELS)
BGR2BGRATYPE_1D(RGB2RGBA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(RGB2RGBA_U8_2D) || defined(ALL_KERNELS)
BGR2BGRATYPE_2D(RGB2RGBA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(RGB2RGBA_F32_1D) || defined(ALL_KERNELS)
BGR2BGRATYPE_1D(RGB2RGBA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGB2RGBA_F32_2D) || defined(ALL_KERNELS)
BGR2BGRATYPE_2D(RGB2RGBA, F32, float, float3, float4, 1.f)
#endif

#if defined(BGRA2BGR_U8_1D) || defined(BGRA2BGR_F32_1D) ||                     \
    defined(RGBA2RGB_U8_1D) || defined(RGBA2RGB_F32_1D) || defined(ALL_KERNELS)
#define BGRA2BGRTYPE_1D(Function, base_type, T, T3, T4)                        \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T4 input_value = vload4(element_x, src);                                     \
  T3 value = (T3)(input_value.x, input_value.y, input_value.z);                \
                                                                               \
  vstore3(value, element_x, dst);                                              \
}
#endif

#if defined(BGRA2BGR_U8_2D) || defined(BGRA2BGR_F32_2D) ||                     \
    defined(RGBA2RGB_U8_2D) || defined(RGBA2RGB_F32_2D) || defined(ALL_KERNELS)
#define BGRA2BGRTYPE_2D(Function, base_type, T, T3, T4)                        \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T4 input_value = vload4(element_x, data);                                    \
  T3 value = (T3)(input_value.x, input_value.y, input_value.z);                \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore3(value, element_x, data);                                             \
}
#endif

#if defined(BGRA2BGR_U8_1D) || defined(ALL_KERNELS)
BGRA2BGRTYPE_1D(BGRA2BGR, U8, uchar, uchar3, uchar4)
#endif

#if defined(BGRA2BGR_U8_2D) || defined(ALL_KERNELS)
BGRA2BGRTYPE_2D(BGRA2BGR, U8, uchar, uchar3, uchar4)
#endif

#if defined(BGRA2BGR_F32_1D) || defined(ALL_KERNELS)
BGRA2BGRTYPE_1D(BGRA2BGR, F32, float, float3, float4)
#endif

#if defined(BGRA2BGR_F32_2D) || defined(ALL_KERNELS)
BGRA2BGRTYPE_2D(BGRA2BGR, F32, float, float3, float4)
#endif

#if defined(RGBA2RGB_U8_1D) || defined(ALL_KERNELS)
BGRA2BGRTYPE_1D(RGBA2RGB, U8, uchar, uchar3, uchar4)
#endif

#if defined(RGBA2RGB_U8_2D) || defined(ALL_KERNELS)
BGRA2BGRTYPE_2D(RGBA2RGB, U8, uchar, uchar3, uchar4)
#endif

#if defined(RGBA2RGB_F32_1D) || defined(ALL_KERNELS)
BGRA2BGRTYPE_1D(RGBA2RGB, F32, float, float3, float4)
#endif

#if defined(RGBA2RGB_F32_2D) || defined(ALL_KERNELS)
BGRA2BGRTYPE_2D(RGBA2RGB, F32, float, float3, float4)
#endif

#if defined(BGR2RGBA_U8_1D) || defined(BGR2RGBA_F32_1D) ||                     \
    defined(RGB2BGRA_U8_1D) || defined(RGB2BGRA_F32_1D) || defined(ALL_KERNELS)
#define BGR2RGBATYPE_1D(Function, base_type, T, T3, T4, alpha)                 \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T3 input_value = vload3(element_x, src);                                     \
  T4 value = (T4)(input_value.z, input_value.y, input_value.x, alpha);         \
                                                                               \
  vstore4(value, element_x, dst);                                              \
}
#endif

#if defined(BGR2RGBA_U8_2D) || defined(BGR2RGBA_F32_2D) ||                     \
    defined(RGB2BGRA_U8_2D) || defined(RGB2BGRA_F32_2D) || defined(ALL_KERNELS)
#define BGR2RGBATYPE_2D(Function, base_type, T, T3, T4, alpha)                 \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T3 input_value = vload3(element_x, data);                                    \
  T4 value = (T4)(input_value.z, input_value.y, input_value.x, alpha);         \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore4(value, element_x, data);                                             \
}
#endif

#if defined(BGR2RGBA_U8_1D) || defined(ALL_KERNELS)
BGR2RGBATYPE_1D(BGR2RGBA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2RGBA_U8_2D) || defined(ALL_KERNELS)
BGR2RGBATYPE_2D(BGR2RGBA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2RGBA_F32_1D) || defined(ALL_KERNELS)
BGR2RGBATYPE_1D(BGR2RGBA, F32, float, float3, float4, 1.f)
#endif

#if defined(BGR2RGBA_F32_2D) || defined(ALL_KERNELS)
BGR2RGBATYPE_2D(BGR2RGBA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGB2BGRA_U8_1D) || defined(ALL_KERNELS)
BGR2RGBATYPE_1D(RGB2BGRA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(RGB2BGRA_U8_2D) || defined(ALL_KERNELS)
BGR2RGBATYPE_2D(RGB2BGRA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(RGB2BGRA_F32_1D) || defined(ALL_KERNELS)
BGR2RGBATYPE_1D(RGB2BGRA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGB2BGRA_F32_2D) || defined(ALL_KERNELS)
BGR2RGBATYPE_2D(RGB2BGRA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGBA2BGR_U8_1D) || defined(RGBA2BGR_F32_1D) ||                     \
    defined(BGRA2RGB_U8_1D) || defined(BGRA2RGB_F32_1D) || defined(ALL_KERNELS)
#define RGBA2BGRTYPE_1D(Function, base_type, T, T3, T4)                        \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T4 input_value = vload4(element_x, src);                                     \
  T3 value = (T3)(input_value.z, input_value.y, input_value.x);                \
                                                                               \
  vstore3(value, element_x, dst);                                              \
}
#endif

#if defined(RGBA2BGR_U8_2D) || defined(RGBA2BGR_F32_2D) ||                     \
    defined(BGRA2RGB_U8_2D) || defined(BGRA2RGB_F32_2D) || defined(ALL_KERNELS)
#define RGBA2BGRTYPE_2D(Function, base_type, T, T3, T4)                        \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T4 input_value = vload4(element_x, data);                                    \
  T3 value = (T3)(input_value.z, input_value.y, input_value.x);                \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore3(value, element_x, data);                                             \
}
#endif

#if defined(RGBA2BGR_U8_1D) || defined(ALL_KERNELS)
RGBA2BGRTYPE_1D(RGBA2BGR, U8, uchar, uchar3, uchar4)
#endif

#if defined(RGBA2BGR_U8_2D) || defined(ALL_KERNELS)
RGBA2BGRTYPE_2D(RGBA2BGR, U8, uchar, uchar3, uchar4)
#endif

#if defined(RGBA2BGR_F32_1D) || defined(ALL_KERNELS)
RGBA2BGRTYPE_1D(RGBA2BGR, F32, float, float3, float4)
#endif

#if defined(RGBA2BGR_F32_2D) || defined(ALL_KERNELS)
RGBA2BGRTYPE_2D(RGBA2BGR, F32, float, float3, float4)
#endif

#if defined(BGRA2RGB_U8_1D) || defined(ALL_KERNELS)
RGBA2BGRTYPE_1D(BGRA2RGB, U8, uchar, uchar3, uchar4)
#endif

#if defined(BGRA2RGB_U8_2D) || defined(ALL_KERNELS)
RGBA2BGRTYPE_2D(BGRA2RGB, U8, uchar, uchar3, uchar4)
#endif

#if defined(BGRA2RGB_F32_1D) || defined(ALL_KERNELS)
RGBA2BGRTYPE_1D(BGRA2RGB, F32, float, float3, float4)
#endif

#if defined(BGRA2RGB_F32_2D) || defined(ALL_KERNELS)
RGBA2BGRTYPE_2D(BGRA2RGB, F32, float, float3, float4)
#endif

/******************************* BGR <-> RGB ******************************/

#if defined(RGB2BGR_U8_1D) || defined(RGB2BGR_F32_1D) ||                       \
    defined(BGR2RGB_U8_1D) || defined(BGR2RGB_F32_1D) || defined(ALL_KERNELS)
#define RGB2BGRTYPE_1D(Function, base_type, T, T3)                             \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T3 input_value = vload3(element_x, src);                                     \
  T3 value = (T3)(input_value.z, input_value.y, input_value.x);                \
                                                                               \
  vstore3(value, element_x, dst);                                              \
}
#endif

#if defined(RGB2BGR_U8_2D) || defined(RGB2BGR_F32_2D) ||                       \
    defined(BGR2RGB_U8_2D) || defined(BGR2RGB_F32_2D) || defined(ALL_KERNELS)
#define RGB2BGRTYPE_2D(Function, base_type, T, T3)                             \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T3 input_value = vload3(element_x, data);                                    \
  T3 value = (T3)(input_value.z, input_value.y, input_value.x);                \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore3(value, element_x, data);                                             \
}
#endif

#if defined(BGR2RGB_U8_1D) || defined(ALL_KERNELS)
RGB2BGRTYPE_1D(BGR2RGB, U8, uchar, uchar3)
#endif

#if defined(BGR2RGB_U8_2D) || defined(ALL_KERNELS)
RGB2BGRTYPE_2D(BGR2RGB, U8, uchar, uchar3)
#endif

#if defined(BGR2RGB_F32_1D) || defined(ALL_KERNELS)
RGB2BGRTYPE_1D(BGR2RGB, F32, float, float3)
#endif

#if defined(BGR2RGB_F32_2D) || defined(ALL_KERNELS)
RGB2BGRTYPE_2D(BGR2RGB, F32, float, float3)
#endif

#if defined(RGB2BGR_U8_1D) || defined(ALL_KERNELS)
RGB2BGRTYPE_1D(RGB2BGR, U8, uchar, uchar3)
#endif

#if defined(RGB2BGR_U8_2D) || defined(ALL_KERNELS)
RGB2BGRTYPE_2D(RGB2BGR, U8, uchar, uchar3)
#endif

#if defined(RGB2BGR_F32_1D) || defined(ALL_KERNELS)
RGB2BGRTYPE_1D(RGB2BGR, F32, float, float3)
#endif

#if defined(RGB2BGR_F32_2D) || defined(ALL_KERNELS)
RGB2BGRTYPE_2D(RGB2BGR, F32, float, float3)
#endif

/******************************* BGRA <-> RGBA ******************************/

#if defined(BGRA2RGBA_U8_1D) || defined(BGRA2RGBA_F32_1D) ||                   \
    defined(RGBA2BGRA_U8_1D) || defined(RGBA2BGRA_F32_1D) ||                   \
    defined(ALL_KERNELS)
#define BGRA2RGBATYPE_1D(Function, base_type, T, T4)                           \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T4 input_value = vload4(element_x, src);                                     \
  T4 value = (T4)(input_value.z, input_value.y, input_value.x, input_value.w); \
                                                                               \
  vstore4(value, element_x, dst);                                              \
}
#endif

#if defined(BGRA2RGBA_U8_2D) || defined(BGRA2RGBA_F32_2D) ||                   \
    defined(RGBA2BGRA_U8_2D) || defined(RGBA2BGRA_F32_2D) ||                   \
    defined(ALL_KERNELS)
#define BGRA2RGBATYPE_2D(Function, base_type, T, T4)                           \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T4 input_value = vload4(element_x, data);                                    \
  T4 value = (T4)(input_value.z, input_value.y, input_value.x, input_value.w); \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore4(value, element_x, data);                                             \
}
#endif

#if defined(BGRA2RGBA_U8_1D) || defined(ALL_KERNELS)
BGRA2RGBATYPE_1D(BGRA2RGBA, U8, uchar, uchar4)
#endif

#if defined(BGRA2RGBA_U8_2D) || defined(ALL_KERNELS)
BGRA2RGBATYPE_2D(BGRA2RGBA, U8, uchar, uchar4)
#endif

#if defined(BGRA2RGBA_F32_1D) || defined(ALL_KERNELS)
BGRA2RGBATYPE_1D(BGRA2RGBA, F32, float, float4)
#endif

#if defined(BGRA2RGBA_F32_2D) || defined(ALL_KERNELS)
BGRA2RGBATYPE_2D(BGRA2RGBA, F32, float, float4)
#endif

#if defined(RGBA2BGRA_U8_1D) || defined(ALL_KERNELS)
BGRA2RGBATYPE_1D(RGBA2BGRA, U8, uchar, uchar4)
#endif

#if defined(RGBA2BGRA_U8_2D) || defined(ALL_KERNELS)
BGRA2RGBATYPE_2D(RGBA2BGRA, U8, uchar, uchar4)
#endif

#if defined(RGBA2BGRA_F32_1D) || defined(ALL_KERNELS)
BGRA2RGBATYPE_1D(RGBA2BGRA, F32, float, float4)
#endif

#if defined(RGBA2BGRA_F32_2D) || defined(ALL_KERNELS)
BGRA2RGBATYPE_2D(RGBA2BGRA, F32, float, float4)
#endif

/*********************** BGR/RGB/BGRA/RGBA <-> Gray ************************/

enum Bgr2GrayCoefficients {
  kB2Y15    = 3735,
  kG2Y15    = 19235,
  kR2Y15    = 9798,
  kRgbShift = 15,
};

#if defined(BGR2GRAY_U8_1D) || defined(BGR2GRAY_F32_1D) ||                     \
    defined(RGB2GRAY_U8_1D) || defined(RGB2GRAY_F32_1D) || defined(ALL_KERNELS)
#define BGR2GRAYTYPE_1D(Function, base_type, T, T3)                            \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T3 input_value = vload3(element_x, src);                                     \
  T value = Function ## base_type ## Compute(input_value);                     \
                                                                               \
  dst[element_x] = value;                                                      \
}
#endif

#if defined(BGR2GRAY_U8_2D) || defined(BGR2GRAY_F32_2D) ||                     \
    defined(RGB2GRAY_U8_2D) || defined(RGB2GRAY_F32_2D) || defined(ALL_KERNELS)
#define BGR2GRAYTYPE_2D(Function, base_type, T, T3)                            \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T3 input_value = vload3(element_x, data);                                    \
  T value = Function ## base_type ## Compute(input_value);                     \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  data[element_x] = value;                                                     \
}
#endif

#if defined(BGR2GRAY_U8_1D) || defined(BGR2GRAY_U8_2D) || defined(ALL_KERNELS)
uchar BGR2GRAYU8Compute(const uchar3 src) {
  int b = src.x;
  int g = src.y;
  int r = src.z;
  uchar dst = divideUp(b * kB2Y15 + g * kG2Y15 + r * kR2Y15, kRgbShift);

  return dst;
}
#endif

#if defined(BGR2GRAY_F32_1D) || defined(BGR2GRAY_F32_2D) || defined(ALL_KERNELS)
float BGR2GRAYF32Compute(const float3 src) {
  float b = src.x;
  float g = src.y;
  float r = src.z;
  float dst = b * 0.114f + g * 0.587f + r * 0.299f;

  return dst;
}
#endif

#if defined(RGB2GRAY_U8_1D) || defined(RGB2GRAY_U8_2D) || defined(ALL_KERNELS)
uchar RGB2GRAYU8Compute(const uchar3 src) {
  int r = src.x;
  int g = src.y;
  int b = src.z;
  uchar dst = divideUp(r * kR2Y15 + g * kG2Y15 + b * kB2Y15, kRgbShift);

  return dst;
}
#endif

#if defined(RGB2GRAY_F32_1D) || defined(RGB2GRAY_F32_2D) || defined(ALL_KERNELS)
float RGB2GRAYF32Compute(const float3 src) {
  float r = src.x;
  float g = src.y;
  float b = src.z;
  float dst = r * 0.299f + g * 0.587f + b * 0.114f;

  return dst;
}
#endif

#if defined(BGR2GRAY_U8_1D) || defined(ALL_KERNELS)
BGR2GRAYTYPE_1D(BGR2GRAY, U8, uchar, uchar3)
#endif

#if defined(BGR2GRAY_U8_2D) || defined(ALL_KERNELS)
BGR2GRAYTYPE_2D(BGR2GRAY, U8, uchar, uchar3)
#endif

#if defined(BGR2GRAY_F32_1D) || defined(ALL_KERNELS)
BGR2GRAYTYPE_1D(BGR2GRAY, F32, float, float3)
#endif

#if defined(BGR2GRAY_F32_2D) || defined(ALL_KERNELS)
BGR2GRAYTYPE_2D(BGR2GRAY, F32, float, float3)
#endif

#if defined(RGB2GRAY_U8_1D) || defined(ALL_KERNELS)
BGR2GRAYTYPE_1D(RGB2GRAY, U8, uchar, uchar3)
#endif

#if defined(RGB2GRAY_U8_2D) || defined(ALL_KERNELS)
BGR2GRAYTYPE_2D(RGB2GRAY, U8, uchar, uchar3)
#endif

#if defined(RGB2GRAY_F32_1D) || defined(ALL_KERNELS)
BGR2GRAYTYPE_1D(RGB2GRAY, F32, float, float3)
#endif

#if defined(RGB2GRAY_F32_2D) || defined(ALL_KERNELS)
BGR2GRAYTYPE_2D(RGB2GRAY, F32, float, float3)
#endif

#if defined(BGRA2GRAY_U8_1D) || defined(BGRA2GRAY_F32_1D) ||                   \
    defined(RGBA2GRAY_U8_1D) || defined(RGBA2GRAY_F32_1D) ||                   \
    defined(ALL_KERNELS)
#define BGRA2GRAYTYPE_1D(Function, base_type, T, T4)                           \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T4 input_value = vload4(element_x, src);                                     \
  T value = Function ## base_type ## Compute(input_value);                     \
                                                                               \
  dst[element_x] = value;                                                      \
}
#endif

#if defined(BGRA2GRAY_U8_2D) || defined(BGRA2GRAY_F32_2D) ||                   \
    defined(RGBA2GRAY_U8_2D) || defined(RGBA2GRAY_F32_2D) ||                   \
    defined(ALL_KERNELS)
#define BGRA2GRAYTYPE_2D(Function, base_type, T, T4)                           \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T4 input_value = vload4(element_x, data);                                    \
  T value = Function ## base_type ## Compute(input_value);                     \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  data[element_x] = value;                                                     \
}
#endif

#if defined(BGRA2GRAY_U8_1D) || defined(BGRA2GRAY_U8_2D) || defined(ALL_KERNELS)
uchar BGRA2GRAYU8Compute(const uchar4 src) {
  int b = src.x;
  int g = src.y;
  int r = src.z;
  uchar dst = divideUp(b * kB2Y15 + g * kG2Y15 + r * kR2Y15, kRgbShift);

  return dst;
}
#endif

#if defined(BGRA2GRAY_F32_1D) || defined(BGRA2GRAY_F32_2D) ||                  \
    defined(ALL_KERNELS)
float BGRA2GRAYF32Compute(const float4 src) {
  float b = src.x;
  float g = src.y;
  float r = src.z;
  float dst = b * 0.114f + g * 0.587f + r * 0.299f;

  return dst;
}
#endif

#if defined(RGBA2GRAY_U8_1D) || defined(RGBA2GRAY_U8_2D) || defined(ALL_KERNELS)
uchar RGBA2GRAYU8Compute(const uchar4 src) {
  int r = src.x;
  int g = src.y;
  int b = src.z;
  uchar dst = divideUp(r * kR2Y15 + g * kG2Y15 + b * kB2Y15, kRgbShift);

  return dst;
}
#endif

#if defined(RGBA2GRAY_F32_1D) || defined(RGBA2GRAY_F32_2D) ||                  \
    defined(ALL_KERNELS)
float RGBA2GRAYF32Compute(const float4 src) {
  float r = src.x;
  float g = src.y;
  float b = src.z;
  float dst = r * 0.299f + g * 0.587f + b * 0.114f;

  return dst;
}
#endif

#if defined(BGRA2GRAY_U8_1D) || defined(ALL_KERNELS)
BGRA2GRAYTYPE_1D(BGRA2GRAY, U8, uchar, uchar4)
#endif

#if defined(BGRA2GRAY_U8_2D) || defined(ALL_KERNELS)
BGRA2GRAYTYPE_2D(BGRA2GRAY, U8, uchar, uchar4)
#endif

#if defined(BGRA2GRAY_F32_1D) || defined(ALL_KERNELS)
BGRA2GRAYTYPE_1D(BGRA2GRAY, F32, float, float4)
#endif

#if defined(BGRA2GRAY_F32_2D) || defined(ALL_KERNELS)
BGRA2GRAYTYPE_2D(BGRA2GRAY, F32, float, float4)
#endif

#if defined(RGBA2GRAY_U8_1D) || defined(ALL_KERNELS)
BGRA2GRAYTYPE_1D(RGBA2GRAY, U8, uchar, uchar4)
#endif

#if defined(RGBA2GRAY_U8_2D) || defined(ALL_KERNELS)
BGRA2GRAYTYPE_2D(RGBA2GRAY, U8, uchar, uchar4)
#endif

#if defined(RGBA2GRAY_F32_1D) || defined(ALL_KERNELS)
BGRA2GRAYTYPE_1D(RGBA2GRAY, F32, float, float4)
#endif

#if defined(RGBA2GRAY_F32_2D) || defined(ALL_KERNELS)
BGRA2GRAYTYPE_2D(RGBA2GRAY, F32, float, float4)
#endif

#if defined(GRAY2BGR_U8_1D) || defined(GRAY2BGR_F32_1D) ||                     \
    defined(GRAY2RGB_U8_1D) || defined(GRAY2RGB_F32_1D) || defined(ALL_KERNELS)
#define GRAY2BGRTYPE_1D(Function, base_type, T, T3)                            \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T input_value = src[element_x];                                              \
  T3 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  vstore3(value, element_x, dst);                                              \
}
#endif

#if defined(GRAY2BGR_U8_2D) || defined(GRAY2BGR_F32_2D) ||                     \
    defined(GRAY2RGB_U8_2D) || defined(GRAY2RGB_F32_2D) || defined(ALL_KERNELS)
#define GRAY2BGRTYPE_2D(Function, base_type, T, T3)                            \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T input_value = data[element_x];                                             \
  T3 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore3(value, element_x, data);                                             \
}
#endif

#if defined(GRAY2BGR_U8_1D) || defined(GRAY2BGR_U8_2D) || defined(ALL_KERNELS)
uchar3 GRAY2BGRU8Compute(const uchar src) {
  uchar3 dst;
  dst.x = src;
  dst.y = src;
  dst.z = src;

  return dst;
}
#endif

#if defined(GRAY2BGR_F32_1D) || defined(GRAY2BGR_F32_2D) || defined(ALL_KERNELS)
float3 GRAY2BGRF32Compute(const float src) {
  float3 dst;
  dst.x = src;
  dst.y = src;
  dst.z = src;

  return dst;
}
#endif

#if defined(GRAY2RGB_U8_1D) || defined(GRAY2RGB_U8_2D) || defined(ALL_KERNELS)
uchar3 GRAY2RGBU8Compute(const uchar src) {
  uchar3 dst;
  dst.x = src;
  dst.y = src;
  dst.z = src;

  return dst;
}
#endif

#if defined(GRAY2RGB_F32_1D) || defined(GRAY2RGB_F32_2D) || defined(ALL_KERNELS)
float3 GRAY2RGBF32Compute(const float src) {
  float3 dst;
  dst.x = src;
  dst.y = src;
  dst.z = src;

  return dst;
}
#endif

#if defined(GRAY2BGR_U8_1D) || defined(ALL_KERNELS)
GRAY2BGRTYPE_1D(GRAY2BGR, U8, uchar, uchar3)
#endif

#if defined(GRAY2BGR_U8_2D) || defined(ALL_KERNELS)
GRAY2BGRTYPE_2D(GRAY2BGR, U8, uchar, uchar3)
#endif

#if defined(GRAY2BGR_F32_1D) || defined(ALL_KERNELS)
GRAY2BGRTYPE_1D(GRAY2BGR, F32, float, float3)
#endif

#if defined(GRAY2BGR_F32_2D) || defined(ALL_KERNELS)
GRAY2BGRTYPE_2D(GRAY2BGR, F32, float, float3)
#endif

#if defined(GRAY2RGB_U8_1D) || defined(ALL_KERNELS)
GRAY2BGRTYPE_1D(GRAY2RGB, U8, uchar, uchar3)
#endif

#if defined(GRAY2RGB_U8_2D) || defined(ALL_KERNELS)
GRAY2BGRTYPE_2D(GRAY2RGB, U8, uchar, uchar3)
#endif

#if defined(GRAY2RGB_F32_1D) || defined(ALL_KERNELS)
GRAY2BGRTYPE_1D(GRAY2RGB, F32, float, float3)
#endif

#if defined(GRAY2RGB_F32_2D) || defined(ALL_KERNELS)
GRAY2BGRTYPE_2D(GRAY2RGB, F32, float, float3)
#endif

#if defined(GRAY2BGRA_U8_1D) || defined(GRAY2BGRA_F32_1D) ||                   \
    defined(GRAY2RGBA_U8_1D) || defined(GRAY2RGBA_F32_1D) ||                   \
    defined(ALL_KERNELS)
#define GRAY2BGRATYPE_1D(Function, base_type, T, T4)                           \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T input_value = src[element_x];                                              \
  T4 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  vstore4(value, element_x, dst);                                              \
}
#endif

#if defined(GRAY2BGRA_U8_2D) || defined(GRAY2BGRA_F32_2D) ||                   \
    defined(GRAY2RGBA_U8_2D) || defined(GRAY2RGBA_F32_2D) ||                   \
    defined(ALL_KERNELS)
#define GRAY2BGRATYPE_2D(Function, base_type, T, T4)                           \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T input_value = data[element_x];                                             \
  T4 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore4(value, element_x, data);                                             \
}
#endif

#if defined(GRAY2BGRA_U8_1D) || defined(GRAY2BGRA_U8_2D) || defined(ALL_KERNELS)
uchar4 GRAY2BGRAU8Compute(const uchar src) {
  uchar4 dst;
  dst.x = src;
  dst.y = src;
  dst.z = src;
  dst.w = 255;

  return dst;
}
#endif

#if defined(GRAY2BGRA_F32_1D) || defined(GRAY2BGRA_F32_2D) ||                  \
    defined(ALL_KERNELS)
float4 GRAY2BGRAF32Compute(const float src) {
  float4 dst;
  dst.x = src;
  dst.y = src;
  dst.z = src;
  dst.w = 1.0f;

  return dst;
}
#endif

#if defined(GRAY2RGBA_U8_1D) || defined(GRAY2RGBA_U8_2D) || defined(ALL_KERNELS)
uchar4 GRAY2RGBAU8Compute(const uchar src) {
  uchar4 dst;
  dst.x = src;
  dst.y = src;
  dst.z = src;
  dst.w = 255;

  return dst;
}
#endif

#if defined(GRAY2RGBA_F32_1D) || defined(GRAY2RGBA_F32_2D) ||                  \
    defined(ALL_KERNELS)
float4 GRAY2RGBAF32Compute(const float src) {
  float4 dst;
  dst.x = src;
  dst.y = src;
  dst.z = src;
  dst.w = 1.0f;

  return dst;
}
#endif

#if defined(GRAY2BGRA_U8_1D) || defined(ALL_KERNELS)
GRAY2BGRATYPE_1D(GRAY2BGRA, U8, uchar, uchar4)
#endif

#if defined(GRAY2BGRA_U8_2D) || defined(ALL_KERNELS)
GRAY2BGRATYPE_2D(GRAY2BGRA, U8, uchar, uchar4)
#endif

#if defined(GRAY2BGRA_F32_1D) || defined(ALL_KERNELS)
GRAY2BGRATYPE_1D(GRAY2BGRA, F32, float, float4)
#endif

#if defined(GRAY2BGRA_F32_2D) || defined(ALL_KERNELS)
GRAY2BGRATYPE_2D(GRAY2BGRA, F32, float, float4)
#endif

#if defined(GRAY2RGBA_U8_1D) || defined(ALL_KERNELS)
GRAY2BGRATYPE_1D(GRAY2RGBA, U8, uchar, uchar4)
#endif

#if defined(GRAY2RGBA_U8_2D) || defined(ALL_KERNELS)
GRAY2BGRATYPE_2D(GRAY2RGBA, U8, uchar, uchar4)
#endif

#if defined(GRAY2RGBA_F32_1D) || defined(ALL_KERNELS)
GRAY2BGRATYPE_1D(GRAY2RGBA, F32, float, float4)
#endif

#if defined(GRAY2RGBA_F32_2D) || defined(ALL_KERNELS)
GRAY2BGRATYPE_2D(GRAY2RGBA, F32, float, float4)
#endif

/*********************** BGR/RGB/BGRA/RGBA <-> YCrCb ************************/

#define R2Y_FLOAT_COEFF 0.299f
#define G2Y_FLOAT_COEFF 0.587f
#define B2Y_FLOAT_COEFF 0.114f
#define CR_FLOAT_COEFF 0.713f
#define CB_FLOAT_COEFF 0.564f
#define YCRCB_UCHAR_DELTA 128
#define YCRCB_FLOAT_DELTA 0.5f
#define CR2R_FLOAT_COEFF 1.403f
#define CB2R_FLOAT_COEFF 1.773f
#define Y2G_CR_FLOAT_COEFF -0.714f
#define Y2G_CB_FLOAT_COEFF -0.344f

enum YCrCbIntegerCoefficients1 {
  kB2YCoeff = 1868,
  kG2YCoeff = 9617,
  kR2YCoeff = 4899,
  kCRCoeff  = 11682,
  kCBCoeff  = 9241,
};

enum YCrCbIntegerCoefficients2 {
  kCr2RCoeff  = 22987,
  kCb2BCoeff  = 29049,
  kY2GCrCoeff = -11698,
  kY2GCbCoeff = -5636,
};

enum YCrCbShifts {
  kYCrCbShift   = 14,
  kShift14Delta = 2097152,
};

#if defined(BGR2YCrCb_U8_1D) || defined(BGR2YCrCb_F32_1D) ||                   \
    defined(RGB2YCrCb_U8_1D) || defined(RGB2YCrCb_F32_1D) ||                   \
    defined(YCrCb2BGR_U8_1D) || defined(YCrCb2BGR_F32_1D) ||                   \
    defined(YCrCb2RGB_U8_1D) || defined(YCrCb2RGB_F32_1D) ||                   \
    defined(BGR2HSV_U8_1D) || defined(BGR2HSV_F32_1D) ||                       \
    defined(RGB2HSV_U8_1D) || defined(RGB2HSV_F32_1D) ||                       \
    defined(HSV2BGR_U8_1D) || defined(HSV2BGR_F32_1D) ||                       \
    defined(HSV2RGB_U8_1D) || defined(HSV2RGB_F32_1D) ||                       \
    defined(BGR2LAB_U8_1D) || defined(BGR2LAB_F32_1D) ||                       \
    defined(RGB2LAB_U8_1D) || defined(RGB2LAB_F32_1D) ||                       \
    defined(LAB2BGR_U8_1D) || defined(LAB2BGR_F32_1D) ||                       \
    defined(LAB2RGB_U8_1D) || defined(LAB2RGB_F32_1D) || defined(ALL_KERNELS)
#define Convert3To3_1D(Function, base_type, T, T3)                             \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T3 input_value = vload3(element_x, src);                                     \
  T3 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  vstore3(value, element_x, dst);                                              \
}
#endif

#if defined(BGR2YCrCb_U8_2D) || defined(BGR2YCrCb_F32_2D) ||                   \
    defined(RGB2YCrCb_U8_2D) || defined(RGB2YCrCb_F32_2D) ||                   \
    defined(YCrCb2BGR_U8_2D) || defined(YCrCb2BGR_F32_2D) ||                   \
    defined(YCrCb2RGB_U8_2D) || defined(YCrCb2RGB_F32_2D) ||                   \
    defined(BGR2HSV_U8_2D) || defined(BGR2HSV_F32_2D) ||                       \
    defined(RGB2HSV_U8_2D) || defined(RGB2HSV_F32_2D) ||                       \
    defined(HSV2BGR_U8_2D) || defined(HSV2BGR_F32_2D) ||                       \
    defined(HSV2RGB_U8_2D) || defined(HSV2RGB_F32_2D) ||                       \
    defined(BGR2LAB_U8_2D) || defined(BGR2LAB_F32_2D) ||                       \
    defined(RGB2LAB_U8_2D) || defined(RGB2LAB_F32_2D) ||                       \
    defined(LAB2BGR_U8_2D) || defined(LAB2BGR_F32_2D) ||                       \
    defined(LAB2RGB_U8_2D) || defined(LAB2RGB_F32_2D) || defined(ALL_KERNELS)
#define Convert3To3_2D(Function, base_type, T, T3)                             \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T3 input_value = vload3(element_x, data);                                    \
  T3 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore3(value, element_x, data);                                             \
}
#endif

#if defined(BGR2YCrCb_U8_1D) || defined(BGR2YCrCb_U8_2D) || defined(ALL_KERNELS)
uchar3 BGR2YCrCbU8Compute(const uchar3 src) {
  int3 value;
  value.x = divideUp(src.z * kR2YCoeff + src.y * kG2YCoeff + src.x * kB2YCoeff,
                     kYCrCbShift);
  value.y = divideUp((src.z - value.x) * kCRCoeff + kShift14Delta, kYCrCbShift);
  value.z = divideUp((src.x - value.x) * kCBCoeff + kShift14Delta, kYCrCbShift);

  uchar3 dst = convert_uchar3_sat(value);

  return dst;
}
#endif

#if defined(BGR2YCrCb_F32_1D) || defined(BGR2YCrCb_F32_2D) ||                  \
    defined(ALL_KERNELS)
float3 BGR2YCrCbF32Compute(const float3 src) {
  float3 dst;
  dst.x = src.z * R2Y_FLOAT_COEFF + src.y * G2Y_FLOAT_COEFF +
          src.x * B2Y_FLOAT_COEFF;
  dst.y = (src.z - dst.x) * CR_FLOAT_COEFF + YCRCB_FLOAT_DELTA;
  dst.z = (src.x - dst.x) * CB_FLOAT_COEFF + YCRCB_FLOAT_DELTA;

  return dst;
}
#endif

#if defined(RGB2YCrCb_U8_1D) || defined(RGB2YCrCb_U8_2D) || defined(ALL_KERNELS)
uchar3 RGB2YCrCbU8Compute(const uchar3 src) {
  int3 value;
  value.x = divideUp(src.x * kR2YCoeff + src.y * kG2YCoeff + src.z * kB2YCoeff,
                     kYCrCbShift);
  value.y = divideUp((src.x - value.x) * kCRCoeff + kShift14Delta, kYCrCbShift);
  value.z = divideUp((src.z - value.x) * kCBCoeff + kShift14Delta, kYCrCbShift);

  uchar3 dst = convert_uchar3_sat(value);

  return dst;
}
#endif

#if defined(RGB2YCrCb_F32_1D) || defined(RGB2YCrCb_F32_2D) ||                  \
    defined(ALL_KERNELS)
float3 RGB2YCrCbF32Compute(const float3 src) {
  float3 dst;
  dst.x = src.x * R2Y_FLOAT_COEFF + src.y * G2Y_FLOAT_COEFF +
          src.z * B2Y_FLOAT_COEFF;
  dst.y = (src.x - dst.x) * CR_FLOAT_COEFF + YCRCB_FLOAT_DELTA;
  dst.z = (src.z - dst.x) * CB_FLOAT_COEFF + YCRCB_FLOAT_DELTA;

  return dst;
}
#endif

#if defined(BGR2YCrCb_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(BGR2YCrCb, U8, uchar, uchar3)
#endif

#if defined(BGR2YCrCb_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(BGR2YCrCb, U8, uchar, uchar3)
#endif

#if defined(BGR2YCrCb_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(BGR2YCrCb, F32, float, float3)
#endif

#if defined(BGR2YCrCb_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(BGR2YCrCb, F32, float, float3)
#endif

#if defined(RGB2YCrCb_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(RGB2YCrCb, U8, uchar, uchar3)
#endif

#if defined(RGB2YCrCb_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(RGB2YCrCb, U8, uchar, uchar3)
#endif

#if defined(RGB2YCrCb_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(RGB2YCrCb, F32, float, float3)
#endif

#if defined(RGB2YCrCb_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(RGB2YCrCb, F32, float, float3)
#endif


#if defined(BGRA2YCrCb_U8_1D) || defined(BGRA2YCrCb_F32_1D) ||                 \
    defined(RGBA2YCrCb_U8_1D) || defined(RGBA2YCrCb_F32_1D) ||                 \
    defined(BGRA2HSV_U8_1D) || defined(BGRA2HSV_F32_1D) ||                     \
    defined(RGBA2HSV_U8_1D) || defined(RGBA2HSV_F32_1D) ||                     \
    defined(BGRA2LAB_U8_1D) || defined(BGRA2LAB_F32_1D) ||                     \
    defined(RGBA2LAB_U8_1D) || defined(RGBA2LAB_F32_1D) || defined(ALL_KERNELS)
#define Convert4To3_1D(Function, base_type, T, T4, T3)                         \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T4 input_value = vload4(element_x, src);                                     \
  T3 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  vstore3(value, element_x, dst);                                              \
}
#endif

#if defined(BGRA2YCrCb_U8_2D) || defined(BGRA2YCrCb_F32_2D) ||                 \
    defined(RGBA2YCrCb_U8_2D) || defined(RGBA2YCrCb_F32_2D) ||                 \
    defined(BGRA2YHSV_U8_2D) || defined(BGRA2YHSV_F32_2D) ||                   \
    defined(RGBA2YHSV_U8_2D) || defined(RGBA2YHSV_F32_2D) ||                   \
    defined(BGRA2YLAB_U8_2D) || defined(BGRA2YLAB_F32_2D) ||                   \
    defined(RGBA2YLAB_U8_2D) || defined(RGBA2YLAB_F32_2D) ||                   \
    defined(ALL_KERNELS)
#define Convert4To3_2D(Function, base_type, T, T4, T3)                         \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T4 input_value = vload4(element_x, data);                                    \
  T3 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore3(value, element_x, data);                                             \
}
#endif

#if defined(BGRA2YCrCb_U8_1D) || defined(BGRA2YCrCb_U8_2D) ||                  \
    defined(ALL_KERNELS)
uchar3 BGRA2YCrCbU8Compute(const uchar4 src) {
  int3 value;
  value.x = divideUp(src.z * kR2YCoeff + src.y * kG2YCoeff + src.x * kB2YCoeff,
                     kYCrCbShift);
  value.y = divideUp((src.z - value.x) * kCRCoeff + kShift14Delta, kYCrCbShift);
  value.z = divideUp((src.x - value.x) * kCBCoeff + kShift14Delta, kYCrCbShift);

  uchar3 dst = convert_uchar3_sat(value);

  return dst;
}
#endif

#if defined(BGRA2YCrCb_F32_1D) || defined(BGRA2YCrCb_F32_2D) ||                \
    defined(ALL_KERNELS)
float3 BGRA2YCrCbF32Compute(const float4 src) {
  float3 dst;
  dst.x = src.z * R2Y_FLOAT_COEFF + src.y * G2Y_FLOAT_COEFF +
          src.x * B2Y_FLOAT_COEFF;
  dst.y = (src.z - dst.x) * CR_FLOAT_COEFF + YCRCB_FLOAT_DELTA;
  dst.z = (src.x - dst.x) * CB_FLOAT_COEFF + YCRCB_FLOAT_DELTA;

  return dst;
}
#endif

#if defined(RGBA2YCrCb_U8_1D) || defined(RGBA2YCrCb_U8_2D) ||                  \
    defined(ALL_KERNELS)
uchar3 RGBA2YCrCbU8Compute(const uchar4 src) {
  int3 value;
  value.x = divideUp(src.x * kR2YCoeff + src.y * kG2YCoeff + src.z * kB2YCoeff,
                     kYCrCbShift);
  value.y = divideUp((src.x - value.x) * kCRCoeff + kShift14Delta, kYCrCbShift);
  value.z = divideUp((src.z - value.x) * kCBCoeff + kShift14Delta, kYCrCbShift);

  uchar3 dst = convert_uchar3_sat(value);

  return dst;
}
#endif

#if defined(RGBA2YCrCb_F32_1D) || defined(RGBA2YCrCb_F32_2D) ||                \
    defined(ALL_KERNELS)
float3 RGBA2YCrCbF32Compute(const float4 src) {
  float3 dst;
  dst.x = src.x * R2Y_FLOAT_COEFF + src.y * G2Y_FLOAT_COEFF +
          src.z * B2Y_FLOAT_COEFF;
  dst.y = (src.x - dst.x) * CR_FLOAT_COEFF + YCRCB_FLOAT_DELTA;
  dst.z = (src.z - dst.x) * CB_FLOAT_COEFF + YCRCB_FLOAT_DELTA;

  return dst;
}
#endif

#if defined(BGRA2YCrCb_U8_1D) || defined(ALL_KERNELS)
Convert4To3_1D(BGRA2YCrCb, U8, uchar, uchar4, uchar3)
#endif

#if defined(BGRA2YCrCb_U8_2D) || defined(ALL_KERNELS)
Convert4To3_2D(BGRA2YCrCb, U8, uchar, uchar4, uchar3)
#endif

#if defined(BGRA2YCrCb_F32_1D) || defined(ALL_KERNELS)
Convert4To3_1D(BGRA2YCrCb, F32, float, float4, float3)
#endif

#if defined(BGRA2YCrCb_F32_2D) || defined(ALL_KERNELS)
Convert4To3_2D(BGRA2YCrCb, F32, float, float4, float3)
#endif

#if defined(RGBA2YCrCb_U8_1D) || defined(ALL_KERNELS)
Convert4To3_1D(RGBA2YCrCb, U8, uchar, uchar4, uchar3)
#endif

#if defined(RGBA2YCrCb_U8_2D) || defined(ALL_KERNELS)
Convert4To3_2D(RGBA2YCrCb, U8, uchar, uchar4, uchar3)
#endif

#if defined(RGBA2YCrCb_F32_1D) || defined(ALL_KERNELS)
Convert4To3_1D(RGBA2YCrCb, F32, float, float4, float3)
#endif

#if defined(RGBA2YCrCb_F32_2D) || defined(ALL_KERNELS)
Convert4To3_2D(RGBA2YCrCb, F32, float, float4, float3)
#endif


#if defined(YCrCb2BGR_U8_1D) || defined(YCrCb2BGR_U8_2D) ||                    \
    defined(ALL_KERNELS)
uchar3 YCrCb2BGRU8Compute(const uchar3 src) {
  int y  = src.x;
  int cr = src.y - YCRCB_UCHAR_DELTA;
  int cb = src.z - YCRCB_UCHAR_DELTA;

  int3 value;
  value.x = y + divideUp(cb * kCb2BCoeff, kYCrCbShift);
  value.y = y + divideUp(cr * kY2GCrCoeff + cb * kY2GCbCoeff, kYCrCbShift);
  value.z = y + divideUp(cr * kCr2RCoeff, kYCrCbShift);

  uchar3 dst = convert_uchar3_sat(value);

  return dst;
}
#endif

#if defined(YCrCb2BGR_F32_1D) || defined(YCrCb2BGR_F32_2D) ||                  \
    defined(ALL_KERNELS)
float3 YCrCb2BGRF32Compute(const float3 src) {
  float y  = src.x;
  float cr = src.y - YCRCB_FLOAT_DELTA;
  float cb = src.z - YCRCB_FLOAT_DELTA;

  float3 dst;
  dst.x = y + cb * CB2R_FLOAT_COEFF;
  dst.y = y + cr * Y2G_CR_FLOAT_COEFF + cb * Y2G_CB_FLOAT_COEFF;
  dst.z = y + cr * CR2R_FLOAT_COEFF;

  return dst;
}
#endif

#if defined(YCrCb2RGB_U8_1D) || defined(YCrCb2RGB_U8_2D) || defined(ALL_KERNELS)
uchar3 YCrCb2RGBU8Compute(const uchar3 src) {
  int y  = src.x;
  int cr = src.y - YCRCB_UCHAR_DELTA;
  int cb = src.z - YCRCB_UCHAR_DELTA;

  int3 value;
  value.x = y + divideUp(cr * kCr2RCoeff, kYCrCbShift);
  value.y = y + divideUp(cr * kY2GCrCoeff + cb * kY2GCbCoeff, kYCrCbShift);
  value.z = y + divideUp(cb * kCb2BCoeff, kYCrCbShift);

  uchar3 dst = convert_uchar3_sat(value);

  return dst;
}
#endif

#if defined(YCrCb2RGB_F32_1D) || defined(YCrCb2RGB_F32_2D) ||                  \
    defined(ALL_KERNELS)
float3 YCrCb2RGBF32Compute(const float3 src) {
  float y  = src.x;
  float cr = src.y - YCRCB_FLOAT_DELTA;
  float cb = src.z - YCRCB_FLOAT_DELTA;

  float3 dst;
  dst.x = y + cr * CR2R_FLOAT_COEFF;
  dst.y = y + cr * Y2G_CR_FLOAT_COEFF + cb * Y2G_CB_FLOAT_COEFF;
  dst.z = y + cb * CB2R_FLOAT_COEFF;

  return dst;
}
#endif

#if defined(YCrCb2BGR_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(YCrCb2BGR, U8, uchar, uchar3)
#endif

#if defined(YCrCb2BGR_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(YCrCb2BGR, U8, uchar, uchar3)
#endif

#if defined(YCrCb2BGR_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(YCrCb2BGR, F32, float, float3)
#endif

#if defined(YCrCb2BGR_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(YCrCb2BGR, F32, float, float3)
#endif

#if defined(YCrCb2RGB_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(YCrCb2RGB, U8, uchar, uchar3)
#endif

#if defined(YCrCb2RGB_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(YCrCb2RGB, U8, uchar, uchar3)
#endif

#if defined(YCrCb2RGB_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(YCrCb2RGB, F32, float, float3)
#endif

#if defined(YCrCb2RGB_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(YCrCb2RGB, F32, float, float3)
#endif


#if defined(YCrCb2BGRA_U8_1D) || defined(YCrCb2BGRA_F32_1D) ||                 \
    defined(YCrCb2RGBA_U8_1D) || defined(YCrCb2RGBA_F32_1D) ||                 \
    defined(HSV2BGRA_U8_1D) || defined(HSV2BGRA_F32_1D) ||                     \
    defined(HSV2RGBA_U8_1D) || defined(HSV2RGBA_F32_1D) ||                     \
    defined(LAB2BGRA_U8_1D) || defined(LAB2BGRA_F32_1D) ||                     \
    defined(LAB2RGBA_U8_1D) || defined(LAB2RGBA_F32_1D) || defined(ALL_KERNELS)
#define Convert3To4_1D(Function, base_type, T, T3, T4)                         \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  if (element_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T3 input_value = vload3(element_x, src);                                     \
  T4 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  vstore4(value, element_x, dst);                                              \
}
#endif

#if defined(YCrCb2BGRA_U8_2D) || defined(YCrCb2BGRA_F32_2D) ||                 \
    defined(YCrCb2RGBA_U8_2D) || defined(YCrCb2RGBA_F32_2D) ||                 \
    defined(HSV2BGRA_U8_2D) || defined(HSV2BGRA_F32_2D) ||                     \
    defined(HSV2RGBA_U8_2D) || defined(HSV2RGBA_F32_2D) ||                     \
    defined(LAB2BGRA_U8_2D) || defined(LAB2BGRA_F32_2D) ||                     \
    defined(LAB2RGBA_U8_2D) || defined(LAB2RGBA_F32_2D) || defined(ALL_KERNELS)
#define Convert3To4_2D(Function, base_type, T, T3, T4)                         \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const T* src, int rows, int cols, \
                                      int src_stride, global T* dst,           \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global T* data = (global T*)((global uchar*)src + element_y * src_stride);   \
  T3 input_value = vload3(element_x, data);                                    \
  T4 value = Function ## base_type ## Compute(input_value);                    \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore4(value, element_x, data);                                             \
}
#endif

#if defined(YCrCb2BGRA_U8_1D) || defined(YCrCb2BGRA_U8_2D) ||                  \
    defined(ALL_KERNELS)
uchar4 YCrCb2BGRAU8Compute(const uchar3 src) {
  int y  = src.x;
  int cr = src.y - YCRCB_UCHAR_DELTA;
  int cb = src.z - YCRCB_UCHAR_DELTA;

  int4 value;
  value.x = y + divideUp(cb * kCb2BCoeff, kYCrCbShift);
  value.y = y + divideUp(cr * kY2GCrCoeff + cb * kY2GCbCoeff, kYCrCbShift);
  value.z = y + divideUp(cr * kCr2RCoeff, kYCrCbShift);
  value.w = 255.f;

  uchar4 dst = convert_uchar4_sat(value);

  return dst;
}
#endif

#if defined(YCrCb2BGRA_F32_1D) || defined(YCrCb2BGRA_F32_2D) ||                \
    defined(ALL_KERNELS)
float4 YCrCb2BGRAF32Compute(const float3 src) {
  float y  = src.x;
  float cr = src.y - YCRCB_FLOAT_DELTA;
  float cb = src.z - YCRCB_FLOAT_DELTA;

  float4 dst;
  dst.x = y + cb * CB2R_FLOAT_COEFF;
  dst.y = y + cr * Y2G_CR_FLOAT_COEFF + cb * Y2G_CB_FLOAT_COEFF;
  dst.z = y + cr * CR2R_FLOAT_COEFF;
  dst.w = 1.0f;

  return dst;
}
#endif

#if defined(YCrCb2RGBA_U8_1D) || defined(YCrCb2RGBA_U8_2D) ||                  \
    defined(ALL_KERNELS)
uchar4 YCrCb2RGBAU8Compute(const uchar3 src) {
  int y  = src.x;
  int cr = src.y - YCRCB_UCHAR_DELTA;
  int cb = src.z - YCRCB_UCHAR_DELTA;

  int4 value;
  value.x = y + divideUp(cr * kCr2RCoeff, kYCrCbShift);
  value.y = y + divideUp(cr * kY2GCrCoeff + cb * kY2GCbCoeff, kYCrCbShift);
  value.z = y + divideUp(cb * kCb2BCoeff, kYCrCbShift);
  value.w = 255.f;

  uchar4 dst = convert_uchar4_sat(value);

  return dst;
}
#endif

#if defined(YCrCb2RGBA_F32_1D) || defined(YCrCb2RGBA_F32_2D) ||                \
    defined(ALL_KERNELS)
float4 YCrCb2RGBAF32Compute(const float3 src) {
  float y  = src.x;
  float cr = src.y - YCRCB_FLOAT_DELTA;
  float cb = src.z - YCRCB_FLOAT_DELTA;

  float4 dst;
  dst.x = y + cr * CR2R_FLOAT_COEFF;
  dst.y = y + cr * Y2G_CR_FLOAT_COEFF + cb * Y2G_CB_FLOAT_COEFF;
  dst.z = y + cb * CB2R_FLOAT_COEFF;
  dst.w = 1.0f;

  return dst;
}
#endif

#if defined(YCrCb2BGRA_U8_1D) || defined(ALL_KERNELS)
Convert3To4_1D(YCrCb2BGRA, U8, uchar, uchar3, uchar4)
#endif

#if defined(YCrCb2BGRA_U8_2D) || defined(ALL_KERNELS)
Convert3To4_2D(YCrCb2BGRA, U8, uchar, uchar3, uchar4)
#endif

#if defined(YCrCb2BGRA_F32_1D) || defined(ALL_KERNELS)
Convert3To4_1D(YCrCb2BGRA, F32, float, float3, float4)
#endif

#if defined(YCrCb2BGRA_F32_2D) || defined(ALL_KERNELS)
Convert3To4_2D(YCrCb2BGRA, F32, float, float3, float4)
#endif

#if defined(YCrCb2RGBA_U8_1D) || defined(ALL_KERNELS)
Convert3To4_1D(YCrCb2RGBA, U8, uchar, uchar3, uchar4)
#endif

#if defined(YCrCb2RGBA_U8_2D) || defined(ALL_KERNELS)
Convert3To4_2D(YCrCb2RGBA, U8, uchar, uchar3, uchar4)
#endif

#if defined(YCrCb2RGBA_F32_1D) || defined(ALL_KERNELS)
Convert3To4_1D(YCrCb2RGBA, F32, float, float3, float4)
#endif

#if defined(YCrCb2RGBA_F32_2D) || defined(ALL_KERNELS)
Convert3To4_2D(YCrCb2RGBA, F32, float, float3, float4)
#endif

/*********************** BGR/RGB/BGRA/RGBA <-> HSV ************************/

#if defined(BGR2HSV_U8_1D) || defined(BGR2HSV_U8_2D) || defined(ALL_KERNELS)
uchar3 BGR2HSVU8Compute(const uchar3 src) {
  int h, v, s;
  v = max(max(src.x, src.y), src.z);
  s = min(min(src.x, src.y), src.z);
  uchar diff = convert_uchar_sat(v - s);
  if (v == 0) {
    s = 0;
  }
  else {
    s = (diff * ((255 << 12) / v) + (1 << 11)) >> 12;
  }

  if (diff == 0) {
    h = 0;
  }
  else {
    int value = convert_int_sat_rte((180 << 12) / (6.f * diff));
    if (v == src.z) {
      h = ((src.y - src.x) * value + (1 << 11)) >> 12;
    }
    if (v == src.y) {
      h = ((src.x - src.z + 2 * diff) * value + (1 << 11)) >> 12;
    }
    if (v == src.x) {
      h = ((src.z - src.y + 4 * diff) * value + (1 << 11)) >> 12;
    }
  }
  if (h < 0) {
    h += 180;
  }

  uchar3 dst = convert_uchar3_sat((int3)(h, s, v));

  return dst;
}
#endif

#if defined(BGR2HSV_F32_1D) || defined(BGR2HSV_F32_2D) || defined(ALL_KERNELS)
float3 BGR2HSVF32Compute(const float3 src) {
  float diff;
  float3 dst;

  dst.z = max(max(src.z, src.y), src.x);
  diff = dst.z - min(min(src.z, src.y), src.x);
  dst.y = diff / (float)(dst.z + FLT_EPSILON);

  diff = (float)(60.0f / (diff + FLT_EPSILON));
  if (dst.z == src.z) {
    dst.x = (src.y - src.x) * diff;
  }
  if (dst.z == src.y) {
    dst.x = (src.x - src.z) * diff + 120.0f;
  }
  if (dst.z == src.x) {
    dst.x = (src.z - src.y) * diff + 240.0f;
  }
  if (dst.x < 0.0f) {
    dst.x += 360.0f;
  }

  return dst;
}
#endif

#if defined(RGB2HSV_U8_1D) || defined(RGB2HSV_U8_2D) || defined(ALL_KERNELS)
uchar3 RGB2HSVU8Compute(const uchar3 src) {
  int h, v, s;
  v = max(max(src.z, src.y), src.x);
  s = min(min(src.z, src.y), src.x);
  uchar diff = convert_uchar_sat(v - s);
  if (v == 0) {
    s = 0;
  }
  else {
    s = (diff * ((255 << 12) / v) + (1 << 11)) >> 12;
  }

  if (diff == 0) {
    h = 0;
  }
  else {
    int value = convert_int_sat_rte((180 << 12) / (6.f * diff));
    if (v == src.x) {
      h = ((src.y - src.z) * value + (1 << 11)) >> 12;
    }
    if (v == src.y) {
      h = ((src.z - src.x + 2 * diff) * value + (1 << 11)) >> 12;
    }
    if (v == src.z) {
      h = ((src.x - src.y + 4 * diff) * value + (1 << 11)) >> 12;
    }
  }
  if (h < 0) {
    h += 180;
  }

  uchar3 dst = convert_uchar3_sat((int3)(h, s, v));

  return dst;
}
#endif

#if defined(RGB2HSV_F32_1D) || defined(RGB2HSV_F32_2D) || defined(ALL_KERNELS)
float3 RGB2HSVF32Compute(const float3 src) {
  float diff;
  float3 dst;

  dst.z = max(max(src.x, src.y), src.z);
  diff = dst.z - min(min(src.x, src.y), src.z);
  dst.y = diff / (float)(dst.z + FLT_EPSILON);

  diff = (float)(60.0f / (diff + FLT_EPSILON));
  if (dst.z == src.x) {
    dst.x = (src.y - src.z) * diff;
  }
  if (dst.z == src.y) {
    dst.x = (src.z - src.x) * diff + 120.0f;
  }
  if (dst.z == src.z) {
    dst.x = (src.x - src.y) * diff + 240.0f;
  }
  if (dst.x < 0.0f) {
    dst.x += 360.0f;
  }

  return dst;
}
#endif

#if defined(BGR2HSV_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(BGR2HSV, U8, uchar, uchar3)
#endif

#if defined(BGR2HSV_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(BGR2HSV, U8, uchar, uchar3)
#endif

#if defined(BGR2HSV_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(BGR2HSV, F32, float, float3)
#endif

#if defined(BGR2HSV_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(BGR2HSV, F32, float, float3)
#endif

#if defined(RGB2HSV_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(RGB2HSV, U8, uchar, uchar3)
#endif

#if defined(RGB2HSV_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(RGB2HSV, U8, uchar, uchar3)
#endif

#if defined(RGB2HSV_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(RGB2HSV, F32, float, float3)
#endif

#if defined(RGB2HSV_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(RGB2HSV, F32, float, float3)
#endif


#if defined(BGRA2HSV_U8_1D) || defined(BGRA2HSV_U8_2D) || defined(ALL_KERNELS)
uchar3 BGRA2HSVU8Compute(const uchar4 src) {
  int h, v, s;
  v = max(max(src.x, src.y), src.z);
  s = min(min(src.x, src.y), src.z);
  uchar diff = convert_uchar_sat(v - s);
  if (v == 0) {
    s = 0;
  }
  else {
    s = (diff * ((255 << 12) / v) + (1 << 11)) >> 12;
  }

  if (diff == 0) {
    h = 0;
  }
  else {
    int value = convert_int_sat_rte((180 << 12) / (6.f * diff));
    if (v == src.z) {
      h = ((src.y - src.x) * value + (1 << 11)) >> 12;
    }
    if (v == src.y) {
      h = ((src.x - src.z + 2 * diff) * value + (1 << 11)) >> 12;
    }
    if (v == src.x) {
      h = ((src.z - src.y + 4 * diff) * value + (1 << 11)) >> 12;
    }
  }
  if (h < 0) {
    h += 180;
  }

  uchar3 dst = convert_uchar3_sat((int3)(h, s, v));

  return dst;
}
#endif

#if defined(BGRA2HSV_F32_1D) || defined(BGRA2HSV_F32_2D) || defined(ALL_KERNELS)
float3 BGRA2HSVF32Compute(const float4 src) {
  float diff;
  float3 dst;

  dst.z = max(max(src.z, src.y), src.x);
  diff = dst.z - min(min(src.z, src.y), src.x);
  dst.y = diff / (float)(dst.z + FLT_EPSILON);

  diff = (float)(60.0f / (diff + FLT_EPSILON));
  if (dst.z == src.z) {
    dst.x = (src.y - src.x) * diff;
  }
  if (dst.z == src.y) {
    dst.x = (src.x - src.z) * diff + 120.0f;
  }
  if (dst.z == src.x) {
    dst.x = (src.z - src.y) * diff + 240.0f;
  }
  if (dst.x < 0.0f) {
    dst.x += 360.0f;
  }

  return dst;
}
#endif

#if defined(RGBA2HSV_U8_1D) || defined(RGBA2HSV_U8_2D) || defined(ALL_KERNELS)
uchar3 RGBA2HSVU8Compute(const uchar4 src) {
  int h, v, s;
  v = max(max(src.z, src.y), src.x);
  s = min(min(src.z, src.y), src.x);
  uchar diff = convert_uchar_sat(v - s);
  if (v == 0) {
    s = 0;
  }
  else {
    s = (diff * ((255 << 12) / v) + (1 << 11)) >> 12;
  }

  if (diff == 0) {
    h = 0;
  }
  else {
    int value = convert_int_sat_rte((180 << 12) / (6.f * diff));
    if (v == src.x) {
      h = ((src.y - src.z) * value + (1 << 11)) >> 12;
    }
    if (v == src.y) {
      h = ((src.z - src.x + 2 * diff) * value + (1 << 11)) >> 12;
    }
    if (v == src.z) {
      h = ((src.x - src.y + 4 * diff) * value + (1 << 11)) >> 12;
    }
  }
  if (h < 0) {
    h += 180;
  }

  uchar3 dst = convert_uchar3_sat((int3)(h, s, v));

  return dst;
}
#endif

#if defined(RGBA2HSV_F32_1D) || defined(RGBA2HSV_F32_2D) || defined(ALL_KERNELS)
float3 RGBA2HSVF32Compute(const float4 src) {
  float diff;
  float3 dst;

  dst.z = max(max(src.x, src.y), src.z);
  diff = dst.z - min(min(src.x, src.y), src.z);
  dst.y = diff / (float)(dst.z + FLT_EPSILON);

  diff = (float)(60.0f / (diff + FLT_EPSILON));
  if (dst.z == src.x) {
    dst.x = (src.y - src.z) * diff;
  }
  if (dst.z == src.y) {
    dst.x = (src.z - src.x) * diff + 120.0f;
  }
  if (dst.z == src.z) {
    dst.x = (src.x - src.y) * diff + 240.0f;
  }
  if (dst.x < 0.0f) {
    dst.x += 360.0f;
  }

  return dst;
}
#endif

#if defined(BGRA2HSV_U8_1D) || defined(ALL_KERNELS)
Convert4To3_1D(BGRA2HSV, U8, uchar, uchar4, uchar3)
#endif

#if defined(BGRA2HSV_U8_2D) || defined(ALL_KERNELS)
Convert4To3_2D(BGRA2HSV, U8, uchar, uchar4, uchar3)
#endif

#if defined(BGRA2HSV_F32_1D) || defined(ALL_KERNELS)
Convert4To3_1D(BGRA2HSV, F32, float, float4, float3)
#endif

#if defined(BGRA2HSV_F32_2D) || defined(ALL_KERNELS)
Convert4To3_2D(BGRA2HSV, F32, float, float4, float3)
#endif

#if defined(RGBA2HSV_U8_1D) || defined(ALL_KERNELS)
Convert4To3_1D(RGBA2HSV, U8, uchar, uchar4, uchar3)
#endif

#if defined(RGBA2HSV_U8_2D) || defined(ALL_KERNELS)
Convert4To3_2D(RGBA2HSV, U8, uchar, uchar4, uchar3)
#endif

#if defined(RGBA2HSV_F32_1D) || defined(ALL_KERNELS)
Convert4To3_1D(RGBA2HSV, F32, float, float4, float3)
#endif

#if defined(RGBA2HSV_F32_2D) || defined(ALL_KERNELS)
Convert4To3_2D(RGBA2HSV, F32, float, float4, float3)
#endif


#if defined(HSV2BGR_U8_1D) || defined(HSV2BGR_U8_2D) || defined(ALL_KERNELS)
uchar3 HSV2BGRU8Compute(const uchar3 src) {
  float h = src.x;
  float s = src.y;
  float v = src.z;

  float b, g, r;
  float hscale = 0.03333333f;  // 6.f / 180;
  float div_norm = 0.003921569f;  // 1.0f / 255;
  s *= div_norm;
  v *= div_norm;
  if (s == 0) {
    b = g = r = v;
  }
  else {
    h *= hscale;
    if (h < 0) {
      do {
        h += 6;
      } while (h < 0);
    }
    else if (h >= 6) {
      do {
        h -= 6;
      } while (h >= 6);
    }
    int sector = convert_int_rtn(h);
    h -= sector;
    if ((unsigned)sector >= 6u) {
      sector = 0;
      h = 0.f;
    }
    float x, y, z, w;
    x = v;
    y = v * (1.f - s);
    z = v * (1.f - s * h);
    w = v * (1.f - s * (1.f - h));
    if (sector == 0) {
      b = y;
      g = w;
      r = x;
    }
    else if (sector == 1) {
      b = y;
      g = x;
      r = z;
    }
    else if (sector == 2) {
      b = w;
      g = x;
      r = y;
    }
    else if (sector == 3) {
      b = x;
      g = z;
      r = y;
    }
    else if (sector == 4) {
      b = x;
      g = y;
      r = w;
    }
    else if (sector == 5) {
      b = z;
      g = y;
      r = x;
    }
  }
  b *= 255;
  g *= 255;
  r *= 255;

  uchar3 dst = convert_uchar3_sat_rte((float3)(b, g, r));

  return dst;
}
#endif

#if defined(HSV2BGR_F32_1D) || defined(HSV2BGR_F32_2D) || defined(ALL_KERNELS)
float3 HSV2BGRF32Compute(const float3 src) {
  float _1_60 = 0.016666667f;  // 1.f / 60.f;
  float diff = src.y * src.z;
  float min = src.z - diff;
  float h = src.x * _1_60;

  float3 dst;
  dst.x = src.z;
  float value = diff * (h - 4);
  bool mask0 = h < 4;
  dst.z = mask0 ? min : (min + value);
  dst.y = mask0 ? (min - value) : min;

  value = diff * (h - 2);
  mask0 = h < 2;
  bool mask1 = h < 3;
  dst.y = mask1 ? src.z : dst.y;
  mask1 = ~mask0 & mask1;
  dst.x = mask0 ? min : dst.x;
  dst.x = mask1 ? (min + value) : dst.x;
  dst.z = mask0 ? (min - value) : dst.z;
  dst.z = mask1 ? min : dst.z;

  mask0 = h < 1;
  value = diff * h;
  dst.z = mask0 ? src.z : dst.z;
  dst.x = mask0 ? min : dst.x;
  dst.y = mask0 ? (min + value) : dst.y;

  mask0 = h >= 5;
  value = diff * (h - 6);
  dst.z = mask0 ? src.z : dst.z;
  dst.y = mask0 ? min : dst.y;
  dst.x = mask0 ? (min - value) : dst.x;

  return dst;
}
#endif

#if defined(HSV2RGB_U8_1D) || defined(HSV2RGB_U8_2D) || defined(ALL_KERNELS)
uchar3 HSV2RGBU8Compute(const uchar3 src) {
  float h = src.x;
  float s = src.y;
  float v = src.z;

  float r, g, b;
  float hscale = 0.03333333f;  // 6.f / 180;
  float div_norm = 0.003921569f;  // 1.0f / 255;
  s *= div_norm;
  v *= div_norm;
  if (s == 0) {
    b = g = r = v;
  }
  else {
    h *= hscale;
    if (h < 0) {
      do {
        h += 6;
      } while (h < 0);
    }
    else if (h >= 6) {
      do {
        h -= 6;
      } while (h >= 6);
    }
    int sector = convert_int_rtn(h);
    h -= sector;
    if ((unsigned)sector >= 6u) {
      sector = 0;
      h = 0.f;
    }
    float x, y, z, w;
    x = v;
    y = v * (1.f - s);
    z = v * (1.f - s * h);
    w = v * (1.f - s * (1.f - h));
    if (sector == 0) {
      r = x;
      g = w;
      b = y;
    }
    else if (sector == 1) {
      r = z;
      g = x;
      b = y;
    }
    else if (sector == 2) {
      r = y;
      g = x;
      b = w;
    }
    else if (sector == 3) {
      r = y;
      g = z;
      b = x;
    }
    else if (sector == 4) {
      r = w;
      g = y;
      b = x;
    }
    else if (sector == 5) {
      r = x;
      g = y;
      b = z;
    }
  }
  r *= 255;
  g *= 255;
  b *= 255;

  uchar3 dst = convert_uchar3_sat_rte((float3)(r, g, b));

  return dst;
}
#endif

#if defined(HSV2RGB_F32_1D) || defined(HSV2RGB_F32_2D) || defined(ALL_KERNELS)
float3 HSV2RGBF32Compute(const float3 src) {
  float _1_60 = 0.016666667f;  // 1.f / 60.f;
  float diff = src.y * src.z;
  float min = src.z - diff;
  float h = src.x * _1_60;

  float3 dst;
  dst.z = src.z;
  float value = diff * (h - 4);
  bool mask0 = h < 4;
  dst.x = mask0 ? min : (min + value);
  dst.y = mask0 ? (min - value) : min;

  value = diff * (h - 2);
  mask0 = h < 2;
  bool mask1 = h < 3;
  dst.y = mask1 ? src.z : dst.y;
  mask1 = ~mask0 & mask1;
  dst.z = mask0 ? min : dst.z;
  dst.z = mask1 ? (min + value) : dst.z;
  dst.x = mask0 ? (min - value) : dst.x;
  dst.x = mask1 ? min : dst.x;

  mask0 = h < 1;
  value = diff * h;
  dst.x = mask0 ? src.z : dst.x;
  dst.z = mask0 ? min : dst.z;
  dst.y = mask0 ? (min + value) : dst.y;

  mask0 = h >= 5;
  value = diff * (h - 6);
  dst.x = mask0 ? src.z : dst.x;
  dst.y = mask0 ? min : dst.y;
  dst.z = mask0 ? (min - value) : dst.z;

  return dst;
}
#endif

#if defined(HSV2BGR_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(HSV2BGR, U8, uchar, uchar3)
#endif

#if defined(HSV2BGR_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(HSV2BGR, U8, uchar, uchar3)
#endif

#if defined(HSV2BGR_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(HSV2BGR, F32, float, float3)
#endif

#if defined(HSV2BGR_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(HSV2BGR, F32, float, float3)
#endif

#if defined(HSV2RGB_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(HSV2RGB, U8, uchar, uchar3)
#endif

#if defined(HSV2RGB_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(HSV2RGB, U8, uchar, uchar3)
#endif

#if defined(HSV2RGB_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(HSV2RGB, F32, float, float3)
#endif

#if defined(HSV2RGB_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(HSV2RGB, F32, float, float3)
#endif


#if defined(HSV2BGRA_U8_1D) || defined(HSV2BGRA_U8_2D) || defined(ALL_KERNELS)
uchar4 HSV2BGRAU8Compute(const uchar3 src) {
  float h = src.x;
  float s = src.y;
  float v = src.z;

  float b, g, r;
  float hscale = 0.03333333f;  // 6.f / 180;
  float div_norm = 0.003921569f;  // 1.0f / 255;
  s *= div_norm;
  v *= div_norm;
  if (s == 0) {
    b = g = r = v;
  }
  else {
    h *= hscale;
    if (h < 0) {
      do {
        h += 6;
      } while (h < 0);
    }
    else if (h >= 6) {
      do {
        h -= 6;
      } while (h >= 6);
    }
    int sector = convert_int_rtn(h);
    h -= sector;
    if ((unsigned)sector >= 6u) {
      sector = 0;
      h = 0.f;
    }
    float x, y, z, w;
    x = v;
    y = v * (1.f - s);
    z = v * (1.f - s * h);
    w = v * (1.f - s * (1.f - h));
    if (sector == 0) {
      b = y;
      g = w;
      r = x;
    }
    else if (sector == 1) {
      b = y;
      g = x;
      r = z;
    }
    else if (sector == 2) {
      b = w;
      g = x;
      r = y;
    }
    else if (sector == 3) {
      b = x;
      g = z;
      r = y;
    }
    else if (sector == 4) {
      b = x;
      g = y;
      r = w;
    }
    else if (sector == 5) {
      b = z;
      g = y;
      r = x;
    }
  }
  b *= 255;
  g *= 255;
  r *= 255;

  uchar4 dst=convert_uchar4_sat_rte((float4)(b, g, r, 255.f));

  return dst;
}
#endif

#if defined(HSV2BGRA_F32_1D) || defined(HSV2BGRA_F32_2D) || defined(ALL_KERNELS)
float4 HSV2BGRAF32Compute(const float3 src) {
  float _1_60 = 0.016666667f;  // 1.f / 60.f;
  float diff = src.y * src.z;
  float min = src.z - diff;
  float h = src.x * _1_60;

  float4 dst;
  dst.x = src.z;
  float value = diff * (h - 4);
  bool mask0 = h < 4;
  dst.z = mask0 ? min : (min + value);
  dst.y = mask0 ? (min - value) : min;

  value = diff * (h - 2);
  mask0 = h < 2;
  bool mask1 = h < 3;
  dst.y = mask1 ? src.z : dst.y;
  mask1 = ~mask0 & mask1;
  dst.x = mask0 ? min : dst.x;
  dst.x = mask1 ? (min + value) : dst.x;
  dst.z = mask0 ? (min - value) : dst.z;
  dst.z = mask1 ? min : dst.z;

  mask0 = h < 1;
  value = diff * h;
  dst.z = mask0 ? src.z : dst.z;
  dst.x = mask0 ? min : dst.x;
  dst.y = mask0 ? (min + value) : dst.y;

  mask0 = h >= 5;
  value = diff * (h - 6);
  dst.z = mask0 ? src.z : dst.z;
  dst.y = mask0 ? min : dst.y;
  dst.x = mask0 ? (min - value) : dst.x;
  dst.w = 1.f;

  return dst;
}
#endif

#if defined(HSV2RGBA_U8_1D) || defined(HSV2RGBA_U8_2D) || defined(ALL_KERNELS)
uchar4 HSV2RGBAU8Compute(const uchar3 src) {
  float h = src.x;
  float s = src.y;
  float v = src.z;

  float r, g, b;
  float hscale = 0.03333333f;  // 6.f / 180;
  float div_norm = 0.003921569f;  // 1.0f / 255;
  s *= div_norm;
  v *= div_norm;
  if (s == 0) {
    b = g = r = v;
  }
  else {
    h *= hscale;
    if (h < 0) {
      do {
        h += 6;
      } while (h < 0);
    }
    else if (h >= 6) {
      do {
        h -= 6;
      } while (h >= 6);
    }
    int sector = convert_int_rtn(h);
    h -= sector;
    if ((unsigned)sector >= 6u) {
      sector = 0;
      h = 0.f;
    }
    float x, y, z, w;
    x = v;
    y = v * (1.f - s);
    z = v * (1.f - s * h);
    w = v * (1.f - s * (1.f - h));
    if (sector == 0) {
      r = x;
      g = w;
      b = y;
    }
    else if (sector == 1) {
      r = z;
      g = x;
      b = y;
    }
    else if (sector == 2) {
      r = y;
      g = x;
      b = w;
    }
    else if (sector == 3) {
      r = y;
      g = z;
      b = x;
    }
    else if (sector == 4) {
      r = w;
      g = y;
      b = x;
    }
    else if (sector == 5) {
      r = x;
      g = y;
      b = z;
    }
  }
  r *= 255;
  g *= 255;
  b *= 255;

  uchar4 dst=convert_uchar4_sat_rte((float4)(r, g, b, 255.f));

  return dst;
}
#endif

#if defined(HSV2RGBA_F32_1D) || defined(HSV2RGBA_F32_2D) || defined(ALL_KERNELS)
float4 HSV2RGBAF32Compute(const float3 src) {
  float _1_60 = 0.016666667f;  // 1.f / 60.f;
  float diff = src.y * src.z;
  float min = src.z - diff;
  float h = src.x * _1_60;

  float4 dst;
  dst.z = src.z;
  float value = diff * (h - 4);
  bool mask0 = h < 4;
  dst.x = mask0 ? min : (min + value);
  dst.y = mask0 ? (min - value) : min;

  value = diff * (h - 2);
  mask0 = h < 2;
  bool mask1 = h < 3;
  dst.y = mask1 ? src.z : dst.y;
  mask1 = ~mask0 & mask1;
  dst.z = mask0 ? min : dst.z;
  dst.z = mask1 ? (min + value) : dst.z;
  dst.x = mask0 ? (min - value) : dst.x;
  dst.x = mask1 ? min : dst.x;

  mask0 = h < 1;
  value = diff * h;
  dst.x = mask0 ? src.z : dst.x;
  dst.z = mask0 ? min : dst.z;
  dst.y = mask0 ? (min + value) : dst.y;

  mask0 = h >= 5;
  value = diff * (h - 6);
  dst.x = mask0 ? src.z : dst.x;
  dst.y = mask0 ? min : dst.y;
  dst.z = mask0 ? (min - value) : dst.z;
  dst.w = 1.f;

  return dst;
}
#endif

#if defined(HSV2BGRA_U8_1D) || defined(ALL_KERNELS)
Convert3To4_1D(HSV2BGRA, U8, uchar, uchar3, uchar4)
#endif

#if defined(HSV2BGRA_U8_2D) || defined(ALL_KERNELS)
Convert3To4_2D(HSV2BGRA, U8, uchar, uchar3, uchar4)
#endif

#if defined(HSV2BGRA_F32_1D) || defined(ALL_KERNELS)
Convert3To4_1D(HSV2BGRA, F32, float, float3, float4)
#endif

#if defined(HSV2BGRA_F32_2D) || defined(ALL_KERNELS)
Convert3To4_2D(HSV2BGRA, F32, float, float3, float4)
#endif

#if defined(HSV2RGBA_U8_1D) || defined(ALL_KERNELS)
Convert3To4_1D(HSV2RGBA, U8, uchar, uchar3, uchar4)
#endif

#if defined(HSV2RGBA_U8_2D) || defined(ALL_KERNELS)
Convert3To4_2D(HSV2RGBA, U8, uchar, uchar3, uchar4)
#endif

#if defined(HSV2RGBA_F32_1D) || defined(ALL_KERNELS)
Convert3To4_1D(HSV2RGBA, F32, float, float3, float4)
#endif

#if defined(HSV2RGBA_F32_2D) || defined(ALL_KERNELS)
Convert3To4_2D(HSV2RGBA, F32, float, float3, float4)
#endif


/*********************** BGR/RGB/BGRA/RGBA <-> LAB ************************/

enum LABShifts {
  kLabShift   = 12,
  kGammaShift = 3,
  kLabShift2  = (kLabShift + kGammaShift),
};

int gamma(float x) {
    float value = x > 0.04045f ? pow((x + 0.055f) / 1.055f, 2.4f) : x / 12.92f;
    return convert_int_rte(value * 2040);
}
int labCbrt_b(int i) {
  float x = i * (1.f / (255.f * (1 << kGammaShift)));
  float value = x < 0.008856f ? x * 7.787f + 0.13793103448275862f :
                pow(x,0.3333333333333f);

  return (1 << kLabShift2) * value;
}

#if defined(BGR2LAB_U8_1D) || defined(BGR2LAB_U8_2D) || defined(ALL_KERNELS)
uchar3 BGR2LABU8Compute(const uchar3 src) {
  int Lscale = 296;  // (116 * 255 + 50) / 100;
  int Lshift = -1336935;  // -((16 * 255 * (1 << kLabShift2) + 50) / 100);

  int B = gamma(src.x / 255.f);
  int G = gamma(src.y / 255.f);
  int R = gamma(src.z / 255.f);

  int L = divideUp1(B * 778 + G * 1541 + R * 1777, kLabShift);
  int a = divideUp1(B * 296 + G * 2929 + R * 871, kLabShift);
  int b = divideUp1(B * 3575 + G * 448 + R * 73, kLabShift);

  int fX = labCbrt_b(L);
  int fY = labCbrt_b(a);
  int fZ = labCbrt_b(b);

  B = Lscale * fY + Lshift;
  G = 500 * (fX - fY) + 128 * (1 << kLabShift2);
  R = 200 * (fY - fZ) + 128 * (1 << kLabShift2);

  L = divideUp1(B, kLabShift2);
  a = divideUp1(G, kLabShift2);
  b = divideUp1(R, kLabShift2);

  uchar3 dst = convert_uchar3_sat((int3)(L, a, b));

  return dst;
}
#endif

#if defined(BGR2LAB_F32_1D) || defined(BGR2LAB_F32_2D) || defined(ALL_KERNELS)
float3 BGR2LABF32Compute(float3 src) {
  float div_1_3     = 0.333333f;
  float div_16_116  = 0.137931f;

  src.x = (src.x > 0.04045f) ? pow((src.x + 0.055f) / 1.055f, 2.4f) :
           src.x / 12.92f;
  src.y = (src.y > 0.04045f) ? pow((src.y + 0.055f) / 1.055f, 2.4f) :
           src.y / 12.92f;
  src.z = (src.z > 0.04045f) ? pow((src.z + 0.055f) / 1.055f, 2.4f) :
           src.z / 12.92f;

  float3 dst;
  dst.x = src.x * 0.189828f + src.y * 0.376219f + src.z * 0.433953f;
  dst.y = src.x * 0.072169f + src.y * 0.715160f + src.z * 0.212671f;
  dst.z = src.x * 0.872766f + src.y * 0.109477f + src.z * 0.017758f;

  float pow_y = pow(dst.y, div_1_3);
  float FX = dst.x > 0.008856f ? pow(dst.x, div_1_3) : (7.787f * dst.x +
             div_16_116);
  float FY = dst.y > 0.008856f ? pow_y : (7.787f * dst.y + div_16_116);
  float FZ = dst.z > 0.008856f ? pow(dst.z, div_1_3) : (7.787f * dst.z +
             div_16_116);

  dst.x = dst.y > 0.008856f ? (116.f * pow_y - 16.f) : (903.3f * dst.y);
  dst.y = 500.f * (FX - FY);
  dst.z = 200.f * (FY - FZ);

  return dst;
}
#endif

#if defined(RGB2LAB_U8_1D) || defined(RGB2LAB_U8_2D) || defined(ALL_KERNELS)
uchar3 RGB2LABU8Compute(const uchar3 src) {
  int Lscale = 296;  // (116 * 255 + 50) / 100;
  int Lshift = -1336935;  // -((16 * 255 * (1 << kLabShift2) + 50) / 100);

  int B = gamma(src.z / 255.f);
  int G = gamma(src.y / 255.f);
  int R = gamma(src.x / 255.f);

  int L = divideUp1(B * 778 + G * 1541 + R * 1777, kLabShift);
  int a = divideUp1(B * 296 + G * 2929 + R * 871, kLabShift);
  int b = divideUp1(B * 3575 + G * 448 + R * 73, kLabShift);

  int fX = labCbrt_b(L);
  int fY = labCbrt_b(a);
  int fZ = labCbrt_b(b);

  B = Lscale * fY + Lshift;
  G = 500 * (fX - fY) + 128 * (1 << kLabShift2);
  R = 200 * (fY - fZ) + 128 * (1 << kLabShift2);

  L = divideUp1(B, kLabShift2);
  a = divideUp1(G, kLabShift2);
  b = divideUp1(R, kLabShift2);

  uchar3 dst = convert_uchar3_sat((int3)(L, a, b));

  return dst;
}
#endif

#if defined(RGB2LAB_F32_1D) || defined(RGB2LAB_F32_2D) || defined(ALL_KERNELS)
float3 RGB2LABF32Compute(float3 src) {
  float div_1_3     = 0.333333f;
  float div_16_116  = 0.137931f;

  src.x = (src.x > 0.04045f) ? pow((src.x + 0.055f) / 1.055f, 2.4f) :
           src.x / 12.92f;
  src.y = (src.y > 0.04045f) ? pow((src.y + 0.055f) / 1.055f, 2.4f) :
           src.y / 12.92f;
  src.z = (src.z > 0.04045f) ? pow((src.z + 0.055f) / 1.055f, 2.4f) :
           src.z / 12.92f;

  float3 dst;
  dst.x = src.z * 0.189828f + src.y * 0.376219f + src.x * 0.433953f;
  dst.y = src.z * 0.072169f + src.y * 0.715160f + src.x * 0.212671f;
  dst.z = src.z * 0.872766f + src.y * 0.109477f + src.x * 0.017758f;

  float pow_y = pow(dst.y, div_1_3);
  float FX = dst.x > 0.008856f ? pow(dst.x, div_1_3) : (7.787f * dst.x +
             div_16_116);
  float FY = dst.y > 0.008856f ? pow_y : (7.787f * dst.y + div_16_116);
  float FZ = dst.z > 0.008856f ? pow(dst.z, div_1_3) : (7.787f * dst.z +
             div_16_116);

  dst.x = dst.y > 0.008856f ? (116.f * pow_y - 16.f) : (903.3f * dst.y);
  dst.y = 500.f * (FX - FY);
  dst.z = 200.f * (FY - FZ);

  return dst;
}
#endif

#if defined(BGR2LAB_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(BGR2LAB, U8, uchar, uchar3)
#endif

#if defined(BGR2LAB_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(BGR2LAB, U8, uchar, uchar3)
#endif

#if defined(BGR2LAB_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(BGR2LAB, F32, float, float3)
#endif

#if defined(BGR2LAB_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(BGR2LAB, F32, float, float3)
#endif

#if defined(RGB2LAB_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(RGB2LAB, U8, uchar, uchar3)
#endif

#if defined(RGB2LAB_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(RGB2LAB, U8, uchar, uchar3)
#endif

#if defined(RGB2LAB_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(RGB2LAB, F32, float, float3)
#endif

#if defined(RGB2LAB_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(RGB2LAB, F32, float, float3)
#endif


#if defined(BGRA2LAB_U8_1D) || defined(BGRA2LAB_U8_2D) || defined(ALL_KERNELS)
uchar3 BGRA2LABU8Compute(const uchar4 src) {
  int Lscale = 296;  // (116 * 255 + 50) / 100;
  int Lshift = -1336935;  // -((16 * 255 * (1 << kLabShift2) + 50) / 100);

  int B = gamma(src.x / 255.f);
  int G = gamma(src.y / 255.f);
  int R = gamma(src.z / 255.f);

  int L = divideUp1(B * 778 + G * 1541 + R * 1777, kLabShift);
  int a = divideUp1(B * 296 + G * 2929 + R * 871, kLabShift);
  int b = divideUp1(B * 3575 + G * 448 + R * 73, kLabShift);

  int fX = labCbrt_b(L);
  int fY = labCbrt_b(a);
  int fZ = labCbrt_b(b);

  B = Lscale * fY + Lshift;
  G = 500 * (fX - fY) + 128 * (1 << kLabShift2);
  R = 200 * (fY - fZ) + 128 * (1 << kLabShift2);

  L = divideUp1(B, kLabShift2);
  a = divideUp1(G, kLabShift2);
  b = divideUp1(R, kLabShift2);

  uchar3 dst = convert_uchar3_sat((int3)(L, a, b));

  return dst;
}
#endif

#if defined(BGRA2LAB_F32_1D) || defined(BGRA2LAB_F32_2D) || defined(ALL_KERNELS)
float3 BGRA2LABF32Compute(float4 src) {
  float div_1_3     = 0.333333f;
  float div_16_116  = 0.137931f;

  src.x = (src.x > 0.04045f) ? pow((src.x + 0.055f) / 1.055f, 2.4f) :
           src.x / 12.92f;
  src.y = (src.y > 0.04045f) ? pow((src.y + 0.055f) / 1.055f, 2.4f) :
           src.y / 12.92f;
  src.z = (src.z > 0.04045f) ? pow((src.z + 0.055f) / 1.055f, 2.4f) :
           src.z / 12.92f;

  float3 dst;
  dst.x = src.x * 0.189828f + src.y * 0.376219f + src.z * 0.433953f;
  dst.y = src.x * 0.072169f + src.y * 0.715160f + src.z * 0.212671f;
  dst.z = src.x * 0.872766f + src.y * 0.109477f + src.z * 0.017758f;

  float pow_y = pow(dst.y, div_1_3);
  float FX = dst.x > 0.008856f ? pow(dst.x, div_1_3) : (7.787f * dst.x +
             div_16_116);
  float FY = dst.y > 0.008856f ? pow_y : (7.787f * dst.y + div_16_116);
  float FZ = dst.z > 0.008856f ? pow(dst.z, div_1_3) : (7.787f * dst.z +
             div_16_116);

  dst.x = dst.y > 0.008856f ? (116.f * pow_y - 16.f) : (903.3f * dst.y);
  dst.y = 500.f * (FX - FY);
  dst.z = 200.f * (FY - FZ);

  return dst;
}
#endif

#if defined(RGBA2LAB_U8_1D) || defined(RGBA2LAB_U8_2D) || defined(ALL_KERNELS)
uchar3 RGBA2LABU8Compute(const uchar4 src) {
  int Lscale = 296;  // (116 * 255 + 50) / 100;
  int Lshift = -1336935;  // -((16 * 255 * (1 << kLabShift2) + 50) / 100);

  int B = gamma(src.z / 255.f);
  int G = gamma(src.y / 255.f);
  int R = gamma(src.x / 255.f);

  int L = divideUp1(B * 778 + G * 1541 + R * 1777, kLabShift);
  int a = divideUp1(B * 296 + G * 2929 + R * 871, kLabShift);
  int b = divideUp1(B * 3575 + G * 448 + R * 73, kLabShift);

  int fX = labCbrt_b(L);
  int fY = labCbrt_b(a);
  int fZ = labCbrt_b(b);

  B = Lscale * fY + Lshift;
  G = 500 * (fX - fY) + 128 * (1 << kLabShift2);
  R = 200 * (fY - fZ) + 128 * (1 << kLabShift2);

  L = divideUp1(B, kLabShift2);
  a = divideUp1(G, kLabShift2);
  b = divideUp1(R, kLabShift2);

  uchar3 dst = convert_uchar3_sat((int3)(L, a, b));

  return dst;
}
#endif

#if defined(RGBA2LAB_F32_1D) || defined(RGBA2LAB_F32_2D) || defined(ALL_KERNELS)
float3 RGBA2LABF32Compute(float4 src) {
  float div_1_3     = 0.333333f;
  float div_16_116  = 0.137931f;

  src.x = (src.x > 0.04045f) ? pow((src.x + 0.055f) / 1.055f, 2.4f) :
           src.x / 12.92f;
  src.y = (src.y > 0.04045f) ? pow((src.y + 0.055f) / 1.055f, 2.4f) :
           src.y / 12.92f;
  src.z = (src.z > 0.04045f) ? pow((src.z + 0.055f) / 1.055f, 2.4f) :
           src.z / 12.92f;

  float3 dst;
  dst.x = src.z * 0.189828f + src.y * 0.376219f + src.x * 0.433953f;
  dst.y = src.z * 0.072169f + src.y * 0.715160f + src.x * 0.212671f;
  dst.z = src.z * 0.872766f + src.y * 0.109477f + src.x * 0.017758f;

  float pow_y = pow(dst.y, div_1_3);
  float FX = dst.x > 0.008856f ? pow(dst.x, div_1_3) : (7.787f * dst.x +
             div_16_116);
  float FY = dst.y > 0.008856f ? pow_y : (7.787f * dst.y + div_16_116);
  float FZ = dst.z > 0.008856f ? pow(dst.z, div_1_3) : (7.787f * dst.z +
             div_16_116);

  dst.x = dst.y > 0.008856f ? (116.f * pow_y - 16.f) : (903.3f * dst.y);
  dst.y = 500.f * (FX - FY);
  dst.z = 200.f * (FY - FZ);

  return dst;
}
#endif

#if defined(BGRA2LAB_U8_1D) || defined(ALL_KERNELS)
Convert4To3_1D(BGRA2LAB, U8, uchar, uchar4, uchar3)
#endif

#if defined(BGRA2LAB_U8_2D) || defined(ALL_KERNELS)
Convert4To3_2D(BGRA2LAB, U8, uchar, uchar4, uchar3)
#endif

#if defined(BGRA2LAB_F32_1D) || defined(ALL_KERNELS)
Convert4To3_1D(BGRA2LAB, F32, float, float4, float3)
#endif

#if defined(BGRA2LAB_F32_2D) || defined(ALL_KERNELS)
Convert4To3_2D(BGRA2LAB, F32, float, float4, float3)
#endif

#if defined(RGBA2LAB_U8_1D) || defined(ALL_KERNELS)
Convert4To3_1D(RGBA2LAB, U8, uchar, uchar4, uchar3)
#endif

#if defined(RGBA2LAB_U8_2D) || defined(ALL_KERNELS)
Convert4To3_2D(RGBA2LAB, U8, uchar, uchar4, uchar3)
#endif

#if defined(RGBA2LAB_F32_1D) || defined(ALL_KERNELS)
Convert4To3_1D(RGBA2LAB, F32, float, float4, float3)
#endif

#if defined(RGBA2LAB_F32_2D) || defined(ALL_KERNELS)
Convert4To3_2D(RGBA2LAB, F32, float, float4, float3)
#endif


#if defined(LAB2BGR_U8_1D) || defined(LAB2BGR_U8_2D) || defined(ALL_KERNELS)
uchar3 LAB2BGRU8Compute(const uchar3 src) {
  float _16_116 = 0.137931034f;  // 16.0f / 116.0f;
  float lThresh = 7.9996248f;    // 0.008856f * 903.3f;
  float fThresh = 0.206892706f;  // 0.008856f * 7.787f + _16_116;

  float B = src.x * 0.392156863f;   // (100.f / 255.f);
  float G = src.y - 128;
  float R = src.z - 128;

  float Y, fy;

  if (B <= lThresh) {
    Y = B / 903.3f;
    fy = 7.787f * Y + _16_116;
  }
  else {
    fy = (B + 16.0f) / 116.0f;
    Y = fy * fy * fy;
  }

  float X = G / 500.0f + fy;
  float Z = fy - R / 200.0f;

  if (X <= fThresh) {
    X = (X - _16_116) / 7.787f;
  }
  else {
    X = X * X * X;
  }

  if (Z <= fThresh) {
    Z = (Z - _16_116) / 7.787f;
  }
  else {
    Z = Z * Z * Z;
  }

  B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
  G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
  R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

  B = (B > 0.00304f) ? (1.055f * pow(B, 0.41667f) - 0.055f) : 12.92f * B;
  G = (G > 0.00304f) ? (1.055f * pow(G, 0.41667f) - 0.055f) : 12.92f * G;
  R = (R > 0.00304f) ? (1.055f * pow(R, 0.41667f) - 0.055f) : 12.92f * R;

  B = B * 255.f;
  G = G * 255.f;
  R = R * 255.f;

  uchar3 dst = convert_uchar3_sat_rte((float3)(B, G, R));

  return dst;
}
#endif

#if defined(LAB2BGR_F32_1D) || defined(LAB2BGR_F32_2D) || defined(ALL_KERNELS)
float3 LAB2BGRF32Compute(const float3 src) {
  float _16_116 = 0.137931034f;  // 16.0f / 116.0f;
  float lThresh = 7.9996248f;    // 0.008856f * 903.3f;
  float fThresh = 0.206892706f;  // 0.008856f * 7.787f + _16_116;

  float Y, fy;

  if (src.x <= lThresh) {
    Y = src.x / 903.3f;
    fy = 7.787f * Y + _16_116;
  }
  else {
    fy = (src.x + 16.0f) / 116.0f;
    Y = fy * fy * fy;
  }

  float X = src.y / 500.0f + fy;
  float Z = fy - src.z / 200.0f;

  if (X <= fThresh) {
    X = (X - _16_116) / 7.787f;
  }
  else {
    X = X * X * X;
  }

  if (Z <= fThresh) {
    Z = (Z - _16_116) / 7.787f;
  }
  else {
    Z = Z * Z * Z;
  }

  float3 dst;
  dst.x = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
  dst.y = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
  dst.z = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

  dst.x = (dst.x > 0.00304f) ? (1.055f * pow(dst.x, 0.41667f) - 0.055f) :
          12.92f * dst.x;
  dst.y = (dst.y > 0.00304f) ? (1.055f * pow(dst.y, 0.41667f) - 0.055f) :
          12.92f * dst.y;
  dst.z = (dst.z > 0.00304f) ? (1.055f * pow(dst.z, 0.41667f) - 0.055f) :
          12.92f * dst.z;

  return dst;
}
#endif

#if defined(LAB2RGB_U8_1D) || defined(LAB2RGB_U8_2D) || defined(ALL_KERNELS)
uchar3 LAB2RGBU8Compute(const uchar3 src) {
  float _16_116 = 0.137931034f;  // 16.0f / 116.0f;
  float lThresh = 7.9996248f;    // 0.008856f * 903.3f;
  float fThresh = 0.206892706f;  // 0.008856f * 7.787f + _16_116;

  float B = src.x * 0.392156863f;   // (100.f / 255.f);
  float G = src.y - 128;
  float R = src.z - 128;

  float Y, fy;

  if (B <= lThresh) {
    Y = B / 903.3f;
    fy = 7.787f * Y + _16_116;
  }
  else {
    fy = (B + 16.0f) / 116.0f;
    Y = fy * fy * fy;
  }

  float X = G / 500.0f + fy;
  float Z = fy - R / 200.0f;

  if (X <= fThresh) {
    X = (X - _16_116) / 7.787f;
  }
  else {
    X = X * X * X;
  }

  if (Z <= fThresh) {
    Z = (Z - _16_116) / 7.787f;
  }
  else {
    Z = Z * Z * Z;
  }

  B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
  G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
  R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

  B = (B > 0.00304f) ? (1.055f * pow(B, 0.41667f) - 0.055f) : 12.92f * B;
  G = (G > 0.00304f) ? (1.055f * pow(G, 0.41667f) - 0.055f) : 12.92f * G;
  R = (R > 0.00304f) ? (1.055f * pow(R, 0.41667f) - 0.055f) : 12.92f * R;

  B = B * 255.f;
  G = G * 255.f;
  R = R * 255.f;

  uchar3 dst = convert_uchar3_sat_rte((float3)(R, G, B));

  return dst;
}
#endif

#if defined(LAB2RGB_F32_1D) || defined(LAB2RGB_F32_2D) || defined(ALL_KERNELS)
float3 LAB2RGBF32Compute(const float3 src) {
  float _16_116 = 0.137931034f;  // 16.0f / 116.0f;
  float lThresh = 7.9996248f;    // 0.008856f * 903.3f;
  float fThresh = 0.206892706f;  // 0.008856f * 7.787f + _16_116;

  float Y, fy;

  if (src.x <= lThresh) {
    Y = src.x / 903.3f;
    fy = 7.787f * Y + _16_116;
  }
  else {
    fy = (src.x + 16.0f) / 116.0f;
    Y = fy * fy * fy;
  }

  float X = src.y / 500.0f + fy;
  float Z = fy - src.z / 200.0f;

  if (X <= fThresh) {
    X = (X - _16_116) / 7.787f;
  }
  else {
    X = X * X * X;
  }

  if (Z <= fThresh) {
    Z = (Z - _16_116) / 7.787f;
  }
  else {
    Z = Z * Z * Z;
  }

  float3 dst;
  dst.x = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;
  dst.y = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
  dst.z = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;

  dst.x = (dst.x > 0.00304f) ? (1.055f * pow(dst.x, 0.41667f) - 0.055f) :
          12.92f * dst.x;
  dst.y = (dst.y > 0.00304f) ? (1.055f * pow(dst.y, 0.41667f) - 0.055f) :
          12.92f * dst.y;
  dst.z = (dst.z > 0.00304f) ? (1.055f * pow(dst.z, 0.41667f) - 0.055f) :
          12.92f * dst.z;

  return dst;
}
#endif

#if defined(LAB2BGR_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(LAB2BGR, U8, uchar, uchar3)
#endif

#if defined(LAB2BGR_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(LAB2BGR, U8, uchar, uchar3)
#endif

#if defined(LAB2BGR_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(LAB2BGR, F32, float, float3)
#endif

#if defined(LAB2BGR_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(LAB2BGR, F32, float, float3)
#endif

#if defined(LAB2RGB_U8_1D) || defined(ALL_KERNELS)
Convert3To3_1D(LAB2RGB, U8, uchar, uchar3)
#endif

#if defined(LAB2RGB_U8_2D) || defined(ALL_KERNELS)
Convert3To3_2D(LAB2RGB, U8, uchar, uchar3)
#endif

#if defined(LAB2RGB_F32_1D) || defined(ALL_KERNELS)
Convert3To3_1D(LAB2RGB, F32, float, float3)
#endif

#if defined(LAB2RGB_F32_2D) || defined(ALL_KERNELS)
Convert3To3_2D(LAB2RGB, F32, float, float3)
#endif


#if defined(LAB2BGRA_U8_1D) || defined(LAB2BGRA_U8_2D) || defined(ALL_KERNELS)
uchar4 LAB2BGRAU8Compute(const uchar3 src) {
  float _16_116 = 0.137931034f;  // 16.0f / 116.0f;
  float lThresh = 7.9996248f;    // 0.008856f * 903.3f;
  float fThresh = 0.206892706f;  // 0.008856f * 7.787f + _16_116;

  float B = src.x * 0.392156863f;   // (100.f / 255.f);
  float G = src.y - 128;
  float R = src.z - 128;

  float Y, fy;

  if (B <= lThresh) {
    Y = B / 903.3f;
    fy = 7.787f * Y + _16_116;
  }
  else {
    fy = (B + 16.0f) / 116.0f;
    Y = fy * fy * fy;
  }

  float X = G / 500.0f + fy;
  float Z = fy - R / 200.0f;

  if (X <= fThresh) {
    X = (X - _16_116) / 7.787f;
  }
  else {
    X = X * X * X;
  }

  if (Z <= fThresh) {
    Z = (Z - _16_116) / 7.787f;
  }
  else {
    Z = Z * Z * Z;
  }

  B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
  G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
  R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

  B = (B > 0.00304f) ? (1.055f * pow(B, 0.41667f) - 0.055f) : 12.92f * B;
  G = (G > 0.00304f) ? (1.055f * pow(G, 0.41667f) - 0.055f) : 12.92f * G;
  R = (R > 0.00304f) ? (1.055f * pow(R, 0.41667f) - 0.055f) : 12.92f * R;

  B = B * 255.f;
  G = G * 255.f;
  R = R * 255.f;

  uchar4 dst=convert_uchar4_sat_rte((float4)(B, G, R, 255.f));

  return dst;
}
#endif

#if defined(LAB2BGRA_F32_1D) || defined(LAB2BGRA_F32_2D) || defined(ALL_KERNELS)
float4 LAB2BGRAF32Compute(const float3 src) {
  float _16_116 = 0.137931034f;  // 16.0f / 116.0f;
  float lThresh = 7.9996248f;    // 0.008856f * 903.3f;
  float fThresh = 0.206892706f;  // 0.008856f * 7.787f + _16_116;

  float Y, fy;

  if (src.x <= lThresh) {
    Y = src.x / 903.3f;
    fy = 7.787f * Y + _16_116;
  }
  else {
    fy = (src.x + 16.0f) / 116.0f;
    Y = fy * fy * fy;
  }

  float X = src.y / 500.0f + fy;
  float Z = fy - src.z / 200.0f;

  if (X <= fThresh) {
    X = (X - _16_116) / 7.787f;
  }
  else {
    X = X * X * X;
  }

  if (Z <= fThresh) {
    Z = (Z - _16_116) / 7.787f;
  }
  else {
    Z = Z * Z * Z;
  }

  float4 dst;
  dst.x = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
  dst.y = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
  dst.z = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

  dst.x = (dst.x > 0.00304f) ? (1.055f * pow(dst.x, 0.41667f) - 0.055f) :
          12.92f * dst.x;
  dst.y = (dst.y > 0.00304f) ? (1.055f * pow(dst.y, 0.41667f) - 0.055f) :
          12.92f * dst.y;
  dst.z = (dst.z > 0.00304f) ? (1.055f * pow(dst.z, 0.41667f) - 0.055f) :
          12.92f * dst.z;
  dst.w = 1.f;

  return dst;
}
#endif

#if defined(LAB2RGBA_U8_1D) || defined(LAB2RGBA_U8_2D) || defined(ALL_KERNELS)
uchar4 LAB2RGBAU8Compute(const uchar3 src) {
  float _16_116 = 0.137931034f;  // 16.0f / 116.0f;
  float lThresh = 7.9996248f;    // 0.008856f * 903.3f;
  float fThresh = 0.206892706f;  // 0.008856f * 7.787f + _16_116;

  float B = src.x * 0.392156863f;   // (100.f / 255.f);
  float G = src.y - 128;
  float R = src.z - 128;

  float Y, fy;

  if (B <= lThresh) {
    Y = B / 903.3f;
    fy = 7.787f * Y + _16_116;
  }
  else {
    fy = (B + 16.0f) / 116.0f;
    Y = fy * fy * fy;
  }

  float X = G / 500.0f + fy;
  float Z = fy - R / 200.0f;

  if (X <= fThresh) {
    X = (X - _16_116) / 7.787f;
  }
  else {
    X = X * X * X;
  }

  if (Z <= fThresh) {
    Z = (Z - _16_116) / 7.787f;
  }
  else {
    Z = Z * Z * Z;
  }

  B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
  G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
  R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

  B = (B > 0.00304f) ? (1.055f * pow(B, 0.41667f) - 0.055f) : 12.92f * B;
  G = (G > 0.00304f) ? (1.055f * pow(G, 0.41667f) - 0.055f) : 12.92f * G;
  R = (R > 0.00304f) ? (1.055f * pow(R, 0.41667f) - 0.055f) : 12.92f * R;

  B = B * 255.f;
  G = G * 255.f;
  R = R * 255.f;

  uchar4 dst=convert_uchar4_sat_rte((float4)(R, G, B, 255.f));

  return dst;
}
#endif

#if defined(LAB2RGBA_F32_1D) || defined(LAB2RGBA_F32_2D) || defined(ALL_KERNELS)
float4 LAB2RGBAF32Compute(const float3 src) {
  float _16_116 = 0.137931034f;  // 16.0f / 116.0f;
  float lThresh = 7.9996248f;    // 0.008856f * 903.3f;
  float fThresh = 0.206892706f;  // 0.008856f * 7.787f + _16_116;

  float Y, fy;

  if (src.x <= lThresh) {
    Y = src.x / 903.3f;
    fy = 7.787f * Y + _16_116;
  }
  else {
    fy = (src.x + 16.0f) / 116.0f;
    Y = fy * fy * fy;
  }

  float X = src.y / 500.0f + fy;
  float Z = fy - src.z / 200.0f;

  if (X <= fThresh) {
    X = (X - _16_116) / 7.787f;
  }
  else {
    X = X * X * X;
  }

  if (Z <= fThresh) {
    Z = (Z - _16_116) / 7.787f;
  }
  else {
    Z = Z * Z * Z;
  }

  float4 dst;
  dst.x = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;
  dst.y = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
  dst.z = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;

  dst.x = (dst.x > 0.00304f) ? (1.055f * pow(dst.x, 0.41667f) - 0.055f) :
          12.92f * dst.x;
  dst.y = (dst.y > 0.00304f) ? (1.055f * pow(dst.y, 0.41667f) - 0.055f) :
          12.92f * dst.y;
  dst.z = (dst.z > 0.00304f) ? (1.055f * pow(dst.z, 0.41667f) - 0.055f) :
          12.92f * dst.z;
  dst.w = 1.f;

  return dst;
}
#endif

#if defined(LAB2BGRA_U8_1D) || defined(ALL_KERNELS)
Convert3To4_1D(LAB2BGRA, U8, uchar, uchar3, uchar4)
#endif

#if defined(LAB2BGRA_U8_2D) || defined(ALL_KERNELS)
Convert3To4_2D(LAB2BGRA, U8, uchar, uchar3, uchar4)
#endif

#if defined(LAB2BGRA_F32_1D) || defined(ALL_KERNELS)
Convert3To4_1D(LAB2BGRA, F32, float, float3, float4)
#endif

#if defined(LAB2BGRA_F32_2D) || defined(ALL_KERNELS)
Convert3To4_2D(LAB2BGRA, F32, float, float3, float4)
#endif

#if defined(LAB2RGBA_U8_1D) || defined(ALL_KERNELS)
Convert3To4_1D(LAB2RGBA, U8, uchar, uchar3, uchar4)
#endif

#if defined(LAB2RGBA_U8_2D) || defined(ALL_KERNELS)
Convert3To4_2D(LAB2RGBA, U8, uchar, uchar3, uchar4)
#endif

#if defined(LAB2RGBA_F32_1D) || defined(ALL_KERNELS)
Convert3To4_1D(LAB2RGBA, F32, float, float3, float4)
#endif

#if defined(LAB2RGBA_F32_2D) || defined(ALL_KERNELS)
Convert3To4_2D(LAB2RGBA, F32, float, float3, float4)
#endif

/*********************** BGR/RGB/BGRA/RGBA <-> NV12 ************************/

// BGR/RGB -> NV12/NV21
#define NVXX_YR 269484
#define NVXX_YG 528482
#define NVXX_YB 102760
#define NVXX_VR 460324
#define NVXX_VG -385875
#define NVXX_VB -74448
#define NVXX_UR -155188
#define NVXX_UG -305135
#define NVXX_UB 460324

// NV12/NV21-> BGR/RGB
#define NVXX_CY 1220542
#define NVXX_CUB 2116026
#define NVXX_CUG -409993
#define NVXX_CVG -852492
#define NVXX_CVR 1673527
#define NVXX_SHIFT 20

enum {
  kG2NShift = 20,
  kG2NShift16 = (16 << kG2NShift),
  kG2NHalfShift = (1 << (kG2NShift - 1)),
  kG2NShift128  = (128 << kG2NShift),
};

#if defined(BGR2NV12_U8_2D) || defined(RGB2NV12_U8_2D) ||                      \
    defined(BGR2NV21_U8_2D) || defined(RGB2NV21_U8_2D) || defined(ALL_KERNELS)
#define ConvertToNVXX_2D(Function, base_type)                                  \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst, int dst_stride) {     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar3 input_value = vload3(element_x, data);                                \
  uchar3 result = Function ## Compute(input_value, element_y, element_x);      \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  data[element_x] = result.x;                                                  \
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {                        \
    data = dst + (rows + (element_y >> 1)) * dst_stride;                       \
    vstore2((uchar2)(result.y, result.z), (element_x >> 1), data);             \
  }                                                                            \
}
#endif

#if defined(BGR2NV12_U8_2D) || defined(BGR2NV12_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 BGR2NV12Compute(const uchar3 src, uint row, uint col) {
  int y = (src.x * NVXX_YB + src.y * NVXX_YG + src.z * NVXX_YR +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UB + src.y * NVXX_UG + src.z * NVXX_UR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VB + src.y * NVXX_VG + src.z * NVXX_VR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, u, v));

  return dst;
}
#endif

#if defined(RGB2NV12_U8_2D) || defined(RGB2NV12_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 RGB2NV12Compute(const uchar3 src, uint row, uint col) {
  int y = (src.x * NVXX_YR + src.y * NVXX_YG + src.z * NVXX_YB +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UR + src.y * NVXX_UG + src.z * NVXX_UB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VR + src.y * NVXX_VG + src.z * NVXX_VB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, u, v));

  return dst;
}
#endif

#if defined(BGR2NV12_U8_2D) || defined(ALL_KERNELS)
ConvertToNVXX_2D(BGR2NV12, U8)
#endif

#if defined(RGB2NV12_U8_2D) || defined(ALL_KERNELS)
ConvertToNVXX_2D(RGB2NV12, U8)
#endif

#if defined(BGRA2NV12_U8_2D) || defined(RGBA2NV12_U8_2D) ||                    \
    defined(BGRA2NV21_U8_2D) || defined(RGBA2NV21_U8_2D) || defined(ALL_KERNELS)
#define Convert4To3NVXX_2D(Function, base_type)                                \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst, int dst_stride) {     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar4 input_value = vload4(element_x, data);                                \
  uchar3 result = Function ## Compute(input_value, element_y, element_x);      \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  data[element_x] = result.x;                                                  \
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {                        \
    data = dst + (rows + (element_y >> 1)) * dst_stride;                       \
    vstore2((uchar2)(result.y, result.z), (element_x >> 1), data);             \
  }                                                                            \
}
#endif

#if defined(BGRA2NV12_U8_2D) || defined(BGRA2NV12_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar3 BGRA2NV12Compute(const uchar4 src, uint row, uint col) {
  int y = (src.x * NVXX_YB + src.y * NVXX_YG + src.z * NVXX_YR +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UB + src.y * NVXX_UG + src.z * NVXX_UR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VB + src.y * NVXX_VG + src.z * NVXX_VR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, u, v));

  return dst;
}
#endif

#if defined(RGBA2NV12_U8_2D) || defined(RGBA2NV12_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar3 RGBA2NV12Compute(const uchar4 src, uint row, uint col) {
  int y = (src.x * NVXX_YR + src.y * NVXX_YG + src.z * NVXX_YB +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UR + src.y * NVXX_UG + src.z * NVXX_UB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VR + src.y * NVXX_VG + src.z * NVXX_VB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, u, v));

  return dst;
}
#endif

#if defined(BGRA2NV12_U8_2D) || defined(ALL_KERNELS)
Convert4To3NVXX_2D(BGRA2NV12, U8)
#endif

#if defined(RGBA2NV12_U8_2D) || defined(ALL_KERNELS)
Convert4To3NVXX_2D(RGBA2NV12, U8)
#endif

#if defined(NV122BGR_U8_2D) || defined(NV122RGB_U8_2D) ||                      \
    defined(NV212BGR_U8_2D) || defined(NV212RGB_U8_2D) || defined(ALL_KERNELS)
#define ConvertFromNVXX_2D(Function, base_type)                                \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst, int dst_stride) {     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar3 input_value;                                                          \
  input_value.x = data[element_x];                                             \
                                                                               \
  data = src + (rows + (element_y >> 1)) * src_stride;                         \
  uchar2 tmp = vload2((element_x >> 1), data);                                 \
  input_value.y = tmp.x;                                                       \
  input_value.z = tmp.y;                                                       \
  uchar3 result = Function ## Compute(input_value);                            \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore3(result, element_x, data);                                            \
}
#endif

#if defined(NV122BGR_U8_2D) || defined(NV122BGR_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 NV122BGRCompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.z - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(b, g, r));

  return dst;
}
#endif

#if defined(NV122RGB_U8_2D) || defined(NV122RGB_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 NV122RGBCompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.z - 128;

  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;

  int r = (y + ruv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int b = (y + buv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(r, g, b));

  return dst;
}
#endif

#if defined(NV122BGR_U8_2D) || defined(ALL_KERNELS)
ConvertFromNVXX_2D(NV122BGR, U8)
#endif

#if defined(NV122RGB_U8_2D) || defined(ALL_KERNELS)
ConvertFromNVXX_2D(NV122RGB, U8)
#endif

#if defined(NV122BGRA_U8_2D) || defined(NV122RGBA_U8_2D) ||                    \
    defined(NV212BGRA_U8_2D) || defined(NV212RGBA_U8_2D) || defined(ALL_KERNELS)
#define Convert3To4NVXX_2D(Function, base_type)                                \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst, int dst_stride) {     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar3 input_value;                                                          \
  input_value.x = data[element_x];                                             \
                                                                               \
  data = src + (rows + (element_y >> 1)) * src_stride;                         \
  uchar2 tmp = vload2((element_x >> 1), data);                                 \
  input_value.y = tmp.x;                                                       \
  input_value.z = tmp.y;                                                       \
  uchar4 result = Function ## Compute(input_value);                            \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore4(result, element_x, data);                                            \
}
#endif

#if defined(NV122BGRA_U8_2D) || defined(NV122BGRA_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar4 NV122BGRACompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.z - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar4 dst = convert_uchar4_sat((int4)(b, g, r, 255));

  return dst;
}
#endif

#if defined(NV122RGBA_U8_2D) || defined(NV122RGBA_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar4 NV122RGBACompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.z - 128;

  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;

  int r = (y + ruv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int b = (y + buv) >> NVXX_SHIFT;
  uchar4 dst = convert_uchar4_sat((int4)(r, g, b, 255));

  return dst;
}
#endif

#if defined(NV122BGRA_U8_2D) || defined(ALL_KERNELS)
Convert3To4NVXX_2D(NV122BGRA, U8)
#endif

#if defined(NV122RGBA_U8_2D) || defined(ALL_KERNELS)
Convert3To4NVXX_2D(NV122RGBA, U8)
#endif


#if defined(BGR2NV12_DISCRETE_U8_2D) || defined(RGB2NV12_DISCRETE_U8_2D) ||    \
    defined(BGR2NV21_DISCRETE_U8_2D) || defined(RGB2NV21_DISCRETE_U8_2D) ||    \
    defined(ALL_KERNELS)
#define ConvertToDISCRETE_NVXX_2D(Function, base_type)                         \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst_y, int dst_y_stride,   \
                                      global uchar* dst_uv,                    \
                                      int dst_uv_stride) {                     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar3 input_value = vload3(element_x, data);                                \
  uchar3 result = Function ## Compute(input_value, element_y, element_x);      \
                                                                               \
  data = dst_y + element_y * dst_y_stride;                                     \
  data[element_x] = result.x;                                                  \
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {                        \
    data = dst_uv + (element_y >> 1) * dst_uv_stride;                          \
    vstore2((uchar2)(result.y, result.z), (element_x >> 1), data);             \
  }                                                                            \
}
#endif

#if defined(BGR2NV12_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertToDISCRETE_NVXX_2D(BGR2NV12, U8)
#endif

#if defined(RGB2NV12_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertToDISCRETE_NVXX_2D(RGB2NV12, U8)
#endif

#if defined(BGRA2NV12_DISCRETE_U8_2D) || defined(RGBA2NV12_DISCRETE_U8_2D) ||  \
    defined(BGRA2NV21_DISCRETE_U8_2D) || defined(RGBA2NV21_DISCRETE_U8_2D) ||  \
    defined(ALL_KERNELS)
#define Convert4To3DISCRETE_NVXX_2D(Function, base_type)                       \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst_y, int dst_y_stride,   \
                                      global uchar* dst_uv,                    \
                                      int dst_uv_stride) {                     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar4 input_value = vload4(element_x, data);                                \
  uchar3 result = Function ## Compute(input_value, element_y, element_x);      \
                                                                               \
  data = dst_y + element_y * dst_y_stride;                                     \
  data[element_x] = result.x;                                                  \
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {                        \
    data = dst_uv + (element_y >> 1) * dst_uv_stride;                          \
    vstore2((uchar2)(result.y, result.z), (element_x >> 1), data);             \
  }                                                                            \
}
#endif

#if defined(BGRA2NV12_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert4To3DISCRETE_NVXX_2D(BGRA2NV12, U8)
#endif

#if defined(RGBA2NV12_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert4To3DISCRETE_NVXX_2D(RGBA2NV12, U8)
#endif

#if defined(NV122BGR_DISCRETE_U8_2D) || defined(NV122RGB_DISCRETE_U8_2D) ||    \
    defined(NV212BGR_DISCRETE_U8_2D) || defined(NV212RGB_DISCRETE_U8_2D) ||    \
    defined(ALL_KERNELS)
#define ConvertFromDISCRETE_NVXX_2D(Function, base_type)                       \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const uchar* src_y, int rows,     \
                                      int cols, int src_y_stride,              \
                                      global const uchar* src_uv,              \
                                      int src_uv_stride, global uchar* dst,    \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src_y + element_y * src_y_stride;                       \
  uchar3 input_value;                                                          \
  input_value.x = data[element_x];                                             \
                                                                               \
  data = src_uv + (element_y >> 1) * src_uv_stride;                            \
  uchar2 tmp = vload2((element_x >> 1), data);                                 \
  input_value.y = tmp.x;                                                       \
  input_value.z = tmp.y;                                                       \
  uchar3 result = Function ## Compute(input_value);                            \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore3(result, element_x, data);                                            \
}
#endif

#if defined(NV122BGR_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertFromDISCRETE_NVXX_2D(NV122BGR, U8)
#endif

#if defined(NV122RGB_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertFromDISCRETE_NVXX_2D(NV122RGB, U8)
#endif

#if defined(NV122BGRA_DISCRETE_U8_2D) || defined(NV122RGBA_DISCRETE_U8_2D) ||  \
    defined(NV212BGRA_DISCRETE_U8_2D) || defined(NV212RGBA_DISCRETE_U8_2D) ||  \
    defined(ALL_KERNELS)
#define Convert3To4DISCRETE_NVXX_2D(Function, base_type)                       \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const uchar* src_y, int rows,     \
                                      int cols, int src_y_stride,              \
                                      global const uchar* src_uv,              \
                                      int src_uv_stride, global uchar* dst,    \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src_y + element_y * src_y_stride;                       \
  uchar3 input_value;                                                          \
  input_value.x = data[element_x];                                             \
                                                                               \
  data = src_uv + (element_y >> 1) * src_uv_stride;                            \
  uchar2 tmp = vload2((element_x >> 1), data);                                 \
  input_value.y = tmp.x;                                                       \
  input_value.z = tmp.y;                                                       \
  uchar4 result = Function ## Compute(input_value);                            \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore4(result, element_x, data);                                            \
}
#endif

#if defined(NV122BGRA_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert3To4DISCRETE_NVXX_2D(NV122BGRA, U8)
#endif

#if defined(NV122RGBA_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert3To4DISCRETE_NVXX_2D(NV122RGBA, U8)
#endif

/*********************** BGR/RGB/BGRA/RGBA <-> NV21 ************************/

#if defined(BGR2NV21_U8_2D) || defined(BGR2NV21_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 BGR2NV21Compute(const uchar3 src, uint row, uint col) {
  int y = (src.x * NVXX_YB + src.y * NVXX_YG + src.z * NVXX_YR +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UB + src.y * NVXX_UG + src.z * NVXX_UR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VB + src.y * NVXX_VG + src.z * NVXX_VR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, v, u));

  return dst;
}
#endif

#if defined(RGB2NV21_U8_2D) || defined(RGB2NV21_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 RGB2NV21Compute(const uchar3 src, uint row, uint col) {
  int y = (src.x * NVXX_YR + src.y * NVXX_YG + src.z * NVXX_YB +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UR + src.y * NVXX_UG + src.z * NVXX_UB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VR + src.y * NVXX_VG + src.z * NVXX_VB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, v, u));

  return dst;
}
#endif

#if defined(BGR2NV21_U8_2D) || defined(ALL_KERNELS)
ConvertToNVXX_2D(BGR2NV21, U8)
#endif

#if defined(RGB2NV21_U8_2D) || defined(ALL_KERNELS)
ConvertToNVXX_2D(RGB2NV21, U8)
#endif

#if defined(BGRA2NV21_U8_2D) || defined(BGRA2NV21_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar3 BGRA2NV21Compute(const uchar4 src, uint row, uint col) {
  int y = (src.x * NVXX_YB + src.y * NVXX_YG + src.z * NVXX_YR +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UB + src.y * NVXX_UG + src.z * NVXX_UR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VB + src.y * NVXX_VG + src.z * NVXX_VR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, v, u));

  return dst;
}
#endif

#if defined(RGBA2NV21_U8_2D) || defined(RGBA2NV21_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar3 RGBA2NV21Compute(const uchar4 src, uint row, uint col) {
  int y = (src.x * NVXX_YR + src.y * NVXX_YG + src.z * NVXX_YB +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UR + src.y * NVXX_UG + src.z * NVXX_UB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VR + src.y * NVXX_VG + src.z * NVXX_VB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, v, u));

  return dst;
}
#endif

#if defined(BGRA2NV21_U8_2D) || defined(ALL_KERNELS)
Convert4To3NVXX_2D(BGRA2NV21, U8)
#endif

#if defined(RGBA2NV21_U8_2D) || defined(ALL_KERNELS)
Convert4To3NVXX_2D(RGBA2NV21, U8)
#endif

#if defined(NV212BGR_U8_2D) || defined(NV212BGR_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 NV212BGRCompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int v = src.y - 128;
  int u = src.z - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(b, g, r));

  return dst;
}
#endif

#if defined(NV212RGB_U8_2D) || defined(NV212RGB_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 NV212RGBCompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int v = src.y - 128;
  int u = src.z - 128;

  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;

  int r = (y + ruv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int b = (y + buv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(r, g, b));

  return dst;
}
#endif

#if defined(NV212BGR_U8_2D) || defined(ALL_KERNELS)
ConvertFromNVXX_2D(NV212BGR, U8)
#endif

#if defined(NV212RGB_U8_2D) || defined(ALL_KERNELS)
ConvertFromNVXX_2D(NV212RGB, U8)
#endif

#if defined(NV212BGRA_U8_2D) || defined(NV212BGRA_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar4 NV212BGRACompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int v = src.y - 128;
  int u = src.z - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar4 dst = convert_uchar4_sat((int4)(b, g, r, 255));

  return dst;
}
#endif

#if defined(NV212RGBA_U8_2D) || defined(NV212RGBA_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar4 NV212RGBACompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int v = src.y - 128;
  int u = src.z - 128;

  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;

  int r = (y + ruv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int b = (y + buv) >> NVXX_SHIFT;
  uchar4 dst = convert_uchar4_sat((int4)(r, g, b, 255));

  return dst;
}
#endif

#if defined(NV212BGRA_U8_2D) || defined(ALL_KERNELS)
Convert3To4NVXX_2D(NV212BGRA, U8)
#endif

#if defined(NV212RGBA_U8_2D) || defined(ALL_KERNELS)
Convert3To4NVXX_2D(NV212RGBA, U8)
#endif


#if defined(BGR2NV21_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertToDISCRETE_NVXX_2D(BGR2NV21, U8)
#endif

#if defined(RGB2NV21_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertToDISCRETE_NVXX_2D(RGB2NV21, U8)
#endif

#if defined(BGRA2NV21_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert4To3DISCRETE_NVXX_2D(BGRA2NV21, U8)
#endif

#if defined(RGBA2NV21_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert4To3DISCRETE_NVXX_2D(RGBA2NV21, U8)
#endif

#if defined(NV212BGR_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertFromDISCRETE_NVXX_2D(NV212BGR, U8)
#endif

#if defined(NV212RGB_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertFromDISCRETE_NVXX_2D(NV212RGB, U8)
#endif

#if defined(NV212BGRA_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert3To4DISCRETE_NVXX_2D(NV212BGRA, U8)
#endif

#if defined(NV212RGBA_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert3To4DISCRETE_NVXX_2D(NV212RGBA, U8)
#endif

/*********************** BGR/RGB/BGRA/RGBA <-> I420 ************************/

#if defined(BGR2I420_U8_2D) || defined(RGB2I420_U8_2D) || defined(ALL_KERNELS)
#define ConvertToI420_2D(Function, base_type)                                  \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst, int dst_stride) {     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar3 input_value = vload3(element_x, data);                                \
  uchar3 result = Function ## Compute(input_value, element_y, element_x);      \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  data[element_x] = result.x;                                                  \
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {                        \
    int half_cols = cols >> 1;                                                 \
    if (((element_y >> 1) + 1) & 1) {                                          \
      data = (global uchar*)dst + (rows + (element_y >> 2)) * dst_stride;      \
      data[element_x >> 1] = result.y;                                         \
      data += (rows >> 2) * dst_stride;                                        \
      data[element_x >> 1] = result.z;                                         \
    } else {                                                                   \
      data = (global uchar*)dst + (rows + (element_y >> 2)) * dst_stride;      \
      data[(element_x >> 1) + half_cols] = result.y;                           \
      data += (rows >> 2) * dst_stride;                                        \
      data[(element_x >> 1) + half_cols] = result.z;                           \
    }                                                                          \
  }                                                                            \
}
#endif

#if defined(BGR2I420_U8_2D) || defined(BGR2I420_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 BGR2I420Compute(const uchar3 src, uint row, uint col) {
  int y = (src.x * NVXX_YB + src.y * NVXX_YG + src.z * NVXX_YR +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UB + src.y * NVXX_UG + src.z * NVXX_UR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VB + src.y * NVXX_VG + src.z * NVXX_VR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, u, v));

  return dst;
}
#endif

#if defined(RGB2I420_U8_2D) || defined(RGB2I420_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 RGB2I420Compute(const uchar3 src, uint row, uint col) {
  int y = (src.x * NVXX_YR + src.y * NVXX_YG + src.z * NVXX_YB +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UR + src.y * NVXX_UG + src.z * NVXX_UB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VR + src.y * NVXX_VG + src.z * NVXX_VB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, u, v));

  return dst;
}
#endif

#if defined(BGR2I420_U8_2D) || defined(ALL_KERNELS)
ConvertToI420_2D(BGR2I420, U8)
#endif

#if defined(RGB2I420_U8_2D) || defined(ALL_KERNELS)
ConvertToI420_2D(RGB2I420, U8)
#endif

#if defined(BGRA2I420_U8_2D) || defined(RGBA2I420_U8_2D) || defined(ALL_KERNELS)
#define Convert4To3I420_2D(Function, base_type)                                \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst, int dst_stride) {     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar4 input_value = vload4(element_x, data);                                \
  uchar3 result = Function ## Compute(input_value, element_y, element_x);      \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  data[element_x] = result.x;                                                  \
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {                        \
    int half_cols = cols >> 1;                                                 \
    if (((element_y >> 1) + 1) & 1) {                                          \
      data = (global uchar*)dst + (rows + (element_y >> 2)) * dst_stride;      \
      data[element_x >> 1] = result.y;                                         \
      data += (rows >> 2) * dst_stride;                                        \
      data[element_x >> 1] = result.z;                                         \
    } else {                                                                   \
      data = (global uchar*)dst + (rows + (element_y >> 2)) * dst_stride;      \
      data[(element_x >> 1) + half_cols] = result.y;                           \
      data += (rows >> 2) * dst_stride;                                        \
      data[(element_x >> 1) + half_cols] = result.z;                           \
    }                                                                          \
  }                                                                            \
}
#endif

#if defined(BGRA2I420_U8_2D) || defined(BGRA2I420_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar3 BGRA2I420Compute(const uchar4 src, uint row, uint col) {
  int y = (src.x * NVXX_YB + src.y * NVXX_YG + src.z * NVXX_YR +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UB + src.y * NVXX_UG + src.z * NVXX_UR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VB + src.y * NVXX_VG + src.z * NVXX_VR + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, u, v));

  return dst;
}
#endif

#if defined(RGBA2I420_U8_2D) || defined(RGBA2I420_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar3 RGBA2I420Compute(const uchar4 src, uint row, uint col) {
  int y = (src.x * NVXX_YR + src.y * NVXX_YG + src.z * NVXX_YB +
           kG2NHalfShift + kG2NShift16) >> kG2NShift;
  int u, v;

  // Interpolate every 2 rows and 2 columns.
  if (((row + 1) & 1) && ((col + 1) & 1)) {
    u = (src.x * NVXX_UR + src.y * NVXX_UG + src.z * NVXX_UB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
    v = (src.x * NVXX_VR + src.y * NVXX_VG + src.z * NVXX_VB + kG2NHalfShift +
         kG2NShift128) >> kG2NShift;
  }
  uchar3 dst = convert_uchar3_sat((int3)(y, u, v));

  return dst;
}
#endif

#if defined(BGRA2I420_U8_2D) || defined(ALL_KERNELS)
Convert4To3I420_2D(BGRA2I420, U8)
#endif

#if defined(RGBA2I420_U8_2D) || defined(ALL_KERNELS)
Convert4To3I420_2D(RGBA2I420, U8)
#endif

#if defined(I4202BGR_U8_2D) || defined(I4202RGB_U8_2D) || defined(ALL_KERNELS)
#define ConvertFromI420_2D(Function, base_type)                                \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst, int dst_stride) {     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar3 input_value;                                                          \
  input_value.x = data[element_x];                                             \
                                                                               \
  int uv_index = element_x >> 1;                                               \
  if ((element_y >> 1) & 1) {                                                  \
    uv_index += (cols >> 1);                                                   \
  }                                                                            \
  data = src + (rows + (element_y >> 2)) * src_stride;                         \
  input_value.y = data[uv_index];                                              \
  data += (rows >> 2) * src_stride;                                            \
  input_value.z = data[uv_index];                                              \
  uchar3 result = Function ## Compute(input_value);                            \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore3(result, element_x, data);                                            \
}
#endif

#if defined(I4202BGR_U8_2D) || defined(I4202BGR_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 I4202BGRCompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.z - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(b, g, r));

  return dst;
}
#endif

#if defined(I4202RGB_U8_2D) || defined(I4202RGB_DISCRETE_U8_2D) ||             \
    defined(ALL_KERNELS)
uchar3 I4202RGBCompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.z - 128;

  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;

  int r = (y + ruv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int b = (y + buv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(r, g, b));

  return dst;
}
#endif

#if defined(I4202BGR_U8_2D) || defined(ALL_KERNELS)
ConvertFromI420_2D(I4202BGR, U8)
#endif

#if defined(I4202RGB_U8_2D) || defined(ALL_KERNELS)
ConvertFromI420_2D(I4202RGB, U8)
#endif

#if defined(I4202BGRA_U8_2D) || defined(I4202RGBA_U8_2D) || defined(ALL_KERNELS)
#define Convert3To4I420_2D(Function, base_type)                                \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst, int dst_stride) {     \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar3 input_value;                                                          \
  input_value.x = data[element_x];                                             \
                                                                               \
  int uv_index = element_x >> 1;                                               \
  if ((element_y >> 1) & 1) {                                                  \
    uv_index += (cols >> 1);                                                   \
  }                                                                            \
  data = src + (rows + (element_y >> 2)) * src_stride;                         \
  input_value.y = data[uv_index];                                              \
  data += (rows >> 2) * src_stride;                                            \
  input_value.z = data[uv_index];                                              \
  uchar4 result = Function ## Compute(input_value);                            \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore4(result, element_x, data);                                            \
}
#endif

#if defined(I4202BGRA_U8_2D) || defined(I4202BGRA_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar4 I4202BGRACompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.z - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar4 dst = convert_uchar4_sat((int4)(b, g, r, 255));

  return dst;
}
#endif

#if defined(I4202RGBA_U8_2D) || defined(I4202RGBA_DISCRETE_U8_2D) ||           \
    defined(ALL_KERNELS)
uchar4 I4202RGBACompute(const uchar3 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.z - 128;

  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;

  int r = (y + ruv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int b = (y + buv) >> NVXX_SHIFT;
  uchar4 dst = convert_uchar4_sat((int4)(r, g, b, 255));

  return dst;
}
#endif

#if defined(I4202BGRA_U8_2D) || defined(ALL_KERNELS)
Convert3To4I420_2D(I4202BGRA, U8)
#endif

#if defined(I4202RGBA_U8_2D) || defined(ALL_KERNELS)
Convert3To4I420_2D(I4202RGBA, U8)
#endif


#if defined(BGR2I420_DISCRETE_U8_2D) || defined(RGB2I420_DISCRETE_U8_2D) ||    \
    defined(ALL_KERNELS)
#define ConvertToDISCRETE_I420_2D(Function, base_type)                         \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst_y, int dst_y_stride,   \
                                      global uchar* dst_u, int dst_u_stride,   \
                                      global uchar* dst_v, int dst_v_stride) { \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar3 input_value = vload3(element_x, data);                                \
  uchar3 result = Function ## Compute(input_value, element_y, element_x);      \
                                                                               \
  data = dst_y + element_y * dst_y_stride;                                     \
  data[element_x] = result.x;                                                  \
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {                        \
    data = dst_u + (element_y >> 1) * dst_u_stride;                            \
    data[element_x >> 1] = result.y;                                           \
    data = dst_v + (element_y >> 1) * dst_v_stride;                            \
    data[element_x >> 1] = result.z;                                           \
  }                                                                            \
}
#endif

#if defined(BGR2I420_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertToDISCRETE_I420_2D(BGR2I420, U8)
#endif

#if defined(RGB2I420_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertToDISCRETE_I420_2D(RGB2I420, U8)
#endif

#if defined(BGRA2I420_DISCRETE_U8_2D) || defined(RGBA2I420_DISCRETE_U8_2D) ||  \
    defined(ALL_KERNELS)
#define Convert4To3DISCRETE_I420_2D(Function, base_type)                       \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const uchar* src, int rows,       \
                                      int cols, int src_stride,                \
                                      global uchar* dst_y, int dst_y_stride,   \
                                      global uchar* dst_u, int dst_u_stride,   \
                                      global uchar* dst_v, int dst_v_stride) { \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar4 input_value = vload4(element_x, data);                                \
  uchar3 result = Function ## Compute(input_value, element_y, element_x);      \
                                                                               \
  data = dst_y + element_y * dst_y_stride;                                     \
  data[element_x] = result.x;                                                  \
  if (((element_x + 1) & 1) && ((element_y + 1) & 1)) {                        \
    data = dst_u + (element_y >> 1) * dst_u_stride;                            \
    data[element_x >> 1] = result.y;                                           \
    data = dst_v + (element_y >> 1) * dst_v_stride;                            \
    data[element_x >> 1] = result.z;                                           \
  }                                                                            \
}
#endif

#if defined(BGRA2I420_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert4To3DISCRETE_I420_2D(BGRA2I420, U8)
#endif

#if defined(RGBA2I420_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert4To3DISCRETE_I420_2D(RGBA2I420, U8)
#endif

#if defined(I4202BGR_DISCRETE_U8_2D) || defined(I4202RGB_DISCRETE_U8_2D) ||    \
    defined(ALL_KERNELS)
#define ConvertFromDISCRETE_I420_2D(Function, base_type)                       \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const uchar* src_y, int rows,     \
                                      int cols, int src_y_stride,              \
                                      global const uchar* src_u,               \
                                      int src_u_stride,                        \
                                      global const uchar* src_v,               \
                                      int src_v_stride, global uchar* dst,     \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src_y + element_y * src_y_stride;                       \
  uchar3 input_value;                                                          \
  input_value.x = data[element_x];                                             \
                                                                               \
  data = src_u + (element_y >> 1) * src_u_stride;                              \
  input_value.y = data[element_x >> 1];                                        \
  data = src_v + (element_y >> 1) * src_v_stride;                              \
  input_value.z = data[element_x >> 1];                                        \
  uchar3 result = Function ## Compute(input_value);                            \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore3(result, element_x, data);                                            \
}
#endif

#if defined(I4202BGR_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertFromDISCRETE_I420_2D(I4202BGR, U8)
#endif

#if defined(I4202RGB_DISCRETE_U8_2D) || defined(ALL_KERNELS)
ConvertFromDISCRETE_I420_2D(I4202RGB, U8)
#endif

#if defined(I4202BGRA_DISCRETE_U8_2D) || defined(I4202RGBA_DISCRETE_U8_2D) ||  \
    defined(ALL_KERNELS)
#define Convert3To4DISCRETE_I420_2D(Function, base_type)                       \
__kernel                                                                       \
void Function ## base_type ## Kernel1(global const uchar* src_y, int rows,     \
                                      int cols, int src_y_stride,              \
                                      global const uchar* src_u,               \
                                      int src_u_stride,                        \
                                      global const uchar* src_v,               \
                                      int src_v_stride, global uchar* dst,     \
                                      int dst_stride) {                        \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  if (element_y >= rows || element_x >= cols) {                                \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src_y + element_y * src_y_stride;                       \
  uchar3 input_value;                                                          \
  input_value.x = data[element_x];                                             \
                                                                               \
  data = src_u + (element_y >> 1) * src_u_stride;                              \
  input_value.y = data[element_x >> 1];                                        \
  data = src_v + (element_y >> 1) * src_v_stride;                              \
  input_value.z = data[element_x >> 1];                                        \
  uchar4 result = Function ## Compute(input_value);                            \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore4(result, element_x, data);                                            \
}
#endif

#if defined(I4202BGRA_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert3To4DISCRETE_I420_2D(I4202BGRA, U8)
#endif

#if defined(I4202RGBA_DISCRETE_U8_2D) || defined(ALL_KERNELS)
Convert3To4DISCRETE_I420_2D(I4202RGBA, U8)
#endif

/***************************** YUV2 -> GRAY ******************************/

#if defined(YUV2GRAY_U8_2D) || defined(ALL_KERNELS)
__kernel
void YUV2GRAYU8Kernel(global const uchar* src, int rows, int cols,
                      int src_stride, global uchar* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  global uchar* input = src + element_y * src_stride;
  global uchar* output = dst + element_y * dst_stride;
  if (index_x < cols - 3) {
    uchar4 value = vload4(element_x, input);
    vstore4(value, element_x, output);
  }
  else {
    uchar value0, value1, value2, value3;
    value0 = input[index_x];
    if (index_x < cols - 1) {
      value1 = input[index_x + 1];
    }
    if (index_x < cols - 2) {
      value2 = input[index_x + 2];
    }

    output[index_x] = value0;
    if (index_x < cols - 1) {
      output[index_x + 1] = value1;
    }
    if (index_x < cols - 2) {
      output[index_x + 2] = value2;
    }
  }
}
#endif

/************************** BGR/GRAY <-> UYVY/YUYV ***************************/

#if defined(UYVY2BGR_U8_2D) || defined(YUYV2BGR_U8_2D) || defined(ALL_KERNELS)
#define ConvertFROM_YUV422_2D(Function, base_type)                             \
__kernel                                                                       \
void Function ## base_type ## Kernel(global const uchar* src, int rows,        \
                                     int cols, int src_stride,                 \
                                     global uchar* dst, int dst_stride) {      \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  int index_x = element_x << 1;                                                \
  if (element_y >= rows || index_x >= cols) {                                  \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar4 input_value = vload4(element_x, data);                                \
  uchar3 result0 = Function ## Compute0(input_value);                          \
  uchar3 result1 = Function ## Compute1(input_value);                          \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore3(result0, index_x, data);                                             \
  vstore3(result1, index_x + 1, data);                                         \
}
#endif

#if defined(UYVY2BGR_U8_2D) || defined(ALL_KERNELS)
uchar3 UYVY2BGRCompute0(const uchar4 src) {
  int y = max(0, (src.y - 16)) * NVXX_CY;
  int u = src.x - 128;
  int v = src.z - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(b, g, r));

  return dst;
}

uchar3 UYVY2BGRCompute1(const uchar4 src) {
  int y = max(0, (src.w - 16)) * NVXX_CY;
  int u = src.x - 128;
  int v = src.z - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(b, g, r));

  return dst;
}

ConvertFROM_YUV422_2D(UYVY2BGR, U8)
#endif

#if defined(YUYV2BGR_U8_2D) || defined(ALL_KERNELS)
uchar3 YUYV2BGRCompute0(const uchar4 src) {
  int y = max(0, (src.x - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.w - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(b, g, r));

  return dst;
}

uchar3 YUYV2BGRCompute1(const uchar4 src) {
  int y = max(0, (src.z - 16)) * NVXX_CY;
  int u = src.y - 128;
  int v = src.w - 128;

  int buv = (1 << (NVXX_SHIFT - 1)) + NVXX_CUB * u;
  int guv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVG * v + NVXX_CUG * u;
  int ruv = (1 << (NVXX_SHIFT - 1)) + NVXX_CVR * v;

  int b = (y + buv) >> NVXX_SHIFT;
  int g = (y + guv) >> NVXX_SHIFT;
  int r = (y + ruv) >> NVXX_SHIFT;
  uchar3 dst = convert_uchar3_sat((int3)(b, g, r));

  return dst;
}

ConvertFROM_YUV422_2D(YUYV2BGR, U8)
#endif

#if defined(UYVY2GRAY_U8_2D) || defined(YUYV2GRAY_U8_2D) || defined(ALL_KERNELS)
#define ConvertFROM_YUV422_2D(Function, base_type)                             \
__kernel                                                                       \
void Function ## base_type ## Kernel(global const uchar* src, int rows,        \
                                     int cols, int src_stride,                 \
                                     global uchar* dst, int dst_stride) {      \
  int element_x = get_global_id(0);                                            \
  int element_y = get_global_id(1);                                            \
  int index_x = element_x << 1;                                                \
  if (element_y >= rows || index_x >= cols) {                                  \
    return;                                                                    \
  }                                                                            \
                                                                               \
  global uchar* data = src + element_y * src_stride;                           \
  uchar2 input_value0 = vload2(index_x, data);                                 \
  uchar2 input_value1 = vload2(index_x + 1, data);                             \
  uchar2 result = Function ## Compute(input_value0, input_value1);             \
                                                                               \
  data = dst + element_y * dst_stride;                                         \
  vstore2(result, element_x, data);                                            \
}
#endif

#if defined(UYVY2GRAY_U8_2D) || defined(ALL_KERNELS)
uchar2 UYVY2GRAYCompute(const uchar2 src0, const uchar2 src1) {
  uchar2 dst;
  dst.x = src0.y;
  dst.y = src1.y;

  return dst;
}

ConvertFROM_YUV422_2D(UYVY2GRAY, U8)
#endif

#if defined(YUYV2GRAY_U8_2D) || defined(ALL_KERNELS)
uchar2 YUYV2GRAYCompute(const uchar2 src0, const uchar2 src1) {
  uchar2 dst;
  dst.x = src0.x;
  dst.y = src1.x;

  return dst;
}

ConvertFROM_YUV422_2D(YUYV2GRAY, U8)
#endif
