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
    defined(RGB2RGBA_U8_1D) || defined(RGB2RGBA_F32_1D) || defined(SPIR)
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
    defined(RGB2RGBA_U8_2D) || defined(RGB2RGBA_F32_2D) || defined(SPIR)
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

#if defined(BGR2BGRA_U8_1D) || defined(SPIR)
BGR2BGRATYPE_1D(BGR2BGRA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2BGRA_U8_2D) || defined(SPIR)
BGR2BGRATYPE_2D(BGR2BGRA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2BGRA_F32_1D) || defined(SPIR)
BGR2BGRATYPE_1D(BGR2BGRA, F32, float, float3, float4, 1.f)
#endif

#if defined(BGR2BGRA_F32_2D) || defined(SPIR)
BGR2BGRATYPE_2D(BGR2BGRA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGB2RGBA_U8_1D) || defined(SPIR)
BGR2BGRATYPE_1D(RGB2RGBA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(RGB2RGBA_U8_2D) || defined(SPIR)
BGR2BGRATYPE_2D(RGB2RGBA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(RGB2RGBA_F32_1D) || defined(SPIR)
BGR2BGRATYPE_1D(RGB2RGBA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGB2RGBA_F32_2D) || defined(SPIR)
BGR2BGRATYPE_2D(RGB2RGBA, F32, float, float3, float4, 1.f)
#endif

#if defined(BGRA2BGR_U8_1D) || defined(BGRA2BGR_F32_1D) ||                     \
    defined(RGBA2RGB_U8_1D) || defined(RGBA2RGB_F32_1D) || defined(SPIR)
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
    defined(RGBA2RGB_U8_2D) || defined(RGBA2RGB_F32_2D) || defined(SPIR)
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

#if defined(BGRA2BGR_U8_1D) || defined(SPIR)
BGRA2BGRTYPE_1D(BGRA2BGR, U8, uchar, uchar3, uchar4)
#endif

#if defined(BGRA2BGR_U8_2D) || defined(SPIR)
BGRA2BGRTYPE_2D(BGRA2BGR, U8, uchar, uchar3, uchar4)
#endif

#if defined(BGRA2BGR_F32_1D) || defined(SPIR)
BGRA2BGRTYPE_1D(BGRA2BGR, F32, float, float3, float4)
#endif

#if defined(BGRA2BGR_F32_2D) || defined(SPIR)
BGRA2BGRTYPE_2D(BGRA2BGR, F32, float, float3, float4)
#endif

#if defined(RGBA2RGB_U8_1D) || defined(SPIR)
BGRA2BGRTYPE_1D(RGBA2RGB, U8, uchar, uchar3, uchar4)
#endif

#if defined(RGBA2RGB_U8_2D) || defined(SPIR)
BGRA2BGRTYPE_2D(RGBA2RGB, U8, uchar, uchar3, uchar4)
#endif

#if defined(RGBA2RGB_F32_1D) || defined(SPIR)
BGRA2BGRTYPE_1D(RGBA2RGB, F32, float, float3, float4)
#endif

#if defined(RGBA2RGB_F32_2D) || defined(SPIR)
BGRA2BGRTYPE_2D(RGBA2RGB, F32, float, float3, float4)
#endif

#if defined(BGR2RGBA_U8_1D) || defined(BGR2RGBA_F32_1D) ||                     \
    defined(RGB2BGRA_U8_1D) || defined(RGB2BGRA_F32_1D) || defined(SPIR)
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
    defined(RGB2BGRA_U8_2D) || defined(RGB2BGRA_F32_2D) || defined(SPIR)
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

#if defined(BGR2RGBA_U8_1D) || defined(SPIR)
BGR2RGBATYPE_1D(BGR2RGBA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2RGBA_U8_2D) || defined(SPIR)
BGR2RGBATYPE_2D(BGR2RGBA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2RGBA_F32_1D) || defined(SPIR)
BGR2RGBATYPE_1D(BGR2RGBA, F32, float, float3, float4, 1.f)
#endif

#if defined(BGR2RGBA_F32_2D) || defined(SPIR)
BGR2RGBATYPE_2D(BGR2RGBA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGB2BGRA_U8_1D) || defined(SPIR)
BGR2RGBATYPE_1D(RGB2BGRA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(RGB2BGRA_U8_2D) || defined(SPIR)
BGR2RGBATYPE_2D(RGB2BGRA, U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(RGB2BGRA_F32_1D) || defined(SPIR)
BGR2RGBATYPE_1D(RGB2BGRA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGB2BGRA_F32_2D) || defined(SPIR)
BGR2RGBATYPE_2D(RGB2BGRA, F32, float, float3, float4, 1.f)
#endif

#if defined(RGBA2BGR_U8_1D) || defined(RGBA2BGR_F32_1D) ||                     \
    defined(BGRA2RGB_U8_1D) || defined(BGRA2RGB_F32_1D) || defined(SPIR)
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
    defined(BGRA2RGB_U8_2D) || defined(BGRA2RGB_F32_2D) || defined(SPIR)
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

#if defined(RGBA2BGR_U8_1D) || defined(SPIR)
RGBA2BGRTYPE_1D(RGBA2BGR, U8, uchar, uchar3, uchar4)
#endif

#if defined(RGBA2BGR_U8_2D) || defined(SPIR)
RGBA2BGRTYPE_2D(RGBA2BGR, U8, uchar, uchar3, uchar4)
#endif

#if defined(RGBA2BGR_F32_1D) || defined(SPIR)
RGBA2BGRTYPE_1D(RGBA2BGR, F32, float, float3, float4)
#endif

#if defined(RGBA2BGR_F32_2D) || defined(SPIR)
RGBA2BGRTYPE_2D(RGBA2BGR, F32, float, float3, float4)
#endif

#if defined(BGRA2RGB_U8_1D) || defined(SPIR)
RGBA2BGRTYPE_1D(BGRA2RGB, U8, uchar, uchar3, uchar4)
#endif

#if defined(BGRA2RGB_U8_2D) || defined(SPIR)
RGBA2BGRTYPE_2D(BGRA2RGB, U8, uchar, uchar3, uchar4)
#endif

#if defined(BGRA2RGB_F32_1D) || defined(SPIR)
RGBA2BGRTYPE_1D(BGRA2RGB, F32, float, float3, float4)
#endif

#if defined(BGRA2RGB_F32_2D) || defined(SPIR)
RGBA2BGRTYPE_2D(BGRA2RGB, F32, float, float3, float4)
#endif

/******************************* BGR <-> RGB ******************************/

#if defined(RGB2BGR_U8_1D) || defined(RGB2BGR_F32_1D) ||                       \
    defined(BGR2RGB_U8_1D) || defined(BGR2RGB_F32_1D) || defined(SPIR)
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
    defined(BGR2RGB_U8_2D) || defined(BGR2RGB_F32_2D) || defined(SPIR)
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

#if defined(BGR2RGB_U8_1D) || defined(SPIR)
RGB2BGRTYPE_1D(BGR2RGB, U8, uchar, uchar3)
#endif

#if defined(BGR2RGB_U8_2D) || defined(SPIR)
RGB2BGRTYPE_2D(BGR2RGB, U8, uchar, uchar3)
#endif

#if defined(BGR2RGB_F32_1D) || defined(SPIR)
RGB2BGRTYPE_1D(BGR2RGB, F32, float, float3)
#endif

#if defined(BGR2RGB_F32_2D) || defined(SPIR)
RGB2BGRTYPE_2D(BGR2RGB, F32, float, float3)
#endif

#if defined(RGB2BGR_U8_1D) || defined(SPIR)
RGB2BGRTYPE_1D(RGB2BGR, U8, uchar, uchar3)
#endif

#if defined(RGB2BGR_U8_2D) || defined(SPIR)
RGB2BGRTYPE_2D(RGB2BGR, U8, uchar, uchar3)
#endif

#if defined(RGB2BGR_F32_1D) || defined(SPIR)
RGB2BGRTYPE_1D(RGB2BGR, F32, float, float3)
#endif

#if defined(RGB2BGR_F32_2D) || defined(SPIR)
RGB2BGRTYPE_2D(RGB2BGR, F32, float, float3)
#endif

/******************************* BGRA <-> RGBA ******************************/

#if defined(BGRA2RGBA_U8_1D) || defined(BGRA2RGBA_F32_1D) ||                   \
    defined(RGBA2BGRA_U8_1D) || defined(RGBA2BGRA_F32_1D) || defined(SPIR)
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
    defined(RGBA2BGRA_U8_2D) || defined(RGBA2BGRA_F32_2D) || defined(SPIR)
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

#if defined(BGRA2RGBA_U8_1D) || defined(SPIR)
BGRA2RGBATYPE_1D(BGRA2RGBA, U8, uchar, uchar4)
#endif

#if defined(BGRA2RGBA_U8_2D) || defined(SPIR)
BGRA2RGBATYPE_2D(BGRA2RGBA, U8, uchar, uchar4)
#endif

#if defined(BGRA2RGBA_F32_1D) || defined(SPIR)
BGRA2RGBATYPE_1D(BGRA2RGBA, F32, float, float4)
#endif

#if defined(BGRA2RGBA_F32_2D) || defined(SPIR)
BGRA2RGBATYPE_2D(BGRA2RGBA, F32, float, float4)
#endif

#if defined(RGBA2BGRA_U8_1D) || defined(SPIR)
BGRA2RGBATYPE_1D(RGBA2BGRA, U8, uchar, uchar4)
#endif

#if defined(RGBA2BGRA_U8_2D) || defined(SPIR)
BGRA2RGBATYPE_2D(RGBA2BGRA, U8, uchar, uchar4)
#endif

#if defined(RGBA2BGRA_F32_1D) || defined(SPIR)
BGRA2RGBATYPE_1D(RGBA2BGRA, F32, float, float4)
#endif

#if defined(RGBA2BGRA_F32_2D) || defined(SPIR)
BGRA2RGBATYPE_2D(RGBA2BGRA, F32, float, float4)
#endif

/*********************** BGR/RGB/BGRA/RGBA <-> Gray ************************/

enum Bgr2GrayCoefficients {
  kB2Y15    = 3735,
  kG2Y15    = 19235,
  kR2Y15    = 9798,
  kRgbShift = 15,
};

/* #if defined(BGR2GRAY_U8_1D) || defined(BGR2GRAY_F32_1D) ||                     \
    defined(RGB2GRAY_U8_1D) || defined(RGB2GRAY_F32_1D) || defined(SPIR)
#define BGR2GRAYTYPE_1D(Function, base_type, T, T3)                 \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  int index_x = element_x << 1;                                            \
  if (index_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T3 input_value0 = vload3(index_x, src);                                     \
  T3 input_value1 = vload3(index_x + 1, src);                                     \
  T value0 = Function ## Compute(input_value0);                                  \
  T value1 = Function ## Compute(input_value1);                                  \
                                                                               \
  dst[index_x] = value0;                                              \
  dst[index_x + 1] = value1;                                              \
}
#endif */

#if defined(BGR2GRAY_U8_1D) || defined(BGR2GRAY_F32_1D) ||                     \
    defined(RGB2GRAY_U8_1D) || defined(RGB2GRAY_F32_1D) || defined(SPIR)
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
  T value = Function ## Compute(input_value);                                  \
                                                                               \
  dst[element_x] = value;                                                      \
}
#endif

#if defined(BGR2GRAY_U8_2D) || defined(BGR2GRAY_F32_2D) ||                     \
    defined(RGB2GRAY_U8_2D) || defined(RGB2GRAY_F32_2D) || defined(SPIR)
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
  T value = Function ## Compute(input_value);                                  \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  data[element_x] = value;                                                     \
}
#endif

#if defined(BGR2GRAY_U8_1D) || defined(BGR2GRAY_U8_2D) || defined(SPIR)
uchar BGR2GRAYCompute(const uchar3 src) {
  int b = src.x;
  int g = src.y;
  int r = src.z;
  uchar dst = divideUp(b * kB2Y15 + g * kG2Y15 + r * kR2Y15, kRgbShift);

  return dst;
}
#endif

#if defined(BGR2GRAY_F32_1D) || defined(BGR2GRAY_F32_2D) || defined(SPIR)
float BGR2GRAYCompute(const float3 src) {
  float b = src.x;
  float g = src.y;
  float r = src.z;
  float dst = b * 0.114f + g * 0.587f + r * 0.299f;

  return dst;
}
#endif

#if defined(RGB2GRAY_U8_1D) || defined(RGB2GRAY_U8_2D) || defined(SPIR)
uchar RGB2GRAYCompute(const uchar3 src) {
  int r = src.x;
  int g = src.y;
  int b = src.z;
  uchar dst = divideUp(r * kR2Y15 + g * kG2Y15 + b * kB2Y15, kRgbShift);

  return dst;
}
#endif

#if defined(RGB2GRAY_F32_1D) || defined(RGB2GRAY_F32_2D) || defined(SPIR)
float RGB2GRAYCompute(const float3 src) {
  float r = src.x;
  float g = src.y;
  float b = src.z;
  float dst = r * 0.299f + g * 0.587f + b * 0.114f;

  return dst;
}
#endif

#if defined(BGR2GRAY_U8_1D) || defined(SPIR)
BGR2GRAYTYPE_1D(BGR2GRAY, U8, uchar, uchar3)
#endif

#if defined(BGR2GRAY_U8_2D) || defined(SPIR)
BGR2GRAYTYPE_2D(BGR2GRAY, U8, uchar, uchar3)
#endif

#if defined(BGR2GRAY_F32_1D) || defined(SPIR)
BGR2GRAYTYPE_1D(BGR2GRAY, F32, float, float3)
#endif

#if defined(BGR2GRAY_F32_2D) || defined(SPIR)
BGR2GRAYTYPE_2D(BGR2GRAY, F32, float, float3)
#endif

#if defined(RGB2GRAY_U8_1D) || defined(SPIR)
BGR2GRAYTYPE_1D(RGB2GRAY, U8, uchar, uchar3)
#endif

#if defined(RGB2GRAY_U8_2D) || defined(SPIR)
BGR2GRAYTYPE_2D(RGB2GRAY, U8, uchar, uchar3)
#endif

#if defined(RGB2GRAY_F32_1D) || defined(SPIR)
BGR2GRAYTYPE_1D(RGB2GRAY, F32, float, float3)
#endif

#if defined(RGB2GRAY_F32_2D) || defined(SPIR)
BGR2GRAYTYPE_2D(RGB2GRAY, F32, float, float3)
#endif

/* #if defined(BGRA2GRAY_U8_1D) || defined(BGRA2GRAY_F32_1D) ||                   \
    defined(RGBA2GRAY_U8_1D) || defined(RGBA2GRAY_F32_1D) || defined(SPIR)
#define BGRA2GRAYTYPE_1D(Function, base_type, T, T4)                           \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  int index_x = element_x << 1;                                            \
  if (index_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T4 input_value0 = vload4(index_x, src);                                     \
  T4 input_value1 = vload4(index_x + 1, src);                                     \
  T value0 = Function ## Compute(input_value0);                                  \
  T value1 = Function ## Compute(input_value1);                                  \
                                                                               \
  dst[index_x] = value0;                                              \
  dst[index_x + 1] = value1;                                              \
}
#endif */

#if defined(BGRA2GRAY_U8_1D) || defined(BGRA2GRAY_F32_1D) ||                   \
    defined(RGBA2GRAY_U8_1D) || defined(RGBA2GRAY_F32_1D) || defined(SPIR)
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
  T value = Function ## Compute(input_value);                                  \
                                                                               \
  dst[element_x] = value;                                                      \
}
#endif

#if defined(BGRA2GRAY_U8_2D) || defined(BGRA2GRAY_F32_2D) ||                   \
    defined(RGBA2GRAY_U8_2D) || defined(RGBA2GRAY_F32_2D) || defined(SPIR)
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
  T value = Function ## Compute(input_value);                                  \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  data[element_x] = value;                                                     \
}
#endif

#if defined(BGRA2GRAY_U8_1D) || defined(BGRA2GRAY_U8_2D) || defined(SPIR)
uchar BGRA2GRAYCompute(const uchar4 src) {
  int b = src.x;
  int g = src.y;
  int r = src.z;
  uchar dst = divideUp(b * kB2Y15 + g * kG2Y15 + r * kR2Y15, kRgbShift);

  return dst;
}
#endif

#if defined(BGRA2GRAY_F32_1D) || defined(BGRA2GRAY_F32_2D) || defined(SPIR)
float BGRA2GRAYCompute(const float4 src) {
  float b = src.x;
  float g = src.y;
  float r = src.z;
  float dst = b * 0.114f + g * 0.587f + r * 0.299f;

  return dst;
}
#endif

#if defined(RGBA2GRAY_U8_1D) || defined(RGBA2GRAY_U8_2D) || defined(SPIR)
uchar RGBA2GRAYCompute(const uchar4 src) {
  int r = src.x;
  int g = src.y;
  int b = src.z;
  uchar dst = divideUp(r * kR2Y15 + g * kG2Y15 + b * kB2Y15, kRgbShift);

  return dst;
}
#endif

#if defined(RGBA2GRAY_F32_1D) || defined(RGBA2GRAY_F32_2D) || defined(SPIR)
float RGBA2GRAYCompute(const float4 src) {
  float r = src.x;
  float g = src.y;
  float b = src.z;
  float dst = r * 0.299f + g * 0.587f + b * 0.114f;

  return dst;
}
#endif

#if defined(BGRA2GRAY_U8_1D) || defined(SPIR)
BGRA2GRAYTYPE_1D(BGRA2GRAY, U8, uchar, uchar4)
#endif

#if defined(BGRA2GRAY_U8_2D) || defined(SPIR)
BGRA2GRAYTYPE_2D(BGRA2GRAY, U8, uchar, uchar4)
#endif

#if defined(BGRA2GRAY_F32_1D) || defined(SPIR)
BGRA2GRAYTYPE_1D(BGRA2GRAY, F32, float, float4)
#endif

#if defined(BGRA2GRAY_F32_2D) || defined(SPIR)
BGRA2GRAYTYPE_2D(BGRA2GRAY, F32, float, float4)
#endif

#if defined(RGBA2GRAY_U8_1D) || defined(SPIR)
BGRA2GRAYTYPE_1D(RGBA2GRAY, U8, uchar, uchar4)
#endif

#if defined(RGBA2GRAY_U8_2D) || defined(SPIR)
BGRA2GRAYTYPE_2D(RGBA2GRAY, U8, uchar, uchar4)
#endif

#if defined(RGBA2GRAY_F32_1D) || defined(SPIR)
BGRA2GRAYTYPE_1D(RGBA2GRAY, F32, float, float4)
#endif

#if defined(RGBA2GRAY_F32_2D) || defined(SPIR)
BGRA2GRAYTYPE_2D(RGBA2GRAY, F32, float, float4)
#endif

/* #if defined(GRAY2BGR_U8_1D) || defined(GRAY2BGR_F32_1D) ||                   \
    defined(GRAY2RGB_U8_1D) || defined(GRAY2RGB_F32_1D) || defined(SPIR)
#define GRAY2BGRTYPE_1D(Function, base_type, T, T3)                           \
__kernel                                                                       \
void Function ## base_type ## Kernel0(global const T* src, int cols,           \
                                      global T* dst) {                         \
  int element_x = get_global_id(0);                                            \
  int index_x = element_x << 1;                                            \
  if (index_x >= cols) {                                                     \
    return;                                                                    \
  }                                                                            \
                                                                               \
  T input_value0 = src[index_x];                                     \
  T input_value1 = src[index_x + 1];                                     \
  T3 value0 = Function ## Compute(input_value0);                                  \
  T3 value1 = Function ## Compute(input_value1);                                  \
                                                                               \
  vstore3(value0, index_x, dst);                                              \
  vstore3(value1, index_x + 1, dst);                                              \
}
#endif */

#if defined(GRAY2BGR_U8_1D) || defined(GRAY2BGR_F32_1D) ||                     \
    defined(GRAY2RGB_U8_1D) || defined(GRAY2RGB_F32_1D) || defined(SPIR)
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
  T3 value = Function ## Compute(input_value);                                 \
                                                                               \
  vstore3(value, element_x, dst);                                              \
}
#endif

#if defined(GRAY2BGR_U8_2D) || defined(GRAY2BGR_F32_2D) ||                     \
    defined(GRAY2RGB_U8_2D) || defined(GRAY2RGB_F32_2D) || defined(SPIR)
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
  T3 value = Function ## Compute(input_value);                                 \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore3(value, element_x, data);                                             \
}
#endif

#if defined(GRAY2BGR_U8_1D) || defined(GRAY2BGR_U8_2D) || defined(SPIR)
uchar3 GRAY2BGRCompute(const uchar src) {
    uchar3 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;

    return dst;
}
#endif

#if defined(GRAY2BGR_F32_1D) || defined(GRAY2BGR_F32_2D) || defined(SPIR)
float3 GRAY2BGRCompute(const float src) {
    float3 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;

    return dst;
}
#endif

#if defined(GRAY2RGB_U8_1D) || defined(GRAY2RGB_U8_2D) || defined(SPIR)
uchar3 GRAY2RGBCompute(const uchar src) {
    uchar3 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;

    return dst;
}
#endif

#if defined(GRAY2RGB_F32_1D) || defined(GRAY2RGB_F32_2D) || defined(SPIR)
float3 GRAY2RGBCompute(const float src) {
    float3 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;

    return dst;
}
#endif

#if defined(GRAY2BGR_U8_1D) || defined(SPIR)
GRAY2BGRTYPE_1D(GRAY2BGR, U8, uchar, uchar3)
#endif

#if defined(GRAY2BGR_U8_2D) || defined(SPIR)
GRAY2BGRTYPE_2D(GRAY2BGR, U8, uchar, uchar3)
#endif

#if defined(GRAY2BGR_F32_1D) || defined(SPIR)
GRAY2BGRTYPE_1D(GRAY2BGR, F32, float, float3)
#endif

#if defined(GRAY2BGR_F32_2D) || defined(SPIR)
GRAY2BGRTYPE_2D(GRAY2BGR, F32, float, float3)
#endif

#if defined(GRAY2RGB_U8_1D) || defined(SPIR)
GRAY2BGRTYPE_1D(GRAY2RGB, U8, uchar, uchar3)
#endif

#if defined(GRAY2RGB_U8_2D) || defined(SPIR)
GRAY2BGRTYPE_2D(GRAY2RGB, U8, uchar, uchar3)
#endif

#if defined(GRAY2RGB_F32_1D) || defined(SPIR)
GRAY2BGRTYPE_1D(GRAY2RGB, F32, float, float3)
#endif

#if defined(GRAY2RGB_F32_2D) || defined(SPIR)
GRAY2BGRTYPE_2D(GRAY2RGB, F32, float, float3)
#endif

#if defined(GRAY2BGRA_U8_1D) || defined(GRAY2BGRA_F32_1D) ||                   \
    defined(GRAY2RGBA_U8_1D) || defined(GRAY2RGBA_F32_1D) || defined(SPIR)
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
  T4 value = Function ## Compute(input_value);                                 \
                                                                               \
  vstore4(value, element_x, dst);                                              \
}
#endif

#if defined(GRAY2BGRA_U8_2D) || defined(GRAY2BGRA_F32_2D) ||                   \
    defined(GRAY2RGBA_U8_2D) || defined(GRAY2RGBA_F32_2D) || defined(SPIR)
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
  T4 value = Function ## Compute(input_value);                                 \
                                                                               \
  data = (global T*)((global uchar*)dst + element_y * dst_stride);             \
  vstore4(value, element_x, data);                                             \
}
#endif

#if defined(GRAY2BGRA_U8_1D) || defined(GRAY2BGRA_U8_2D) || defined(SPIR)
uchar4 GRAY2BGRACompute(const uchar src) {
    uchar4 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;
    dst.w = 255;

    return dst;
}
#endif

#if defined(GRAY2BGRA_F32_1D) || defined(GRAY2BGRA_F32_2D) || defined(SPIR)
float4 GRAY2BGRACompute(const float src) {
    float4 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;
    dst.w = 1.0f;

    return dst;
}
#endif

#if defined(GRAY2RGBA_U8_1D) || defined(GRAY2RGBA_U8_2D) || defined(SPIR)
uchar4 GRAY2RGBACompute(const uchar src) {
    uchar4 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;
    dst.w = 255;

    return dst;
}
#endif

#if defined(GRAY2RGBA_F32_1D) || defined(GRAY2RGBA_F32_2D) || defined(SPIR)
float4 GRAY2RGBACompute(const float src) {
    float4 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;
    dst.w = 1.0f;

    return dst;
}
#endif

#if defined(GRAY2BGRA_U8_1D) || defined(SPIR)
GRAY2BGRATYPE_1D(GRAY2BGRA, U8, uchar, uchar4)
#endif

#if defined(GRAY2BGRA_U8_2D) || defined(SPIR)
GRAY2BGRATYPE_2D(GRAY2BGRA, U8, uchar, uchar4)
#endif

#if defined(GRAY2BGRA_F32_1D) || defined(SPIR)
GRAY2BGRATYPE_1D(GRAY2BGRA, F32, float, float4)
#endif

#if defined(GRAY2BGRA_F32_2D) || defined(SPIR)
GRAY2BGRATYPE_2D(GRAY2BGRA, F32, float, float4)
#endif

#if defined(GRAY2RGBA_U8_1D) || defined(SPIR)
GRAY2BGRATYPE_1D(GRAY2RGBA, U8, uchar, uchar4)
#endif

#if defined(GRAY2RGBA_U8_2D) || defined(SPIR)
GRAY2BGRATYPE_2D(GRAY2RGBA, U8, uchar, uchar4)
#endif

#if defined(GRAY2RGBA_F32_1D) || defined(SPIR)
GRAY2BGRATYPE_1D(GRAY2RGBA, F32, float, float4)
#endif

#if defined(GRAY2RGBA_F32_2D) || defined(SPIR)
GRAY2BGRATYPE_2D(GRAY2RGBA, F32, float, float4)
#endif

