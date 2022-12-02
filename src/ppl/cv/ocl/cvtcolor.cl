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
