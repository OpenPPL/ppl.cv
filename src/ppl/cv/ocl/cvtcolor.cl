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

#if defined(BGR2BGRA_U8_1D) || defined(BGR2BGRA_F32_1D) || defined(SPIR)
#define BGR2BGRA_1D(base_type, T, T3, T4, alpha)                               \
__kernel                                                                       \
void BGR2BGRA ## base_type ## Kernel0(global const T* src, int cols,           \
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

#if defined(BGR2BGRA_U8_2D) || defined(BGR2BGRA_F32_2D) || defined(SPIR)
#define BGR2BGRA_2D(base_type, T, T3, T4, alpha)                               \
__kernel                                                                       \
void BGR2BGRA ## base_type ## Kernel1(global const T* src, int rows, int cols, \
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
BGR2BGRA_1D(U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2BGRA_U8_2D) || defined(SPIR)
BGR2BGRA_2D(U8, uchar, uchar3, uchar4, 255)
#endif

#if defined(BGR2BGRA_F32_1D) || defined(SPIR)
BGR2BGRA_1D(F32, float, float3, float4, 1.f)
#endif

#if defined(BGR2BGRA_F32_2D) || defined(SPIR)
BGR2BGRA_2D(F32, float, float3, float4, 1.f)
#endif

