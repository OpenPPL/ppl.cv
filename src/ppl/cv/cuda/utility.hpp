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
 *
 * Definition of macro, typedef, enum, function templates, and inline
 * functions to facilitate computation.
 */


#ifndef _ST_HPC_PPL3_CV_CUDA_UTILITY_HPP_
#define _ST_HPC_PPL3_CV_CUDA_UTILITY_HPP_

#include <cstdlib>

#include "cuda_runtime.h"

#include "ppl/common/log.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

#define DEVICE_INLINE
#if defined(DEVICE_INLINE)
# define __INLINE__ __device__ __forceinline__
#else
# define __INLINE__ inline
#endif

#define PPL_ASSERT(expression)                                                 \
if (!(expression)) {                                                           \
  LOG(ERROR) << "Assertion failed: " << #expression;                           \
  return ppl::common::RC_INVALID_VALUE;                                        \
}

typedef unsigned char uchar;
typedef signed char schar;

// configuration of thread block
enum DimX {
  kDimX0 = 16,
  kDimX1 = 32,
  kDimX2 = 32,
  kDimX3 = 32,
  kDimX4 = 128,
  kDimX5 = 256,
  kDimX6 = 64,
  kDimX7 = 32,
  kDimX8 = 32,
  kDimX9 = 64,
};

enum DimY {
  kDimY0 = 16,
  kDimY1 = 32,
  kDimY2 = 16,
  kDimY3 = 8,
  kDimY4 = 1,
  kDimY5 = 1,
  kDimY6 = 1,
  kDimY7 = 1,
  kDimY8 = 4,
  kDimY9 = 2,
};

enum ShiftX {
  kShiftX0 = 4,
  kShiftX1 = 5,
  kShiftX2 = 5,
  kShiftX3 = 5,
  kShiftX4 = 7,
  kShiftX5 = 8,
  kShiftX6 = 6,
  kShiftX7 = 5,
  kShiftX8 = 5,
  kShiftX9 = 6,
};

enum ShiftY {
  kShiftY0 = 4,
  kShiftY1 = 5,
  kShiftY2 = 4,
  kShiftY3 = 3,
  kShiftY4 = 0,
  kShiftY5 = 0,
  kShiftY6 = 0,
  kShiftY7 = 0,
  kShiftY8 = 2,
  kShiftY9 = 1,
};

enum BlockConfiguration0 {
  kBlockDimX0 = kDimX8,
  kBlockDimY0 = kDimY8,
  kBlockShiftX0 = kShiftX8,
  kBlockShiftY0 = kShiftY8,
};

enum BlockConfiguration1 {
  kBlockDimX1 = kDimX3,
  kBlockDimY1 = kDimY3,
  kBlockShiftX1 = kShiftX3,
  kBlockShiftY1 = kShiftY3,
};

/*
 * rounding up total / grain, where gian = (1 << bits).
 */
__host__ __device__
inline
int divideUp(int total, int grain, int shift) {
  return (total + grain - 1) >> shift;
}

__host__ __device__
inline
int divideUp(int total, int shift) {
  return (total + (1 << (shift - 1))) >> shift;
}

__host__ __device__
inline
int roundUp(int total, int grain, int shift) {
  return ((total + grain - 1) >> shift) << shift;
}

template <typename T>
__INLINE__
T max(const T& value1, const T& value2) {
  if (value1 < value2) {
    return value2;
  }
  else {
    return value1;
  }
}

template <typename T>
__INLINE__
T min(const T& value1, const T& value2) {
  if (value1 > value2) {
    return value2;
  }
  else {
    return value1;
  }
}

template <typename T>
__INLINE__
T max(const T& value1, const T& value2, const T& value3) {
  if (value1 < value2) {
    return value2 > value3 ? value2 : value3;
  }
  else {
    return value1 > value3 ? value1 : value3;
  }
}

template <typename T>
__INLINE__
T min(const T& value1, const T& value2, const T& value3) {
  if (value1 > value2) {
    return value2 < value3 ? value2 : value3;
  }
  else {
    return value1 < value3 ? value1 : value3;
  }
}

template<typename T>
__INLINE__
T clip(const T& value, const T& min_value, const T& max_value) {
    return min(max(value, min_value), max_value);
}

__INLINE__
uchar saturate_cast(int value) {
  unsigned int result = 0;
  asm("cvt.sat.u8.s32 %0, %1;" : "=r"(result) : "r"(value));
  return result;
}

__INLINE__
uchar saturate_cast(float value) {
  unsigned int result = 0;
  asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(result) : "f"(value));
  return result;
}

__INLINE__
short saturate_cast_i2s(int value) {
  short result = 0;
  asm("cvt.sat.s16.s32 %0, %1;" : "=h"(result) : "r"(value));
  return result;
}

__INLINE__
short saturate_cast_f2s(float value) {
  short result = 0;
  asm("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(result) : "f"(value));
  return result;
}

template <typename T0, typename T1>
__INLINE__
T0 saturate_cast_vector(T1 value);

template <>
__INLINE__
uchar2 saturate_cast_vector(float2 value) {
  uchar2 result;
  result.x = saturate_cast(value.x);
  result.y = saturate_cast(value.y);

  return result;
}

template <>
__INLINE__
float2 saturate_cast_vector(float2 value) {
  return value;
}

template <>
__INLINE__
uchar3 saturate_cast_vector(float3 value) {
  uchar3 result;
  result.x = saturate_cast(value.x);
  result.y = saturate_cast(value.y);
  result.z = saturate_cast(value.z);

  return result;
}

template <>
__INLINE__
float3 saturate_cast_vector(float3 value) {
  return value;
}

template <>
__INLINE__
uchar4 saturate_cast_vector(float4 value) {
  uchar4 result;
  result.x = saturate_cast(value.x);
  result.y = saturate_cast(value.y);
  result.z = saturate_cast(value.z);
  result.w = saturate_cast(value.w);

  return result;
}

template <>
__INLINE__
float4 saturate_cast_vector(float4 value) {
  return value;
}

__INLINE__
float2 operator*(float value0, uchar2 value1) {
  float2 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;

  return result;
}

__INLINE__
void operator+=(float2 &result, uchar2 &value) {
  result.x += value.x;
  result.y += value.y;
}

__INLINE__
void operator+=(float2 &result, float2 &value) {
  result.x += value.x;
  result.y += value.y;
}

__INLINE__
void operator/=(float2 &result, int value) {
  result.x /= value;
  result.y /= value;
}

__INLINE__
float3 operator*(float value0, uchar3 value1) {
  float3 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;
  result.z = value0 * value1.z;

  return result;
}

__INLINE__
float3 operator*(float value0, float3 value1) {
  float3 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;
  result.z = value0 * value1.z;

  return result;
}

__INLINE__
float3 operator+(float3 &value0, float3 &value1) {
  float3 result;
  result.x = value0.x + value1.x;
  result.y = value0.y + value1.y;
  result.z = value0.z + value1.z;

  return result;
}

__INLINE__
void operator+=(float3 &result, uchar3 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
}

__INLINE__
void operator+=(float3 &result, float3 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
}

__INLINE__
void operator/=(float3 &result, int value) {
  result.x = result.x / value;
  result.y = result.y / value;
  result.z = result.z / value;
}

__INLINE__
void operator/=(float3 &result, float value) {
  result.x = result.x / value;
  result.y = result.y / value;
  result.z = result.z / value;
}

__INLINE__
float4 operator*(float value0, uchar4 value1) {
  float4 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;
  result.z = value0 * value1.z;
  result.w = value0 * value1.w;

  return result;
}

__INLINE__
float4 operator*(float value0, float4 value1) {
  float4 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;
  result.z = value0 * value1.z;
  result.w = value0 * value1.w;

  return result;
}

__INLINE__
float4 operator+(float4 &value0, float4 &value1) {
  float4 result;
  result.x = value0.x + value1.x;
  result.y = value0.y + value1.y;
  result.z = value0.z + value1.z;
  result.w = value0.w + value1.w;

  return result;
}

__INLINE__
void operator+=(float4 &result, uchar4 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
  result.w += value.w;
}

__INLINE__
void operator+=(float4 &result, float4 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
  result.w += value.w;
}

__INLINE__
void operator/=(float4 &result, int value) {
  result.x /= value;
  result.y /= value;
  result.z /= value;
  result.w /= value;
}

__INLINE__
void operator/=(float4 &result, float value) {
  result.x /= value;
  result.y /= value;
  result.z /= value;
  result.w /= value;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL3_CV_CUDA_UTILITY_HPP_
