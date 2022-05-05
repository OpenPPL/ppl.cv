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


#ifndef _ST_HPC_PPL_CV_CUDA_UTILITY_HPP_
#define _ST_HPC_PPL_CV_CUDA_UTILITY_HPP_

#include "cuda_runtime.h"

#include "ppl/common/log.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

#define DEVICE_INLINE
#if defined(DEVICE_INLINE)
# define __DEVICE__ __device__ __forceinline__
#else
# define __DEVICE__ inline
#endif

#define PPL_ASSERT(expression)                                                 \
if (!(expression)) {                                                           \
  LOG(ERROR) << "Assertion failed: " << #expression;                           \
  return ppl::common::RC_INVALID_VALUE;                                        \
}

typedef unsigned char uchar;
typedef signed char schar;
typedef unsigned short ushort;
typedef unsigned int uint;

// configuration of thread blocks.
enum DimX {
  kDimX0 = 16,
  kDimX1 = 32,
  kDimX2 = 32,
};

enum DimY {
  kDimY0 = 16,
  kDimY1 = 4,
  kDimY2 = 8,
};

enum ShiftX {
  kShiftX0 = 4,
  kShiftX1 = 5,
  kShiftX2 = 5,
};

enum ShiftY {
  kShiftY0 = 4,
  kShiftY1 = 2,
  kShiftY2 = 3,
};

enum BlockConfiguration0 {
  kBlockDimX0 = kDimX1,
  kBlockDimY0 = kDimY1,
  kBlockShiftX0 = kShiftX1,
  kBlockShiftY0 = kShiftY1,
};

enum BlockConfiguration1 {
  kBlockDimX1 = kDimX2,
  kBlockDimY1 = kDimY2,
  kBlockShiftX1 = kShiftX2,
  kBlockShiftY1 = kShiftY2,
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
  return (total + ((1 << shift) - 1)) >> shift;
}

__host__ __device__
inline
int divideUp1(int total, int shift) {
  return (total + (1 << (shift - 1))) >> shift;
}

__host__ __device__
inline
int roundUp(int total, int grain, int shift) {
  return ((total + grain - 1) >> shift) << shift;
}

template <typename T>
__DEVICE__
T max(const T& value0, const T& value1) {
  if (value0 < value1) {
    return value1;
  }
  else {
    return value0;
  }
}

template <typename T>
__DEVICE__
T min(const T& value0, const T& value1) {
  if (value0 > value1) {
    return value1;
  }
  else {
    return value0;
  }
}

template <typename T>
__DEVICE__
T max(const T& value0, const T& value1, const T& value2) {
  if (value0 < value1) {
    return value1 > value2 ? value1 : value2;
  }
  else {
    return value0 > value2 ? value0 : value2;
  }
}

template <typename T>
__DEVICE__
T min(const T& value0, const T& value1, const T& value2) {
  if (value0 > value1) {
    return value1 < value2 ? value1 : value2;
  }
  else {
    return value0 < value2 ? value0 : value2;
  }
}

template<typename T>
__DEVICE__
T clip(const T& value, const T& min_value, const T& max_value) {
    return min(max(value, min_value), max_value);
}

__DEVICE__
uchar saturateCast(int value) {
  unsigned int result = 0;
  asm("cvt.sat.u8.s32 %0, %1;" : "=r"(result) : "r"(value));
  return result;
}

__DEVICE__
uchar saturateCast(short value) {
  unsigned int result = 0;
  asm("cvt.sat.u8.s16 %0, %1;" : "=r"(result) : "h"(value));
  return result;
}

__DEVICE__
uchar saturateCast(float value) {
  unsigned int result = 0;
  asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(result) : "f"(value));
  return result;
}

__DEVICE__
short saturateCastI2S(int value) {
  short result = 0;
  asm("cvt.sat.s16.s32 %0, %1;" : "=h"(result) : "r"(value));
  return result;
}

__DEVICE__
short saturateCastF2S(float value) {
  short result = 0;
  asm("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(result) : "f"(value));
  return result;
}

template <typename T0, typename T1>
__DEVICE__
T0 saturateCastVector(T1 value);

template <>
__DEVICE__
uchar saturateCastVector(float4 value) {
  uchar result = saturateCast(value.x);

  return result;
}

template <>
__DEVICE__
uchar2 saturateCastVector(float2 value) {
  uchar2 result;
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);

  return result;
}

template <>
__DEVICE__
uchar3 saturateCastVector(short3 value) {
  uchar3 result;
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);

  return result;
}

template <>
__DEVICE__
uchar3 saturateCastVector(float3 value) {
  uchar3 result;
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);

  return result;
}

template <>
__DEVICE__
uchar3 saturateCastVector(float4 value) {
  uchar3 result;
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);

  return result;
}

template <>
__DEVICE__
uchar4 saturateCastVector(short4 value) {
  uchar4 result;
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);
  result.w = saturateCast(value.w);

  return result;
}

template <>
__DEVICE__
uchar3 saturateCastVector(int3 value) {
  uchar3 result;
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);

  return result;
}

template <>
__DEVICE__
uchar4 saturateCastVector(int4 value) {
  uchar4 result;
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);
  result.w = saturateCast(value.w);

  return result;
}

template <>
__DEVICE__
uchar4 saturateCastVector(float4 value) {
  uchar4 result;
  result.x = saturateCast(value.x);
  result.y = saturateCast(value.y);
  result.z = saturateCast(value.z);
  result.w = saturateCast(value.w);

  return result;
}

template <>
__DEVICE__
short3 saturateCastVector(float4 value) {
  short3 result;
  result.x = saturateCastF2S(value.x);
  result.y = saturateCastF2S(value.y);
  result.z = saturateCastF2S(value.z);

  return result;
}

template <>
__DEVICE__
short4 saturateCastVector(float4 value) {
  short4 result;
  result.x = saturateCastF2S(value.x);
  result.y = saturateCastF2S(value.y);
  result.z = saturateCastF2S(value.z);
  result.w = saturateCastF2S(value.w);

  return result;
}

template <>
__DEVICE__
float saturateCastVector(float4 value) {
  return value.x;
}

template <>
__DEVICE__
float2 saturateCastVector(float2 value) {
  return value;
}

template <>
__DEVICE__
float3 saturateCastVector(float3 value) {
  return value;
}

template <>
__DEVICE__
float3 saturateCastVector(float4 value) {
  float3 result;
  result.x = value.x;
  result.y = value.y;
  result.z = value.z;

  return result;
}

template <>
__DEVICE__
float4 saturateCastVector(float4 value) {
  return value;
}

__DEVICE__
float3 operator+(float3 &value0, float3 &value1) {
  float3 result;
  result.x = value0.x + value1.x;
  result.y = value0.y + value1.y;
  result.z = value0.z + value1.z;

  return result;
}

__DEVICE__
float4 operator+(float4 &value0, float4 &value1) {
  float4 result;
  result.x = value0.x + value1.x;
  result.y = value0.y + value1.y;
  result.z = value0.z + value1.z;
  result.w = value0.w + value1.w;

  return result;
}

__DEVICE__
float2 operator*(float value0, uchar2 value1) {
  float2 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;

  return result;
}

__DEVICE__
float3 operator*(float value0, uchar3 value1) {
  float3 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;
  result.z = value0 * value1.z;

  return result;
}

__DEVICE__
float3 operator*(float value0, float3 value1) {
  float3 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;
  result.z = value0 * value1.z;

  return result;
}

__DEVICE__
float4 operator*(float value0, uchar4 value1) {
  float4 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;
  result.z = value0 * value1.z;
  result.w = value0 * value1.w;

  return result;
}

__DEVICE__
float4 operator*(uchar3 value0, float value1) {
  float4 result;
  result.x = value0.x * value1;
  result.y = value0.y * value1;
  result.z = value0.z * value1;

  return result;
}

__DEVICE__
float4 operator*(uchar4 value0, float value1) {
  float4 result;
  result.x = value0.x * value1;
  result.y = value0.y * value1;
  result.z = value0.z * value1;
  result.w = value0.w * value1;

  return result;
}

__DEVICE__
float4 operator*(float3 value0, float value1) {
  float4 result;
  result.x = value0.x * value1;
  result.y = value0.y * value1;
  result.z = value0.z * value1;

  return result;
}

__DEVICE__
float4 operator*(float4 value0, float value1) {
  float4 result;
  result.x = value0.x * value1;
  result.y = value0.y * value1;
  result.z = value0.z * value1;
  result.w = value0.w * value1;

  return result;
}

__DEVICE__
float4 operator*(float value0, float4 value1) {
  float4 result;
  result.x = value0 * value1.x;
  result.y = value0 * value1.y;
  result.z = value0 * value1.z;
  result.w = value0 * value1.w;

  return result;
}

__DEVICE__
void operator+=(uint3 &result, uchar3 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
}

__DEVICE__
void operator+=(uint3 &result, uint3 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
}

__DEVICE__
void operator+=(uint4 &result, uchar4 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
  result.w += value.w;
}

__DEVICE__
void operator+=(uint4 &result, uint4 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
  result.w += value.w;
}

__DEVICE__
void operator+=(float2 &result, uchar2 &value) {
  result.x += value.x;
  result.y += value.y;
}

__DEVICE__
void operator+=(float2 &result, float2 &value) {
  result.x += value.x;
  result.y += value.y;
}

__DEVICE__
void operator+=(float3 &result, uchar3 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
}

__DEVICE__
void operator+=(float3 &result, float3 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
}

__DEVICE__
void operator+=(float4 &result, uchar value) {
  result.x += value;
}

__DEVICE__
void operator+=(float4 &result, uchar3 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
}

__DEVICE__
void operator+=(float4 &result, uchar4 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
  result.w += value.w;
}

__DEVICE__
void operator+=(float4 &result, float value) {
  result.x += value;
}

__DEVICE__
void operator+=(float4 &result, float3 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
}

__DEVICE__
void operator+=(float4 &result, float4 &value) {
  result.x += value.x;
  result.y += value.y;
  result.z += value.z;
  result.w += value.w;
}

__DEVICE__
void operator-=(float3 &result, uchar3 &value) {
  result.x -= value.x;
  result.y -= value.y;
  result.z -= value.z;
}

__DEVICE__
void operator-=(float3 &result, float3 &value) {
  result.x -= value.x;
  result.y -= value.y;
  result.z -= value.z;
}

__DEVICE__
void operator-=(float4 &result, uchar4 &value) {
  result.x -= value.x;
  result.y -= value.y;
  result.z -= value.z;
  result.w -= value.w;
}

__DEVICE__
void operator-=(float4 &result, float4 &value) {
  result.x -= value.x;
  result.y -= value.y;
  result.z -= value.z;
  result.w -= value.w;
}

__DEVICE__
void operator/=(float2 &result, int value) {
  result.x /= value;
  result.y /= value;
}

__DEVICE__
void operator/=(float3 &result, int value) {
  result.x = result.x / value;
  result.y = result.y / value;
  result.z = result.z / value;
}

__DEVICE__
void operator/=(float3 &result, float value) {
  result.x = result.x / value;
  result.y = result.y / value;
  result.z = result.z / value;
}

__DEVICE__
void operator/=(float4 &result, int value) {
  result.x /= value;
  result.y /= value;
  result.z /= value;
  result.w /= value;
}

__DEVICE__
void operator/=(float4 &result, float value) {
  result.x /= value;
  result.y /= value;
  result.z /= value;
  result.w /= value;
}

__DEVICE__
void operator/=(float4 &result, float4 value) {
  result.x /= value.x;
  result.y /= value.y;
  result.z /= value.z;
  result.w /= value.w;
}

__DEVICE__
void assignVector(float3 &result, uchar3 &value) {
  result.x = value.x;
  result.y = value.y;
  result.z = value.z;
}

__DEVICE__
void assignVector(float3 &result, float3 &value) {
  result.x = value.x;
  result.y = value.y;
  result.z = value.z;
}

__DEVICE__
void assignVector(float4 &result, uchar4 &value) {
  result.x = value.x;
  result.y = value.y;
  result.z = value.z;
  result.w = value.w;
}

__DEVICE__
void assignVector(float4 &result, float4 &value) {
  result.x = value.x;
  result.y = value.y;
  result.z = value.z;
  result.w = value.w;
}

__DEVICE__
void mulAdd(float &result, uchar &value0, float value1) {
  result += value0 * value1;
}

__DEVICE__
void mulAdd(float &result, short &value0, float value1) {
  result += value0 * value1;
}

__DEVICE__
void mulAdd(float &result, float &value0, float value1) {
  result += value0 * value1;
}

__DEVICE__
void mulAdd(float3 &result, float3 &value0, float3 &value1) {
  result.x += value0.x * value1.x;
  result.y += value0.y * value1.y;
  result.z += value0.z * value1.z;
}

__DEVICE__
void mulAdd(float4 &result, float4 &value0, float4 &value1) {
  result.x += value0.x * value1.x;
  result.y += value0.y * value1.y;
  result.z += value0.z * value1.z;
  result.w += value0.w * value1.w;
}

__DEVICE__
void mulAdd(float4 &result, uchar &value0, float value1) {
  result.x += value0 * value1;
}

__DEVICE__
void mulAdd(float4 &result, uchar3 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
}

__DEVICE__
void mulAdd(float4 &result, uchar4 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
  result.w += value0.w * value1;
}

__DEVICE__
void mulAdd(float4 &result, short3 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
}

__DEVICE__
void mulAdd(float4 &result, short4 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
  result.w += value0.w * value1;
}

__DEVICE__
void mulAdd(float4 &result, float &value0, float value1) {
  result.x += value0 * value1;
}

__DEVICE__
void mulAdd(float4 &result, float3 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
}

__DEVICE__
void mulAdd(float4 &result, float4 &value0, float value1) {
  result.x += value0.x * value1;
  result.y += value0.y * value1;
  result.z += value0.z * value1;
  result.w += value0.w * value1;
}

template <typename T>
__DEVICE__
T transform(float4 &src);

template <>
__DEVICE__
float3 transform(float4 &src) {
  float3 dst;
  dst.x = src.x;
  dst.y = src.y;
  dst.z = src.z;

  return dst;
}

template <>
__DEVICE__
float4 transform(float4 &src) {
  return src;
}

struct ConstantBorder {
  __DEVICE__
  int operator()(int range, int radius, int index) {
    if (index < 0) {
      return -1;
    }
    else if (index < range) {
      return index;
    }
    else {
      return -1;
    }
  }
};

struct ReplicateBorder {
  __DEVICE__
  int operator()(int range, int radius, int index) {
    if (index < 0) {
      return 0;
    }
    else if (index < range) {
      return index;
    }
    else {
      return range - 1;
    }
  }
};

struct ReflectBorder {
  __DEVICE__
  int operator()(int range, int radius, int index) {
    if (range >= radius) {
      if (index < 0) {
        return -1 - index;
      }
      else if (index < range) {
        return index;
      }
      else {
        return (range << 1) - index - 1;
      }
    }
    else {
      if (index >= 0 && index < range) {
        return index;
      }
      else {
        if (range == 1) {
          index = 0;
        }
        else {
          do {
            if (index < 0)
              index = -1 - index;
            else
              index = (range << 1) - index - 1;
          } while (index >= range || index < 0);
        }

        return index;
      }
    }
  }
};

struct WarpBorder {
  __DEVICE__
  int operator()(int range, int radius, int index) {
    if (range >= radius) {
      if (index < 0) {
        return index + range;
      }
      else if (index < range) {
        return index;
      }
      else {
        return index - range;
      }
    }
    else {
      if (index >= 0 && index < range) {
        return index;
      }
      else {
        if (range == 1) {
          index = 0;
        }
        else {
          do {
            if (index < 0)
              index += range;
            else
              index -= range;
          } while (index >= range || index < 0);
        }

        return index;
      }
    }
  }
};

struct Reflect101Border {
  __DEVICE__
  int operator()(int range, int radius, int index) {
    if (range >= radius) {
      if (index < 0) {
        return 0 - index;
      }
      else if (index < range) {
        return index;
      }
      else {
        return (range << 1) - index - 2;
      }
    }
    else {
      if (index >= 0 && index < range) {
        return index;
      }
      else {
        if (range == 1) {
          index = 0;
        }
        else {
          do {
            if (index < 0)
              index = 0 - index;
            else
              index = (range << 1) - index - 2;
          } while (index >= range || index < 0);
        }

        return index;
      }
    }
  }
};

typedef struct Reflect101Border DefaultBorder;

template <typename T>
inline bool checkOverlapping(T* address0, size_t pitch0, size_t rows0,
                             T* address1, size_t pitch1, size_t rows1) {
  T* end0 = (T*)((uchar*)address0 + pitch0 * rows0);
  T* end1 = (T*)((uchar*)address1 + pitch1 * rows1);
  T* temp;

  if (address0 > address1) {
    temp = address0;
    address0 = address1;
    address1 = temp;

    temp = end0;
    end0 = end1;
    end1 = temp;
  }

  if (address0 == address1) {
    return true;
  }

  if (address0 < address1) {
    if (address1 < end0) {
      return true;
    }
  }
  else {
    return false;
  }
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_UTILITY_HPP_
