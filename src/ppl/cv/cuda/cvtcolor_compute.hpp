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

#ifndef _ST_HPC_PPL_CV_CUDA_CVTCOLOR_COMPUTE_HPP_
#define _ST_HPC_PPL_CV_CUDA_CVTCOLOR_COMPUTE_HPP_

#include <cfloat>

#include "cuda_runtime.h"

#include "utility/utility.hpp"

namespace ppl {
namespace cv {
namespace cuda {

#define DEVICE_INLINE
#if defined(DEVICE_INLINE)
# define __DEVICE__ __device__ __forceinline__
#else
# define __DEVICE__ __device__
#endif

// BGR/RGB/BGRA/RGBA <-> YCrCb
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

// BGR/RGB -> NV12/NV21
#define NVXX1_YR 269484
#define NVXX1_YG 528482
#define NVXX1_YB 102760
#define NVXX1_VR 460324
#define NVXX1_VG -385875
#define NVXX1_VB -74448
#define NVXX1_UR -155188
#define NVXX1_UG -305135
#define NVXX1_UB 460324

// NV12/NV21-> BGR/RGB
#define NVXX1_CY 1220542
#define NVXX1_CUB 2116026
#define NVXX1_CUG -409993
#define NVXX1_CVG -852492
#define NVXX1_CVR 1673527
#define NVXX1_SHIFT 20

/******************* BGR(RBB) <-> BGRA(RGBA) ******************/

struct BGR2BGRACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    uchar4 dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float4 dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
    dst.w = 1.0f;

    return dst;
  }
};

struct RGB2RGBACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    uchar4 dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float4 dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
    dst.w = 1.0f;

    return dst;
  }
};

struct BGRA2BGRCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    uchar3 dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float3 dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;

    return dst;
  }
};

struct RGBA2RGBCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    uchar3 dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float3 dst;
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;

    return dst;
  }
};

struct BGR2RGBACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    uchar4 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float4 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;
    dst.w = 1.0f;

    return dst;
  }
};

struct RGB2BGRACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    uchar4 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float4 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;
    dst.w = 1.0f;

    return dst;
  }
};

struct RGBA2BGRCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    uchar3 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float3 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;

    return dst;
  }
};

struct BGRA2RGBCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    uchar3 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float3 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;

    return dst;
  }
};

/******************* BGR <-> RGB ******************/

struct BGR2RGBCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    uchar3 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float3 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;

    return dst;
  }
};

struct RGB2BGRCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    uchar3 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float3 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;

    return dst;
  }
};

/******************* BGRA <-> RGBA ******************/

struct BGRA2RGBACompute {
  __DEVICE__
  uchar4 operator()(const uchar4& src) {
    uchar4 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;
    dst.w = src.w;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float4& src) {
    float4 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;
    dst.w = src.w;

    return dst;
  }
};

struct RGBA2BGRACompute {
  __DEVICE__
  uchar4 operator()(const uchar4& src) {
    uchar4 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;
    dst.w = src.w;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float4& src) {
    float4 dst;
    dst.x = src.z;
    dst.y = src.y;
    dst.z = src.x;
    dst.w = src.w;

    return dst;
  }
};

/******************* BGR/RGB/BGRA/RGBA <-> Gray ******************/

enum Bgr2GrayCoefficients {
  kB2Y15    = 3735,
  kG2Y15    = 19235,
  kR2Y15    = 9798,
  kRgbShift = 15,
};

struct BGR2GRAYCompute {
  __DEVICE__
  unsigned char operator()(const uchar3& src) {
    int b = src.x;
    int g = src.y;
    int r = src.z;
    unsigned char dst = divideUp(b * kB2Y15 + g * kG2Y15 + r * kR2Y15,
                                 kRgbShift);

    return dst;
  }

  __DEVICE__
  float operator()(const float3& src) {
    float b = src.x;
    float g = src.y;
    float r = src.z;
    float dst = b * 0.114f + g * 0.587f + r * 0.299f;

    return dst;
  }
};

struct RGB2GRAYCompute {
  __DEVICE__
  unsigned char operator()(const uchar3& src) {
    int r = src.x;
    int g = src.y;
    int b = src.z;
    unsigned char dst = divideUp(r * kR2Y15 + g * kG2Y15 + b * kB2Y15,
                                 kRgbShift);

    return dst;
  }

  __DEVICE__
  float operator()(const float3& src) {
    float r = src.x;
    float g = src.y;
    float b = src.z;
    float dst = r * 0.299f + g * 0.587f + b * 0.114f;

    return dst;
  }
};

struct BGRA2GRAYCompute {
  __DEVICE__
  unsigned char operator()(const uchar4& src) {
    int b = src.x;
    int g = src.y;
    int r = src.z;
    unsigned char dst = divideUp(b * kB2Y15 + g * kG2Y15 + r * kR2Y15,
                                 kRgbShift);

    return dst;
  }

  __DEVICE__
  float operator()(const float4& src) {
    float b = src.x;
    float g = src.y;
    float r = src.z;
    float dst = b * 0.114f + g * 0.587f + r * 0.299f;

    return dst;
  }
};

struct RGBA2GRAYCompute {
  __DEVICE__
  unsigned char operator()(const uchar4& src) {
    int r = src.x;
    int g = src.y;
    int b = src.z;
    unsigned char dst = divideUp(r * kR2Y15 + g * kG2Y15 + b * kB2Y15,
                                 kRgbShift);

    return dst;
  }

  __DEVICE__
  float operator()(const float4& src) {
    float r = src.x;
    float g = src.y;
    float b = src.z;
    float dst = r * 0.299f + g * 0.587f + b * 0.114f;

    return dst;
  }
};

struct GRAY2BGRCompute {
  __DEVICE__
  uchar3 operator()(const unsigned char& src) {
    uchar3 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float& src) {
    float3 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;

    return dst;
  }
};

struct GRAY2RGBCompute {
  __DEVICE__
  uchar3 operator()(const unsigned char& src) {
    uchar3 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float& src) {
    float3 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;

    return dst;
  }
};

struct GRAY2BGRACompute {
  __DEVICE__
  uchar4 operator()(const unsigned char& src) {
    uchar4 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float& src) {
    float4 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;
    dst.w = 1.0f;

    return dst;
  }
};

struct GRAY2RGBACompute {
  __DEVICE__
  uchar4 operator()(const unsigned char& src) {
    uchar4 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float& src) {
    float4 dst;
    dst.x = src;
    dst.y = src;
    dst.z = src;
    dst.w = 1.0f;

    return dst;
  }
};

/******************* BGR/RGB/BGRA/RGBA <-> YCrCb ******************/

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

struct BGR2YCrCbCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    unsigned char b = src.x;
    unsigned char g = src.y;
    unsigned char r = src.z;

    int x = divideUp(r * kR2YCoeff + g * kG2YCoeff + b * kB2YCoeff,
                     kYCrCbShift);
    int y = divideUp((r - x) * kCRCoeff + kShift14Delta, kYCrCbShift);
    int z = divideUp((b - x) * kCBCoeff + kShift14Delta, kYCrCbShift);

    uchar3 dst;  // (x = Y, y = Cr, z = Cb)
    dst.x = saturateCast(x);
    dst.y = saturateCast(y);
    dst.z = saturateCast(z);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float b = src.x;
    float g = src.y;
    float r = src.z;

    float3 dst;  // (x = Y, y = Cr, z = Cb)
    dst.x = r * R2Y_FLOAT_COEFF + g * G2Y_FLOAT_COEFF + b * B2Y_FLOAT_COEFF;
    dst.y = (r - dst.x) * CR_FLOAT_COEFF + YCRCB_FLOAT_DELTA;
    dst.z = (b - dst.x) * CB_FLOAT_COEFF + YCRCB_FLOAT_DELTA;

    return dst;
  }
};

struct RGB2YCrCbCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    unsigned char r = src.x;
    unsigned char g = src.y;
    unsigned char b = src.z;

    int x = divideUp(r * kR2YCoeff + g * kG2YCoeff + b * kB2YCoeff,
                     kYCrCbShift);
    int y = divideUp((r - x) * kCRCoeff + kShift14Delta, kYCrCbShift);
    int z = divideUp((b - x) * kCBCoeff + kShift14Delta, kYCrCbShift);

    uchar3 dst;  // (x = Y, y = Cr, z = Cb)
    dst.x = saturateCast(x);
    dst.y = saturateCast(y);
    dst.z = saturateCast(z);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float r = src.x;
    float g = src.y;
    float b = src.z;

    float3 dst;  // (x = Y, y = Cr, z = Cb)
    dst.x = r * R2Y_FLOAT_COEFF + g * G2Y_FLOAT_COEFF + b * B2Y_FLOAT_COEFF;
    dst.y = (r - dst.x) * CR_FLOAT_COEFF + YCRCB_FLOAT_DELTA;
    dst.z = (b - dst.x) * CB_FLOAT_COEFF + YCRCB_FLOAT_DELTA;

    return dst;
  }
};

struct BGRA2YCrCbCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    unsigned char b = src.x;
    unsigned char g = src.y;
    unsigned char r = src.z;

    int x = divideUp(r * kR2YCoeff + g * kG2YCoeff + b * kB2YCoeff,
                     kYCrCbShift);
    int y = divideUp((r - x) * kCRCoeff + kShift14Delta, kYCrCbShift);
    int z = divideUp((b - x) * kCBCoeff + kShift14Delta, kYCrCbShift);

    uchar3 dst;  // (x = Y, y = Cr, z = Cb)
    dst.x = saturateCast(x);
    dst.y = saturateCast(y);
    dst.z = saturateCast(z);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float b = src.x;
    float g = src.y;
    float r = src.z;

    float3 dst;  // (x = Y, y = Cr, z = Cb)
    dst.x = r * R2Y_FLOAT_COEFF + g * G2Y_FLOAT_COEFF + b * B2Y_FLOAT_COEFF;
    dst.y = (r - dst.x) * CR_FLOAT_COEFF + YCRCB_FLOAT_DELTA;
    dst.z = (b - dst.x) * CB_FLOAT_COEFF + YCRCB_FLOAT_DELTA;

    return dst;
  }
};

struct RGBA2YCrCbCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    unsigned char r = src.x;
    unsigned char g = src.y;
    unsigned char b = src.z;

    int x = divideUp(r * kR2YCoeff + g * kG2YCoeff + b * kB2YCoeff,
                     kYCrCbShift);
    int y = divideUp((r - x) * kCRCoeff + kShift14Delta, kYCrCbShift);
    int z = divideUp((b - x) * kCBCoeff + kShift14Delta, kYCrCbShift);

    uchar3 dst;  // (x = Y, y = Cr, z = Cb)
    dst.x = saturateCast(x);
    dst.y = saturateCast(y);
    dst.z = saturateCast(z);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float r = src.x;
    float g = src.y;
    float b = src.z;

    float3 dst;  // (x = Y, y = Cr, z = Cb)
    dst.x = r * R2Y_FLOAT_COEFF + g * G2Y_FLOAT_COEFF + b * B2Y_FLOAT_COEFF;
    dst.y = (r - dst.x) * CR_FLOAT_COEFF + YCRCB_FLOAT_DELTA;
    dst.z = (b - dst.x) * CB_FLOAT_COEFF + YCRCB_FLOAT_DELTA;

    return dst;
  }
};

struct YCrCb2BGRCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    int y  = src.x;
    int cr = src.y - YCRCB_UCHAR_DELTA;
    int cb = src.z - YCRCB_UCHAR_DELTA;

    int b = y + divideUp(cb * kCb2BCoeff, kYCrCbShift);
    int g = y + divideUp(cr * kY2GCrCoeff + cb * kY2GCbCoeff,
                         kYCrCbShift);
    int r = y + divideUp(cr * kCr2RCoeff, kYCrCbShift);

    b = b < 0 ? 0 : b;
    b = b > 255 ? 255 : b;
    g = g < 0 ? 0 : g;
    g = g > 255 ? 255 : g;
    r = r < 0 ? 0 : r;
    r = r > 255 ? 255 : r;

    uchar3 dst;  // (x = B, y = G, z = R)
    dst.x = b;
    dst.y = g;
    dst.z = r;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float y  = src.x;
    float cr = src.y - YCRCB_FLOAT_DELTA;
    float cb = src.z - YCRCB_FLOAT_DELTA;

    float3 dst;  // (x = B, y = G, z = R)
    dst.x = y + cb * CB2R_FLOAT_COEFF;
    dst.y = y + cr * Y2G_CR_FLOAT_COEFF + cb * Y2G_CB_FLOAT_COEFF;
    dst.z = y + cr * CR2R_FLOAT_COEFF;

    return dst;
  }
};

struct YCrCb2RGBCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    int y  = src.x;
    int cr = src.y - YCRCB_UCHAR_DELTA;
    int cb = src.z - YCRCB_UCHAR_DELTA;

    int b = y + divideUp(cb * kCb2BCoeff, kYCrCbShift);
    int g = y + divideUp(cr * kY2GCrCoeff + cb * kY2GCbCoeff, kYCrCbShift);
    int r = y + divideUp(cr * kCr2RCoeff, kYCrCbShift);

    b = b < 0 ? 0 : b;
    b = b > 255 ? 255 : b;
    g = g < 0 ? 0 : g;
    g = g > 255 ? 255 : g;
    r = r < 0 ? 0 : r;
    r = r > 255 ? 255 : r;

    uchar3 dst;  // (x = R, y = G, z = B)
    dst.x = r;
    dst.y = g;
    dst.z = b;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float y  = src.x;
    float cr = src.y - YCRCB_FLOAT_DELTA;
    float cb = src.z - YCRCB_FLOAT_DELTA;

    float3 dst;  // (x = R, y = G, z = B)
    dst.x = y + cr * CR2R_FLOAT_COEFF;
    dst.y = y + cr * Y2G_CR_FLOAT_COEFF + cb * Y2G_CB_FLOAT_COEFF;
    dst.z = y + cb * CB2R_FLOAT_COEFF;

    return dst;
  }
};

struct YCrCb2BGRACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    int y  = src.x;
    int cr = src.y - YCRCB_UCHAR_DELTA;
    int cb = src.z - YCRCB_UCHAR_DELTA;

    int b = y + divideUp(cb * kCb2BCoeff, kYCrCbShift);
    int g = y + divideUp(cr * kY2GCrCoeff + cb * kY2GCbCoeff, kYCrCbShift);
    int r = y + divideUp(cr * kCr2RCoeff, kYCrCbShift);

    b = b < 0 ? 0 : b;
    b = b > 255 ? 255 : b;
    g = g < 0 ? 0 : g;
    g = g > 255 ? 255 : g;
    r = r < 0 ? 0 : r;
    r = r > 255 ? 255 : r;

    uchar4 dst;  // (x = B, y = G, z = R, w = Alpha)
    dst.x = b;
    dst.y = g;
    dst.z = r;
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float y  = src.x;
    float cr = src.y - YCRCB_FLOAT_DELTA;
    float cb = src.z - YCRCB_FLOAT_DELTA;

    float4 dst;  // (x = B, y = G, z = R, w = Alpha)
    dst.x = y + cb * CB2R_FLOAT_COEFF;
    dst.y = y + cr * Y2G_CR_FLOAT_COEFF + cb * Y2G_CB_FLOAT_COEFF;
    dst.z = y + cr * CR2R_FLOAT_COEFF;
    dst.w = 1.0f;

    return dst;
  }
};

struct YCrCb2RGBACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    int y  = src.x;
    int cr = src.y - YCRCB_UCHAR_DELTA;
    int cb = src.z - YCRCB_UCHAR_DELTA;

    int b = y + divideUp(cb * kCb2BCoeff, kYCrCbShift);
    int g = y + divideUp(cr * kY2GCrCoeff + cb * kY2GCbCoeff, kYCrCbShift);
    int r = y + divideUp(cr * kCr2RCoeff, kYCrCbShift);

    b = b < 0 ? 0 : b;
    b = b > 255 ? 255 : b;
    g = g < 0 ? 0 : g;
    g = g > 255 ? 255 : g;
    r = r < 0 ? 0 : r;
    r = r > 255 ? 255 : r;

    uchar4 dst;  // (x = R, y = G, z = B, w = Alpha)
    dst.x = r;
    dst.y = g;
    dst.z = b;
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float y  = src.x;
    float cr = src.y - YCRCB_FLOAT_DELTA;
    float cb = src.z - YCRCB_FLOAT_DELTA;

    float4 dst;  // (x = R, y = G, z = B, w = Alpha)
    dst.x = y + cr * CR2R_FLOAT_COEFF;
    dst.y = y + cr * Y2G_CR_FLOAT_COEFF + cb * Y2G_CB_FLOAT_COEFF;
    dst.z = y + cb * CB2R_FLOAT_COEFF;
    dst.w = 1.0f;

    return dst;
  }
};

/******************* BGR/RGB/BGRA/RGBA <-> HSV ******************/

enum HsvShifts {
  kHSVSift = 12,
};

__constant__ int c_HsvDivTable[256] = {0, 1044480, 522240, 348160, 261120, 208896, 174080, 149211, 130560, 116053, 104448, 94953, 87040, 80345, 74606, 69632, 65280, 61440, 58027, 54973, 52224, 49737, 47476, 45412, 43520, 41779, 40172, 38684, 37303, 36017, 34816, 33693, 32640, 31651, 30720, 29842, 29013, 28229, 27486, 26782, 26112, 25475, 24869, 24290, 23738, 23211, 22706, 22223, 21760, 21316, 20890, 20480, 20086, 19707, 19342, 18991, 18651, 18324, 18008, 17703, 17408, 17123, 16846, 16579, 16320, 16069, 15825, 15589, 15360, 15137, 14921, 14711, 14507, 14308, 14115, 13926, 13743, 13565, 13391, 13221, 13056, 12895, 12738, 12584, 12434, 12288, 12145, 12006, 11869, 11736, 11605, 11478, 11353, 11231, 11111, 10995, 10880, 10768, 10658, 10550, 10445, 10341, 10240, 10141, 10043, 9947, 9854, 9761, 9671, 9582, 9495, 9410, 9326, 9243, 9162, 9082, 9004, 8927, 8852, 8777, 8704, 8632, 8561, 8492, 8423, 8356, 8290, 8224, 8160, 8097, 8034, 7973, 7913, 7853, 7795, 7737, 7680, 7624, 7569, 7514, 7461, 7408, 7355, 7304, 7253, 7203, 7154, 7105, 7057, 7010, 6963, 6917, 6872, 6827, 6782, 6739, 6695, 6653, 6611, 6569, 6528, 6487, 6447, 6408, 6369, 6330, 6292, 6254, 6217, 6180, 6144, 6108, 6073, 6037, 6003, 5968, 5935, 5901, 5868, 5835, 5803, 5771, 5739, 5708, 5677, 5646, 5615, 5585, 5556, 5526, 5497, 5468, 5440, 5412, 5384, 5356, 5329, 5302, 5275, 5249, 5222, 5196, 5171, 5145, 5120, 5095, 5070, 5046, 5022, 4998, 4974, 4950, 4927, 4904, 4881, 4858, 4836, 4813, 4791, 4769, 4748, 4726, 4705, 4684, 4663, 4642, 4622, 4601, 4581, 4561, 4541, 4522, 4502, 4483, 4464, 4445, 4426, 4407, 4389, 4370, 4352, 4334, 4316, 4298, 4281, 4263, 4246, 4229, 4212, 4195, 4178, 4161, 4145, 4128, 4112, 4096};
__constant__ int c_HsvDivTable180[256] = {0, 122880, 61440, 40960, 30720, 24576, 20480, 17554, 15360, 13653, 12288, 11171, 10240, 9452, 8777, 8192, 7680, 7228, 6827, 6467, 6144, 5851, 5585, 5343, 5120, 4915, 4726, 4551, 4389, 4237, 4096, 3964, 3840, 3724, 3614, 3511, 3413, 3321, 3234, 3151, 3072, 2997, 2926, 2858, 2793, 2731, 2671, 2614, 2560, 2508, 2458, 2409, 2363, 2318, 2276, 2234, 2194, 2156, 2119, 2083, 2048, 2014, 1982, 1950, 1920, 1890, 1862, 1834, 1807, 1781, 1755, 1731, 1707, 1683, 1661, 1638, 1617, 1596, 1575, 1555, 1536, 1517, 1499, 1480, 1463, 1446, 1429, 1412, 1396, 1381, 1365, 1350, 1336, 1321, 1307, 1293, 1280, 1267, 1254, 1241, 1229, 1217, 1205, 1193, 1182, 1170, 1159, 1148, 1138, 1127, 1117, 1107, 1097, 1087, 1078, 1069, 1059, 1050, 1041, 1033, 1024, 1016, 1007, 999, 991, 983, 975, 968, 960, 953, 945, 938, 931, 924, 917, 910, 904, 897, 890, 884, 878, 871, 865, 859, 853, 847, 842, 836, 830, 825, 819, 814, 808, 803, 798, 793, 788, 783, 778, 773, 768, 763, 759, 754, 749, 745, 740, 736, 731, 727, 723, 719, 714, 710, 706, 702, 698, 694, 690, 686, 683, 679, 675, 671, 668, 664, 661, 657, 654, 650, 647, 643, 640, 637, 633, 630, 627, 624, 621, 617, 614, 611, 608, 605, 602, 599, 597, 594, 591, 588, 585, 582, 580, 577, 574, 572, 569, 566, 564, 561, 559, 556, 554, 551, 549, 546, 544, 541, 539, 537, 534, 532, 530, 527, 525, 523, 521, 518, 516, 514, 512, 510, 508, 506, 504, 502, 500, 497, 495, 493, 492, 490, 488, 486, 484, 482};
__constant__ int c_HsvSectorData[6][3] = {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1},
                                          {0,1,3}, {2,1,0}};

struct BGR2HSVCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    int b = src.x;
    int g = src.y;
    int r = src.z;
    int h, s, v, diff;
    int vr, vg;

    v = max(r, g, b);
    diff = v - min(r, g, b);
    vr = (v == r) * -1;
    vg = (v == g) * -1;
    s = (diff * c_HsvDivTable[v] + (1 << (kHSVSift - 1))) >> kHSVSift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) +
                          ((~vg) & (r - g + 4 * diff))));
    h = (h * c_HsvDivTable180[diff] + (1 << (kHSVSift-1))) >> kHSVSift;
    h += (h < 0) * 180;

    if (h > 359) {
      h -= 360;
    }
    if (h < 0) {
      h += 360;
    }

    uchar3 dst;
    dst.x = h;
    dst.y = s;
    dst.z = v;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float b = src.x;
    float g = src.y;
    float r = src.z;
    float h, s, v, diff;

    v = max(r, g, b);
    diff = v - min(r, g, b);
    s = diff / (float)(v + FLT_EPSILON);

    diff = (float)(60.0f / (diff + FLT_EPSILON));
    if (v == r) {
      h = (g - b) * diff;
    }
    if (v == g) {
      h = (b - r) * diff + 120.0f;
    }
    if (v == b) {
      h = (r - g) * diff + 240.0f;
    }
    if (h < 0.0f) {
      h += 360.0f;
    }

    float3 dst;
    dst.x = h;
    dst.y = s;
    dst.z = v;

    return dst;
  }
};

struct RGB2HSVCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    int r = src.x;
    int g = src.y;
    int b = src.z;
    int h, s, v, diff;
    int vr, vg;

    v = max(r, g, b);
    diff = v - min(r, g, b);
    vr = (v == r) * -1;
    vg = (v == g) * -1;
    s = (diff * c_HsvDivTable[v] + (1 << (kHSVSift - 1))) >> kHSVSift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) +
                          ((~vg) & (r - g + 4 * diff))));
    h = (h * c_HsvDivTable180[diff] + (1 << (kHSVSift-1))) >> kHSVSift;
    h += (h < 0) * 180;

    if (h > 359) {
      h -= 360;
    }
    if (h < 0) {
      h += 360;
    }

    uchar3 dst;
    dst.x = h;
    dst.y = s;
    dst.z = v;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float r = src.x;
    float g = src.y;
    float b = src.z;
    float h, s, v, diff;

    v = max(r, g, b);
    diff = v - min(r, g, b);
    s = diff / (float)(v + FLT_EPSILON);

    diff = (float)(60.0f / (diff + FLT_EPSILON));
    if (v == r) {
      h = (g - b) * diff;
    }
    if (v == g) {
      h = (b - r) * diff + 120.0f;
    }
    if (v == b) {
      h = (r - g) * diff + 240.0f;
    }
    if (h < 0.0f) {
      h += 360.0f;
    }

    float3 dst;
    dst.x = h;
    dst.y = s;
    dst.z = v;

    return dst;
  }
};

struct BGRA2HSVCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    int b = src.x;
    int g = src.y;
    int r = src.z;
    int h, s, v, diff;
    int vr, vg;

    v = max(r, g, b);
    diff = v - min(r, g, b);
    vr = (v == r) * -1;
    vg = (v == g) * -1;
    s = (diff * c_HsvDivTable[v] + (1 << (kHSVSift - 1))) >> kHSVSift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) +
                          ((~vg) & (r - g + 4 * diff))));
    h = (h * c_HsvDivTable180[diff] + (1 << (kHSVSift-1))) >> kHSVSift;
    h += (h < 0) * 180;

    if (h > 359) {
      h -= 360;
    }
    if (h < 0) {
      h += 360;
    }

    uchar3 dst;
    dst.x = h;
    dst.y = s;
    dst.z = v;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float b = src.x;
    float g = src.y;
    float r = src.z;
    float h, s, v, diff;

    v = max(r, g, b);
    diff = v - min(r, g, b);
    s = diff / (float)(v + FLT_EPSILON);

    diff = (float)(60.0f / (diff + FLT_EPSILON));
    if (v == r) {
      h = (g - b) * diff;
    }
    if (v == g) {
      h = (b - r) * diff + 120.0f;
    }
    if (v == b) {
      h = (r - g) * diff + 240.0f;
    }
    if (h < 0.0f) {
      h += 360.0f;
    }

    float3 dst;
    dst.x = h;
    dst.y = s;
    dst.z = v;

    return dst;
  }
};

struct RGBA2HSVCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    int r = src.x;
    int g = src.y;
    int b = src.z;
    int h, s, v, diff;
    int vr, vg;

    v = max(r, g, b);
    diff = v - min(r, g, b);
    vr = (v == r) * -1;
    vg = (v == g) * -1;
    s = (diff * c_HsvDivTable[v] + (1 << (kHSVSift - 1))) >> kHSVSift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) +
                          ((~vg) & (r - g + 4 * diff))));
    h = (h * c_HsvDivTable180[diff] + (1 << (kHSVSift-1))) >> kHSVSift;
    h += (h < 0) * 180;

    if (h > 359) {
      h -= 360;
    }
    if (h < 0) {
      h += 360;
    }

    uchar3 dst;
    dst.x = h;
    dst.y = s;
    dst.z = v;

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float r = src.x;
    float g = src.y;
    float b = src.z;
    float h, s, v, diff;

    v = max(r, g, b);
    diff = v - min(r, g, b);
    s = diff / (float)(v + FLT_EPSILON);

    diff = (float)(60.0f / (diff + FLT_EPSILON));
    if (v == r) {
      h = (g - b) * diff;
    }
    if (v == g) {
      h = (b - r) * diff + 120.0f;
    }
    if (v == b) {
      h = (r - g) * diff + 240.0f;
    }
    if (h < 0.0f) {
      h += 360.0f;
    }

    float3 dst;
    dst.x = h;
    dst.y = s;
    dst.z = v;

    return dst;
  }
};

struct HSV2BGRCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    const float hscale = 6.f / 180.f;

    float h = src.x;
    float s = src.y * (1.f / 255.f);
    float v = src.z * (1.f / 255.f);
    float b = v, g = v, r = v;

    if (s != 0) {
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

      int sector = __float2int_rd(h);
      h -= sector;

      if ((unsigned)sector >= 6u) {
        sector = 0;
        h = 0.f;
      }

      float tab[4];
      tab[0] = v;
      tab[1] = v * (1.f - s);
      tab[2] = v * (1.f - s * h);
      tab[3] = v * (1.f - s * (1.f - h));

      b = tab[c_HsvSectorData[sector][0]];
      g = tab[c_HsvSectorData[sector][1]];
      r = tab[c_HsvSectorData[sector][2]];
    }

    uchar3 dst;
    dst.x = saturateCast(b * 255.f);
    dst.y = saturateCast(g * 255.f);
    dst.z = saturateCast(r * 255.f);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float h = src.x;
    float s = src.y;
    float v = src.z;

    float _1_60 = 1.f / 60.f;
    float diff0 = s * v;
    float min0 = v - diff0;
    float h0 = h *_1_60;

    //V = B
    float b0 = v;
    float tmp0 = diff0 * (h0 - 4);
    bool mask0 = h0 < 4;
    float r0 = mask0 ? min0 : (min0 + tmp0);
    float g0 = mask0 ? (min0 - tmp0) : min0;

    //V = G
    tmp0 = diff0 * (h0 - 2);
    mask0 = h0 < 2;
    bool mask1 = h0 < 3;
    g0 = mask1 ? v : g0;
    mask1 = ~mask0 & mask1;
    b0 = mask0 ? min0 : b0;
    b0 = mask1 ? (min0 + tmp0) : b0;
    r0 = mask0 ? (min0 - tmp0) : r0;
    r0 = mask1 ? min0 : r0;

    //V = R
    mask0 = h0 < 1;
    tmp0 = diff0 * h0;
    r0 = mask0 ? v : r0;
    b0 = mask0 ? min0 : b0;
    g0 = mask0 ? (min0 + tmp0) : g0;

    mask0 = h0 >= 5;
    tmp0 = diff0 * (h0 - 6);
    r0 = mask0 ? v : r0;
    g0 = mask0 ? min0 : g0;
    b0 = mask0 ? (min0 - tmp0) : b0;

    float3 dst;
    dst.x = b0;
    dst.y = g0;
    dst.z = r0;

    return dst;
  }
};

struct HSV2RGBCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    const float hscale = 6.f / 180.f;

    float h = src.x;
    float s = src.y * (1.f / 255.f);
    float v = src.z * (1.f / 255.f);
    float b = v, g = v, r = v;

    if (s != 0) {
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

      int sector = __float2int_rd(h);
      h -= sector;

      if ((unsigned)sector >= 6u) {
        sector = 0;
        h = 0.f;
      }

      float tab[4];
      tab[0] = v;
      tab[1] = v * (1.f - s);
      tab[2] = v * (1.f - s * h);
      tab[3] = v * (1.f - s * (1.f - h));

      b = tab[c_HsvSectorData[sector][0]];
      g = tab[c_HsvSectorData[sector][1]];
      r = tab[c_HsvSectorData[sector][2]];
    }

    uchar3 dst;
    dst.x = saturateCast(r * 255.f);
    dst.y = saturateCast(g * 255.f);
    dst.z = saturateCast(b * 255.f);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float h = src.x;
    float s = src.y;
    float v = src.z;

    float _1_60 = 1.f / 60.f;
    float diff0 = s * v;
    float min0 = v - diff0;
    float h0 = h *_1_60;

    //V = B
    float b0 = v;
    float tmp0 = diff0 * (h0 - 4);
    bool mask0 = h0 < 4;
    float r0 = mask0 ? min0 : (min0 + tmp0);
    float g0 = mask0 ? (min0 - tmp0) : min0;

    //V = G
    tmp0 = diff0 * (h0 - 2);
    mask0 = h0 < 2;
    bool mask1 = h0 < 3;
    g0 = mask1 ? v : g0;
    mask1 = ~mask0 & mask1;
    b0 = mask0 ? min0 : b0;
    b0 = mask1 ? (min0 + tmp0) : b0;
    r0 = mask0 ? (min0 - tmp0) : r0;
    r0 = mask1 ? min0 : r0;

    //V = R
    mask0 = h0 < 1;
    tmp0 = diff0 * h0;
    r0 = mask0 ? v : r0;
    b0 = mask0 ? min0 : b0;
    g0 = mask0 ? (min0 + tmp0) : g0;

    mask0 = h0 >= 5;
    tmp0 = diff0 * (h0 - 6);
    r0 = mask0 ? v : r0;
    g0 = mask0 ? min0 : g0;
    b0 = mask0 ? (min0 - tmp0) : b0;

    float3 dst;
    dst.x = r0;
    dst.y = g0;
    dst.z = b0;

    return dst;
  }
};

struct HSV2BGRACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    const float hscale = 6.f / 180.f;

    float h = src.x;
    float s = src.y * (1.f / 255.f);
    float v = src.z * (1.f / 255.f);
    float b = v, g = v, r = v;

    if (s != 0) {
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

      int sector = __float2int_rd(h);
      h -= sector;

      if ((unsigned)sector >= 6u) {
        sector = 0;
        h = 0.f;
      }

      float tab[4];
      tab[0] = v;
      tab[1] = v * (1.f - s);
      tab[2] = v * (1.f - s * h);
      tab[3] = v * (1.f - s * (1.f - h));

      b = tab[c_HsvSectorData[sector][0]];
      g = tab[c_HsvSectorData[sector][1]];
      r = tab[c_HsvSectorData[sector][2]];
    }

    uchar4 dst;
    dst.x = saturateCast(b * 255.f);
    dst.y = saturateCast(g * 255.f);
    dst.z = saturateCast(r * 255.f);
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float h = src.x;
    float s = src.y;
    float v = src.z;

    float _1_60 = 1.f / 60.f;
    float diff0 = s * v;
    float min0 = v - diff0;
    float h0 = h *_1_60;

    //V = B
    float b0 = v;
    float tmp0 = diff0 * (h0 - 4);
    bool mask0 = h0 < 4;
    float r0 = mask0 ? min0 : (min0 + tmp0);
    float g0 = mask0 ? (min0 - tmp0) : min0;

    //V = G
    tmp0 = diff0 * (h0 - 2);
    mask0 = h0 < 2;
    bool mask1 = h0 < 3;
    g0 = mask1 ? v : g0;
    mask1 = ~mask0 & mask1;
    b0 = mask0 ? min0 : b0;
    b0 = mask1 ? (min0 + tmp0) : b0;
    r0 = mask0 ? (min0 - tmp0) : r0;
    r0 = mask1 ? min0 : r0;

    //V = R
    mask0 = h0 < 1;
    tmp0 = diff0 * h0;
    r0 = mask0 ? v : r0;
    b0 = mask0 ? min0 : b0;
    g0 = mask0 ? (min0 + tmp0) : g0;

    mask0 = h0 >= 5;
    tmp0 = diff0 * (h0 - 6);
    r0 = mask0 ? v : r0;
    g0 = mask0 ? min0 : g0;
    b0 = mask0 ? (min0 - tmp0) : b0;

    float4 dst;
    dst.x = b0;
    dst.y = g0;
    dst.z = r0;
    dst.w = 1.0f;

    return dst;
  }
};

struct HSV2RGBACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    const float hscale = 6.f / 180.f;

    float h = src.x;
    float s = src.y * (1.f / 255.f);
    float v = src.z * (1.f / 255.f);
    float b = v, g = v, r = v;

    if (s != 0) {
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

      int sector = __float2int_rd(h);
      h -= sector;

      if ((unsigned)sector >= 6u) {
        sector = 0;
        h = 0.f;
      }

      float tab[4];
      tab[0] = v;
      tab[1] = v * (1.f - s);
      tab[2] = v * (1.f - s * h);
      tab[3] = v * (1.f - s * (1.f - h));

      b = tab[c_HsvSectorData[sector][0]];
      g = tab[c_HsvSectorData[sector][1]];
      r = tab[c_HsvSectorData[sector][2]];
    }

    uchar4 dst;
    dst.x = saturateCast(r * 255.f);
    dst.y = saturateCast(g * 255.f);
    dst.z = saturateCast(b * 255.f);
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float h = src.x;
    float s = src.y;
    float v = src.z;

    float _1_60 = 1.f / 60.f;
    float diff0 = s * v;
    float min0 = v - diff0;
    float h0 = h *_1_60;

    //V = B
    float b0 = v;
    float tmp0 = diff0 * (h0 - 4);
    bool mask0 = h0 < 4;
    float r0 = mask0 ? min0 : (min0 + tmp0);
    float g0 = mask0 ? (min0 - tmp0) : min0;

    //V = G
    tmp0 = diff0 * (h0 - 2);
    mask0 = h0 < 2;
    bool mask1 = h0 < 3;
    g0 = mask1 ? v : g0;
    mask1 = ~mask0 & mask1;
    b0 = mask0 ? min0 : b0;
    b0 = mask1 ? (min0 + tmp0) : b0;
    r0 = mask0 ? (min0 - tmp0) : r0;
    r0 = mask1 ? min0 : r0;

    //V = R
    mask0 = h0 < 1;
    tmp0 = diff0 * h0;
    r0 = mask0 ? v : r0;
    b0 = mask0 ? min0 : b0;
    g0 = mask0 ? (min0 + tmp0) : g0;

    mask0 = h0 >= 5;
    tmp0 = diff0 * (h0 - 6);
    r0 = mask0 ? v : r0;
    g0 = mask0 ? min0 : g0;
    b0 = mask0 ? (min0 - tmp0) : b0;

    float4 dst;
    dst.x = r0;
    dst.y = g0;
    dst.z = b0;
    dst.w = 1.0f;

    return dst;
  }
};

/******************* BGR to Lab ******************/

enum LABShifts {
  kGammaTabSize = 1024,
  kLabShift     = 12,
  kGammaShift   = 3,
  kLabShift2    = (kLabShift + kGammaShift),
};

__constant__ ushort c_sRGBGammaTab_b[] = {0,1,1,2,2,3,4,4,5,6,6,7,8,8,9,10,11,11,12,13,14,15,16,17,19,20,21,22,24,25,26,28,29,31,33,34,36,38,40,41,43,45,47,49,51,54,56,58,60,63,65,68,70,73,75,78,81,83,86,89,92,95,98,101,105,108,111,115,118,121,125,129,132,136,140,144,147,151,155,160,164,168,172,176,181,185,190,194,199,204,209,213,218,223,228,233,239,244,249,255,260,265,271,277,282,288,294,300,306,312,318,324,331,337,343,350,356,363,370,376,383,390,397,404,411,418,426,433,440,448,455,463,471,478,486,494,502,510,518,527,535,543,552,560,569,578,586,595,604,613,622,631,641,650,659,669,678,688,698,707,717,727,737,747,757,768,778,788,799,809,820,831,842,852,863,875,886,897,908,920,931,943,954,966,978,990,1002,1014,1026,1038,1050,1063,1075,1088,1101,1113,1126,1139,1152,1165,1178,1192,1205,1218,1232,1245,1259,1273,1287,1301,1315,1329,1343,1357,1372,1386,1401,1415,1430,1445,1460,1475,1490,1505,1521,1536,1551,1567,1583,1598,1614,1630,1646,1662,1678,1695,1711,1728,1744,1761,1778,1794,1811,1828,1846,1863,1880,1897,1915,1933,1950,1968,1986,2004,2022,2040};

__device__ static
int labCbrt_b(int i) {
  float x = i * (1.f / (255.f * (1 << kGammaShift)));
  float tmp = x < 0.008856f ? x * 7.787f + 0.13793103448275862f : ::cbrtf(x);

  return (1 << kLabShift2) * tmp;
}

__device__ __forceinline__
float splineInterpolate(float x, const float* tab, int n) {
  int ix = ::min(::max(int(x), 0), n-1);
  x -= ix;
  tab += (ix << 2);//ix * 4;
  return ((tab[3] * x + tab[2]) * x + tab[1]) * x + tab[0];
}

struct BGR2LABCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    const int Lscale = (116 * 255 + 50) / 100;
    const int Lshift = -((16 * 255 * (1 << kLabShift2) + 50) / 100);

    int B = src.x;
    int G = src.y;
    int R = src.z;

    B = c_sRGBGammaTab_b[B];
    G = c_sRGBGammaTab_b[G];
    R = c_sRGBGammaTab_b[R];

    int fX = labCbrt_b(divideUp1(B * 778 + G * 1541 + R * 1777, kLabShift));
    int fY = labCbrt_b(divideUp1(B * 296 + G * 2929 + R * 871, kLabShift));
    int fZ = labCbrt_b(divideUp1(B * 3575 + G * 448 + R * 73, kLabShift));

    int L = divideUp1(Lscale * fY + Lshift, kLabShift2);
    int a = divideUp1(500 * (fX - fY) + 128 * (1 << kLabShift2), kLabShift2);
    int b = divideUp1(200 * (fY - fZ) + 128 * (1 << kLabShift2), kLabShift2);

    uchar3 dst;
    dst.x = L; //saturateCast(L);
    dst.y = a; //saturateCast(a);
    dst.z = b; //saturateCast(b);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float div_1_3     = 0.333333f;
    float div_16_116  = 0.137931f;

    float B = src.x;
    float G = src.y;
    float R = src.z;

    B = (B > 0.04045f) ? ::powf((B + 0.055f) / 1.055f, 2.4f) : B / 12.92f;
    G = (G > 0.04045f) ? ::powf((G + 0.055f) / 1.055f, 2.4f) : G / 12.92f;
    R = (R > 0.04045f) ? ::powf((R + 0.055f) / 1.055f, 2.4f) : R / 12.92f;

    float X = B * 0.189828f + G * 0.376219f + R * 0.433953f;
    float Y = B * 0.072169f + G * 0.715160f + R * 0.212671f;
    float Z = B * 0.872766f + G * 0.109477f + R * 0.017758f;

    float pow_y = ::powf(Y, div_1_3);
    float FX = X > 0.008856f ? ::powf(X, div_1_3) : (7.787f * X + div_16_116);
    float FY = Y > 0.008856f ? pow_y : (7.787f * Y + div_16_116);
    float FZ = Z > 0.008856f ? ::powf(Z, div_1_3) : (7.787f * Z + div_16_116);

    float L = Y > 0.008856f ? (116.f * pow_y - 16.f) : (903.3f * Y);
    float a = 500.f * (FX - FY);
    float b = 200.f * (FY - FZ);

    float3 dst;
    dst.x = L;
    dst.y = a;
    dst.z = b;

    return dst;
  }
};

struct RGB2LABCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    const int Lscale = (116 * 255 + 50) / 100;
    const int Lshift = -((16 * 255 * (1 << kLabShift2) + 50) / 100);

    int R = src.x;
    int G = src.y;
    int B = src.z;

    R = c_sRGBGammaTab_b[R];
    G = c_sRGBGammaTab_b[G];
    B = c_sRGBGammaTab_b[B];

    int fX = labCbrt_b(divideUp1(B * 778 + G * 1541 + R * 1777, kLabShift));
    int fY = labCbrt_b(divideUp1(B * 296 + G * 2929 + R * 871, kLabShift));
    int fZ = labCbrt_b(divideUp1(B * 3575 + G * 448 + R * 73, kLabShift));

    int L = divideUp1(Lscale * fY + Lshift, kLabShift2);
    int a = divideUp1(500 * (fX - fY) + 128 * (1 << kLabShift2), kLabShift2);
    int b = divideUp1(200 * (fY - fZ) + 128 * (1 << kLabShift2), kLabShift2);

    uchar3 dst;
    dst.x = L; //saturateCast(L);
    dst.y = a; //saturateCast(a);
    dst.z = b; //saturateCast(b);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float div_1_3     = 0.333333f;
    float div_16_116  = 0.137931f;

    float R = src.x;
    float G = src.y;
    float B = src.z;

    R = (R > 0.04045f) ? ::powf((R + 0.055f) / 1.055f, 2.4f) : R / 12.92f;
    G = (G > 0.04045f) ? ::powf((G + 0.055f) / 1.055f, 2.4f) : G / 12.92f;
    B = (B > 0.04045f) ? ::powf((B + 0.055f) / 1.055f, 2.4f) : B / 12.92f;

    float X = R * 0.433953f + G * 0.376219f + B * 0.189828f;
    float Y = R * 0.212671f + G * 0.715160f + B * 0.072169f;
    float Z = R * 0.017758f + G * 0.109477f + B * 0.872766f;

    float pow_y = ::powf(Y, div_1_3);
    float FX = X > 0.008856f ? ::powf(X, div_1_3) : (7.787f * X + div_16_116);
    float FY = Y > 0.008856f ? pow_y : (7.787f * Y + div_16_116);
    float FZ = Z > 0.008856f ? ::powf(Z, div_1_3) : (7.787f * Z + div_16_116);

    float L = Y > 0.008856f ? (116.f * pow_y - 16.f) : (903.3f * Y);
    float a = 500.f * (FX - FY);
    float b = 200.f * (FY - FZ);

    float3 dst;
    dst.x = L;
    dst.y = a;
    dst.z = b;

    return dst;
  }
};

struct BGRA2LABCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    const int Lscale = (116 * 255 + 50) / 100;
    const int Lshift = -((16 * 255 * (1 << kLabShift2) + 50) / 100);

    int B = src.x;
    int G = src.y;
    int R = src.z;

    B = c_sRGBGammaTab_b[B];
    G = c_sRGBGammaTab_b[G];
    R = c_sRGBGammaTab_b[R];

    int fX = labCbrt_b(divideUp1(B * 778 + G * 1541 + R * 1777, kLabShift));
    int fY = labCbrt_b(divideUp1(B * 296 + G * 2929 + R * 871, kLabShift));
    int fZ = labCbrt_b(divideUp1(B * 3575 + G * 448 + R * 73, kLabShift));

    int L = divideUp1(Lscale * fY + Lshift, kLabShift2);
    int a = divideUp1(500 * (fX - fY) + 128 * (1 << kLabShift2), kLabShift2);
    int b = divideUp1(200 * (fY - fZ) + 128 * (1 << kLabShift2), kLabShift2);

    uchar3 dst;
    dst.x = L; //saturateCast(L);
    dst.y = a; //saturateCast(a);
    dst.z = b; //saturateCast(b);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float div_1_3     = 0.333333f;
    float div_16_116  = 0.137931f;

    float B = src.x;
    float G = src.y;
    float R = src.z;

    B = (B > 0.04045f) ? ::powf((B + 0.055f) / 1.055f, 2.4f) : B / 12.92f;
    G = (G > 0.04045f) ? ::powf((G + 0.055f) / 1.055f, 2.4f) : G / 12.92f;
    R = (R > 0.04045f) ? ::powf((R + 0.055f) / 1.055f, 2.4f) : R / 12.92f;

    float X = B * 0.189828f + G * 0.376219f + R * 0.433953f;
    float Y = B * 0.072169f + G * 0.715160f + R * 0.212671f;
    float Z = B * 0.872766f + G * 0.109477f + R * 0.017758f;

    float pow_y = ::powf(Y, div_1_3);
    float FX = X > 0.008856f ? ::powf(X, div_1_3) : (7.787f * X + div_16_116);
    float FY = Y > 0.008856f ? pow_y : (7.787f * Y + div_16_116);
    float FZ = Z > 0.008856f ? ::powf(Z, div_1_3) : (7.787f * Z + div_16_116);

    float L = Y > 0.008856f ? (116.f * pow_y - 16.f) : (903.3f * Y);
    float a = 500.f * (FX - FY);
    float b = 200.f * (FY - FZ);

    float3 dst;
    dst.x = L;
    dst.y = a;
    dst.z = b;

    return dst;
  }
};

struct RGBA2LABCompute {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    const int Lscale = (116 * 255 + 50) / 100;
    const int Lshift = -((16 * 255 * (1 << kLabShift2) + 50) / 100);

    int R = src.x;
    int G = src.y;
    int B = src.z;

    R = c_sRGBGammaTab_b[R];
    G = c_sRGBGammaTab_b[G];
    B = c_sRGBGammaTab_b[B];

    int fX = labCbrt_b(divideUp1(B * 778 + G * 1541 + R * 1777, kLabShift));
    int fY = labCbrt_b(divideUp1(B * 296 + G * 2929 + R * 871, kLabShift));
    int fZ = labCbrt_b(divideUp1(B * 3575 + G * 448 + R * 73, kLabShift));

    int L = divideUp1(Lscale * fY + Lshift, kLabShift2);
    int a = divideUp1(500 * (fX - fY) + 128 * (1 << kLabShift2), kLabShift2);
    int b = divideUp1(200 * (fY - fZ) + 128 * (1 << kLabShift2), kLabShift2);

    uchar3 dst;
    dst.x = L; //saturateCast(L);
    dst.y = a; //saturateCast(a);
    dst.z = b; //saturateCast(b);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float4& src) {
    float div_1_3     = 0.333333f;
    float div_16_116  = 0.137931f;

    float R = src.x;
    float G = src.y;
    float B = src.z;

    R = (R > 0.04045f) ? ::powf((R + 0.055f) / 1.055f, 2.4f) : R / 12.92f;
    G = (G > 0.04045f) ? ::powf((G + 0.055f) / 1.055f, 2.4f) : G / 12.92f;
    B = (B > 0.04045f) ? ::powf((B + 0.055f) / 1.055f, 2.4f) : B / 12.92f;

    float X = R * 0.433953f + G * 0.376219f + B * 0.189828f;
    float Y = R * 0.212671f + G * 0.715160f + B * 0.072169f;
    float Z = R * 0.017758f + G * 0.109477f + B * 0.872766f;

    float pow_y = ::powf(Y, div_1_3);
    float FX = X > 0.008856f ? ::powf(X, div_1_3) : (7.787f * X + div_16_116);
    float FY = Y > 0.008856f ? pow_y : (7.787f * Y + div_16_116);
    float FZ = Z > 0.008856f ? ::powf(Z, div_1_3) : (7.787f * Z + div_16_116);

    float L = Y > 0.008856f ? (116.f * pow_y - 16.f) : (903.3f * Y);
    float a = 500.f * (FX - FY);
    float b = 200.f * (FY - FZ);

    float3 dst;
    dst.x = L;
    dst.y = a;
    dst.z = b;

    return dst;
  }
};

struct LAB2BGRCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    float _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float lThresh = 7.9996248f;   // 0.008856f * 903.3f;
    float fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

    float L, a, b;
    L = src.x * 0.392156863f; // (100.f / 255.f);
    a = src.y - 128;
    b = src.z - 128;

    float Y, fy;

    if (L <= lThresh) {
      Y = L / 903.3f;
      fy = 7.787f * Y + _16_116;
    }
    else {
      fy = (L + 16.0f) / 116.0f;
      Y = fy * fy * fy;
    }

    float X = a / 500.0f + fy;
    float Z = fy - b / 200.0f;

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

    float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
    float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
    float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

    B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;
    G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
    R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;

    B = B * 255.f;
    G = G * 255.f;
    R = R * 255.f;

    uchar3 dst;
    dst.x = saturateCast(B);
    dst.y = saturateCast(G);
    dst.z = saturateCast(R);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float lThresh = 7.9996248f;   // 0.008856f * 903.3f;
    float fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

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

    float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
    float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
    float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

    B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;
    G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
    R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;

    float3 dst;
    dst.x = B;
    dst.y = G;
    dst.z = R;

    return dst;
  }
};

struct LAB2RGBCompute {
  __DEVICE__
  uchar3 operator()(const uchar3& src) {
    float _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float lThresh = 7.9996248f;   // 0.008856f * 903.3f;
    float fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

    float L, a, b;
    L = src.x * 0.392156863f; // (100.f / 255.f);
    a = src.y - 128;
    b = src.z - 128;

    float Y, fy;

    if (L <= lThresh) {
      Y = L / 903.3f;
      fy = 7.787f * Y + _16_116;
    }
    else {
      fy = (L + 16.0f) / 116.0f;
      Y = fy * fy * fy;
    }

    float X = a / 500.0f + fy;
    float Z = fy - b / 200.0f;

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

    float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;
    float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
    float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;

    R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;
    G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
    B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;

    R = R * 255.f;
    G = G * 255.f;
    B = B * 255.f;

    uchar3 dst;
    dst.x = saturateCast(R);
    dst.y = saturateCast(G);
    dst.z = saturateCast(B);

    return dst;
  }

  __DEVICE__
  float3 operator()(const float3& src) {
    float _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float lThresh = 7.9996248f;   // 0.008856f * 903.3f;
    float fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

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

    float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;
    float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
    float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;

    R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;
    G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
    B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;

    float3 dst;
    dst.x = R;
    dst.y = G;
    dst.z = B;

    return dst;
  }
};

struct LAB2BGRACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    float _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float lThresh = 7.9996248f;   // 0.008856f * 903.3f;
    float fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

    float L, a, b;
    L = src.x * 0.392156863f; // (100.f / 255.f);
    a = src.y - 128;
    b = src.z - 128;

    float Y, fy;

    if (L <= lThresh) {
      Y = L / 903.3f;
      fy = 7.787f * Y + _16_116;
    }
    else {
      fy = (L + 16.0f) / 116.0f;
      Y = fy * fy * fy;
    }

    float X = a / 500.0f + fy;
    float Z = fy - b / 200.0f;

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

    float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
    float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
    float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

    B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;
    G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
    R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;

    B = B * 255.f;
    G = G * 255.f;
    R = R * 255.f;

    uchar4 dst;
    dst.x = saturateCast(B);
    dst.y = saturateCast(G);
    dst.z = saturateCast(R);
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float lThresh = 7.9996248f;   // 0.008856f * 903.3f;
    float fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

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

    float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
    float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
    float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

    B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;
    G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
    R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;

    float4 dst;
    dst.x = B;
    dst.y = G;
    dst.z = R;
    dst.w = 1.f;

    return dst;
  }
};

struct LAB2RGBACompute {
  __DEVICE__
  uchar4 operator()(const uchar3& src) {
    float _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float lThresh = 7.9996248f;   // 0.008856f * 903.3f;
    float fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

    float L, a, b;
    L = src.x * 0.392156863f; // (100.f / 255.f);
    a = src.y - 128;
    b = src.z - 128;

    float Y, fy;

    if (L <= lThresh) {
      Y = L / 903.3f;
      fy = 7.787f * Y + _16_116;
    }
    else {
      fy = (L + 16.0f) / 116.0f;
      Y = fy * fy * fy;
    }

    float X = a / 500.0f + fy;
    float Z = fy - b / 200.0f;

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

    float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
    float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
    float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

    B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;
    G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
    R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;

    B = B * 255.f;
    G = G * 255.f;
    R = R * 255.f;

    uchar4 dst;
    dst.x = saturateCast(R);
    dst.y = saturateCast(G);
    dst.z = saturateCast(B);
    dst.w = 255;

    return dst;
  }

  __DEVICE__
  float4 operator()(const float3& src) {
    float _16_116 = 0.137931034f; // 16.0f / 116.0f;
    float lThresh = 7.9996248f;   // 0.008856f * 903.3f;
    float fThresh = 0.206892706f; // 0.008856f * 7.787f + _16_116;

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

    float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
    float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
    float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

    B = (B > 0.00304f) ? (1.055f * ::powf(B, 0.41667f) - 0.055f) : 12.92f * B;
    G = (G > 0.00304f) ? (1.055f * ::powf(G, 0.41667f) - 0.055f) : 12.92f * G;
    R = (R > 0.00304f) ? (1.055f * ::powf(R, 0.41667f) - 0.055f) : 12.92f * R;

    float4 dst;

    dst.x = R;
    dst.y = G;
    dst.z = B;
    dst.w = 1.f;

    return dst;
  }
};

/*****************************************************************
 * NV12: YYYYUV, used on IOS.
 * NV21: YYYYVU, used on android.
 * Only unsigned char data is supported for BGR/RGB <-> NV12/NV21.
 *****************************************************************
 */

/******************* BGR/RGB/BGRA/RGBA <-> NV12 ******************/

enum {
  kG2NShift  = 7,
  kG2NShift2 = (1 << (kG2NShift - 1)),
  kN2GShift  = 6,
  kN2GShift2 = (1 << (kN2GShift - 1)),
  kG2N1Shift = 20,
  kG2N1Shift2  = (1 << (kG2N1Shift - 1)),
  kG2N1Shift16 = (16 << kG2N1Shift),
  kG2N1HalfShift = (1 << (kG2N1Shift - 1)),
  kG2N1Shift128  = (128 << kG2N1Shift),
};

/*
 * y in YUV420 is computed per BGR pixel, u and v are computed by taking the
 * left-top pixel of every 2x2 BGR matrix.
 */
struct BGR2NV12Compute {
  __DEVICE__
  uchar3 operator()(const uchar3& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YB + src.y * NVXX1_YG + src.z * NVXX1_YR +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UB + src.y * NVXX1_UG + src.z * NVXX1_UR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VB + src.y * NVXX1_VG + src.z * NVXX1_VR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      u = u < 0 ? 0 : u;
      v = v < 0 ? 0 : v;
      u = u > 255 ? 255 : u;
      v = v > 255 ? 255 : v;
      dst.y = (unsigned char) u;
      dst.z = (unsigned char) v;
    }

    return dst;
  }
};

struct RGB2NV12Compute {
  __DEVICE__
  uchar3 operator()(const uchar3& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YR + src.y * NVXX1_YG + src.z * NVXX1_YB +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UR + src.y * NVXX1_UG + src.z * NVXX1_UB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VR + src.y * NVXX1_VG + src.z * NVXX1_VB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      u = u < 0 ? 0 : u;
      v = v < 0 ? 0 : v;
      u = u > 255 ? 255 : u;
      v = v > 255 ? 255 : v;
      dst.y = (unsigned char) u;
      dst.z = (unsigned char) v;
    }

    return dst;
  }
};

struct BGRA2NV12Compute {
  __DEVICE__
  uchar3 operator()(const uchar4& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YB + src.y * NVXX1_YG + src.z * NVXX1_YR +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UB + src.y * NVXX1_UG + src.z * NVXX1_UR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VB + src.y * NVXX1_VG + src.z * NVXX1_VR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      u = u < 0 ? 0 : u;
      v = v < 0 ? 0 : v;
      u = u > 255 ? 255 : u;
      v = v > 255 ? 255 : v;
      dst.y = (unsigned char) u;
      dst.z = (unsigned char) v;
    }

    return dst;
  }
};

struct RGBA2NV12Compute {
  __DEVICE__
  uchar3 operator()(const uchar4& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YR + src.y * NVXX1_YG + src.z * NVXX1_YB +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UR + src.y * NVXX1_UG + src.z * NVXX1_UB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VR + src.y * NVXX1_VG + src.z * NVXX1_VB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      u = u < 0 ? 0 : u;
      v = v < 0 ? 0 : v;
      u = u > 255 ? 255 : u;
      v = v > 255 ? 255 : v;
      dst.y = (unsigned char) u;
      dst.z = (unsigned char) v;
    }

    return dst;
  }
};

/*
 * y -> BGR: one to one.
 * uv -> BGR: one to four, should be put on the shared memory.
 */
struct NV122BGRCompute {
  __DEVICE__
  uchar3 operator()(const unsigned char& src_y, const uchar2 &src_uv) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int u = src_uv.x - 128;
    int v = src_uv.y - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);

    return dst;
  }
};

struct NV122RGBCompute {
  __DEVICE__
  uchar3 operator()(const unsigned char& src_y, const uchar2 &src_uv) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int u = src_uv.x - 128;
    int v = src_uv.y - 128;

    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;

    int r = (y + ruv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int b = (y + buv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(r);
    dst.y = saturateCast(g);
    dst.z = saturateCast(b);

    return dst;
  }
};

struct NV122BGRACompute {
  __DEVICE__
  uchar4 operator()(const unsigned char& src_y, const uchar2 &src_uv) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int u = src_uv.x - 128;
    int v = src_uv.y - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar4 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);
    dst.w = 255;

    return dst;
  }
};

struct NV122RGBACompute {
  __DEVICE__
  uchar4 operator()(const unsigned char& src_y, const uchar2 &src_uv) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int u = src_uv.x - 128;
    int v = src_uv.y - 128;

    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;

    int r = (y + ruv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int b = (y + buv) >> NVXX1_SHIFT;

    uchar4 dst;
    dst.x = saturateCast(r);
    dst.y = saturateCast(g);
    dst.z = saturateCast(b);
    dst.w = 255;

    return dst;
  }
};

/******************* BGR/RGB/BGRA/RGBA <-> NV21 ******************/

struct BGR2NV21Compute {
  __DEVICE__
  uchar3 operator()(const uchar3& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YB + src.y * NVXX1_YG + src.z * NVXX1_YR +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UB + src.y * NVXX1_UG + src.z * NVXX1_UR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VB + src.y * NVXX1_VG + src.z * NVXX1_VR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      u = u < 0 ? 0 : u;
      v = v < 0 ? 0 : v;
      u = u > 255 ? 255 : u;
      v = v > 255 ? 255 : v;
      dst.y = (unsigned char) v;
      dst.z = (unsigned char) u;
    }

    return dst;
  }
};

struct RGB2NV21Compute {
  __DEVICE__
  uchar3 operator()(const uchar3& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YR + src.y * NVXX1_YG + src.z * NVXX1_YB +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UR + src.y * NVXX1_UG + src.z * NVXX1_UB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VR + src.y * NVXX1_VG + src.z * NVXX1_VB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      u = u < 0 ? 0 : u;
      v = v < 0 ? 0 : v;
      u = u > 255 ? 255 : u;
      v = v > 255 ? 255 : v;
      dst.y = (unsigned char) v;
      dst.z = (unsigned char) u;
    }

    return dst;
  }
};

struct BGRA2NV21Compute {
  __DEVICE__
  uchar3 operator()(const uchar4& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YB + src.y * NVXX1_YG + src.z * NVXX1_YR +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UB + src.y * NVXX1_UG + src.z * NVXX1_UR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VB + src.y * NVXX1_VG + src.z * NVXX1_VR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      u = u < 0 ? 0 : u;
      v = v < 0 ? 0 : v;
      u = u > 255 ? 255 : u;
      v = v > 255 ? 255 : v;
      dst.y = (unsigned char) v;
      dst.z = (unsigned char) u;
    }

    return dst;
  }
};

struct RGBA2NV21Compute {
  __DEVICE__
  uchar3 operator()(const uchar4& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YR + src.y * NVXX1_YG + src.z * NVXX1_YB +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UR + src.y * NVXX1_UG + src.z * NVXX1_UB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VR + src.y * NVXX1_VG + src.z * NVXX1_VB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      u = u < 0 ? 0 : u;
      v = v < 0 ? 0 : v;
      u = u > 255 ? 255 : u;
      v = v > 255 ? 255 : v;
      dst.y = (unsigned char) v;
      dst.z = (unsigned char) u;
    }

    return dst;
  }
};

struct NV212BGRCompute {
  __DEVICE__
  uchar3 operator()(const unsigned char& src_y, const uchar2 &src_uv) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int v = src_uv.x - 128;
    int u = src_uv.y - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);

    return dst;
  }
};

struct NV212RGBCompute {
  __DEVICE__
  uchar3 operator()(const unsigned char& src_y, const uchar2 &src_uv) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int v = src_uv.x - 128;
    int u = src_uv.y - 128;

    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;

    int r = (y + ruv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int b = (y + buv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(r);
    dst.y = saturateCast(g);
    dst.z = saturateCast(b);

    return dst;
  }
};

struct NV212BGRACompute {
  __DEVICE__
  uchar4 operator()(const unsigned char& src_y, const uchar2 &src_uv) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int v = src_uv.x - 128;
    int u = src_uv.y - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar4 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);
    dst.w = 255;

    return dst;
  }
};

struct NV212RGBACompute {
  __DEVICE__
  uchar4 operator()(const unsigned char& src_y, const uchar2 &src_uv) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int v = src_uv.x - 128;
    int u = src_uv.y - 128;

    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;

    int r = (y + ruv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int b = (y + buv) >> NVXX1_SHIFT;

    uchar4 dst;
    dst.x = saturateCast(r);
    dst.y = saturateCast(g);
    dst.z = saturateCast(b);
    dst.w = 255;

    return dst;
  }
};

/******************* BGR to I420 ******************/

/*
 * y in YUV420 is computed per BGR pixel, u and v are computed by taking the
 * left-top pixel of every 2x2 BGR matrix.
 */
struct BGR2I420Compute {
  __DEVICE__
  uchar3 operator()(const uchar3& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YB + src.y * NVXX1_YG + src.z * NVXX1_YR +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UB + src.y * NVXX1_UG + src.z * NVXX1_UR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VB + src.y * NVXX1_VG + src.z * NVXX1_VR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      dst.y = saturateCast(u);
      dst.z = saturateCast(v);
    }

    return dst;
  }
};

struct RGB2I420Compute {
  __DEVICE__
  uchar3 operator()(const uchar3& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YR + src.y * NVXX1_YG + src.z * NVXX1_YB +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UR + src.y * NVXX1_UG + src.z * NVXX1_UB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VR + src.y * NVXX1_VG + src.z * NVXX1_VB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      dst.y = saturateCast(u);
      dst.z = saturateCast(v);
    }

    return dst;
  }
};

struct BGRA2I420Compute {
  __DEVICE__
  uchar3 operator()(const uchar4& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YB + src.y * NVXX1_YG + src.z * NVXX1_YR +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UB + src.y * NVXX1_UG + src.z * NVXX1_UR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VB + src.y * NVXX1_VG + src.z * NVXX1_VR +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      dst.y = saturateCast(u);
      dst.z = saturateCast(v);
    }

    return dst;
  }
};

struct RGBA2I420Compute {
  __DEVICE__
  uchar3 operator()(const uchar4& src, unsigned int row, unsigned int col) {
    uchar3 dst;

    int y = (src.x * NVXX1_YR + src.y * NVXX1_YG + src.z * NVXX1_YB +
            kG2N1HalfShift + kG2N1Shift16) >> kG2N1Shift;
    y = y < 0 ? 0 : y;
    y = y > 255 ? 255 : y;
    dst.x = (unsigned char) y;

    // Interpolate every 2 rows and 2 columns.
    if (((row + 1) & 1) && ((col + 1) & 1)) {
      int u = (src.x * NVXX1_UR + src.y * NVXX1_UG + src.z * NVXX1_UB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;
      int v = (src.x * NVXX1_VR + src.y * NVXX1_VG + src.z * NVXX1_VB +
              kG2N1HalfShift + kG2N1Shift128) >> kG2N1Shift;

      dst.y = saturateCast(u);
      dst.z = saturateCast(v);
    }

    return dst;
  }
};

/*
 * y -> BGR: one to one.
 * uv -> BGR: one to four, should be put on the shared memory.
 */
struct I4202BGRCompute {
  __DEVICE__
  uchar3 operator()(const unsigned char& src_y, const uchar &src_u,
                    const uchar &src_v) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int u = src_u - 128;
    int v = src_v - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);

    return dst;
  }
};

struct I4202RGBCompute {
  __DEVICE__
  uchar3 operator()(const unsigned char& src_y, const uchar &src_u,
                    const uchar &src_v) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int u = src_u - 128;
    int v = src_v - 128;

    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;

    int r = (y + ruv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int b = (y + buv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(r);
    dst.y = saturateCast(g);
    dst.z = saturateCast(b);

    return dst;
  }
};

struct I4202BGRACompute {
  __DEVICE__
  uchar4 operator()(const unsigned char& src_y, const uchar &src_u,
                    const uchar &src_v) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int u = src_u - 128;
    int v = src_v - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar4 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);
    dst.w = 255;

    return dst;
  }
};

struct I4202RGBACompute {
  __DEVICE__
  uchar4 operator()(const unsigned char& src_y, const uchar &src_u,
                    const uchar &src_v) {
    int y = max(0, (src_y - 16)) * NVXX1_CY;
    int u = src_u - 128;
    int v = src_v - 128;

    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;

    int r = (y + ruv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int b = (y + buv) >> NVXX1_SHIFT;

    uchar4 dst;
    dst.x = saturateCast(r);
    dst.y = saturateCast(g);
    dst.z = saturateCast(b);
    dst.w = 255;

    return dst;
  }
};

/******************* BGR/GRAY <-> UYVY ******************/

struct UYVY2BGRCompute0 {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    int y = max(0, (src.y - 16)) * NVXX1_CY;
    int u = src.x - 128;
    int v = src.z - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);

    return dst;
  }
};

struct UYVY2BGRCompute1 {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    int y = max(0, (src.w - 16)) * NVXX1_CY;
    int u = src.x - 128;
    int v = src.z - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);

    return dst;
  }
};

struct UYVY2GRAYCompute {
  __DEVICE__
  uchar2 operator()(const uchar2& src0, const uchar2& src1) {
    uchar2 dst;
    dst.x = src0.y;
    dst.y = src1.y;

    return dst;
  }
};

/******************* BGR/GRAY <-> YUYV ******************/

struct YUYV2BGRCompute0 {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    int y = max(0, (src.x - 16)) * NVXX1_CY;
    int u = src.y - 128;
    int v = src.w - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);

    return dst;
  }
};

struct YUYV2BGRCompute1 {
  __DEVICE__
  uchar3 operator()(const uchar4& src) {
    int y = max(0, (src.z - 16)) * NVXX1_CY;
    int u = src.y - 128;
    int v = src.w - 128;

    int buv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CUB * u;
    int guv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVG * v + NVXX1_CUG * u;
    int ruv = (1 << (NVXX1_SHIFT - 1)) + NVXX1_CVR * v;

    int b = (y + buv) >> NVXX1_SHIFT;
    int g = (y + guv) >> NVXX1_SHIFT;
    int r = (y + ruv) >> NVXX1_SHIFT;

    uchar3 dst;
    dst.x = saturateCast(b);
    dst.y = saturateCast(g);
    dst.z = saturateCast(r);

    return dst;
  }
};

struct YUYV2GRAYCompute {
  __DEVICE__
  uchar2 operator()(const uchar2& src0, const uchar2& src1) {
    uchar2 dst;
    dst.x = src0.x;
    dst.y = src1.x;

    return dst;
  }
};

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_CVTCOLOR_COMPUTE_HPP_
