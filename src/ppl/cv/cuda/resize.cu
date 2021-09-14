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

#include "ppl/cv/cuda/resize.h"

#include "utility.hpp"

#define MIN(a,b)  ((a) < (b) ? (a) : (b))
#define MAX(a,b)  ((a) > (b) ? (a) : (b))
#define INC(x, l) ((x + 1) >= (l) ? (x) : ((x) + 1))
#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename T>
__DEVICE__
T bilinearSampleUchar(T t[][2], int x0, int x1, int y0, int y1);

template <>
__DEVICE__
uchar2 bilinearSampleUchar(uchar2 t[][2], int x0, int x1, int y0, int y1) {
  int a0 = y0 * x0;
  int a1 = y0 * x1;
  int a2 = y1 * x0;
  int a3 = y1 * x1;

  int2 ret;
  uchar2 final_ret;
  ret.x = t[0][0].x * a0 + t[0][1].x * a1 + t[1][0].x * a2 + t[1][1].x * a3;
  final_ret.x = (ret.x + (1<<(CAST_BITS-1))) >> CAST_BITS;
  ret.y = t[0][0].y * a0 + t[0][1].y * a1 + t[1][0].y * a2 + t[1][1].y * a3;
  final_ret.y = (ret.y + (1<<(CAST_BITS-1))) >> CAST_BITS;

  return final_ret;
}

template <>
__DEVICE__
uchar3 bilinearSampleUchar(uchar3 t[][2], int x0, int x1, int y0, int y1) {
  int a0 = y0 * x0;
  int a1 = y0 * x1;
  int a2 = y1 * x0;
  int a3 = y1 * x1;

  int3 ret;
  uchar3 final_ret;
  ret.x = t[0][0].x * a0 + t[0][1].x * a1 + t[1][0].x * a2 + t[1][1].x * a3;
  final_ret.x = (ret.x + (1<<(CAST_BITS-1))) >> CAST_BITS;
  ret.y = t[0][0].y * a0 + t[0][1].y * a1 + t[1][0].y * a2 + t[1][1].y * a3;
  final_ret.y = (ret.y + (1<<(CAST_BITS-1))) >> CAST_BITS;
  ret.z = t[0][0].z * a0 + t[0][1].z * a1 + t[1][0].z * a2 + t[1][1].z * a3;
  final_ret.z = (ret.z + (1<<(CAST_BITS-1))) >> CAST_BITS;

  return final_ret;
}

template <>
__DEVICE__
uchar4 bilinearSampleUchar(uchar4 t[][2], int x0, int x1, int y0, int y1) {
  int a0 = y0 * x0;
  int a1 = y0 * x1;
  int a2 = y1 * x0;
  int a3 = y1 * x1;

  int4 ret;
  uchar4 final_ret;
  ret.x = t[0][0].x * a0 + t[0][1].x * a1 + t[1][0].x * a2 + t[1][1].x * a3;
  final_ret.x = (ret.x + (1<<(CAST_BITS-1))) >> CAST_BITS;
  ret.y = t[0][0].y * a0 + t[0][1].y * a1 + t[1][0].y * a2 + t[1][1].y * a3;
  final_ret.y = (ret.y + (1<<(CAST_BITS-1))) >> CAST_BITS;
  ret.z = t[0][0].z * a0 + t[0][1].z * a1 + t[1][0].z * a2 + t[1][1].z * a3;
  final_ret.z = (ret.z + (1<<(CAST_BITS-1))) >> CAST_BITS;
  ret.w = t[0][0].w * a0 + t[0][1].w * a1 + t[1][0].w * a2 + t[1][1].w * a3;
  final_ret.w = (ret.w + (1<<(CAST_BITS-1))) >> CAST_BITS;

  return final_ret;
}

/***************************** ResizeLinear() ******************************/

__global__
void resizeLinearKernel(const uchar* src, int src_rows, int src_cols,
                        int channels, int src_stride, uchar* dst, int dst_rows,
                        int dst_cols, int dst_stride, float col_scale,
                        float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float fy = ((element_y + 0.5f) * row_scale - 0.5f);
  float fx = ((element_x + 0.5f) * col_scale - 0.5f);
  int sy = floor(fy);
  int sx = floor(fx);
  fy -= sy;
  fx -= sx;
  if (sy < 0) {
    sy = 0;
    fy = 0;
  }
  if (sx < 0) {
    sx = 0;
    fx = 0;
  }
  if (sy >= src_rows) {
    sy = src_rows - 1;
    fy = 0;
  }
  if (sx >= src_cols) {
    sx = src_cols - 1;
    fx = 0;
  }

  int sy_ = INC(sy, src_rows);
  int cbufy[2];
  fy = fy * INTER_RESIZE_COEF_SCALE;
  cbufy[0] = rint(INTER_RESIZE_COEF_SCALE - fy);
  cbufy[1] = rint(fy);

  int sx_ = INC(sx, src_cols);
  int cbufx[2];
  fx = fx * INTER_RESIZE_COEF_SCALE;
  cbufx[0] = rint(INTER_RESIZE_COEF_SCALE - rint(fx));
  cbufx[1] = rint(fx);

  if (channels == 1) {
    int src_index0 = sy * src_stride + sx;
    int src_index1 = sy * src_stride + sx_;
    int src_index2 = sy_ * src_stride + sx;
    int src_index3 = sy_ * src_stride + sx_;
    int dst_index = element_y * dst_stride + element_x;

    int sum = 0;
    sum = cbufy[0] * cbufx[0] * src[src_index0] +
          cbufy[0] * cbufx[1] * src[src_index1] +
          cbufy[1] * cbufx[0] * src[src_index2] +
          cbufy[1] * cbufx[1] * src[src_index3];
    dst[dst_index] = (sum + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  }
  else if (channels == 2) {
    uchar2* input0 = (uchar2*)((uchar*)src + sy * src_stride);
    uchar2* input1 = (uchar2*)((uchar*)src + sy_ * src_stride);
    uchar2* output = (uchar2*)((uchar*)dst + element_y * dst_stride);

    uchar2 t[2][2];
    t[0][0] = input0[sx];
    t[0][1] = input0[sx_];
    t[1][0] = input1[sx];
    t[1][1] = input1[sx_];

    output[element_x] = bilinearSampleUchar(t, cbufx[0], cbufx[1], cbufy[0],
                                            cbufy[1]);
  }
  else if (channels == 3) {
    uchar3* input0 = (uchar3*)((uchar*)src + sy * src_stride);
    uchar3* input1 = (uchar3*)((uchar*)src + sy_ * src_stride);
    uchar3* output = (uchar3*)((uchar*)dst + element_y * dst_stride);

    uchar3 t[2][2];
    t[0][0] = input0[sx];
    t[0][1] = input0[sx_];
    t[1][0] = input1[sx];
    t[1][1] = input1[sx_];

    output[element_x] = bilinearSampleUchar(t, cbufx[0], cbufx[1], cbufy[0],
                                            cbufy[1]);
  }
  else {
    uchar4* input0 = (uchar4*)((uchar*)src + sy * src_stride);
    uchar4* input1 = (uchar4*)((uchar*)src + sy_ * src_stride);
    uchar4* output = (uchar4*)((uchar*)dst + element_y * dst_stride);

    uchar4 t[2][2];
    t[0][0] = input0[sx];
    t[0][1] = input0[sx_];
    t[1][0] = input1[sx];
    t[1][1] = input1[sx_];

    output[element_x] = bilinearSampleUchar(t, cbufx[0], cbufx[1], cbufy[0],
                                            cbufy[1]);
  }
}

__global__
void resizeLinearKernel(const float* src, int src_rows, int src_cols,
                        int channels, int src_stride, float* dst, int dst_rows,
                        int dst_cols, int dst_stride, double col_scale,
                        float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float fx = ((element_x + 0.5f) * col_scale - 0.5f);
  float fy = ((element_y + 0.5f) * row_scale - 0.5f);
  int sx = floor(fx);
  int sy = floor(fy);
  fx -= sx;
  fy -= sy;
  if (sy < 0) {
    sy = 0;
    fy = 0;
  }
  if (sx < 0) {
    sx = 0;
    fx = 0;
  }
  if (sy >= src_rows) {
    sy = src_rows - 1;
    fy = 0;
  }
  if (sx >= src_cols) {
    sx = src_cols - 1;
    fx = 0;
  }

  int sy_ = INC(sy,src_rows);
  float cbufy[2];
  cbufy[0] = 1.f - fy;
  cbufy[1] = 1.f - cbufy[0];

  int sx_ = INC(sx,src_cols);
  float cbufx[2];
  cbufx[0] = 1.f - fx;
  cbufx[1] = 1.f - cbufx[0];

  if (channels == 1) {
    int index = sy * src_stride;
    float src1 = src[index + sx];
    float src2 = src[index + sx_];
    float value1 = cbufy[0] * cbufx[0] * src1;
    float value2 = cbufy[0] * cbufx[1] * src2;
    float sum = 0.f;
    sum += value1 + value2;

    index = sy_ * src_stride;
    src1 = src[index + sx];
    src2 = src[index + sx_];
    value1 = cbufy[1] * cbufx[0] * src1;
    value2 = cbufy[1] * cbufx[1] * src2;
    sum += value1 + value2;

    index = element_y * dst_stride + element_x;
    dst[index] = sum;
  }
  else if (channels == 3) {
    int index = sy * src_stride;
    float3 src1 = ((float3*)(src + index))[sx];
    float3 src2 = ((float3*)(src + index))[sx_];
    float3 value1 = cbufy[0] * cbufx[0] * src1;
    float3 value2 = cbufy[0] * cbufx[1] * src2;
    float3 sum = make_float3(0.f, 0.f, 0.f);
    sum += value1;
    sum += value2;

    index = sy_ * src_stride;
    src1 = ((float3*)(src + index))[sx];
    src2 = ((float3*)(src + index))[sx_];
    value1 = cbufy[1] * cbufx[0] * src1;
    value2 = cbufy[1] * cbufx[1] * src2;
    sum += value1;
    sum += value2;

    float3* output = (float3*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
  else {
    int index = sy * src_stride;
    float4 src1 = ((float4*)(src + index))[sx];
    float4 src2 = ((float4*)(src + index))[sx_];
    float4 value1 = cbufy[0] * cbufx[0] * src1;
    float4 value2 = cbufy[0] * cbufx[1] * src2;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    sum += value1;
    sum += value2;

    index = sy_ * src_stride;
    src1 = ((float4*)(src + index))[sx];
    src2 = ((float4*)(src + index))[sx_];
    value1 = cbufy[1] * cbufx[0] * src1;
    value2 = cbufy[1] * cbufx[1] * src2;
    sum += value1;
    sum += value2;

    float4* output = (float4*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
}

RetCode resizeLinear(const uchar* src, int src_rows, int src_cols, int channels,
                     int src_stride, uchar* dst, int dst_rows, int dst_cols,
                     int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);

  cudaError_t code = cudaSuccess;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, src_rows * src_stride * sizeof(uchar),
                             cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  const int kBlockX = 32;
  const int kBlockY = 16;
  dim3 block(kBlockX, kBlockY);
  dim3 grid;
  grid.x = (dst_cols + kBlockX -1) / kBlockX;
  grid.y = (dst_rows + kBlockY - 1) / kBlockY;

  float col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;

  resizeLinearKernel<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
      channels, src_stride, dst, dst_rows, dst_cols, dst_stride, col_scale,
      row_scale);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode resizeLinear(const float* src, int src_rows, int src_cols, int channels,
                     int src_stride, float* dst, int dst_rows, int dst_cols,
                     int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);

  cudaError_t code = cudaSuccess;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, src_rows * src_stride * sizeof(float),
                             cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  const int kBlockX = 32;
  const int kBlockY = 4;
  dim3 block(kBlockX, kBlockY);
  dim3 grid;
  grid.x = (dst_cols + kBlockX -1) / kBlockX;
  grid.y = (dst_rows + kBlockY - 1) / kBlockY;

  double col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;

  resizeLinearKernel<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
      channels, src_stride, dst, dst_rows, dst_cols, dst_stride, col_scale,
      row_scale);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode ResizeLinear<uchar, 1>(cudaStream_t stream,
                               int inHeight,
                               int inWidth,
                               int inWidthStride,
                               const uchar* inData,
                               int outHeight,
                               int outWidth,
                               int outWidthStride,
                               uchar* outData) {
  RetCode code = resizeLinear(inData, inHeight, inWidth, 1, inWidthStride,
                              outData, outHeight, outWidth, outWidthStride,
                              stream);

  return code;
}

template <>
RetCode ResizeLinear<uchar, 3>(cudaStream_t stream,
                               int inHeight,
                               int inWidth,
                               int inWidthStride,
                               const uchar* inData,
                               int outHeight,
                               int outWidth,
                               int outWidthStride,
                               uchar* outData) {
  RetCode code = resizeLinear(inData, inHeight, inWidth, 3, inWidthStride,
                              outData, outHeight, outWidth, outWidthStride,
                              stream);

  return code;
}

template <>
RetCode ResizeLinear<uchar, 4>(cudaStream_t stream,
                               int inHeight,
                               int inWidth,
                               int inWidthStride,
                               const uchar* inData,
                               int outHeight,
                               int outWidth,
                               int outWidthStride,
                               uchar* outData) {
  RetCode code = resizeLinear(inData, inHeight, inWidth, 4, inWidthStride,
                              outData, outHeight, outWidth, outWidthStride,
                              stream);

  return code;
}

template <>
RetCode ResizeLinear<float, 1>(cudaStream_t stream,
                               int inHeight,
                               int inWidth,
                               int inWidthStride,
                               const float* inData,
                               int outHeight,
                               int outWidth,
                               int outWidthStride,
                               float* outData) {
  RetCode code = resizeLinear(inData, inHeight, inWidth, 1, inWidthStride,
                              outData, outHeight, outWidth, outWidthStride,
                              stream);

  return code;
}

template <>
RetCode ResizeLinear<float, 3>(cudaStream_t stream,
                               int inHeight,
                               int inWidth,
                               int inWidthStride,
                               const float* inData,
                               int outHeight,
                               int outWidth,
                               int outWidthStride,
                               float* outData) {
  RetCode code = resizeLinear(inData, inHeight, inWidth, 3, inWidthStride,
                              outData, outHeight, outWidth, outWidthStride,
                              stream);

  return code;
}

template <>
RetCode ResizeLinear<float, 4>(cudaStream_t stream,
                               int inHeight,
                               int inWidth,
                               int inWidthStride,
                               const float* inData,
                               int outHeight,
                               int outWidth,
                               int outWidthStride,
                               float* outData) {
  RetCode code = resizeLinear(inData, inHeight, inWidth, 4, inWidthStride,
                              outData, outHeight, outWidth, outWidthStride,
                              stream);

  return code;
}

/************************** resizeNearestPoint() ***************************/

template <typename T0, typename T1>
__global__
void resizeNearestPointKernel(const T1* src, int src_rows, int src_cols,
                              int channels, int src_stride, T1* dst,
                              int dst_rows, int dst_cols, int dst_stride,
                              float col_scale, float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int sy = element_y * row_scale;
  sy = MIN(sy, src_rows - 1);
  int sx = element_x * col_scale;
  sx = MIN(sx, src_cols - 1);

  T0* input  = (T0*)(src + sy* src_stride);
  T0* output = (T0*)(dst + element_y * dst_stride);
  output[element_x] = input[sx];
}

RetCode resizeNearestPoint(const uchar* src, int src_rows, int src_cols,
                           int channels, int src_stride, uchar* dst,
                           int dst_rows, int dst_cols, int dst_stride,
                           cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);

  cudaError_t code = cudaSuccess;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, src_rows * src_stride * sizeof(uchar),
                             cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  const int kBlockX = 32;
  const int kBlockY = 4;
  dim3 block(kBlockX, kBlockY);
  dim3 grid;
  grid.x = (dst_cols + kBlockX -1) / kBlockX;
  grid.y = (dst_rows + kBlockY - 1) / kBlockY;

  float col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;

  if (channels == 1) {
    resizeNearestPointKernel<uchar, uchar><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, col_scale, row_scale);
  }
  else if (channels == 3) {
    resizeNearestPointKernel<uchar3, uchar><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, col_scale, row_scale);
  }
  else {
    resizeNearestPointKernel<uchar4, uchar><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, col_scale, row_scale);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode resizeNearestPoint(const float* src, int src_rows, int src_cols,
                           int channels, int src_stride, float* dst,
                           int dst_rows, int dst_cols, int dst_stride,
                           cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);

  cudaError_t code = cudaSuccess;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, src_rows * src_stride * sizeof(float),
                             cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  const int kBlockX = 32;
  const int kBlockY = 4;
  dim3 block(kBlockX, kBlockY);
  dim3 grid;
  grid.x = (dst_cols + kBlockX -1) / kBlockX;
  grid.y = (dst_rows + kBlockY - 1) / kBlockY;

  float col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;

  if (channels == 1) {
    resizeNearestPointKernel<float, float><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, col_scale, row_scale);
  }
  else if (channels == 3) {
    resizeNearestPointKernel<float3, float><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, col_scale, row_scale);
  }
  else {
    resizeNearestPointKernel<float4, float><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, col_scale, row_scale);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode ResizeNearestPoint<uchar, 1>(cudaStream_t stream,
                                     int inHeight,
                                     int inWidth,
                                     int inWidthStride,
                                     const uchar* inData,
                                     int outHeight,
                                     int outWidth,
                                     int outWidthStride,
                                     uchar* outData) {
  RetCode code = resizeNearestPoint(inData, inHeight, inWidth, 1, inWidthStride,
                                    outData, outHeight, outWidth,
                                    outWidthStride, stream);

  return code;
}

template <>
RetCode ResizeNearestPoint<uchar, 3>(cudaStream_t stream,
                                     int inHeight,
                                     int inWidth,
                                     int inWidthStride,
                                     const uchar* inData,
                                     int outHeight,
                                     int outWidth,
                                     int outWidthStride,
                                     uchar* outData) {
  RetCode code = resizeNearestPoint(inData, inHeight, inWidth, 3, inWidthStride,
                                    outData, outHeight, outWidth,
                                    outWidthStride, stream);

  return code;
}

template <>
RetCode ResizeNearestPoint<uchar, 4>(cudaStream_t stream,
                                     int inHeight,
                                     int inWidth,
                                     int inWidthStride,
                                     const uchar* inData,
                                     int outHeight,
                                     int outWidth,
                                     int outWidthStride,
                                     uchar* outData) {
  RetCode code = resizeNearestPoint(inData, inHeight, inWidth, 4, inWidthStride,
                                    outData, outHeight, outWidth,
                                    outWidthStride, stream);

  return code;
}

template <>
RetCode ResizeNearestPoint<float, 1>(cudaStream_t stream,
                                     int inHeight,
                                     int inWidth,
                                     int inWidthStride,
                                     const float* inData,
                                     int outHeight,
                                     int outWidth,
                                     int outWidthStride,
                                     float* outData) {
  RetCode code = resizeNearestPoint(inData, inHeight, inWidth, 1, inWidthStride,
                                    outData, outHeight, outWidth,
                                    outWidthStride, stream);

  return code;
}

template <>
RetCode ResizeNearestPoint<float, 3>(cudaStream_t stream,
                                     int inHeight,
                                     int inWidth,
                                     int inWidthStride,
                                     const float* inData,
                                     int outHeight,
                                     int outWidth,
                                     int outWidthStride,
                                     float* outData) {
  RetCode code = resizeNearestPoint(inData, inHeight, inWidth, 3, inWidthStride,
                                    outData, outHeight, outWidth,
                                    outWidthStride, stream);

  return code;
}

template <>
RetCode ResizeNearestPoint<float, 4>(cudaStream_t stream,
                                     int inHeight,
                                     int inWidth,
                                     int inWidthStride,
                                     const float* inData,
                                     int outHeight,
                                     int outWidth,
                                     int outWidthStride,
                                     float* outData) {
  RetCode code = resizeNearestPoint(inData, inHeight, inWidth, 4, inWidthStride,
                                    outData, outHeight, outWidth,
                                    outWidthStride, stream);

  return code;
}

/****************************** ResizeArea() *******************************/

template <typename T>
__global__
void resizeAreaKernel0C1(const T* src, int src_rows, int src_cols, int channels,
                         int src_stride, T* dst, int dst_rows, int dst_cols,
                         int dst_stride, int col_scale, int row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  int area = (x_end - x_start) * (y_end - y_start);

  float sum = 0.f;
  T* input;
  for (int i = y_start; i < y_end; ++i) {
    input = (T*)(src + i * src_stride);
    for (int j = x_start; j < x_end; ++j) {
      sum += input[j];
    }
  }
  sum /= area;

  T* output = (T*)(dst + element_y * dst_stride);
  if (sizeof(T) == 1) {
    output[element_x] = saturate_cast(sum);
  }
  else {
    output[element_x] = sum;
  }
}

template <typename T0, typename T1>
__global__
void resizeAreaKernel0C2(const T1* src, int src_rows, int src_cols,
                         int channels, int src_stride, T1* dst, int dst_rows,
                         int dst_cols, int dst_stride, int col_scale,
                         int row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  int area = (x_end - x_start) * (y_end - y_start);

  float2 sum = make_float2(0.f, 0.f);
  T0* input;
  for (int i = y_start; i < y_end; ++i) {
    input = (T0*)(src + i * src_stride);
    for (int j = x_start; j < x_end; ++j) {
      sum += input[j];
    }
  }
  sum /= area;

  T0* output = (T0*)(dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<T0, float2>(sum);
}

template <typename T0, typename T1>
__global__
void resizeAreaKernel0C3(const T1* src, int src_rows, int src_cols,
                         int channels, int src_stride, T1* dst, int dst_rows,
                         int dst_cols, int dst_stride, int col_scale,
                         int row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  int area = (x_end - x_start) * (y_end - y_start);

  float3 sum = make_float3(0.f, 0.f, 0.f);
  T0* input;
  for (int i = y_start; i < y_end; ++i) {
    input = (T0*)(src + i * src_stride);
    for (int j = x_start; j < x_end; ++j) {
      sum += input[j];
    }
  }
  sum /= area;

  T0* output = (T0*)(dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<T0, float3>(sum);
}

template <typename T0, typename T1>
__global__
void resizeAreaKernel0C4(const T1* src, int src_rows, int src_cols,
                         int channels, int src_stride, T1* dst, int dst_rows,
                         int dst_cols, int dst_stride, int col_scale,
                         int row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int x_start = element_x * col_scale;
  int y_start = element_y * row_scale;
  int x_end = x_start + col_scale;
  int y_end = y_start + row_scale;
  x_end = (x_end <= src_cols) ? x_end : src_cols;
  y_end = (y_end <= src_rows) ? y_end : src_rows;
  int area = (x_end - x_start) * (y_end - y_start);

  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  T0* input;
  for (int i = y_start; i < y_end; ++i) {
    input = (T0*)(src + i * src_stride);
    for (int j = x_start; j < x_end; ++j) {
      sum += input[j];
    }
  }
  sum /= area;

  T0* output = (T0*)(dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<T0, float4>(sum);
}

template <typename T>
__global__
void resizeAreaKernel1C1(const T* src, int src_rows, int src_cols, int channels,
                         int src_stride, T* dst, int dst_rows, int dst_cols,
                         int dst_stride, float col_scale, float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float fsy1 = element_y * row_scale;
  float fsy2 = fsy1 + row_scale;
  int sy1 = ceilf(fsy1);
  int sy2 = floorf(fsy2);

  float fsx1 = element_x * col_scale;
  float fsx2 = fsx1 + col_scale;
  int sx1 = ceilf(fsx1);
  int sx2 = floorf(fsx2);

  T* input;
  float sum = 0.f;
  float area = fminf(col_scale, src_cols - fsx1) *
               fminf(row_scale, src_rows - fsy1);

  if (sy1 - fsy1 > 1e-3) {
    input = (T*)(src + (sy1 - 1) * src_stride);
    if (sx1 - fsx1 > 1e-3) {
      sum = sum + input[sx1 - 1] * (sy1 - fsy1) * (sx1 - fsx1);
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      sum = sum + input[dx] * (sy1 - fsy1);
    }

    if (fsx2 - sx2 > 1e-3) {
      sum = sum + input[sx2] * (sy1 - fsy1) * (fsx2 - sx2);
    }
  }

  input = (T*)(src + sy1 * src_stride);
  for (int dy = sy1; dy < sy2; ++dy) {
    if (sx1 - fsx1 > 1e-3) {
      sum = sum + input[sx1 - 1] * ((sx1 - fsx1));
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      sum = sum + input[dx];
    }

    if (fsx2 - sx2 > 1e-3) {
      sum = sum + input[sx2] * ((fsx2 - sx2));
    }
    input += src_stride;
  }

  if (fsy2 - sy2 > 1e-3) {
    if (sx1 - fsx1 > 1e-3) {
      sum = sum + input[sx1 - 1] * (fsy2 - sy2) * (sx1 - fsx1);
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      sum = sum + input[dx] * (fsy2 - sy2);
    }

    if (fsx2 - sx2 > 1e-3) {
      sum = sum + input[sx2] * (fsy2 - sy2) * (fsx2 - sx2);
    }
  }
  sum = sum / area;

  T* output = (T*)(dst + element_y * dst_stride);
  if (sizeof(T) == 1) {
    output[element_x] = saturate_cast(sum);
  }
  else {
    output[element_x] = sum;
  }
}

template <typename T0, typename T1>
__global__
void resizeAreaKernel1C2(const T1* src, int src_rows, int src_cols,
                         int channels, int src_stride, T1* dst, int dst_rows,
                         int dst_cols, int dst_stride, float col_scale,
                         float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float fsy1 = element_y * row_scale;
  float fsy2 = fsy1 + row_scale;
  int sy1 = ceilf(fsy1);
  int sy2 = floorf(fsy2);

  float fsx1 = element_x * col_scale;
  float fsx2 = fsx1 + col_scale;
  int sx1 = ceilf(fsx1);
  int sx2 = floorf(fsx2);

  T0* input;
  float2 value;
  float2 sum = make_float2(0.f, 0.f);
  float area = fminf(col_scale, src_cols - fsx1) *
               fminf(row_scale, src_rows - fsy1);

  if (sy1 - fsy1 > 1e-3) {
    input = (T0*)(src + (sy1 - 1) * src_stride);
    if (sx1 - fsx1 > 1e-3) {
      value = (sy1 - fsy1) * (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      value = (sy1 - fsy1) * input[dx];
      sum += value;
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (sy1 - fsy1) * (fsx2 - sx2) * input[sx2];
      sum += value;
    }
  }

  input = (T0*)(src + sy1 * src_stride);
  for (int dy = sy1; dy < sy2; ++dy) {
    if (sx1 - fsx1 > 1e-3) {
      value = (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      sum += input[dx];
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (fsx2 - sx2) * input[sx2];
      sum += value;
    }
    input = (T0*)((T1*)input + src_stride);
  }

  if (fsy2 - sy2 > 1e-3) {
    if (sx1 - fsx1 > 1e-3) {
      value = (fsy2 - sy2) * (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      value = (fsy2 - sy2) * input[dx];
      sum += value;
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (fsy2 - sy2) * (fsx2 - sx2) * input[sx2];
      sum += value;
    }
  }
  sum /= area;

  T0* output = (T0*)(dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<T0, float2>(sum);
}

template <typename T0, typename T1>
__global__
void resizeAreaKernel1C3(const T1* src, int src_rows, int src_cols,
                         int channels, int src_stride, T1* dst, int dst_rows,
                         int dst_cols, int dst_stride, float col_scale,
                         float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float fsy1 = element_y * row_scale;
  float fsy2 = fsy1 + row_scale;
  int sy1 = ceilf(fsy1);
  int sy2 = floorf(fsy2);

  float fsx1 = element_x * col_scale;
  float fsx2 = fsx1 + col_scale;
  int sx1 = ceilf(fsx1);
  int sx2 = floorf(fsx2);

  T0* input;
  float3 value;
  float3 sum = make_float3(0.f, 0.f, 0.f);
  float area = fminf(col_scale, src_cols - fsx1) *
               fminf(row_scale, src_rows - fsy1);

  if (sy1 - fsy1 > 1e-3) {
    input = (T0*)(src + (sy1 - 1) * src_stride);
    if (sx1 - fsx1 > 1e-3) {
      value = (sy1 - fsy1) * (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      value = (sy1 - fsy1) * input[dx];
      sum += value;
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (sy1 - fsy1) * (fsx2 - sx2) * input[sx2];
      sum += value;
    }
  }

  input = (T0*)(src + sy1 * src_stride);
  for (int dy = sy1; dy < sy2; ++dy) {
    if (sx1 - fsx1 > 1e-3) {
      value = (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      sum += input[dx];
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (fsx2 - sx2) * input[sx2];
      sum += value;
    }
    input = (T0*)((T1*)input + src_stride);
  }

  if (fsy2 - sy2 > 1e-3) {
    if (sx1 - fsx1 > 1e-3) {
      value = (fsy2 - sy2) * (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      value = (fsy2 - sy2) * input[dx];
      sum += value;
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (fsy2 - sy2) * (fsx2 - sx2) * input[sx2];
      sum += value;
    }
  }
  sum /= area;

  T0* output = (T0*)(dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<T0, float3>(sum);
}

template <typename T0, typename T1>
__global__
void resizeAreaKernel1C4(const T1* src, int src_rows, int src_cols,
                         int channels, int src_stride, T1* dst, int dst_rows,
                         int dst_cols, int dst_stride, float col_scale,
                         float row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float fsy1 = element_y * row_scale;
  float fsy2 = fsy1 + row_scale;
  int sy1 = ceilf(fsy1);
  int sy2 = floorf(fsy2);

  float fsx1 = element_x * col_scale;
  float fsx2 = fsx1 + col_scale;
  int sx1 = ceilf(fsx1);
  int sx2 = floorf(fsx2);

  T0* input;
  float4 value;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  float area = fminf(col_scale, src_cols - fsx1) *
               fminf(row_scale, src_rows - fsy1);

  if (sy1 - fsy1 > 1e-3) {
    input = (T0*)(src + (sy1 - 1) * src_stride);
    if (sx1 - fsx1 > 1e-3) {
      value = (sy1 - fsy1) * (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      value = (sy1 - fsy1) * input[dx];
      sum += value;
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (sy1 - fsy1) * (fsx2 - sx2) * input[sx2];
      sum += value;
    }
  }

  input = (T0*)(src + sy1 * src_stride);
  for (int dy = sy1; dy < sy2; ++dy) {
    if (sx1 - fsx1 > 1e-3) {
      value = (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      sum += input[dx];
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (fsx2 - sx2) * input[sx2];
      sum += value;
    }
    input = (T0*)((T1*)input + src_stride);
  }

  if (fsy2 - sy2 > 1e-3) {
    if (sx1 - fsx1 > 1e-3) {
      value = (fsy2 - sy2) * (sx1 - fsx1) * input[sx1 - 1];
      sum += value;
    }

    for (int dx = sx1; dx < sx2; ++dx) {
      value = (fsy2 - sy2) * input[dx];
      sum += value;
    }

    if (fsx2 - sx2 > 1e-3) {
      value = (fsy2 - sy2) * (fsx2 - sx2) * input[sx2];
      sum += value;
    }
  }
  sum /= area;

  T0* output = (T0*)(dst + element_y * dst_stride);
  output[element_x] = saturate_cast_vector<T0, float4>(sum);
}

__global__
void resizeAreaKernel2(const uchar* src, int src_rows, int src_cols,
                       int channels, int src_stride, uchar* dst, int dst_rows,
                       int dst_cols, int dst_stride, float col_scale,
                       float row_scale, float inv_col_scale,
                       float inv_row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int sy = floor(element_y * row_scale);
  int sx = floor(element_x * col_scale);
  float fy = element_y + 1 - (sy + 1) * inv_row_scale;
  float fx = element_x + 1 - (sx + 1) * inv_col_scale;
  fy = fy <= 0 ? 0.f : fy - floor(fy);
  fx = fx <= 0 ? 0.f : fx - floor(fx);
  if (sy < 0) {
    sy = 0;
    fy = 0;
  }
  if (sx < 0) {
    sx = 0;
    fx = 0;
  }
  if (sy >= src_rows) {
    sy = src_rows - 1;
    fy = 0;
  }
  if (sx >= src_cols) {
    sx = src_cols - 1;
    fx = 0;
  }

  int sy_ = INC(sy, src_rows);
  int cbufy[2];
  fy = fy * INTER_RESIZE_COEF_SCALE;
  cbufy[0] = rint(INTER_RESIZE_COEF_SCALE - fy);
  cbufy[1] = rint(fy);

  int sx_ = INC(sx, src_cols);
  int cbufx[2];
  fx = fx * INTER_RESIZE_COEF_SCALE;
  cbufx[0] = rint(INTER_RESIZE_COEF_SCALE - rint(fx));
  cbufx[1] = rint(fx);

  if (channels == 1) {
    int src_index0 = sy * src_stride + sx;
    int src_index1 = sy * src_stride + sx_;
    int src_index2 = sy_ * src_stride + sx;
    int src_index3 = sy_ * src_stride + sx_;
    int dst_index = element_y * dst_stride + element_x;

    int sum = 0;
    sum = cbufy[0] * cbufx[0] * src[src_index0] +
          cbufy[0] * cbufx[1] * src[src_index1] +
          cbufy[1] * cbufx[0] * src[src_index2] +
          cbufy[1] * cbufx[1] * src[src_index3];
    dst[dst_index] = (sum + (1 << (CAST_BITS - 1))) >> CAST_BITS;
  }
  else if (channels == 2) {
    uchar2* input0 = (uchar2*)((uchar*)src + sy * src_stride);
    uchar2* input1 = (uchar2*)((uchar*)src + sy_ * src_stride);
    uchar2* output = (uchar2*)((uchar*)dst + element_y * dst_stride);

    uchar2 t[2][2];
    t[0][0] = input0[sx];
    t[0][1] = input0[sx_];
    t[1][0] = input1[sx];
    t[1][1] = input1[sx_];

    output[element_x] = bilinearSampleUchar(t, cbufx[0], cbufx[1], cbufy[0],
                                            cbufy[1]);
  }
  else if (channels == 3) {
    uchar3* input0 = (uchar3*)((uchar*)src + sy * src_stride);
    uchar3* input1 = (uchar3*)((uchar*)src + sy_ * src_stride);
    uchar3* output = (uchar3*)((uchar*)dst + element_y * dst_stride);

    uchar3 t[2][2];
    t[0][0] = input0[sx];
    t[0][1] = input0[sx_];
    t[1][0] = input1[sx];
    t[1][1] = input1[sx_];

    output[element_x] = bilinearSampleUchar(t, cbufx[0], cbufx[1], cbufy[0],
                                            cbufy[1]);
  }
  else {
    uchar4* input0 = (uchar4*)((uchar*)src + sy * src_stride);
    uchar4* input1 = (uchar4*)((uchar*)src + sy_ * src_stride);
    uchar4* output = (uchar4*)((uchar*)dst + element_y * dst_stride);

    uchar4 t[2][2];
    t[0][0] = input0[sx];
    t[0][1] = input0[sx_];
    t[1][0] = input1[sx];
    t[1][1] = input1[sx_];

    output[element_x] = bilinearSampleUchar(t, cbufx[0], cbufx[1], cbufy[0],
                                            cbufy[1]);
  }
}

__global__
void resizeAreaKernel2(const float* src, int src_rows, int src_cols,
                       int channels, int src_stride, float* dst, int dst_rows,
                       int dst_cols, int dst_stride, double col_scale,
                       float row_scale, float inv_col_scale,
                       float inv_row_scale) {
  int element_x = blockIdx.x * blockDim.x + threadIdx.x;
  int element_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int sy = floor(element_y * row_scale);
  int sx = floor(element_x * col_scale);
  float fy = element_y + 1 - (sy + 1) * inv_row_scale;
  float fx = element_x + 1 - (sx + 1) * inv_col_scale;
  fy = fy <= 0 ? 0.f : fy - floor(fy);
  fx = fx <= 0 ? 0.f : fx - floor(fx);
  if (sy < 0) {
    sy = 0;
    fy = 0;
  }
  if (sx < 0) {
    sx = 0;
    fx = 0;
  }
  if (sy >= src_rows) {
    sy = src_rows - 1;
    fy = 0;
  }
  if (sx >= src_cols) {
    sx = src_cols - 1;
    fx = 0;
  }

  int sy_ = INC(sy,src_rows);
  float cbufy[2];
  cbufy[0] = 1.f - fy;
  cbufy[1] = 1.f - cbufy[0];

  int sx_ = INC(sx,src_cols);
  float cbufx[2];
  cbufx[0] = 1.f - fx;
  cbufx[1] = 1.f - cbufx[0];

  if (channels == 1) {
    int index = sy * src_stride;
    float src1 = src[index + sx];
    float src2 = src[index + sx_];
    float value1 = cbufy[0] * cbufx[0] * src1;
    float value2 = cbufy[0] * cbufx[1] * src2;
    float sum = 0.f;
    sum += value1 + value2;

    index = sy_ * src_stride;
    src1 = src[index + sx];
    src2 = src[index + sx_];
    value1 = cbufy[1] * cbufx[0] * src1;
    value2 = cbufy[1] * cbufx[1] * src2;
    sum += value1 + value2;

    index = element_y * dst_stride + element_x;
    dst[index] = sum;
  }
  else if (channels == 3) {
    int index = sy * src_stride;
    float3 src1 = ((float3*)(src + index))[sx];
    float3 src2 = ((float3*)(src + index))[sx_];
    float3 value1 = cbufy[0] * cbufx[0] * src1;
    float3 value2 = cbufy[0] * cbufx[1] * src2;
    float3 sum = make_float3(0.f, 0.f, 0.f);
    sum += value1;
    sum += value2;

    index = sy_ * src_stride;
    src1 = ((float3*)(src + index))[sx];
    src2 = ((float3*)(src + index))[sx_];
    value1 = cbufy[1] * cbufx[0] * src1;
    value2 = cbufy[1] * cbufx[1] * src2;
    sum += value1;
    sum += value2;

    float3* output = (float3*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
  else {
    int index = sy * src_stride;
    float4 src1 = ((float4*)(src + index))[sx];
    float4 src2 = ((float4*)(src + index))[sx_];
    float4 value1 = cbufy[0] * cbufx[0] * src1;
    float4 value2 = cbufy[0] * cbufx[1] * src2;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    sum += value1;
    sum += value2;

    index = sy_ * src_stride;
    src1 = ((float4*)(src + index))[sx];
    src2 = ((float4*)(src + index))[sx_];
    value1 = cbufy[1] * cbufx[0] * src1;
    value2 = cbufy[1] * cbufx[1] * src2;
    sum += value1;
    sum += value2;

    float4* output = (float4*)(dst + element_y * dst_stride);
    output[element_x] = sum;
  }
}

RetCode resizeArea(const uchar* src, int src_rows, int src_cols, int channels,
                   int src_stride, uchar* dst, int dst_rows, int dst_cols,
                   int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);

  cudaError_t code = cudaSuccess;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, src_rows * src_stride * sizeof(uchar),
                             cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  const int kBlockX = 32;
  const int kBlockY = 16;
  dim3 block(kBlockX, kBlockY);
  dim3 grid;
  grid.x = (dst_cols + kBlockX -1) / kBlockX;
  grid.y = (dst_rows + kBlockY - 1) / kBlockY;

  float col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;
  float inv_col_scale = 1.0 / col_scale;
  float inv_row_scale = 1.0 / row_scale;

  if (src_cols > dst_cols && src_rows > dst_rows) {
    if (src_cols % dst_cols == 0 && src_rows % dst_rows == 0) {
      if (channels == 1) {
        resizeAreaKernel0C1<uchar><<<grid, block, 0, stream>>>(src, src_rows,
            src_cols, channels, src_stride, dst, dst_rows, dst_cols, dst_stride,
            col_scale, row_scale);
      }
      else if (channels == 3) {
        resizeAreaKernel0C3<uchar3, uchar><<<grid, block, 0, stream>>>(src,
            src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
            dst_stride, col_scale, row_scale);
      }
      else {
        resizeAreaKernel0C4<uchar4, uchar><<<grid, block, 0, stream>>>(src,
            src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
            dst_stride, col_scale, row_scale);
      }
    }
    else {
      if (channels == 1) {
        resizeAreaKernel1C1<uchar><<<grid, block, 0, stream>>>(src, src_rows,
            src_cols, channels, src_stride, dst, dst_rows, dst_cols, dst_stride,
            col_scale, row_scale);
      }
      else if (channels == 3) {
        resizeAreaKernel1C3<uchar3, uchar><<<grid, block, 0, stream>>>(src,
            src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
            dst_stride, col_scale, row_scale);
      }
      else {
        resizeAreaKernel1C4<uchar4, uchar><<<grid, block, 0, stream>>>(src,
            src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
            dst_stride, col_scale, row_scale);
      }
    }
  }
  else {
    resizeAreaKernel2<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
        channels, src_stride, dst, dst_rows, dst_cols, dst_stride, col_scale,
        row_scale, inv_col_scale, inv_row_scale);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode resizeArea(const float* src, int src_rows, int src_cols, int channels,
                   int src_stride, float* dst, int dst_rows, int dst_cols,
                   int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);

  cudaError_t code = cudaSuccess;
  if (src_rows == dst_rows && src_cols == dst_cols &&
      src_stride == dst_stride) {
    if (src != dst) {
      code = cudaMemcpyAsync(dst, src, src_rows * src_stride * sizeof(float),
                             cudaMemcpyDeviceToDevice);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return RC_DEVICE_MEMORY_ERROR;
      }
    }
    return RC_SUCCESS;
  }

  const int kBlockX = 32;
  const int kBlockY = 16;
  dim3 block(kBlockX, kBlockY);
  dim3 grid;
  grid.x = (dst_cols + kBlockX -1) / kBlockX;
  grid.y = (dst_rows + kBlockY - 1) / kBlockY;

  double col_scale = (double)src_cols / dst_cols;
  float row_scale = (double)src_rows / dst_rows;
  float inv_col_scale = 1.0 / col_scale;
  float inv_row_scale = 1.0 / row_scale;

  if (src_cols > dst_cols && src_rows > dst_rows) {
    if (src_cols % dst_cols == 0 && src_rows % dst_rows == 0) {
      if (channels == 1) {
        resizeAreaKernel0C1<float><<<grid, block, 0, stream>>>(src, src_rows,
            src_cols, channels, src_stride, dst, dst_rows, dst_cols, dst_stride,
            col_scale, row_scale);
      }
      else if (channels == 3) {
        resizeAreaKernel0C3<float3, float><<<grid, block, 0, stream>>>(src,
            src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
            dst_stride, col_scale, row_scale);
      }
      else {
        resizeAreaKernel0C4<float4, float><<<grid, block, 0, stream>>>(src,
            src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
            dst_stride, col_scale, row_scale);
      }
    }
    else {
      if (channels == 1) {
        resizeAreaKernel1C1<float><<<grid, block, 0, stream>>>(src, src_rows,
            src_cols, channels, src_stride, dst, dst_rows, dst_cols, dst_stride,
            col_scale, row_scale);
      }
      else if (channels == 3) {
        resizeAreaKernel1C3<float3, float><<<grid, block, 0, stream>>>(src,
            src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
            dst_stride, col_scale, row_scale);
      }
      else {
        resizeAreaKernel1C4<float4, float><<<grid, block, 0, stream>>>(src,
            src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
            dst_stride, col_scale, row_scale);
      }
    }
  }
  else {
    resizeAreaKernel2<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
        channels, src_stride, dst, dst_rows, dst_cols, dst_stride, col_scale,
        row_scale, inv_col_scale, inv_row_scale);
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode ResizeArea<uchar, 1>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const uchar* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             uchar* outData) {
  RetCode code = resizeArea(inData, inHeight, inWidth, 1, inWidthStride,
                            outData, outHeight, outWidth, outWidthStride,
                            stream);

  return code;
}

template <>
RetCode ResizeArea<uchar, 3>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const uchar* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             uchar* outData) {
  RetCode code = resizeArea(inData, inHeight, inWidth, 3, inWidthStride,
                            outData, outHeight, outWidth, outWidthStride,
                            stream);

  return code;
}

template <>
RetCode ResizeArea<uchar, 4>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const uchar* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             uchar* outData) {
  RetCode code = resizeArea(inData, inHeight, inWidth, 4, inWidthStride,
                            outData, outHeight, outWidth, outWidthStride,
                            stream);

  return code;
}

template <>
RetCode ResizeArea<float, 1>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const float* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             float* outData) {
  RetCode code = resizeArea(inData, inHeight, inWidth, 1, inWidthStride,
                            outData, outHeight, outWidth, outWidthStride,
                            stream);

  return code;
}

template <>
RetCode ResizeArea<float, 3>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const float* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             float* outData) {
  RetCode code = resizeArea(inData, inHeight, inWidth, 3, inWidthStride,
                            outData, outHeight, outWidth, outWidthStride,
                            stream);

  return code;
}

template <>
RetCode ResizeArea<float, 4>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const float* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             float* outData) {
  RetCode code = resizeArea(inData, inHeight, inWidth, 4, inWidthStride,
                            outData, outHeight, outWidth, outWidthStride,
                            stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
