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

#include "ppl/cv/cuda/remap.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

__global__
void remapLinearKernel(int src_rows, int src_cols, int src_stride,
                       const uchar* src, int dst_rows, int dst_cols,
                       int channels, int dst_stride, uchar* dst,
                       const float* map_x, const float* map_y,
                       BorderType border_type, uchar border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int idxXY = element_y * dst_cols + element_x;
  int idxDst = element_y * dst_stride + element_x * channels;
  float mapx = map_x[idxXY];
  float mapy = map_y[idxXY];
  int sx0 = (int)(mapx);
  int sy0 = (int)(mapy);
  float ax0 = (float)(mapx-sx0);
  float ay0 = (float)(mapy-sy0);

  float v0, v1, v2, v3;
  float tab[4];
  float taby[2], tabx[2];
  taby[0] = 1.0f - ay0;  //1-u
  taby[1] = ay0;		     //u
  tabx[0] = 1.0f - ax0;  //1-v
  tabx[1] = ax0;		     //v
  tab[0] = taby[0] * tabx[0];
  tab[1] = taby[0] * tabx[1];
  tab[2] = taby[1] * tabx[0];
  tab[3] = taby[1] * tabx[1];

  if (border_type == BORDER_TYPE_CONSTANT) {
    bool flag0 = sx0 >= 0 && sx0 < src_cols && sy0 >= 0 && sy0 < src_rows;
    bool flag1 = sx0+1 >= 0 && sx0+1 < src_cols && sy0 >= 0 && sy0 < src_rows;
    bool flag2 = sx0 >= 0 && sx0 < src_cols && sy0+1 >= 0 && sy0+1 < src_rows;
    bool flag3 = sx0+1 >= 0 && sx0+1 < src_cols && sy0+1 >= 0 &&
                  sy0+1 < src_rows;
    int position1 = sy0 * src_stride + sx0*channels;
    int position2 = (sy0+1) * src_stride + sx0*channels;
    for (int k = 0; k < channels; k++) {
      v0 = flag0 ? src[position1 + k] : border_value;
      v1 = flag1 ? src[position1+ channels + k] : border_value;
      v2 = flag2 ? src[position2 + k] : border_value;
      v3 = flag3 ? src[position2+ channels + k] : border_value;
      float sum = v0 * tab[0] +  v1 * tab[1] +  v2 * tab[2] +  v3 * tab[3];
      dst[idxDst + k] = saturate_cast(sum);
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    int sx1 = sx0 + 1;
    int sy1 = sy0 + 1;
    sx0 = clip(sx0, 0, src_cols - 1);
    sx1 = clip(sx1, 0, src_cols - 1);
    sy0 = clip(sy0, 0, src_rows - 1);
    sy1 = clip(sy1, 0, src_rows - 1);
    const uchar* t0 = src + sy0 * src_stride + sx0 * channels;
    const uchar* t1 = src + sy0 * src_stride + sx1 * channels;
    const uchar* t2 = src + sy1 * src_stride + sx0 * channels;
    const uchar* t3 = src + sy1 * src_stride + sx1 * channels;
    for (int k = 0; k < channels; ++k) {
      float sum = t0[k] * tab[0] + t1[k] * tab[1] + t2[k] * tab[2] +
                  t3[k] * tab[3];
      dst[idxDst + k] = saturate_cast(sum);
    }
  }
  else if (border_type == BORDER_TYPE_TRANSPARENT) {
    bool flag0 = sx0 >= 0 && sx0 < src_cols && sy0 >= 0 && sy0 < src_rows;
    bool flag1 = sx0+1 >= 0 && sx0+1 < src_cols && sy0 >= 0 && sy0 < src_rows;
    bool flag2 = sx0 >= 0 && sx0 < src_cols && sy0+1 >= 0 && sy0+1 < src_rows;
    bool flag3 = sx0+1 >= 0 && sx0+1 < src_cols && sy0+1 >= 0 &&
                  sy0+1 < src_rows;
    if (flag0 && flag1 && flag2 && flag3) {
      int position1 = (sy0 * src_stride + sx0 * channels);
      int position2 = ((sy0+1) * src_stride + sx0 * channels);
      for (int k = 0; k < channels; k++) {
        v0 = src[position1 + k];
        v1 = src[position1 + channels + k];
        v2 = src[position2 + k];
        v3 = src[position2 + channels +k];
        float sum = v0 * tab[0] +  v1 * tab[1] +  v2 * tab[2] +  v3 * tab[3];
        dst[idxDst + k] = saturate_cast(sum);
      }
    }
  }
  else {
  }
}

__global__
void remapLinearKernel(int src_rows, int src_cols, int src_stride,
                       const float* src, int dst_rows, int dst_cols,
                       int channels, int dst_stride, float* dst,
                       const float* map_x, const float* map_y,
                       BorderType border_type, float border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int idxXY = element_y * dst_cols + element_x;
  int idxDst = element_y * dst_stride + element_x * channels;
  float mapx = map_x[idxXY];
  float mapy = map_y[idxXY];
  int sx0 = (int)(mapx);
  int sy0 = (int)(mapy);
  float ax0 = (float)(mapx-sx0);
  float ay0 = (float)(mapy-sy0);

  float tab[4];
  float taby[2], tabx[2];
  float v0, v1, v2, v3;
  taby[0] = 1.0f - ay0;  //1-u
  taby[1] = ay0;		     //u
  tabx[0] = 1.0f - ax0;  //1-v
  tabx[1] = ax0;		     //v
  tab[0] = taby[0] * tabx[0];
  tab[1] = taby[0] * tabx[1];
  tab[2] = taby[1] * tabx[0];
  tab[3] = taby[1] * tabx[1];

  if (border_type == BORDER_TYPE_CONSTANT) {
    bool flag0 = sx0 >= 0 && sx0 < src_cols && sy0 >= 0 && sy0 < src_rows;
    bool flag1 = sx0+1 >= 0 && sx0+1 < src_cols && sy0 >= 0 && sy0 < src_rows;
    bool flag2 = sx0 >= 0 && sx0 < src_cols && sy0+1 >= 0 && sy0+1 < src_rows;
    bool flag3 = sx0+1 >= 0 && sx0+1 < src_cols && sy0+1 >= 0 &&
                  sy0+1 < src_rows;
    int position1 = sy0 * src_stride + sx0*channels;
    int position2 = (sy0+1) * src_stride + sx0*channels;
    for (int k = 0; k < channels; k++) {
      v0 = flag0 ? src[position1 + k] : border_value;
      v1 = flag1 ? src[position1+ channels + k] : border_value;
      v2 = flag2 ? src[position2 + k] : border_value;
      v3 = flag3 ? src[position2+ channels + k] : border_value;
      float sum = v0 * tab[0] +  v1 * tab[1] +  v2 * tab[2] +  v3 * tab[3];
      dst[idxDst + k] = sum;
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    int sx1 = sx0 + 1;
    int sy1 = sy0 + 1;
    sx0 = clip(sx0, 0, src_cols - 1);
    sx1 = clip(sx1, 0, src_cols - 1);
    sy0 = clip(sy0, 0, src_rows - 1);
    sy1 = clip(sy1, 0, src_rows - 1);
    const float* t0 = src + sy0 * src_stride + sx0 * channels;
    const float* t1 = src + sy0 * src_stride + sx1 * channels;
    const float* t2 = src + sy1 * src_stride + sx0 * channels;
    const float* t3 = src + sy1 * src_stride + sx1 * channels;
    for (int k = 0; k < channels; ++k) {
      float sum = t0[k] * tab[0] + t1[k] * tab[1] + t2[k] * tab[2] +
                  t3[k] * tab[3];
      dst[idxDst + k] = sum;
    }
  }
  else if (border_type == BORDER_TYPE_TRANSPARENT) {
    bool flag0 = sx0 >= 0 && sx0 < src_cols && sy0 >= 0 && sy0 < src_rows;
    bool flag1 = sx0+1 >= 0 && sx0+1 < src_cols && sy0 >= 0 && sy0 < src_rows;
    bool flag2 = sx0 >= 0 && sx0 < src_cols && sy0+1 >= 0 && sy0+1 < src_rows;
    bool flag3 = sx0+1 >= 0 && sx0+1 < src_cols && sy0+1 >= 0 &&
                 sy0+1 < src_rows;
    if (flag0 && flag1 && flag2 && flag3) {
      int position1 = (sy0 * src_stride + sx0 * channels);
      int position2 = ((sy0+1) * src_stride + sx0 * channels);
      for (int k = 0; k < channels; k++) {
        v0 = src[position1 + k];
        v1 = src[position1 + channels + k];
        v2 = src[position2 + k];
        v3 = src[position2 + channels +k];
        float sum = v0 * tab[0] +  v1 * tab[1] +  v2 * tab[2] +  v3 * tab[3];
        dst[idxDst + k] = sum;
      }
    }
  }
  else {
  }
}

RetCode RemapLinear(int src_rows, int src_cols, int src_stride,
                    const uchar* src, int dst_rows, int dst_cols, int channels,
                    int dst_stride, uchar* dst, const float* map_x,
                    const float* map_y, BorderType border_type,
                    uchar border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(map_x != nullptr);
  PPL_ASSERT(map_y != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(border_type == BORDER_TYPE_CONSTANT ||
             border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  remapLinearKernel<<<grid, block, 0, stream>>>(src_rows, src_cols, src_stride,
                                                src, dst_rows, dst_cols,
                                                channels, dst_stride, dst,
                                                map_x, map_y, border_type,
                                                border_value);
}

RetCode RemapLinear(int src_rows, int src_cols, int src_stride,
                    const float* src, int dst_rows, int dst_cols, int channels,
                    int dst_stride, float* dst, const float* map_x,
                    const float* map_y, BorderType border_type,
                    float border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(map_x != nullptr);
  PPL_ASSERT(map_y != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(border_type == BORDER_TYPE_CONSTANT ||
             border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  remapLinearKernel<<<grid, block, 0, stream>>>(src_rows, src_cols, src_stride,
                                                src, dst_rows, dst_cols,
                                                channels, dst_stride, dst,
                                                map_x, map_y, border_type,
                                                border_value);
}

template <>
RetCode RemapLinear<uchar, 1>(cudaStream_t stream,
                              int inHeight,
                              int inWidth,
                              int inWidthStride,
                              const uchar* inData,
                              int outHeight,
                              int outWidth,
                              int outWidthStride,
                              uchar* outData,
                              const float* mapX,
                              const float* mapY,
                              BorderType border_type,
                              uchar borderValue) {
  RetCode code = RemapLinear(inHeight, inWidth, inWidthStride, inData,
                             outHeight, outWidth, 1, outWidthStride, outData,
                             mapX, mapY, border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapLinear<uchar, 3>(cudaStream_t stream,
                              int inHeight,
                              int inWidth,
                              int inWidthStride,
                              const uchar* inData,
                              int outHeight,
                              int outWidth,
                              int outWidthStride,
                              uchar* outData,
                              const float* mapX,
                              const float* mapY,
                              BorderType border_type,
                              uchar borderValue) {
  RetCode code = RemapLinear(inHeight, inWidth, inWidthStride, inData,
                             outHeight, outWidth, 3, outWidthStride, outData,
                             mapX, mapY, border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapLinear<uchar, 4>(cudaStream_t stream,
                              int inHeight,
                              int inWidth,
                              int inWidthStride,
                              const uchar* inData,
                              int outHeight,
                              int outWidth,
                              int outWidthStride,
                              uchar* outData,
                              const float* mapX,
                              const float* mapY,
                              BorderType border_type,
                              uchar borderValue) {
  RetCode code = RemapLinear(inHeight, inWidth, inWidthStride, inData,
                             outHeight, outWidth, 4, outWidthStride, outData,
                             mapX, mapY, border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapLinear<float, 1>(cudaStream_t stream,
                              int inHeight,
                              int inWidth,
                              int inWidthStride,
                              const float* inData,
                              int outHeight,
                              int outWidth,
                              int outWidthStride,
                              float* outData,
                              const float* mapX,
                              const float* mapY,
                              BorderType border_type,
                              float borderValue) {
  RetCode code = RemapLinear(inHeight, inWidth, inWidthStride, inData,
                             outHeight, outWidth, 1, outWidthStride, outData,
                             mapX, mapY, border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapLinear<float, 3>(cudaStream_t stream,
                              int inHeight,
                              int inWidth,
                              int inWidthStride,
                              const float* inData,
                              int outHeight,
                              int outWidth,
                              int outWidthStride,
                              float* outData,
                              const float* mapX,
                              const float* mapY,
                              BorderType border_type,
                              float borderValue) {
  RetCode code = RemapLinear(inHeight, inWidth, inWidthStride, inData,
                             outHeight, outWidth, 3, outWidthStride, outData,
                             mapX, mapY, border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapLinear<float, 4>(cudaStream_t stream,
                              int inHeight,
                              int inWidth,
                              int inWidthStride,
                              const float* inData,
                              int outHeight,
                              int outWidth,
                              int outWidthStride,
                              float* outData,
                              const float* mapX,
                              const float* mapY,
                              BorderType border_type,
                              float borderValue) {
  RetCode code = RemapLinear(inHeight, inWidth, inWidthStride, inData,
                             outHeight, outWidth, 4, outWidthStride, outData,
                             mapX, mapY, border_type, borderValue, stream);

  return code;
}

__global__
void remapNPKernel(int src_rows, int src_cols, int src_stride,
                   const uchar* src, int dst_rows, int dst_cols, int channels,
                   int dst_stride, uchar* dst, const float* map_x,
                   const float* map_y, BorderType border_type,
                   uchar border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int idxXY = element_y * dst_cols + element_x;
  int idxDst = element_y * dst_stride + element_x * channels;
  float mapx = map_x[idxXY];
  float mapy = map_y[idxXY];
  int sx = __float2int_rn(mapx);
  int sy = __float2int_rn(mapy);

  if (border_type == BORDER_TYPE_CONSTANT) {
    int idxSrc = sy * src_stride + sx * channels;
    if (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) {
      for (int k = 0; k < channels; k++) {
        dst[idxDst + k] = src[idxSrc + k];
      }
    }
    else {
      for (int k = 0; k < channels; k++) {
        dst[idxDst + k] = border_value;
      }
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    sx = clip(sx, 0, src_cols - 1);
    sy = clip(sy, 0, src_rows - 1);
    int idxSrc = sy * src_stride + sx * channels;
    for (int k = 0; k < channels; ++k) {
      dst[idxDst + k] = src[idxSrc + k];
    }
  }
  else if (border_type == BORDER_TYPE_TRANSPARENT) {
    if (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) {
      int idxSrc = sy * src_stride + sx * channels;
      for (int k = 0; k < channels; k++) {
        dst[idxDst + k] = src[idxSrc + k];
      }
    }
  }
  else {
  }
}

__global__
void remapNPKernel(int src_rows, int src_cols, int src_stride,
                   const float* src, int dst_rows, int dst_cols, int channels,
                   int dst_stride, float* dst, const float* map_x,
                   const float* map_y, BorderType border_type,
                   float border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  int idxXY = element_y * dst_cols + element_x;
  int idxDst = element_y * dst_stride + element_x * channels;
  float mapx = map_x[idxXY];
  float mapy = map_y[idxXY];
  int sx = __float2int_rn(mapx);
  int sy = __float2int_rn(mapy);

  if (border_type == BORDER_TYPE_CONSTANT) {
    int idxSrc = sy * src_stride + sx * channels;
    if (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) {
      for (int k = 0; k < channels; k++) {
        dst[idxDst + k] = src[idxSrc + k];
      }
    }
    else {
      for (int k = 0; k < channels; k++) {
        dst[idxDst + k] = border_value;
      }
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    sx = clip(sx, 0, src_cols - 1);
    sy = clip(sy, 0, src_rows - 1);
    int idxSrc = sy * src_stride + sx * channels;
    for (int k = 0; k < channels; ++k) {
      dst[idxDst + k] = src[idxSrc + k];
    }
  }
  else if (border_type == BORDER_TYPE_TRANSPARENT) {
    if (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) {
      int idxSrc = sy * src_stride + sx * channels;
      for (int k = 0; k < channels; k++) {
        dst[idxDst + k] = src[idxSrc + k];
      }
    }
  }
  else {
  }
}

RetCode RemapNP(int src_rows, int src_cols, int src_stride, const uchar* src,
                int dst_rows, int dst_cols, int channels, int dst_stride,
                uchar* dst, const float* map_x, const float* map_y,
                BorderType border_type, uchar border_value,
                cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(map_x != nullptr);
  PPL_ASSERT(map_y != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(border_type == BORDER_TYPE_CONSTANT ||
             border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  remapNPKernel<<<grid, block, 0, stream>>>(src_rows, src_cols, src_stride,
                                            src, dst_rows, dst_cols, channels,
                                            dst_stride, dst, map_x, map_y,
                                            border_type, border_value);
}

RetCode RemapNP(int src_rows, int src_cols, int src_stride, const float* src,
                int dst_rows, int dst_cols, int channels, int dst_stride,
                float* dst, const float* map_x, const float* map_y,
                BorderType border_type, float border_value,
                cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(map_x != nullptr);
  PPL_ASSERT(map_y != nullptr);
  PPL_ASSERT(src_rows > 0 && src_cols > 0);
  PPL_ASSERT(dst_rows > 0 && dst_cols > 0);
  PPL_ASSERT(src_stride >= src_cols * channels);
  PPL_ASSERT(dst_stride >= dst_cols * channels);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(border_type == BORDER_TYPE_CONSTANT ||
             border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  remapNPKernel<<<grid, block, 0, stream>>>(src_rows, src_cols, src_stride,
                                            src, dst_rows, dst_cols, channels,
                                            dst_stride, dst, map_x, map_y,
                                            border_type, border_value);
}

template <>
RetCode RemapNearestPoint<uchar, 1>(cudaStream_t stream,
                                    int inHeight,
                                    int inWidth,
                                    int inWidthStride,
                                    const uchar* inData,
                                    int outHeight,
                                    int outWidth,
                                    int outWidthStride,
                                    uchar* outData,
                                    const float* mapX,
                                    const float* mapY,
                                    BorderType border_type,
                                    uchar borderValue) {
  RetCode code = RemapNP(inHeight, inWidth, inWidthStride, inData, outHeight,
                         outWidth, 1, outWidthStride, outData, mapX, mapY,
                         border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapNearestPoint<uchar, 3>(cudaStream_t stream,
                                    int inHeight,
                                    int inWidth,
                                    int inWidthStride,
                                    const uchar* inData,
                                    int outHeight,
                                    int outWidth,
                                    int outWidthStride,
                                    uchar* outData,
                                    const float* mapX,
                                    const float* mapY,
                                    BorderType border_type,
                                    uchar borderValue) {
  RetCode code = RemapNP(inHeight, inWidth, inWidthStride, inData, outHeight,
                         outWidth, 3, outWidthStride, outData, mapX, mapY,
                         border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapNearestPoint<uchar, 4>(cudaStream_t stream,
                                    int inHeight,
                                    int inWidth,
                                    int inWidthStride,
                                    const uchar* inData,
                                    int outHeight,
                                    int outWidth,
                                    int outWidthStride,
                                    uchar* outData,
                                    const float* mapX,
                                    const float* mapY,
                                    BorderType border_type,
                                    uchar borderValue) {
  RetCode code = RemapNP(inHeight, inWidth, inWidthStride, inData, outHeight,
                         outWidth, 4, outWidthStride, outData, mapX, mapY,
                         border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapNearestPoint<float, 1>(cudaStream_t stream,
                                    int inHeight,
                                    int inWidth,
                                    int inWidthStride,
                                    const float* inData,
                                    int outHeight,
                                    int outWidth,
                                    int outWidthStride,
                                    float* outData,
                                    const float* mapX,
                                    const float* mapY,
                                    BorderType border_type,
                                    float borderValue) {
  RetCode code = RemapNP(inHeight, inWidth, inWidthStride, inData, outHeight,
                         outWidth, 1, outWidthStride, outData, mapX, mapY,
                         border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapNearestPoint<float, 3>(cudaStream_t stream,
                                    int inHeight,
                                    int inWidth,
                                    int inWidthStride,
                                    const float* inData,
                                    int outHeight,
                                    int outWidth,
                                    int outWidthStride,
                                    float* outData,
                                    const float* mapX,
                                    const float* mapY,
                                    BorderType border_type,
                                    float borderValue) {
  RetCode code = RemapNP(inHeight, inWidth, inWidthStride, inData, outHeight,
                         outWidth, 3, outWidthStride, outData, mapX, mapY,
                         border_type, borderValue, stream);

  return code;
}

template <>
RetCode RemapNearestPoint<float, 4>(cudaStream_t stream,
                                    int inHeight,
                                    int inWidth,
                                    int inWidthStride,
                                    const float* inData,
                                    int outHeight,
                                    int outWidth,
                                    int outWidthStride,
                                    float* outData,
                                    const float* mapX,
                                    const float* mapY,
                                    BorderType border_type,
                                    float borderValue) {
  RetCode code = RemapNP(inHeight, inWidth, inWidthStride, inData, outHeight,
                         outWidth, 4, outWidthStride, outData, mapX, mapY,
                         border_type, borderValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
