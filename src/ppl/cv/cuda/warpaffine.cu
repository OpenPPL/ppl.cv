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

#include "ppl/cv/cuda/warpaffine.h"
#include "warp.hpp"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

struct AffineTransform {
  float coeffe0;
  float coeffe1;
  float coeffe2;
  float coeffe3;
  float coeffe4;
  float coeffe5;
  float x;
  float y;

  AffineTransform(const float* coefficients) : coeffe0(coefficients[0]),
      coeffe1(coefficients[1]), coeffe2(coefficients[2]), 
      coeffe3(coefficients[3]), coeffe4(coefficients[4]),
      coeffe5(coefficients[5]) {}

  __DEVICE__
  void calculateCoordinates(int element_x, int element_y) {
    x = coeffe0 * element_x + coeffe1 * element_y + coeffe2;
    y = coeffe3 * element_x + coeffe4 * element_y + coeffe5;
  }

  __DEVICE__
  float getX() const {
    return x;
  }

  __DEVICE__
  float getY() const {
    return y;
  }
};
               
RetCode warpAffine(const uchar* src, int src_rows, int src_cols, int channels,
                   int src_stride, const float* affine_matrix, uchar* dst,
                   int dst_rows, int dst_cols, int dst_stride,
                   InterpolationType interpolation, BorderType border_type,
                   uchar border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src != dst);
  PPL_ASSERT(affine_matrix != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(uchar));
  PPL_ASSERT(interpolation == INTERPOLATION_TYPE_LINEAR ||
             interpolation == INTERPOLATION_TYPE_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_TYPE_CONSTANT ||
             border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(dst_cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(dst_rows, kBlockDimY0, kBlockShiftY0);

  cudaError_t code;
  AffineTransform affine_transform(affine_matrix);

  if (interpolation == INTERPOLATION_TYPE_LINEAR) {
    if (channels != 3 && (src_stride & (TEXTURE_ALIGNMENT - 1)) == 0) {
      if (channels == 1) {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
        uchar_c1_ref.normalized = false;
        uchar_c1_ref.filterMode = cudaFilterModeLinear;
        uchar_c1_ref.addressMode[0] = cudaAddressModeClamp;
        uchar_c1_ref.addressMode[1] = cudaAddressModeClamp;
        code = cudaBindTexture2D(0, uchar_c1_ref, src, desc, src_cols, src_rows,
                                 (size_t)src_stride);
      }
      else {  // channels == 4
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        uchar_c4_ref.normalized = false;
        uchar_c4_ref.filterMode = cudaFilterModeLinear;
        uchar_c4_ref.addressMode[0] = cudaAddressModeClamp;
        uchar_c4_ref.addressMode[1] = cudaAddressModeClamp;
        code = cudaBindTexture2D(0, uchar_c4_ref, src, desc, src_cols, src_rows,
                                 (size_t)src_stride);
      }
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA texture error: " << cudaGetErrorString(code);
        return RC_DEVICE_RUNTIME_ERROR;
      }
      
      warpLinearTexKernel<AffineTransform><<<grid, block, 0, stream>>>(src, 
          src_rows, src_cols, channels, src_stride, affine_transform, dst, 
          dst_rows, dst_cols, dst_stride, border_type, border_value);
    }
    else {
      warpLinearKernel<AffineTransform><<<grid, block, 0, stream>>>(src, 
          src_rows, src_cols, channels, src_stride, affine_transform, dst, 
          dst_rows, dst_cols, dst_stride, border_type, border_value);
    }
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      warpNearestKernel<uchar, uchar, AffineTransform><<<grid, block, 0, 
          stream>>>(src, src_rows, src_cols, channels, src_stride, 
          affine_transform, dst, dst_rows, dst_cols, dst_stride, border_type, 
          border_value);
    }
    else if (channels == 3) {
      uchar3 border_value1 = make_uchar3(border_value, border_value,
                                         border_value);
      warpNearestKernel<uchar, uchar3, AffineTransform><<<grid, block, 0, 
          stream>>>(src, src_rows, src_cols, channels, src_stride, 
          affine_transform, dst, dst_rows, dst_cols, dst_stride, border_type, 
          border_value1);
    }
    else {
      uchar4 border_value1 = make_uchar4(border_value, border_value,
                                         border_value, border_value);
      warpNearestKernel<uchar, uchar4, AffineTransform><<<grid, block, 0, 
          stream>>>(src, src_rows, src_cols, channels, src_stride, 
          affine_transform, dst, dst_rows, dst_cols, dst_stride, border_type, 
          border_value1);
    }
  }
  else {
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

RetCode warpAffine(const float* src, int src_rows, int src_cols, int channels,
                   int src_stride, const float* affine_matrix, float* dst,
                   int dst_rows, int dst_cols, int dst_stride,
                   InterpolationType interpolation, BorderType border_type,
                   float border_value, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(src != dst);
  PPL_ASSERT(affine_matrix != nullptr);
  PPL_ASSERT(src_rows >= 1 && src_cols >= 1);
  PPL_ASSERT(dst_rows >= 1 && dst_cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src_stride >= src_cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= dst_cols * channels * (int)sizeof(float));
  PPL_ASSERT(interpolation == INTERPOLATION_TYPE_LINEAR ||
             interpolation == INTERPOLATION_TYPE_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_TYPE_CONSTANT ||
             border_type == BORDER_TYPE_REPLICATE ||
             border_type == BORDER_TYPE_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  cudaError_t code;
  AffineTransform affine_transform(affine_matrix);

  if (interpolation == INTERPOLATION_TYPE_LINEAR) {
    if (channels != 3 && (src_stride & (TEXTURE_ALIGNMENT - 1)) == 0) {
      if (channels == 1) {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        float_c1_ref.normalized = false;
        float_c1_ref.filterMode = cudaFilterModeLinear;
        float_c1_ref.addressMode[0] = cudaAddressModeClamp;
        float_c1_ref.addressMode[1] = cudaAddressModeClamp;
        code = cudaBindTexture2D(0, float_c1_ref, src, desc, src_cols, src_rows,
                                 (size_t)src_stride);
      }
      else {  // channels == 4
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        float_c4_ref.normalized = false;
        float_c4_ref.filterMode = cudaFilterModeLinear;
        float_c4_ref.addressMode[0] = cudaAddressModeClamp;
        float_c4_ref.addressMode[1] = cudaAddressModeClamp;
        code = cudaBindTexture2D(0, float_c4_ref, src, desc, src_cols, src_rows,
                                 (size_t)src_stride);
      }
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA texture error: " << cudaGetErrorString(code);
        return RC_DEVICE_RUNTIME_ERROR;
      }
      
      warpLinearTexKernel<AffineTransform><<<grid, block, 0, stream>>>(src, 
          src_rows, src_cols, channels, src_stride, affine_transform, dst, 
          dst_rows, dst_cols, dst_stride, border_type, border_value);
    }
    else {
      warpLinearKernel<AffineTransform><<<grid, block, 0, stream>>>(src, 
          src_rows, src_cols, channels, src_stride, affine_transform, dst, 
          dst_rows, dst_cols, dst_stride, border_type, border_value);
    }
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      warpNearestKernel<float, float, AffineTransform><<<grid, block, 0, 
          stream>>>(src, src_rows, src_cols, channels, src_stride, 
          affine_transform, dst, dst_rows, dst_cols, dst_stride, border_type, 
          border_value);
    }
    else if (channels == 3) {
      float3 border_value1 = make_float3(border_value, border_value,
                                         border_value);
      warpNearestKernel<float, float3, AffineTransform><<<grid, block, 0, 
          stream>>>(src, src_rows, src_cols, channels, src_stride, 
          affine_transform, dst, dst_rows, dst_cols, dst_stride, border_type, 
          border_value1);
    }
    else {
      float4 border_value1 = make_float4(border_value, border_value,
                                         border_value, border_value);
      warpNearestKernel<float, float4, AffineTransform><<<grid, block, 0, 
          stream>>>(src, src_rows, src_cols, channels, src_stride, 
          affine_transform, dst, dst_rows, dst_cols, dst_stride, border_type, 
          border_value1);
    }
  }
  else {
  }

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode WarpAffine<uchar, 1>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const uchar* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             uchar* outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpAffine(inData, inHeight, inWidth, 1, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, stream);

  return code;
}

template <>
RetCode WarpAffine<uchar, 3>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const uchar* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             uchar* outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpAffine(inData, inHeight, inWidth, 3, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, stream);

  return code;
}

template <>
RetCode WarpAffine<uchar, 4>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const uchar* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             uchar* outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             uchar borderValue) {
  RetCode code = warpAffine(inData, inHeight, inWidth, 4, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, stream);

  return code;
}

template <>
RetCode WarpAffine<float, 1>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const float* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             float* outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpAffine(inData, inHeight, inWidth, 1, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, stream);

  return code;
}

template <>
RetCode WarpAffine<float, 3>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const float* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             float* outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpAffine(inData, inHeight, inWidth, 3, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, stream);

  return code;
}

template <>
RetCode WarpAffine<float, 4>(cudaStream_t stream,
                             int inHeight,
                             int inWidth,
                             int inWidthStride,
                             const float* inData,
                             int outHeight,
                             int outWidth,
                             int outWidthStride,
                             float* outData,
                             const float* affineMatrix,
                             InterpolationType interpolation,
                             BorderType borderType,
                             float borderValue) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = warpAffine(inData, inHeight, inWidth, 4, inWidthStride,
                            affineMatrix, outData, outHeight, outWidth,
                            outWidthStride, interpolation, borderType,
                            borderValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
