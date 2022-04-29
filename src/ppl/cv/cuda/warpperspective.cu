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

#include "ppl/cv/cuda/warpperspective.h"
#include "warp.hpp"

#include "utility/utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

struct PerspectiveTransform {
  float coeffe0;
  float coeffe1;
  float coeffe2;
  float coeffe3;
  float coeffe4;
  float coeffe5;
  float coeffe6;
  float coeffe7;
  float coeffe8;
  float x;
  float y;

  PerspectiveTransform(const float* coefficients) : coeffe0(coefficients[0]),
      coeffe1(coefficients[1]), coeffe2(coefficients[2]),
      coeffe3(coefficients[3]), coeffe4(coefficients[4]),
      coeffe5(coefficients[5]), coeffe6(coefficients[6]),
      coeffe7(coefficients[7]), coeffe8(coefficients[8]) {}

  __DEVICE__
  void calculateCoordinates(int element_x, int element_y) {
    x = coeffe0 * element_x + coeffe1 * element_y + coeffe2;
    y = coeffe3 * element_x + coeffe4 * element_y + coeffe5;

    float weight = coeffe6 * element_x + coeffe7 * element_y + coeffe8;
    weight = 1.f / weight;
    x *= weight;
    y *= weight;
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

RetCode warpPerspective(const uchar* src, int src_rows, int src_cols,
                        int channels, int src_stride,
                        const float* affine_matrix, uchar* dst, int dst_rows,
                        int dst_cols, int dst_stride,
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
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(dst_cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(dst_rows, kBlockDimY0, kBlockShiftY0);

  cudaError_t code;
  PerspectiveTransform perspective_transform(affine_matrix);

  if (interpolation == INTERPOLATION_LINEAR) {
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

      warpLinearTexKernel<PerspectiveTransform><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, perspective_transform, dst,
          dst_rows, dst_cols, dst_stride, border_type, border_value);

      if (channels == 1) {
        cudaUnbindTexture(uchar_c1_ref);
      }
      else {
        cudaUnbindTexture(uchar_c4_ref);
      }
    }
    else {
      warpLinearKernel<PerspectiveTransform><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, perspective_transform, dst,
          dst_rows, dst_cols, dst_stride, border_type, border_value);
    }
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      warpNearestKernel<uchar, uchar, PerspectiveTransform><<<grid, block, 0,
          stream>>>(src, src_rows, src_cols, channels, src_stride,
          perspective_transform, dst, dst_rows, dst_cols, dst_stride,
          border_type, border_value);
    }
    else if (channels == 3) {
      uchar3 border_value1 = make_uchar3(border_value, border_value,
                                         border_value);
      warpNearestKernel<uchar, uchar3, PerspectiveTransform><<<grid, block, 0,
          stream>>>(src, src_rows, src_cols, channels, src_stride,
          perspective_transform, dst, dst_rows, dst_cols, dst_stride,
          border_type, border_value1);
    }
    else {
      uchar4 border_value1 = make_uchar4(border_value, border_value,
                                         border_value, border_value);
      warpNearestKernel<uchar, uchar4, PerspectiveTransform><<<grid, block, 0,
          stream>>>(src, src_rows, src_cols, channels, src_stride,
          perspective_transform, dst, dst_rows, dst_cols, dst_stride,
          border_type, border_value1);
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

RetCode warpPerspective(const float* src, int src_rows, int src_cols,
                        int channels, int src_stride,
                        const float* affine_matrix, float* dst, int dst_rows,
                        int dst_cols, int dst_stride,
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
  PPL_ASSERT(interpolation == INTERPOLATION_LINEAR ||
             interpolation == INTERPOLATION_NEAREST_POINT);
  PPL_ASSERT(border_type == BORDER_CONSTANT ||
             border_type == BORDER_REPLICATE ||
             border_type == BORDER_TRANSPARENT);

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  cudaError_t code;
  PerspectiveTransform perspective_transform(affine_matrix);

  if (interpolation == INTERPOLATION_LINEAR) {
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

      warpLinearTexKernel<PerspectiveTransform><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, perspective_transform, dst,
          dst_rows, dst_cols, dst_stride, border_type, border_value);

      if (channels == 1) {
        cudaUnbindTexture(float_c1_ref);
      }
      else {
        cudaUnbindTexture(float_c4_ref);
      }
    }
    else {
      warpLinearKernel<PerspectiveTransform><<<grid, block, 0, stream>>>(src,
          src_rows, src_cols, channels, src_stride, perspective_transform, dst,
          dst_rows, dst_cols, dst_stride, border_type, border_value);
    }
  }
  else if (interpolation == INTERPOLATION_NEAREST_POINT) {
    if (channels == 1) {
      warpNearestKernel<float, float, PerspectiveTransform><<<grid, block, 0,
          stream>>>(src, src_rows, src_cols, channels, src_stride,
          perspective_transform, dst, dst_rows, dst_cols, dst_stride,
          border_type, border_value);
    }
    else if (channels == 3) {
      float3 border_value1 = make_float3(border_value, border_value,
                                         border_value);
      warpNearestKernel<float, float3, PerspectiveTransform><<<grid, block, 0,
          stream>>>(src, src_rows, src_cols, channels, src_stride,
          perspective_transform, dst, dst_rows, dst_cols, dst_stride,
          border_type, border_value1);
    }
    else {
      float4 border_value1 = make_float4(border_value, border_value,
                                         border_value, border_value);
      warpNearestKernel<float, float4, PerspectiveTransform><<<grid, block, 0,
          stream>>>(src, src_rows, src_cols, channels, src_stride,
          perspective_transform, dst, dst_rows, dst_cols, dst_stride,
          border_type, border_value1);
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
RetCode WarpPerspective<uchar, 1>(cudaStream_t stream,
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
  RetCode code = warpPerspective(inData, inHeight, inWidth, 1, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<uchar, 3>(cudaStream_t stream,
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
  RetCode code = warpPerspective(inData, inHeight, inWidth, 3, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<uchar, 4>(cudaStream_t stream,
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
  RetCode code = warpPerspective(inData, inHeight, inWidth, 4, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<float, 1>(cudaStream_t stream,
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
  RetCode code = warpPerspective(inData, inHeight, inWidth, 1, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<float, 3>(cudaStream_t stream,
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
  RetCode code = warpPerspective(inData, inHeight, inWidth, 3, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

template <>
RetCode WarpPerspective<float, 4>(cudaStream_t stream,
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
  RetCode code = warpPerspective(inData, inHeight, inWidth, 4, inWidthStride,
                                 affineMatrix, outData, outHeight, outWidth,
                                 outWidthStride, interpolation, borderType,
                                 borderValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
