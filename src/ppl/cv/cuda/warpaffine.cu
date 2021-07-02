/**
 * @file   warpaffine.cu
 * @brief  The kernel and invocation definitions of applying an affine
           transformation to an image.
 * @author Liheng Jian(jianliheng@sensetime.com)
 *
 * @copyright Copyright (c) 2014-2021 SenseTime Group Limited.
 */

#include "warpaffine.h"

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/**************************** WarpAffineLinear() ****************************/

__global__
void warpAffineLinearKernel(const uchar* src, int src_rows, int src_cols,
                            int channels, int src_stride, uchar* dst,
                            int dst_rows, int dst_cols, int dst_stride,
                            float coeffe0, float coeffe1, float coeffe2,
                            float coeffe3, float coeffe4, float coeffe5,
                            BorderType border_type, uchar border_value) {
  int element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float src_x = coeffe0 * element_x + coeffe1 * element_y + coeffe2;
  float src_y = coeffe3 * element_x + coeffe4 * element_y + coeffe5;

  int src_x1 = __float2int_rd(src_x);
  int src_y1 = __float2int_rd(src_y);
  int src_x2 = src_x1 + 1;
  int src_y2 = src_y1 + 1;

  if (border_type == BORDER_TYPE_CONSTANT ||
      border_type == BORDER_TYPE_TRANSPARENT) {
    bool flag0 = src_y1 >= 0 && src_y1 < src_rows && src_x1 >= 0 &&
                 src_x1 < src_cols;
    bool flag1 = src_y1 >= 0 && src_y1 < src_rows && src_x2 >= 0 &&
                 src_x2 < src_cols;
    bool flag2 = src_y2 >= 0 && src_y2 < src_rows && src_x1 >= 0 &&
                 src_x1 < src_cols;
    bool flag3 = src_y2 >= 0 && src_y2 < src_rows && src_x2 >= 0 &&
                 src_x2 < src_cols;

    if ((border_type == BORDER_TYPE_TRANSPARENT) &&
        ((!flag0) || (!flag1) || (!flag2) || (!flag3))) {
      return;
    }

    if (channels == 1) {
      uchar* input = (uchar*)(src + src_y1 * src_stride);
      uchar src_value1 = flag0 ? input[src_x1] : border_value;
      uchar src_value2 = flag1 ? input[src_x2] : border_value;
      float value1 = (src_x2 - src_x) * (src_y2 - src_y) * src_value1;
      float value2 = (src_x - src_x1) * (src_y2 - src_y) * src_value2;
      float sum = 0.f;
      sum += value1;
      sum += value2;

      input = (uchar*)(src + src_y2 * src_stride);
      src_value1 = flag2 ? input[src_x1] : border_value;
      src_value2 = flag3 ? input[src_x2] : border_value;
      value1 = (src_x2 - src_x) * (src_y - src_y1) * src_value1;
      value2 = (src_x - src_x1) * (src_y - src_y1) * src_value2;
      sum += value1;
      sum += value2;

      uchar* output = (uchar*)(dst + element_y * dst_stride);
      output[element_x] = saturate_cast(sum);
    }
    else if (channels == 2) {
      uchar2 border_value1 = make_uchar2(border_value, border_value);
      uchar2* input = (uchar2*)(src + src_y1 * src_stride);
      uchar2 src_value1 = flag0 ? input[src_x1] : border_value1;
      uchar2 src_value2 = flag1 ? input[src_x2] : border_value1;
      float2 value1 = (src_x2 - src_x) * (src_y2 - src_y) * src_value1;
      float2 value2 = (src_x - src_x1) * (src_y2 - src_y) * src_value2;
      float2 sum = make_float2(0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (uchar2*)(src + src_y2 * src_stride);
      src_value1 = flag2 ? input[src_x1] : border_value1;
      src_value2 = flag3 ? input[src_x2] : border_value1;
      value1 = (src_x2 - src_x) * (src_y - src_y1) * src_value1;
      value2 = (src_x - src_x1) * (src_y - src_y1) * src_value2;
      sum += value1;
      sum += value2;

      uchar2* output = (uchar2*)(dst + element_y * dst_stride);
      output[element_x] = saturate_cast_vector<uchar2, float2>(sum);
    }
    else if (channels == 3) {
      uchar3 border_value1 = make_uchar3(border_value, border_value,
                                         border_value);
      uchar3* input = (uchar3*)(src + src_y1 * src_stride);
      uchar3 src_value1 = flag0 ? input[src_x1] : border_value1;
      uchar3 src_value2 = flag1 ? input[src_x2] : border_value1;
      float3 value1 = (src_x2 - src_x) * (src_y2 - src_y) * src_value1;
      float3 value2 = (src_x - src_x1) * (src_y2 - src_y) * src_value2;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (uchar3*)(src + src_y2 * src_stride);
      src_value1 = flag2 ? input[src_x1] : border_value1;
      src_value2 = flag3 ? input[src_x2] : border_value1;
      value1 = (src_x2 - src_x) * (src_y - src_y1) * src_value1;
      value2 = (src_x - src_x1) * (src_y - src_y1) * src_value2;
      sum += value1;
      sum += value2;

      uchar3* output = (uchar3*)(dst + element_y * dst_stride);
      if (src_x > src_cols - 1 || src_y > src_rows - 1) {
        output[element_x] = border_value1;  // align with npp.
      }
      else {
        output[element_x] = saturate_cast_vector<uchar3, float3>(sum);
      }
    }
    else {
      uchar4 border_value1 = make_uchar4(border_value, border_value,
                                         border_value, border_value);
      uchar4* input = (uchar4*)(src + src_y1 * src_stride);
      uchar4 src_value1 = flag0 ? input[src_x1] : border_value1;
      uchar4 src_value2 = flag1 ? input[src_x2] : border_value1;
      float4 value1 = (src_x2 - src_x) * (src_y2 - src_y) * src_value1;
      float4 value2 = (src_x - src_x1) * (src_y2 - src_y) * src_value2;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (uchar4*)(src + src_y2 * src_stride);
      src_value1 = flag2 ? input[src_x1] : border_value1;
      src_value2 = flag3 ? input[src_x2] : border_value1;
      value1 = (src_x2 - src_x) * (src_y - src_y1) * src_value1;
      value2 = (src_x - src_x1) * (src_y - src_y1) * src_value2;
      sum += value1;
      sum += value2;

      uchar4* output = (uchar4*)(dst + element_y * dst_stride);
      output[element_x] = saturate_cast_vector<uchar4, float4>(sum);
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    float diff_x1 = src_x - src_x1;
    float diff_x2 = src_x2 - src_x;
    float diff_y1 = src_y - src_y1;
    float diff_y2 = src_y2 - src_y;

    src_x1 = clip(src_x1, 0, src_cols - 1);
    src_y1 = clip(src_y1, 0, src_rows - 1);
    src_x2 = clip(src_x2, 0, src_cols - 1);
    src_y2 = clip(src_y2, 0, src_rows - 1);

    if (channels == 1) {
      uchar* input = (uchar*)(src + src_y1 * src_stride);
      uchar src_value1 = input[src_x1];
      uchar src_value2 = input[src_x2];
      float value1 = diff_x2 * diff_y2 * src_value1;
      float value2 = diff_x1 * diff_y2 * src_value2;
      float sum = 0.f;
      sum += value1;
      sum += value2;

      input = (uchar*)(src + src_y2 * src_stride);
      src_value1 = input[src_x1];
      src_value2 = input[src_x2];
      value1 = diff_x2 * diff_y1 * src_value1;
      value2 = diff_x1 * diff_y1 * src_value2;
      sum += value1;
      sum += value2;

      uchar* output = (uchar*)(dst + element_y * dst_stride);
      output[element_x] = saturate_cast(sum);
    }
    else if (channels == 2) {
      uchar2* input = (uchar2*)(src + src_y1 * src_stride);
      uchar2 src_value1 = input[src_x1];
      uchar2 src_value2 = input[src_x2];
      float2 value1 = diff_x2 * diff_y2 * src_value1;
      float2 value2 = diff_x1 * diff_y2 * src_value2;
      float2 sum = make_float2(0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (uchar2*)(src + src_y2 * src_stride);
      src_value1 = input[src_x1];
      src_value2 = input[src_x2];
      value1 = diff_x2 * diff_y1 * src_value1;
      value2 = diff_x1 * diff_y1 * src_value2;
      sum += value1;
      sum += value2;

      uchar2* output = (uchar2*)(dst + element_y * dst_stride);
      output[element_x] = saturate_cast_vector<uchar2, float2>(sum);
    }
    else if (channels == 3) {
      uchar3* input = (uchar3*)(src + src_y1 * src_stride);
      uchar3 src_value1 = input[src_x1];
      uchar3 src_value2 = input[src_x2];
      float3 value1 = diff_x2 * diff_y2 * src_value1;
      float3 value2 = diff_x1 * diff_y2 * src_value2;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (uchar3*)(src + src_y2 * src_stride);
      src_value1 = input[src_x1];
      src_value2 = input[src_x2];
      value1 = diff_x2 * diff_y1 * src_value1;
      value2 = diff_x1 * diff_y1 * src_value2;
      sum += value1;
      sum += value2;

      uchar3* output = (uchar3*)(dst + element_y * dst_stride);
      output[element_x] = saturate_cast_vector<uchar3, float3>(sum);
    }
    else {
      uchar4* input = (uchar4*)(src + src_y1 * src_stride);
      uchar4 src_value1 = input[src_x1];
      uchar4 src_value2 = input[src_x2];
      float4 value1 = diff_x2 * diff_y2 * src_value1;
      float4 value2 = diff_x1 * diff_y2 * src_value2;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (uchar4*)(src + src_y2 * src_stride);
      src_value1 = input[src_x1];
      src_value2 = input[src_x2];
      value1 = diff_x2 * diff_y1 * src_value1;
      value2 = diff_x1 * diff_y1 * src_value2;
      sum += value1;
      sum += value2;

      uchar4* output = (uchar4*)(dst + element_y * dst_stride);
      output[element_x] = saturate_cast_vector<uchar4, float4>(sum);
    }
  }
  else {
  }
}

__global__
void warpAffineLinearKernel(const float* src, int src_rows, int src_cols,
                            int channels, int src_stride, float* dst,
                            int dst_rows, int dst_cols, int dst_stride,
                            float coeffe0, float coeffe1, float coeffe2,
                            float coeffe3, float coeffe4, float coeffe5,
                            BorderType border_type, float border_value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float src_x = coeffe0 * element_x + coeffe1 * element_y + coeffe2;
  float src_y = coeffe3 * element_x + coeffe4 * element_y + coeffe5;

  int src_x1 = __float2int_rd(src_x);
  int src_y1 = __float2int_rd(src_y);
  int src_x2 = src_x1 + 1;
  int src_y2 = src_y1 + 1;

  if (border_type == BORDER_TYPE_CONSTANT ||
      border_type == BORDER_TYPE_TRANSPARENT) {
    bool flag0 = src_y1 >= 0 && src_y1 < src_rows && src_x1 >= 0 &&
                 src_x1 < src_cols;
    bool flag1 = src_y1 >= 0 && src_y1 < src_rows && src_x2 >= 0 &&
                 src_x2 < src_cols;
    bool flag2 = src_y2 >= 0 && src_y2 < src_rows && src_x1 >= 0 &&
                 src_x1 < src_cols;
    bool flag3 = src_y2 >= 0 && src_y2 < src_rows && src_x2 >= 0 &&
                 src_x2 < src_cols;

    if ((border_type == BORDER_TYPE_TRANSPARENT) &&
        ((!flag0) || (!flag1) || (!flag2) || (!flag3))) {
      return;
    }

    if (channels == 1) {
      float* input = (float*)(src + src_y1 * src_stride);
      float src_value1 = flag0 ? input[src_x1] : border_value;
      float src_value2 = flag1 ? input[src_x2] : border_value;
      float value1 = (src_x2 - src_x) * (src_y2 - src_y) * src_value1;
      float value2 = (src_x - src_x1) * (src_y2 - src_y) * src_value2;
      float sum = 0.f;
      sum += value1;
      sum += value2;

      input = (float*)(src + src_y2 * src_stride);
      src_value1 = flag2 ? input[src_x1] : border_value;
      src_value2 = flag3 ? input[src_x2] : border_value;
      value1 = (src_x2 - src_x) * (src_y - src_y1) * src_value1;
      value2 = (src_x - src_x1) * (src_y - src_y1) * src_value2;
      sum += value1;
      sum += value2;

      float* output = (float*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else if (channels == 3) {
      float3 border_value1 = make_float3(border_value, border_value,
                                         border_value);
      float3* input = (float3*)(src + src_y1 * src_stride);
      float3 src_value1 = flag0 ? input[src_x1] : border_value1;
      float3 src_value2 = flag1 ? input[src_x2] : border_value1;
      float3 value1 = (src_x2 - src_x) * (src_y2 - src_y) * src_value1;
      float3 value2 = (src_x - src_x1) * (src_y2 - src_y) * src_value2;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (float3*)(src + src_y2 * src_stride);
      src_value1 = flag2 ? input[src_x1] : border_value1;
      src_value2 = flag3 ? input[src_x2] : border_value1;
      value1 = (src_x2 - src_x) * (src_y - src_y1) * src_value1;
      value2 = (src_x - src_x1) * (src_y - src_y1) * src_value2;
      sum += value1;
      sum += value2;

      float3* output = (float3*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else {
      float4 border_value1 = make_float4(border_value, border_value,
                                         border_value, border_value);
      float4* input = (float4*)(src + src_y1 * src_stride);
      float4 src_value1 = flag0 ? input[src_x1] : border_value1;
      float4 src_value2 = flag1 ? input[src_x2] : border_value1;
      float4 value1 = (src_x2 - src_x) * (src_y2 - src_y) * src_value1;
      float4 value2 = (src_x - src_x1) * (src_y2 - src_y) * src_value2;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (float4*)(src + src_y2 * src_stride);
      src_value1 = flag2 ? input[src_x1] : border_value1;
      src_value2 = flag3 ? input[src_x2] : border_value1;
      value1 = (src_x2 - src_x) * (src_y - src_y1) * src_value1;
      value2 = (src_x - src_x1) * (src_y - src_y1) * src_value2;
      sum += value1;
      sum += value2;

      float4* output = (float4*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    float diff_x1 = src_x - src_x1;
    float diff_x2 = src_x2 - src_x;
    float diff_y1 = src_y - src_y1;
    float diff_y2 = src_y2 - src_y;

    src_x1 = clip(src_x1, 0, src_cols - 1);
    src_y1 = clip(src_y1, 0, src_rows - 1);
    src_x2 = clip(src_x2, 0, src_cols - 1);
    src_y2 = clip(src_y2, 0, src_rows - 1);

    if (channels == 1) {
      float* input = (float*)(src + src_y1 * src_stride);
      float src_value1 = input[src_x1];
      float src_value2 = input[src_x2];
      float value1 = diff_x2 * diff_y2 * src_value1;
      float value2 = diff_x1 * diff_y2 * src_value2;
      float sum = 0.f;
      sum += value1;
      sum += value2;

      input = (float*)(src + src_y2 * src_stride);
      src_value1 = input[src_x1];
      src_value2 = input[src_x2];
      value1 = diff_x2 * diff_y1 * src_value1;
      value2 = diff_x1 * diff_y1 * src_value2;
      sum += value1;
      sum += value2;

      float* output = (float*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else if (channels == 3) {
      float3* input = (float3*)(src + src_y1 * src_stride);
      float3 src_value1 = input[src_x1];
      float3 src_value2 = input[src_x2];
      float3 value1 = diff_x2 * diff_y2 * src_value1;
      float3 value2 = diff_x1 * diff_y2 * src_value2;
      float3 sum = make_float3(0.f, 0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (float3*)(src + src_y2 * src_stride);
      src_value1 = input[src_x1];
      src_value2 = input[src_x2];
      value1 = diff_x2 * diff_y1 * src_value1;
      value2 = diff_x1 * diff_y1 * src_value2;
      sum += value1;
      sum += value2;

      float3* output = (float3*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
    else {
      float4* input = (float4*)(src + src_y1 * src_stride);
      float4 src_value1 = input[src_x1];
      float4 src_value2 = input[src_x2];
      float4 value1 = diff_x2 * diff_y2 * src_value1;
      float4 value2 = diff_x1 * diff_y2 * src_value2;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      sum += value1;
      sum += value2;

      input = (float4*)(src + src_y2 * src_stride);
      src_value1 = input[src_x1];
      src_value2 = input[src_x2];
      value1 = diff_x2 * diff_y1 * src_value1;
      value2 = diff_x1 * diff_y1 * src_value2;
      sum += value1;
      sum += value2;

      float4* output = (float4*)(dst + element_y * dst_stride);
      output[element_x] = sum;
    }
  }
  else {
  }
}

RetCode warpAffineLinear(const uchar* src, int src_rows, int src_cols,
                         int channels, int src_stride, uchar* dst, int dst_rows,
                         int dst_cols, int dst_stride,
                         const float* affine_matrix, BorderType border_type,
                         uchar border_value, cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || src == dst ||
      src_rows < 1 || src_cols < 1 || dst_rows < 1 || dst_cols < 1 ||
      (channels != 1 && channels != 3 && channels != 4) ||
      src_stride < src_cols * channels || dst_stride < dst_cols * channels ||
      (border_type != BORDER_TYPE_CONSTANT &&
       border_type != BORDER_TYPE_REPLICATE &&
       border_type != BORDER_TYPE_TRANSPARENT)) {
    return RC_INVALID_VALUE;
  }

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(dst_cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(dst_rows, kBlockDimY0, kBlockShiftY0);

  warpAffineLinearKernel<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
      channels, src_stride, dst, dst_rows, dst_cols, dst_stride,
      affine_matrix[0], affine_matrix[1], affine_matrix[2], affine_matrix[3],
      affine_matrix[4], affine_matrix[5], border_type, border_value);

  return RC_SUCCESS;
}

RetCode warpAffineLinear(const float* src, int src_rows, int src_cols,
                         int channels, int src_stride, float* dst, int dst_rows,
                         int dst_cols, int dst_stride,
                         const float* affine_matrix, BorderType border_type,
                         float border_value, cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || src == dst ||
      src_rows < 1 || src_cols < 1 || dst_rows < 1 || dst_cols < 1 ||
      (channels != 1 && channels != 3 && channels != 4) ||
      src_stride < src_cols * channels || dst_stride < dst_cols * channels ||
      (border_type != BORDER_TYPE_CONSTANT &&
       border_type != BORDER_TYPE_REPLICATE &&
       border_type != BORDER_TYPE_TRANSPARENT)) {
    return RC_INVALID_VALUE;
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  warpAffineLinearKernel<<<grid, block, 0, stream>>>(src, src_rows, src_cols,
      channels, src_stride, dst, dst_rows, dst_cols, dst_stride,
      affine_matrix[0], affine_matrix[1], affine_matrix[2], affine_matrix[3],
      affine_matrix[4], affine_matrix[5], border_type, border_value);

  return RC_SUCCESS;
}

template <>
RetCode WarpAffineLinear<uchar, 1>(cudaStream_t stream,
                                   int inHeight,
                                   int inWidth,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outHeight,
                                   int outWidth,
                                   int outWidthStride,
                                   uchar* outData,
                                   const float* affineMatrix,
                                   BorderType borderType,
                                   uchar borderValue) {
  RetCode code = warpAffineLinear(inData, inHeight, inWidth, 1, inWidthStride,
                                  outData, outHeight, outWidth, outWidthStride,
                                  affineMatrix, borderType, borderValue,
                                  stream);

  return code;
}

template <>
RetCode WarpAffineLinear<uchar, 3>(cudaStream_t stream,
                                   int inHeight,
                                   int inWidth,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outHeight,
                                   int outWidth,
                                   int outWidthStride,
                                   uchar* outData,
                                   const float* affineMatrix,
                                   BorderType borderType,
                                   uchar borderValue) {
  RetCode code = warpAffineLinear(inData, inHeight, inWidth, 3, inWidthStride,
                                  outData, outHeight, outWidth, outWidthStride,
                                  affineMatrix, borderType, borderValue,
                                  stream);

  return code;
}

template <>
RetCode WarpAffineLinear<uchar, 4>(cudaStream_t stream,
                                   int inHeight,
                                   int inWidth,
                                   int inWidthStride,
                                   const uchar* inData,
                                   int outHeight,
                                   int outWidth,
                                   int outWidthStride,
                                   uchar* outData,
                                   const float* affineMatrix,
                                   BorderType borderType,
                                   uchar borderValue) {
  RetCode code = warpAffineLinear(inData, inHeight, inWidth, 4, inWidthStride,
                                  outData, outHeight, outWidth, outWidthStride,
                                  affineMatrix, borderType, borderValue,
                                  stream);

  return code;
}

template <>
RetCode WarpAffineLinear<float, 1>(cudaStream_t stream,
                                   int inHeight,
                                   int inWidth,
                                   int inWidthStride,
                                   const float* inData,
                                   int outHeight,
                                   int outWidth,
                                   int outWidthStride,
                                   float* outData,
                                   const float* affineMatrix,
                                   BorderType borderType,
                                   float borderValue) {
  RetCode code = warpAffineLinear(inData, inHeight, inWidth, 1, inWidthStride,
                                  outData, outHeight, outWidth, outWidthStride,
                                  affineMatrix, borderType, borderValue,
                                  stream);

  return code;
}

template <>
RetCode WarpAffineLinear<float, 3>(cudaStream_t stream,
                                   int inHeight,
                                   int inWidth,
                                   int inWidthStride,
                                   const float* inData,
                                   int outHeight,
                                   int outWidth,
                                   int outWidthStride,
                                   float* outData,
                                   const float* affineMatrix,
                                   BorderType borderType,
                                   float borderValue) {
  RetCode code = warpAffineLinear(inData, inHeight, inWidth, 3, inWidthStride,
                                  outData, outHeight, outWidth, outWidthStride,
                                  affineMatrix, borderType, borderValue,
                                  stream);

  return code;
}

template <>
RetCode WarpAffineLinear<float, 4>(cudaStream_t stream,
                                   int inHeight,
                                   int inWidth,
                                   int inWidthStride,
                                   const float* inData,
                                   int outHeight,
                                   int outWidth,
                                   int outWidthStride,
                                   float* outData,
                                   const float* affineMatrix,
                                   BorderType borderType,
                                   float borderValue) {
  RetCode code = warpAffineLinear(inData, inHeight, inWidth, 4, inWidthStride,
                                  outData, outHeight, outWidth, outWidthStride,
                                  affineMatrix, borderType, borderValue,
                                  stream);

  return code;
}

/*********************** WarpAffineNearestPoint() ***************************/

template <typename T0, typename T1>
__global__
void warpAffineNearestPointKernel(const T1* src, int src_rows, int src_cols,
                                  int channels, int src_stride, T1* dst,
                                  int dst_rows, int dst_cols, int dst_stride,
                                  float coeffe0, float coeffe1, float coeffe2,
                                  float coeffe3, float coeffe4, float coeffe5,
                                  BorderType border_type, T0 border_value) {
  int element_x, element_y;
  if (sizeof(T1) == 1) {
    element_x = (blockIdx.x << kBlockShiftX0) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY0) + threadIdx.y;
  }
  else {
    element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
    element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  }
  if (element_y >= dst_rows || element_x >= dst_cols) {
    return;
  }

  float src_x_float = coeffe0 * element_x + coeffe1 * element_y + coeffe2;
  float src_y_float = coeffe3 * element_x + coeffe4 * element_y + coeffe5;

  int src_x = src_x_float;
  int src_y = src_y_float;

  if (border_type == BORDER_TYPE_CONSTANT) {
    T0* output = (T0*)(dst + element_y * dst_stride);

    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      T0* input  = (T0*)(src + src_y * src_stride);
      output[element_x] = input[src_x];
    }
    else {
      output[element_x] = border_value;
    }
  }
  else if (border_type == BORDER_TYPE_REPLICATE) {
    src_x = clip(src_x, 0, src_cols - 1);
    src_y = clip(src_y, 0, src_rows - 1);

    T0* input  = (T0*)(src + src_y * src_stride);
    T0* output = (T0*)(dst + element_y * dst_stride);
    output[element_x] = input[src_x];
  }
  else if (border_type == BORDER_TYPE_TRANSPARENT) {
    T0* output = (T0*)(dst + element_y * dst_stride);

    if (src_x >= 0 && src_x < src_cols && src_y >= 0 && src_y < src_rows) {
      T0* input  = (T0*)(src + src_y * src_stride);
      output[element_x] = input[src_x];
    }
  }
  else {
  }
}

RetCode warpAffineNearestPoint(const uchar* src, int src_rows, int src_cols,
                               int channels, int src_stride, uchar* dst,
                               int dst_rows, int dst_cols, int dst_stride,
                               const float* affine_matrix,
                               BorderType border_type, uchar border_value,
                               cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || src == dst ||
      src_rows < 1 || src_cols < 1 || dst_rows < 1 || dst_cols < 1 ||
      (channels != 1 && channels != 3 && channels != 4) ||
      src_stride < src_cols * channels || dst_stride < dst_cols * channels ||
      (border_type != BORDER_TYPE_CONSTANT &&
       border_type != BORDER_TYPE_REPLICATE &&
       border_type != BORDER_TYPE_TRANSPARENT)) {
    return RC_INVALID_VALUE;
  }

  dim3 block, grid;
  block.x = kBlockDimX0;
  block.y = kBlockDimY0;
  grid.x  = divideUp(dst_cols, kBlockDimX0, kBlockShiftX0);
  grid.y  = divideUp(dst_rows, kBlockDimY0, kBlockShiftY0);

  if (channels == 1) {
    warpAffineNearestPointKernel<uchar, uchar><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, affine_matrix[0], affine_matrix[1], affine_matrix[2],
        affine_matrix[3], affine_matrix[4], affine_matrix[5], border_type,
        border_value);
  }
  else if (channels == 3) {
    uchar3 border_value1 = make_uchar3(border_value, border_value,
                                       border_value);
    warpAffineNearestPointKernel<uchar3, uchar><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, affine_matrix[0], affine_matrix[1], affine_matrix[2],
        affine_matrix[3], affine_matrix[4], affine_matrix[5], border_type,
        border_value1);
  }
  else {
    uchar4 border_value1 = make_uchar4(border_value, border_value,
                                       border_value, border_value);
    warpAffineNearestPointKernel<uchar4, uchar><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, affine_matrix[0], affine_matrix[1], affine_matrix[2],
        affine_matrix[3], affine_matrix[4], affine_matrix[5], border_type,
        border_value1);
  }

  return RC_SUCCESS;
}

RetCode warpAffineNearestPoint(const float* src, int src_rows, int src_cols,
                               int channels, int src_stride, float* dst,
                               int dst_rows, int dst_cols, int dst_stride,
                               const float* affine_matrix,
                               BorderType border_type, float border_value,
                               cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || src == dst ||
      src_rows < 1 || src_cols < 1 || dst_rows < 1 || dst_cols < 1 ||
      (channels != 1 && channels != 3 && channels != 4) ||
      src_stride < src_cols * channels || dst_stride < dst_cols * channels ||
      (border_type != BORDER_TYPE_CONSTANT &&
       border_type != BORDER_TYPE_REPLICATE &&
       border_type != BORDER_TYPE_TRANSPARENT)) {
    return RC_INVALID_VALUE;
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(dst_cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(dst_rows, kBlockDimY1, kBlockShiftY1);

  if (channels == 1) {
    warpAffineNearestPointKernel<float, float><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, affine_matrix[0], affine_matrix[1], affine_matrix[2],
        affine_matrix[3], affine_matrix[4], affine_matrix[5], border_type,
        border_value);
  }
  else if (channels == 3) {
    float3 border_value1 = make_float3(border_value, border_value,
                                       border_value);
    warpAffineNearestPointKernel<float3, float><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, affine_matrix[0], affine_matrix[1], affine_matrix[2],
        affine_matrix[3], affine_matrix[4], affine_matrix[5], border_type,
        border_value1);
  }
  else {
    float4 border_value1 = make_float4(border_value, border_value,
                                       border_value, border_value);
    warpAffineNearestPointKernel<float4, float><<<grid, block, 0, stream>>>(src,
        src_rows, src_cols, channels, src_stride, dst, dst_rows, dst_cols,
        dst_stride, affine_matrix[0], affine_matrix[1], affine_matrix[2],
        affine_matrix[3], affine_matrix[4], affine_matrix[5], border_type,
        border_value1);
  }

  return RC_SUCCESS;
}

template <>
RetCode WarpAffineNearestPoint<uchar, 1>(cudaStream_t stream,
                                         int inHeight,
                                         int inWidth,
                                         int inWidthStride,
                                         const uchar* inData,
                                         int outHeight,
                                         int outWidth,
                                         int outWidthStride,
                                         uchar* outData,
                                         const float* affineMatrix,
                                         BorderType borderType,
                                         uchar borderValue) {
  RetCode code = warpAffineNearestPoint(inData, inHeight, inWidth, 1,
                                        inWidthStride, outData, outHeight,
                                        outWidth, outWidthStride, affineMatrix,
                                        borderType, borderValue, stream);

  return code;
}

template <>
RetCode WarpAffineNearestPoint<uchar, 3>(cudaStream_t stream,
                                         int inHeight,
                                         int inWidth,
                                         int inWidthStride,
                                         const uchar* inData,
                                         int outHeight,
                                         int outWidth,
                                         int outWidthStride,
                                         uchar* outData,
                                         const float* affineMatrix,
                                         BorderType borderType,
                                         uchar borderValue) {
  RetCode code = warpAffineNearestPoint(inData, inHeight, inWidth, 3,
                                        inWidthStride, outData, outHeight,
                                        outWidth, outWidthStride, affineMatrix,
                                        borderType, borderValue, stream);

  return code;
}

template <>
RetCode WarpAffineNearestPoint<uchar, 4>(cudaStream_t stream,
                                         int inHeight,
                                         int inWidth,
                                         int inWidthStride,
                                         const uchar* inData,
                                         int outHeight,
                                         int outWidth,
                                         int outWidthStride,
                                         uchar* outData,
                                         const float* affineMatrix,
                                         BorderType borderType,
                                         uchar borderValue) {
  RetCode code = warpAffineNearestPoint(inData, inHeight, inWidth, 4,
                                        inWidthStride, outData, outHeight,
                                        outWidth, outWidthStride, affineMatrix,
                                        borderType, borderValue, stream);

  return code;
}

template <>
RetCode WarpAffineNearestPoint<float, 1>(cudaStream_t stream,
                                         int inHeight,
                                         int inWidth,
                                         int inWidthStride,
                                         const float* inData,
                                         int outHeight,
                                         int outWidth,
                                         int outWidthStride,
                                         float* outData,
                                         const float* affineMatrix,
                                         BorderType borderType,
                                         float borderValue) {
  RetCode code = warpAffineNearestPoint(inData, inHeight, inWidth, 1,
                                        inWidthStride, outData, outHeight,
                                        outWidth, outWidthStride, affineMatrix,
                                        borderType, borderValue, stream);

  return code;
}

template <>
RetCode WarpAffineNearestPoint<float, 3>(cudaStream_t stream,
                                         int inHeight,
                                         int inWidth,
                                         int inWidthStride,
                                         const float* inData,
                                         int outHeight,
                                         int outWidth,
                                         int outWidthStride,
                                         float* outData,
                                         const float* affineMatrix,
                                         BorderType borderType,
                                         float borderValue) {
  RetCode code = warpAffineNearestPoint(inData, inHeight, inWidth, 3,
                                        inWidthStride, outData, outHeight,
                                        outWidth, outWidthStride, affineMatrix,
                                        borderType, borderValue, stream);

  return code;
}

template <>
RetCode WarpAffineNearestPoint<float, 4>(cudaStream_t stream,
                                         int inHeight,
                                         int inWidth,
                                         int inWidthStride,
                                         const float* inData,
                                         int outHeight,
                                         int outWidth,
                                         int outWidthStride,
                                         float* outData,
                                         const float* affineMatrix,
                                         BorderType borderType,
                                         float borderValue) {
  RetCode code = warpAffineNearestPoint(inData, inHeight, inWidth, 4,
                                        inWidthStride, outData, outHeight,
                                        outWidth, outWidthStride, affineMatrix,
                                        borderType, borderValue, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
