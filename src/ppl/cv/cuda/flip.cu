/**
 * @file   flip.cu
 * @brief  The kernel and invocation definitions of image flipping operation.
 * @author Liheng Jian(jianliheng@sensetime.com)
 *
 * @copyright Copyright (c) 2014-2021 SenseTime Group Limited.
 */

#include "flip.h"

#include <cassert>

#include "utility.hpp"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

template <typename T0, typename T1>
__global__
void flipKernel(const T1* src, int rows, int cols, int src_stride, T1* dst,
                int dst_stride, int flip_code) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  T0* output = (T0*)((uchar*)dst + element_y * dst_stride);

  int x, y;
  if (flip_code == 0) {
    x = element_x;
    y = rows - element_y - 1;
  }
  else if (flip_code > 0) {
    x = cols - element_x - 1;
    y = element_y;
  }
  else {
    x = cols - element_x - 1;
    y = rows - element_y - 1;
  }
  T0* input = (T0*)((uchar*)src + y * src_stride);

  if (element_x < cols && element_y < rows) {
    T0 result = input[x];
    output[element_x] = result;
  }
}

RetCode flip(const uchar* src, int rows, int cols, int channels, int src_stride,
             uchar* dst, int dst_stride, int flip_code, cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || rows < 1 || cols < 1 ||
      (channels != 1 && channels != 3 && channels != 4) ||
      src_stride < cols * channels * sizeof(uchar) ||
      dst_stride < cols * channels * sizeof(uchar)) {
    return RC_INVALID_VALUE;
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  if (channels == 1) {
    flipKernel<uchar, uchar><<<grid, block, 0, stream>>>(src, rows, cols,
        src_stride, dst, dst_stride, flip_code);
  }
  else if (channels == 3) {
    flipKernel<uchar3, uchar><<<grid, block, 0, stream>>>(src, rows, cols,
        src_stride, dst, dst_stride, flip_code);
  }
  else if (channels == 4) {
    flipKernel<uchar4, uchar><<<grid, block, 0, stream>>>(src, rows, cols,
        src_stride, dst, dst_stride, flip_code);
  }
  else {
  }

  return RC_SUCCESS;
}

RetCode flip(const float* src, int rows, int cols, int channels, int src_stride,
             float* dst, int dst_stride, int flip_code, cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || rows < 1 || cols < 1 ||
      (channels != 1 && channels != 3 && channels != 4) ||
      src_stride < cols * channels * sizeof(float) ||
      dst_stride < cols * channels * sizeof(float)) {
    return RC_INVALID_VALUE;
  }

  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(cols, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  if (channels == 1) {
    flipKernel<float, float><<<grid, block, 0, stream>>>(src, rows, cols,
        src_stride, dst, dst_stride, flip_code);
  }
  else if (channels == 3) {
    flipKernel<float3, float><<<grid, block, 0, stream>>>(src, rows, cols,
        src_stride, dst, dst_stride, flip_code);
  }
  else if (channels == 4) {
    flipKernel<float4, float><<<grid, block, 0, stream>>>(src, rows, cols,
        src_stride, dst, dst_stride, flip_code);
  }
  else {
  }

  return RC_SUCCESS;
}

template <>
RetCode Flip<uchar, 1>(cudaStream_t stream,
                       int height,
                       int width,
                       int inWidthStride,
                       const uchar* inData,
                       int outWidthStride,
                       uchar* outData,
                       int flipCode) {
  RetCode code = flip(inData, height, width, 1, inWidthStride, outData,
                      outWidthStride, flipCode, stream);

  return code;
}

template <>
RetCode Flip<uchar, 3>(cudaStream_t stream,
                   int height,
                   int width,
                   int inWidthStride,
                   const uchar* inData,
                   int outWidthStride,
                   uchar* outData,
                   int flipCode) {
  RetCode code = flip(inData, height, width, 3, inWidthStride, outData,
                      outWidthStride, flipCode, stream);

  return code;
}

template <>
RetCode Flip<uchar, 4>(cudaStream_t stream,
                   int height,
                   int width,
                   int inWidthStride,
                   const uchar* inData,
                   int outWidthStride,
                   uchar* outData,
                   int flipCode) {
  RetCode code = flip(inData, height, width, 4, inWidthStride, outData,
                      outWidthStride, flipCode, stream);

  return code;
}

template <>
RetCode Flip<float, 1>(cudaStream_t stream,
                    int height,
                    int width,
                    int inWidthStride,
                    const float* inData,
                    int outWidthStride,
                    float* outData,
                    int flipCode) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = flip(inData, height, width, 1, inWidthStride, outData,
                      outWidthStride, flipCode, stream);

  return code;
}

template <>
RetCode Flip<float, 3>(cudaStream_t stream,
                    int height,
                    int width,
                    int inWidthStride,
                    const float* inData,
                    int outWidthStride,
                    float* outData,
                    int flipCode) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = flip(inData, height, width, 3, inWidthStride, outData,
                      outWidthStride, flipCode, stream);

  return code;
}

template <>
RetCode Flip<float, 4>(cudaStream_t stream,
                    int height,
                    int width,
                    int inWidthStride,
                    const float* inData,
                    int outWidthStride,
                    float* outData,
                    int flipCode) {
  inWidthStride  *= sizeof(float);
  outWidthStride *= sizeof(float);
  RetCode code = flip(inData, height, width, 4, inWidthStride, outData,
                      outWidthStride, flipCode, stream);

  return code;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
