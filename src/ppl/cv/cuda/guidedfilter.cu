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

#include "ppl/cv/cuda/guidedfilter.h"

#include "utility/utility.hpp"
#include "utility/use_memory_pool.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

/**************************** function declaration **************************/

RetCode multiply(const float* src0, int rows, int cols, int channels,
                 int src0_stride, const float* src1, int src1_stride,
                 float* dst, int dst_stride, float scale, cudaStream_t stream);

RetCode divide(const float* src0, int rows, int cols, int channels,
               int src0_stride, const float* src1, int src1_stride,
               float* dst, int dst_stride, float scale, cudaStream_t stream);

RetCode add(const float* src0, int rows, int cols, int channels,
            int src0_stride, const float* src1, int src1_stride, float* dst,
            int dst_stride, cudaStream_t stream);

RetCode boxFilter(const float* src, int rows, int cols, int channels,
                  int src_stride, int ksize_x, int ksize_y, bool normalize,
                  float* dst, int dst_stride, BorderType border_type,
                  cudaStream_t stream);

RetCode split3Channels(const float* src, int rows, int cols, int src_stride,
                       float* dst0, float* dst1, float* dst2, int dst_stride,
                       cudaStream_t stream);

RetCode split4Channels(const float* src, int rows, int cols, int src_stride,
                       float* dst0, float* dst1, float* dst2, float* dst3,
                       int dst_stride, cudaStream_t stream);

RetCode merge3Channels(const float* src0, const float* src1, const float* src2,
                       int rows, int cols, int src_stride, float* dst,
                       int dst_stride, cudaStream_t stream);

RetCode merge4Channels(const float* src0, const float* src1, const float* src2,
                       const float* src3, int rows, int cols, int src_stride,
                       float* dst, int dst_stride, cudaStream_t stream);

RetCode convertTo(const uchar* src, int rows, int cols, int channels,
                  int src_stride, float* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream);

RetCode convertTo(const float* src, int rows, int cols, int channels,
                  int src_stride, uchar* dst, int dst_stride, float alpha,
                  float beta, cudaStream_t stream);

/******************************* addScalar() *******************************/

__global__
void addScalarKernel(float* dst, int rows, int columns, int stride,
                     float value) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= columns) {
    return;
  }

  float* output = (float*)((uchar*)dst + element_y * stride);
  float result = output[element_x];
  result += value;

  output[element_x] = result;
}

RetCode addScalar(float* dst, int rows, int cols, int channels, int stride,
                  float value, cudaStream_t stream) {
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(stride >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(columns, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  addScalarKernel<<<grid, block, 0, stream>>>(dst, rows, columns, stride,
                                              value);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/******************************* subtract() ********************************/

__global__
void subtractKernel(const float* src0, int rows, int cols, int src0_stride,
                    const float* src1, int src1_stride, float* dst,
                    int dst_stride) {
  int element_x = (blockIdx.x << kBlockShiftX1) + threadIdx.x;
  int element_y = (blockIdx.y << kBlockShiftY1) + threadIdx.y;
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  float* input0 = (float*)((uchar*)src0 + element_y * src0_stride);
  float* input1 = (float*)((uchar*)src1 + element_y * src1_stride);
  float* output = (float*)((uchar*)dst + element_y * dst_stride);

  float value0 = input0[element_x];
  float value1 = input1[element_x];
  float result = value0 - value1;

  output[element_x] = result;
}

RetCode subtract(const float* src0, int rows, int cols, int channels,
                 int src0_stride, const float* src1, int src1_stride,
                 float* dst, int dst_stride, cudaStream_t stream) {
  PPL_ASSERT(src0 != nullptr);
  PPL_ASSERT(src1 != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(channels == 1 || channels == 3 || channels == 4);
  PPL_ASSERT(src0_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(src1_stride >= cols * channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * channels * (int)sizeof(float));

  int columns = cols * channels;
  dim3 block, grid;
  block.x = kBlockDimX1;
  block.y = kBlockDimY1;
  grid.x  = divideUp(columns, kBlockDimX1, kBlockShiftX1);
  grid.y  = divideUp(rows, kBlockDimY1, kBlockShiftY1);

  subtractKernel<<<grid, block, 0, stream>>>(src0, rows, columns, src0_stride,
      src1, src1_stride, dst, dst_stride);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

/***************************** guidedFilter() ******************************/

/*
 * guide image: 1 channel.
 * input image: 1 channel.
 * output image: 1 channel.
 */
void guidedFilter_1to1(const float* src, int src_rows, int src_cols,
                       int src_stride, const float* guide, int guide_stride,
                       float* dst, int dst_stride, int radius, double eps,
                       BorderType border_type, cudaStream_t stream) {
  float* buffer;
  size_t pitch;

  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(src_cols * sizeof(float), src_rows * 8, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch  = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch, src_cols * sizeof(float),
                           src_rows * 8);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return;
    }
  }

  int offset = pitch * src_rows;
  float* II     = buffer;
  float* IP     = (float*)((uchar*)buffer + offset);
  float* meanI  = (float*)((uchar*)buffer + offset * 2);
  float* meanP  = (float*)((uchar*)buffer + offset * 3);
  float* meanII = (float*)((uchar*)buffer + offset * 4);
  float* meanIP = (float*)((uchar*)buffer + offset * 5);
  float* varI   = (float*)((uchar*)buffer + offset * 6);
  float* covIP  = (float*)((uchar*)buffer + offset * 7);

  multiply(guide, src_rows, src_cols, 1, guide_stride, guide, guide_stride, II,
           pitch, 1.f, stream);
  multiply(guide, src_rows, src_cols, 1, guide_stride, src, src_stride, IP,
           pitch, 1.f, stream);

  int side_length = (radius << 1) + 1;
  boxFilter(guide, src_rows, src_cols, 1, src_stride, side_length, side_length,
            true, meanI, pitch, border_type, stream);
  boxFilter(src, src_rows, src_cols, 1, src_stride, side_length, side_length,
            true, meanP, pitch, border_type, stream);
  boxFilter(II, src_rows, src_cols, 1, src_stride, side_length, side_length,
            true, meanII, pitch, border_type, stream);
  boxFilter(IP, src_rows, src_cols, 1, src_stride, side_length, side_length,
            true, meanIP, pitch, border_type, stream);

  float* meanII_mul = II;
  float* meanIP_mul = IP;
  multiply(meanI, src_rows, src_cols, 1, pitch, meanI, pitch, meanII_mul, pitch,
           1.f, stream);
  multiply(meanI, src_rows, src_cols, 1, pitch, meanP, pitch, meanIP_mul, pitch,
           1.f, stream);
  subtract(meanII, src_rows, src_cols, 1, pitch, meanII_mul, pitch, varI, pitch,
           stream);
  subtract(meanIP, src_rows, src_cols, 1, pitch, meanIP_mul, pitch, covIP,
           pitch, stream);

  float* a = meanII;
  float* b = meanIP;
  float* aMeanI = covIP;
  addScalar(varI, src_rows, src_cols, 1, pitch, eps, stream);
  divide(covIP, src_rows, src_cols, 1, pitch, varI, pitch, a, pitch, 1.f,
         stream);
  multiply(a, src_rows, src_cols, 1, pitch, meanI, pitch, aMeanI, pitch, 1.f,
           stream);
  subtract(meanP, src_rows, src_cols, 1, pitch, aMeanI, pitch, b, pitch,
           stream);

  float* meanA = II;
  float* meanB = IP;
  boxFilter(a, src_rows, src_cols, 1, src_stride, side_length, side_length,
            true, meanA, pitch, border_type, stream);
  boxFilter(b, src_rows, src_cols, 1, src_stride, side_length, side_length,
            true, meanB, pitch, border_type, stream);

  float* meanAI = meanI;
  multiply(meanA, src_rows, src_cols, 1, pitch, guide, guide_stride, meanAI,
           pitch, 1.f, stream);
  add(meanAI, src_rows, src_cols, 1, pitch, meanB, pitch, dst, dst_stride,
      stream);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return;
  }

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
  }
}

void filtering(const float* src, int rows, int cols, int src_channels,
               int src_stride, const float* guide, int guide_channels,
               int guide_stride, float* dst, int dst_stride, int radius,
               float eps, BorderType border_type, cudaStream_t stream) {
  if (guide_channels == 1) {
    if (src_channels == 1) {
      guidedFilter_1to1(src, rows, cols, src_stride, guide, guide_stride,
                        dst, dst_stride, radius, eps, border_type, stream);
    }
    else if (src_channels == 3) {  // src_channels == 3
      float* buffer;
      size_t pitch;

      cudaError_t code;
      GpuMemoryBlock buffer_block;
      if (memoryPoolUsed()) {
        pplCudaMallocPitch(cols * sizeof(float), rows * 6, buffer_block);
        buffer = (float*)(buffer_block.data);
        pitch  = buffer_block.pitch;
      }
      else {
        code = cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows * 6);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
          return;
        }
      }

      int offset = pitch * rows;
      float* src0 = buffer;
      float* src1 = (float*)((uchar*)buffer + offset);
      float* src2 = (float*)((uchar*)buffer + offset * 2);
      float* dst0 = (float*)((uchar*)buffer + offset * 3);
      float* dst1 = (float*)((uchar*)buffer + offset * 4);
      float* dst2 = (float*)((uchar*)buffer + offset * 5);

      split3Channels(src, rows, cols, src_stride, src0, src1, src2, pitch,
                     stream);
      guidedFilter_1to1(src0, rows, cols, pitch, guide, guide_stride, dst0,
                        pitch, radius, eps, border_type, stream);
      guidedFilter_1to1(src1, rows, cols, pitch, guide, guide_stride, dst1,
                        pitch, radius, eps, border_type, stream);
      guidedFilter_1to1(src2, rows, cols, pitch, guide, guide_stride, dst2,
                        pitch, radius, eps, border_type, stream);
      merge3Channels(dst0, dst1, dst2, rows, cols, pitch, dst, dst_stride,
                     stream);

      code = cudaGetLastError();
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return;
      }

      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block);
      }
      else {
        cudaFree(buffer);
      }
    }
    else {  // src_channels == 4
      float* buffer;
      size_t pitch;

      cudaError_t code;
      GpuMemoryBlock buffer_block;
      if (memoryPoolUsed()) {
        pplCudaMallocPitch(cols * sizeof(float), rows * 8, buffer_block);
        buffer = (float*)(buffer_block.data);
        pitch  = buffer_block.pitch;
      }
      else {
        code = cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows * 8);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
          return;
        }
      }

      int offset = pitch * rows;
      float* src0 = buffer;
      float* src1 = (float*)((uchar*)buffer + offset);
      float* src2 = (float*)((uchar*)buffer + offset * 2);
      float* src3 = (float*)((uchar*)buffer + offset * 3);
      float* dst0 = (float*)((uchar*)buffer + offset * 4);
      float* dst1 = (float*)((uchar*)buffer + offset * 5);
      float* dst2 = (float*)((uchar*)buffer + offset * 6);
      float* dst3 = (float*)((uchar*)buffer + offset * 7);

      split4Channels(src, rows, cols, src_stride, src0, src1, src2, src3, pitch,
                     stream);
      guidedFilter_1to1(src0, rows, cols, pitch, guide, guide_stride, dst0,
                        pitch, radius, eps, border_type, stream);
      guidedFilter_1to1(src1, rows, cols, pitch, guide, guide_stride, dst1,
                        pitch, radius, eps, border_type, stream);
      guidedFilter_1to1(src2, rows, cols, pitch, guide, guide_stride, dst2,
                        pitch, radius, eps, border_type, stream);
      guidedFilter_1to1(src3, rows, cols, pitch, guide, guide_stride, dst3,
                        pitch, radius, eps, border_type, stream);
      merge4Channels(dst0, dst1, dst2, dst3, rows, cols, pitch, dst, dst_stride,
                     stream);

      code = cudaGetLastError();
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return;
      }

      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block);
      }
      else {
        cudaFree(buffer);
      }
    }
  }
  else {  // guide_channels == 3
    if (src_channels == 1) {
    }
    else if (src_channels == 3) { // src_channels == 3
    }
    else { // src_channels == 4
    }
  }
}

RetCode guidedFilter(const uchar* src, int rows, int cols, int src_channels,
                     int src_stride, const uchar* guide, int guide_channels,
                     int guide_stride, uchar* dst, int dst_stride, int radius,
                     float eps, BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(guide != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));
  PPL_ASSERT(guide_stride >= cols * guide_channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * src_channels * (int)sizeof(uchar));
  PPL_ASSERT(guide_channels == 1);
  PPL_ASSERT(src_channels == 1 || src_channels == 3 || src_channels == 4);
  PPL_ASSERT(radius > 0);
  PPL_ASSERT(eps > 0.0);
  PPL_ASSERT(border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_REFLECT);

  float* fguide;
  float* fsrc;
  float* fdst;
  size_t fguide_stride, fsrc_stride, fdst_stride;

  cudaError_t code;
  GpuMemoryBlock buffer_block0, buffer_block1, buffer_block2;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * guide_channels * sizeof(float), rows,
                       buffer_block0);
    pplCudaMallocPitch(cols * src_channels * sizeof(float), rows,
                       buffer_block1);
    pplCudaMallocPitch(cols * src_channels * sizeof(float), rows,
                       buffer_block2);
    fguide = (float*)(buffer_block0.data);
    fsrc   = (float*)(buffer_block1.data);
    fdst   = (float*)(buffer_block2.data);
    fguide_stride = buffer_block0.pitch;
    fsrc_stride   = buffer_block1.pitch;
    fdst_stride   = buffer_block2.pitch;
  }
  else {
    code = cudaMallocPitch(&fguide, &fguide_stride,
                           cols * guide_channels * sizeof(float), rows);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
    code = cudaMallocPitch(&fsrc, &fsrc_stride,
                           cols * src_channels * sizeof(float), rows);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
    code = cudaMallocPitch(&fdst, &fdst_stride,
                           cols * src_channels * sizeof(float), rows);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return RC_DEVICE_MEMORY_ERROR;
    }
  }

  convertTo(guide, rows, cols, guide_channels, guide_stride, fguide,
            fguide_stride, 1, 0.0, stream);
  convertTo(src, rows, cols, src_channels, src_stride, fsrc, fsrc_stride,
            1, 0.0, stream);

  filtering(fsrc, rows, cols, src_channels, fsrc_stride, fguide, guide_channels,
            fguide_stride, fdst, fdst_stride, radius, eps, border_type, stream);

  convertTo(fdst, rows, cols, src_channels, fdst_stride, dst, dst_stride,
            1, 0.0, stream);

  code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block0);
    pplCudaFree(buffer_block1);
    pplCudaFree(buffer_block2);
  }
  else {
    cudaFree(fguide);
    cudaFree(fsrc);
    cudaFree(fdst);
  }

  return RC_SUCCESS;
}

RetCode guidedFilter(const float* src, int rows, int cols, int src_channels,
                     int src_stride, const float* guide, int guide_channels,
                     int guide_stride, float* dst, int dst_stride, int radius,
                     float eps, BorderType border_type, cudaStream_t stream) {
  PPL_ASSERT(src != nullptr);
  PPL_ASSERT(guide != nullptr);
  PPL_ASSERT(dst != nullptr);
  PPL_ASSERT(rows >= 1 && cols >= 1);
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(float));
  PPL_ASSERT(guide_stride >= cols * guide_channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * src_channels * (int)sizeof(float));
  PPL_ASSERT(guide_channels == 1);
  PPL_ASSERT(src_channels == 1 || src_channels == 3 || src_channels == 4);
  PPL_ASSERT(radius > 0);
  PPL_ASSERT(eps > 0.0);
  PPL_ASSERT(border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_REFLECT);

  filtering(src, rows, cols, src_channels, src_stride, guide, guide_channels,
            guide_stride, dst, dst_stride, radius, eps, border_type, stream);

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    return RC_DEVICE_RUNTIME_ERROR;
  }

  return RC_SUCCESS;
}

template <>
RetCode GuidedFilter<uchar, 1, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(inData, height, width, 1, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<uchar, 3, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(inData, height, width, 3, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<uchar, 4, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const uchar* inData,
                                  int guideWidthStride,
                                  const uchar* guideData,
                                  int outWidthStride,
                                  uchar* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  RetCode code = guidedFilter(inData, height, width, 4, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 1, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(inData, height, width, 1, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 3, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(inData, height, width, 3, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 4, 1>(cudaStream_t stream,
                                  int height,
                                  int width,
                                  int inWidthStride,
                                  const float* inData,
                                  int guideWidthStride,
                                  const float* guideData,
                                  int outWidthStride,
                                  float* outData,
                                  int radius,
                                  float eps,
                                  BorderType border_type) {
  inWidthStride    *= sizeof(float);
  guideWidthStride *= sizeof(float);
  outWidthStride   *= sizeof(float);
  RetCode code = guidedFilter(inData, height, width, 4, inWidthStride,
                              guideData, 1, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
