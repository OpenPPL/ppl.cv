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
void initialize1to1(const float* guide, int rows, int cols, int guide_stride,
                    int radius, float eps, float* meanI, float* varI,
                    int i_pitch, BorderType border_type, cudaStream_t stream) {
  float* buffer;
  size_t pitch;

  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * sizeof(float), rows * 2, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch  = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows * 2);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return;
    }
  }

  int offset = pitch * rows;
  float* II     = buffer;
  float* meanII = (float*)((uchar*)buffer + offset);

  int ksize = (radius << 1) + 1;
  multiply(guide, rows, cols, 1, guide_stride, guide, guide_stride, II, pitch,
           1.f, stream);
  boxFilter(II, rows, cols, 1, pitch, ksize, ksize, true, meanII, pitch,
            border_type, stream);
  boxFilter(guide, rows, cols, 1, guide_stride, ksize, ksize, true, meanI,
            i_pitch, border_type, stream);

  float* meanII_mul = II;
  multiply(meanI, rows, cols, 1, i_pitch, meanI, i_pitch, meanII_mul, pitch,
           1.f, stream);
  subtract(meanII, rows, cols, 1, pitch, meanII_mul, pitch, varI, i_pitch,
           stream);
  addScalar(varI, rows, cols, 1, i_pitch, eps, stream);

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
  }
}

/*
 * guide image: 1 channel.
 * input image: 1 channel.
 * output image: 1 channel.
 */
void filter1to1(const float* src, int rows, int cols, int src_stride,
                const float* guide, int guide_stride, float* meanI, float* varI,
                int i_pitch, float* dst, int dst_stride, int radius,
                BorderType border_type, cudaStream_t stream) {
  float* buffer;
  size_t pitch;

  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * sizeof(float), rows * 4, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch  = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch, cols * sizeof(float), rows * 4);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return;
    }
  }

  int offset = pitch * rows;
  float* IP     = buffer;
  float* meanP  = (float*)((uchar*)buffer + offset);
  float* meanIP = (float*)((uchar*)buffer + offset * 2);
  float* covIP  = (float*)((uchar*)buffer + offset * 3);

  int ksize = (radius << 1) + 1;
  multiply(guide, rows, cols, 1, guide_stride, src, src_stride, IP, pitch, 1.f,
           stream);
  boxFilter(src, rows, cols, 1, src_stride, ksize, ksize, true, meanP, pitch,
            border_type, stream);
  boxFilter(IP, rows, cols, 1, src_stride, ksize, ksize, true, meanIP, pitch,
            border_type, stream);

  float* meanIP_mul = IP;
  multiply(meanI, rows, cols, 1, i_pitch, meanP, pitch, meanIP_mul, pitch,
           1.f, stream);
  subtract(meanIP, rows, cols, 1, pitch, meanIP_mul, pitch, covIP, pitch,
           stream);

  float* a = IP;
  float* b = meanIP;
  float* aMeanI = covIP;
  divide(covIP, rows, cols, 1, pitch, varI, i_pitch, a, pitch, 1.f, stream);
  multiply(a, rows, cols, 1, pitch, meanI, i_pitch, aMeanI, pitch, 1.f, stream);
  subtract(meanP, rows, cols, 1, pitch, aMeanI, pitch, b, pitch, stream);

  float* meanA = meanP;
  float* meanB = IP;
  boxFilter(a, rows, cols, 1, src_stride, ksize, ksize, true, meanA, pitch,
            border_type, stream);
  boxFilter(b, rows, cols, 1, src_stride, ksize, ksize, true, meanB, pitch,
            border_type, stream);

  float* meanAI = meanIP;
  multiply(meanA, rows, cols, 1, pitch, guide, guide_stride, meanAI, pitch, 1.f,
           stream);
  add(meanAI, rows, cols, 1, pitch, meanB, pitch, dst, dst_stride, stream);

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
  cudaError_t code;
  if (guide_channels == 1) {
    float* buffer0;
    size_t pitch0;
    GpuMemoryBlock buffer_block0;
    if (memoryPoolUsed()) {
      pplCudaMallocPitch(cols * sizeof(float), rows * 2, buffer_block0);
      buffer0 = (float*)(buffer_block0.data);
      pitch0  = buffer_block0.pitch;
    }
    else {
      code = cudaMallocPitch(&buffer0, &pitch0, cols * sizeof(float), rows * 2);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return;
      }
    }

    int offset = pitch0 * rows;
    float* meanI = buffer0;
    float* varI  = (float*)((uchar*)buffer0 + offset);
    initialize1to1(guide, rows, cols, guide_stride, radius, eps, meanI, varI,
                   pitch0, border_type, stream);

    if (src_channels == 1) {
      filter1to1(src, rows, cols, src_stride, guide, guide_stride, meanI, varI,
                 pitch0, dst, dst_stride, radius, border_type, stream);
    }
    else if (src_channels == 3) {  // src_channels == 3
      float* buffer1;
      size_t pitch1;

      GpuMemoryBlock buffer_block1;
      if (memoryPoolUsed()) {
        pplCudaMallocPitch(cols * sizeof(float), rows * 6, buffer_block1);
        buffer1 = (float*)(buffer_block1.data);
        pitch1  = buffer_block1.pitch;
      }
      else {
        code = cudaMallocPitch(&buffer1, &pitch1, cols * sizeof(float),
                               rows * 6);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
          return;
        }
      }

      offset = pitch1 * rows;
      float* src0 = buffer1;
      float* src1 = (float*)((uchar*)buffer1 + offset);
      float* src2 = (float*)((uchar*)buffer1 + offset * 2);
      float* dst0 = (float*)((uchar*)buffer1 + offset * 3);
      float* dst1 = (float*)((uchar*)buffer1 + offset * 4);
      float* dst2 = (float*)((uchar*)buffer1 + offset * 5);

      split3Channels(src, rows, cols, src_stride, src0, src1, src2, pitch1,
                     stream);
      filter1to1(src0, rows, cols, pitch1, guide, guide_stride, meanI, varI,
                 pitch0, dst0, pitch1, radius, border_type, stream);
      filter1to1(src1, rows, cols, pitch1, guide, guide_stride, meanI, varI,
                 pitch0, dst1, pitch1, radius, border_type, stream);
      filter1to1(src2, rows, cols, pitch1, guide, guide_stride, meanI, varI,
                 pitch0, dst2, pitch1, radius, border_type, stream);
      merge3Channels(dst0, dst1, dst2, rows, cols, pitch1, dst, dst_stride,
                     stream);

      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block1);
      }
      else {
        cudaFree(buffer1);
      }
    }
    else {  // src_channels == 4
      float* buffer1;
      size_t pitch1;

      GpuMemoryBlock buffer_block1;
      if (memoryPoolUsed()) {
        pplCudaMallocPitch(cols * sizeof(float), rows * 8, buffer_block1);
        buffer1 = (float*)(buffer_block1.data);
        pitch1  = buffer_block1.pitch;
      }
      else {
        code = cudaMallocPitch(&buffer1, &pitch1, cols * sizeof(float),
                               rows * 8);
        if (code != cudaSuccess) {
          LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
          return;
        }
      }

      offset = pitch1 * rows;
      float* src0 = buffer1;
      float* src1 = (float*)((uchar*)buffer1 + offset);
      float* src2 = (float*)((uchar*)buffer1 + offset * 2);
      float* src3 = (float*)((uchar*)buffer1 + offset * 3);
      float* dst0 = (float*)((uchar*)buffer1 + offset * 4);
      float* dst1 = (float*)((uchar*)buffer1 + offset * 5);
      float* dst2 = (float*)((uchar*)buffer1 + offset * 6);
      float* dst3 = (float*)((uchar*)buffer1 + offset * 7);

      split4Channels(src, rows, cols, src_stride, src0, src1, src2, src3,
                     pitch1, stream);
      filter1to1(src0, rows, cols, pitch1, guide, guide_stride, meanI, varI,
                 pitch0, dst0, pitch1, radius, border_type, stream);
      filter1to1(src1, rows, cols, pitch1, guide, guide_stride, meanI, varI,
                 pitch0, dst1, pitch1, radius, border_type, stream);
      filter1to1(src2, rows, cols, pitch1, guide, guide_stride, meanI, varI,
                 pitch0, dst2, pitch1, radius, border_type, stream);
      filter1to1(src3, rows, cols, pitch1, guide, guide_stride, meanI, varI,
                 pitch0, dst3, pitch1, radius, border_type, stream);
      merge4Channels(dst0, dst1, dst2, dst3, rows, cols, pitch1, dst,
                     dst_stride, stream);

      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block1);
      }
      else {
        cudaFree(buffer1);
      }
    }

    if (memoryPoolUsed()) {
      pplCudaFree(buffer_block0);
    }
    else {
      cudaFree(buffer0);
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
