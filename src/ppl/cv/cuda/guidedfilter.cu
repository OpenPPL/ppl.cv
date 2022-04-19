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

RetCode add(const float* src0, int rows, int cols, int channels,
            int src0_stride, const float* src1, int src1_stride, float* dst,
            int dst_stride, cudaStream_t stream);

RetCode multiply(const float* src0, int rows, int cols, int channels,
                 int src0_stride, const float* src1, int src1_stride,
                 float* dst, int dst_stride, float scale, cudaStream_t stream);

RetCode divide(const float* src0, int rows, int cols, int channels,
               int src0_stride, const float* src1, int src1_stride,
               float* dst, int dst_stride, float scale, cudaStream_t stream);

RetCode mla(const float* src0, int rows, int cols, int channels,
            int src0_stride, const float* src1, int src1_stride, float* dst,
            int dst_stride, cudaStream_t stream);

RetCode mls(const float* src0, int rows, int cols, int channels,
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
 */
void initialize1to1(const float* guide, int rows, int cols, int guide_stride,
                    int ksize, float eps, float* mean_i, float* var_i,
                    int pitch0, BorderType border_type, cudaStream_t stream) {
  float* buffer;
  size_t pitch1;

  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * sizeof(float), rows, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch1 = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch1, cols * sizeof(float), rows);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return;
    }
  }

  float* ii = buffer;
  float* mean_ii = var_i;

  multiply(guide, rows, cols, 1, guide_stride, guide, guide_stride, ii, pitch1,
           1.f, stream);
  boxFilter(ii, rows, cols, 1, pitch1, ksize, ksize, true, mean_ii, pitch0,
            border_type, stream);
  boxFilter(guide, rows, cols, 1, guide_stride, ksize, ksize, true, mean_i,
            pitch0, border_type, stream);
  mls(mean_i, rows, cols, 1, pitch0, mean_i, pitch0, var_i, pitch0, stream);
  addScalar(var_i, rows, cols, 1, pitch0, eps, stream);

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
  }
}

/*
 * guide image: 3 channel.
 */
void initialize3to1(const float* guide, int rows, int cols, int guide_stride,
                    int ksize, float eps, float* guide0, float* guide1,
                    float* guide2, float* mean_guide0, float* mean_guide1,
                    float* mean_guide2, float* inv_var_guide00,
                    float* inv_var_guide01, float* inv_var_guide02,
                    float* inv_var_guide11, float* inv_var_guide12,
                    float* inv_var_guide22, int pitch0, BorderType border_type,
                    cudaStream_t stream) {
  split3Channels(guide, rows, cols, guide_stride, guide0, guide1, guide2,
                 pitch0, stream);
  boxFilter(guide0, rows, cols, 1, pitch0, ksize, ksize, true, mean_guide0,
            pitch0, border_type, stream);
  boxFilter(guide1, rows, cols, 1, pitch0, ksize, ksize, true, mean_guide1,
            pitch0, border_type, stream);
  boxFilter(guide2, rows, cols, 1, pitch0, ksize, ksize, true, mean_guide2,
            pitch0, border_type, stream);

  float* buffer;
  size_t pitch1;

  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * sizeof(float), rows * 12, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch1 = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch1, cols * sizeof(float), rows * 12);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return;
    }
  }

  int offset = pitch1 * rows;
  float* guide00 = buffer;
  float* guide01 = (float*)((uchar*)buffer + offset);
  float* guide02 = (float*)((uchar*)buffer + offset * 2);
  float* guide11 = (float*)((uchar*)buffer + offset * 3);
  float* guide12 = (float*)((uchar*)buffer + offset * 4);
  float* guide22 = (float*)((uchar*)buffer + offset * 5);
  float* var_guide00 = (float*)((uchar*)buffer + offset * 6);
  float* var_guide01 = (float*)((uchar*)buffer + offset * 7);
  float* var_guide02 = (float*)((uchar*)buffer + offset * 8);
  float* var_guide11 = (float*)((uchar*)buffer + offset * 9);
  float* var_guide12 = (float*)((uchar*)buffer + offset * 10);
  float* var_guide22 = (float*)((uchar*)buffer + offset * 11);

  multiply(guide0, rows, cols, 1, pitch0, guide0, pitch0, guide00, pitch1, 1.f,
           stream);
  multiply(guide0, rows, cols, 1, pitch0, guide1, pitch0, guide01, pitch1, 1.f,
           stream);
  multiply(guide0, rows, cols, 1, pitch0, guide2, pitch0, guide02, pitch1, 1.f,
           stream);
  multiply(guide1, rows, cols, 1, pitch0, guide1, pitch0, guide11, pitch1, 1.f,
           stream);
  multiply(guide1, rows, cols, 1, pitch0, guide2, pitch0, guide12, pitch1, 1.f,
           stream);
  multiply(guide2, rows, cols, 1, pitch0, guide2, pitch0, guide22, pitch1, 1.f,
           stream);

  boxFilter(guide00, rows, cols, 1, pitch1, ksize, ksize, true, var_guide00,
            pitch1, border_type, stream);
  boxFilter(guide01, rows, cols, 1, pitch1, ksize, ksize, true, var_guide01,
            pitch1, border_type, stream);
  boxFilter(guide02, rows, cols, 1, pitch1, ksize, ksize, true, var_guide02,
            pitch1, border_type, stream);
  boxFilter(guide11, rows, cols, 1, pitch1, ksize, ksize, true, var_guide11,
            pitch1, border_type, stream);
  boxFilter(guide12, rows, cols, 1, pitch1, ksize, ksize, true, var_guide12,
            pitch1, border_type, stream);
  boxFilter(guide22, rows, cols, 1, pitch1, ksize, ksize, true, var_guide22,
            pitch1, border_type, stream);

  mls(mean_guide0, rows, cols, 1, pitch0, mean_guide0, pitch0, var_guide00,
      pitch1, stream);
  mls(mean_guide0, rows, cols, 1, pitch0, mean_guide1, pitch0, var_guide01,
      pitch1, stream);
  mls(mean_guide0, rows, cols, 1, pitch0, mean_guide2, pitch0, var_guide02,
      pitch1, stream);
  mls(mean_guide1, rows, cols, 1, pitch0, mean_guide1, pitch0, var_guide11,
      pitch1, stream);
  mls(mean_guide1, rows, cols, 1, pitch0, mean_guide2, pitch0, var_guide12,
      pitch1, stream);
  mls(mean_guide2, rows, cols, 1, pitch0, mean_guide2, pitch0, var_guide22,
      pitch1, stream);

  addScalar(var_guide00, rows, cols, 1, pitch1, eps, stream);
  addScalar(var_guide11, rows, cols, 1, pitch1, eps, stream);
  addScalar(var_guide22, rows, cols, 1, pitch1, eps, stream);

  multiply(var_guide11, rows, cols, 1, pitch1, var_guide22, pitch1,
           inv_var_guide00, pitch0, 1.f, stream);
  multiply(var_guide12, rows, cols, 1, pitch1, var_guide02, pitch1,
           inv_var_guide01, pitch0, 1.f, stream);
  multiply(var_guide01, rows, cols, 1, pitch1, var_guide12, pitch1,
           inv_var_guide02, pitch0, 1.f, stream);
  multiply(var_guide00, rows, cols, 1, pitch1, var_guide22, pitch1,
           inv_var_guide11, pitch0, 1.f, stream);
  multiply(var_guide02, rows, cols, 1, pitch1, var_guide01, pitch1,
           inv_var_guide12, pitch0, 1.f, stream);
  multiply(var_guide00, rows, cols, 1, pitch1, var_guide11, pitch1,
           inv_var_guide22, pitch0, 1.f, stream);

  mls(var_guide12, rows, cols, 1, pitch1, var_guide12, pitch1, inv_var_guide00,
      pitch0, stream);
  mls(var_guide01, rows, cols, 1, pitch1, var_guide22, pitch1, inv_var_guide01,
      pitch0, stream);
  mls(var_guide11, rows, cols, 1, pitch1, var_guide02, pitch1, inv_var_guide02,
      pitch0, stream);
  mls(var_guide02, rows, cols, 1, pitch1, var_guide02, pitch1, inv_var_guide11,
      pitch0, stream);
  mls(var_guide00, rows, cols, 1, pitch1, var_guide12, pitch1, inv_var_guide12,
      pitch0, stream);
  mls(var_guide01, rows, cols, 1, pitch1, var_guide01, pitch1, inv_var_guide22,
      pitch0, stream);

  float* conv_det = guide00;
  multiply(inv_var_guide00, rows, cols, 1, pitch0, var_guide00, pitch1, conv_det,
           pitch1, 1.f, stream);
  mla(inv_var_guide01, rows, cols, 1, pitch0, var_guide01, pitch1, conv_det,
      pitch1, stream);
  mla(inv_var_guide02, rows, cols, 1, pitch0, var_guide02, pitch1, conv_det,
      pitch1, stream);

  divide(inv_var_guide00, rows, cols, 1, pitch0, conv_det, pitch1,
         inv_var_guide00, pitch0, 1.f, stream);
  divide(inv_var_guide01, rows, cols, 1, pitch0, conv_det, pitch1,
         inv_var_guide01, pitch0, 1.f, stream);
  divide(inv_var_guide02, rows, cols, 1, pitch0, conv_det, pitch1,
         inv_var_guide02, pitch0, 1.f, stream);
  divide(inv_var_guide11, rows, cols, 1, pitch0, conv_det, pitch1,
         inv_var_guide11, pitch0, 1.f, stream);
  divide(inv_var_guide12, rows, cols, 1, pitch0, conv_det, pitch1,
         inv_var_guide12, pitch0, 1.f, stream);
  divide(inv_var_guide22, rows, cols, 1, pitch0, conv_det, pitch1,
         inv_var_guide22, pitch0, 1.f, stream);

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
  }
}

/*
 * input image: 1 channel.
 * guide image: 1 channel.
 * output image: 1 channel.
 */
void filter1to1(const float* src, int rows, int cols, int src_stride,
                const float* guide, int guide_stride, const float* mean_i,
                const float* var_i, int pitch0, float* dst, int dst_stride,
                int ksize, BorderType border_type, cudaStream_t stream) {
  float* buffer;
  size_t pitch1;

  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * sizeof(float), rows * 3, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch1 = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch1, cols * sizeof(float), rows * 3);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return;
    }
  }

  int offset = pitch1 * rows;
  float* ip      = buffer;
  float* mean_p  = (float*)((uchar*)buffer + offset);
  float* mean_ip = (float*)((uchar*)buffer + offset * 2);

  multiply(guide, rows, cols, 1, guide_stride, src, src_stride, ip, pitch1, 1.f,
           stream);
  boxFilter(src, rows, cols, 1, src_stride, ksize, ksize, true, mean_p, pitch1,
            border_type, stream);
  boxFilter(ip, rows, cols, 1, pitch1, ksize, ksize, true, mean_ip, pitch1,
            border_type, stream);
  mls(mean_i, rows, cols, 1, pitch0, mean_p, pitch1, mean_ip, pitch1, stream);

  float* alpha = ip;
  float* beta  = mean_p;
  divide(mean_ip, rows, cols, 1, pitch1, var_i, pitch0, alpha, pitch1, 1.f,
         stream);
  mls(alpha, rows, cols, 1, pitch1, mean_i, pitch0, beta, pitch1, stream);

  float* mean_alpha = mean_ip;
  float* mean_beta = dst;
  boxFilter(alpha, rows, cols, 1, pitch1, ksize, ksize, true, mean_alpha,
            pitch1, border_type, stream);
  boxFilter(beta, rows, cols, 1, pitch1, ksize, ksize, true, mean_beta,
            dst_stride, border_type, stream);
  mla(mean_alpha, rows, cols, 1, pitch1, guide, guide_stride, dst, dst_stride,
      stream);

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block);
  }
  else {
    cudaFree(buffer);
  }
}

/*
 * input image: 1 channel.
 * guide image: 3 channel.
 * output image: 1 channel.
 */
void filter3to1(const float* src, int rows, int cols, int src_stride, int ksize,
                const float* guide0, const float* guide1, const float* guide2,
                const float* mean_guide0, const float* mean_guide1,
                const float* mean_guide2, const float* inv_var_guide00,
                const float* inv_var_guide01, const float* inv_var_guide02,
                const float* inv_var_guide11, const float* inv_var_guide12,
                const float* inv_var_guide22, int pitch0, float* dst,
                int dst_stride, BorderType border_type, cudaStream_t stream) {
  float* buffer;
  size_t pitch1;

  cudaError_t code;
  GpuMemoryBlock buffer_block;
  if (memoryPoolUsed()) {
    pplCudaMallocPitch(cols * sizeof(float), rows * 7, buffer_block);
    buffer = (float*)(buffer_block.data);
    pitch1 = buffer_block.pitch;
  }
  else {
    code = cudaMallocPitch(&buffer, &pitch1, cols * sizeof(float), rows * 7);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
      return;
    }
  }

  int offset = pitch1 * rows;
  float* mean_p = buffer;
  float* guide0_p = (float*)((uchar*)buffer + offset);
  float* guide1_p = (float*)((uchar*)buffer + offset * 2);
  float* guide2_p = (float*)((uchar*)buffer + offset * 3);
  float* mean_guide0_p = (float*)((uchar*)buffer + offset * 4);
  float* mean_guide1_p = (float*)((uchar*)buffer + offset * 5);
  float* mean_guide2_p = (float*)((uchar*)buffer + offset * 6);

  boxFilter(src, rows, cols, 1, src_stride, ksize, ksize, true, mean_p, pitch1,
            border_type, stream);
  multiply(guide0, rows, cols, 1, pitch0, src, src_stride, guide0_p, pitch1,
           1.f, stream);
  multiply(guide1, rows, cols, 1, pitch0, src, src_stride, guide1_p, pitch1,
           1.f, stream);
  multiply(guide2, rows, cols, 1, pitch0, src, src_stride, guide2_p, pitch1,
           1.f, stream);
  boxFilter(guide0_p, rows, cols, 1, pitch1, ksize, ksize, true, mean_guide0_p,
            pitch1, border_type, stream);
  boxFilter(guide1_p, rows, cols, 1, pitch1, ksize, ksize, true, mean_guide1_p,
            pitch1, border_type, stream);
  boxFilter(guide2_p, rows, cols, 1, pitch1, ksize, ksize, true, mean_guide2_p,
            pitch1, border_type, stream);

  float* cov_guide0_p = mean_guide0_p;
  float* cov_guide1_p = mean_guide1_p;
  float* cov_guide2_p = mean_guide2_p;
  mls(mean_guide0, rows, cols, 1, pitch0, mean_p, pitch1, cov_guide0_p, pitch1,
      stream);
  mls(mean_guide1, rows, cols, 1, pitch0, mean_p, pitch1, cov_guide1_p, pitch1,
      stream);
  mls(mean_guide2, rows, cols, 1, pitch0, mean_p, pitch1, cov_guide2_p, pitch1,
      stream);

  float* alpha0 = guide0_p;
  float* alpha1 = guide1_p;
  float* alpha2 = guide2_p;
  multiply(cov_guide0_p, rows, cols, 1, pitch1, inv_var_guide00, pitch0, alpha0,
           pitch1, 1.f, stream);
  multiply(cov_guide0_p, rows, cols, 1, pitch1, inv_var_guide01, pitch0, alpha1,
           pitch1, 1.f, stream);
  multiply(cov_guide0_p, rows, cols, 1, pitch1, inv_var_guide02, pitch0, alpha2,
           pitch1, 1.f, stream);

  mla(cov_guide1_p, rows, cols, 1, pitch1, inv_var_guide01, pitch0, alpha0,
      pitch1, stream);
  mla(cov_guide1_p, rows, cols, 1, pitch1, inv_var_guide11, pitch0, alpha1,
      pitch1, stream);
  mla(cov_guide1_p, rows, cols, 1, pitch1, inv_var_guide12, pitch0, alpha2,
      pitch1, stream);

  mla(cov_guide2_p, rows, cols, 1, pitch1, inv_var_guide02, pitch0, alpha0,
      pitch1, stream);
  mla(cov_guide2_p, rows, cols, 1, pitch1, inv_var_guide12, pitch0, alpha1,
      pitch1, stream);
  mla(cov_guide2_p, rows, cols, 1, pitch1, inv_var_guide22, pitch0, alpha2,
      pitch1, stream);

  float* beta = mean_p;
  mls(alpha0, rows, cols, 1, pitch1, mean_guide0, pitch1, beta, pitch1, stream);
  mls(alpha1, rows, cols, 1, pitch1, mean_guide1, pitch1, beta, pitch1, stream);
  mls(alpha2, rows, cols, 1, pitch1, mean_guide2, pitch1, beta, pitch1, stream);

  float* mean_alpha0 = mean_guide0_p;
  float* mean_alpha1 = mean_guide1_p;
  float* mean_alpha2 = mean_guide2_p;
  boxFilter(alpha0, rows, cols, 1, pitch1, ksize, ksize, true, mean_alpha0,
            pitch1, border_type, stream);
  boxFilter(alpha1, rows, cols, 1, pitch1, ksize, ksize, true, mean_alpha1,
            pitch1, border_type, stream);
  boxFilter(alpha2, rows, cols, 1, pitch1, ksize, ksize, true, mean_alpha2,
            pitch1, border_type, stream);

  multiply(mean_alpha0, rows, cols, 1, pitch1, guide0, pitch0, dst, dst_stride,
           1.f, stream);
  mla(mean_alpha1, rows, cols, 1, pitch1, guide1, pitch0, dst, dst_stride,
      stream);
  mla(mean_alpha2, rows, cols, 1, pitch1, guide2, pitch0, dst, dst_stride,
      stream);

  float* mean_beta = mean_guide0_p;
  boxFilter(beta, rows, cols, 1, pitch1, ksize, ksize, true, mean_beta,
            pitch1, border_type, stream);
  add(dst, rows, cols, 1, dst_stride, mean_beta, pitch1, dst, dst_stride,
      stream);

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
  int ksize = (radius << 1) + 1;
  GpuMemoryBlock buffer_block0;
  float* buffer0;
  size_t pitch0;

  cudaError_t code;
  if (guide_channels == 1) {
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
    float* mean_i = buffer0;
    float* var_i  = (float*)((uchar*)buffer0 + offset);
    initialize1to1(guide, rows, cols, guide_stride, ksize, eps, mean_i, var_i,
                   pitch0, border_type, stream);

    if (src_channels == 1) {
      filter1to1(src, rows, cols, src_stride, guide, guide_stride, mean_i,
                 var_i, pitch0, dst, dst_stride, ksize, border_type, stream);
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
      filter1to1(src0, rows, cols, pitch1, guide, guide_stride, mean_i, var_i,
                 pitch0, dst0, pitch1, ksize, border_type, stream);
      filter1to1(src1, rows, cols, pitch1, guide, guide_stride, mean_i, var_i,
                 pitch0, dst1, pitch1, ksize, border_type, stream);
      filter1to1(src2, rows, cols, pitch1, guide, guide_stride, mean_i, var_i,
                 pitch0, dst2, pitch1, ksize, border_type, stream);
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
      filter1to1(src0, rows, cols, pitch1, guide, guide_stride, mean_i, var_i,
                 pitch0, dst0, pitch1, ksize, border_type, stream);
      filter1to1(src1, rows, cols, pitch1, guide, guide_stride, mean_i, var_i,
                 pitch0, dst1, pitch1, ksize, border_type, stream);
      filter1to1(src2, rows, cols, pitch1, guide, guide_stride, mean_i, var_i,
                 pitch0, dst2, pitch1, ksize, border_type, stream);
      filter1to1(src3, rows, cols, pitch1, guide, guide_stride, mean_i, var_i,
                 pitch0, dst3, pitch1, ksize, border_type, stream);
      merge4Channels(dst0, dst1, dst2, dst3, rows, cols, pitch1, dst,
                     dst_stride, stream);

      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block1);
      }
      else {
        cudaFree(buffer1);
      }
    }
  }
  else {  // guide_channels == 3
    if (memoryPoolUsed()) {
      pplCudaMallocPitch(cols * sizeof(float), rows * 12, buffer_block0);
      buffer0 = (float*)(buffer_block0.data);
      pitch0  = buffer_block0.pitch;
    }
    else {
      code = cudaMallocPitch(&buffer0, &pitch0, cols * sizeof(float),
                             rows * 12);
      if (code != cudaSuccess) {
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
        return;
      }
    }

    int offset = pitch0 * rows;
    float* guide0 = buffer0;
    float* guide1 = (float*)((uchar*)buffer0 + offset);
    float* guide2 = (float*)((uchar*)buffer0 + offset * 2);
    float* mean_guide0 = (float*)((uchar*)buffer0 + offset * 3);
    float* mean_guide1 = (float*)((uchar*)buffer0 + offset * 4);
    float* mean_guide2 = (float*)((uchar*)buffer0 + offset * 5);
    float* inv_var_guide00 = (float*)((uchar*)buffer0 + offset * 6);
    float* inv_var_guide01 = (float*)((uchar*)buffer0 + offset * 7);
    float* inv_var_guide02 = (float*)((uchar*)buffer0 + offset * 8);
    float* inv_var_guide11 = (float*)((uchar*)buffer0 + offset * 9);
    float* inv_var_guide12 = (float*)((uchar*)buffer0 + offset * 10);
    float* inv_var_guide22 = (float*)((uchar*)buffer0 + offset * 11);
    initialize3to1(guide, rows, cols, guide_stride, ksize, eps, guide0, guide1,
                   guide2, mean_guide0, mean_guide1, mean_guide2,
                   inv_var_guide00, inv_var_guide01, inv_var_guide02,
                   inv_var_guide11, inv_var_guide12, inv_var_guide22, pitch0,
                   border_type, stream);

    if (src_channels == 1) {
      filter3to1(src, rows, cols, src_stride, ksize, guide0, guide1, guide2,
                 mean_guide0, mean_guide1, mean_guide2, inv_var_guide00,
                 inv_var_guide01, inv_var_guide02, inv_var_guide11,
                 inv_var_guide12, inv_var_guide22, pitch0, dst, dst_stride,
                 border_type, stream);
    }
    else if (src_channels == 3) { // src_channels == 3
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
      filter3to1(src0, rows, cols, pitch1, ksize, guide0, guide1, guide2,
                 mean_guide0, mean_guide1, mean_guide2, inv_var_guide00,
                 inv_var_guide01, inv_var_guide02, inv_var_guide11,
                 inv_var_guide12, inv_var_guide22, pitch0, dst0, pitch1,
                 border_type, stream);
      filter3to1(src1, rows, cols, pitch1, ksize, guide0, guide1, guide2,
                 mean_guide0, mean_guide1, mean_guide2, inv_var_guide00,
                 inv_var_guide01, inv_var_guide02, inv_var_guide11,
                 inv_var_guide12, inv_var_guide22, pitch0, dst1, pitch1,
                 border_type, stream);
      filter3to1(src2, rows, cols, pitch1, ksize, guide0, guide1, guide2,
                 mean_guide0, mean_guide1, mean_guide2, inv_var_guide00,
                 inv_var_guide01, inv_var_guide02, inv_var_guide11,
                 inv_var_guide12, inv_var_guide22, pitch0, dst2, pitch1,
                 border_type, stream);
      merge3Channels(dst0, dst1, dst2, rows, cols, pitch1, dst, dst_stride,
                     stream);

      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block1);
      }
      else {
        cudaFree(buffer1);
      }
    }
    else { // src_channels == 4
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
      filter3to1(src0, rows, cols, pitch1, ksize, guide0, guide1, guide2,
                 mean_guide0, mean_guide1, mean_guide2, inv_var_guide00,
                 inv_var_guide01, inv_var_guide02, inv_var_guide11,
                 inv_var_guide12, inv_var_guide22, pitch0, dst0, pitch1,
                 border_type, stream);
      filter3to1(src1, rows, cols, pitch1, ksize, guide0, guide1, guide2,
                 mean_guide0, mean_guide1, mean_guide2, inv_var_guide00,
                 inv_var_guide01, inv_var_guide02, inv_var_guide11,
                 inv_var_guide12, inv_var_guide22, pitch0, dst1, pitch1,
                 border_type, stream);
      filter3to1(src2, rows, cols, pitch1, ksize, guide0, guide1, guide2,
                 mean_guide0, mean_guide1, mean_guide2, inv_var_guide00,
                 inv_var_guide01, inv_var_guide02, inv_var_guide11,
                 inv_var_guide12, inv_var_guide22, pitch0, dst2, pitch1,
                 border_type, stream);
      filter3to1(src3, rows, cols, pitch1, ksize, guide0, guide1, guide2,
                 mean_guide0, mean_guide1, mean_guide2, inv_var_guide00,
                 inv_var_guide01, inv_var_guide02, inv_var_guide11,
                 inv_var_guide12, inv_var_guide22, pitch0, dst3, pitch1,
                 border_type, stream);
      merge4Channels(dst0, dst1, dst2, dst3, rows, cols, pitch1, dst,
                     dst_stride, stream);

      if (memoryPoolUsed()) {
        pplCudaFree(buffer_block1);
      }
      else {
        cudaFree(buffer1);
      }
    }
  }

  if (memoryPoolUsed()) {
    pplCudaFree(buffer_block0);
  }
  else {
    cudaFree(buffer0);
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
  PPL_ASSERT(guide_channels == 1 || guide_channels == 3);
  PPL_ASSERT(src_channels == 1 || src_channels == 3 || src_channels == 4);
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(uchar));
  PPL_ASSERT(guide_stride >= cols * guide_channels * (int)sizeof(uchar));
  PPL_ASSERT(dst_stride >= cols * src_channels * (int)sizeof(uchar));
  PPL_ASSERT(radius > 0);
  PPL_ASSERT(eps > 0.0);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

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
  PPL_ASSERT(guide_channels == 1 || guide_channels == 3);
  PPL_ASSERT(src_channels == 1 || src_channels == 3 || src_channels == 4);
  PPL_ASSERT(src_stride >= cols * src_channels * (int)sizeof(float));
  PPL_ASSERT(guide_stride >= cols * guide_channels * (int)sizeof(float));
  PPL_ASSERT(dst_stride >= cols * src_channels * (int)sizeof(float));
  PPL_ASSERT(radius > 0);
  PPL_ASSERT(eps > 0.0);
  PPL_ASSERT(border_type == BORDER_REPLICATE ||
             border_type == BORDER_REFLECT ||
             border_type == BORDER_REFLECT_101 ||
             border_type == BORDER_DEFAULT);

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
RetCode GuidedFilter<uchar, 1, 3>(cudaStream_t stream,
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
                              guideData, 3, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<uchar, 3, 3>(cudaStream_t stream,
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
                              guideData, 3, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<uchar, 4, 3>(cudaStream_t stream,
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
                              guideData, 3, guideWidthStride, outData,
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

template <>
RetCode GuidedFilter<float, 1, 3>(cudaStream_t stream,
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
                              guideData, 3, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 3, 3>(cudaStream_t stream,
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
                              guideData, 3, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

template <>
RetCode GuidedFilter<float, 4, 3>(cudaStream_t stream,
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
                              guideData, 3, guideWidthStride, outData,
                              outWidthStride, radius, eps, border_type, stream);

  return code;
}

}  // cuda
}  // cv
}  // ppl
