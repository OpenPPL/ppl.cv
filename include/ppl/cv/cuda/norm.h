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

#ifndef _ST_HPC_PPL_CV_CUDA_NORM_H_
#define _ST_HPC_PPL_CV_CUDA_NORM_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Calculates the absolute norm of an array.
 * @tparam T The data type, used for the source image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream          cuda stream object.
 * @param inHeight        input image's height.
 * @param inWidth         input image's width.
 * @param inWidthStride   input image's width stride, it is `width * channels`
 *                        for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                        for 2D cudaMallocPitch() allocated data.
 * @param inData          input image data.
 * @param normType        norm type. NORM_INF, NORM_L1, and NORM_L2 are
 *                        supported for now.
 * @param maskWidthStride the width stride of mask, similar to inWidthStride.
 * @param mask            optional operation mask; it must have the same size as
 *                        inData, and is uchar and single channel type.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 This function needs a memory buffer to store the norm values on GPU,
 *         which is not less than 2048 bytes. When CUDA Memory Pool is used,
 *         the capacity of CUDA Memory Pool must be not less than the size of
 *         the memory buffer.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uint8_t(uchar))<td>1
 * <tr><td>uint8_t(uint8_t(uchar))<td>3
 * <tr><td>uint8_t(uint8_t(uchar))<td>4
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/norm.h";
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/norm.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   size_t input_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   double norm_value;
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Norm<float, 3>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, &norm_value, ppl::cv::NORM_L2);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Norm(cudaStream_t stream,
                          int inHeight,
                          int inWidth,
                          int inWidthStride,
                          const T* inData,
                          double* normValue,
                          NormTypes normType = NORM_L2,
                          int maskWidthStride = 0,
                          const uchar* mask = nullptr);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_NORM_H_