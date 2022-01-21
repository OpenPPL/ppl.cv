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

#ifndef _ST_HPC_PPL_CV_CUDA_SETVALUE_H_
#define _ST_HPC_PPL_CV_CUDA_SETVALUE_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Sets all or some of the array elements to the specified value.
 * @tparam T The data type of output array and the specified value,
 *         currently only uint8_t(uchar) and float are supported.
 * @tparam outChannels The number of channels of output image, 1, 3 and 4 are
 *         supported.
 * @tparam maskChannels The number of channels of mask, 1, 3 and 4 are
 *         supported.
 * @param stream          cuda stream object.
 * @param height          output image's height.
 * @param width           output image's width.
 * @param outWidthStride  output image's width stride, it is `width * channels`
 *                        for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                        for 2D cudaMallocPitch() allocated data.
 * @param outData         output image data.
 * @param value           assigned scalar converted to the actual array type.
 * @param maskWidthStride the width stride of mask image, similar to
 *                        outWidthStride.
 * @param mask            mask data.

 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>uint8_t(uchar)<td>4
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/setvalue.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/setvalue.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_output;
 *   float* gpu_mask;
 *   size_t output_pitch, mask_pitch;
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_mask, &mask_pitch, width * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   SetTo<float, 3, 1>(stream, height, width, output_pitch / sizeof(float),
 *                      gpu_output, 5.f, mask_pitch / sizeof(uchar), gpu_mask);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_output);
 *   cudaFree(gpu_mask);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int outChannels, int maskChannels>
ppl::common::RetCode SetTo(cudaStream_t stream,
                           int height,
                           int width,
                           int outWidthStride,
                           T* outData,
                           const T value,
                           int maskWidthStride = 0,
                           const unsigned char* mask = nullptr);

/**
 * @brief Returns an array of all 1's of the specified size and type.
 * @tparam T The data type of output image, currently only uint8_t(uchar) and
 *         float are supported.
 * @tparam channels The number of channels of output image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         output image's height.
 * @param width          output image's width.
 * @param outWidthStride output image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 In case of multi-channels type, only the first channel will be
 *         initialized with 1's, the others will be set to 0's.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>uint8_t(uchar)<td>4
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/setvalue.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/setvalue.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_output;
 *   size_t output_pitch;
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Ones<float, 3>(stream, height, width, output_pitch / sizeof(float),
 *                  gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Ones(cudaStream_t stream,
                          int height,
                          int width,
                          int outWidthStride,
                          T* outData);

/**
 * @brief Returns a zero matrix of the specified size and type.
 * @tparam T The data type of output image, currently only uint8_t(uchar) and
 *         float are supported.
 * @tparam channels The number of channels of output image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         output image's height.
 * @param width          output image's width.
 * @param outWidthStride output image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>uint8_t(uchar)<td>4
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/zeros.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/zeros.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_output;
 *   size_t output_pitch;
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Zeros<float, 3>(stream, height, width, output_pitch / sizeof(float),
 *                   gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Zeros(cudaStream_t stream,
                           int height,
                           int width,
                           int outWidthStride,
                           T* outData);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_SETVALUE_H_
