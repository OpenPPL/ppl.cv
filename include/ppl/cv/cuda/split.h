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

#ifndef _ST_HPC_PPL_CV_CUDA_SPLIT_H_
#define _ST_HPC_PPL_CV_CUDA_SPLIT_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Divides a multi-channel array into 3 single-channel arrays.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData0       the first single-channel output image data.
 * @param outData1       the second single-channel output image data.
 * @param outData2       the third single-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>float<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/split.h";
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/split.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output0;
 *   float* gpu_output1;
 *   float* gpu_output2;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output2, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Split3Channels<float>(stream, height, width, input_pitch / sizeof(float),
 *                         gpu_input, output_pitch / sizeof(float), gpu_output0,
 *                         gpu_output1, gpu_output2);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *   cudaFree(gpu_output2);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode Split3Channels(cudaStream_t stream,
                                    int height,
                                    int width,
                                    int inWidthStride,
                                    const T* inData,
                                    int outWidthStride,
                                    T* outData0,
                                    T* outData1,
                                    T* outData2);

/**
 * @brief Divides a multi-channel array into 4 single-channel arrays.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData0       the first single-channel output image data.
 * @param outData1       the second single-channel output image data.
 * @param outData2       the third single-channel output image data.
 * @param outData3       the fourth single-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>4
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/split.h";
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/split.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output0;
 *   float* gpu_output1;
 *   float* gpu_output2;
 *   float* gpu_output3;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output2, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output3, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Split4Channels<float>(stream, height, width, input_pitch / sizeof(float),
 *                         gpu_input, output_pitch / sizeof(float), gpu_output0,
 *                         gpu_output1, gpu_output2, gpu_output3);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *   cudaFree(gpu_output2);
 *   cudaFree(gpu_output3);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode Split4Channels(cudaStream_t stream,
                                    int height,
                                    int width,
                                    int inWidthStride,
                                    const T* inData,
                                    int outWidthStride,
                                    T* outData0,
                                    T* outData1,
                                    T* outData2,
                                    T* outData3);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_SPLIT_H_
