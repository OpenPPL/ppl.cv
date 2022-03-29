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

#ifndef _ST_HPC_PPL_CV_CUDA_ADAPTIVETHRESHOLD_H_
#define _ST_HPC_PPL_CV_CUDA_ADAPTIVETHRESHOLD_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Applies an adaptive threshold to an image.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @param maxValue       Non-zero value assigned to the pixels for which the
 *                       condition is satisfied.
 * @param adaptiveMethod adaptive thresholding algorithm to use,
 *                       ADAPTIVE_THRESH_MEAN_C and ADAPTIVE_THRESH_GAUSSIAN_C
 *                       are supported.
 * @param thresholdType  thresholding type, THRESH_BINARY and THRESH_BINARY_INV
 *                       are supported.
 * @param blockSize      size of a pixel neighborhood that is used to calculate
 *                       a threshold value for the pixel. It must be odd and
 *                       greater than 1.
 * @param delta          constant subtracted from the mean or weighted mean.
 * @param border_type    ways to deal with border. BORDER_REPLICATE,
 *                       BORDER_REFLECT, BORDER_REFLECT_101 and BORDER_DEFAULT
 *                       are supported now.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 The output image has the same size and channels as the input image.
 *       3 Only the uchar and single-channels data is supported.
 *       4 When blockSize is bigger than 32, some implementations of this
 *         function need a memory buffer to store the intermediate result, which
 *         is not less than ppl::cv::cuda::ceil2DVolume(width * sizeof(float),
 *         height). When CUDA Memory Pool is used, the capacity of CUDA Memory
 *         Pool must be not less than the size of the memory buffer.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/adaptivethreshold.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/adaptivethreshold.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 1;
 *   AdaptiveThresholdTypes adaptive_method = ppl::cv::ADAPTIVE_THRESH_MEAN_C;
 *   ThresholdTypes threshold_type = ppl::cv::THRESH_BINARY;
 *   int ksize = 3;
 *   float max_value = 25.f;
 *   float delta = 5.f;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   AdaptiveThreshold(stream, height, width, input_pitch / sizeof(uchar),
 *                     gpu_input, output_pitch / sizeof(uchar), gpu_output,
 *                     max_value, adaptive_method, threshold_type, ksize, delta,
 *                     ppl::cv::BORDER_REPLICATE);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
ppl::common::RetCode
AdaptiveThreshold(cudaStream_t stream,
                  int height,
                  int width,
                  int inWidthStride,
                  const uchar* inData,
                  int outWidthStride,
                  uchar* outData,
                  float maxValue,
                  int adaptiveMethod,
                  int thresholdType,
                  int blockSize,
                  float delta,
                  BorderType border_type = BORDER_REPLICATE);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_ADAPTIVETHRESHOLD_H_
