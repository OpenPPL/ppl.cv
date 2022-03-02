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

#ifndef _ST_HPC_PPL_CV_CUDA_DISTANCETRANSFORM_H_
#define _ST_HPC_PPL_CV_CUDA_DISTANCETRANSFORM_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Calculates the distance to the closest zero pixel for each pixel of
 *        the source image.
 * @tparam T The data type of input and output image, currently only float
 *         is supported.
 * @param stream          cuda stream object.
 * @param height          input&output image's height.
 * @param width           input&output image's width.
 * @param inWidthStride   input image's width stride, it is `width` for
 *                        cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                        for 2D cudaMallocPitch() allocated data.
 * @param inData          input image data.
 * @param outWidthStride  the width stride of out, similar to inWidthStride.
 * @param outData         output image with calculated distances..
 * @param distanceType    Type of distance, see enum DistTypes. DIST_C, DIST_L1
 *                        and DIST_L2 are supported.
 * @param maskSize        Size of the distance transform mask,
 *                        see DistanceTransformMasks. DIST_MASK_3, DIST_MASK_5
 *                        and DIST_MASK_PRECISE are supported.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>float<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/distancetransform.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/distancetransform.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int channels = 1;
 *
 *   DistTypes distance_type = ppl::cv::DIST_L2;
 *   DistanceTransformMasks mask_size = ppl::cv::DIST_MASK_PRECISE;
 *
 *   uchar* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   DistanceTransform<float>(stream, height, width,
 *                            input_pitch / sizeof(uchar), gpu_input,
 *                            input_pitch / sizeof(float), gpu_output,
 *                            distance_type, mask_size);
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
template <typename T>
ppl::common::RetCode DistanceTransform(cudaStream_t stream,
                                       int height,
                                       int width,
                                       int inWidthStride,
                                       const unsigned char* inData,
                                       int outWidthStride,
                                       T* outData,
                                       DistTypes distanceType,
                                       DistanceTransformMasks maskSize);
}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_DISTANCETRANSFORM_H_
