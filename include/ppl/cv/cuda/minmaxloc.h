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

#ifndef _ST_HPC_PPL3_CV_CUDA_MINMAXLOC_H_
#define _ST_HPC_PPL3_CV_CUDA_MINMAXLOC_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Finds the global minimum and maximum element values and their
 *        positions in a 2D array.
 * @tparam T The data type, used for both source matrix, currently only uchar
 *         and float are supported.
 * @param stream           cuda stream object.
 * @param inHeight         input image's height.
 * @param inWidth          input image's width.
 * @param inWidthStride    input image's width stride, it is `width * channels`
 *                         for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                         for 2D cudaMallocPitch() allocated data.
 * @param inData           input image data.
 * @param minVal           pointer to the returned minimum value.
 * @param maxVal           pointer to the returned maximum value.
 * @param minIdxX          pointer to the returned minimum column index.
 * @param minIdxY          pointer to the returned minimum row index.
 * @param maxIdxX          pointer to the returned maximum column index.
 * @param maxIdxY          pointer to the returned maximum row index.
 * @param maskWidthStride  the width stride of mask image, similar to
 *                         inWidthStride.
 * @param mask             specified array region.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 inData and mask have the same height, width and channels.
 *       3 The function only works with single channel arrays.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>float<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/minmaxloc.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/minmaxloc.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int channels = 1;
 *
 *   float* dev_input;
 *   float* dev_mask;
 *   size_t input_pitch, mask_pitch;
 *   cudaMallocPitch(&dev_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&dev_mask, &mask_pitch,
 *                   width * channels * sizeof(uchar), height);
 *
 *   float min_value, max_value;
 *   int min_index_x, min_index_y, max_index_x, max_index_y;
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   MinMaxLoc<float>(stream, height, width, input_pitch / sizeof(float),
 *                    dev_input, &min_value, &max_value, &min_index_x,
 *                    &min_index_y, &max_index_x, &max_index_y,
 *                    mask_pitch / sizeof(float), dev_mask);
 *   cudaStreamSynchronize(stream);
 *
 *   cudaFree(dev_input);
 *   cudaFree(dev_mask);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode MinMaxLoc(cudaStream_t stream,
                               int inHeight,
                               int inWidth,
                               int inWidthStride,
                               const T* inData,
                               T* minVal,
                               T* maxVal,
                               int* minIdxX,
                               int* minIdxY,
                               int* maxIdxX,
                               int* maxIdxY,
                               int maskWidthStride = 0,
                               const unsigned char* mask = NULL);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL3_CV_CUDA_MINMAXLOC_H_
