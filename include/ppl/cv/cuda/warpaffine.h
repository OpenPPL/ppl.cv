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

#ifndef __ST_HPC_PPL3_CV_CUDA_WARPAFFINE_H_
#define __ST_HPC_PPL3_CV_CUDA_WARPAFFINE_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Affine transformation with linear interpolation.
 * @tparam T The data type of input image and output image, currently only \a
 *           uint8_t(uchar) and \a float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream          cuda stream object.
 * @param inHeight        input image's height.
 * @param inWidth         input image's width.
 * @param inWidthStride   input image's width stride, it is `width * channels`
 *                        for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                        for 2D cudaMallocPitch() allocated data.
 * @param inData          input image data.
 * @param outHeight       output image's height.
 * @param outWidth        output image's width.
 * @param outWidthStride  width stride of output image, similar to inWidthStride.
 * @param outData         output image data.
 * @param affineMatrix    2 x 3 transformation matrix.
 * @param borderType      ways to deal with border. BORDER_TYPE_CONSTANT,
 *                        BORDER_TYPE_REPLICATE and BORDER_TYPE_TRANSPARENT are
 *                        supported.
 * @param borderValue     value used in case of a constant border; by default,
 *                        it is 0.
 * @return The execution status, succeeds or fails with an error code.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @note It is aligned to the standard formula, which is more accurate than its
 *       counterpart in opencv.
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
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/warpaffine.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/warpaffine.h"
 * #include <memory>
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   const int src_width  = 320;
 *   const int src_height = 240;
 *   const int dst_width  = 640;
 *   const int dst_height = 480;
 *   const int channels   = 3;
 *
 *   float* dev_input;
 *   float* dev_output;
 *   float* affine_matrix;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&dev_input, &input_pitch,
 *                   src_width * channels * sizeof(float), src_height);
 *   cudaMallocPitch(&dev_output, &output_pitch,
 *                   dst_width * channels * sizeof(float), dst_height);
 *   std::unique_ptr<float[]> affine_matrix(new float[6]);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   WarpAffineLinear<float, 3>(stream, src_height, src_width,
 *     input_pitch / sizeof(float), dev_input, dst_height, dst_width,
 *     output_pitch / sizeof(float), dev_output, affine_matrix,
 *     BORDER_TYPE_CONSTANT, 0);
 *   cudaStreamSynchronize(stream);
 *
 *   cudaFree(dev_input);
 *   cudaFree(dev_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode
WarpAffineLinear(cudaStream_t stream,
                 int inHeight,
                 int inWidth,
                 int inWidthStride,
                 const T* inData,
                 int outHeight,
                 int outWidth,
                 int outWidthStride,
                 T* outData,
                 const float* affineMatrix,
                 BorderType borderType = BORDER_TYPE_CONSTANT,
                 T borderValue = 0);

/**
 * @brief Affine transformation with nearest neighbor interpolation.
 * @tparam T The data type of input image and output image, currently only \a
 *         uint8_t(uchar) and \a float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream          cuda stream object.
 * @param inHeight        input image's height.
 * @param inWidth         input image's width.
 * @param inWidthStride   input image's width stride, it is `width * channels`
 *                        for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                        for 2D cudaMallocPitch() allocated data.
 * @param inData          input image data.
 * @param outHeight       output image's height.
 * @param outWidth        output image's width.
 * @param outWidthStride  width stride of output image, similar to inWidthStride.
 * @param outData         output image data.
 * @param affineMatrix    2 x 3 transformation matrix.
 * @param borderType      ways to deal with border. BORDER_TYPE_CONSTANT,
 *                        BORDER_TYPE_REPLICATE and BORDER_TYPE_TRANSPARENT are
 *                        supported.
 * @param borderValue     value used in case of a constant border; by default,
 *                        it is 0.
 * @return The execution status, succeeds or fails with an error code.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @note It is aligned to the standard formula, which is more accurate than its
 *       counterpart in opencv.
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
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/warpaffine.h"
 * <tr><td>Project       <td> ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/warpaffine.h"
 * #include <memory>
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   const int src_width  = 320;
 *   const int src_height = 240;
 *   const int dst_width  = 640;
 *   const int dst_height = 480;
 *   const int channels   = 3;
 *
 *   float* dev_input;
 *   float* dev_output;
 *   float* affine_matrix;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&dev_input, &input_pitch,
 *                   src_width * channels * sizeof(float), src_height);
 *   cudaMallocPitch(&dev_output, &output_pitch,
 *                   dst_width * channels * sizeof(float), dst_height);
 *   std::unique_ptr<float[]> affine_matrix(new float[6]);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   WarpAffineNearestPoint<float, 3>(stream, src_height, src_width,
 *     input_pitch / sizeof(float), dev_input, dst_height, dst_width,
 *     output_pitch / sizeof(float), dev_output, affine_matrix,
 *     BORDER_TYPE_CONSTANT, 0);
 *   cudaStreamSynchronize(stream);
 *
 *   cudaFree(dev_input);
 *   cudaFree(dev_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode
WarpAffineNearestPoint(cudaStream_t stream,
                       int inHeight,
                       int inWidth,
                       int inWidthStride,
                       const T* inData,
                       int outHeight,
                       int outWidth,
                       int outWidthStride,
                       T* outData,
                       const float* affineMatrix,
                       BorderType borderType = BORDER_TYPE_CONSTANT,
                       T borderValue = 0);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif //! __ST_HPC_PPL3_CV_CUDA_WARPAFFINE_H_
