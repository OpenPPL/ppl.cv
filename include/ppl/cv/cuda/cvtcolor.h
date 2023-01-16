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

#ifndef _ST_HPC_PPL_CV_CUDA_CVTCOLOR_H_
#define _ST_HPC_PPL_CV_CUDA_CVTCOLOR_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

// BGR(RBB) <-> BGRA(RGBA)

/**
 * @brief Convert BGR images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2BGRA<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGR2BGRA(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGB images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2RGBA<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGB2RGBA(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert BGRA images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2BGR<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGRA2BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGBA images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2RGB<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGBA2RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert BGR images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2RGBA<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGR2RGBA(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGB images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2BGRA<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGB2BGRA(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGBA images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2BGR<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGBA2BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert BGRA images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2RGB<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGRA2RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

// BGR <-> RGB

/**
 * @brief Convert BGR images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2RGB<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGR2RGB(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert RGB images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2BGR<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGB2BGR(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

// BGRA <-> RGBA

/**
 * @brief Convert BGRA images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>4
 * <tr><td>float<td>4<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2RGBA<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGRA2RGBA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert RGBA images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>4
 * <tr><td>float<td>4<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2BGRA<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGBA2BGRA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

// BGR/RGB/BGRA/RGBA <-> Gray

/**
 * @brief Convert BGR images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * <tr><td>float<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2GRAY<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGR2GRAY(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGB images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * <tr><td>float<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2GRAY<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGB2GRAY(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert BGRA images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * <tr><td>float<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2GRAY<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGRA2GRAY(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert RGBA images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * <tr><td>float<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2GRAY<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGBA2GRAY(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert GRAY images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * <tr><td>float<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   GRAY2BGR<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode GRAY2BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert GRAY images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * <tr><td>float<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   GRAY2RGB<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode GRAY2RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert GRAY images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * <tr><td>float<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   GRAY2BGRA<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode GRAY2BGRA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert GRAY images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * <tr><td>float<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   GRAY2RGBA<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode GRAY2RGBA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

// BGR/RGB/BGRA/RGBA <-> YCrCb

/**
 * @brief Convert BGR images to YCrCb images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2YCrCb<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGR2YCrCb(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert RGB images to YCrCb images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2YCrCb<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGB2YCrCb(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert BGRA images to YCrCb images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2YCrCb<float>(stream, height, width, input_pitch / sizeof(float),
 *                     gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGRA2YCrCb(cudaStream_t stream,
                                int height,
                                int width,
                                int inWidthStride,
                                const T* inData,
                                int outWidthStride,
                                T* outData);

/**
 * @brief Convert RGBA images to YCrCb images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2YCrCb<float>(stream, height, width, input_pitch / sizeof(float),
 *                     gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGBA2YCrCb(cudaStream_t stream,
                                int height,
                                int width,
                                int inWidthStride,
                                const T* inData,
                                int outWidthStride,
                                T* outData);

/**
 * @brief Convert YCrCb images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   YCrCb2BGR<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode YCrCb2BGR(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert YCrCb images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   YCrCb2RGB<float>(stream, height, width, input_pitch / sizeof(float),
 *                    gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode YCrCb2RGB(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert YCrCb images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   YCrCb2BGRA<float>(stream, height, width, input_pitch / sizeof(float),
 *                     gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode YCrCb2BGRA(cudaStream_t stream,
                                int height,
                                int width,
                                int inWidthStride,
                                const T* inData,
                                int outWidthStride,
                                T* outData);

/**
 * @brief Convert YCrCb images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   YCrCb2RGBA<float>(stream, height, width, input_pitch / sizeof(float),
 *                     gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode YCrCb2RGBA(cudaStream_t stream,
                                int height,
                                int width,
                                int inWidthStride,
                                const T* inData,
                                int outWidthStride,
                                T* outData);

// BGR/RGB/BGRA/RGBA <-> HSV

/**
 * @brief Convert BGR images to HSV images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning 1 All input parameters must be valid, or undefined behaviour may occur.
 *          2 Due to use of the constant memory, this function is not thread safe.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2HSV<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGR2HSV(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert RGB images to HSV images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning 1 All input parameters must be valid, or undefined behaviour may occur.
 *          2 Due to use of the constant memory, this function is not thread safe.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2HSV<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGB2HSV(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert BGRA images to HSV images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning 1 All input parameters must be valid, or undefined behaviour may occur.
 *          2 Due to use of the constant memory, this function is not thread safe.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2HSV<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGRA2HSV(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGBA images to HSV images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning 1 All input parameters must be valid, or undefined behaviour may occur.
 *          2 Due to use of the constant memory, this function is not thread safe.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2HSV<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGBA2HSV(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert HSV images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning 1 All input parameters must be valid, or undefined behaviour may occur.
 *          2 Due to use of the constant memory, this function is not thread safe.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   HSV2BGR<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode HSV2BGR(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert HSV images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning 1 All input parameters must be valid, or undefined behaviour may occur.
 *          2 Due to use of the constant memory, this function is not thread safe.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   HSV2RGB<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode HSV2RGB(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert HSV images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning 1 All input parameters must be valid, or undefined behaviour may occur.
 *          2 Due to use of the constant memory, this function is not thread safe.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   HSV2BGRA<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode HSV2BGRA(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert HSV images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 * @warning 1 All input parameters must be valid, or undefined behaviour may occur.
 *          2 Due to use of the constant memory, this function is not thread safe.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   HSV2RGBA<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode HSV2RGBA(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

// BGR/RGB/BGRA/RGBA <-> LAB

/**
 * @brief Convert BGR images to LAB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2LAB<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGR2LAB(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert RGB images to LAB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2LAB<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGB2LAB(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert BGRA images to LAB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2LAB<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode BGRA2LAB(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGBA images to LAB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * <tr><td>float<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2LAB<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode RGBA2LAB(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert LAB images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   LAB2BGR<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode LAB2BGR(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert LAB images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>3
 * <tr><td>float<td>3<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   LAB2RGB<float>(stream, height, width, input_pitch / sizeof(float),
 *                  gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode LAB2RGB(cudaStream_t stream,
                             int height,
                             int width,
                             int inWidthStride,
                             const T* inData,
                             int outWidthStride,
                             T* outData);

/**
 * @brief Convert LAB images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   LAB2BGRA<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode LAB2BGRA(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert LAB images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>4
 * <tr><td>float<td>3<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   LAB2RGBA<float>(stream, height, width, input_pitch / sizeof(float),
 *                   gpu_input, output_pitch / sizeof(float), gpu_output);
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
ppl::common::RetCode LAB2RGBA(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

// BGR/RGB/BGRA/RGBA <-> NV12

/**
 * @brief Convert BGR images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode BGR2NV12(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert BGR images to NV12 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch, output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2NV12(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outYStride,
                              T* outY,
                              int outUVStride,
                              T* outUV);

/**
 * @brief Convert RGB images to NV12 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode RGB2NV12(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGB images to NV12 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch, output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2NV12(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outYStride,
                              T* outY,
                              int outUVStride,
                              T* outUV);

/**
 * @brief Convert BGRA images to NV12 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode BGRA2NV12(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert BGRA images to NV12 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch, output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2NV12(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outYStride,
                               T* outY,
                               int outUVStride,
                               T* outUV);

/**
 * @brief Convert RGBA images to NV12 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode RGBA2NV12(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert RGBA images to NV12 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch, output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2NV12(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outYStride,
                               T* outY,
                               int outUVStride,
                               T* outUV);

/**
 * @brief Convert NV12 images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode NV122BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert NV12 images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122BGR<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inYStride,
                              const T* inY,
                              int inUVStride,
                              const T* inUV,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert NV12 images to RGB images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122RGB<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode NV122RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert NV12 images to RGB images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122RGB<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inYStride,
                              const T* inY,
                              int inUVStride,
                              const T* inUV,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert NV12 images to BGRA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122BGRA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode NV122BGRA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert NV12 images to BGRA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122BGRA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122BGRA(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUVStride,
                               const T* inUV,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert NV12 images to RGBA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122RGBA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode NV122RGBA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert NV12 images to RGBA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122RGBA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122RGBA(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUVStride,
                               const T* inUV,
                               int outWidthStride,
                               T* outData);

// BGR/RGB/BGRA/RGBA <-> NV21

/**
 * @brief Convert BGR images to NV21 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode BGR2NV21(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert BGR images to NV21 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch, output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2NV21(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outYStride,
                              T* outY,
                              int outUVStride,
                              T* outUV);

/**
 * @brief Convert RGB images to NV21 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode RGB2NV21(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGB images to NV21 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch, output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2NV21(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outYStride,
                              T* outY,
                              int outUVStride,
                              T* outUV);

/**
 * @brief Convert BGRA images to NV21 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode BGRA2NV21(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert BGRA images to NV21 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch, output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2NV21(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outYStride,
                               T* outY,
                               int outUVStride,
                               T* outUV);

/**
 * @brief Convert RGBA images to NV21 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode RGBA2NV21(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert RGBA images to NV21 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch, output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2NV21(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outYStride,
                               T* outY,
                               int outUVStride,
                               T* outUV);

/**
 * @brief Convert NV21 images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode NV212BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert NV21 images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212BGR<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inYStride,
                              const T* inY,
                              int inUVStride,
                              const T* inUV,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert NV21 images to RGB images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212RGB<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode NV212RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert NV21 images to RGB images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212RGB<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inYStride,
                              const T* inY,
                              int inUVStride,
                              const T* inUV,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert NV21 images to BGRA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212BGRA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode NV212BGRA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert NV21 images to BGRA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212BGRA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212BGRA(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUVStride,
                               const T* inUV,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert NV21 images to RGBA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212RGBA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode NV212RGBA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert NV21 images to RGBA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212RGBA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212RGBA(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUVStride,
                               const T* inUV,
                               int outWidthStride,
                               T* outData);

// BGR/RGB/BGRA/RGBA <-> I420

/**
 * @brief Convert RGB images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode RGB2I420(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert RGB images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUStride     U-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outU           U-channel output image data.
 * @param outVStride     V-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outV           V-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   uchar* gpu_output2;
 *   size_t input_pitch, output_pitch0, output_pitch1, output_pitch2;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output2, &output_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGB2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1,
 *                   output_pitch2 / sizeof(uchar), gpu_output2);
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
ppl::common::RetCode RGB2I420(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outYStride,
                              T* outY,
                              int outUStride,
                              T* outU,
                              int outVStride,
                              T* outV);

/**
 * @brief Convert BGR images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode BGR2I420(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert BGR images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUStride     U-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outU           U-channel output image data.
 * @param outVStride     V-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outV           V-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   uchar* gpu_output2;
 *   size_t input_pitch, output_pitch0, output_pitch1, output_pitch2;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output2, &output_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGR2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1,
 *                   output_pitch2 / sizeof(uchar), gpu_output2);
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
ppl::common::RetCode BGR2I420(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outYStride,
                              T* outY,
                              int outUStride,
                              T* outU,
                              int outVStride,
                              T* outV);

/**
 * @brief Convert BGRA images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode BGRA2I420(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert BGRA images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUStride     U-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outU           U-channel output image data.
 * @param outVStride     V-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outV           V-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   uchar* gpu_output2;
 *   size_t input_pitch, output_pitch0, output_pitch1, output_pitch2;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output2, &output_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   BGRA2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1,
 *                    output_pitch2 / sizeof(uchar), gpu_output2);
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
ppl::common::RetCode BGRA2I420(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outYStride,
                               T* outY,
                               int outUStride,
                               T* outU,
                               int outVStride,
                               T* outV);

/**
 * @brief Convert RGBA images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode RGBA2I420(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert RGBA images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUStride     U-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outU           U-channel output image data.
 * @param outVStride     V-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outV           V-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   uchar* gpu_output2;
 *   size_t input_pitch, output_pitch0, output_pitch1, output_pitch2;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output2, &output_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   RGBA2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1,
 *                    output_pitch2 / sizeof(uchar), gpu_output2);
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
ppl::common::RetCode RGBA2I420(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outYStride,
                               T* outY,
                               int outUStride,
                               T* outU,
                               int outVStride,
                               T* outV);

/**
 * @brief Convert I420 images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode I4202BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert I420 images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, it is `width`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, it is `width / 2` for cudaMalloc() allocated data, usually
 *                       no less than `width / 2` for 2D cudaMallocPitch() allocated data.
 * @param inU            U-channel input image data.
 * @param inVStride      U-channel width stride of input image, similar to inUStride.
 * @param inV            V-channel input image data.
 * @param outWidthStride output image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_input2;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, input_pitch2, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_input2, &input_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202BGR<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   input_pitch2 / sizeof(uchar), gpu_input2,
 *                   output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_input2);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inYStride,
                              const T* inY,
                              int inUStride,
                              const T* inU,
                              int inVStride,
                              const T* inV,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert I420 images to RGB images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202RGB<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode I4202RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert I420 images to RGB images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, it is `width`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, it is `width / 2` for cudaMalloc() allocated data, usually
 *                       no less than `width / 2` for 2D cudaMallocPitch() allocated data.
 * @param inU            U-channel input image data.
 * @param inVStride      U-channel width stride of input image, similar to inUStride.
 * @param inV            V-channel input image data.
 * @param outWidthStride output image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_input2;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, input_pitch2, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_input2, &input_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202RGB<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   input_pitch2 / sizeof(uchar), gpu_input2,
 *                   output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_input2);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202RGB(cudaStream_t stream,
                              int height,
                              int width,
                              int inYStride,
                              const T* inY,
                              int inUStride,
                              const T* inU,
                              int inVStride,
                              const T* inV,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert I420 images to BGRA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202BGRA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode I4202BGRA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert I420 images to BGRA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, it is `width`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, it is `width / 2` for cudaMalloc() allocated data, usually
 *                       no less than `width / 2` for 2D cudaMallocPitch() allocated data.
 * @param inU            U-channel input image data.
 * @param inVStride      U-channel width stride of input image, similar to inUStride.
 * @param inV            V-channel input image data.
 * @param outWidthStride output image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_input2;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, input_pitch2, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_input2, &input_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202BGRA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    input_pitch2 / sizeof(uchar), gpu_input2,
 *                    output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_input2);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202BGRA(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUStride,
                               const T* inU,
                               int inVStride,
                               const T* inV,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert I420 images to RGBA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *   int src_height = height;
 *   int dst_height = height;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202RGBA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode I4202RGBA(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

/**
 * @brief Convert I420 images to RGBA images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, it is `width`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, it is `width / 2` for cudaMalloc() allocated data, usually
 *                       no less than `width / 2` for 2D cudaMallocPitch() allocated data.
 * @param inU            U-channel input image data.
 * @param inVStride      U-channel width stride of input image, similar to inUStride.
 * @param inV            V-channel input image data.
 * @param outWidthStride output image's width stride, it is `width * channels`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_input2;
 *   uchar* gpu_output;
 *   size_t input_pitch0, input_pitch1, input_pitch2, output_pitch;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_input2, &input_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202RGBA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    input_pitch2 / sizeof(uchar), gpu_input2,
 *                    output_pitch / sizeof(uchar), gpu_output);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_input2);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202RGBA(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUStride,
                               const T* inU,
                               int inVStride,
                               const T* inV,
                               int outWidthStride,
                               T* outData);

// YUV -> GRAY

/**
 * @brief Convert YUV images to GRAY images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height * 3 / 2);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   YUV2GRAY<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode YUV2GRAY(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

// BGR/GRAY <-> UYVY

/**
 * @brief Convert UYVY images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>2<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 2;
 *   int dst_channels = 3;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * 2 * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * 2 * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   UYVY2BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode UYVY2BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert UYVY images to GRAY images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>2<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 2;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * 2 * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * 2 * dst_channels * sizeof(uchar), height);
 *
 *   UYVY2GRAY<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode UYVY2GRAY(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

// BGR/GRAY <-> YUYV

/**
 * @brief Convert YUYV images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>2<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 2;
 *   int dst_channels = 3;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * 2 * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * 2 * sizeof(uchar), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   YUYV2BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode YUYV2BGR(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Convert YUYV images to GRAY images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>2<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 2;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * 2 * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * 2 * dst_channels * sizeof(uchar), height);
 *
 *   YUYV2GRAY<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
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
ppl::common::RetCode YUYV2GRAY(cudaStream_t stream,
                               int height,
                               int width,
                               int inWidthStride,
                               const T* inData,
                               int outWidthStride,
                               T* outData);

//NV12/21 <-> I420

/**
 * @brief Convert NV21 images to I420 images,format: YYYYUVUVUVUV -> YYYYUUUUVVVV.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUStride     U-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outU           U-channel output image data.
 * @param outVStride     V-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outV           V-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 1;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   uchar* gpu_output2;
 *   size_t input_pitch0, input_pitch1;
 *   size_t output_pitch0, output_pitch1, output_pitch2;
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output2, &output_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV122I420<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1,
 *                    output_pitch2 / sizeof(uchar), gpu_output2);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *   cudaFree(gpu_output2);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122I420(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUVStride,
                               const T* inUV,
                               int outYStride,
                               T* outY,
                               int outUStride,
                               T* outU,
                               int outVStride,
                               T* outV);

/**
 * @brief Convert NV21 images to I420 images,format: YYYYVUVUVUVU -> YYYYUUUUVVVV.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's width stride, it is `width` for
 *                       cudaMalloc() allocated data, `pitch / sizeof(T)` for
 *                       2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data..
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUStride     U-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outU           U-channel output image data.
 * @param outVStride     V-channel width stride of output image, similar to
 *                       inWidthStride / 2.
 * @param outV           V-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   uchar* gpu_output2;
 *   size_t input_pitch0, input_pitch1;
 *   size_t output_pitch0, output_pitch1, output_pitch2;
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output2, &output_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   NV212I420<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1,
 *                    output_pitch2 / sizeof(uchar), gpu_output2);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_output0);
 *   cudaFree(gpu_output1);
 *   cudaFree(gpu_output2);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212I420(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUVStride,
                               const T* inUV,
                               int outYStride,
                               T* outY,
                               int outUStride,
                               T* outU,
                               int outVStride,
                               T* outV);

/**
 * @brief Convert I420 images to NV12 images,format: YYYYUUUUVVVV -> YYYYUVUVUVUV
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, it is `width`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, it is `width / 2` for cudaMalloc() allocated data, usually
 *                       no less than `width / 2` for 2D cudaMallocPitch() allocated data.
 * @param inU            U-channel input image data.
 * @param inVStride      U-channel width stride of input image, similar to inUStride.
 * @param inV            V-channel input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_input2;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch0, input_pitch1, input_pitch2;
 *   size_t output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_input2, &input_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202NV12<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    input_pitch2 / sizeof(uchar), gpu_input2,
 *                    output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_input2);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202NV12(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUStride,
                               const T* inU,
                               int inVStride,
                               const T* inV,
                               int outYStride,
                               T* outY,
                               int outUVStride,
                               T* outUV);

/**
 * @brief Convert I420 images to NV21 images,format: YYYYUUUUVVVV -> YYYYVUVUVUVU
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, it is `width`
 *                       for cudaMalloc() allocated data, `pitch / sizeof(T)`
 *                       for 2D cudaMallocPitch() allocated data.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, it is `width / 2` for cudaMalloc() allocated data, usually
 *                       no less than `width / 2` for 2D cudaMallocPitch() allocated data.
 * @param inU            U-channel input image data.
 * @param inVStride      U-channel width stride of input image, similar to inUStride.
 * @param inV            V-channel input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported<td>CUDA7.0
 * <tr><td>Header files  <td> #include "ppl/cv/cuda/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/cvtcolor.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *
 *   uchar* gpu_input0;
 *   uchar* gpu_input1;
 *   uchar* gpu_input2;
 *   uchar* gpu_output0;
 *   uchar* gpu_output1;
 *   size_t input_pitch0, input_pitch1, input_pitch2;
 *   size_t output_pitch0, output_pitch1;
 *   cudaMallocPitch(&gpu_input0, &input_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_input1, &input_pitch1,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_input2, &input_pitch2,
 *                   (width >> 1) * sizeof(uchar), (height >> 1));
 *   cudaMallocPitch(&gpu_output0, &output_pitch0,
 *                   width * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output1, &output_pitch1,
 *                   width * sizeof(uchar), (height >> 1));
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   I4202NV21<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    input_pitch2 / sizeof(uchar), gpu_input2,
 *                    output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
 *   cudaStreamSynchronize(stream);
 *   cudaStreamDestroy(stream);
 *
 *   cudaFree(gpu_input0);
 *   cudaFree(gpu_input1);
 *   cudaFree(gpu_input2);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202NV21(cudaStream_t stream,
                               int height,
                               int width,
                               int inYStride,
                               const T* inY,
                               int inUStride,
                               const T* inU,
                               int inVStride,
                               const T* inV,
                               int outYStride,
                               T* outY,
                               int outUVStride,
                               T* outUV);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_CVTCOLOR_H_
