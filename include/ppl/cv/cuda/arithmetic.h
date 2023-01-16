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

#ifndef _ST_HPC_PPL_CV_CUDA_ARITHMETIC_H_
#define _ST_HPC_PPL_CV_CUDA_ARITHMETIC_H_

#include "cuda_runtime.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace cuda {

/**
 * @brief Calculates the element-wise addition of two matrices.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, it is `width *
 *                       channels` for cudaMalloc() allocated data, `pitch /
 *                       sizeof(T)` for 2D cudaMallocPitch() allocated data.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
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
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/arithmetic.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Add<float, 3>(stream, height, width,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 output_pitch / sizeof(float), gpu_output);
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
template <typename T, int channels>
ppl::common::RetCode Add(cudaStream_t stream,
                         int height,
                         int width,
                         int inWidthStride0,
                         const T* inData0,
                         int inWidthStride1,
                         const T* inData1,
                         int outWidthStride,
                         T* outData);

/**
 * @brief Calculates the element-wise weighted sum of two matrices.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, it is `width *
 *                       channels` for cudaMalloc() allocated data, `pitch /
 *                       sizeof(T)` for 2D cudaMallocPitch() allocated data.
 * @param inData0        first input image data.
 *  @param alpha         weight of the first image elements.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param beta           weight of the second image elements.
 * @param gamma          scalar added to each sum.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
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
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/arithmetic.h";
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/arithmetic.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   AddWeighted<float, 3>(stream, height, width,
 *                         input_pitch / sizeof(float), gpu_input, 0.1,
 *                         input_pitch / sizeof(float), gpu_input, 0.2, 0.3,
 *                         output_pitch / sizeof(float), gpu_output);
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
template <typename T, int channels>
ppl::common::RetCode AddWeighted(cudaStream_t stream,
                                 int height,
                                 int width,
                                 int inWidthStride0,
                                 const T* inData0,
                                 float alpha,
                                 int inWidthStride1,
                                 const T* inData1,
                                 float beta,
                                 float gamma,
                                 int outWidthStride,
                                 T* outData);

/**
 * @brief Calculates the element-wise difference between an array and a scalar.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, it is `width *
 *                       channels` for cudaMalloc() allocated data, `pitch /
 *                       sizeof(T)` for 2D cudaMallocPitch() allocated data.
 * @param inData         input image data.
 * @param scalar         input scalar data, equals to `channels * sizeof(T)` on
 *                       the host memory.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
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
 * <tr><td>Header files <td>#include "ppl/cv/cuda/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/arithmetic.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *   float scalar[3] = {1.f, 2.f, 3.f};
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Subtract<float, 3>(stream, height, width, input_pitch / sizeof(float),
 *                      gpu_input, scalar, output_pitch / sizeof(float),
 *                      gpu_output);
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
template <typename T, int channels>
ppl::common::RetCode Subtract(cudaStream_t stream,
                              int height,
                              int width,
                              int inWidthStride,
                              const T* inData,
                              const T* scalar,
                              int outWidthStride,
                              T* outData);

/**
 * @brief Calculates the element-wise scaled product of two matrices.
 * @tparam T The data type, used for both input and output image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, it is `width *
 *                       channels` for cudaMalloc() allocated data, `pitch /
 *                       sizeof(T)` for 2D cudaMallocPitch() allocated data.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @param scale          optional scale factor.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
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
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/arithmetic.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Mul<float, 3>(stream, height, width,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 output_pitch / sizeof(float), gpu_output);
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
template <typename T, int channels>
ppl::common::RetCode Mul(cudaStream_t stream,
                         int height,
                         int width,
                         int inWidthStride0,
                         const T* inData0,
                         int inWidthStride1,
                         const T* inData1,
                         int outWidthStride,
                         T* outData,
                         float scale = 1.f);

/**
 * @brief Calculates the element-wise division of two matrices.
 * @tparam T The data type, used for both input and output image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, it is `width *
 *                       channels` for cudaMalloc() allocated data, `pitch /
 *                       sizeof(T)` for 2D cudaMallocPitch() allocated data.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @param scale          optional scale factor.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
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
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/arithmetic.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Div<float, 3>(stream, height, width,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 output_pitch / sizeof(float), gpu_output);
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
template <typename T, int channels>
ppl::common::RetCode Div(cudaStream_t stream,
                         int height,
                         int width,
                         int inWidthStride0,
                         const T* inData0,
                         int inWidthStride1,
                         const T* inData1,
                         int outWidthStride,
                         T* outData,
                         float scale = 1.f);

/**
 * @brief Calculates the element-wise multiplication-addition of two matrices.
 * @tparam T The data type, used for both input and output image, currently only
 *         float is supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, it is `width *
 *                       channels` for cudaMalloc() allocated data, `pitch /
 *                       sizeof(T)` for 2D cudaMallocPitch() allocated data.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/arithmetic.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Mla<float, 3>(stream, height, width,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 output_pitch / sizeof(float), gpu_output);
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
template <typename T, int channels>
ppl::common::RetCode Mla(cudaStream_t stream,
                         int height,
                         int width,
                         int inWidthStride0,
                         const T* inData0,
                         int inWidthStride1,
                         const T* inData1,
                         int outWidthStride,
                         T* outData);

/**
 * @brief Calculates the element-wise multiplication-subtraction of two
 *        matrices.
 * @tparam T The data type, used for both input and output image, currently only
 *         float is supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param stream         cuda stream object.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, it is `width *
 *                       channels` for cudaMalloc() allocated data, `pitch /
 *                       sizeof(T)` for 2D cudaMallocPitch() allocated data.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, a 2D array allocated by cudaMallocPitch() is
 *       recommended.
 * @warning All parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>CUDA platforms supported <td>CUDA 7.0
 * <tr><td>Header files <td>#include "ppl/cv/cuda/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/cuda/arithmetic.h"
 * using namespace ppl::cv::cuda;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   float* gpu_input;
 *   float* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * channels * sizeof(float), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * channels * sizeof(float), height);
 *
 *   cudaStream_t stream;
 *   cudaStreamCreate(&stream);
 *   Mls<float, 3>(stream, height, width,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 input_pitch / sizeof(float), gpu_input,
 *                 output_pitch / sizeof(float), gpu_output);
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
template <typename T, int channels>
ppl::common::RetCode Mls(cudaStream_t stream,
                         int height,
                         int width,
                         int inWidthStride0,
                         const T* inData0,
                         int inWidthStride1,
                         const T* inData1,
                         int outWidthStride,
                         T* outData);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_ARITHMETIC_H_
