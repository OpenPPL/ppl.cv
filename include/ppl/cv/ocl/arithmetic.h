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

#ifndef _ST_HPC_PPL_CV_OCL_ARITHMETIC_H_
#define _ST_HPC_PPL_CV_OCL_ARITHMETIC_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace ocl {

/**
 * @brief Calculates the element-wise addition of two matrices.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/arithmetic.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * channels * sizeof(float);
 *   float* input  = (float*)malloc(data_size);
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input0 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output = = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                        NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input0, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input1, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   Add<float, 3>(queue, height, width, width * channels, gpu_input0,
 *                 width * channels, gpu_input1, width * channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input0);
 *   clReleaseMemObject(gpu_input1);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Add(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride0,
                         const cl_mem inData0,
                         int inWidthStride1,
                         const cl_mem inData1,
                         int outWidthStride,
                         cl_mem outData);

/**
 * @brief Calculates the element-wise weighted sum of two matrices.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, which is not less
 *                       than `width * channels`.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/arithmetic.h";
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/arithmetic.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * channels * sizeof(float);
 *   float* input  = (float*)malloc(data_size);
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input0 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output = = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                        NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input0, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input1, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   AddWeighted<float, 3>(queue, height, width, width * channels, gpu_input0,
 *                         0.1f, width * channels, gpu_input1, 0.2f, 0.3f,
 *                         width * channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input0);
 *   clReleaseMemObject(gpu_input1);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode AddWeighted(cl_command_queue queue,
                                 int height,
                                 int width,
                                 int inWidthStride0,
                                 const cl_mem inData0,
                                 float alpha,
                                 int inWidthStride1,
                                 const cl_mem inData1,
                                 float beta,
                                 float gamma,
                                 int outWidthStride,
                                 cl_mem outData);

/**
 * @brief Calculates the element-wise difference of two matrices.
 * @tparam T The data type, used for both source image and destination image,
 *         currently only uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/arithmetic.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * channels * sizeof(float);
 *   float* input  = (float*)malloc(data_size);
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input0 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output = = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                        NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input0, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input1, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   Subtract<float, 3>(queue, height, width, width * channels, gpu_input0,
 *                      width * channels, gpu_input1, width * channels,
 *                      gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input0);
 *   clReleaseMemObject(gpu_input1);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Subtract(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride0,
                              const cl_mem inData0,
                              int inWidthStride1,
                              const cl_mem inData1,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Calculates the element-wise scaled product of two matrices.
 * @tparam T The data type, used for both input and output image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @param scale          optional scale factor.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/arithmetic.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * channels * sizeof(float);
 *   float* input  = (float*)malloc(data_size);
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input0 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output = = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                        NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input0, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input1, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   Mul<float, 3>(queue, height, width, width * channels, gpu_input0,
 *                 width * channels, gpu_input1, width * channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input0);
 *   clReleaseMemObject(gpu_input1);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Mul(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride0,
                         const cl_mem inData0,
                         int inWidthStride1,
                         const cl_mem inData1,
                         int outWidthStride,
                         cl_mem outData,
                         float scale = 1.f);

/**
 * @brief Calculates the element-wise scaled division of two matrices.
 * @tparam T The data type, used for both input and output image, currently only
 *         uint8_t(uchar) and float are supported.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are
 *         supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride0 first input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData0        first input image data.
 * @param inWidthStride1 second input image's width stride, similar to
 *                       inWidthStride0.
 * @param inData1        second input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride0.
 * @param outData        output image data.
 * @param scale          optional scale factor.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files <td>#include "ppl/cv/ocl/arithmetic.h"
 * <tr><td>Project      <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/arithmetic.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width    = 640;
 *   int height   = 480;
 *   int channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int data_size = height * width * channels * sizeof(float);
 *   float* input  = (float*)malloc(data_size);
 *   float* output = (float*)malloc(data_size);
 *   cl_mem gpu_input0 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_output = = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size,
 *                                        NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input0, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input1, CL_FALSE, 0,
 *                                     data_size, input, 0, NULL, NULL);
 *
 *   Div<float, 3>(queue, height, width, width * channels, gpu_input0,
 *                 width * channels, gpu_input1, width * channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, data_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input0);
 *   clReleaseMemObject(gpu_input1);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T, int channels>
ppl::common::RetCode Div(cl_command_queue queue,
                         int height,
                         int width,
                         int inWidthStride0,
                         const cl_mem inData0,
                         int inWidthStride1,
                         const cl_mem inData1,
                         int outWidthStride,
                         cl_mem outData,
                         float scale = 1.f);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_ARITHMETIC_H_
