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

#ifndef _ST_HPC_PPL_CV_OCL_CVTCOLOR_H_
#define _ST_HPC_PPL_CV_OCL_CVTCOLOR_H_

#include "CL/cl.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace ocl {

// BGR(RBB) <-> BGRA(RGBA)

/**
 * @brief Convert BGR images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2BGRA<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2BGRA(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGB images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2RGBA<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2RGBA(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert BGRA images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2BGR<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGBA images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2RGB<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert BGR images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2RGBA<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2RGBA(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGB images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2BGRA<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2BGRA(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGBA images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2BGR<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert BGRA images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2RGB<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

// BGR <-> RGB

/**
 * @brief Convert BGR images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2RGB<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2RGB(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert RGB images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2BGR<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2BGR(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

// BGRA <-> RGBA

/**
 * @brief Convert BGRA images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2RGBA<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2RGBA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert RGBA images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2BGRA<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2BGRA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

// BGR/RGB/BGRA/RGBA <-> Gray

/**
 * @brief Convert BGR images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2GRAY<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2GRAY(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGB images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 1;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2GRAY<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2GRAY(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert BGRA images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2GRAY<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2GRAY(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert RGBA images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 1;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2GRAY<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2GRAY(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert GRAY images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   GRAY2BGR<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode GRAY2BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert GRAY images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   GRAY2RGB<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode GRAY2RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert GRAY images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   GRAY2BGRA<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode GRAY2BGRA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert GRAY images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   GRAY2RGBA<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode GRAY2RGBA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

// BGR/RGB/BGRA/RGBA <-> YCrCb

/**
 * @brief Convert BGR images to YCrCb images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2YCrCb<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2YCrCb(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert RGB images to YCrCb images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2YCrCb<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2YCrCb(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert BGRA images to YCrCb images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2YCrCb<float>(queue, height, width, width * src_channels, gpu_input,
 *                     width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2YCrCb(cl_command_queue queue,
                                int height,
                                int width,
                                int inWidthStride,
                                const cl_mem inData,
                                int outWidthStride,
                                cl_mem outData);

/**
 * @brief Convert RGBA images to YCrCb images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2YCrCb<float>(queue, height, width, width * src_channels, gpu_input,
 *                     width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2YCrCb(cl_command_queue queue,
                                int height,
                                int width,
                                int inWidthStride,
                                const cl_mem inData,
                                int outWidthStride,
                                cl_mem outData);

/**
 * @brief Convert YCrCb images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   YCrCb2BGR<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode YCrCb2BGR(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert YCrCb images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   YCrCb2RGB<float>(queue, height, width, width * src_channels, gpu_input,
 *                    width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode YCrCb2RGB(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert YCrCb images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   YCrCb2BGRA<float>(queue, height, width, width * src_channels, gpu_input,
 *                     width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode YCrCb2BGRA(cl_command_queue queue,
                                int height,
                                int width,
                                int inWidthStride,
                                const cl_mem inData,
                                int outWidthStride,
                                cl_mem outData);

/**
 * @brief Convert YCrCb images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   YCrCb2RGBA<float>(queue, height, width, width * src_channels, gpu_input,
 *                     width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode YCrCb2RGBA(cl_command_queue queue,
                                int height,
                                int width,
                                int inWidthStride,
                                const cl_mem inData,
                                int outWidthStride,
                                cl_mem outData);

// BGR/RGB/BGRA/RGBA <-> HSV

/**
 * @brief Convert BGR images to HSV images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2HSV<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2HSV(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert RGB images to HSV images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2HSV<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2HSV(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert BGRA images to HSV images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2HSV<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2HSV(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGBA images to HSV images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2HSV<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2HSV(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert HSV images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   HSV2BGR<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode HSV2BGR(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert HSV images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   HSV2RGB<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode HSV2RGB(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert HSV images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   HSV2BGRA<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode HSV2BGRA(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert HSV images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   HSV2RGBA<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode HSV2RGBA(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

// BGR/RGB/BGRA/RGBA <-> LAB

/**
 * @brief Convert BGR images to LAB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2LAB<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2LAB(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert RGB images to LAB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2LAB<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2LAB(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert BGRA images to LAB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2LAB<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2LAB(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGBA images to LAB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 4;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2LAB<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2LAB(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert LAB images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   LAB2BGR<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode LAB2BGR(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert LAB images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 3;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   LAB2RGB<float>(queue, height, width, width * src_channels, gpu_input,
 *                  width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode LAB2RGB(cl_command_queue queue,
                             int height,
                             int width,
                             int inWidthStride,
                             const cl_mem inData,
                             int outWidthStride,
                             cl_mem outData);

/**
 * @brief Convert LAB images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   LAB2BGRA<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode LAB2BGRA(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert LAB images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) and float are supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 3;
 *   int dst_channels = 4;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(float);
 *   int dst_size = height * width * dst_channels * sizeof(float);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   LAB2RGBA<float>(queue, height, width, width * src_channels, gpu_input,
 *                   width * dst_channels, gpu_output);
 *   error_code = clEnqueueReadBuffer(queue, gpu_output, CL_TRUE, 0, dst_size,
 *                                    output, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode LAB2RGBA(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

// BGR/RGB/BGRA/RGBA <-> NV12

/**
 * @brief Convert BGR images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *           currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2NV12<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2NV12(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert BGR images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int y_size = height * width * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2NV12<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                   width, gpu_y, width, gpu_uv);
 *
 *   free(input);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2NV12(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outYStride,
                              cl_mem outY,
                              int outUVStride,
                              cl_mem outUV);

/**
 * @brief Convert RGB images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2NV12<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2NV12(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGB images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int y_size = height * width * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2NV12<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                   width, gpu_y, width, gpu_uv);
 *
 *   free(input);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2NV12(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outYStride,
                              cl_mem outY,
                              int outUVStride,
                              cl_mem outUV);

/**
 * @brief Convert BGRA images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2NV12<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2NV12(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert BGRA images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int y_size = height * width * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2NV12<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                    width, gpu_y, width, gpu_uv);
 *
 *   free(input);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2NV12(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outYStride,
                               cl_mem outY,
                               int outUVStride,
                               cl_mem outUV);

/**
 * @brief Convert RGBA images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2NV12<uchar>(squeue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2NV12(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert RGBA images to NV12 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int y_size = height * width * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2NV12<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                    width, gpu_y, width, gpu_uv);
 *
 *   free(input);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2NV12(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outYStride,
                               cl_mem outY,
                               int outUVStride,
                               cl_mem outUV);

/**
 * @brief Convert NV12 images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   NV122BGR<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert NV12 images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's Y-channel width stride, which is not less
 *                       than `width * channels`.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   NV122BGR<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                   width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inYStride,
                              const cl_mem inY,
                              int inUVStride,
                              const cl_mem inUV,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert NV12 images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   NV122RGB<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert NV12 images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's Y-channel width stride, which is not less
 *                       than `width * channels`.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   NV122RGB<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                   width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inYStride,
                              const cl_mem inY,
                              int inUVStride,
                              const cl_mem inUV,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert NV12 images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   NV122BGRA<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122BGRA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert NV12 images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's Y-channel width stride, which is not less
 *                       than `width * channels`.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   NV122BGRA<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                    width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122BGRA(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUVStride,
                               const cl_mem inUV,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert NV12 images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   NV122RGBA<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122RGBA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert NV12 images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's Y-channel width stride, which is not less
 *                       than `width * channels`.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   NV122RGBA<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                    width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV122RGBA(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUVStride,
                               const cl_mem inUV,
                               int outWidthStride,
                               cl_mem outData);

// BGR/RGB/BGRA/RGBA <-> NV21

/**
 * @brief Convert BGR images to NV21 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2NV21<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2NV21(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert BGR images to NV21 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int y_size = height * width * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2NV21<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                   width, gpu_y, width, gpu_uv);
 *
 *   free(input);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2NV21(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outYStride,
                              cl_mem outY,
                              int outUVStride,
                              cl_mem outUV);

/**
 * @brief Convert RGB images to NV21 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2NV21<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2NV21(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGB images to NV21 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int y_size = height * width * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2NV21<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                   width, gpu_y, width, gpu_uv);
 *
 *   free(input);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2NV21(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outYStride,
                              cl_mem outY,
                              int outUVStride,
                              cl_mem outUV);

/**
 * @brief Convert BGRA images to NV21 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2NV21<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2NV21(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert BGRA images to NV21 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   BGRA2NV21<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                    width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2NV21(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outYStride,
                               cl_mem outY,
                               int outUVStride,
                               cl_mem outUV);

/**
 * @brief Convert RGBA images to NV21 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2NV21<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2NV21(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert RGBA images to NV21 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outYStride     Y-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outY           Y-channel output image data.
 * @param outUVStride    UV-channel width stride of output image, similar to
 *                       inWidthStride.
 * @param outUV          UV-channel output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   RGBA2NV21<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                    width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2NV21(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outYStride,
                               cl_mem outY,
                               int outUVStride,
                               cl_mem outUV);

/**
 * @brief Convert NV21 images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   NV212BGR<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert NV21 images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's Y-channel width stride, which is not less
 *                       than `width * channels`.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   NV212BGR<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                   width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inYStride,
                              const cl_mem inY,
                              int inUVStride,
                              const cl_mem inUV,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert NV21 images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   NV212RGB<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert NV21 images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's Y-channel width stride, which is not less
 *                       than `width * channels`.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   NV212RGB<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                   width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inYStride,
                              const cl_mem inY,
                              int inUVStride,
                              const cl_mem inUV,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert NV21 images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   NV212BGRA<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212BGRA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert NV21 images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's Y-channel width stride, which is not less
 *                       than `width * channels`.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   NV212BGRA<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                    width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212BGRA(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUVStride,
                               const cl_mem inUV,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert NV21 images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   NV212RGBA<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212RGBA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert NV21 images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      input image's Y-channel width stride, which is not less
 *                       than `width * channels`.
 * @param inY            Y-channel input image data.
 * @param inUVStride     input image's UV-channel width stride, similar to
 *                       inYStride.
 * @param inUV           UV-channel input image data.
 * @param outWidthStride width stride of output image, similar to inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int y_size = height * width * src_channels * sizeof(uchar);
 *   int uv_size = height * width / 2 * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(y_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_READ_ONLY, y_size,
 *                                 NULL, &error_code);
 *   cl_mem gpu_uv = clCreateBuffer(context, CL_MEM_READ_ONLY, uv_size,
 *                                  NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     y_size, input, 0, NULL, NULL);
 *
 *   NV212RGBA<uchar>(queue, height, width, width, gpu_y, width, gpu_uv,
 *                    width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_uv);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode NV212RGBA(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUVStride,
                               const cl_mem inUV,
                               int outWidthStride,
                               cl_mem outData);

// BGR/RGB/BGRA/RGBA <-> I420

/**
 * @brief Convert BGR images to I420 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2I420<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2I420(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert BGR images to I420 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = height * width * sizeof(uchar);
 *   int uv_size = height * width / 4 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_u = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGR2I420<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                   width, gpu_y, width / 2, gpu_u, width / 2, gpu_v);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_u);
 *   clReleaseMemObject(gpu_v);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2I420(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outYStride,
                              cl_mem outY,
                              int outUStride,
                              cl_mem outU,
                              int outVStride,
                              cl_mem outV);

/**
 * @brief Convert RGB images to I420 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2I420<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2I420(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert RGB images to I420 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * sizeof(uchar);
 *   int uv_size = height * width / 4 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_u = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGB2I420<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                   width, gpu_y, width / 2, gpu_u, width / 2, gpu_v);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_u);
 *   clReleaseMemObject(gpu_v);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGB2I420(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outYStride,
                              cl_mem outY,
                              int outUStride,
                              cl_mem outU,
                              int outVStride,
                              cl_mem outV);

/**
 * @brief Convert BGRA images to I420 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2I420<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2I420(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert BGRA images to I420 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * sizeof(uchar);
 *   int uv_size = height * width / 4 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_u = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   BGRA2I420<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                    width, gpu_y, width / 2, gpu_u, width / 2, gpu_v);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_u);
 *   clReleaseMemObject(gpu_v);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGRA2I420(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outYStride,
                               cl_mem outY,
                               int outUStride,
                               cl_mem outU,
                               int outVStride,
                               cl_mem outV);

/**
 * @brief Convert RGBA images to I420 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2I420<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2I420(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert RGBA images to I420 images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * sizeof(uchar);
 *   int uv_size = height * width / 4 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   cl_mem gpu_u = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   RGBA2I420<uchar>(queue, height, width, width * src_channels, gpu_input,
 *                    width, gpu_y, width / 2, gpu_u, width / 2, gpu_v);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_u);
 *   clReleaseMemObject(gpu_v);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode RGBA2I420(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outYStride,
                               cl_mem outY,
                               int outUStride,
                               cl_mem outU,
                               int outVStride,
                               cl_mem outV);

/**
 * @brief Convert I420 images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   I4202BGR<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert I420 images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, which is not less
 *                       than `width`.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, similar to
 *                       inYStride / 2.
 * @param inU            U-channel input image data.
 * @param inVStride      V-channel width stride of input image, similar to
 *                       inYStride / 2.
 * @param inV            V-channel input image data.
 * @param outWidthStride output image's width stride, which is not less
 *                       than `width * channels`.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * sizeof(uchar);
 *   int uv_size = height * width / 4 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_u = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_READ_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   I4202BGR<uchar>(queue, height, width, width, gpu_y, width / 2, gpu_u,
 *                   width / 2, gpu_v, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_u);
 *   clReleaseMemObject(gpu_v);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inYStride,
                              const cl_mem inY,
                              int inUStride,
                              const cl_mem inU,
                              int inVStride,
                              const cl_mem inV,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert I420 images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   I4202RGB<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert I420 images to RGB images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, which is not less
 *                       than `width`.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, similar to
 *                       inYStride / 2.
 * @param inU            U-channel input image data.
 * @param inVStride      V-channel width stride of input image, similar to
 *                       inYStride / 2.
 * @param inV            V-channel input image data.
 * @param outWidthStride output image's width stride, which is not less
 *                       than `width * channels`.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 3;
 *   if (src_channels == 1) {
 *     src_height = height + (height >> 1);
 *   }
 *   else {
 *     dst_height = height + (height >> 1);
 *   }
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * sizeof(uchar);
 *   int uv_size = height * width / 4 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_u = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_READ_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   I4202RGB<uchar>(queue, height, width, width, gpu_y, width / 2, gpu_u,
 *                   width / 2, gpu_v, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_u);
 *   clReleaseMemObject(gpu_v);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202RGB(cl_command_queue queue,
                              int height,
                              int width,
                              int inYStride,
                              const cl_mem inY,
                              int inUStride,
                              const cl_mem inU,
                              int inVStride,
                              const cl_mem inV,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert I420 images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   I4202BGRA<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202BGRA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert I420 images to BGRA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, which is not less
 *                       than `width`.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, similar to
 *                       inYStride / 2.
 * @param inU            U-channel input image data.
 * @param inVStride      V-channel width stride of input image, similar to
 *                       inYStride / 2.
 * @param inV            V-channel input image data.
 * @param outWidthStride output image's width stride, which is not less
 *                       than `width * channels`.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * sizeof(uchar);
 *   int uv_size = height * width / 4 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_u = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_READ_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   I4202BGRA<uchar>(queue, height, width, width, gpu_y, width / 2, gpu_u,
 *                    width / 2, gpu_v, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_u);
 *   clReleaseMemObject(gpu_v);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202BGRA(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUStride,
                               const cl_mem inU,
                               int inVStride,
                               const cl_mem inV,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert I420 images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = src_height * width * src_channels * sizeof(uchar);
 *   int dst_size = dst_height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   I4202RGBA<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202RGBA(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

/**
 * @brief Convert I420 images to RGBA images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inYStride      Y-channel width stride of input image, which is not less
 *                       than `width`.
 * @param inY            Y-channel input image data.
 * @param inUStride      U-channel width stride of input image, similar to
 *                       inYStride / 2.
 * @param inU            U-channel input image data.
 * @param inVStride      V-channel width stride of input image, similar to
 *                       inYStride / 2.
 * @param inV            V-channel input image data.
 * @param outWidthStride output image's width stride, which is not less
 *                       than `width * channels`.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
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
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * sizeof(uchar);
 *   int uv_size = height * width / 4 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_u = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, uv_size, NULL,
 *                                 &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_READ_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_y, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   I4202RGBA<uchar>(queue, height, width, width, gpu_y, width / 2, gpu_u,
 *                    width / 2, gpu_v, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_y);
 *   clReleaseMemObject(gpu_u);
 *   clReleaseMemObject(gpu_v);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode I4202RGBA(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUStride,
                               const cl_mem inU,
                               int inVStride,
                               const cl_mem inV,
                               int outWidthStride,
                               cl_mem outData);

// YUV -> GRAY

/**
 * @brief Convert YUV images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 *       2 height and width must be even numbers.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 1;
 *   int dst_channels = 1;
 *
 *   createSharedFrameChain(false);
 *   cl_context context = getSharedFrameChain()->getContext();
 *   cl_command_queue queue = getSharedFrameChain()->getQueue();
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   YUV2GRAY<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode YUV2GRAY(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

// BGR/GRAY <-> UYVY

/**
 * @brief Convert UYVY images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>2<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 2;
 *   int dst_channels = 3;
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * 2 * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   UYVY2BGR<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode UYVY2BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert UYVY images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>2<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 2;
 *   int dst_channels = 1;
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * 2 * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   UYVY2GRAY<uchar>(queue, height, width, width * src_channels,
 *                    gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode UYVY2GRAY(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

// BGR/GRAY <-> YUYV

/**
 * @brief Convert YUYV images to BGR images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>2<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 2;
 *   int dst_channels = 3;
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * 2 * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   YUYV2BGR<uchar>(queue, height, width, width * src_channels,
 *                   gpu_input, width * dst_channels, gpu_output);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode YUYV2BGR(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert YUYV images to GRAY images.
 * @tparam T The data type, used for both input image and output image,
 *         currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
 * @param height         input&output image's height.
 * @param width          input&output image's width.
 * @param inWidthStride  input image's width stride, which is not less
 *                       than `width * channels`.
 * @param inData         input image data.
 * @param outWidthStride the width stride of output image, similar to
 *                       inWidthStride.
 * @param outData        output image data.
 * @return The execution status, succeeds or fails with an error code.
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>2<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 1.2
 * <tr><td>Header files  <td> #include "ppl/cv/ocl/cvtcolor.h"
 * <tr><td>Project       <td>ppl.cv
 * </table>
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/ocl/cvtcolor.h"
 * #include "ppl/common/oclcommon.h"
 *
 * using namespace ppl::common::ocl;
 * using namespace ppl::cv::ocl;
 *
 * int main(int argc, char** argv) {
 *   int width  = 640;
 *   int height = 480;
 *   int src_channels = 2;
 *   int dst_channels = 1;
 *
 *   cl_int error_code = 0;
 *   int src_size = height * width * src_channels * 2 * sizeof(uchar);
 *   int dst_size = height * width * dst_channels * 2 * sizeof(uchar);
 *   float* input = (float*)malloc(src_size);
 *   float* output = (float*)malloc(dst_size);
 *   cl_mem gpu_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_size,
 *                                     NULL, &error_code);
 *   cl_mem gpu_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dst_size,
 *                                      NULL, &error_code);
 *   error_code = clEnqueueWriteBuffer(queue, gpu_input, CL_FALSE, 0,
 *                                     src_size, input, 0, NULL, NULL);
 *
 *   free(input);
 *   free(output);
 *   clReleaseMemObject(gpu_input);
 *   clReleaseMemObject(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode YUYV2GRAY(cl_command_queue queue,
                               int height,
                               int width,
                               int inWidthStride,
                               const cl_mem inData,
                               int outWidthStride,
                               cl_mem outData);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_CVTCOLOR_H_
