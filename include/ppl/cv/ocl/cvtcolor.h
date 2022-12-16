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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   BGR2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   BGR2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   RGB2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   RGB2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   BGRA2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   BGRA2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   RGBA2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   RGBA2NV12<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   NV122BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV122BGR<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   NV122RGB<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV122RGB<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   NV122BGRA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV122BGRA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   NV122RGBA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV122RGBA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   BGR2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   BGR2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   RGB2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   RGB2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   BGRA2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   BGRA2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   RGBA2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   RGBA2NV21<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   NV212BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV212BGR<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   NV212RGB<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV212RGB<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   NV212BGRA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV212BGRA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   NV212RGBA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV212RGBA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch / sizeof(uchar), gpu_output);
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
 * @brief Convert RGB images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   RGB2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   RGB2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1,
 *                   output_pitch2 / sizeof(uchar), gpu_output2);
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
 * @brief Convert BGR images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   BGR2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   BGR2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                   output_pitch1 / sizeof(uchar), gpu_output1,
 *                   output_pitch2 / sizeof(uchar), gpu_output2);
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
 * @brief Convert BGRA images to I420 images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   BGRA2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   BGRA2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1,
 *                    output_pitch2 / sizeof(uchar), gpu_output2);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   RGBA2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   RGBA2I420<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1,
 *                    output_pitch2 / sizeof(uchar), gpu_output2);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   I4202BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * @param outWidthStride output image's width stride, which is not less
 *                       than `width * channels`.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   I4202BGR<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   input_pitch2 / sizeof(uchar), gpu_input2,
 *                   output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   I4202RGB<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * @param outWidthStride output image's width stride, which is not less
 *                       than `width * channels`.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   I4202RGB<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                   gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                   input_pitch2 / sizeof(uchar), gpu_input2,
 *                   output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   I4202BGRA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * @param outWidthStride output image's width stride, which is not less
 *                       than `width * channels`.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   I4202BGRA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    input_pitch2 / sizeof(uchar), gpu_input2,
 *                    output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), src_height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), dst_height);
 *
 *   I4202RGBA<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                    gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * @param outWidthStride output image's width stride, which is not less
 *                       than `width * channels`.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   I4202RGBA<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    input_pitch2 / sizeof(uchar), gpu_input2,
 *                    output_pitch / sizeof(uchar), gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @brief Convert BGR images to UYVY images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * @note 1 For best performance, a 2D array allocated by cudaMallocPitch() is
 *         recommended.
 *       2 width must be an even number.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>2
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   int dst_channels = 2;
 *
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * sizeof(uchar), height);
 *
 *   BGR2UYVY<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
 *
 *   return 0;
 * }
 * @endcode
 */
template <typename T>
ppl::common::RetCode BGR2UYVY(cl_command_queue queue,
                              int height,
                              int width,
                              int inWidthStride,
                              const cl_mem inData,
                              int outWidthStride,
                              cl_mem outData);

/**
 * @brief Convert UYVY images to BGR images.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * 2 * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * 2 * sizeof(uchar), height);
 *
 *   UYVY2BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   uchar* gpu_input;
 *   uchar* gpu_output;
 *   size_t input_pitch, output_pitch;
 *   cudaMallocPitch(&gpu_input, &input_pitch,
 *                   width * src_channels * 2 * sizeof(uchar), height);
 *   cudaMallocPitch(&gpu_output, &output_pitch,
 *                   width * dst_channels * 2 * sizeof(uchar), height);
 *
 *   YUYV2BGR<uchar>(stream, height, width, input_pitch / sizeof(uchar),
 *                   gpu_input, output_pitch / sizeof(uchar), gpu_output);
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
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
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *
 *   cudaFree(gpu_input);
 *   cudaFree(gpu_output);
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

//NV12/21 <-> I420

/**
 * @brief Convert NV21 images to I420 images,format: YYYYUVUVUVUV -> YYYYUUUUVVVV.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV122I420<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1,
 *                    output_pitch2 / sizeof(uchar), gpu_output2);
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
ppl::common::RetCode NV122I420(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUVStride,
                               const cl_mem inUV,
                               int outYStride,
                               cl_mem outY,
                               int outUStride,
                               cl_mem outU,
                               int outVStride,
                               cl_mem outV);

/**
 * @brief Convert NV21 images to I420 images,format: YYYYVUVUVUVU -> YYYYUUUUVVVV.
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   NV212I420<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1,
 *                    output_pitch2 / sizeof(uchar), gpu_output2);
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
ppl::common::RetCode NV212I420(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUVStride,
                               const cl_mem inUV,
                               int outYStride,
                               cl_mem outY,
                               int outUStride,
                               cl_mem outU,
                               int outVStride,
                               cl_mem outV);

/**
 * @brief Convert I420 images to NV12 images,format: YYYYUUUUVVVV -> YYYYUVUVUVUV
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   I4202NV12<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    input_pitch2 / sizeof(uchar), gpu_input2,
 *                    output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
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
ppl::common::RetCode I4202NV12(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUStride,
                               const cl_mem inU,
                               int inVStride,
                               const cl_mem inV,
                               int outYStride,
                               cl_mem outY,
                               int outUVStride,
                               cl_mem outUV);

/**
 * @brief Convert I420 images to NV21 images,format: YYYYUUUUVVVV -> YYYYVUVUVUVU
 * @tparam T The data type, used for both input image and output image, currently only uint8_t(uchar) is supported.
 * @param queue          opencl command queue.
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
 * @note For best performance, rows of input&output aligned with 64 bits are
 *       recommended.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>OpenCL platforms supported <td>OpenCL 2.0
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
 *   I4202NV21<uchar>(stream, height, width, input_pitch0 / sizeof(uchar),
 *                    gpu_input0, input_pitch1 / sizeof(uchar), gpu_input1,
 *                    input_pitch2 / sizeof(uchar), gpu_input2,
 *                    output_pitch0 / sizeof(uchar), gpu_output0,
 *                    output_pitch1 / sizeof(uchar), gpu_output1);
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
ppl::common::RetCode I4202NV21(cl_command_queue queue,
                               int height,
                               int width,
                               int inYStride,
                               const cl_mem inY,
                               int inUStride,
                               const cl_mem inU,
                               int inVStride,
                               const cl_mem inV,
                               int outYStride,
                               cl_mem outY,
                               int outUVStride,
                               cl_mem outUV);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_CVTCOLOR_H_
