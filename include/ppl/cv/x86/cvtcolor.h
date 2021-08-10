// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_HPC_PPL_CV_X86_CVTCOLOR_H_
#define __ST_HPC_PPL_CV_X86_CVTCOLOR_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

//RGB_GRAY
/**
 * @brief Convert RGB images to GRAY images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * <tr><td>float<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     float* iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *
 *     ppl::cv::x86::RGB2GRAY<float, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2GRAY(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert GRAY images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * <tr><td>float<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     float* iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *
 *     ppl::cv::x86::GRAY2RGB<float, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode GRAY2RGB(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGBA images to GRAY images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * <tr><td>float<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     float* iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *
 *     ppl::cv::x86::RGBA2GRAY<float, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2GRAY(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert GRAY images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * <tr><td>float<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     float* iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *
 *     ppl::cv::x86::GRAY2RGBA<float, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode GRAY2RGBA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

//BGR_GRAY
/**
 * @brief Convert BGR images to GRAY images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * <tr><td>float<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     float* iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *
 *     ppl::cv::x86::BGR2GRAY<float, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2GRAY(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert GRAY images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * <tr><td>float<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     float* iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *
 *     ppl::cv::x86::GRAY2BGR<float, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode GRAY2BGR(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to GRAY images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * <tr><td>float<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     float* iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *
 *     ppl::cv::x86::BGRA2GRAY<float, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2GRAY(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert GRAY images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * <tr><td>float<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     float* iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *
 *     ppl::cv::x86::GRAY2BGRA<float, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode GRAY2BGRA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

//BGR_I420
/**
 * @brief Convert BGR images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::BGR2I420<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2I420(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGR images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * ncSrc`
 * @param inData            input image data
 * @param outYStride        output Y stride, usually it equals to `width`
 * @param outY              output Y
 * @param outUStride        output U stride, usually it equals to `width / 2`
 * @param outU              output U
 * @param outVStride        output V stride, usually it equals to `width / 2`
 * @param outV              output V
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> X86 platforms supported<td> All
 * <tr><td> Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     int32_t outy_stride = W;
 *     int32_t outu_stride = W / 2;
 *     int32_t outv_stride = W / 2;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_u = (uint8_t*)malloc(W * H / 4 * output_channels * sizeof(uint8_t)); 
 *     uint8_t* dev_oImage_v = (uint8_t*)malloc(W * H / 4 * output_channels * sizeof(uint8_t));
 *     ppl::cv::x86::BGR2I420<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, 
 *              outy_stride, dev_oImage_y, outu_stride,dev_oImage_u, outv_stride, dev_oImage_v);
 *
 *     free(dev_oImage_y);
 *     free(dev_oImage_u);
 *     free(dev_oImage_v);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2I420(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outDataY,
    int32_t outUStride,
    T* outDataU,
    int32_t outVStride,
    T* outDataV);

/**
 * @brief Convert I420 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::I4202BGR<uint8_t, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202BGR(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert I420 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            bgr image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input Y stride, usually it equals to `width`
 * @param inDataY           input Y
 * @param inUStride         input U stride, usually it equals to `width / 2`
 * @param inDataU           input U
 * @param inVStride         input V stride, usually it equals to `width / 2`
 * @param inDataV           input V
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> X86 platforms supported<td> All
 * <tr><td> Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     int32_t iny_stride = W;
 *     int32_t inu_stride = W / 2;
 *     int32_t inv_stride = W / 2;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_iImage_u = (uint8_t*)malloc(W * H / 4 * output_channels * sizeof(uint8_t)); 
 *     uint8_t* dev_iImage_v = (uint8_t*)malloc(W * H / 4 * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::x86::I4202BGR<uint8_t, 1, 3>(H, W, iny_stride, dev_iImage_y, inu_stride, dev_iImage_u, 
 *                                     inv_stride, dev_iImage_v, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_u);
 *     free(dev_iImage_v);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202BGR(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inDataY,
    int32_t inUStride,
    const T* inDataU,
    int32_t inVStride,
    const T* inDataV,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGB images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::RGB2I420<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2I420(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGB images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * ncSrc`
 * @param inData            input image data
 * @param outYStride        output Y stride, usually it equals to `width`
 * @param outDataY          output Y
 * @param outUStride        output U stride, usually it equals to `width / 2`
 * @param outDataU          output U
 * @param outVStride        output V stride, usually it equals to `width / 2`
 * @param outDataV          output V
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> X86 platforms supported<td> All
 * <tr><td> Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     int32_t outy_stride = W;
 *     int32_t outu_stride = W / 2;
 *     int32_t outv_stride = W / 2;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_u = (uint8_t*)malloc(W * H / 4 * output_channels * sizeof(uint8_t)); 
 *     uint8_t* dev_oImage_v = (uint8_t*)malloc(W * H / 4 * output_channels * sizeof(uint8_t));
 *     ppl::cv::x86::RGB2I420<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, 
 *              outy_stride, dev_oImage_y, outu_stride, dev_oImage_u, outv_stride, dev_oImage_v);
 *
 *     free(dev_oImage_y);
 *     free(dev_oImage_u);
 *     free(dev_oImage_v);
 *     free(dev_iImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2I420(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outDataY,
    int32_t outUStride,
    T* outDataU,
    int32_t outVStride,
    T* outDataV);

/**
 * @brief Convert I420 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::I4202RGB<uint8_t, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202RGB(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert I420 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            rgb image's height
 * @param width             rgb image's width need to be processed
 * @param inYStride         stride of y dimension, usually it equals to `width`
 * @param inDataY           input image data of y
 * @param inUStride         stride of u dimension, usually it equals to `width / 2`
 * @param inDataU           input image data of u
 * @param inVStride         stride of v dimension, usually it equals to `width / 2`
 * @param inDataV           input image data of v
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> X86 platforms supported<td> All
 * <tr><td> Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     int32_t yStride = W;
 *     int32_t uStride = W / 2;
 *     int32_t vStride = W / 2;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_u = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_iImage_v = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::x86::I4202RGB<uint8_t, 1, 3>(H, W, yStride, dev_iImage_y, uStride, dev_iImage_u, vStride, dev_iImage_v, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_u);
 *     free(dev_iImage_v);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202RGB(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inDataY,
    int32_t inUStride,
    const T* inDataU,
    int32_t inVStride,
    const T* inDataV,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint_8* iImage = (uint_8*)malloc(W * H * input_channels * sizeof(uint_8));
 *     uint_8* oImage = (uint_8*)malloc(W * H * sizeof(uint_8) * 3 / 2);
 *
 *     ppl::cv::x86::BGRA2I420<uint_8, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2I420(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        output Y stride, usually it equals to `width`
 * @param outDataY          output Y
 * @param outUStride        output U stride, usually it equals to `width / 2`
 * @param outDataU          output U
 * @param outVStride        output V stride, usually it equals to `width / 2`
 * @param outDataV          output V
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> X86 platforms supported<td> All
 * <tr><td> Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     int32_t yStride = W;
 *     int32_t uStride = W / 2;
 *     int32_t vStride = W / 2;
 *     uint_8* dev_iImage = (uint_8*)malloc(W * H * input_channels * sizeof(uint_8));
 *     uint_8* dev_oImage_y = (uint_8*)malloc(W * H * sizeof(uint_8));
 *     uint_8* dev_oImage_u = (uint_8*)malloc(W * H / 4 * sizeof(uint_8));
 *     uint_8* dev_oImage_v = (uint_8*)malloc(W * H / 4 * sizeof(uint_8));
 *     ppl::cv::x86::BGRA2I420<uint_8, 4, 1>(H, W, W * input_channels, dev_iImage, yStride, dev_oImage_y, uStride, dev_oImage_u, vStride, dev_oImage_v);
 *
 *     free(dev_iImage);
 *     free(dev_oImage_y);
 *     free(dev_oImage_u);
 *     free(dev_oImage_v);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2I420(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    T* inData,
    int32_t outYStride,
    T* outDataY,
    int32_t outUStride,
    T* outDataU,
    int32_t outVStride,
    T* outDataV);

/**
 * @brief Convert I420 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::I4202BGRA<uint8_t, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202BGRA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert I420 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            bgra image's height
 * @param width             bgra image's width need to be processed
 * @param inYStride         stride of y dimension, usually it equals to `width`
 * @param inDataY           input image data of y
 * @param inUStride         stride of u dimension, usually it equals to `width / 2`
 * @param inDataU           input image data of u
 * @param inVStride         stride of v dimension, usually it equals to `width / 2`
 * @param inDataV           input image data of v
 * @param outWidthStride    the width stride of output image, usually it equals to `width * ncDst`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> X86 platforms supported<td> All
 * <tr><td> Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     const int32_t yStride = W;
 *     const int32_t uStride = W / 2;
 *     const int32_t vStride = W / 2;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_u = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_iImage_v = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::x86::I4202BGRA<uint8_t, 1, 4>(H, W, yStride, dev_iImage_y, uStride, dev_iImage_u, vStride, dev_iImage_v, w*output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_u);
 *     free(dev_iImage_v);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202BGRA(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inDataY,
    int32_t inUStride,
    const T* inDataU,
    int32_t inVStride,
    const T* inDataV,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGBA images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::RGBA2I420<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2I420(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGBA images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * ncSrc`
 * @param inData            input image data
 * @param outYStride        output Y stride, usually it equals to `width`
 * @param outDataY          output Y
 * @param outUStride        output U stride, usually it equals to `width / 2`
 * @param outDataU          output U
 * @param outVStride        output V stride, usually it equals to `width / 2`
 * @param outDataV          output V
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> X86 platforms supported<td> All
 * <tr><td> Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     int32_t yStride = W;
 *     int32_t uStride = W / 2;
 *     int32_t vStride = W / 2;
 *     uint_8* dev_iImage = (uint_8*)malloc(W * H * input_channels * sizeof(uint_8));
 *     uint_8* dev_oImage_y = (uint_8*)malloc(W * H * sizeof(uint_8));
 *     uint_8* dev_oImage_u = (uint_8*)malloc(W * H / 4 * sizeof(uint_8));
 *     uint_8* dev_oImage_v = (uint_8*)malloc(W * H / 4 * sizeof(uint_8));
 *     ppl::cv::x86::RGBA2I420<uint_8, 4, 1>(H, W, W * input_channels, dev_iImage, yStride, dev_oImage_y, uStride, dev_oImage_u, vStride, dev_oImage_v);
 *
 *     free(dev_iImage);
 *     free(dev_oImage_y);
 *     free(dev_oImage_u);
 *     free(dev_oImage_v);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2I420(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outDataY,
    int32_t outUStride,
    T* outDataU,
    int32_t outVStride,
    T* outDataV);

/**
 * @brief Convert I420 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::I4202RGBA<uint8_t, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202RGBA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert I420 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         stride of y dimension, usually it equals to `width`
 * @param inDataY           input image data of y
 * @param inUStride         stride of u dimension, usually it equals to `width / 2`
 * @param inDataU           input image data of u
 * @param inVStride         stride of v dimension, usually it equals to `width / 2`
 * @param inDataV           input image data of v
 * @param outWidthStride    the width stride of output image, usually it equals to `width * ncDst`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> X86 platforms supported<td> All
 * <tr><td> Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     int32_t yStride = W;
 *     int32_t uStride = W / 2;
 *     int32_t vStride = W / 2;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_u = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_iImage_v = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::x86::I4202RGBA<uint8_t, 1, 4>(H, W, yStride, dev_iImage_y, uStride, dev_iImage_u, vStride, dev_iImage_v, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_u);
 *     free(dev_iImage_v);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202RGBA(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inDataY,
    int32_t inUStride,
    const T* inDataU,
    int32_t inVStride,
    const T* inDataV,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGR images to YV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::BGR2YV12<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2YV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert YV12 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::YV122BGR<uint8_t, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode YV122BGR(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGB images to YV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::RGB2YV12<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2YV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert YV12 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::YV122RGB<uint8_t, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode YV122RGB(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to YV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint_8* iImage = (uint_8*)malloc(W * H * input_channels * sizeof(uint_8));
 *     uint_8* oImage = (uint_8*)malloc(W * H * sizeof(uint_8) * 3 / 2);
 *
 *     ppl::cv::x86::BGRA2YV12<uint_8, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2YV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert YV12 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::YV122BGRA<uint8_t, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode YV122BGRA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGBA images to YV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::RGBA2YV12<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2YV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert YV12 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::YV122RGBA<uint8_t, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode YV122RGBA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

//BGR_NV12
/**
 * @brief Convert BGR images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::BGR2NV12<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2NV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert BGR images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        the width stride of output Y plane image, usually it equals to `width`
 * @param outY              output image data
 * @param outUVStride       the width stride of output UV plane image, usually it equals to `width`
 * @param outUV             output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));

 *     ppl::cv::x86::BGR2NV12<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage_y, W * output_channels, oImage_uv);
 *
 *     free(iImage);
 *     free(oImage_y);
 *     free(oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2NV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outY,
    int32_t outUVStride,
    T* outUV);

/**
 * @brief Convert NV12 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122BGR<uint8_t, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122BGR(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert NV12 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image Y plane stride, usually it equals to `width`
 * @param inY               input image Y plane data
 * @param inUVStride        input image UV width stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122BGR<uint8_t, 1, 3>(H, W, W * input_channels, iImage_y, W * input_channels, iImage_uv, W * output_channels, oImage);
 *
 *     free(iImage_y);
 *     free(iImage_uv);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/

template <typename T>
::ppl::common::RetCode NV122BGR(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inY,
    int32_t inUVStride,
    const T* inUV,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::BGRA2NV12<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2NV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        output image Y plane stride, usually it equals to `width`
 * @param outY              output image Y plane data
 * @param outUVStride       output image UV plane stride, usually it equals to `width`
 * @param outUV             output image UV plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::BGRA2NV12<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage_y, W * output_channels, oImage_uv);
 *
 *     free(iImage);
 *     free(oImage_y);
 *     free(oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/

template <typename T>
::ppl::common::RetCode BGRA2NV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYtride,
    T* outY,
    int32_t outUVStride,
    T* outUV);

/**
 * @brief Convert NV12 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122BGRA<uint8_t, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122BGRA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert NV12 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image Y plane stride, usually it equals to `width`
 * @param inY               input image Y plane data
 * @param inUVStride        input image UV plane stride, usually it equals to `width`
 * @param inUV              input image UV plane data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122BGRA<uint8_t, 1, 4>(H, W, W * input_channels, iImage_y, W * input_channels, iImage_uv, W * output_channels, oImage);
 *
 *     free(iImage_y);
 *     free(iImage_uv);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122BGRA(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inY,
    int32_t inUVStride,
    const T* inU,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert BGR images to NV21 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>x86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (float*)malloc(W * H * 3 / 2 * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::BGR2NV21<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2NV21(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert BGR images to NV21 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        output image Y plane stride, usually it equals to `width`
 * @param outY              output image Y plane data
 * @param outUVStride       output image UV plane stride, usually it equals to `width`
 * @param outUV             output image UV plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::BGR2NV21<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage_y, W * output_channels, oImage_uv);
 *
 *     free(iImage);
 *     free(oImage_y);
 *     free(oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2NV21(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outY,
    int32_t outUVStride,
    T* outUV);

/**
 * @brief Convert NV21 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212BGR<uint8_t, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212BGR(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert NV21 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image Y plane stride, usually it equals to `width`
 * @param inY               input image Y plane data
 * @param inUVStride        input image UV plane stride, usually it equals to `width`
 * @param inUV              input image UV plane data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212BGR<uint8_t, 1, 3>(H, W, W * input_channels, iImage_y, W * input_channels, iImage_uv, W * output_channels, oImage);
 *
 *     free(iImage_y);
 *     free(iImage_uv);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212BGR(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inY,
    int32_t inUVStride,
    const T* inU,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to NV21 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::BGRA2NV21<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2NV21(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to NV21 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        output image Y plane stride, usually it equals to `width`
 * @param outY              output image Y plane data
 * @param outUVStride       output image UV plane stride, usually it equals to `width`
 * @param outUV             output image UV plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::BGRA2NV21<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage_y,  W * output_channels, oImage_uv);
 *
 *     free(iImage);
 *     free(oImage_y);
 *     free(oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2NV21(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outY,
    int32_t outUVStride,
    T* outUV);

/**
 * @brief Convert NV21 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212BGRA<uint8_t, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212BGRA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert NV21 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image Y plane stride, usually it equals to `width`
 * @param inY               input image Y plane data
 * @param inUVStride        input image UV plane stride, usually it equals to `width`
 * @param inUV              input image UV plane data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
       uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212BGRA<uint8_t, 1, 4>(H, W, W * input_channels, iImage_y, W * input_channels, iImage_uv, W * output_channels, oImage);
 *
 *     free(iImage_y);
 *     free(iImage_uv);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212BGRA(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inY,
    int32_t inUVStride,
    const T* inU,
    int32_t outWidthStride,
    T* outData);

//RGB_NV12
/**
 * @brief Convert RGB images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::RGB2NV12<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2NV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGB images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        the Y plane stride of output image, usually it equals to `width * channels`
 * @param outY              output Y plane image data
 * @param outUVStride       the UV plabe stride of output image, usually it equals to `width * channels`
 * @param outUV             output UV plane image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *     ppl::cv::x86::RGB2NV12<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage_y, W * output_channels, oImage_uv);
 *
 *     free(iImage);
 *     free(oImage_y);
 *     free(oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2NV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outY,
    int32_t outUVStride,
    T* outUV);

/**
 * @brief Convert NV12 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122RGB<uint8_t, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122RGB(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert NV12 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input y plane's width stride, usually it equals to `width`
 * @param inY               input y plane data
 * @param inUVStride        input uv plane's width stride, usually it equals to `width`
 * @param inUV              input uv plane data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(2 * (W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122RGB<uint8_t, 1, 3>(H, W, W * input_channels, iImage_y, W * input_channels, iImage_uv, W * output_channels, oImage);
 *
 *     free(iImage_y);
 *     free(iImage_uv);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122RGB(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inY,
    int32_t inUVStride,
    const T* inU,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGBA images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::RGBA2NV12<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2NV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert RGBA images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        the width stride of y plane, usually it equals to `width`
 * @param outY              output y plane data
 * @param outUVStride       the width stride of uv plane, usually it equals to `width`
 * @param outUV             output uv plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_vu = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *     ppl::cv::x86::RGBA2NV12<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage_y);
 *     free(oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2NV12(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outY,
    int32_t outUVStride,
    T* outUV);

/**
 * @brief Convert NV12 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122RGBA<uint8_t, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122RGBA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert NV12 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input y plane 's width stride, usually it equals to `width`
 * @param inY               input y plane data
 * @param inUVStride        input uv plane 's width stride, usually it equals to `width`
 * @param inUV              input uv plane data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122RGBA<uint8_t, 1, 4>(H, W, W, iImage_y, W, iImage_uv, W * output_channels, oImage);
 *
 *     free(iImage_y);
 *     free(iImage_uv);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122RGBA(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inY,
    int32_t inUVStride,
    const T* inUV,
    int32_t outWidthStride,
    T* outData);

//RGB_NV21
/**
 * @brief Convert RGB images to NV21 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::RGB2NV21<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2NV21(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert RGB images to NV21 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        output image Y plane stride, usually it equals to `width`
 * @param outY              output image Y plane data
 * @param outUVStride       output image UV plane stride, usually it equals to `width`
 * @param outUV             output image UV plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *     ppl::cv::x86::RGB2NV21<uint8_t, 3, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage_y, W * output_channels, oImage_uv);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2NV21(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outY,
    int32_t outUVStride,
    T* outUV);

/**
 * @brief Convert NV21 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212RGB<uint8_t, 1, 3>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212RGB(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert NV21 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image Y plane stride, usually it equals to `width`
 * @param inY               input image Y plane data
 * @param inUVStride        input image UV plane stride, usually it equals to `width`
 * @param inUV              input image UV plane data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212RGB<uint8_t, 1, 3>(H, W, W * input_channels, iImage_y, W * input_channels, iImage_uv, W * output_channels, oImage);
 *
 *     free(iImage_y);
 *     free(iImage_uv);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212RGB(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inY,
    int32_t inUVStride,
    const T* inUV,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert RGBA images to NV21 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::x86::RGBA2NV21<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2NV21(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert RGBA images to NV21 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        output image Y plane stride, usually it equals to `width`
 * @param outY              output image Y plane data
 * @param outUVStride       output image UV plane stride, usually it equals to `width`
 * @param outUV             output image UV plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *     ppl::cv::x86::RGBA2NV21<uint8_t, 4, 1>(H, W, W * input_channels, iImage, W * output_channels, oImage_y, W * output_channels, oImage_uv);
 *
 *     free(iImage);
 *     free(oImage_y);
 *     free(oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2NV21(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outYStride,
    T* outY,
    int32_t outUVStride,
    T* outUV);

/**
 * @brief Convert NV21 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> all
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212RGBA<uint8_t, 1, 4>(H, W, W * input_channels, iImage, W * output_channels, oImage);
 *
 *     free(iImage);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212RGBA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);
/**
 * @brief Convert NV21 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image Y plane stride, usually it equals to `width`
 * @param inY               input image Y plane data
 * @param inUVStride        input image UV plane stride, usually it equals to `width`
 * @param inUV              input image UV plane data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212RGBA<uint8_t, 1, 4>(H, W, W * input_channels, iImage_y, W * input_channels, iImage_uv, W * output_channels, oImage);
 *
 *     free(iImage_y);
 *     free(iImage_uv);
 *     free(oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212RGBA(
    int32_t height,
    int32_t width,
    int32_t inYStride,
    const T* inY,
    int32_t inUVStride,
    const T* inUV,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert I420 images to NV21 images,format: YYYYUUUUVVVV -> YYYYVUVUVUVU
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inStrideY         input Y plane width stride, usually it equals to `width`
 * @param inDataY           input Y plane data
 * @param inStrideU         input U plane width stride, usually it equals to `width / 2`
 * @param inDataU           input U plane data
 * @param inStrideV         input V plane width stride, usually it equals to `width / 2`
 * @param inDataV           input V plane data
 * @param outStrideY        output Y plane width stride, usually it equals to `width`
 * @param outDataY          output Y plane data
 * @param outStrideUV       output UV plane width stride, usually it equals to `width`
 * @param outDataUV         output UV plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_u = (uint8_t*)malloc((W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_v = (uint8_t*)malloc((W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::I4202NV12<uint8_t, 1, 1>(H, W, W, iImage_y, W / 2, iImage_u, W / 2, iImage_v, W, oImage_y, W, oImage_uv);
 *     free(iImage_y);
 *     free(iImage_u);
 *     free(iImage_v);
 *     free(oImage_y);
 *     free(oImage_uv);
 *
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202NV21(
    int32_t height,
    int32_t width,
    int32_t inStrideY,
    const T* inDataY,
    int32_t inStrideU,
    const T* inDataU,
    int32_t inStrideV,
    const T* inDataV,
    int32_t outStrideY,
    T* outDataY,
    int32_t outStrideVU,
    T* outDataVU);

/**
 * @brief Convert NV21 images to I420 images,format: YYYYVUVUVUVU -> YYYYUUUUVVVV
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inStrideY         input Y plane width stride, usually it equals to `width`
 * @param inDataY           input Y plane data
 * @param inStrideUV        input UV plane width stride, usually it equals to `width`
 * @param inDataUV          input UV plane data
 * @param outStrideY        output Y plane width stride, usually it equals to `width`
 * @param outDataY          output Y plane data
 * @param outStrideU        output U plane width stride, usually it equals to `width / 2`
 * @param outDataU          output U plane data
 * @param outStrideV        output V plane width stride, usually it equals to `width / 2`
 * @param outDataV          output V plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 1;
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_u = (uint8_t*)malloc((W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_v = (uint8_t*)malloc((W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV212I420<uint8_t, 1, 1>(H, W, W, iImage_y, W, iImage_uv, W, oImage_y, W / 2, oImage_u, W / 2, oImage_v);
 *
 *     free(oImage_y);
 *     free(oImage_u);
 *     free(oImage_v);
 *     free(iImage_y);
 *     free(iImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212I420(
    int32_t height,
    int32_t width,
    int32_t inStrideY,
    const T* inDataY,
    int32_t inStrideVU,
    const T* inDataVU,
    int32_t outStrideY,
    T* outDataY,
    int32_t outStrideU,
    T* outDataU,
    int32_t outStrideV,
    T* outDataV);
/**
 * @brief Convert I420 images to NV12 images,format: YYYYUUUUVVVV -> YYYYUVUVUVUV
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inStrideY         input Y plane width stride, usually it equals to `width`
 * @param inDataY           input Y plane data
 * @param inStrideU         input U plane width stride, usually it equals to `width / 2`
 * @param inDataU           input U plane data
 * @param inStrideV         input V plane width stride, usually it equals to `width / 2`
 * @param inDataV           input V plane data
 * @param outStrideY        output Y plane width stride, usually it equals to `width`
 * @param outDataY          output Y plane data
 * @param outStrideUV       output UV plane width stride, usually it equals to `width`
 * @param outDataUV         output UV plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 1;
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_u = (uint8_t*)malloc((W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_v = (uint8_t*)malloc((W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* oImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::I4202NV12<uint8_t, 1, 1>(H, W, W, iImage_y, W / 2, iImage_u, W / 2, iImage_v, W, oImage_y, W, oImage_uv);
 *
 *     free(iImage_y);
 *     free(iImage_u);
 *     free(iImage_v);
 *     free(oImage_y);
 *     free(oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202NV12(
    int32_t height,
    int32_t width,
    int32_t inStrideY,
    const T* inDataY,
    int32_t inStrideU,
    const T* inDataU,
    int32_t inStrideV,
    const T* inDataV,
    int32_t outStrideY,
    T* outDataY,
    int32_t outStrideUV,
    T* outDataUV);

/**
 * @brief Convert NV12 images to I420 images,format: YYYYUVUVUVUV-> YYYYUUUUVVVV
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inStrideY         input Y plane width stride, usually it equals to `width`
 * @param inDataY           input Y plane data
 * @param inStrideUV        input UV plane width stride, usually it equals to `width`
 * @param inDataUV          input UV plane data
 * @param outStrideY        output Y plane width stride, usually it equals to `width`
 * @param outDataY          output Y plane data
 * @param outStrideU        output U plane width stride, usually it equals to `width / 2`
 * @param outDataU          output U plane data
 * @param outStrideV        output V plane width stride, usually it equals to `width / 2`
 * @param outDataV          output V plane data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uint8_t)<td>1<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/cvtcolor.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 1;
 *     uint8_t* oImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_u = (uint8_t*)malloc((W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* oImage_v = (uint8_t*)malloc((W / 2) * (H / 2) * input_channels * sizeof(uint8_t));
 *     uint8_t* iImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* iImage_uv = (uint8_t*)malloc(W * H / 2 * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::x86::NV122I420<uint8_t, 1, 1>(H, W, W, iImage_y, W, iImage_uv, W, oImage_y, W / 2, oImage_u, W / 2, oImage_v);
 *
 *     free(oImage_y);
 *     free(oImage_u);
 *     free(oImage_v);
 *     free(iImage_y);
 *     free(iImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122I420(
    int32_t height,
    int32_t width,
    int32_t inStrideY,
    const T* inDataY,
    int32_t inStrideUV,
    const T* inDataUV,
    int32_t outStrideY,
    T* outDataY,
    int32_t outStrideU,
    T* outDataU,
    int32_t outStrideV,
    T* outDataV);

}
}
} // namespace ppl::cv::x86

#endif //! __ST_HPC_PPL3_CV_X86_CVTCOLOR_H_
