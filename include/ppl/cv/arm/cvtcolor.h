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

#ifndef __ST_HPC_PPL_CV_AARCH64_CVTCOLOR_H_
#define __ST_HPC_PPL_CV_AARCH64_CVTCOLOR_H_

#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace arm {

/**
 * @brief Convert BGR images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 4;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::BGR2BGRA<float>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2BGRA(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 3;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::BGRA2BGR<float>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2BGR(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outData);

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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::RGB2GRAY<float>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::GRAY2RGB<float>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::RGBA2GRAY<float>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::GRAY2RGBA<float, 1, 4>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 3;
 *     const int32_t output_channels = 1;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::BGR2GRAY<float>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 3;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::GRAY2BGR<float>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 4;
 *     const int32_t output_channels = 1;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::BGRA2GRAY<float, 4, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)
 * <tr><td>uint8_t
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> AARCH64 platforms supported<td> all
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t input_channels = 1;
 *     const int32_t output_channels = 4;
 *     float* dev_iImage = (float*)malloc(W * H * input_channels * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * output_channels * sizeof(float));
 *     
 *     ppl::cv::arm::GRAY2BGRA<float, 1, 4>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::BGR2I420<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2I420(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     int outy_stride = W;
 *     int outu_stride = W / 2;
 *     int outv_stride = W / 2;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_u = (uint8_t*)malloc(W * H / 4 * output_channels * sizeof(uint8_t)); 
 *     uint8_t* dev_oImage_v = (uint8_t*)malloc(W * H / 4 * output_channels * sizeof(uint8_t));
 *     ppl::cv::arm::BGR2I420<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, 
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
 * @brief Convert I420 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            bgr image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::I4202BGR<uint8_t, 1, 3>(H, W, W, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202BGR(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert I420 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            bgr image's height
 * @param width             bgr image's width need to be processed
 * @param inYStride         stride of y dimension, usually it equals to `width`
 * @param inY               input image data of y
 * @param inUStride         stride of u dimension, usually it equals to `width / 2`
 * @param inU               input image data of u
 * @param inVStride         stride of v dimension, usually it equals to `width / 2`
 * @param inv               input image data of v
 * @param outWidthStride    the width stride of output image, usually it equals to `width * ncDst`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     int yStride = W;
 *     int uStride = W / 2;
 *     int vStride = W / 2;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_u = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_iImage_v = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::I4202BGR<uint8_t, 1, 3>(H, W, yStride, dev_iImage_y, uStride, dev_iImage_u, vStride, dev_iImage_v, 
 *                              W * output_channels, dev_oImage);
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
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUStride,
    const T* inU,
    int inVStride,
    const T* inv,
    int outWidthStride,
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
 * @param outWidthStride    the width stride of output image, usually it equals to `width`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * 3 / 2 * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::RGB2I420<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2I420(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outY              output Y
 * @param outUStride        output U stride, usually it equals to `width / 2`
 * @param outU              output U
 * @param outVStride        output V stride, usually it equals to `width / 2`
 * @param outV              output V
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     int yStride = W;
 *     int uStride = W / 2;
 *     int vStride = W / 2;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_oImage_u = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t)); 
 *     uint8_t* dev_oImage_v = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     ppl::cv::arm::RGB2I420<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, 
 *          yStride, dev_oImage_y, uStride, dev_oImage_u, vStride, dev_oImage_v);
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
::ppl::common::RetCode RGB2I420(
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
 * @brief Convert I420 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            rgb image's height
 * @param width             rgb image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output, usually it equals to `width * ncDst`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::I4202RGB<uint8_t, 1, 3>(H, W, W, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202RGB(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert I420 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            rgb image's height
 * @param width             rgb image's width need to be processed
 * @param inYStride         stride of y dimension, usually it equals to `width`
 * @param inY               input image data of y
 * @param inUStride         stride of u dimension, usually it equals to `width / 2`
 * @param inU               input image data of u
 * @param inVStride         stride of v dimension, usually it equals to `width / 2`
 * @param inv               input image data of v
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     int yStride = W;
 *     int uStride = W / 2;
 *     int vStride = W / 2;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_u = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_iImage_v = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::I4202RGB<uint8_t, 1, 3>(H, W, yStride, dev_iImage_y, uStride, dev_iImage_u, vStride, dev_iImage_v, W * output_channels, dev_oImage);
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
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUStride,
    const T* inU,
    int inVStride,
    const T* inv,
    int outWidthStride,
    T* outData);

/**
 * @brief Convert BGRA images to I420 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * ncSrc`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint_8* dev_iImage = (uint_8*)malloc(W * H * input_channels * sizeof(uint_8));
 *     uint_8* dev_oImage = (uint_8*)malloc(W * H * sizeof(uint_8) * 3 / 2);
 *     
 *     ppl::cv::arm::BGRA2I420<uint_8, 4, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2I420(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outY              output Y
 * @param outUStride        output U stride, usually it equals to `width / 2`
 * @param outU              output U
 * @param outVStride        output V stride, usually it equals to `width / 2`
 * @param outV              output V
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     int yStride = W;
 *     int uStride = W / 2;
 *     int vStride = W / 2;
 *     uint_8* dev_iImage = (uint_8*)malloc(W * H * input_channels * sizeof(uint_8));
 *     uint_8* dev_oImage_y = (uint_8*)malloc(W * H * sizeof(uint_8));
 *     uint_8* dev_oImage_u = (uint_8*)malloc(W * H / 4 * sizeof(uint_8));
 *     uint_8* dev_oImage_v = (uint_8*)malloc(W * H / 4 * sizeof(uint_8));
 *     ppl::cv::arm::BGRA2I420<uint_8, 4, 1>(H, W, W * input_channels, dev_iImage, yStride, dev_oImage_y, uStride, dev_oImage_u, vStride, dev_oImage_v);
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
 * @brief Convert I420 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            bgra image's height
 * @param width             bgra image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::I4202BGRA<uint8_t, 1, 4>(H, W, W, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202BGRA(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert I420 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            bgra image's height
 * @param width             bgra image's width need to be processed
 * @param inYStride         stride of y dimension, usually it equals to `width`
 * @param inY               input image data of y
 * @param inUStride         stride of u dimension, usually it equals to `width / 2`
 * @param inU               input image data of u
 * @param inVStride         stride of v dimension, usually it equals to `width / 2`
 * @param inv               input image data of v
 * @param outWidthStride    the width stride of output image, usually it equals to `width * ncDst`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     const int yStride = W;
 *     const int uStride = W / 2;
 *     const int vStride = W / 2;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_u = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_iImage_v = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::I4202BGRA<uint8_t, 1, 4>(H, W, yStride, dev_iImage_y, uStride, dev_iImage_u, vStride, dev_iImage_v, w*output_channels, dev_oImage);
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
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUStride,
    const T* inU,
    int inVStride,
    const T* inv,
    int outWidthStride,
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
 * @param outWidthStride    the width stride of output image, usually it equals to `width`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::RGBA2I420<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2I420(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outY              output Y
 * @param outUStride        output U stride, usually it equals to `width / 2`
 * @param outU              output U
 * @param outVStride        output V stride, usually it equals to `width / 2`
 * @param outV              output V
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     int yStride = W;
 *     int uStride = W / 2;
 *     int vStride = W / 2;
 *     uint_8* dev_iImage = (uint_8*)malloc(W * H * input_channels * sizeof(uint_8));
 *     uint_8* dev_oImage_y = (uint_8*)malloc(W * H * sizeof(uint_8));
 *     uint_8* dev_oImage_u = (uint_8*)malloc(W * H / 4 * sizeof(uint_8));
 *     uint_8* dev_oImage_v = (uint_8*)malloc(W * H / 4 * sizeof(uint_8));
 *     ppl::cv::arm::RGBA2I420<uint_8, 4, 1>(H, W, W * input_channels, dev_iImage, yStride, dev_oImage_y, uStride, dev_oImage_u, vStride, dev_oImage_v);
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
 * @brief Convert I420 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            rgba image's height
 * @param width             rgba image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * ncSrc`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::I4202RGBA<uint8_t, 1, 4>(H, W, W, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode I4202RGBA(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);

/**
 * @brief Convert I420 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         stride of y dimension, usually it equals to `width`
 * @param inY               input image data of y
 * @param inUStride         stride of u dimension, usually it equals to `width / 2`
 * @param inU               input image data of u
 * @param inVStride         stride of v dimension, usually it equals to `width / 2`
 * @param inv               input image data of v
 * @param outWidthStride    the width stride of output image, usually it equals to `width * ncDst`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     int yStride = W;
 *     int uStride = W / 2;
 *     int vStride = W / 2;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_u = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_iImage_v = (uint8_t*)malloc(W * H / 4 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::I4202RGBA<uint8_t, 1, 4>(H, W, yStride, dev_iImage_y, uStride, dev_iImage_u, vStride, dev_iImage_v, W * output_channels, dev_oImage);
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
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUStride,
    const T* inU,
    int inVStride,
    const T* inv,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::arm::BGR2YV12<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2YV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::arm::YV122BGR<uint8_t, 1, 3>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode YV122BGR(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::arm::RGB2YV12<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2YV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::arm::YV122RGB<uint8_t, 1, 3>(H, W, W, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode YV122RGB(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint_8* dev_iImage = (uint_8*)malloc(W * H * input_channels * sizeof(uint_8));
 *     uint_8* dev_oImage = (uint_8*)malloc(W * H * sizeof(uint_8) * 3 / 2);
 *
 *     ppl::cv::arm::BGRA2YV12<uint_8, 4, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2YV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::arm::YV122BGRA<uint8_t, 1, 4>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode YV122BGRA(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::arm::RGBA2YV12<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2YV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::arm::YV122RGBA<uint8_t, 1, 4>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode YV122RGBA(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::BGR2NV12<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2NV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outYStride        the width stride of output Y image, usually it equals to `width`
 * @param outY              output Y data
 * @param outUVStride       the width stride of output UV image, usually it equals to `width`
 * @param outUV             output UV data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
     
 *     ppl::cv::arm::BGR2NV12<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage_y, W, dev_oImage_uv);
 *
 *     free(dev_iImage);
 *     free(dev_oImage_y);
 *     free(dev_oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2NV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outYStride,
    T* outY,
    int outUVStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV122BGR<uint8_t, 1, 3>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/

template <typename T>
::ppl::common::RetCode NV122BGR(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert NV12 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image's Y stride, usually it equals to `width`
 * @param inY               input image Y data
 * @param inUVStride        input image's UV stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_iImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV122BGR<uint8_t, 1, 3>(H, W, W, dev_iImage_y, W, dev_iImage_uv, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_uv);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/

template <typename T>
::ppl::common::RetCode NV122BGR(
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUVStride,
    const T* inUV,
    int outWidthStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::BGRA2NV12<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2NV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert BGRA images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 4 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's stride, usually it equals to `width * channels`
 * @param inData            input image Y data
 * @param outYStride        the width stride of output Y image, usually it equals to `width`
 * @param outY              output Y data
 * @param outUVStride       the width stride of output UV image, usually it equals to `width`
 * @param outUV             output UV data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_oImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::BGRA2NV12<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage_y, W, dev_oImage_uv);
 *
 *     free(dev_iImage);
 *     free(dev_oImage_y);
 *     free(dev_oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2NV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outYStride,
    T* outY,
    int outUVStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV122BGRA<uint8_t, 1, 4>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122BGRA(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert NV12 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image's Y stride, usually it equals to `width`
 * @param inY               input image Y data
 * @param inUVStride        input image's UV stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_iImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV122BGRA<uint8_t, 1, 4>(H, W, W, dev_iImage_y, W, dev_iImage_uv, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_uv);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122BGRA(
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUVStride,
    const T* inUV,
    int outWidthStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::BGR2NV21<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2NV21(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outYStride        the width stride of output Y image, usually it equals to `width`
 * @param outY              output Y data
 * @param outUVStride       the width stride of output UV image, usually it equals to `width`
 * @param outUV             output UV data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::BGR2NV21<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage_y, W, dev_oImage_uv);
 *
 *     free(dev_iImage);
 *     free(dev_oImage_y);
 *     free(dev_oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGR2NV21(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outYStride,
    T* outY,
    int outUVStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV212BGR<uint8_t, 1, 3>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212BGR(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert NV21 images to BGR images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image's Y stride, usually it equals to `width`
 * @param inY               input image Y data
 * @param inUVStride        input image's UV stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_iImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::arm::NV212BGR<uint8_t, 1, 3>(H, W, W, dev_iImage_y, W, dev_iImage_uv, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_uv);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212BGR(
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUVStride,
    const T* inVU,
    int outWidthStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::BGRA2NV21<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2NV21(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outYStride        the width stride of output Y image, usually it equals to `width`
 * @param outY              output Y data
 * @param outUVStride       the width stride of output UV image, usually it equals to `width`
 * @param outUV             output UV data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
  *     uint8_t* dev_oImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::BGRA2NV21<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage_y, W, dev_oImage_uv);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode BGRA2NV21(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outYStride,
    T* outY,
    int outUVStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV212BGRA<uint8_t, 1, 4>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212BGRA(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert NV21 images to BGRA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image's Y stride, usually it equals to `width`
 * @param inY               input image Y data
 * @param inUVStride        input image's UV stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_iImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV212BGRA<uint8_t, 1, 4>(H, W, W, dev_iImage_y, W, dev_iImage_uv, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_uv);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212BGRA(
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUVStride,
    const T* inUV,
    int outWidthStride,
    T* outData);

/**
 * @brief Convert RGB images to NV12 images
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::RGB2NV12<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2NV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);

/**
 * @brief Convert RGB images to NV12 images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 3 is supported.
 * @tparam ncDst The number of channels of output image, 1 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outYStride        the width stride of output Y image, usually it equals to `width`
 * @param outY              output Y data
 * @param outUVStride       the width stride of output UV image, usually it equals to `width`
 * @param outUV             output UV data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     ppl::cv::arm::RGB2NV12<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage_y, W, dev_oImage_uv);
 *
 *     free(dev_iImage);
 *     free(dev_oImage_y);
 *     free(dev_oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2NV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outYStride,
    T* outY,
    int outUVStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *
 *     ppl::cv::arm::NV122RGB<uint8_t, 1, 3>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122RGB(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);

/**
 * @brief Convert NV12 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t and \a float are supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image's Y stride, usually it equals to `width`
 * @param inY               input image Y data
 * @param inUVStride        input image's UV stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV122RGB<uint8_t, 1, 3>(H, W, W, dev_iImage_y, W, dev_iImage_uv, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_uv);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122RGB(
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUVStride,
    const T* inUV,
    int outWidthStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::RGBA2NV12<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2NV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outYStride        the width stride of output Y image, usually it equals to `width`
 * @param outY              output Y data
 * @param outUVStride       the width stride of output UV image, usually it equals to `width`
 * @param outUV             output UV data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_vu = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     ppl::cv::arm::RGBA2NV12<uint8_t, 4, 1>(H, W, W, dev_oImage_y, W, dev_oImage_vu,  W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage_y);  
 *     free(dev_oImage_vu);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2NV12(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outYStride,
    T* outY,
    int outUVStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV122RGBA<uint8_t, 1, 4>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122RGBA(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert NV12 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image's Y stride, usually it equals to `width`
 * @param inY               input image Y data
 * @param inUVStride        input image's UV stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     uint8_t* dev_iImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV122RGBA<uint8_t, 1, 4>(H, W, W, dev_iImage_y, W, dev_iImage_uv, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage_y);
 *     free(dev_iImage_uv);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV122RGBA(
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUVStride,
    const T* inUV,
    int outWidthStride,
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
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     
 *     ppl::cv::arm::RGB2NV21<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2NV21(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outYStride        the width stride of output Y image, usually it equals to `width`
 * @param outY              output Y data
 * @param outUVStride       the width stride of output UV image, usually it equals to `width`
 * @param outUV             output UV data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>3<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 3;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     ppl::cv::arm::RGB2NV21<uint8_t, 3, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage_y, W, dev_oImage_uv);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGB2NV21(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outYStride,
    T* outY,
    int outUVStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV212RGB<uint8_t, 1, 3>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212RGB(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert NV21 images to RGB images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 3 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image's Y stride, usually it equals to `width`
 * @param inY               input image Y data
 * @param inUVStride        input image's UV stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>3
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 3;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_iImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV212RGB<uint8_t, 1, 3>(H, W, W, dev_iImage_y, W, dev_iImage_uv, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212RGB(
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUVStride,
    const T* inUV,
    int outWidthStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *
 *     ppl::cv::arm::RGBA2NV21<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2NV21(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
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
 * @param outYStride        the width stride of output Y image, usually it equals to `width`
 * @param outY              output Y data
 * @param outUVStride       the width stride of output UV image, usually it equals to `width`
 * @param outUV             output UV data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>4<td>1
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 4;
 *     const int output_channels = 1;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_y = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     uint8_t* dev_oImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     ppl::cv::arm::RGBA2NV21<uint8_t, 4, 1>(H, W, W * input_channels, dev_iImage, W, dev_oImage_y, W, dev_oImage_uv);
 *
 *     free(dev_iImage);
 *     free(dev_oImage_y);
 *     free(dev_oImage_uv);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode RGBA2NV21(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outYStride,
    T* outY,
    int outUVStride,
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
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t) * 3 / 2);
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV212RGBA<uint8_t, 1, 4>(H, W, W * input_channels, dev_iImage, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212RGBA(
    int height,
    int width,
    int inWidthStride,
    const T* inData,
    int outWidthStride,
    T* outData);
/**
 * @brief Convert NV21 images to RGBA images
 * @tparam T The data type, used for both input image and output image, currently only \a uint8_t is supported.
 * @tparam ncSrc The number of channels of input image, 1 is supported.
 * @tparam ncDst The number of channels of output image, 4 is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inYStride         input image's Y stride, usually it equals to `width`
 * @param inY               input image Y data
 * @param inUVStride        input image's UV stride, usually it equals to `width`
 * @param inUV              input image UV data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>ncSrc<th>ncDst
 * <tr><td>uint8_t(uchar)<td>1<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td> ARM platforms supported<td> armv7 armv8
 * <tr><td> Header files<td> #include &lt;ppl/cv/arm/cvtcolor.h&gt;
 * <tr><td> Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/arm/cvtcolor.h>
 * #include <stdlib.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int input_channels = 1;
 *     const int output_channels = 4;
 *     uint8_t* dev_iImage_y = (uint8_t*)malloc(W * H * input_channels * sizeof(uint8_t));
 *     uint8_t* dev_iImage_uv = (uint8_t*)malloc(W * H / 2 * sizeof(uint8_t));
 *     uint8_t* dev_oImage = (uint8_t*)malloc(W * H * output_channels * sizeof(uint8_t));
 *     
 *     ppl::cv::arm::NV212RGBA<uint8_t, 1, 4>(H, W, W, dev_iImage_y, W, dev_iImage_uv, W * output_channels, dev_oImage);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ****************************************************************************************************/
template <typename T>
::ppl::common::RetCode NV212RGBA(
    int height,
    int width,
    int inYStride,
    const T* inY,
    int inUVStride,
    const T* inUV,
    int outWidthStride,
    T* outData);

}
}
} // namespace ppl::cv::arm

#endif //! __ST_HPC_PPL_CV_AARCH64_GET_ROTATION_MATRIX2D_H_