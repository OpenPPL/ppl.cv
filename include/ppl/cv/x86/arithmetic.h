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

#ifndef __ST_HPC_PPL_CV_X86_ARITHMETIC_H_
#define __ST_HPC_PPL_CV_X86_ARITHMETIC_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Calculates the per-element sum of two arrays
* @tparam T The data type of input and output image, currently \a float and \a uint8_t are supported.
* @tparam nc The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride0        first input image's width stride, usually it equals to `width * nc`
* @param inData0          first input image data
* @param inWidthStride1        second input image's width stride, usually it equals to `width * nc`
* @param inData1          second input image data
* @param outWidthStride        output image's width stride, usually it equals to `width * nc`
* @param outData          output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/arithmetic.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/arithmetic.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage0 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_iImage1 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::Add<float, 3>(H, W, W * C, dev_iImage0, W * C, dev_iImage1, W * C, dev_oImage);
*
*     free(dev_iImage0);
*     free(dev_iImage1);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t channels>
::ppl::common::RetCode Add(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const T *inData0,
    int32_t inWidthStride1,
    const T *inData1,
    int32_t outWidthStride,
    T *outData);

/**
* @brief Calculates the per-element multiply of two arrays
* @tparam T The data type of input and output image, currently \a float and \a uint8_t are supported.
* @tparam channels The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride0    first input image's width stride, usually it equals to `width * channels`
* @param inData0           first input image data
* @param inWidthStride1    second input image's width stride, usually it equals to `width * channels`
* @param inData1           second input image data
* @param outWidthStride    output image's width stride, usually it equals to `width * channels`
* @param outData           output image data
* @param scale             optional scale factor.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/arithmetic.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/arithmetic.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage0 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_iImage1 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::Mul<float, 3>(H, W, W * C, dev_iImage0, W * C, dev_iImage1, W * C, dev_oImage);
*
*     free(dev_iImage0);
*     free(dev_iImage1);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t channels>
::ppl::common::RetCode Mul(int32_t height,
         int32_t width,
         int32_t inWidthStride0,
         const T *inData0,
         int32_t inWidthStride1,
         const T *inData1,
         int32_t outWidthStride,
         T *outData,
         float scale = 1.0f);


/**
* @brief Calculates the element-wise multiplication-addition of three vectors. Formula : out += a * b.
* @tparam T The data type of input and output image, currently only \a float is supported.
* @tparam channels The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride0    first input image's width stride, usually it equals to `width * channels`
* @param inData0           first input image data
* @param inWidthStride1    second input image's width stride, usually it equals to `width * channels`
* @param inData1           second input image data
* @param outWidthStride    output image's width stride, usually it equals to `width * channels`
* @param outData           output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/arithmetic.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/arithmetic.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage0 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_iImage1 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::Mla<float, 3>(H, W, W * C, dev_iImage0, W * C, dev_iImage1, W * C, dev_oImage);
*
*     free(dev_iImage0);
*     free(dev_iImage1);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t channels>
::ppl::common::RetCode Mla(int32_t height,
         int32_t width,
         int32_t inWidthStride0,
         const T *inData0,
         int32_t inWidthStride1,
         const T *inData1,
         int32_t outWidthStride,
         T *outData);

/**
* @brief Calculates the element-wise multiplication-subtraction of three vectors. Formula : out -= a * b.
* @tparam T The data type of input and output image, currently only \a float is supported.
* @tparam channels The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride0    first input image's width stride, usually it equals to `width * channels`
* @param inData0           first input image data
* @param inWidthStride1    second input image's width stride, usually it equals to `width * channels`
* @param inData1           second input image data
* @param outWidthStride    output image's width stride, usually it equals to `width * channels`
* @param outData           output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/arithmetic.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/arithmetic.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage0 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_iImage1 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::Mls<float, 3>(H, W, W * C, dev_iImage0, W * C, dev_iImage1, W * C, dev_oImage);
*
*     free(dev_iImage0);
*     free(dev_iImage1);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t channels>
::ppl::common::RetCode Mls(int32_t height,
         int32_t width,
         int32_t inWidthStride0,
         const T *inData0,
         int32_t inWidthStride1,
         const T *inData1,
         int32_t outWidthStride,
         T *outData);

/**
* @brief Performs per-element division of two arrays
* @tparam T The data type of input and output image, currently only \a float is supported.
* @tparam nc The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride0    first input image's width stride, usually it equals to `width * nc`
* @param inData0           first input image data
* @param inWidthStride1    second input image's width stride, usually it equals to `width * nc`
* @param inData1           second input image data
* @param outWidthStride    output image's width stride, usually it equals to `width * nc`
* @param outData           output image data
* @param scale             scalar factor
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/arithmetic.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/arithmetic.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage0 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_iImage1 = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::Div<float, 3>(H, W, W * C, dev_iImage0, W * C, dev_iImage1, W * C, dev_oImage);
*
*     free(dev_iImage0);
*     free(dev_iImage1);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T, int32_t channels>
::ppl::common::RetCode Div(int32_t height,
         int32_t width,
         int32_t inWidthStride0,
         const T *inData0,
         int32_t inWidthStride1,
         const T *inData1,
         int32_t outWidthStride,
         T *outData,
         float scale = 1.0f);

/**
* @brief Calculates the per-element difference between array and a scalar.
* @tparam T The data type of input and output image, currently \a float and \a uint8_t are supported.
* @tparam nc The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride         input image's width stride, usually it equals to `width * nc`
* @param inData            input image data
* @param scalar            input scalar data,  usually its numbel equals to `nc`
* @param outWidthStride        output image's width stride, usually it equals to `width * nc`
* @param outData           output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>uint8_t(uint8_t)<td>1
* <tr><td>uint8_t(uint8_t)<td>3
* <tr><td>uint8_t(uint8_t)<td>4
* <tr><td>float<td>1
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/arithmetic.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/arithmetic.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_iScalar= (float*)malloc(C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::Subtract<T, c>(height, width, stride, dev_iImage, dev_iScalar, stride, dev_oImage);
*
*     free(dev_iImage);
*     free(dev_iScalar);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T, int32_t channels>
::ppl::common::RetCode Subtract(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T *inData,
    const T *scalar,
    int32_t outWidthStride,
    T *outData);


} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //!__ST_HPC_PPL_CV_X86_ARITHMETIC_H__

