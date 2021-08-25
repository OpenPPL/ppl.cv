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

#ifndef __ST_HPC_PPL_CV_X86_RESIZE_H_
#define __ST_HPC_PPL_CV_X86_RESIZE_H_

#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Resize the image with linear interpolation method.
* @tparam TSrc The data type of input image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image, 1, 3 and 4 are supported.
* @tparam TDst The data type of output image, currently only \a uint8_t and \a float are supported.The param is same with TSrc.
* @param inHeight          input image's height
* @param inWidth           input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outHeight         output image's height
* @param outWidth          output image's width need to be processed
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
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
* <tr><td>X86 platforms supported<td> &gt; all
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/resize.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/resize.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t inWidth = 320;
*     const int32_t inHeight = 240;
*     const int32_t outWidth = 640;
*     const int32_t outHeight = 480;
*     const int32_t C = 3;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::ResizeLinear<float, 3>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage);
*
*     free(dev_iImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template<typename T, int32_t channels>
::ppl::common::RetCode ResizeLinear(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData);

/**
* @brief Resize the image with nearest neighbor interpolation method
* @tparam T The data type of input and output image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image, 1, 3 and 4 are supported.
* @param inHeight          input image's height
* @param inWidth           input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData           input image data
* @param outHeight         output image's height
* @param outWidth          output image's width need to be processed
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
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
* <tr><td>X86 platforms supported<td> all
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/resize.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/resize.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t inWidth = 320;
*     const int32_t inHeight = 240;
*     const int32_t outWidth = 640;
*     const int32_t outHeight = 480;
*     const int32_t C = 3;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::ResizeNearestPoint<float, 3>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template<typename T, int32_t channels>
::ppl::common::RetCode ResizeNearestPoint(
    int32_t inHeight,
    int32_t inWidth,
    int32_t inWidthStride,
    const T* inData,
    int32_t outHeight,
    int32_t outWidth,
    int32_t outWidthStride,
    T* outData);


} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL3_CV_X86_RESIZE_H_
