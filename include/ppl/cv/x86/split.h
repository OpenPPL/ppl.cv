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

#ifndef __ST_HPC_PPL_CV_X86_SPLIT_H_
#define __ST_HPC_PPL_CV_X86_SPLIT_H_

#include "ppl/common/retcode.h"
namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Creates 3 single-channel array from a 3-channel array.
* @tparam T The data type of input and output image, currently only \a uint8_t and \a float are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride     input image's width stride, usually it equals to `width * 3`
* @param inData            input image data
* @param outWidthStride    output image's width stride, usually it equals to `width`
* @param outData0          first output image data
* @param outData1          second output image data
* @param outData2          third output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>float
* <tr><td>uint8_t
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> all
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/split.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/split.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage0 = (float*)malloc(W * H * sizeof(float));
*     float* dev_oImage1 = (float*)malloc(W * H * sizeof(float));
*     float* dev_oImage2 = (float*)malloc(W * H * sizeof(float));
*
*     ppl::cv::x86::Split3Channels<float>(H, W, W * C, dev_iImage, W, dev_oImage0, dev_oImage1, dev_oImage2);
*
*     free(dev_iImage);
*     free(dev_oImage0);
*     free(dev_oImage1);
*     free(dev_oImage2);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T>
::ppl::common::RetCode Split3Channels(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outDataChannel0,
    T* outDataChannel1,
    T* outDataChannel2);

/**
* @brief Creates 4 single-channel array from a 4-channel array.
* @tparam T The data type of input and output image, currently only \a uint8_t and \a float are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride     input image's width stride, usually it equals to `width * 4`
* @param inData            input image data
* @param outWidthStride    output image's width stride, usually it equals to `width`
* @param outData0          first output image data
* @param outData1          second output image data
* @param outData2          third output image data
* @param outData2          fourth output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>float
* <tr><td>uint8_t
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> all
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/split.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/split.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 4;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage0 = (float*)malloc(W * H * sizeof(float));
*     float* dev_oImage1 = (float*)malloc(W * H * sizeof(float));
*     float* dev_oImage2 = (float*)malloc(W * H * sizeof(float));
*     float* dev_oImage3 = (float*)malloc(W * H * sizeof(float));
*
*     ppl::cv::x86::Split4Channels<float>(H, W, W * C, dev_iImage, W, dev_oImage0, dev_oImage1, dev_oImage2, dev_oImage3);
*
*     free(dev_iImage);
*     free(dev_oImage0);
*     free(dev_oImage1);
*     free(dev_oImage2);
*     free(dev_oImage3);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T>
::ppl::common::RetCode Split4Channels(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inData,
    int32_t outWidthStride,
    T* outDataChannel0,
    T* outDataChannel1,
    T* outDataChannel2,
    T* outDataChannel3);

}
}
} // namespace ppl::cv::x86

#endif //!__ST_HPC_PPL_CV_X86_SPLIT_H_
