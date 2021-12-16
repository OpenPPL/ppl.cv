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

#ifndef __ST_HPC_PPL_CV_X86_MERGE_H_
#define __ST_HPC_PPL_CV_X86_MERGE_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Creates one 3 channel array out of 3 single-channel ones.
* @tparam T The data type of input and output image, currently only \a uint8(uchar) and \a float are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride0    first input image's width stride, usually it equals to `width`
* @param inData0           first input image data
* @param inWidthStride1    second input image's width stride, usually it equals to `width`
* @param inData1           second input image data
* @param inWidthStride2    third input image's width stride, usually it equals to `width`
* @param inData2           third input image data
* @param outWidthStride    output image's width stride, usually it equals to `width * 3`
* @param outData           output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>float
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> all 
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/merge.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/merge.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     float* dev_iImage0 = (float*)malloc(W * H * sizeof(float));
*     float* dev_iImage1 = (float*)malloc(W * H * sizeof(float));
*     float* dev_iImage2 = (float*)malloc(W * H * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::Merge3Channels<float>(H, W, W, dev_iImage0, dev_iImage1, dev_iImage2, W * C, dev_oImage);
*
*     free(dev_iImage0);
*     free(dev_iImage1);
*     free(dev_iImage2);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T>
::ppl::common::RetCode Merge3Channels(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inDataC0,
    const T* inDataC1,
    const T* inDataC2,
    int32_t outWidthStride,
    T* outData);

/**
* @brief Creates one 4 channel array out of 4 single-channel ones.
* @tparam T The data type of input and output image, currently only \a uint8(uchar) and \a float are supported.
* @param height            input image's height
* @param width             input image's width
* @param inWidthStride0    first input image's width stride, usually it equals to `width`
* @param inData0           first input image data
* @param inWidthStride1    second input image's width stride, usually it equals to `width`
* @param inData1           second input image data
* @param inWidthStride2    third input image's width stride, usually it equals to `width`
* @param inData2           third input image data
* @param inWidthStride3    fourth input image's width stride, usually it equals to `width`
* @param inData3           fourth input image data
* @param outWidthStride    output image's width stride, usually it equals to `width * 4`
* @param outData           output image data
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>float
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/merge.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/merge.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 4;
*     float* dev_iImage0 = (float*)malloc(W * H * sizeof(float));
*     float* dev_iImage1 = (float*)malloc(W * H * sizeof(float));
*     float* dev_iImage2 = (float*)malloc(W * H * sizeof(float));
*     float* dev_iImage3 = (float*)malloc(W * H * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::Merge4Channels<float>(H, W, W, dev_iImage0, dev_iImage1, dev_iImage2, dev_iImage3, W * C, dev_oImage);
*
*     free(dev_iImage0);
*     free(dev_iImage1);
*     free(dev_iImage2);
*     free(dev_iImage3);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T>
::ppl::common::RetCode Merge4Channels(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* inDataC0,
    const T* inDataC1,
    const T* inDataC2,
    const T* inDataC3,
    int32_t outWidthStride,
    T* outData);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_MERGE_H_
