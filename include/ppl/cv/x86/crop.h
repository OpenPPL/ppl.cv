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

#ifndef __ST_HPC_PPL_CV_X86_CROP_H_
#define __ST_HPC_PPL_CV_X86_CROP_H_

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief The target area is truncated in the source image and scaled.
* @tparam T The data type of input and output image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image and output image, 1, 2, 3 and 4 are supported.
* @param inHeight          input image's height
* @param inWidth           input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outHeight         output image's height
* @param outWidth          output image's width need to be processed
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param left              The left value of the coordinate point of the upper left corner of the target area
* @param top               The top value of the coordinate point of the upper left corner of the target area
* @param ratio             Scale scaling factor
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels<
* <tr><td>uint8_t(uchar)<td>1
* <tr><td>uint8_t(uchar)<td>2
* <tr><td>uint8_t(uchar)<td>3
* <tr><td>uint8_t(uchar)<td>4
* <tr><td>float<td>1
* <tr><td>float<td>2
* <tr><td>float<td>3
* <tr><td>float<td>4
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>x86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/crop.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/crop.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     const int32_t outHeight = 100;
*     const int32_t outWidth = 100;
*     const left = 10;
*     const top = 10;
*     const float ratio = 1.0f;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(outWidth * outHeight * C * sizeof(float));
*
*     ppl::cv::x86::Crop<float, 3>(H, W, W * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, left, top, ratio);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t channels>
::ppl::common::RetCode Crop(
    const int32_t inHeight,
    const int32_t inWidth,
    const int32_t inWidthStride,
    const T* inData,
    const int32_t outHeight,
    const int32_t outWidth,
    const int32_t outWidthStride,
    T* outData,
    const int32_t left,
    const int32_t top,
    const float ratio);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_CROP_H_
