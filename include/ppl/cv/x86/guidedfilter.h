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

#ifndef __ST_HPC_PPL_CV_X86_GUIDEDFILTER_H_
#define __ST_HPC_PPL_CV_X86_GUIDEDFILTER_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Guided filter
* @tparam T The data type of input image, currently \a float and \a uint8_t is supported.
* @tparam srcChannels The number of channels of input image, 1 or 3 is supported.
* @tparam guidedChannels The number of channels of guide image. when srcChannels == 1, guideChannels can be 1 or 3. when srcChannels == 3, guideChannels can be 3.
* @param height               input/guide/ouput image's height
* @param width                input/guide/ouput image's width
* @param inWidthStride        input image's width stride, usually it equals to `width * channels`
* @param src                  input image data
* @param guidedWidthStride    guided image's width stride, usually it equals to `width * channels`
* @param guided               guided image data
* @param dstWidthStride       the width stride of output image, usually it equals to `width * channels`
* @param dst                  output image data
* @param radius               filter window radius
* @param eps                  Regularization parameter (fuzzy coefficient), if input is in range of [0, 255], eps should be multiplied by 255
* @param border_type       ways to deal with border. Only BORDER_REFLECT, BORDER_REFLECT_101 or BORDER_DEFAULT are supported now.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
* <tr><td>float<td>3
* <tr><td>float<td>1
* <tr><td>uint8_t<td>3
* <tr><td>uint8_t<td>1
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/guidedfilter.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/guidedfilter.h>
* int32_t main(int32_t argc, char** argv) {
*     const int32_t W = 640;
*     const int32_t H = 480;
*     const int32_t C = 3;
*     uint8_t* iImage = (uint8_t*)malloc(W * H * C * sizeof(uint8_t));
*     uint8_t* guidedImage = (uint8_t*)malloc(W * H * C * sizeof(uint8_t));
*     uint8_t* oImage = (uint8_t*)malloc(W * H * C * sizeof(uint8_t));
*     int32_t r = 8;
*     uint8_t eps = 0.4 * 0.4 * 255 * 255;
*
*     ppl::cv::x86::GuidedFilter<uint8_t, 3, 3>(H, W, W * C, iImage, H, W, W * C,guidedImage, W * C, oImage, r, eps, ppl::cv::BORDER_DEFAULT);
*
*     free(iImage);
*     free(guidedImage);
*     free(oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T, int32_t srcChannels, int32_t guidedChannels>
::ppl::common::RetCode GuidedFilter(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const T* src,
    int32_t guidedWidthStride,
    const T* guided,
    int32_t dstWidthStride,
    T* dst,
    int32_t radius,
    float eps,
    BorderType border_type);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_GUIDEDFILTER_H_
