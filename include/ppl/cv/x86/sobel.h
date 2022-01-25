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

#ifndef __ST_HPC_PPL_CV_X86_SOBEL_H_
#define __ST_HPC_PPL_CV_X86_SOBEL_H_

#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"
namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
 * @tparam Tsrc The data type of input image, currently only \a uint8_t and \a float are supported.
 * @tparam Tdst The data type of output image, currently when Tsrc == uint8_t, only \a int16_t is supported. And when Tsrc == float, only \a float is suppoted.
 * @tparam channels The number of channels of input image, 1, 3 and 4 are supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width * channels`
 * @param inData            input image data
 * @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
 * @param outData           output image data
 * @param dx                order of the derivative x
 * @param dy                order of the derivative y
 * @param ksize             the length of kernel. when ksize == -1, it will use 3x3 scharr kernel, (dx, dy) = (0, 1) or (1, 0). when ksize == 1, 3, 5, 7, dx + dy should > 0. other ksize is not supported.
 * @param scale             scale factor for the computed derivative values
 * @param delta             delta value that is added to the results prior to storing them
 * @param border_type       ways to deal with border. Only BORDER_REFLECT_101 or BORDER_DEFAULT are supported now.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The following table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(Tsrc)<th>Data type(Tdst)<th>channels
 * <tr><td>uint8_t(uchar)<td>int16_t<td>1
 * <tr><td>uint8_t(uchar)<td>int16_t<td>3
 * <tr><td>uint8_t(uchar)<td>int16_t<td>4
 * <tr><td>float<td>float<td>1
 * <tr><td>float<td>float<td>3
 * <tr><td>float<td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/sobel.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/sobel.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int C = 3;
 *     const int ksize = 3;
 *     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
 *     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
 *
 *     ppl::cv::x86::Sobel<float, float, 3>(H, W, W * C, dev_iImage, 1, 0, ksize, 1.0, 0.0, W * C, dev_oImage, ppl::cv::BORDER_DEFAULT);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/

template <typename Tsrc, typename Tdst, int nc>
::ppl::common::RetCode Sobel(
    int height,
    int width,
    int inWidthStride,
    const Tsrc* inData,
    int outWidthStride,
    Tdst* outData,
    int dx,
    int dy,
    int ksize,
    double scale,
    double delta,
    BorderType border_type = BORDER_DEFAULT);

}
}
} // namespace ppl::cv::x86

#endif //!__ST_HPC_PPL_CV_X86_SOBEL_H_
