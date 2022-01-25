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

#ifndef __ST_HPC_PPL_CV_X86_REMAP_H_
#define __ST_HPC_PPL_CV_X86_REMAP_H_
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Remap of coordinate map with linear interpolation method.
* @tparam T The data type of input image, currently only \a uint8_t and \a float is supported.
* @tparam channels The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param inHeight          input image's height
* @param inWidth           input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outHeight         output image's height
* @param outWidth          output image's width need to be processed
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param mapX              transformation matrix in the x direction
* @param mapY              transformation matrix in the y direction
* @param border_type       ways to deal with border. BORDER_CONSTANT, BORDER_REPLICATE and BORDER_TRANSPARENT are supported now.
* @param border_value      border value for BORDER_CONSTANT 
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
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/remap.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/remap.h>
* int main(int argc, char** argv) {
*     const int inWidth = 640;
*     const int inHeight = 480;
*     const int C = 3;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* dev_iImage = (unsigned char*)malloc(inWidth * inHeight * C * sizeof(unsigned char));
*     unsigned char* dev_oImage = (unsigned char*)malloc(inWidth * inHeight * C * sizeof(unsigned char));
*     float* mapX= (float*)malloc(outWidth * outHeight * sizeof(float));
*     float* mapY= (float*)malloc(outWidth * outHeight * sizeof(float));
*
*     ppl::cv::x86::RemapLinear<unsigend char, 3>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, mapX, mapY, ppl::cv::BORDER_CONSTANT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     free(mapX);
*     free(mapY);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T, int channels>
::ppl::common::RetCode RemapLinear(
    int inHeight,
    int inWidth,
    int inWidthStride,
    const T* inData,
    int outHeight,
    int outWidth,
    int outWidthStride,
    T* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type = ppl::cv::BORDER_CONSTANT,
    T borderValue          = 0);

/**
* @brief Remap of coordinate map with linear interpolation method.
* @tparam T The data type of input image, currently only \a uint8_t and \a float is supported.
* @param channels          The number of channels of input image and output image.
* @param inHeight          input image's height
* @param inWidth           input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outHeight         output image's height
* @param outWidth          output image's width need to be processed
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param mapX              transformation matrix in the x direction
* @param mapY              transformation matrix in the y direction
* @param border_type       ways to deal with border. BORDER_CONSTANT, BORDER_REPLICATE and BORDER_TRANSPARENT are supported now.
* @param border_value      border value for BORDER_CONSTANT 
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* <tr><td>float
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> all 
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/remap.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.7.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/remap.h>
* int main(int argc, char** argv) {
*     const int inWidth = 640;
*     const int inHeight = 480;
*     const int C = 3;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* dev_iImage = (unsigned char*)malloc(inWidth * inHeight * C * sizeof(unsigned char));
*     unsigned char* dev_oImage = (unsigned char*)malloc(inWidth * inHeight * C * sizeof(unsigned char));
*     float* mapX= (float*)malloc(outWidth * outHeight * sizeof(float));
*     float* mapY= (float*)malloc(outWidth * outHeight * sizeof(float));
*
*     ppl::cv::x86::RemapLinear<unsigend char>(C, inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, mapX, mapY, ppl::cv::BORDER_CONSTANT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     free(mapX);
*     free(mapY);
*     return 0;
* }
* @endcode
***************************************************************************************************/

template <typename T>
::ppl::common::RetCode RemapLinear(
    int channels,
    int inHeight,
    int inWidth,
    int inWidthStride,
    const T* inData,
    int outHeight,
    int outWidth,
    int outWidthStride,
    T* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type = ppl::cv::BORDER_CONSTANT,
    T borderValue          = 0);

/**
* @brief Remap of coordinate map with nearest interpolation method.
* @tparam T The data type of input image, currently only \a uint8_t and \a float is supported.
* @tparam channels The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param inHeight          input image's height
* @param inWidth           input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outHeight         output image's height
* @param outWidth          output image's width need to be processed
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param mapX              transformation matrix in the x direction
* @param mapY              transformation matrix in the y direction
* @param border_type       ways to deal with border. BORDER_CONSTANT, BORDER_REPLICATE and BORDER_TRANSPARENT are supported now.
* @param border_value      border value for BORDER_CONSTANT 
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
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/remap.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/remap.h>
* int main(int argc, char** argv) {
*     const int inWidth = 640;
*     const int inHeight = 480;
*     const int C = 3;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* dev_iImage = (unsigned char*)malloc(inWidth * inHeight * C * sizeof(unsigned char));
*     unsigned char* dev_oImage = (unsigned char*)malloc(inWidth * inHeight * C * sizeof(unsigned char));
*     float* mapX= (float*)malloc(outWidth * outHeight * C * sizeof(float));
*     float* mapY= (float*)malloc(outWidth * outHeight * C * sizeof(float));
*
*     ppl::cv::x86::RemapNearestPoint<unsigend char, 3>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, mapX, mapY, ppl::cv::BORDER_CONSTANT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     free(mapX);
*     free(mapY);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T, int channels>
::ppl::common::RetCode RemapNearestPoint(
    int inHeight,
    int inWidth,
    int inWidthStride,
    const T* inData,
    int outHeight,
    int outWidth,
    int outWidthStride,
    T* outData,
    const float* mapx,
    const float* mapy,
    BorderType border_type = ppl::cv::BORDER_CONSTANT,
    T borderValue          = 0);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL3_CV_X86_REMAP_H_
