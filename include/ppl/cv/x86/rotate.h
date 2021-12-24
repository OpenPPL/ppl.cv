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

#ifndef __ST_HPC_PPL_CV_X86_ROTATE_H_
#define __ST_HPC_PPL_CV_X86_ROTATE_H_

#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"
namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief image rotation.
* @tparam T The data type of input image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image and output image, 1, 2, 3 and 4 are supported.
* @param inHeight          input image's height
* @param inWidth           input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outHeight         output image's height
* @param outWidth          output image's width need to be processed
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param degree            Rotation angle, 90, 180 and 270 are supported.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)<th>channels
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
* <tr><td>X86 platforms supported<td> x86
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/rotate.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/rotate.h>
* int main(int argc, char** argv) {
*     const int inHeight = 640;
*     const int inWidth = 480;
*     const int C = 3;
*     const int outHeight = 640;
*     const int outWidth = 480;
*     const int degree = 90;
*     float* dev_iImage = (float*)malloc(W * H * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(W * H * C * sizeof(float));
*
*     ppl::cv::x86::RotateNx90degree<float, 3>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, degree);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T, int numChannels>
::ppl::common::RetCode RotateNx90degree(int inHeight,
                                        int inWidth,
                                        int inWidthStride,
                                        const T* inData,
                                        int outHeight,
                                        int outWidth,
                                        int outWidthStride,
                                        T* outData,
                                        int degree);

/**
* @brief image rotation with NV12 Image
* @tparam T The data type of input image, currently only \a uint8_t is supported.
* @param inHeight          input Y height 
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `width`
* @param inY               input Y data
* @param inUVStride        input UV width stride,usually it equals to `width`
* @param inUV              input UV data
* @param outHeight         output Y height
* @param outWidth          output Y width
* @param outYStride        output Y width stride, usually it equals to `width`
* @param outY              output Y data
* @param outUVStride       output UV width stride, usually it equals to `width`
* @param outUV             output UV data
* @param degree            Rotation angle, 90, 180 and 270 are supported.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type is supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/rotate.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/rotate.h>
* int main(int argc, char** argv) {
*     const int inHeight = 640;
*     const int inWidth = 480;
*     const int outHeight = 480;
*     const int outWidth = 640;
*     const int degree = 90;
*     uchar* inY  = (uchar*)malloc(inWidth * inHeight * sizeof(uchar));
*     uchar* inUV = (uchar*)malloc(inWidth * inHeight / 2 * sizeof(uchar));
*     uchar* outY = (uchar*)malloc(outWidth * outHeight * sizeof(uchar));
*     uchar* outUV = (uchar*)malloc(outWidth * outHeight / 2 * sizeof(uchar));
*
*     ppl::cv::x86::RotateNx90degree_NV12<uchar>(
*        inHeight, inWidth, inWidth, inY, inWidth, inUV,
*        outHeight, outWidth, outWidth, outY, outWidth, outUV, degree);
*
*     free(inY);
*     free(inUV);
*     free(outY);
*     free(outUV);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode RotateNx90degree_NV12(int inHeight,
                                             int inWidth,
                                             int inYStride,
                                             const T* inY,
                                             int inUVStride,
                                             const T* inUV,
                                             int outHeight,
                                             int outWidth,
                                             int outYStride,
                                             T* outY,
                                             int outUVStride,
                                             T* outUV,
                                             int degree);

/**
* @brief image rotation with NV21 Image
* @tparam T The data type of input image, currently only \a uint8_t is supported.
* @param inHeight          input Y height 
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `width`
* @param inY               input Y data
* @param inVUStride        input VU width stride,usually it equals to `width`
* @param inVU              input VU data
* @param outHeight         output Y height
* @param outWidth          output Y width
* @param outYStride        output Y width stride, usually it equals to `width`
* @param outY              output Y data
* @param outVUStride       output VU width stride, usually it equals to `width`
* @param outVU             output VU data
* @param degree            Rotation angle, 90, 180 and 270 are supported.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type is supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/rotate.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/rotate.h>
* int main(int argc, char** argv) {
*     const int inHeight = 640;
*     const int inWidth = 480;
*     const int outHeight = 480;
*     const int outWidth = 640;
*     const int degree = 90;
*     uchar* inY  = (uchar*)malloc(inWidth * inHeight * sizeof(uchar));
*     uchar* inVU = (uchar*)malloc(inWidth * inHeight / 2 * sizeof(uchar));
*     uchar* outY = (uchar*)malloc(outWidth * outHeight * sizeof(uchar));
*     uchar* outVU = (uchar*)malloc(outWidth * outHeight / 2 * sizeof(uchar));
*
*     ppl::cv::x86::RotateNx90degree_NV21<uchar>(
*        inHeight, inWidth, inWidth, inY, inWidth, inVU,
*        outHeight, outWidth, outWidth, outY, outWidth, outVU, degree);
*
*     free(inY);
*     free(inVU);
*     free(outY);
*     free(outVU);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode RotateNx90degree_NV21(int inHeight,
                                             int inWidth,
                                             int inYStride,
                                             const T* inY,
                                             int inVUStride,
                                             const T* inVU,
                                             int outHeight,
                                             int outWidth,
                                             int outYStride,
                                             T* outY,
                                             int outVUStride,
                                             T* outVU,
                                             int degree);

/**
* @brief image rotation with I420 Image
* @tparam T The data type of input image, currently only \a uint8_t is supported.
* @param inHeight          input Y height 
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `width`
* @param inY               input Y data
* @param inUStride         input U width stride,usually it equals to `width / 2`
* @param inU               input U data
* @param inVStride         input V width stride,usually it equals to `width / 2`
* @param inV               input V data
* @param outHeight         output Y height
* @param outWidth          output Y width
* @param outYStride        output Y width stride, usually it equals to `width`
* @param outY              output Y data
* @param outUStride        output U width stride, usually it equals to `width / 2`
* @param outU              output U data
* @param outVStride        output V width stride, usually it equals to `width / 2`
* @param outV              output V data
* @param degree            Rotation angle, 90, 180 and 270 are supported.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type is supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/rotate.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/rotate.h>
* int main(int argc, char** argv) {
*     const int inHeight = 640;
*     const int inWidth = 480;
*     const int outHeight = 480;
*     const int outWidth = 640;
*     const int degree = 90;
*     uchar* inY  = (uchar*)malloc(inWidth * inHeight * sizeof(uchar));
*     uchar* inU = (uchar*)malloc(inWidth / 2 * inHeight / 2 * sizeof(uchar));
*     uchar* inV = (uchar*)malloc(inWidth / 2 * inHeight / 2 * sizeof(uchar));
*     uchar* outY = (uchar*)malloc(outWidth * outHeight * sizeof(uchar));
*     uchar* outU = (uchar*)malloc(outWidth / 2 * outHeight / 2 * sizeof(uchar));
*     uchar* outV = (uchar*)malloc(outWidth / 2 * outHeight / 2 * sizeof(uchar));
*
*     ppl::cv::x86::RotateNx90degree_I420<uchar>(
*        inHeight, inWidth, inWidth, inY, inWidth / 2, inU, inWidth / 2, inV,
*        outHeight, outWidth, outWidth, outY, outWidth / 2, outU, outWidth / 2, outV, degree);
*
*     free(inY);
*     free(inU);
*     free(inV)
*     free(outY);
*     free(outU);
*     free(outV);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode RotateNx90degree_I420(int inHeight,
                                             int inWidth,
                                             int inYStride,
                                             const T* inY,
                                             int inUStride,
                                             const T* inU,
                                             int inVStride,
                                             const T* inV,
                                             int outHeight,
                                             int outWidth,
                                             int outYStride,
                                             T* outY,
                                             int outUStride,
                                             T* outU,
                                             int outVStride,
                                             T* outV,
                                             int degree);

}
}
} // namespace ppl::cv::x86

#endif //!__ST_HPC_PPL_CV_X86_ROTATE_H_
