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

#ifndef __ST_HPC_PPL_CV_ARRCH64_WARPAFFINE_H_
#define __ST_HPC_PPL_CV_ARRCH64_WARPAFFINE_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace aarch64 {

/**
* @brief Affine transformation with nearest neighbor interpolation method
* @tparam T The data type of input image and output image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param affineMatrix      the mask of warpaffine
* @param border_type       ways to deal with border. Use BORDER_TYPE_WARP as embedded type(immutable), optional type support BORDER_TYPE_CONSTANT now.
* @param borderValue       value used in case of a constant border; by default, it is 0
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
* <tr><td>ARM platforms supported<td> armv7 armv8
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/warpaffine.h>
* int main(int argc, char** argv) {
*     const int inWidth = 640;
*     const int inHeight = 480;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     const int C = 1;
*     float* dev_iImage = (float*)malloc(inWidth * inHeight * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(outWidth * outHeight * C * sizeof(float));
*     float* affineMatrix = (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::arm::WarpAffineNearestPoint<float, 4>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, affineMatrix, ppl::cv::BORDER_TYPE_CONSTANT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T, int channels>
::ppl::common::RetCode WarpAffineNearestPoint(
    int inHeight,
    int inWidth,
    int inWidthStride,
    const T* inData,
    int outHeight,
    int outWidth,
    int outWidthStride,
    T* outData,
    const float* affineMatrix,
    BorderType border_type = BORDER_TYPE_CONSTANT,
    T borderValue          = 0);

/**
* @brief Affine transformation with linear interpolation method
* @tparam T The data type of input image and output image, currently only \a uint8_t and \a float are supported.
* @tparam channels The number of channels of input image and output image, 1, 3 and 4 are supported.
* @param height            input image's height
* @param width             input image's width need to be processed
* @param inWidthStride     input image's width stride, usually it equals to `width * channels`
* @param inData            input image data
* @param outWidthStride    the width stride of output image, usually it equals to `width * channels`
* @param outData           output image data
* @param affineMatrix      the mask of warpaffine
* @param border_type       ways to deal with border. Use BORDER_TYPE_WARP as embedded type(immutable), optional type support BORDER_TYPE_CONSTANT and  BORDER_TYPE_TRANSPARENT now.
* @param borderValue       value used in case of a constant border; by default, it is 0
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
* <tr><td>ARM platforms supported<td> armv7 armv8
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/warpaffine.h>
* int main(int argc, char** argv) {
*     const int inWidth = 640;
*     const int inHeight = 480;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     const int C = 1;
*     float* dev_iImage = (float*)malloc(inWidth * inHeight * C * sizeof(float));
*     float* dev_oImage = (float*)malloc(outWidth * outHeight * C * sizeof(float));
*     float* affineMatrix = (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::arm::WarpAffineLinear<float, 4>(inHeight, inWidth, inWidth * C, dev_iImage, outHeight, outWidth, outWidth * C, dev_oImage, affineMatrix, ppl::cv::BORDER_TYPE_CONSTANT);
*
*     free(dev_iImage);
*     free(dev_oImage);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T, int channels>
::ppl::common::RetCode WarpAffineLinear(
    int inHeight,
    int inWidth,
    int inWidthStride,
    const T* inData,
    int outHeight,
    int outWidth,
    int outWidthStride,
    T* outData,
    const float* affineMatrix,
    BorderType border_type = BORDER_TYPE_CONSTANT,
    T borderValue          = 0);

/**
* @brief Affine transformation with linear interpolation method for NV12 image
* @param inHeight          input Y height
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `inWidth`
* @param yInData           input Y data
* @param inUVStride        input uv width stride, usually it equals to `inWidth`
* @param uvInData          input uv data
* @param outHeight         output Y height
* @param outWidth          output Y width need to be processed
* @param outYStride        output Y width stride, usually it equals to `outWidth`
* @param yOutData          output Y data
* @param outUVStride       output uv width stride, usually it equals to `outWidth`
* @param uvOutData         output uv data
* @param affineMatrix      transformation matrix
* @param border_type       support ppl::cv::BORDER_TYPE_CONSTANT/ppl::cv::BORDER_TYPE_REPLICATE/ppl::cv::BORDER_TYPE_TRANSPARENT
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>arm platforms supported<td> armv7 armv8 
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/warpaffine.h>
* int main(int argc, char** argv) {
*     const int inWidth = 320;
*     const int inHeight = 240;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* inYImage = (unsigned char*)malloc(inWidth * inHeight * sizeof(unsigned char));
*     unsigned char* inUVImage = (unsigned char*)malloc(inWidth * (inHeight / 2) * sizeof(unsigned char));
*     unsigned char* outYImage = (unsigned char*)malloc(outWidth * outHeight * sizeof(unsigned char));
*     unsigned char* outUVImage = (unsigned char*)malloc(outWidth * (outHeight / 2) * sizeof(unsigned char));
*     float* affineMatrix= (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::arm::WarpAffineLinear_NV12<uchar>(inHeight, inWidth, inWidth, inYImage, 
*        inWidth, inUVImage, 
*        outHeight, outWidth, outWidth, outYImage, 
*        outWidth, outUVImage, 
*        affineMatrix, ppl::cv::BORDER_TYPE_CONSTANT);
*     free(inYImage);
*     free(inUVImage);
*     free(outYImage);
*     free(outUVImage);
*     free(affineMatrix);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode WarpAffineLinear_NV12(int inHeight,
                                             int inWidth,
                                             int inYStride,
                                             const T* yInData,
                                             int inUVStride,
                                             const T* uvInData,
                                             int outHeight,
                                             int outWidth,
                                             int outYStride,
                                             T* yOutData,
                                             int outUVStride,
                                             T* uvOutData,
                                             const float* affineMatrix,
                                             BorderType border_type = BORDER_TYPE_CONSTANT);

/**
* @brief Affine transformation with linear interpolation method for NV21 image
* @param inHeight          input Y height
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `inWidth`
* @param yInData           input Y data
* @param inVUStride        input vu width stride, usually it equals to `inWidth`
* @param vuInData          input vu data
* @param outHeight         output Y height
* @param outWidth          output Y width need to be processed
* @param outYStride        output Y width stride, usually it equals to `outWidth`
* @param yOutData          output Y data
* @param outVUStride       output vu width stride, usually it equals to `outWidth`
* @param vuOutData         output vu data
* @param affineMatrix      transformation matrix
* @param border_type       support ppl::cv::BORDER_TYPE_CONSTANT/ppl::cv::BORDER_TYPE_REPLICATE/ppl::cv::BORDER_TYPE_TRANSPARENT
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>arm platforms supported<td> armv7 armv8 
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/warpaffine.h>
* int main(int argc, char** argv) {
*     const int inWidth = 320;
*     const int inHeight = 240;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* inYImage = (unsigned char*)malloc(inWidth * inHeight * sizeof(unsigned char));
*     unsigned char* inVUImage = (unsigned char*)malloc(inWidth * (inHeight / 2) * sizeof(unsigned char));
*     unsigned char* outYImage = (unsigned char*)malloc(outWidth * outHeight * sizeof(unsigned char));
*     unsigned char* outVUImage = (unsigned char*)malloc(outWidth * (outHeight / 2) * sizeof(unsigned char));
*     float* affineMatrix= (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::arm::WarpAffineLinear_NV21<uchar>(inHeight, inWidth, inWidth, inYImage, 
*        inWidth, inVUImage, 
*        outHeight, outWidth, outWidth, outYImage, 
*        outWidth, outVUImage, 
*        affineMatrix, ppl::cv::BORDER_TYPE_CONSTANT);
*     free(inYImage);
*     free(inVUImage);
*     free(outYImage);
*     free(outVUImage);
*     free(affineMatrix);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode WarpAffineLinear_NV21(int inHeight,
                                             int inWidth,
                                             int inYStride,
                                             const T* yInData,
                                             int inVUStride,
                                             const T* vuInData,
                                             int outHeight,
                                             int outWidth,
                                             int outYStride,
                                             T* yOutData,
                                             int outVUStride,
                                             T* vuOutData,
                                             const float* affineMatrix,
                                             BorderType border_type = BORDER_TYPE_CONSTANT);

/**
* @brief Affine transformation with linear interpolation method for I420 image
* @param inHeight          input Y height
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `inWidth`
* @param yInData           input Y data
* @param inUStride         input u width stride, usually it equals to `inWidth / 2`
* @param uInData           input u data
* @param inVStride         input v width stride, usually it equals to `inWidth / 2`
* @param vInData           input v data
* @param outHeight         output Y height
* @param outWidth          output Y width need to be processed
* @param outYStride        output Y width stride, usually it equals to `outWidth`
* @param yOutData          output Y data
* @param outUStride        output u width stride, usually it equals to `outWidth / 2`
* @param uOutData          output u data
* @param outVStride        output v width stride, usually it equals to `outWidth / 2`
* @param vOutData          output v data
* @param affineMatrix      transformation matrix
* @param border_type       support ppl::cv::BORDER_TYPE_CONSTANT/ppl::cv::BORDER_TYPE_REPLICATE/ppl::cv::BORDER_TYPE_TRANSPARENT
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>arm platforms supported<td> armv7 armv8 
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/warpaffine.h>
* int main(int argc, char** argv) {
*     const int inWidth = 320;
*     const int inHeight = 240;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* inYImage = (unsigned char*)malloc(inWidth * inHeight * sizeof(unsigned char));
*     unsigned char* inUImage = (unsigned char*)malloc((inWidth / 2) * (inHeight / 2) * sizeof(unsigned char));
*     unsigned char* inVImage = (unsigned char*)malloc((inWidth / 2) * (inHeight / 2) * sizeof(unsigned char));
*     unsigned char* outYImage = (unsigned char*)malloc(outWidth * outHeight * sizeof(unsigned char));
*     unsigned char* outUImage = (unsigned char*)malloc((outWidth / 2) * (outHeight / 2) * sizeof(unsigned char));
*     unsigned char* outVImage = (unsigned char*)malloc((outWidth / 2) * (outHeight / 2) * sizeof(unsigned char));
*     float* affineMatrix= (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::arm::WarpAffineLinear_I420<uchar>(inHeight, inWidth, inWidth, inYImage, 
*        inWidth / 2, inUImage, inWidth / 2, inVImage, 
*        outHeight, outWidth, outWidth, outYImage, 
*        outWidth / 2, outUImage, outWidth / 2, outVImage, 
*        affineMatrix, ppl::cv::BORDER_TYPE_CONSTANT);
*     free(inYImage);
*     free(inUImage);
*     free(inVImage);
*     free(outYImage);
*     free(outUImage);
*     free(outVImage);
*     free(affineMatrix);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode WarpAffineLinear_I420(int inHeight,
                                             int inWidth,
                                             int inYStride,
                                             const T* yInData,
                                             int inUStride,
                                             const T* uInData,
                                             int inVStride,
                                             const T* vInData,
                                             int outHeight,
                                             int outWidth,
                                             int outYStride,
                                             T* yOutData,
                                             int outUStride,
                                             T* uOutData,
                                             int outVStride,
                                             T* vOutData,
                                             const float* affineMatrix,
                                             BorderType border_type = BORDER_TYPE_CONSTANT);

/**
* @brief Affine transformation with nearest neighbor interpolation method for NV12 image
* @param inHeight          input Y height
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `inWidth`
* @param yInData           input Y data
* @param inUVStride        input uv width stride, usually it equals to `inWidth`
* @param uvInData          input uv data
* @param outHeight         output Y height
* @param outWidth          output Y width need to be processed
* @param outYStride        output Y width stride, usually it equals to `outWidth`
* @param yOutData          output Y data
* @param outUVStride       output uv width stride, usually it equals to `outWidth`
* @param uvOutData         output uv data
* @param affineMatrix      transformation matrix
* @param border_type       support ppl::cv::BORDER_TYPE_CONSTANT/ppl::cv::BORDER_TYPE_REPLICATE/ppl::cv::BORDER_TYPE_TRANSPARENT
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>arm platforms supported<td> armv7 armv8 
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/warpaffine.h>
* int main(int argc, char** argv) {
*     const int inWidth = 320;
*     const int inHeight = 240;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* inYImage = (unsigned char*)malloc(inWidth * inHeight * sizeof(unsigned char));
*     unsigned char* inUVImage = (unsigned char*)malloc(inWidth * (inHeight / 2) * sizeof(unsigned char));
*     unsigned char* outYImage = (unsigned char*)malloc(outWidth * outHeight * sizeof(unsigned char));
*     unsigned char* outUVImage = (unsigned char*)malloc(outWidth * (outHeight / 2) * sizeof(unsigned char));
*     float* affineMatrix= (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::arm::WarpAffineNearestPoint_NV12<uchar>(inHeight, inWidth, inWidth, inYImage, 
*        inWidth, inUVImage, 
*        outHeight, outWidth, outWidth, outYImage, 
*        outWidth, outUVImage, 
*        affineMatrix, ppl::cv::BORDER_TYPE_CONSTANT);
*     free(inYImage);
*     free(inUVImage);
*     free(outYImage);
*     free(outUVImage);
*     free(affineMatrix);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode WarpAffineNearestPoint_NV12(int inHeight,
                                                   int inWidth,
                                                   int inYStride,
                                                   const T* yInData,
                                                   int inUVStride,
                                                   const T* uvInData,
                                                   int outHeight,
                                                   int outWidth,
                                                   int outYStride,
                                                   T* yOutData,
                                                   int outUVStride,
                                                   T* uvOutData,
                                                   const float* affineMatrix,
                                                   BorderType border_type = BORDER_TYPE_CONSTANT);

/**
* @brief Affine transformation with nearest neighbor interpolation method for NV21 image
* @param inHeight          input Y height
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `inWidth`
* @param yInData           input Y data
* @param inVUStride        input vu width stride, usually it equals to `inWidth`
* @param vuInData          input vu data
* @param outHeight         output Y height
* @param outWidth          output Y width need to be processed
* @param outYStride        output Y width stride, usually it equals to `outWidth`
* @param yOutData          output Y data
* @param outVUStride       output vu width stride, usually it equals to `outWidth`
* @param vuOutData         output vu data
* @param affineMatrix      transformation matrix
* @param border_type       support ppl::cv::BORDER_TYPE_CONSTANT/ppl::cv::BORDER_TYPE_REPLICATE/ppl::cv::BORDER_TYPE_TRANSPARENT
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>arm platforms supported<td> armv7 armv8 
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/warpaffine.h>
* int main(int argc, char** argv) {
*     const int inWidth = 320;
*     const int inHeight = 240;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* inYImage = (unsigned char*)malloc(inWidth * inHeight * sizeof(unsigned char));
*     unsigned char* inVUImage = (unsigned char*)malloc(inWidth * (inHeight / 2) * sizeof(unsigned char));
*     unsigned char* outYImage = (unsigned char*)malloc(outWidth * outHeight * sizeof(unsigned char));
*     unsigned char* outVUImage = (unsigned char*)malloc(outWidth * (outHeight / 2) * sizeof(unsigned char));
*     float* affineMatrix= (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::arm::WarpAffineNearestPoint_NV21<uchar>(inHeight, inWidth, inWidth, inYImage, 
*        inWidth, inVUImage, 
*        outHeight, outWidth, outWidth, outYImage, 
*        outWidth, outVUImage, 
*        affineMatrix, ppl::cv::BORDER_TYPE_CONSTANT);
*     free(inYImage);
*     free(inVUImage);
*     free(outYImage);
*     free(outVUImage);
*     free(affineMatrix);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode WarpAffineNearestPoint_NV21(int inHeight,
                                                   int inWidth,
                                                   int inYStride,
                                                   const T* yInData,
                                                   int inVUStride,
                                                   const T* vuInData,
                                                   int outHeight,
                                                   int outWidth,
                                                   int outYStride,
                                                   T* yOutData,
                                                   int outVUStride,
                                                   T* vuOutData,
                                                   const float* affineMatrix,
                                                   BorderType border_type = BORDER_TYPE_CONSTANT);

/**
* @brief Affine transformation with nearest neighbor interpolation method for I420 image
* @param inHeight          input Y height
* @param inWidth           input Y width need to be processed
* @param inYStride         input Y width stride, usually it equals to `inWidth`
* @param yInData           input Y data
* @param inUStride         input u width stride, usually it equals to `inWidth / 2`
* @param uInData           input u data
* @param inVStride         input v width stride, usually it equals to `inWidth / 2`
* @param vInData           input v data
* @param outHeight         output Y height
* @param outWidth          output Y width need to be processed
* @param outYStride        output Y width stride, usually it equals to `outWidth`
* @param yOutData          output Y data
* @param outUStride        output u width stride, usually it equals to `outWidth / 2`
* @param uOutData          output u data
* @param outVStride        output v width stride, usually it equals to `outWidth / 2`
* @param vOutData          output v data
* @param affineMatrix      transformation matrix
* @param border_type       support ppl::cv::BORDER_TYPE_CONSTANT/ppl::cv::BORDER_TYPE_REPLICATE/ppl::cv::BORDER_TYPE_TRANSPARENT
* @warning All input parameters must be valid, or undefined behaviour may occur.
* @remark The fllowing table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>uint8_t(uchar)
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>arm platforms supported<td> armv7 armv8 
* <tr><td>Header files<td> #include &lt;ppl/cv/arm/warpaffine.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/arm/warpaffine.h>
* int main(int argc, char** argv) {
*     const int inWidth = 320;
*     const int inHeight = 240;
*     const int outWidth = 320;
*     const int outHeight = 240;
*     unsigned char* inYImage = (unsigned char*)malloc(inWidth * inHeight * sizeof(unsigned char));
*     unsigned char* inUImage = (unsigned char*)malloc((inWidth / 2) * (inHeight / 2) * sizeof(unsigned char));
*     unsigned char* inVImage = (unsigned char*)malloc((inWidth / 2) * (inHeight / 2) * sizeof(unsigned char));
*     unsigned char* outYImage = (unsigned char*)malloc(outWidth * outHeight * sizeof(unsigned char));
*     unsigned char* outUImage = (unsigned char*)malloc((outWidth / 2) * (outHeight / 2) * sizeof(unsigned char));
*     unsigned char* outVImage = (unsigned char*)malloc((outWidth / 2) * (outHeight / 2) * sizeof(unsigned char));
*     float* affineMatrix= (float*)malloc(6 * sizeof(float));
*
*     ppl::cv::arm::WarpAffineNearestPoint_I420<uchar>(inHeight, inWidth, inWidth, inYImage, 
*        inWidth / 2, inUImage, inWidth / 2, inVImage, 
*        outHeight, outWidth, outWidth, outYImage, 
*        outWidth / 2, outUImage, outWidth / 2, outVImage, 
*        affineMatrix, ppl::cv::BORDER_TYPE_CONSTANT);
*     free(inYImage);
*     free(inUImage);
*     free(inVImage);
*     free(outYImage);
*     free(outUImage);
*     free(outVImage);
*     free(affineMatrix);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode WarpAffineNearestPoint_I420(int inHeight,
                                                   int inWidth,
                                                   int inYStride,
                                                   const T* yInData,
                                                   int inUStride,
                                                   const T* uInData,
                                                   int inVStride,
                                                   const T* vInData,
                                                   int outHeight,
                                                   int outWidth,
                                                   int outYStride,
                                                   T* yOutData,
                                                   int outUStride,
                                                   T* uOutData,
                                                   int outVStride,
                                                   T* vOutData,
                                                   const float* affineMatrix,
                                                   BorderType border_type = BORDER_TYPE_CONSTANT);

}
}
} // namespace ppl::cv::aarch64

#endif //!__ST_HPC_PPL_CV_AARCH64_WARPAFFINE_H_
