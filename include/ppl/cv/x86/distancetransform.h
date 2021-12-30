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

#ifndef __ST_HPC_PPL_CV_X86_DISTANCETRANSFORM_H_
#define __ST_HPC_PPL_CV_X86_DISTANCETRANSFORM_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Calculates the distance to the closest zero pixel for each pixel of the source image.
 * @tparam Tdst The data type of output image, currently only \a float is supported.
 * @param height            input image's height
 * @param width             input image's width need to be processed
 * @param inWidthStride     input image's width stride, usually it equals to `width`
 * @param inData            8-bit, single-channel (binary) source image.
 * @param outWidthStride    output image's width stride, usually it equals to `width`
 * @param outData           Output image with calculated distances. Single-channel image of the same size as src.
 * @param distanceType      Type of distance, see enum DistTypes, currently only \a DIST_L2 is supported.
 * @param maskSize          Size of the distance transform mask, see DistanceTransformMasks, currently only \a DIST_MASK_PRECISE is supported. 
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(Tdst)
 * <tr><td>float
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/distancetransform.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/distancetransform.h>
 * int32_t main(int32_t argc, char** argv) {
 *     const int32_t W = 640;
 *     const int32_t H = 480;
 *     const int32_t Stride = W;
 *     uint8_t* dev_iImage = (uint8_t*)malloc(W * H * sizeof(uint8_t));
 *     float* dev_oImage = (float*)malloc(W * H * sizeof(float));
 *     ppl::cv::x86::DistanceTransform(H, W, Stride, dev_iImage, Stride, dev_oImage, 
            ppl::cv::DIST_L2, ppl::cv::DIST_MASK_PRECISE);
 *
 *     free(dev_iImage);
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename Tdst>
::ppl::common::RetCode DistanceTransform(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inData,
    int32_t outWidthStride,
    Tdst* outData,
    DistTypes distanceType,
    DistanceTransformMasks maskSize);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_DISTANCETRANSFORM_H_
