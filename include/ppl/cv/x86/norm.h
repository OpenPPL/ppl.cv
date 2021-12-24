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

#ifndef __ST_HPC_PPL_CV_X86_NORM_H_
#define __ST_HPC_PPL_CV_X86_NORM_H_
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Calculates the absolute norm of an image.
 * @tparam T The data type, used for input image, currently only uchar and float are supported.
 * @tparam c The number of channels of input image, 1, 3 and 4 is supported for now.
 * @param inHeight      input image's height.
 * @param inWidth       input image's width need to be processed.
 * @param inStride      input image's width stride, usually it equals to `inWidth * c`
 * @param inData        input image data.
 * @param normType      Norm type. NORM_L2 is default.Only NORM_L1, NORM_L2, and NORM_INF are 
 *                        supported for now.
 * @param maskStride    the width stride of input image in bytes, similar to inStride.
 * @param mask          optional operation mask; it must have the same size as 
 *                        inData and CV_8UC1 type.
 * @return the absolute norm of an image in double type.
 * @note Multi-channel input arrays are treated as single-channel arrays, that 
 *       is, the results for all channels are combined. 
 * * <table>
 * <tr><th>Data type(T)<th>channels
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * <tr><td>uchar<td>1
 * <tr><td>uchar<td>3
 * <tr><td>uchar<td>4
 * </table>
 * * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All 
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/norm.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/norm.h>
 * int main(int argc, char** argv) {
 *     const int W = 640;
 *     const int H = 480;
 *     const int C = 3;
 *     float* dev_iImage0 = (float*)malloc(W * H * C * sizeof(float));
 *     double res = ppl::cv::x86::Norm<float, 3>(H, W, W * C, dev_iImage0);
 *
 *     free(dev_iImage0);
 *     return 0;
 * }
 * @endcode
 */

template <typename T, int c>
double Norm(int inHeight,
            int inWidth,
            int inStride,
            const T* inData,
            NormTypes normType = NORM_L2,
            int maskStride     = 0,
            const uchar* mask  = NULL);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL3_CV_X86_NORM_H_
