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

#ifndef __ST_HPC_PPL_CV_X86_MIN_MAX_LOC_H_
#define __ST_HPC_PPL_CV_X86_MIN_MAX_LOC_H_
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
* @brief Finds the global minimum and maximum in an array.
* @tparam T     The data type of input and output image, currently only \a float and \a uchar is supported.
* @param height height of src
* @param width  width of src
* @param steps  steps of src,in element, usually it equals to `width `
* @param src    input single-channel array.
* @param minVal returned minimum value;
* @param maxVal returned maximum value;
* @param minCol location of minimun value in horizontal direction;
* @param minRow location of minimun value in vertical direction; 
* @param maxCol location of maximum value in horizontal direction;
* @param maxRow location of maximum value in vertical direction; 
* @param mastSteps [optional] steps of mask,in Bytes, usually it equals to `width * sizeof(uchar)`
* @param mask   [optional] mask array, if the mask pixel is not 0, process the corresponding src pixel
* @warning All input parameters must be valid, or undefined behaviour may occur.The function MinMaxLoc finds the minimum and maximum element values and their positions. The extremums are searched across the whole array. The function do not work with multi-channel arrays. If you need to find minimum or maximum elements across all the channels, reshape first to reinterpret the array as single-channel. 
* @remark The following table show which data type and channels are supported.
* <table>
* <tr><th>Data type(T)
* <tr><td>float
* <tr><td>uint8
* </table>
* <table>
* <caption align="left">Requirements</caption>
* <tr><td>X86 platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/x86/MinMaxLoc.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/x86/MinMaxLoc.h>
* int main(int argc, char** argv) {
*     const int W = 640;
*     const int H = 480;
*     float* dev_iImage0 = (float*)malloc(W * H * sizeof(float));
*
*     float minVal = 0.0;
*     int minCol = 0;
*     int minRow = 0;
*     float maxVal = 0.0;
*     int maxCol = 0;
*     int maxRow = 0;
*     int steps = W * sizeof(float);
*     ppl::cv::x86::MinMaxLoc<float>(H, W, steps, dev_iImage0, &minVal, 
*          &maxVal, &minCol, &minRow, &maxCol, &maxRow);
*
*     free(dev_iImage0);
*     return 0;
* }
* @endcode
***************************************************************************************************/
template <typename T>
::ppl::common::RetCode MinMaxLoc(
    int height,
    int width,
    int steps,
    const T *src,
    T *minVal,
    T *maxVal,
    int *minCol,
    int *minRow,
    int *maxCol,
    int *maxRow,
    int maskSteps     = 0,
    const uchar *mask = NULL);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL3_CV_X86_MIN_MAX_LOC_H_
