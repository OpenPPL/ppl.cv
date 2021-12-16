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

#ifndef __ST_HPC_PPL_CV_X86_SETVALUE_H_
#define __ST_HPC_PPL_CV_X86_SETVALUE_H_

#include "ppl/common/retcode.h"
#include "ppl/cv/types.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Set value to image
 * @tparam T The data type of input data, currently only \a uint8_t and \a float are supported.
 * @tparam outChannels The number of channels of input data, 1, 3 and 4 are supported.
 * @tparam maskChannels The number of channels of mask, 1 or the sampe as outChannels are supported.
 * @param height input image's height need to be processed
 * @param width  input image's width need to be processed
 * @param stride input image's stride need to be processed, usually it equals to `width * c`
 * @param out    output data
 * @param value  value need to set
 * @param maskStride [optional] mask image's stride need to be processed, usually it equals to `width`
 * @param mask   [optional] Operation mask of the same size as \a out. Its non-zero elements indicate which 
    elements need to be copied. The mask has to be of type uint8_t and can have 1 or multiple channels.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark The fllowing table show which data type and channels are supported.
 * <table>
 * <tr><th>Data type(T)<th>outChannels
 * <tr><td>uint8_t(uchar)<td>1
 * <tr><td>uint8_t(uchar)<td>3
 * <tr><td>uint8_t(uchar)<td>4
 * <tr><td>float<td>1
 * <tr><td>float<td>3
 * <tr><td>float<td>4
 * </table>
 * <table>
 * <caption align="left">Requirements</caption>
 * <tr><td>X86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/setvalue.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/setvalue.h>
 * int main(int argc, char** argv) {
 *     const int width  = 640;
 *     const int height = 480;
 *     const int channels= 3;
 *     float value = 0.5f;
 *     float* dev_oImage  = (float*)malloc(width * height * channels * sizeof(float));
 *
 *     ppl::cv::x86::SetTo<float, channels>(height, width, width * channels, dev_oImage, value);
 *
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T, int outChannels, int maskChannels = 1>
::ppl::common::RetCode SetTo(
    int height,
    int width,
    int stride,
    T* out,
    const T value,
    int maskStride      = 0,
    const uint8_t* mask = NULL);

/**
 * @brief Set 0 
 * @tparam T The data type of input data, currently only \a uint8_t and \a float are supported.
 * @tparam c The number of channels of input data, 1, 3 and 4 are supported.
 * @param height input image's height need to be processed
 * @param width  input image's width need to be processed
 * @param stride input image's stride need to be processed, usually it equals to `width * c`
 * @param out    output data
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
 * <tr><td>X86 platforms supported<td> x86v7 x86v8
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/setvalue.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/setvalue.h>
 * int main(int argc, char** argv) {
 *     const int width  = 640;
 *     const int height = 480;
 *     const int channels= 3;
 *     float value = 1.0f;
 *     float* dev_oImage  = (float*)malloc(width * height * channels * sizeof(float));
 *
 *     ppl::cv::x86::Zeros<float, channels>(height, width, width * channels, dev_oImage);
 *
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T, int c>
::ppl::common::RetCode Zeros(
    int height,
    int width,
    int stride,
    T* out);

/**
 * @brief Set 1
 * @tparam T The data type of input data, currently only \a uint8_t and \a float are supported.
 * @tparam c The number of channels of input data, 1, 3 and 4 are supported.
 * @param height input image's height need to be processed
 * @param width  input image's width need to be processed
 * @param stride input image's stride need to be processed, usually it equals to `width * c`
 * @param out    output data
 * @note    In case of multi-channels type, only the first channel will be set to 1's, the others will be set to 0's.
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
 * <tr><td>X86 platforms supported<td> x86v7 x86v8
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/setvalue.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include <ppl/cv/x86/setvalue.h>
 * int main(int argc, char** argv) {
 *     const int width  = 640;
 *     const int height = 480;
 *     const int channels= 3;
 *     float value = 1.0f;
 *     float* dev_oImage  = (float*)malloc(width * height * channels * sizeof(float));
 *
 *     ppl::cv::x86::Ones<float, channels>(height, width, width * channels, dev_oImage);
 *
 *     free(dev_oImage);
 *     return 0;
 * }
 * @endcode
 ***************************************************************************************************/
template <typename T, int c>
::ppl::common::RetCode Ones(
    int height,
    int width,
    int stride,
    T* out);

}
}
} // namespace ppl::cv::x86
#endif //! __ST_HPC_PPL_CV_X86_SETVALUE_H_
