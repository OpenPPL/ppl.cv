// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_HPC_PPL_CV_X86_IMREAD_H_
#define __ST_HPC_PPL_CV_X86_IMREAD_H_

#include "ppl/cv/types.h"

#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace x86 {

/**
 * @brief Loads an image from a file.
 * @param fileName  name of file to be loaded.
 * @param height    pointer to store the height of the loaded image.
 * @param width     pointer to store the width of the loaded image.
 * @param channels  pointer to store the channels of the loaded image.
 * @param stride    pointer to store the row stride of the loaded image.
 * @param image     pointer to a memory buffer storing the pixel data of the
 *                  loaded image. This buffer is allocated in Imread()
 *                  according to the height and stride of the image.
 * @return The execution status, succeeds or fails with an error code.
 * @note 1 The function determines the type of an image by the content, not by
 *         the file extension.
 *       2 File formats of windows bitmaps(*.bmp), JPEG(*.jpeg, *.jpg) and
 *         portable network graphcs(*.png) are supported for now.
 *       3 In the case of color images, the decoded images will have the
 *         channels stored in B G R / B G R A order.
 *       4 1, 3 and 4 channels are supported in the decoded data.
 *       5 uchar is supported in the decoded data.
 *       6 By default number of pixels must be less than 2^30.
 *       7 image[] must be freed when unused.
 * @warning All input parameters must be valid, or undefined behaviour may occur.
 * @remark
 * <caption align="left">Requirements</caption>
 * <tr><td>x86 platforms supported<td> All
 * <tr><td>Header files<td> #include &lt;ppl/cv/x86/imread.h&gt;
 * <tr><td>Project<td> ppl.cv
 * @since ppl.cv-v1.0.0
 * ###Example
 * @code{.cpp}
 * #include "ppl/cv/x86/imread.h"
 *
 * int32_t main(int32_t argc, char** argv) {
 *     char file_name[] = "test.bmp";
 *     int height, width, channels, stride;
 *     uchar* image;
 *
 *     ppl::cv::x86::Imread(file_name, &height, &width, &channels, &stride,
 *                          &image);
 *
 *     free(image);
 *
 *     return 0;
 * }
 * @endcode
 ******************************************************************************/
::ppl::common::RetCode Imread(const char* fileName,
                              int* height,
                              int* width,
                              int* channels,
                              int* stride,
                              uchar** image);

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! __ST_HPC_PPL_CV_X86_IMREAD_H_
