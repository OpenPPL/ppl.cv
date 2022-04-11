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

#ifndef __ST_HPC_PPL_CV_RISCV_GET_AFFINE_TRANSFORM_H_
#define __ST_HPC_PPL_CV_RISCV_GET_AFFINE_TRANSFORM_H_

namespace ppl {
namespace cv {
namespace riscv {

/**
* @brief Calculates an affine transform from three pairs of the corresponding points.
* @param src_points               src keypoints data, which has 3 points and each points has 2 double values (x, y)
* @param dst_points               dst keypoints data, which has 3 points and each points has 2 double values (x, y)
* @param mat                      transform matrix, 2x3 matrix. can be nullptr.
* @param inverse_mat              transform matrix inversed, 2x3 matrix. can be nullptr.
* @warning All input parameters must be valid, or undefined behaviour may occur.
* <caption align="left">Requirements</caption>
* <tr><td>riscv platforms supported<td> All
* <tr><td>Header files<td> #include &lt;ppl/cv/riscv/get_affine_transform.h&gt;
* <tr><td>Project<td> ppl.cv
* @since ppl.cv-v1.0.0
* ###Example
* @code{.cpp}
* #include <ppl/cv/riscv/get_affine_transform.h>
* int32_t main(int32_t argc, char** argv) {
*     double src_points[6], dst_points[6], mat[6], inverse_mat[6];
*     src_points[0] = 5;   src_points[1] = 9;
*     src_points[2] = 223; src_points[3] = 13;
*     src_points[4] = 49;  src_points[5] = 146;
*
*     dst_points[0] = 27;  dst_points[1] = 19;
*     dst_points[2] = 103; dst_points[3] = 47;
*     dst_points[4] = 18;  dst_points[5] = 91;
*
*     ppl::cv::riscv::GetAffineTransform(src_points, dst_points, mat, inverse_mat);
*     return 0;
* }
* @endcode
***************************************************************************************************/

void GetAffineTransform(
    const double *src_points,
    const double *dst_points,
    double *mat,
    double *inverse_mat);

}
}
} // namespace ppl::cv::riscv

#endif //!__ST_HPC_PPL_CV_RISCV_GET_AFFINE_TRANSFORM_H_
