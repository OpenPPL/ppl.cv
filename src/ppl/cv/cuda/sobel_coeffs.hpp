/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef _ST_HPC_PPL3_CV_CUDA_SOBEL_COEFFS_HPP_
#define _ST_HPC_PPL3_CV_CUDA_SOBEL_COEFFS_HPP_

#include "utility.hpp"

namespace ppl {
namespace cv {
namespace cuda {

#define __UNIFIED__ __device__ __managed__

/**************************** for normal kernels ******************************/

// scharr kernel when ksize is -1.
__UNIFIED__ int ksizen1_order1_dx_unnormalized[9] = { -3, 0,  3,
                                                     -10, 0, 10,
                                                      -3, 0,  3};
__UNIFIED__ int ksizen1_order1_dy_unnormalized[9] = {-3, -10, -3,
                                                      0,   0,  0,
                                                      3,  10,  3};
__UNIFIED__ int ksize1_order1_dx_unnormalized[9] = { 0, 0, 0,
                                                    -1, 0, 1,
                                                     0, 0, 0};
__UNIFIED__ int ksize1_order1_dy_unnormalized[9] = {0, -1, 0,
                                                    0,  0, 0,
                                                    0,  1, 0};
__UNIFIED__ int ksize1_order2_dx_unnormalized[9] = { 0,  0, 0,
                                                     1, -2, 1,
                                                     0,  0, 0};
__UNIFIED__ int ksize1_order2_dy_unnormalized[9] = {0,  1, 0,
                                                    0, -2, 0,
                                                    0,  1, 0};
__UNIFIED__ int ksize3_order1_dx_unnormalized[9] = {-1, 0, 1,
                                                    -2, 0, 2,
                                                    -1, 0, 1};
__UNIFIED__ int ksize3_order1_dy_unnormalized[9] = {-1, -2, -1,
                                                    0,  0,  0,
                                                    1,  2,  1};
__UNIFIED__ int ksize3_order2_dx_unnormalized[9] = {1, -2, 1,
                                                    2, -4, 2,
                                                    1, -2, 1};
__UNIFIED__ int ksize3_order2_dy_unnormalized[9] = {1,  2,  1,
                                                   -2, -4, -2,
                                                    1,  2,  1};
__UNIFIED__ int ksize5_order1_dx_unnormalized[25] = {-1,  -2, 0,  2, 1,
                                                     -4,  -8, 0,  8, 4,
                                                     -6, -12, 0, 12, 6,
                                                     -4,  -8, 0,  8, 4,
                                                     -1,  -2, 0,  2, 1};
__UNIFIED__ int ksize5_order1_dy_unnormalized[25] = {-1, -4,  -6, -4, -1,
                                                     -2, -8, -12, -8, -2,
                                                      0,  0,   0,  0,  0,
                                                      2,  8,  12,  8,  2,
                                                      1,  4,   6,  4,  1};
__UNIFIED__ int ksize5_order2_dx_unnormalized[25] = {1, 0,  -2, 0, 1,
                                                     4, 0,  -8, 0, 4,
                                                     6, 0, -12, 0, 6,
                                                     4, 0,  -8, 0, 4,
                                                     1, 0,  -2, 0, 1,};
__UNIFIED__ int ksize5_order2_dy_unnormalized[25] = { 1,  4,   6,  4,  1,
                                                      0,  0,   0,  0,  0,
                                                     -2, -8, -12, -8, -2,
                                                      0,  0,   0,  0,  0,
                                                      1,  4,   6,  4,  1};
__UNIFIED__ int ksize5_order3_dx_unnormalized[25] = {-1,  2, 0,  -2, 1,
                                                     -4,  8, 0,  -8, 4,
                                                     -6, 12, 0, -12, 6,
                                                     -4,  8, 0,  -8, 4,
                                                     -1,  2, 0,  -2, 1};
__UNIFIED__ int ksize5_order3_dy_unnormalized[25] = {-1, -4,  -6, -4, -1,
                                                      2,  8,  12,  8,  2,
                                                      0,  0,   0,  0,  0,
                                                     -2, -8, -12, -8, -2,
                                                      1,  4,   6,  4,  1};
__UNIFIED__ int ksize7_order1_dx_unnormalized[49] =
                                             {-1,   -4,   -5, 0,   5,  4, 1,
                                              -6,  -24,  -30, 0,  30, 24, 6,
                                             -15, -60,  -75, 0,  75, 60, 15,
                                             -20, -80, -100, 0, 100, 80, 20,
                                             -15, -60,  -75, 0,  75, 60, 15,
                                              -6,  -24,  -30, 0,  30, 24, 6,
                                              -1,   -4,   -5, 0,   5,  4, 1};
__UNIFIED__ int ksize7_order1_dy_unnormalized[49] =
                                             {-1, -6,  -15,  -20, -15, -6, -1,
                                              -4, -24, -60,  -80, -60, -24, -4,
                                              -5, -30, -75, -100, -75, -30, -5,
                                               0,   0,   0,    0,   0,   0,  0,
                                               5,  30,  75,  100,  75,  30, 5,
                                               4,  24,  60,   80,  60,  24,  4,
                                               1,   6,  15,   20,  15,   6,  1};
__UNIFIED__ int ksize7_order2_dx_unnormalized[49] =
                                             { 1,  2,  -1,   -4,  -1,  2,  1,
                                               6, 12,  -6,  -24,  -6, 12,  6,
                                              15, 30, -15, -60, -15, 30, 15,
                                              20, 40, -20, -80, -20, 40, 20,
                                              15, 30, -15, -60, -15, 30, 15,
                                               6, 12,  -6,  -24,  -6, 12,  6,
                                               1,  2,  -1,   -4,  -1,  2,  1};
__UNIFIED__ int ksize7_order2_dy_unnormalized[49] =
                                             { 1,   6,  15,  20,  15,   6,  1,
                                               2,  12,  30,  40,  30,  12,  2,
                                              -1,  -6, -15, -20, -15,  -6, -1,
                                              -4, -24, -60, -80, -60, -24, -4,
                                              -1,  -6, -15, -20, -15,  -6, -1,
                                               2,  12,  30,  40,  30,  12,  2,
                                               1,   6,  15,  20,  15,   6,  1};
__UNIFIED__ int ksize7_order3_dx_unnormalized[49] =
                                             { -1,  0,  3, 0,  -3, 0,  1,
                                               -6,  0, 18, 0, -18, 0,  6,
                                              -15,  0, 45, 0, -45, 0, 15,
                                              -20,  0, 60, 0, -60, 0, 20,
                                              -15,  0, 45, 0, -45, 0, 15,
                                               -6,  0, 18, 0, -18, 0,  6,
                                               -1,  0,  3, 0,  -3, 0,  1};
__UNIFIED__ int ksize7_order3_dy_unnormalized[49] =
                                             {-1,  -6, -15, -20, -15,  -6, -1,
                                               0,   0,   0,   0,   0,   0,  0,
                                               3,  18,  45,  60,  45,  18,  3,
                                               0,   0,   0,   0,   0,   0,  0,
                                              -3, -18, -45, -60, -45, -18, -3,
                                               0,   0,   0,   0,   0,   0,  0,
                                               1,   6,  15,  20,  15,   6,  1};

// scharr kernel when ksize is -1.
__UNIFIED__ float ksizen1_order1_dx_normalized[9] = { -3, 0,  3,
                                                     -10, 0, 10,
                                                      -3, 0,  3};
__UNIFIED__ float ksizen1_order1_dy_normalized[9] = {-3, -10, -3,
                                                      0,   0,  0,
                                                      3,  10,  3};
__UNIFIED__ float ksize1_order1_dx_normalized[9] = { 0, 0, 0,
                                                    -1, 0, 1,
                                                     0, 0, 0};
__UNIFIED__ float ksize1_order1_dy_normalized[9] = {0, -1, 0,
                                                    0,  0, 0,
                                                    0,  1, 0};
__UNIFIED__ float ksize1_order2_dx_normalized[9] = { 0,  0, 0,
                                                     1, -2, 1,
                                                     0,  0, 0};
__UNIFIED__ float ksize1_order2_dy_normalized[9] = {0,  1, 0,
                                                    0, -2, 0,
                                                    0,  1, 0};
__UNIFIED__ float ksize3_order1_dx_normalized[9] = {-1, 0, 1,
                                                    -2, 0, 2,
                                                    -1, 0, 1};
__UNIFIED__ float ksize3_order1_dy_normalized[9] = {-1, -2, -1,
                                                    0,  0,  0,
                                                    1,  2,  1};
__UNIFIED__ float ksize3_order2_dx_normalized[9] = {1, -2, 1,
                                                    2, -4, 2,
                                                    1, -2, 1};
__UNIFIED__ float ksize3_order2_dy_normalized[9] = {1,  2,  1,
                                                   -2, -4, -2,
                                                    1,  2,  1};
__UNIFIED__ float ksize5_order1_dx_normalized[25] = {-1,  -2, 0,  2, 1,
                                                     -4,  -8, 0,  8, 4,
                                                     -6, -12, 0, 12, 6,
                                                     -4,  -8, 0,  8, 4,
                                                     -1,  -2, 0,  2, 1};
__UNIFIED__ float ksize5_order1_dy_normalized[25] = {-1, -4,  -6, -4, -1,
                                                     -2, -8, -12, -8, -2,
                                                      0,  0,   0,  0,  0,
                                                      2,  8,  12,  8,  2,
                                                      1,  4,   6,  4,  1};
__UNIFIED__ float ksize5_order2_dx_normalized[25] = {1, 0,  -2, 0, 1,
                                                     4, 0,  -8, 0, 4,
                                                     6, 0, -12, 0, 6,
                                                     4, 0,  -8, 0, 4,
                                                     1, 0,  -2, 0, 1,};
__UNIFIED__ float ksize5_order2_dy_normalized[25] = { 1,  4,   6,  4,  1,
                                                      0,  0,   0,  0,  0,
                                                     -2, -8, -12, -8, -2,
                                                      0,  0,   0,  0,  0,
                                                      1,  4,   6,  4,  1};
__UNIFIED__ float ksize5_order3_dx_normalized[25] = {-1,  2, 0,  -2, 1,
                                                     -4,  8, 0,  -8, 4,
                                                     -6, 12, 0, -12, 6,
                                                     -4,  8, 0,  -8, 4,
                                                     -1,  2, 0,  -2, 1};
__UNIFIED__ float ksize5_order3_dy_normalized[25] = {-1, -4,  -6, -4, -1,
                                                      2,  8,  12,  8,  2,
                                                      0,  0,   0,  0,  0,
                                                     -2, -8, -12, -8, -2,
                                                      1,  4,   6,  4,  1};
__UNIFIED__ float ksize7_order1_dx_normalized[49] =
                                             {-1,   -4,   -5, 0,   5,  4, 1,
                                              -6,  -24,  -30, 0,  30, 24, 6,
                                             -15, -60,  -75, 0,  75, 60, 15,
                                             -20, -80, -100, 0, 100, 80, 20,
                                             -15, -60,  -75, 0,  75, 60, 15,
                                              -6,  -24,  -30, 0,  30, 24, 6,
                                              -1,   -4,   -5, 0,   5,  4, 1};
__UNIFIED__ float ksize7_order1_dy_normalized[49] =
                                             {-1, -6,  -15,  -20, -15, -6, -1,
                                              -4, -24, -60,  -80, -60, -24, -4,
                                              -5, -30, -75, -100, -75, -30, -5,
                                               0,   0,   0,    0,   0,   0,  0,
                                               5,  30,  75,  100,  75,  30, 5,
                                               4,  24,  60,   80,  60,  24,  4,
                                               1,   6,  15,   20,  15,   6,  1};
__UNIFIED__ float ksize7_order2_dx_normalized[49] =
                                             { 1,  2,  -1,   -4,  -1,  2,  1,
                                               6, 12,  -6,  -24,  -6, 12,  6,
                                              15, 30, -15, -60, -15, 30, 15,
                                              20, 40, -20, -80, -20, 40, 20,
                                              15, 30, -15, -60, -15, 30, 15,
                                               6, 12,  -6,  -24,  -6, 12,  6,
                                               1,  2,  -1,   -4,  -1,  2,  1};
__UNIFIED__ float ksize7_order2_dy_normalized[49] =
                                             { 1,   6,  15,  20,  15,   6,  1,
                                               2,  12,  30,  40,  30,  12,  2,
                                              -1,  -6, -15, -20, -15,  -6, -1,
                                              -4, -24, -60, -80, -60, -24, -4,
                                              -1,  -6, -15, -20, -15,  -6, -1,
                                               2,  12,  30,  40,  30,  12,  2,
                                               1,   6,  15,  20,  15,   6,  1};
__UNIFIED__ float ksize7_order3_dx_normalized[49] =
                                             { -1,  0,  3, 0,  -3, 0,  1,
                                               -6,  0, 18, 0, -18, 0,  6,
                                              -15,  0, 45, 0, -45, 0, 15,
                                              -20,  0, 60, 0, -60, 0, 20,
                                              -15,  0, 45, 0, -45, 0, 15,
                                               -6,  0, 18, 0, -18, 0,  6,
                                               -1,  0,  3, 0,  -3, 0,  1};
__UNIFIED__ float ksize7_order3_dy_normalized[49] =
                                             {-1,  -6, -15, -20, -15,  -6, -1,
                                               0,   0,   0,   0,   0,   0,  0,
                                               3,  18,  45,  60,  45,  18,  3,
                                               0,   0,   0,   0,   0,   0,  0,
                                              -3, -18, -45, -60, -45, -18, -3,
                                               0,   0,   0,   0,   0,   0,  0,
                                               1,   6,  15,  20,  15,   6,  1};

/*************************** for separate kernels *****************************/

// scharr kernel when ksize is -1.
__UNIFIED__ float row_ksizen1_order1_dx[3] = {-1, 0, 1};
__UNIFIED__ float col_ksizen1_order1_dx[3] = {3, 10, 3};
__UNIFIED__ float row_ksizen1_order1_dy[3] = {3, 10, 3};
__UNIFIED__ float col_ksizen1_order1_dy[3] = {-1, 0, 1};
__UNIFIED__ float row_ksize1_order1_dx[3] = {-1, 0, 1};
__UNIFIED__ float col_ksize1_order1_dx[3] = {1};
__UNIFIED__ float row_ksize1_order1_dy[3] = {1};
__UNIFIED__ float col_ksize1_order1_dy[3] = {-1, 0, 1};
__UNIFIED__ float row_ksize1_order2_dx[3] = {1, -2, 1};
__UNIFIED__ float col_ksize1_order2_dx[3] = {1};
__UNIFIED__ float row_ksize1_order2_dy[3] = {1};
__UNIFIED__ float col_ksize1_order2_dy[3] = {1, -2, 1};
__UNIFIED__ float row_ksize3_order1_dx[3] = {-1, 0, 1};
__UNIFIED__ float col_ksize3_order1_dx[3] = {1, 2, 1};
__UNIFIED__ float row_ksize3_order1_dy[3] = {1, 2, 1};
__UNIFIED__ float col_ksize3_order1_dy[3] = {-1, 0, 1};
__UNIFIED__ float row_ksize3_order2_dx[3] = {1, -2, 1};
__UNIFIED__ float col_ksize3_order2_dx[3] = {1, 2, 1};
__UNIFIED__ float row_ksize3_order2_dy[3] = {1, 2, 1};
__UNIFIED__ float col_ksize3_order2_dy[3] = {1, -2, 1};
__UNIFIED__ float row_ksize5_order1_dx[5] = {-1, -2, 0, 2, 1};
__UNIFIED__ float col_ksize5_order1_dx[5] = {1, 4, 6, 4, 1};
__UNIFIED__ float row_ksize5_order1_dy[5] = {1, 4, 6, 4, 1};
__UNIFIED__ float col_ksize5_order1_dy[5] = {-1, -2, 0, 2, 1};
__UNIFIED__ float row_ksize5_order2_dx[5] = {1, 0, -2, 0, 1};
__UNIFIED__ float col_ksize5_order2_dx[5] = {1, 4, 6, 4, 1};
__UNIFIED__ float row_ksize5_order2_dy[5] = {1, 4, 6, 4, 1};
__UNIFIED__ float col_ksize5_order2_dy[5] = {1, 0, -2, 0, 1};
__UNIFIED__ float row_ksize5_order3_dx[5] = {-1, 2, 0, -2, 1};
__UNIFIED__ float col_ksize5_order3_dx[5] = {1, 4, 6, 4, 1};
__UNIFIED__ float row_ksize5_order3_dy[5] = {1, 4, 6, 4, 1};
__UNIFIED__ float col_ksize5_order3_dy[5] = {-1, 2, 0, -2, 1};
__UNIFIED__ float row_ksize7_order1_dx[7] = {-1, -4, -5, 0, 5, 4, 1};
__UNIFIED__ float col_ksize7_order1_dx[7] = {1, 6, 15, 20, 15, 6, 1};
__UNIFIED__ float row_ksize7_order1_dy[7] = {1, 6, 15, 20, 15, 6, 1};
__UNIFIED__ float col_ksize7_order1_dy[7] = {-1, -4, -5, 0, 5, 4, 1};
__UNIFIED__ float row_ksize7_order2_dx[7] = {1, 2, -1, -4, -1, 2, 1};
__UNIFIED__ float col_ksize7_order2_dx[7] = {1, 6, 15, 20, 15, 6, 1};
__UNIFIED__ float row_ksize7_order2_dy[7] = {1, 6, 15, 20, 15, 6, 1};
__UNIFIED__ float col_ksize7_order2_dy[7] = {1, 2, -1, -4, -1, 2, 1};
__UNIFIED__ float row_ksize7_order3_dx[7] = {-1, 0, 3, 0, -3, 0, 1};
__UNIFIED__ float col_ksize7_order3_dx[7] = {1, 6, 15, 20, 15, 6, 1};
__UNIFIED__ float row_ksize7_order3_dy[7] = {1, 6, 15, 20, 15, 6, 1};
__UNIFIED__ float col_ksize7_order3_dy[7] = {-1, 0, 3, 0, -3, 0, 1};

}  // cuda
}  // cv
}  // ppl

#endif  // _ST_HPC_PPL3_CV_CUDA_SOBEL_COEFFS_HPP_
