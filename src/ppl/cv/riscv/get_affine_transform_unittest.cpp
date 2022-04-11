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

#include "ppl/cv/riscv/get_affine_transform.h"
#include "ppl/cv/riscv/test.h"
#include <memory>
#include <gtest/gtest.h>
#include <time.h>
#include "ppl/common/retcode.h"
#include "ppl/cv/debug.h"

#define CHECK_RESULT(a, b) \
    EXPECT_LT(abs(a - b), diff_THR);

TEST(GetAffineTransform, riscv)
{
    double src[6], dst[6];
    double mat[6];
    double inverse_mat[6];
    src[0] = 5;
    src[1] = 9;
    src[2] = 223;
    src[3] = 13;
    src[4] = 49;
    src[5] = 146;

    dst[0] = 27;
    dst[1] = 19;
    dst[2] = 103;
    dst[3] = 47;
    dst[4] = 18;
    dst[5] = 91;

    ppl::cv::riscv::GetAffineTransform(
        src,
        dst,
        mat,
        inverse_mat);
    double ref_mat[6] = {0.351903, -0.178713, 26.848907, 0.119502, 0.487167, 14.017985};
    double ref_inv[6] = {2.526904, 0.926974, -80.838928, -0.619846, 1.825297, -8.944791};
    double diff_THR = 1e-4;
    for (int32_t i = 0; i < 6; i++) {
        CHECK_RESULT(ref_mat[i], mat[i]);
    }
    for (int32_t i = 0; i < 6; i++) {
        CHECK_RESULT(ref_inv[i], inverse_mat[i]);
    }

    for (int32_t loop = 0; loop < 100; loop++) {
        // random generate points
        srand(time(NULL) + loop);
        for (int32_t i = 0; i < 6; i++) {
            src[i] = rand() % 400;
            dst[i] = rand() % 400;
        }
        ppl::cv::riscv::GetAffineTransform(
            src,
            dst,
            mat,
            inverse_mat);

        // use transform mat to compute dst points
        double tmp_dst[6];
        double tmp_src[6];
        // 3x3 mat
        // dst_x = src_x * mat00 + src_y * mat01 + mat02
        // dst_y = src_x * mat10 + src_y * mat11 + mat12
        tmp_dst[0] = src[0] * mat[0] + src[1] * mat[1] + mat[2];
        tmp_dst[1] = src[0] * mat[3] + src[1] * mat[4] + mat[5];
        tmp_dst[2] = src[2] * mat[0] + src[3] * mat[1] + mat[2];
        tmp_dst[3] = src[2] * mat[3] + src[3] * mat[4] + mat[5];
        tmp_dst[4] = src[4] * mat[0] + src[5] * mat[1] + mat[2];
        tmp_dst[5] = src[4] * mat[3] + src[5] * mat[4] + mat[5];
        for (int32_t i = 0; i < 6; i++) {
            CHECK_RESULT(dst[i], tmp_dst[i]);
        }
        // 3x3 inv mat
        // src_x = dst_x * inverse_mat00 + dst_y * inverse_mat01 + inverse_mat02
        // src_y = dst_x * inverse_mat10 + dst_y * inverse_mat11 + inverse_mat12
        tmp_src[0] = dst[0] * inverse_mat[0] + dst[1] * inverse_mat[1] + inverse_mat[2];
        tmp_src[1] = dst[0] * inverse_mat[3] + dst[1] * inverse_mat[4] + inverse_mat[5];
        tmp_src[2] = dst[2] * inverse_mat[0] + dst[3] * inverse_mat[1] + inverse_mat[2];
        tmp_src[3] = dst[2] * inverse_mat[3] + dst[3] * inverse_mat[4] + inverse_mat[5];
        tmp_src[4] = dst[4] * inverse_mat[0] + dst[5] * inverse_mat[1] + inverse_mat[2];
        tmp_src[5] = dst[4] * inverse_mat[3] + dst[5] * inverse_mat[4] + inverse_mat[5];
        for (int32_t i = 0; i < 6; i++) {
            CHECK_RESULT(src[i], tmp_src[i]);
        }
    }
}
