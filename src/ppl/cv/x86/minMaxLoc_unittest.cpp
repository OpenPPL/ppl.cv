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

#include "ppl/cv/x86/minMaxLoc.h"
#include "ppl/cv/x86/cvtcolor.h"
#include "ppl/cv/x86/test.h"
#include <opencv2/imgproc.hpp>
#include <memory>
#include <gtest/gtest.h>
#include "ppl/cv/debug.h"

template <typename T>
class MinMaxLoc_ : public ::testing::TestWithParam<Size> {
public:
    MinMaxLoc_()
    {
    }
    ~MinMaxLoc_()
    {
    }

    void apply(const Size& size)
    {
        T* src0 = (T*)malloc(size.width * size.height * sizeof(T));
        ppl::cv::debug::randomFill<T>(src0, size.width * size.height, 0, 255);

        T minVal   = src0[0];
        int minCol = 0;
        int minRow = 0;
        T maxVal   = src0[0];
        int maxCol = 0;
        int maxRow = 0;

        ppl::cv::x86::MinMaxLoc<T>(size.height, size.width, size.width, src0, &minVal, &maxVal, &minCol, &minRow, &maxCol, &maxRow);

        double mindist, maxdist;
        cv::Point min_loc, max_loc;
        cv::Mat iMat0(size.height, size.width, CV_MAKETYPE(cv::DataType<T>::depth, 1), src0);
        cv::minMaxLoc(iMat0, &mindist, &maxdist, &min_loc, &max_loc);

        if (std::is_same<T, unsigned char>::value) {
            std::cout << "[ppl.cv] minVal:" << (int)minVal << " maxVal:" << (int)maxVal << " minLoc:" << minCol << "," << minRow
                      << " maxLoc:" << maxCol << "," << maxRow << std::endl;
        } else {
            std::cout << "[ppl.cv] minVal:" << minVal << " maxVal:" << maxVal << " minLoc:" << minCol << "," << minRow
                      << " maxLoc:" << maxCol << "," << maxRow << std::endl;
        }

        std::cout << "[opencv] minVal:" << mindist << " maxVal:" << maxdist << " minLoc:" << min_loc.x << "," << min_loc.y
                  << " maxLoc:" << max_loc.x << "," << max_loc.y << std::endl;

        EXPECT_EQ(minVal, mindist);
        EXPECT_EQ(maxVal, maxdist);
        EXPECT_EQ(minCol, min_loc.x);
        EXPECT_EQ(minRow, min_loc.y);
        EXPECT_EQ(maxCol, max_loc.x);
        EXPECT_EQ(maxRow, max_loc.y);

        free(src0);
    }
};

#define R(name, t)               \
    using name = MinMaxLoc_<t>;  \
    TEST_P(name, abc)            \
    {                            \
        this->apply(GetParam()); \
    }                            \
    INSTANTIATE_TEST_CASE_P(standard, name, ::testing::Values(Size{320, 240}, Size{640, 480}, Size{5, 5}));

R(MinMaxLoc_f32, float)
R(MinMaxLoc_u8, uchar)
