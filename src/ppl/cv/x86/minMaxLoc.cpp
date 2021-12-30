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
#include "ppl/cv/x86/norm.h"
#include "ppl/cv/x86/avx/internal_avx.hpp"
#include "ppl/cv/x86/fma/internal_fma.hpp"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "ppl/common/retcode.h"
#include "ppl/common/x86/sysinfo.h"
#include <string.h>
#include <cmath>

#include <vector>
#include <limits.h>
#include <immintrin.h>
#include <cassert>
#include <type_traits>
#include <float.h>
#ifdef _WIN32
#include <algorithm>
#endif
namespace ppl {
namespace cv {
namespace x86 {

template <typename T>
T max_limit();

template <typename T>
T min_limit();

template <>
float max_limit<float>()
{
    return FLT_MAX;
}

template <>
float min_limit<float>()
{
    return -FLT_MAX;
}

template <>
uchar max_limit<uchar>()
{
    return 255;
}

template <>
uchar min_limit<uchar>()
{
    return 0;
}

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
    int maskSteps,
    const uchar *mask)
{
    assert(src != nullptr);
    assert(width > 0);
    assert(height > 0);
    assert(steps >= width);

    *minCol = -1;
    *minRow = -1;
    *maxCol = -1;
    *maxRow = -1;
    *minVal = max_limit<T>();
    *maxVal = min_limit<T>();
    ;

    if (mask == nullptr) {
        for (int l = 0; l < height; ++l) {
            const T *ptr_data = src + steps * l;
            for (int i = 0; i < width; ++i) {
                float val = ptr_data[i];
                if (val < *minVal) {
                    *minVal = val;
                    *minCol = i;
                    *minRow = l;
                }
                if (val > *maxVal) {
                    *maxVal = val;
                    *maxCol = i;
                    *maxRow = l;
                }
            }
        }
    } else {
        for (int l = 0; l < height; ++l) {
            const T *ptr_data     = src + steps * l;
            const uchar *ptr_mask = mask + maskSteps * l;
            for (int i = 0; i < width; ++i) {
                float val = ptr_data[i];
                if (ptr_mask[i]) {
                    if (val < *minVal) {
                        *minVal = val;
                        *minCol = i;
                        *minRow = l;
                    }
                    if (val > *maxVal) {
                        *maxVal = val;
                        *maxCol = i;
                        *maxRow = l;
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template ::ppl::common::RetCode MinMaxLoc<float>(
    int height,
    int width,
    int steps,
    const float *src,
    float *minVal,
    float *maxVal,
    int *minCol,
    int *minRow,
    int *maxCol,
    int *maxRow,
    int maskSteps,
    const uchar *mask);

template ::ppl::common::RetCode MinMaxLoc<uchar>(
    int height,
    int width,
    int steps,
    const uchar *src,
    uchar *minVal,
    uchar *maxVal,
    int *minCol,
    int *minRow,
    int *maxCol,
    int *maxRow,
    int maskSteps,
    const uchar *mask);

}
}
} // namespace ppl::cv::x86
