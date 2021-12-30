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

#ifndef PPL_CV_X86_UTIL_H_
#define PPL_CV_X86_UTIL_H_

#include "ppl/cv/types.h"
#include <stdint.h>
#include <vector>
#include <cstring>
#include <cassert>

namespace ppl {
namespace cv {
namespace x86 {
inline const uint8_t sat_cast_u8(int32_t data)
{
    return data > 255 ? 255 : (data < 0 ? 0 : data);
}
template<typename T>
inline const T round(const T &a, const T &b)
{
    return a / b * b;
}

template<typename T>
inline const T round_up(const T &a, const T &b)
{
    return (a + b - static_cast<T>(1)) / b * b;
}

template<typename T, typename FUNCTION>
inline void Map(T *out_data, const T *in_data, int32_t n, FUNCTION f, T operand0) {
    for (int32_t i = 0; i < n; ++i) {
        out_data[i] = f(in_data[i], operand0);
    }
}

template<typename T, typename FUNCTION>
inline void Map(T *out_data, const T *in_data0, const T *in_data1, int32_t n, FUNCTION f) {
    for (int32_t i = 0; i < n; ++i) {
        out_data[i] = f(in_data0[i], in_data1[i]);
    }
}

template<typename T, typename FUNCTION>
inline void Map(T *out_data, const T *in_data, float alpha, int32_t n, FUNCTION f, T operand0) {
    for (int32_t i = 0; i < n; ++i) {
        out_data[i] = alpha * f(in_data[i], operand0);
    }
}

template<typename T, typename FUNCTION>
inline void Map(T *out_data, const T *in_data0, const T *in_data1, float alpha, int32_t n, FUNCTION f) {
    for (int32_t i = 0; i < n; ++i) {
        out_data[i] = alpha * f(in_data0[i], in_data1[i]);
    }
}

static inline int32_t borderInterpolate(int32_t p, int32_t len)
{
    if (len == 1) {
        return 0;
    }
    if (p < 0 || p >= len) {
        do {
            if (p < 0) {
                p = -p;
            } else {
                p = len - 1 - (p - len) - 1;
            }
        } while (p < 0 || p >= len);
    }

    return p;
}

} //! namespace x86
} //! namespace cv
} //! namespace ppl

#endif //! PPL_CV_X86_UTIL_H_
