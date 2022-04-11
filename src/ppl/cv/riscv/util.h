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

#ifndef PPL_CV_RISCV_UTIL_H_
#define PPL_CV_RISCV_UTIL_H_

#include "ppl/cv/types.h"
#include <stdint.h>
#include <vector>
#include <cstring>
#include <cassert>
#include <limits.h>

namespace ppl {
namespace cv {
namespace riscv {

template <typename T>
inline const uint8_t sat_cast_u8(T data)
{
    return data > 255 ? 255 : (data < 0 ? 0 : (uint8_t)data);
}

static inline int32_t img_floor(float a)
{
    return (((a) >= 0) ? ((int32_t)a) : ((int32_t)a - 1));
}

static inline int32_t img_floor(double a)
{
    return (((a) >= 0) ? ((int32_t)a) : ((int32_t)a - 1));
}

template <typename T>
inline T clip(T value, T min_value, T max_value)
{
    return std::min(std::max(value, min_value), max_value);
}

}
}
} // namespace ppl::cv::riscv

#endif //! PPL_CV_RISCV_UTIL_H_
