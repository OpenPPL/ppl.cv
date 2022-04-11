// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for subitional information
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

#include "ppl/cv/riscv/erode.h"
#include "ppl/cv/riscv/typetraits.h"
#include "ppl/cv/riscv/morph.h"
#include "ppl/common/log.h"
#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include <algorithm>
#include <cmath>
#include <float.h>
#include <string.h>

namespace ppl {
namespace cv {
namespace riscv {

#define Erode(dt, nc, name)                                                                                                                                                                                                                                     \
    template <>                                                                                                                                                                                                                                                 \
    ::ppl::common::RetCode Erode<dt, nc>(int32_t height, int32_t width, int32_t inWidthStride, const dt* inData, int32_t kernelx_len, int32_t kernely_len, const uint8_t* kernel, int32_t outWidthStride, dt* outData, BorderType border_type, dt border_value) \
    {                                                                                                                                                                                                                                                           \
        if (inData == nullptr || outData == nullptr || kernel == nullptr) {                                                                                                                                                                                     \
            return ppl::common::RC_INVALID_VALUE;                                                                                                                                                                                                               \
        }                                                                                                                                                                                                                                                       \
        if (height <= 0 || width <= 0 || inWidthStride < width || outWidthStride == 0) {                                                                                                                                                                        \
            return ppl::common::RC_INVALID_VALUE;                                                                                                                                                                                                               \
        }                                                                                                                                                                                                                                                       \
        if (border_type != BORDER_CONSTANT) {                                                                                                                                                                                                                   \
            border_value = std::numeric_limits<dt>::lowest();                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                                                       \
        if (border_type == BORDER_CONSTANT) {                                                                                                                                                                                                                   \
            return morph<dt, MORPH_ERODE, BORDER_CONSTANT, nc>(height, width, inWidthStride, inData, kernelx_len, kernely_len, kernel, outWidthStride, outData, border_value);                                                                                  \
        } else if (border_type == BORDER_REPLICATE) {                                                                                                                                                                                                           \
            return morph<dt, MORPH_ERODE, BORDER_REPLICATE, nc>(height, width, inWidthStride, inData, kernelx_len, kernely_len, kernel, outWidthStride, outData, border_value);                                                                                 \
        } else if (border_type == BORDER_REFLECT) {                                                                                                                                                                                                             \
            return morph<dt, MORPH_ERODE, BORDER_REFLECT, nc>(height, width, inWidthStride, inData, kernelx_len, kernely_len, kernel, outWidthStride, outData, border_value);                                                                                   \
        } else if (border_type == BORDER_REFLECT101) {                                                                                                                                                                                                          \
            return morph<dt, MORPH_ERODE, BORDER_REFLECT101, nc>(height, width, inWidthStride, inData, kernelx_len, kernely_len, kernel, outWidthStride, outData, border_value);                                                                                \
        } else {                                                                                                                                                                                                                                                \
            return ppl::common::RC_UNSUPPORTED;                                                                                                                                                                                                                 \
        }                                                                                                                                                                                                                                                       \
    }

Erode(float, 1, f32)
    Erode(float, 3, f32)
        Erode(float, 4, f32)

            Erode(uint8_t, 1, u8)
                Erode(uint8_t, 3, u8)
                    Erode(uint8_t, 4, u8)
}
}
} // namespace ppl::cv::riscv
