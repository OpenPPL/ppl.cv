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

#include "ppl/cv/riscv/arithmetic.h"
#include "ppl/cv/riscv/arithmetic_common.h"
#include "ppl/cv/types.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace ppl {
namespace cv {
namespace riscv {

template <>
::ppl::common::RetCode Add<float, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return arithmetic_op<float, ARITHMETIC_ADD>(height, width, 1, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<float, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return arithmetic_op<float, ARITHMETIC_ADD>(height, width, 3, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<float, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const float *inData0,
    int32_t inWidthStride1,
    const float *inData1,
    int32_t outWidthStride,
    float *outData)
{
    return arithmetic_op<float, ARITHMETIC_ADD>(height, width, 4, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<uint8_t, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return arithmetic_op<uint8_t, ARITHMETIC_ADD>(height, width, 1, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<uint8_t, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return arithmetic_op<uint8_t, ARITHMETIC_ADD>(height, width, 3, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

template <>
::ppl::common::RetCode Add<uint8_t, 4>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride0,
    const uint8_t *inData0,
    int32_t inWidthStride1,
    const uint8_t *inData1,
    int32_t outWidthStride,
    uint8_t *outData)
{
    return arithmetic_op<uint8_t, ARITHMETIC_ADD>(height, width, 4, inWidthStride0, inData0, inWidthStride1, inData1, outWidthStride, outData);
}

}
}
} // namespace ppl::cv::riscv
