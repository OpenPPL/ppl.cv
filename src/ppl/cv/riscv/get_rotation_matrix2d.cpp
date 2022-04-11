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

#include "ppl/cv/riscv/get_rotation_matrix2d.h"
#include "ppl/cv/types.h"
#include "ppl/common/retcode.h"

#include <cmath>

#define CV_PI 3.1415926535897932384626433832795

namespace ppl {
namespace cv {
namespace riscv {

::ppl::common::RetCode GetRotationMatrix2D(
    float center_y,
    float center_x,
    double angle,
    double scale,
    double* out_data)
{
    if (nullptr == out_data) {
        return ppl::common::RC_INVALID_VALUE;
    }

    angle *= CV_PI / 180.0;
    double alpha = std::cos(angle) * scale;
    double beta = std::sin(angle) * scale;

    out_data[0] = alpha;
    out_data[1] = beta;
    out_data[2] = (1 - alpha) * center_x - beta * center_y;
    out_data[3] = -beta;
    out_data[4] = alpha;
    out_data[5] = beta * center_x + (1 - alpha) * center_y;

    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::riscv
