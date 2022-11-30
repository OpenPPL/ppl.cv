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
 *
 * Definition of macro, typedef, enum, function templates, and inline
 * functions to facilitate computation.
 */

#ifndef _ST_HPC_PPL_CV_OCL_UTILITY_HPP_
#define _ST_HPC_PPL_CV_OCL_UTILITY_HPP_

#include "ppl/common/log.h"
#include "ppl/common/retcode.h"

namespace ppl {
namespace cv {
namespace ocl {

#define PPL_ASSERT(expression)                                                 \
if (!(expression)) {                                                           \
  LOG(ERROR) << "Assertion failed: " << #expression;                           \
  return ppl::common::RC_INVALID_VALUE;                                        \
}

typedef unsigned char uchar;
typedef signed char schar;
typedef unsigned short ushort;
typedef unsigned int uint;

// configuration of thread blocks.
enum DimX {
  kDimX0 = 16,
  kDimX1 = 32,
  kDimX2 = 32,
};

enum DimY {
  kDimY0 = 16,
  kDimY1 = 4,
  kDimY2 = 8,
};

enum ShiftX {
  kShiftX0 = 4,
  kShiftX1 = 5,
  kShiftX2 = 5,
};

enum ShiftY {
  kShiftY0 = 4,
  kShiftY1 = 2,
  kShiftY2 = 3,
};

enum BlockConfiguration0 {
  kBlockDimX0 = kDimX1,
  kBlockDimY0 = kDimY1,
  kBlockShiftX0 = kShiftX1,
  kBlockShiftY0 = kShiftY1,
};

enum BlockConfiguration1 {
  kBlockDimX1 = kDimX2,
  kBlockDimY1 = kDimY2,
  kBlockShiftX1 = kShiftX2,
  kBlockShiftY1 = kShiftY2,
};

/*
 * rounding up total / grain, where gian = (1 << bits).
 */
inline
int divideUp(int total, int grain, int shift) {
  return (total + grain - 1) >> shift;
}

inline
int divideUp(int total, int shift) {
  return (total + ((1 << shift) - 1)) >> shift;
}

inline
int divideUp1(int total, int shift) {
  return (total + (1 << (shift - 1))) >> shift;
}

inline
int roundUp(int total, int grain, int shift) {
  return ((total + grain - 1) >> shift) << shift;
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_UTILITY_HPP_
