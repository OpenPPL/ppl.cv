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

#ifndef _ST_HPC_PPL_CV_OCL_INFRASTRUCTURE_H_
#define _ST_HPC_PPL_CV_OCL_INFRASTRUCTURE_H_

#include <iostream>

#include "opencv2/core.hpp"

#include "ppl/common/log.h"

#define EPSILON_1F 1.1f
#define EPSILON_2F 2.1f
#define EPSILON_3F 3.1f
#define EPSILON_4F 4.1f
#define EPSILON_E1 1e-1
#define EPSILON_E2 1e-2
#define EPSILON_E3 0.002
#define EPSILON_E4 1e-4
#define EPSILON_E5 1e-5
#define EPSILON_E6 1e-6

#define AUX_ASSERT(expression)                                                 \
if (!(expression)) {                                                           \
  LOG(ERROR) << "Infrastructure assertion failed: " << #expression;            \
  exit(-1);                                                                    \
}

cv::Mat createSourceImage(int rows, int cols, int type);
cv::Mat createSourceImage(int rows, int cols, int type, float begin,
                          float end);
cv::Mat createBinaryImage(int rows, int cols, int type);

template <typename T>
void copyMatToArray(const cv::Mat& image0, T* image1);

template <typename T>
bool checkMatricesIdentity(const T* src0, int rows, int cols, int channels,
                           int src0_stride, const T* src1, int src1_stride,
                           float epsilon, bool display = false);

#endif  // _ST_HPC_PPL_CV_OCL_INFRASTRUCTURE_H_
