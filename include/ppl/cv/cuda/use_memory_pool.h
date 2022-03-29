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

#ifndef _ST_HPC_PPL_CV_CUDA_USE_MEMORY_POOL_EXTERNAL_H_
#define _ST_HPC_PPL_CV_CUDA_USE_MEMORY_POOL_EXTERNAL_H_

#include <cstdio>

namespace ppl {
namespace cv {
namespace cuda {

void activateGpuMemoryPool(size_t size);

void shutDownGpuMemoryPool();

size_t ceil1DVolume(size_t volume);

size_t ceil2DVolume(size_t width, size_t height);

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_USE_MEMORY_POOL_EXTERNAL_H_