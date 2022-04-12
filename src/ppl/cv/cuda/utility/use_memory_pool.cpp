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

#include "ppl/cv/cuda/use_memory_pool.h"
#include "use_memory_pool.h"
#include "memory_pool.h"

#include <memory>

#include "ppl/common/log.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

// global handle of CUDA Memory Pool.
std::unique_ptr<GpuMemoryPool> memory_pool_ptr(new GpuMemoryPool);

// user API
void activateGpuMemoryPool(size_t size) {
  if (memory_pool_ptr->isActivated()) {
    LOG(ERROR) << "CUDA Memory Pool error: It has already been activated.";
    return;
  }

  memory_pool_ptr->mallocMemoryPool(size);
}

// user API
void shutDownGpuMemoryPool() {
  if (!(memory_pool_ptr->isActivated())) {
    LOG(ERROR) << "CUDA Memory Pool error: It is not activated.";
    return;
  }

  memory_pool_ptr->freeMemoryPool();
}

size_t ceil1DVolume(size_t volume) {
  return ROUNDUP(volume, PITCH_GRANULARITY, PITCH_SHIFT);
}

size_t ceil2DVolume(size_t width, size_t height) {
  size_t ceiled_volume = ROUNDUP(width, PITCH_GRANULARITY, PITCH_SHIFT) *
                         height;

  return ceiled_volume;
}

bool memoryPoolUsed() {
  return memory_pool_ptr->isActivated();
}

void pplCudaMalloc(size_t size, GpuMemoryBlock &memory_block) {
  memory_pool_ptr->malloc1DBlock(size, memory_block);
}

void pplCudaMallocPitch(size_t width, size_t height,
                        GpuMemoryBlock &memory_block) {
  memory_pool_ptr->malloc2DBlock(width, height, memory_block);
}

void pplCudaFree(GpuMemoryBlock &memory_block) {
  memory_pool_ptr->freeMemoryBlock(memory_block);
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
