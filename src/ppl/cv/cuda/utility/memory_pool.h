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

#ifndef _ST_HPC_PPL_CV_CUDA_MEMORY_POOL_H_
#define _ST_HPC_PPL_CV_CUDA_MEMORY_POOL_H_

#include <forward_list>
#include <mutex>

#include "cuda_runtime.h"

namespace ppl {
namespace cv {
namespace cuda {

#define PITCH_GRANULARITY 512
#define PITCH_SHIFT 9
#define ROUNDUP(total, grain, shift) (((total + grain - 1) >> shift) << shift)

struct GpuMemoryBlock {
  unsigned char* data;
  size_t pitch;
  size_t size;
};

class GpuMemoryPool {
 public:
  GpuMemoryPool();
  GpuMemoryPool(const GpuMemoryPool&) = delete;
  GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;
  ~GpuMemoryPool();

  bool isActivated() const {
    return (begin_ != nullptr);
  }
  void mallocMemoryPool(size_t size);
  void freeMemoryPool();
  void malloc1DBlock(size_t size, GpuMemoryBlock &memory_block);
  void malloc2DBlock(size_t width, size_t height,
                     GpuMemoryBlock &memory_block);
  void freeMemoryBlock(GpuMemoryBlock &memory_block);

 private:
  unsigned char* begin_;
  unsigned char* end_;
  std::forward_list<GpuMemoryBlock> memory_blocks_;
  std::mutex host_mutex_;
};

}  // namespace cuda
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_CUDA_MEMORY_POOL_H_
