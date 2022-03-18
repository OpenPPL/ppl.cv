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

#include "memory_pool.h"
#include "utility.hpp"

#include "ppl/common/log.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

GpuMemoryPool::GpuMemoryPool() {
  memory_pool = nullptr;
}

GpuMemoryPool::~GpuMemoryPool() {
  if (memory_pool != nullptr) {
    cudaError_t code = cudaFree(memory_pool);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    }
    memory_blocks.clear();
  }
}

void GpuMemoryPool::mallocMemoryPool(size_t size) {
  cudaError_t code = cudaMalloc((void**)&memory_pool, size);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
  }

  capability = size;
}

void GpuMemoryPool::freeMemoryPool() {
  if (memory_pool != nullptr) {
    cudaError_t code = cudaFree(memory_pool);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    }

    capability = 0;
    memory_pool = nullptr;
  }
}

void GpuMemoryPool::malloc1DBlock(size_t size, GpuMemoryBlock &memory_block) {
  if (memory_blocks.empty()) {
    if (size <= capability) {
      memory_block.data   = memory_pool;
      memory_block.offset = 0;
      memory_block.size   = size;
      memory_block.pitch  = 0;

      host_mutex.lock();
      auto current = memory_blocks.before_begin();
      memory_blocks.insert_after(current, memory_block);
      host_mutex.unlock();
    }
    else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  auto previous = memory_blocks.begin();
  auto current  = memory_blocks.begin();
  ++current;
  if (current == memory_blocks.end()) {
    size_t hollow_begin = roundUp((previous->offset + previous->size),
                                  PITCH_GRANULARITY, PITCH_SHIFT);
    if (hollow_begin + size <= capability) {
      memory_block.data   = memory_pool;
      memory_block.offset = hollow_begin;
      memory_block.size   = size;
      memory_block.pitch  = 0;

      host_mutex.lock();
      memory_blocks.insert_after(previous, memory_block);
      host_mutex.unlock();
    }
    else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  while (current != memory_blocks.end()) {
    size_t hollow_begin = roundUp((previous->offset + previous->size),
                                  PITCH_GRANULARITY, PITCH_SHIFT);
    if (hollow_begin + size <= current->offset) {
      memory_block.data   = memory_pool;
      memory_block.offset = hollow_begin;
      memory_block.size   = size;
      memory_block.pitch  = 0;

      host_mutex.lock();
      memory_blocks.insert_after(previous, memory_block);
      host_mutex.unlock();
      break;
    }
    ++current;
  }

  if (current == memory_blocks.end()) {
    LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
  }
}

void GpuMemoryPool::malloc2DBlock(size_t width, size_t height,
                                  GpuMemoryBlock &memory_block) {
  size_t block_pitch = roundUp(width, PITCH_GRANULARITY, PITCH_SHIFT);
  size_t block_size  = block_pitch * height;

  if (memory_blocks.empty()) {
    if (block_size <= capability) {
      memory_block.data   = memory_pool;
      memory_block.offset = 0;
      memory_block.size   = block_size;
      memory_block.pitch  = block_pitch;

      host_mutex.lock();
      auto current = memory_blocks.before_begin();
      memory_blocks.insert_after(current, memory_block);
      host_mutex.unlock();
    }
    else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  auto previous = memory_blocks.begin();
  auto current  = memory_blocks.begin();
  ++current;
  if (current == memory_blocks.end()) {
    size_t hollow_begin = roundUp((previous->offset + previous->size),
                                  PITCH_GRANULARITY, PITCH_SHIFT);
    if (hollow_begin + block_size <= capability) {
      memory_block.data   = memory_pool;
      memory_block.offset = hollow_begin;
      memory_block.size   = block_size;
      memory_block.pitch  = block_pitch;

      host_mutex.lock();
      memory_blocks.insert_after(previous, memory_block);
      host_mutex.unlock();
    }
    else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  while (current != memory_blocks.end()) {
    size_t hollow_begin = roundUp((previous->offset + previous->size),
                                  PITCH_GRANULARITY, PITCH_SHIFT);
    if (hollow_begin + block_size <= current->offset) {
      memory_block.data   = memory_pool;
      memory_block.offset = hollow_begin;
      memory_block.size   = block_size;
      memory_block.pitch  = block_pitch;

      host_mutex.lock();
      memory_blocks.insert_after(previous, memory_block);
      host_mutex.unlock();
    }
    ++current;
  }

  if (current == memory_blocks.end()) {
    LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
  }
}

void GpuMemoryPool::freeMemoryBlock(GpuMemoryBlock &memory_block) {
  if (memory_blocks.empty()) {
    LOG(ERROR) << "Cuda Memory Pool error: empty pool can't contain a block.";

    return;
  }

  auto previous = memory_blocks.before_begin();
  auto current  = memory_blocks.begin();
  while (current != memory_blocks.end()) {
    if (current->offset == memory_block.offset) {
      host_mutex.lock();
      memory_blocks.erase_after(previous);
      host_mutex.unlock();
      break;
    }

    previous = current;
    ++current;
  }

  if (current == memory_blocks.end()) {
    LOG(ERROR) << "Cuda Memory Pool error: can't not find the memory block.";
  }
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
