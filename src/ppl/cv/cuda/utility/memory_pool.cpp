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

#include "ppl/common/log.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

GpuMemoryPool::GpuMemoryPool() {
  begin_ = nullptr;
  end_   = nullptr;
}

GpuMemoryPool::~GpuMemoryPool() {
  if (begin_ != nullptr) {
    cudaError_t code = cudaFree(begin_);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    }
    memory_blocks_.clear();
  }
}

void GpuMemoryPool::mallocMemoryPool(size_t size) {
  cudaError_t code = cudaMalloc((void**)&begin_, size);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
  }

  end_ = begin_ + size;
}

void GpuMemoryPool::freeMemoryPool() {
  if (begin_ != nullptr) {
    cudaError_t code = cudaFree(begin_);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    }

    begin_ = nullptr;
    end_   = nullptr;
  }
}

void GpuMemoryPool::malloc1DBlock(size_t size, GpuMemoryBlock &memory_block) {
  size_t allocated_size = ROUNDUP(size, PITCH_GRANULARITY, PITCH_SHIFT);
  if (memory_blocks_.empty()) {
    if (begin_ + allocated_size <= end_) {
      memory_block.data  = begin_;
      memory_block.pitch = 0;
      memory_block.size  = allocated_size;

      host_mutex_.lock();
      auto current = memory_blocks_.before_begin();
      memory_blocks_.insert_after(current, memory_block);
      host_mutex_.unlock();
    }
    else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  auto previous = memory_blocks_.begin();
  auto current  = memory_blocks_.begin();
  ++current;
  unsigned char* hollow_begin;
  if (current == memory_blocks_.end()) {
    hollow_begin = previous->data + previous->size;
    if (hollow_begin + allocated_size <= end_) {
      memory_block.data  = hollow_begin;
      memory_block.pitch = 0;
      memory_block.size  = allocated_size;

      host_mutex_.lock();
      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();
    }
    else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  while (current != memory_blocks_.end()) {
    hollow_begin = previous->data + previous->size;
    if (hollow_begin + allocated_size <= current->data) {
      memory_block.data  = hollow_begin;
      memory_block.pitch = 0;
      memory_block.size  = allocated_size;

      host_mutex_.lock();
      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();

      return;
    }

    ++previous;
    ++current;
  }

  hollow_begin = previous->data + previous->size;
  if (hollow_begin + allocated_size <= end_) {
    memory_block.data  = hollow_begin;
    memory_block.pitch = 0;
    memory_block.size  = allocated_size;

    host_mutex_.lock();
    memory_blocks_.insert_after(previous, memory_block);
    host_mutex_.unlock();
  }
  else {
    LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
  }
}

void GpuMemoryPool::malloc2DBlock(size_t width, size_t height,
                                  GpuMemoryBlock &memory_block) {
  size_t block_pitch = ROUNDUP(width, PITCH_GRANULARITY, PITCH_SHIFT);
  size_t block_size  = block_pitch * height;

  if (memory_blocks_.empty()) {
    if (begin_ + block_size <= end_) {
      memory_block.data  = begin_;
      memory_block.pitch = block_pitch;
      memory_block.size  = block_size;

      host_mutex_.lock();
      auto current = memory_blocks_.before_begin();
      memory_blocks_.insert_after(current, memory_block);
      host_mutex_.unlock();
    }
    else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  auto previous = memory_blocks_.begin();
  auto current  = memory_blocks_.begin();
  ++current;
  unsigned char* hollow_begin;
  if (current == memory_blocks_.end()) {
    hollow_begin = previous->data + previous->size;
    if (hollow_begin + block_size <= end_) {
      memory_block.data  = hollow_begin;
      memory_block.pitch = block_pitch;
      memory_block.size  = block_size;

      host_mutex_.lock();
      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();
    }
    else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  while (current != memory_blocks_.end()) {
    hollow_begin = previous->data + previous->size;
    if (hollow_begin + block_size <= current->data) {
      memory_block.data  = hollow_begin;
      memory_block.pitch = block_pitch;
      memory_block.size  = block_size;

      host_mutex_.lock();
      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();

      return;
    }

    ++previous;
    ++current;
  }

  hollow_begin = previous->data + previous->size;
  if (hollow_begin + block_size <= end_) {
    memory_block.data  = hollow_begin;
    memory_block.pitch = block_pitch;
    memory_block.size  = block_size;

    host_mutex_.lock();
    memory_blocks_.insert_after(previous, memory_block);
    host_mutex_.unlock();
  }
  else {
    LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
  }
}

void GpuMemoryPool::freeMemoryBlock(GpuMemoryBlock &memory_block) {
  if (memory_blocks_.empty()) {
    LOG(ERROR) << "Cuda Memory Pool error: empty pool can't contain a block.";

    return;
  }

  auto previous = memory_blocks_.before_begin();
  auto current  = memory_blocks_.begin();
  while (current != memory_blocks_.end()) {
    if (current->data == memory_block.data &&
        current->size == memory_block.size) {
      host_mutex_.lock();
      memory_blocks_.erase_after(previous);
      host_mutex_.unlock();
      break;
    }

    ++previous;
    ++current;
  }

  if (current == memory_blocks_.end()) {
    LOG(ERROR) << "Cuda Memory Pool error: can't not find the memory block.";
  }
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
