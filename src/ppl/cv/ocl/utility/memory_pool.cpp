#include "memory_pool.h"

#include "ppl/common/log.h"

using namespace ppl::common;
using namespace ppl::common::ocl;

namespace ppl {
namespace cv {
namespace ocl {

GpuMemoryPool::GpuMemoryPool() {
  begin_ = 0;
  end_   = 0;
  pool   = nullptr;
}

GpuMemoryPool::~GpuMemoryPool() {
  if (pool != nullptr) {
    if (!memory_blocks_.empty()) {
      LOG(ERROR) << "OpenCL Memory Pool error: allocated memory blocks have to be released first.";
    }
    cl_int code = clReleaseMemObject(pool);
    CHECK_ERROR(code, clReleaseMemObject);
  }
}

void GpuMemoryPool::mallocMemoryPool(size_t size) {
  cl_int code;
  pool = clCreateBuffer(getSharedFrameChain()->getContext(),
                        CL_MEM_READ_WRITE, size, nullptr, &code);
  CHECK_ERROR(code, clCreateBuffer);
  end_ = begin_ + size;
}

void GpuMemoryPool::freeMemoryPool() {
  if (pool != nullptr) {
    if (!memory_blocks_.empty()) {
      LOG(ERROR) << "OpenCL Memory Pool error: allocated memory blocks have to be released first.";
    }
    cl_int code = clReleaseMemObject(pool);
    CHECK_ERROR(code, clReleaseMemObject);
    begin_ = 0;
    end_   = 0;
    pool   = nullptr;
  }
}

void GpuMemoryPool::malloc1DBlock(GpuMemoryBlock &memory_block, 
                                  size_t size) {
  size_t allocated_size = ROUNDUP(size, PITCH_GRANULARITY, PITCH_SHIFT);
  host_mutex_.lock();
  if (memory_blocks_.empty()) {
    if (begin_ + allocated_size <= end_) {
      memory_block.offset  = begin_;
      memory_block.pitch = 0;
      memory_block.size  = allocated_size;
      memory_block.data = pool;
      
      auto current = memory_blocks_.before_begin();
      memory_blocks_.insert_after(current, memory_block);
      host_mutex_.unlock();
    }
    else {
      host_mutex_.unlock();
      LOG(ERROR) << "OpenCL Memory Pool error: insufficient space.";
    }

    return;
  }

  auto previous = memory_blocks_.begin();
  auto current  = memory_blocks_.begin();
  ++current;
  size_t hollow_begin;
  if (current == memory_blocks_.end()) {
    hollow_begin = previous->offset + previous->size;
    if (hollow_begin + allocated_size <= end_) {
      memory_block.offset  = hollow_begin;
      memory_block.pitch = 0;
      memory_block.size  = allocated_size;
      memory_block.data = pool;

      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();
    }
    else {
      host_mutex_.unlock();
      LOG(ERROR) << "OpenCL Memory Pool error: insufficient space.";
    }

    return;
  }

  while (current != memory_blocks_.end()) {
    hollow_begin = previous->offset + previous->size;
    if (hollow_begin + allocated_size <= current->offset) {
      memory_block.offset  = hollow_begin;
      memory_block.pitch = 0;
      memory_block.size  = allocated_size;
      memory_block.data = pool;

      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();

      return;
    }

    ++previous;
    ++current;
  }

  hollow_begin = previous->offset + previous->size;
  if (hollow_begin + allocated_size <= end_) {
    memory_block.offset  = hollow_begin;
    memory_block.pitch = 0;
    memory_block.size  = allocated_size;
    memory_block.data = pool;

    memory_blocks_.insert_after(previous, memory_block);
    host_mutex_.unlock();
  }
  else {
    host_mutex_.unlock();
    LOG(ERROR) << "OpenCL Memory Pool error: insufficient space.";
  }
}

void GpuMemoryPool::malloc2DBlock(GpuMemoryBlock &memory_block, size_t width, size_t height) {
  size_t block_pitch = ROUNDUP(width, PITCH_GRANULARITY, PITCH_SHIFT);
  size_t block_size  = block_pitch * height;

  host_mutex_.lock();
  if (memory_blocks_.empty()) {
    if (begin_ + block_size <= end_) {
      memory_block.offset  = begin_;
      memory_block.pitch = block_pitch;
      memory_block.size  = block_size;
      memory_block.data = pool;

      auto current = memory_blocks_.before_begin();
      memory_blocks_.insert_after(current, memory_block);
      host_mutex_.unlock();
    }
    else {
      host_mutex_.unlock();
      LOG(ERROR) << "OpenCL Memory Pool error: insufficient space.";
    }

    return;
  }

  auto previous = memory_blocks_.begin();
  auto current  = memory_blocks_.begin();
  ++current;
  size_t hollow_begin;
  if (current == memory_blocks_.end()) {
    hollow_begin = previous->offset + previous->size;
    if (hollow_begin + block_size <= end_) {
      memory_block.offset  = hollow_begin;
      memory_block.pitch = block_pitch;
      memory_block.size  = block_size;
      memory_block.data = pool;

      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();
    }
    else {
      host_mutex_.unlock();
      LOG(ERROR) << "OpenCL Memory Pool error: insufficient space.";
    }

    return;
  }

  while (current != memory_blocks_.end()) {
    hollow_begin = previous->offset + previous->size;
    if (hollow_begin + block_size <= current->offset) {
      memory_block.offset  = hollow_begin;
      memory_block.pitch = block_pitch;
      memory_block.size  = block_size;
      memory_block.data = pool;

      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();

      return;
    }

    ++previous;
    ++current;
  }

  hollow_begin = previous->offset + previous->size;
  if (hollow_begin + block_size <= end_) {
    memory_block.offset  = hollow_begin;
    memory_block.pitch = block_pitch;
    memory_block.size  = block_size;
    memory_block.data = pool;

    memory_blocks_.insert_after(previous, memory_block);
    host_mutex_.unlock();
  }
  else {
    host_mutex_.unlock();
    LOG(ERROR) << "OpenCL Memory Pool error: insufficient space.";
  }
}

void GpuMemoryPool::freeMemoryBlock(GpuMemoryBlock &memory_block) {
  if (memory_blocks_.empty()) {
    LOG(ERROR) << "OpenCL Memory Pool error: empty pool can't contain a block.";

    return;
  }

  auto previous = memory_blocks_.before_begin();
  auto current  = memory_blocks_.begin();
  while (current != memory_blocks_.end()) {
    if ((current->offset == memory_block.offset) &&
        (current->size == memory_block.size)) {
      memory_block.data = nullptr;
      
      host_mutex_.lock();
      memory_blocks_.erase_after(previous);
      host_mutex_.unlock();
      break;
    }

    ++previous;
    ++current;
  }

  if (current == memory_blocks_.end()) {
    LOG(ERROR) << "OpenCL Memory Pool error: can't not find the memory block.";
  }
}

}  // namespace ocl
}  // namespace cv
}  // namespace ppl
