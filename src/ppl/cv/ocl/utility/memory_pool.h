#ifndef _ST_HPC_PPL_CV_OCL_MEMORY_POOL_H_
#define _ST_HPC_PPL_CV_OCL_MEMORY_POOL_H_

#include <forward_list>
#include <mutex>
#include <memory>

#include "ppl/common/ocl/pplopencl.h"

namespace ppl {
namespace cv {
namespace ocl {

#define PITCH_GRANULARITY 512
#define PITCH_SHIFT 9
#define ROUNDUP(total, grain, shift) (((total + grain - 1) >> shift) << shift)

struct GpuMemoryBlock {
  size_t offset;
  size_t pitch;
  size_t size;
  cl_mem data=nullptr;
};

class GpuMemoryPool {
 public:
  GpuMemoryPool();
  GpuMemoryPool(const GpuMemoryPool&) = delete;
  GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;
  ~GpuMemoryPool();

  bool isActivated() const {
    return (pool != nullptr);
  }
  void mallocMemoryPool(size_t size);
  void freeMemoryPool();
  void malloc1DBlock(GpuMemoryBlock &memory_block, size_t size);
  void malloc2DBlock(GpuMemoryBlock &memory_block, size_t width, size_t height);
  void freeMemoryBlock(GpuMemoryBlock &memory_block);

 private:
  size_t begin_;
  size_t end_;
  cl_mem pool;
  std::forward_list<GpuMemoryBlock> memory_blocks_;
  std::mutex host_mutex_;
};

}
}
}

#endif  // _ST_HPC_PPL_CV_OCL_MEMORY_POOL_H_