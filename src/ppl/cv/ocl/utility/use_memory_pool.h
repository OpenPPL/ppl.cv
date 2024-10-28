#ifndef _ST_HPC_PPL_CV_OCL_USE_MEMORY_POOL_H_
#define _ST_HPC_PPL_CV_OCL_USE_MEMORY_POOL_H_

#include "memory_pool.h"

namespace ppl {
namespace cv {
namespace ocl {

bool memoryPoolUsed();

void pplOclMalloc(GpuMemoryBlock &memory_block, size_t size);

void pplOclMallocPitch(GpuMemoryBlock &memory_block, size_t width, size_t height);

void pplOclFree(GpuMemoryBlock &memory_block);

}  // namespace ocl
}  // namespace cv
}  // namespace ppl

#endif  // _ST_HPC_PPL_CV_OCL_USE_MEMORY_POOL_H_
