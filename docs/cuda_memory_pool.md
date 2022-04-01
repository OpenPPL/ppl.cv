## CUDA Memory Pool

Since intermediate result between kernels needs global memory to store data temporarily in some of ppl.cv.cuda functions, cudaMalloc()/cudaMallocPitch()/cudaFree(), which consume quite a few time, must be used in function implementation. When this kind of functions are repeatedly invoked for image processing, iterative operation of global memory management hurts function performance. We create CUDA Memory Pool to manage the use of global memory operation, which separates global memory management from function execution, to impove the performance of this kind of functions. Its goal is 'one memory allocation and freeing, multiple time of usages'.

It is a optional utility component for performance in ppl.cv.cuda. If a function is invoked only once, it is not recommended.


### 1. User guide

If you just use ppl.cv.cuda functions in your projects, the following procedure should be complied with.

#### (1). Determination of CUDA Memory Pool use and memory requirement

In document of each function, whether CUDA Memory Pool is needed and how to calculate the memory volumn are marked. This can be checked. Every time a function exits, it returns memory to CUDA Memory Pool, so the total memory requirement of several functions is the max volumn in all functions when multiple functions using CUDA Memory Pool are invoked. There are 2 APIs in use_memory_pool.h which can be used to align minimum memory volume of CUDA Memory Pool as following:

`size_t ceil1DVolume(size_t volume);`

`size_t ceil2DVolume(size_t width, size_t height);`

#### (2). Activating CUDA Memory Pool

If CUDA Memory Pool is needed, it must be activated by including its header file and invoking the activating function in code. The header file is 'ppl/cv/cuda/use_memory_pool.h' and the activating function in it is declared as following:

`void activateGpuMemoryPool(size_t size);`

#### (3). Invocation of ppl.cv.cuda function

There is no difference between the invocation of ppl.cv.cuda function using CUDA Memory Pool and that of ppl.cv.cuda function not using CUDA Memory Pool, so each function is invoked as normal.

#### (4). Shutting down CUDA Memory Pool

When CUDA Memory Pool is useless, it must be shut down by invoking the inactivating function in code. The inactivating function is declared as following:

`void shutDownGpuMemoryPool();`

A code snippet demonstrating use of AdaptiveThreshold() and BoxFilter() with CUDA Memory Pool is something like this:

```
#include "ppl/cv/cuda/adaptivethreshold.h"
#include "ppl/cv/cuda/boxfilter.h"
#include "ppl/cv/cuda/use_memory_pool.h"

  ...
  size_t ceiled_volume = ppl::cv::cuda::ceil2DVolume(1920 * sizeof(float), 1080);
  ppl::cv::cuda::activateGpuMemoryPool(ceiled_volume);

  for (int i = 0; i < 1000; i++) {
    ppl::cv::cuda::AdaptiveThreshold();
    ppl::cv::cuda::BoxFilter();
    ...
  }

  ppl::cv::cuda::shutDownGpuMemoryPool();
  ...
```


### 2. Developer guide

If you want to add a new function which needs global memory to store intermediate result to ppl.cv.cuda, the following procedure should be complied with.

Our strategy for using CUDA Memory Pool is to minimize global memory occupation. There is a rule to be obeyed. *allocate memory blocks from CUDA Memory Pool when needed, return memory blocks to CUDA Memory Pool immediately when useless*. Namely, allocating and freeing memory blocks should be done in ppl.cv.cuda function definition.

#### (1). Including CUDA Memory Pool header file

The functions used by function development is defined in 'utility/use_memory_pool.h', which should be included in function implementation.

#### (2). Checking activation of CUDA Memory Pool

The following function declared in 'utility/use_memory_pool.h' detects whether CUDA Memory Pool is activated, and should be used to make a check before using CUDA Memory Pool developping APIs.

`bool memoryPoolUsed();`

#### (3). Allocation memory from CUDA Memory Pool

CUDA Memory Pool uses some data type and functions to manage memory blocks allocated in it. *GpuMemoryBlock* is defined in 'utility/memory_pool.h' and keeps information about global memory blocks allocated in CUDA Memory Pool. There are two memory allocting APIs in CUDA Memory Pool, which correspind to CUDA runtime APIs for allocting 1D arrays and allocting 2D arrays respectively. When a memory area is needed to store intermediate result, a *GpuMemoryBlock* variable should be declared, then *pplCudaMalloc()* or *pplCudaMallocPitch()* should be used to allocate a new memory block from CUDA Memory Pool.

`void pplCudaMalloc(size_t size, GpuMemoryBlock &memory_block);`

`void pplCudaMallocPitch(size_t width, size_t height, GpuMemoryBlock &memory_block);`

#### (4). Kernel definition and invocation

In kernel definition, *GpuMemoryBlock* variables should be passed as a parameter to indicate a memory buffer, and used in kernel to calculate data address. In kernel invocation, allocated memory blocks indicated by *GpuMemoryBlock* variables should be passed to kernel invocation.

#### (5). Return memory to CUDA Memory Pool

When allocated memory blocks is useless, they should be returned to CUDA Memory Pool by the following API declared in 'utility/use_memory_pool.h'.

`void pplCudaFree(GpuMemoryBlock &memory_block);`

A code snippet demonstrating use of CUDA Memory Pool in function implementation is something like this:

```
#include "utility/use_memory_pool.h"

__global__
void kernel(const float* buffer, int pitch) {
  ...
  float* data = (float*)((uchar*)buffer + row_index * pitch);
  ...
}

Function() {
  ...
  if (memoryPoolUsed()) {
    GpuMemoryBlock buffer_block;
    pplCudaMallocPitch(640 * sizeof(float), 480, buffer_block);

    kernel<<<>>>((float*)(buffer_block.data), buffer_block.pitch);
    ...

    pplCudaFree(buffer_block);
  }

  ...
}
```
