## OpenCL Platform Guide

### 1. OpenCL kernel compilation using ppl.common.ocl

ppl.cv.ocl use kernels compilation infrastructure provided by ppl.common.ocl to speed up kernel compilation during programs execution. These infrastructures include a binary kernel pool, OpenCL frame chain management and an executable querying supported OpenCL properties on hardware. The first caches each compiled binary kernel in a kernel pool when kernel compilation is invoked to eliminate repeated compilation when the kernel will be used again. This works transparently, without user intervention. The second speeds up kernel compilation by managing OpenCL platform, device, context, command queue, program, kerenl, etc. This serves in 2 ways as illustrated in the following code snippet. One uses the command queue passed by user, and the other uses a shared frame chain including a command queue and cached other OpenCL components, and is suggested. The third can be used to query what OpenCL properties are supported on the target GPU before OpenCL application development. The executable is named as oclgpuinfo, and located in ppl.cv/x86_ocl-build(/aarch64_ocl-build)/bin/ after compilation.

```
  ppl::common::ocl::FrameChain frame_chain(queue);
  cl_command_queue queue = frame_chain.getQueue();
  ppl::cv::ocl::Abs(queue, ...);

```

```
  ppl::common::ocl::createSharedFrameChain(false);
  cl_command_queue queue = ppl::common::ocl::getSharedFrameChain()->getQueue();
  ppl::cv::ocl::Abs(queue, ...);
```


### 2. Building commands on linux

ppl.cv.ocl supports x86_64 linux and aarch64 android OSs, and is configured and tested on nvidia/qualcomm adreno/arm mali GPUs. Since it uses standard OpenCL grammar, it can be configured and runs on other GPUs supporting OpenCL, such as intel/amd gpu in theory. To configure and run the compilation command, the OpenCL version, the directory of OpenCL headers, the directory of OpenCL libraries, the android ndk and the tool chain for cross compilation must be specified. Here is 3 command examples needed to be run in the root directory of ppl.cv to generate ppl.cv binary libary for nvidia/qualcomm adreno/arm mali GPUs separately.

`$ ./build.sh ocl -DCL_TARGET_OPENCL_VERSION=120 -DPPLCV_OPENCL_INCLUDE_DIRS='/usr/include' -DPPLCV_OPENCL_LIBRARIES='/usr/local/cuda-10.0/lib64/libOpenCL.so'`

`$ ./build.sh ocl -DCL_TARGET_OPENCL_VERSION=220 -DCMAKE_TOOLCHAIN_FILE='/opt/toolchains/android-ndk-r24/build/cmake/android.toolchain.cmake' -DANDROID_NDK='/opt/toolchains/android-ndk-r24' -DCMAKE_ANDROID_NDK='/opt/toolchains/android-ndk-r24' -DPPLCV_OPENCL_INCLUDE_DIRS='/opt/toolchains/android-toolchain-aarch64/include/OpenCL' -DPPLCV_OPENCL_LIBRARIES='/opt/toolchains/android-toolchain-aarch64/lib64/OpenCL/qualcomm/libOpenCL.so'`

`$ ./build.sh ocl -DCL_TARGET_OPENCL_VERSION=220 -DCMAKE_TOOLCHAIN_FILE='/opt/toolchains/android-ndk-r24/build/cmake/android.toolchain.cmake' -DANDROID_NDK='/opt/toolchains/android-ndk-r24' -DCMAKE_ANDROID_NDK='/opt/toolchains/android-ndk-r24' -DPPLCV_OPENCL_INCLUDE_DIRS='/opt/toolchains/android-toolchain-aarch64/include/OpenCL' -DPPLCV_OPENCL_LIBRARIES='/opt/toolchains/android-toolchain-aarch64/lib64/OpenCL/mali/t860/libGLES_mali.so'`

These commands build the ppl.cv static library, and package the header files, the binary library and other relevant files together for usage. The generated directories and files look something like this:

```
ppl.cv/x86_ocl-build(/aarch64_ocl-build)/install/
  include/ppl/cv/ocl/
    abs.h
    ...
  lib/
    libpplcv_static.a
    ...
```

If what you want to build includes not only the static library but also the executable unit test and benchmark, then run the following command in the root directory of ppl.cv.

`$ ./build.sh ocl -DCL_TARGET_OPENCL_VERSION=120 -DPPLCV_OPENCL_INCLUDE_DIRS='/usr/include' -DPPLCV_OPENCL_LIBRARIES='/usr/local/cuda-10.0/lib64/libOpenCL.so' -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON`

`$ ./build.sh ocl -DCL_TARGET_OPENCL_VERSION=220 -DCMAKE_TOOLCHAIN_FILE='/opt/toolchains/android-ndk-r24/build/cmake/android.toolchain.cmake' -DANDROID_NDK='/opt/toolchains/android-ndk-r24' -DCMAKE_ANDROID_NDK='/opt/toolchains/android-ndk-r24' -DPPLCV_OPENCL_INCLUDE_DIRS='/opt/toolchains/android-toolchain-aarch64/include/OpenCL' -DPPLCV_OPENCL_LIBRARIES='/opt/toolchains/android-toolchain-aarch64/lib64/OpenCL/qualcomm/libOpenCL.so' -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON`

`$ ./build.sh ocl -DCL_TARGET_OPENCL_VERSION=220 -DCMAKE_TOOLCHAIN_FILE='/opt/toolchains/android-ndk-r24/build/cmake/android.toolchain.cmake' -DANDROID_NDK='/opt/toolchains/android-ndk-r24' -DCMAKE_ANDROID_NDK='/opt/toolchains/android-ndk-r24' -DPPLCV_OPENCL_INCLUDE_DIRS='/opt/toolchains/android-toolchain-aarch64/include/OpenCL' -DPPLCV_OPENCL_LIBRARIES='/opt/toolchains/android-toolchain-aarch64/lib64/OpenCL/mali/t860/libGLES_mali.so' -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON`

Besides the static library, the executable program files of ppl.cv unittest and benchmark will be generated and the location looks like this:

```
ppl.cv/x86_ocl-build(/aarch64_ocl-build)/bin/
  pplcv_benchmark
  pplcv_unittest
  oclgpuinfo
```


### 3. How to run unittest

The executable unittest includes unit tests for all functions on all platforms, which check the consistency between the implementation in ppl.cv and that in opencv opencv x86/aarch64. Our unittest is based on GoogleTest, and use regular expression to identify function unit tests. To run all the unit tests of all function in ppl.cv.ocl, the following commands is needed:

`$ ./pplcv_unittest --gtest_filter=*PplCvOcl*`

To run the unit test of a particular function, a regular express consisting of 'PplCvOcl' and the function name is needed. For example, the command to run the unit test of Abs() is as following:

`$ ./pplcv_unittest --gtest_filter=*PplCvOclAbs*`

The output of a unit test case is formatted with the arguments passed to its function, So each test case shows both the execution status and the function arguments. When a case fails, the input arguments of the function can be easily determined.

![Output snippet of Abs unittest](./abs_unittest.png)


### 4. How to run benchmark

The executable benchmark exhibits performance of all ppl.cv functions on all platforms, also shows performance comparison between the implementation in ppl.cv and that in opencv x86/aarch64. Our benchmark is based on Google Benchmark, and use regular expression to identify functions. To run all benchmarks of all function in ppl.cv.ocl, the following commands is needed:

`$ ./pplcv_benchmark --benchmark_filter="BM_.+ocl"`

To run the benchmark of a particular function, a regular express consisting of 'BM_.+ocl' and the function name is needed. For example, the command to run the benchmark of Abs() is as following:

`$ ./pplcv_benchmark --benchmark_filter="BM_Abs.+ocl"`

The output of a benchmark is also formatted with the arguments passed to its function, So each benchmark case shows both the execution time and the function arguments. Since manual timing is adopted for GPU in Google Benchmark, so the *Time* column is the real time of function execution on GPU.

![Output snippet of Abs benchmark](./abs_benchmark.png)


### 5. Library customization and tailoring

ppl.cv.ocl targets small volume and flexibility. Each function normally has five files, including a *.h file for function declaration and document, a *.cpp file for function implementation, a *.cl file for kernel difinition, a *_unittest.cpp file for unit test and a *_benchmark.cpp file for performance exhibition. Besides very limited invocation between functions, there is not dependency between functions. In 'ppl/cv/src/ppl/cv/ocl/utility' folder, function utility, unit test infrastructure and performance benchmark infrastructure are defined for each function. In order to create a customized OpenCL cv library from ppl.cv.ocl, the utility files and the files of needed functions are just needed to be kept.

For example, a customization library, which only has CopyMakeBorder(), has the following files.

```
ppl/cv/
  include/ppl/cv/ocl/
    copymakeborder.h
  src/ppl/cv/ocl/
    copymakeborder.cpp
    copymakeborder.cl
    copymakeborder_unittest.cpp
    copymakeborder_benchmark.cpp
    utility/
      (all files under this directory)
  ...

```
