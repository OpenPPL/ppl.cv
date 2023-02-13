## CUDA Platform Guide

### 1. Prerequisites

* Linux or Windows running on x86_64
* CUDA toolkit >= 7.0
* gcc/g++ >= 4.9
* Visual Studio >= 2015
* cmake >= 3.14
* Git >= 2.7.0

### 2. How to build from source on linux

If you just want *ppl.cv* binary libary to link, then run the following command in the root directory of *ppl.cv*.

`$ ./build.sh cuda`

This builds the *ppl.cv* static library, and packages the header files, the binary library and other relevant files together for usage. The generated directories and files look something like this:

```
ppl.cv/cuda-build/install/
  bin/
  include/ppl/cv/cuda/
    abs.h
    ...
  lib/
    libpplcv_static.a
    ...
  share/
```

If what you want to build includes not only the static library but also the executable unit test and benchmark, then run the following command in the root directory of *ppl.cv*.

`$ ./build.sh cuda -DPPLCV_USE_X86_64=ON -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON`

Besides the static library, the executable program files of *ppl.cv* unittest and benchmark will be generated and the location looks like this:

```
ppl.cv/cuda-build/bin/
  pplcv_benchmark
  pplcv_unittest
```

### 3. How to build from source on windows

Similar to compiling and linking on linux, script and commands to invoke Microsoft Visual Studio are used to build *ppl.cv*. For now, "Visual Studio 2015" and "Visual Studio 2019" are supported and tested. If you just want *ppl.cv* binary libary to link, then run the following command in the root directory of *ppl.cv*.

`$ ./build.bat -G "Visual Studio 14 2015 Win64" -DPPLCV_USE_CUDA=ON`

`$ ./build.bat -G "Visual Studio 16 2019" -A x64 -DPPLCV_USE_CUDA=ON`

The generated directories and files look something like this:

```
ppl.cv/cuda-build/install/
  etc/
  include/ppl/cv/cuda/
    abs.h
    ...
  lib/
    libpplcv_static.lib
    ...
  x64/
```

If what you want to build includes not only the static library but also the executable unit test and benchmark, then run the following command in the root directory of *ppl.cv*.

`$ ./build.bat -G "Visual Studio 14 2015 Win64" -DPPLCV_USE_X86_64=ON -DPPLCV_USE_CUDA=ON -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON`

`$ ./build.bat -G "Visual Studio 16 2019" -A x64 -DPPLCV_USE_X86_64=ON -DPPLCV_USE_CUDA=ON -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON`

Besides directories of the header files and the static library, the executable unittest and benchmark are located as:

```
ppl.cv/cuda-build/bin/Release/
  pplcv_benchmark.exe
  pplcv_unittest.exe
  ...
```

### 4. How to run unittest

The executable unittest includes unit tests for all functions on all platforms, which check the consistency between the implementation in *ppl.cv* and that in opencv of functions. Our unittest is based on GoogleTest, and use regular expression to identify function unit tests. To run all the unit tests of all function in *ppl.cv.cuda*, the following commands is needed:

`$ ./pplcv_unittest --gtest_filter=*PplCvCuda*`

To run the unit test of a particular function, a regular express consisting of 'PplCvCuda' and the function name is needed. For example, the command to run the unit test of GaussianBlur() is as following:

`$ ./pplcv_unittest --gtest_filter=*PplCvCudaGaussianBlur*`

The output of a unit test case is formatted with the arguments passed to its function, So each test case shows both the execution status and the function arguments. When a case fails, the input arguments of the function can be easily determined.

![Output snippet of GaussianBlur unittest](images/cuda_gaussianblur_unittest.png)

### 5. How to run benchmark

The executable benchmark exhibits performance of all *ppl.cv* functions on all platforms, also shows performance comparison between the implementation in *ppl.cv* and that in opencv x86 and opencv cuda. Our benchmark is based on Google Benchmark, and use regular expression to identify functions. To run all benchmarks of all function in *ppl.cv.cuda*, the following commands is needed:

`$ ./pplcv_benchmark --benchmark_filter="BM_.+cuda"`

To run the benchmark of a particular function, a regular express consisting of 'BM_.+cuda' and the function name is needed. For example, the command to run the benchmark of GaussianBlur() is as following:

`$ ./pplcv_benchmark --benchmark_filter="BM_GaussianBlur.+cuda"`

The output of a benchmark is also formatted with the arguments passed to its function, So each benchmark case shows both the execution time and the function arguments. Since manual timing is adopted for GPU in Google Benchmark, so the *Time* column is the real time of function execution on GPU.

![Output snippet of GaussianBlur benchmark](images/cuda_gaussianblur_benchmark.png)

### 6. How to use a function

There is a brief document coming with the interface in `include/ppl/cv/cuda/xxx.h` for each function. What it does, supported data types, supported channels, introduction of parameters, return value and other notices are provided. A example code snippet is also offered to show how to invoke this function in your application. Please refer to its document before you use a function.

### 7. How to add a function

There are some conventions made by the cmake building system of *ppl.cv* that should be abided by when a new function is added. There are at least four files for a function definition as listed below where their file names have a common prefix(xxx).

* include/ppl/cv/cuda/xxx.h: A prototype declaration and a brief introduction of the interface and usage example should be given here.
* src/ppl/cv/cuda/xxx.cu: All things about implementation, including macros, kernel definitions, device function definitions, thread configuration, kernel invocation and host functions definitions, should be located here.
* src/ppl/cv/cuda/xxx_unittest.cpp: An unittest based on *GoogleTest* covering thorough parameter combination in usage cases should be provided here to compare the outputs with its counterpart in *OpenCV* for consistency.
* src/ppl/cv/cuda/xxx_benchmark.cpp: A benchmark based on *Google Benchmark* covering common usage cases should be provided here to compare performance with its counterpart in *OpenCV* to validate the implemented optimization.

Some common infrastructure in *ppl.cv* can facilitate development. Firstly, some enumerations for image processing algorithm are given in `include/ppl/cv/types.h`, and can be used in the interface and implementation of a function. Secondly, error checking, type definitiond, enumeration of thread configuration and commonly used device functions are provided in `src/ppl/cv/cuda/utility/utility.hpp` and can be used in function implementation. Thirdly, [CUDA Memory Pool](docs/cuda_memory_pool.md) is provided for memory allocation and freeing as a utility component, and can be used to cut down memory management cost in function implementation when needed. Fourthly, infrastructures for creating different input images and checking consistency in unittest/benchmark are provided in `src/ppl/cv/cuda/utility/infrastructure.hpp`, and can be used in writing a unittest/benchmark.

### 8. How to customize this library

*ppl.cv* targets small volume and flexibility. Each function normally has four files, including a xxx.h file for function declaration and document, a xxx.cu file for function implementation, a xxx_unittest.cpp file for unit test and a xxx_benchmark.cpp file for performance exhibition. Besides very limited invocation between functions, there is not dependency between functions. In 'ppl/cv/src/ppl/cv/cuda/utility' folder, function utility, cuda memory pool, unit test infrastructure and performance benchmark infrastructure are defined for each function. In order to create a customized cuda cv library from *ppl.cv.cuda*, the utility files and the files of needed functions are just needed to be kept.

For example, a customization library, which only has Adaptivethreshold(), has the following files.

```
ppl/cv/
  include/ppl/cv/cuda/
    adaptivethreshold.h
    use_memory_pool.h
  src/ppl/cv/cuda/
    adaptivethreshold.cu
    adaptivethreshold_unittest.cpp
    adaptivethreshold_benchmark.cpp
    utility/
      (all files under this directory)
  ...

```
