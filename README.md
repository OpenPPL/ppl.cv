## What is ppl.cv

*ppl.cv* comes from the image processing demand of different teams in sensetime, and provides a set of high performance implementations of commonly adopted image algorithms, which are used in the pipeline of different deep learning applications.

It is a light-weighted, customizable framework for image processing. Suffered from the large volume and complex dependency of implementation in frameworks like *OpenCV*, we wish to provide a flexible framework to satisfy a variety of image processing applications with only the needed algorithms when deep learning kits are developped and/ or deployed. With *ppl.cv*, developers can add a new platform to support a new kind of hardware and/ or a new implementation for a new image processing algorithm to support a new applications with easy, users can select image processing implementations for their specified platforms and algorithms to produce a small and compact image processing library for deployment.

It is a high-performance library of image processing for deep learning. We have already filled this framework with several supported platforms and a substantial number of carefully selected functions to fulfil image processing in deep learning pipeline. Since vision-based deep learning comes with massive images, each function has been optimized in terms of hardware to gain a good speedup. To facilitate the use of *ppl.cv*, there are a brief invocation example in function document, an unittest to check consistency with its counterpart in *OpenCV* and a benchmark to compare performance with its counterpart in *OpenCV*, in addition to an implementation, for each function.

### What We Target

`Functions in ppl.cv are aligned with those in OpenCV.` In the industry of computer vision, *OpenCV* is the most popular library which has been used by professional people around the world for many years, and the implementations in *OpenCV* comply with precise mathematical principle in algorithm, so we salute it. To cut down the study cost, each function in *ppl.cv* is aligned with its counterpart in *OpenCV* in terms of what the function does and its interface. Instead of an abstract data type to represent an image in *OpenCV*, we adopt a combination of several low level C data types describing a 2D pixel array which are data pointer, height, width and row stride, to easy compatibility among different hardwares and their corresponding programming languages. In order to get a good precision, we prefer to realize the mathematical principles in algorithms and have implemented them in *ppl.cv* as possible as we can. Considering there are hardware and software differences of platforms in float point computation, the outputs of the vast majority of functions in *ppl.cv* are consistent with that of CPU functions in *OpenCV* which strictly adhere to the mathematical principles or approximate them with integer quantization.

`Functions in ppl.cv are self-contained.` Given that the huge deep learning requirements of image processing and the complexity of porting *OpenCV* functions for application deployment, *ppl.cv* aims at providing self-contained implementations for each function and a tailorable image processing library. Each function has its own declaration, document, implementation and independent or limitedly shared unittest and benchmark. We decouple functions by eliminating the dependency between them and reduce the dependence on other definitions by maintaining a minimum common infrastructure for each platform. Developers and users can add or remove platforms/functions at your own will, so *ppl.cv* is friendly to development and deployment.

`Functions in ppl.cv pursue ultimate performance.` Through our theoretical analysis and experimental tests, it has been found that computations of most functions are memory bound while a few are compute bound. In regard to each specific hardware platform, deep optimization has been made in both memory access and computing. For memory access, it improves performance by adopting smaller data structures that meet the requirements, reducing memory allocation and release, address alignment, cache-friendly memory access, vector loading and storage, etc. For calculation, it uses instruction parallelism, fixed point quantization, compressing operation, and replacing double point operation with float point operation while keeping accuracy. Compared with that in *OpenCV*, each function in *ppl.cv* attains better speedup.

`ppl.cv works in coordination with ppl.nn.` As one step of the pipeline of deep learning applications, *ppl.cv* has an agreement with ppl.nn in image interface, they can works effectively together.

### Supported Platforms

For now, *ppl.cv* supports several popular platforms for mainstream desktop and mobile CPU/GPUs, including x86, CUDA, aarch64, risc-v, OpenCL.

### Supported Algorithms

For now, the image processing algorithms *ppl.cv* covers are arithmetic operation, color space conversion, histograms, filtering, morphology, image pyramid sampling, image scaling and transformation, etc. In the future, we are ready to add some new algorithms related to image decoding and VSLAM(Visual Simultaneous Localization And Mapping).

### How To Quickly Build

This project is configured by cmake scripts. A simple build script is provided.

#### Linux

```bash
$ git clone https://github.com/openppl-public/ppl.cv.git
$ cd ppl.cv
$ ./build.sh x86_64                         # for linux-x86_64
$ ./build.sh aarch64                        # for linux-aarch64
$ ./build.sh cuda                           # for linux-x86_64_cuda
$ ./build.sh riscv                          # for linux-riscv
```

#### Windows

Using vs2015 for example:

```
build.bat -G "Visual Studio 14 2015 Win64" -DPPLCV_USE_X86_64=ON
```

Please see the following guides for more detail.

### Documents

Due to the difference of hardware and programming language in platforms, documents are classified according to platform. The aspects in documents cover prerequisites, code building, unittest, benchmark, adding a function, library customization, etc. Please pick up your desired document in the following list for your development or usage.

* X86
  - [X86 Platform Guide](docs/x86_usage.md)
* CUDA
  - [CUDA Platform Guide](docs/cuda_usage.md)
  - [CUDA Memory Pool](docs/cuda_memory_pool.md)
  - [Benchmark](docs/cuda_benchmark.md)
* Arm
  - [Aarch64 Platform Guide](docs/aarch64_usage.md)
* RISCV
  - [RISCV Platform Guide](docs/riscv_usage.md)
  - [Benchmark](docs/riscv_benchmark.md)
* OpenCL
  - [OpenCL Platform Guide](docs/ocl_usage.md)
  - [Benchmark](docs/ocl_benchmark.md)

### Documentation

`ppl.cv` uses `doxygen` to generate API docs and examples in html format:

```bash
doxygen docs/Doxyfile
```

then open `html/index.html` in your web browser.

### Contact Us

Questions, reports, and suggestions are welcome through GitHub Issues!

| WeChat Official Account | QQ Group |
| :----:| :----: |
| OpenPPL | 627853444 |
| ![OpenPPL](docs/images/wechat_qrcode.jpg)| ![QQGroup](docs/images/qq_qrcode.jpg) |

### Contributing

*ppl.cv* is an open source project, any contribution would be greatly appreciated. Please report bugs by submitting a github issue without any hesitation. If you want to contribute new functions, please follow the procedure below:

* Open an github issue, let's discuss what you would like to contribute.
* Have a look at `How to add a function` in the document of your desired platform.
* Submit a pull requests for your code.

### Acknowledgements

* [OpenCV](https://github.com/opencv/opencv)

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
