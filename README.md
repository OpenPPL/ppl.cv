# ppl.cv

*ppl.cv* is a high-performance image processing library of openPPL supporting x86, aarch64 and cuda platforms.

### How To Build

This project is configured by cmake scripts. A simple build script is provided.

#### Linux

```bash
$ cd ${path_to_ppl.cv}
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

### Documentation

`ppl.cv` uses `doxygen` to generate API docs and examples in html format:

```bash
doxygen docs/Doxyfile
```

then open `html/index.html` in your web browser.

### Documents
* [CUDA Platform Guide](docs/cuda_usage.md)
* [RISCV Platform Guide](docs/riscv_usage.md)
* [CUDA Memory Pool](docs/cuda_memory_pool.md)

### Contact Us

* [OpenPPL](https://openppl.ai/)
* [Github issues](https://github.com/openppl-public/ppl.cv/issues)

### Contributions

This project uses [Contributor Covenant](https://www.contributor-covenant.org/) as code of conduct. Any contributions would be highly appreciated.

### Acknowledgements

* [OpenCV](https://github.com/opencv/opencv)

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
