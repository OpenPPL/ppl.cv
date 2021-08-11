# ppl.cv
*ppl.cv* is a high-performance image processing library of openPPL supporting x86 and cuda platforms.

### Source Tree Structure
```
┌────────────────────────────────────────────────────┐
│ directory or file              explanation         │
└────────────────────────────────────────────────────┘
 ppl.cv
  ├── .vscode                    Visual code settings
  ├── assets                     Asset files, such as pictures for testing
  ├── cmake                      CMake script
  │     ├─ x86.cmake             CMake configuration script for x86 architecture.
  │     ├─ cuda.cmake            CMake configuration script for cuda architecture.
  │     ├─ opencv.cmake          CMake configuration script for OpenCV.
  │     ├─ benchmark.cmake       CMake configuration script for benchmarks.
  │     ├─ unittest.cmake        CMake configuration script for unit tests .
  │     └─ deps.cmake            CMake configuration script for third-party dependencies.
  ├── src                        Source code directory
  │    └─ ppl
  │        └─ cv                 Common headers and source files are placed here
  │           ├─ cuda            All cuda source files are placed here
  │           ├─ x86             All x86 source files are placed here
  ├── .gitignore
  ├── CMakeLists.txt
  ├── README.md
  └── build.sh                   Shell scripts for building ppl.cv projects.
```
### How To Build
This project is configured by cmake script. A simple build script is provided.
```sh
$ cd ${path_to_ppl.cv}
$ ./build.sh x86                            # for linux-x86_64
$ ./build.sh cuda                           # for linux-x86_64_cuda
```
### Contact Us

* [OpenPPL](https://openppl.ai/)
* [Github issues](https://github.com/openppl-public/ppl.cv/issues)

### Contributions
This project uses [Contributor Covenant](https://www.contributor-covenant.org/) as code of conduct. Any contributions would be highly appreciated.

### Acknowledgements
* [OpenCV](https://github.com/opencv/opencv)

### License
This project is distributed under the [Apache License, Version 2.0](LICENSE).

