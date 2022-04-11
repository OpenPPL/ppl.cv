## RISCV Platform Guide

### AllWinner D1

Download c906 toolchain package(v2.2.4) from [https://occ.t-head.cn/community/download?id=3996672928124047360](https://occ.t-head.cn/community/download?id=3996672928124047360).

``` shell
tar -xf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.4-20211227.tar.gz
export RISCV_ROOT_PATH=/path/to/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.2.4-20211227
```

If you just want ppl.cv binary libary to link, then run the following command in the root directory of ppl.cv.

``` shell
./build.sh riscv -DHPCC_TOOLCHAIN_DIR=$RISCV_ROOT_PATH
```

This builds the ppl.cv static library, and packages the header files, the binary library and other relevant files together for usage.

If what you want to build includes not only the static library but also the executable unit test and benchmark, then run the following command in the root directory of ppl.cv.

``` shell
./build.sh riscv -DHPCC_TOOLCHAIN_DIR=$RISCV_ROOT_PATH -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON
```

Besides the static library, the executable program files of ppl.cv unittest and benchmark will be generated and the location looks like this:

```
ppl.cv/riscv-build/bin/
  pplcv_benchmark
  pplcv_unittest
```