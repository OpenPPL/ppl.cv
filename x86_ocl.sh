#!/bin/bash

mkdir ocl_x86-build
cd ocl_x86-build

cmd="cmake .. \
      -DPPLCV_HOLD_DEPS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=install \
      -DWITH_CUDA=OFF \
      -DBUILD_ANDROID_PROJECTS=OFF \
      -DBUILD_ANDROID_EXAMPLES=OFF \
      -DCL_TARGET_OPENCL_VERSION=200 \
      -DPPLCV_USE_OPENCL=ON \
      -DPPLCV_BUILD_TESTS=ON \
      -DPPLCV_BUILD_BENCHMARK=ON \
      -DPPLCV_OPENCL_INCLUDE_DIRS='/usr/include' \
      -DPPLCV_OPENCL_LIBRARIES='/usr/local/cuda-10.0/lib64/libOpenCL.so'"
echo "cmd -> $cmd"
eval $cmd

cmake --build . -j 8 --config Release --target install
