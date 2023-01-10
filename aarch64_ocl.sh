#!/bin/bash

mkdir aarch64_ocl-build
cd aarch64_ocl-build

cmd="cmake .. \
      -DPPLCV_HOLD_DEPS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=install \
      -DWITH_CUDA=OFF \
      -DCMAKE_TOOLCHAIN_FILE='/opt/toolchains/android-ndk-r24/build/cmake/android.toolchain.cmake' \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_NDK='/opt/toolchains/android-ndk-r24' \
      -DCMAKE_ANDROID_NDK='/opt/toolchains/android-ndk-r24' \
      -DANDROID_NATIVE_API_LEVEL=android-18 \
      -DBUILD_ANDROID_PROJECTS=OFF \
      -DBUILD_ANDROID_EXAMPLES=OFF \
      -DPPLCV_HOLD_DEPS=ON \
      -DCL_TARGET_OPENCL_VERSION=220 \
      -DPPLCV_USE_AARCH64=ON \
      -DPPLCV_USE_OPENCL=ON \
      -DPPLCV_BUILD_TESTS=ON \
      -DPPLCV_BUILD_BENCHMARK=ON \
      -DPPLCV_OPENCL_INCLUDE_DIRS='/opt/toolchains/android-toolchain-aarch64/include/OpenCL' \
      -DPPLCV_OPENCL_LIBRARIES='/opt/toolchains/android-toolchain-aarch64/lib64/OpenCL/mali/t860/libGLES_mali.so'"
echo "cmd -> $cmd"
eval $cmd

cmake --build . -j 8 --config Release --target install

      # -DPPLCV_OPENCL_LIBRARIES='/opt/toolchains/android-toolchain-aarch64/lib64/OpenCL/qualcomm/libOpenCL.so'"
