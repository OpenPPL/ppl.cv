#!/bin/bash

workdir=`pwd`
x86_64_build_dir="${workdir}/x86-64-build"
cuda_build_dir="${workdir}/cuda-build"
aarch64_build_dir="${workdir}/aarch64-build"
riscv_build_dir="${workdir}/riscv-build"
ocl_build_dir="${workdir}/ocl-build"

if [[ `uname` == "Linux" ]]; then
    processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`
elif [[ `uname` == "Darwin" ]]; then
    processor_num=`sysctl machdep.cpu | grep machdep.cpu.core_count | cut -d " " -f 2`
else
    processor_num=1
fi

build_type='Release'
options="-DCMAKE_BUILD_TYPE=${build_type}"

# --------------------------------------------------------------------------- #

function BuildCuda() {
    mkdir ${cuda_build_dir}
    cd ${cuda_build_dir}
    cmd="cmake $options -DPPLCV_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${cuda_build_dir}/install .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    echo "cmd -> $cmd"
    eval "$cmd"
}

function BuildX86_64() {
    mkdir ${x86_64_build_dir}
    cd ${x86_64_build_dir}
    cmd="cmake $options -DPPLCV_USE_X86_64=ON -DCMAKE_INSTALL_PREFIX=${x86_64_build_dir}/install .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    echo "cmd -> $cmd"
    eval "$cmd"
}

function BuildAarch64() {
    arch=$(uname -m)
    case "$arch" in
        "x86_64")
            extra_options="-DCMAKE_TOOLCHAIN_FILE=${workdir}/cmake/toolchains/aarch64-linux-gnu.cmake"
            ;;
        "aarch64")
            ;;
        *)
            echo "unsupported arch -> $arch"
            exit 1
            ;;
    esac

    mkdir ${aarch64_build_dir}
    cd ${aarch64_build_dir}
    cmd="cmake $options ${extra_options} -DPPLCV_USE_AARCH64=ON -DCMAKE_INSTALL_PREFIX=${aarch64_build_dir}/install .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    echo "cmd -> $cmd"
    eval "$cmd"
}

function BuildRiscv() {
    arch=$(uname -m)
    case "$arch" in
        "x86_64")
            extra_options="-DCMAKE_TOOLCHAIN_FILE=${workdir}/cmake/toolchains/riscv64-rvv071-c906-linux-gnu.cmake"
            ;;
        *)
            echo "unsupported arch -> $arch"
            exit 1
            ;;
    esac

    mkdir ${riscv_build_dir}
    cd ${riscv_build_dir}
    cmd="cmake $options ${extra_options} -DPPLCV_USE_RISCV64=ON -DPPLCOMMON_ENABLE_PYTHON_API=OFF .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    echo "cmd -> $cmd"
    eval "$cmd"
}

function BuildOcl() {
    if [ $# -eq 3 ]; then
        ocl_build_dir="${workdir}/x86_ocl-build"
        mkdir ${ocl_build_dir}
        cd ${ocl_build_dir}
        cmd="cmake $options -DPPLCV_USE_X86_64=ON -DPPLCV_USE_OPENCL=ON -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON -DCMAKE_INSTALL_PREFIX=${ocl_build_dir}/install -DWITH_CUDA=OFF -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF $1 $2 $3 .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    elif [ $# -eq 6 ]; then
        ocl_build_dir="${workdir}/aarch64_ocl-build"
        mkdir ${ocl_build_dir}
        cd ${ocl_build_dir}
        cmd="cmake $options -DPPLCV_USE_AARCH64=ON -DPPLCV_USE_OPENCL=ON -DPPLCV_BUILD_TESTS=ON -DPPLCV_BUILD_BENCHMARK=ON -DCMAKE_INSTALL_PREFIX=${ocl_build_dir}/install -DWITH_CUDA=OFF $1 $2 $3 $4 -DANDROID_ABI=arm64-v8a  -DANDROID_NATIVE_API_LEVEL=android-18 -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF $5 $6 .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    else
        echo "unsupported opencl configuration."
        exit 1
    fi

    echo "cmd -> $cmd"
    eval "$cmd"
}

declare -A engine2func=(
    ["cuda"]=BuildCuda
    ["x86_64"]=BuildX86_64
    ["aarch64"]=BuildAarch64
    ["riscv"]=BuildRiscv
    ["ocl"]=BuildOcl
)

# --------------------------------------------------------------------------- #

function Usage() {
    echo -n "[INFO] usage: $0 [ all"
    for engine in ${!engine2func[@]}; do
        echo -n " | $engine"
    done
    echo "] [cmake options]"
}

if [ $# -lt 1 ]; then
    Usage
    exit 1
fi
engine="$1"

shift
options="$options $*"

if [ "$engine" == "all" ]; then
    for engine in "${!engine2func[@]}"; do
        func=${engine2func[$engine]}
        if [engine == "ocl"]; then
            continue
        else
            eval $func
        fi
        if [ $? -ne 0 ]; then
            echo "[ERROR] build [$engine] failed." >&2
            exit 1
        fi
    done
else
    func=${engine2func["$engine"]}
    if ! [ -z "$func" ]; then
        eval $func "$1" "$2" "$3" "$4" "$5" "$6"
        if [ $? -ne 0 ]; then
            echo "[ERROR] build [$engine] failed." >&2
            exit 1
        fi
    else
        echo "[ERROR] unknown engine name [$engine]" >&2
        Usage
        exit 1
    fi
fi
