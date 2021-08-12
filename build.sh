#!/bin/bash

workdir=`pwd`
x86_build_dir="${workdir}/x86-build"
cuda_build_dir="${workdir}/cuda-build"
aarch64_build_dir="${workdir}/aarch64-build"
deps_dir="${workdir}/deps"
cpu_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`

options='-DCMAKE_BUILD_TYPE=Release'

# --------------------------------------------------------------------------- #

function BuildCuda() {
    mkdir ${cuda_build_dir}
    cd ${cuda_build_dir}
    cmd="cmake $options -DHPCC_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${cuda_build_dir}/install .. && make -j${cpu_num} && make install"
    echo "cmd -> $cmd"
    eval "$cmd"
}

function BuildX86() {
    mkdir ${x86_build_dir}
    cd ${x86_build_dir}
    cmd="cmake $options -DHPCC_USE_X86=ON -DCMAKE_INSTALL_PREFIX=${x86_build_dir}/install .. && make -j${cpu_num} && make install"
    echo "cmd -> $cmd"
    eval "$cmd"
}

function BuildAarch64() {
    arch=$(uname -m)
    case "$arch" in
        "x86_64")
            extra_options="-DCMAKE_TOOLCHAIN_FILE=${deps_dir}/hpcc/cmake/toolchains/aarch64-linux-gnu.cmake"
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
    cmd="cmake $options ${extra_options} -DHPCC_USE_AARCH64=ON -DPPLCOMMON_ENABLE_PYTHON_API=OFF -DCMAKE_INSTALL_PREFIX=${aarch64_build_dir}/install .. && make -j${cpu_num} && make install"
    echo "cmd -> $cmd"
    eval "$cmd"
}

declare -A engine2func=(
    ["cuda"]=BuildCuda
    ["x86"]=BuildX86
    ["aarch64"]=BuildAarch64
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
        eval $func
        if [ $? -ne 0 ]; then
            echo "[ERROR] build [$engine] failed." >&2
            exit 1
        fi
    done
else
    func=${engine2func["$engine"]}
    if ! [ -z "$func" ]; then
        eval $func
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
