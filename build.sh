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
# build_type='RelWithDebInfo'
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
    # cmd="cmake $options -DPPLCV_USE_X86_64=ON -DCMAKE_INSTALL_PREFIX=${x86_64_build_dir}/install .. && make VERBOSE=1 && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    cmd="cmake $options -DPPLCV_USE_X86_64=ON -DCMAKE_INSTALL_PREFIX=${x86_64_build_dir}/install .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    echo "cmd -> $cmd"
    eval "$cmd"
}

function BuildAarch64() {
    arch=$(uname -m)
    case "$arch" in
        "x86_64")
            if [[ "$options" =~ .*android\.toolchain\.cmake.* ]] ; then
                extra_options="-DBUILD_ANDROID_PROJECTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF"
            else
                extra_options="-DCMAKE_TOOLCHAIN_FILE=${workdir}/cmake/toolchains/aarch64-linux-gnu.cmake"
            fi
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
    cmd="cmake -DPPLCV_USE_AARCH64=ON -DCMAKE_INSTALL_PREFIX=${aarch64_build_dir}/install $options ${extra_options} .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
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
    parameters=($options)
    length=${#parameters[*]}
    if [[ $length -eq 4 || $length -eq 6 ]] ; then
        ocl_build_dir="${workdir}/x86_ocl-build"
        mkdir ${ocl_build_dir}
        cd ${ocl_build_dir}
        cmd="cmake -DCMAKE_BUILD_TYPE=Release -DPPLCV_USE_X86_64=ON -DPPLCV_USE_OPENCL=ON -DCMAKE_INSTALL_PREFIX=${ocl_build_dir}/install -DWITH_CUDA=OFF -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF ${parameters[1]} ${parameters[2]} ${parameters[3]} ${parameters[4]} ${parameters[5]} .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
    elif [[ $length -eq 7 || $length -eq 9 ]]; then
        ocl_build_dir="${workdir}/aarch64_ocl-build"
        mkdir ${ocl_build_dir}
        cd ${ocl_build_dir}
        cmd="cmake -DCMAKE_BUILD_TYPE=Release -DPPLCV_USE_AARCH64=ON -DPPLCV_USE_OPENCL=ON -DCMAKE_INSTALL_PREFIX=${ocl_build_dir}/install -DWITH_CUDA=OFF ${parameters[1]} ${parameters[2]} ${parameters[3]} ${parameters[4]} -DANDROID_ABI=arm64-v8a  -DANDROID_NATIVE_API_LEVEL=android-18 -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF ${parameters[5]} ${parameters[6]} ${parameters[7]} ${parameters[8]} .. && cmake --build . -j ${processor_num} --config ${build_type} && cmake --build . --target install -j ${processor_num} --config ${build_type}"
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
