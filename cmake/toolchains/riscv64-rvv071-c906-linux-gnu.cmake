set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv)

if(NOT HPCC_TOOLCHAIN_DIR)
    set(HPCC_TOOLCHAIN_DIR "/usr")
    if(NOT EXISTS ${HPCC_TOOLCHAIN_DIR})
        message(FATAL_ERROR "`HPCC_TOOLCHAIN_DIR` not set.")
    endif()
elseif(NOT EXISTS ${HPCC_TOOLCHAIN_DIR})
    message(FATAL_ERROR "`HPCC_TOOLCHAIN_DIR`(${HPCC_TOOLCHAIN_DIR}) not found")
endif()

set(PPLCV_RISCV_RVV_0P71 ON)

set(CMAKE_C_COMPILER ${HPCC_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${HPCC_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-g++)

set(CMAKE_CXX_FLAGS "-static -march=rv64gcv0p7xtheadc -mabi=lp64d -mtune=c906 -pthread")
set(CMAKE_C_FLAGS "-static -march=rv64gcv0p7xtheadc -mabi=lp64d -mtune=c906 -pthread")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
