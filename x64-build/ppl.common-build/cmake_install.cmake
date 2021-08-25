# Install script for directory: /home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/sensetime/projects/github_opensource/ppl.cv/x64-build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ppl/common" TYPE FILE FILES
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/allocator.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/compact_memory_manager.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/file_mapping.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/generic_cpu_allocator.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/half.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/lock_utils.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/log.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/object_pool.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/retcode.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/stripfilename.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/sys.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/deps/ppl.common/src/ppl/common/types.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/sensetime/projects/github_opensource/ppl.cv/x64-build/ppl.common-build/libpplcommon_static.a")
endif()

