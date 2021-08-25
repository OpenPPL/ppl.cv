# Install script for directory: /home/sensetime/projects/github_opensource/ppl.cv

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ppl/cv/x86" TYPE FILE FILES
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/addweighted.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/arithmetic.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/copymakeborder.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/cvtcolor.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/dilate.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/erode.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/flip.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/get_affine_transform.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/get_rotation_matrix2d.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/resize.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/test.h"
    "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/x86/warpaffine.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ppl/cv" TYPE FILE FILES "/home/sensetime/projects/github_opensource/ppl.cv/src/ppl/cv/types.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/sensetime/projects/github_opensource/ppl.cv/x64-build/libpplcv_static.a")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/sensetime/projects/github_opensource/ppl.cv/x64-build/ppl.common-build/cmake_install.cmake")
  include("/home/sensetime/projects/github_opensource/ppl.cv/x64-build/opencv-build/cmake_install.cmake")
  include("/home/sensetime/projects/github_opensource/ppl.cv/x64-build/googletest-build/cmake_install.cmake")
  include("/home/sensetime/projects/github_opensource/ppl.cv/x64-build/benchmark-build/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/sensetime/projects/github_opensource/ppl.cv/x64-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
