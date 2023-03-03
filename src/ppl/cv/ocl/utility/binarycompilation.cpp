/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "ppl/common/ocl/framechain.h"
#include "ppl/common/ocl/kernelbinaries_interface.h"

#include "ppl/common/log.h"

#include "kernels/abs.cl"
#include "kernels/arithmetic.cl"
#include "kernels/copymakeborder.cl"
#include "kernels/cvtcolor.cl"
#include "kernels/dilate.cl"
#include "kernels/erode.cl"
#include "kernels/flip.cl"
#include "kernels/resize.cl"
#include "kernels/warpaffine.cl"

using namespace ppl::common;
using namespace ppl::common::ocl;

int main(int argc, char **argv) {
  createSharedFrameChain(false);
  FrameChain* frame_chain = getSharedFrameChain();
  frame_chain->setProjectName("cv");
  frame_chain->setCompileOptions("-D ALL_KERNELS");

  bool succeeded = initializeKernelBinariesManager(BINARIES_COMPILE);
  if (!succeeded) {
    LOG(ERROR) << "Failed to intialize kernel binaries manager.";

    return -1;
  }

  SET_PROGRAM_SOURCE(frame_chain, abs);
  frame_chain->setFunctionName("abs");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, arithmetic);
  frame_chain->setFunctionName("arithmetic");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, copymakeborder);
  frame_chain->setFunctionName("copymakeborder");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, cvtcolor);
  frame_chain->setFunctionName("cvtcolor");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, dilate);
  frame_chain->setFunctionName("dilate");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, erode);
  frame_chain->setFunctionName("erode");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, flip);
  frame_chain->setFunctionName("flip");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, resize);
  frame_chain->setFunctionName("resize");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, warpaffine);
  frame_chain->setFunctionName("warpaffine");
  buildKernelBinaries();

  shutDownKernelBinariesManager(BINARIES_COMPILE);

  return 0;
}
