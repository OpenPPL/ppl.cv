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
#include "kernels/adaptivethreshold.cl"
#include "kernels/boxfilter.cl"
#include "kernels/calchist.cl"
#include "kernels/convertto.cl"
#include "kernels/crop.cl"
#include "kernels/equalizehist.cl"
#include "kernels/filter2d.cl"
#include "kernels/gaussianblur.cl"
#include "kernels/integral.cl"
#include "kernels/merge.cl"
#include "kernels/rotate.cl"
#include "kernels/sepfilter2d.cl"
#include "kernels/split.cl"
#include "kernels/transpose.cl"
#include "kernels/warpperspective.cl"


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

  SET_PROGRAM_SOURCE(frame_chain, adaptivethreshold);
  frame_chain->setFunctionName("adaptivethreshold");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, boxfilter);
  frame_chain->setFunctionName("boxfilter");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, calchist);
  frame_chain->setFunctionName("calchist");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, convertto);
  frame_chain->setFunctionName("convertto");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, crop);
  frame_chain->setFunctionName("crop");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, equalizehist);
  frame_chain->setFunctionName("equalizehist");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, filter2d);
  frame_chain->setFunctionName("filter2d");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, gaussianblur);
  frame_chain->setFunctionName("gaussianblur");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, integral);
  frame_chain->setFunctionName("integral");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, merge);
  frame_chain->setFunctionName("merge");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, rotate);
  frame_chain->setFunctionName("rotate");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, sepfilter2d);
  frame_chain->setFunctionName("sepfilter2d");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, split);
  frame_chain->setFunctionName("split");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, transpose);
  frame_chain->setFunctionName("transpose");
  buildKernelBinaries();

  SET_PROGRAM_SOURCE(frame_chain, warpperspective);
  frame_chain->setFunctionName("warpperspective");
  buildKernelBinaries();

  shutDownKernelBinariesManager(BINARIES_COMPILE);

  return 0;
}
