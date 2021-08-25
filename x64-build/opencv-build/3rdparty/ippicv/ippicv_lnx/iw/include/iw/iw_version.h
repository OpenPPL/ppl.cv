/*
// Copyright 2016-2018 Intel Corporation All Rights Reserved.
//
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title
// to such Material remains with Intel Corporation or its suppliers or
// licensors. The Material contains proprietary information of Intel
// or its suppliers and licensors. The Material is protected by worldwide
// copyright laws and treaty provisions. No part of the Material may be used,
// copied, reproduced, modified, published, uploaded, posted, transmitted,
// distributed or disclosed in any way without Intel's prior express written
// permission. No license under any patent, copyright or other intellectual
// property rights in the Material is granted to or conferred upon you,
// either expressly, by implication, inducement, estoppel or otherwise.
// Any license under such intellectual property rights must be express and
// approved by Intel in writing.
//
// Unless otherwise agreed by Intel in writing,
// you may not remove or alter this notice or any other notice embedded in
// Materials by Intel or Intel's suppliers or licensors in any way.
//
*/

#if !defined( __IPP_IW_VERSION__ )
#define __IPP_IW_VERSION__

#include "ippversion.h"

// Intel IPP IW version, equal to target Intel IPP package version
#define IW_VERSION_MAJOR  2019
#define IW_VERSION_MINOR  0
#define IW_VERSION_UPDATE 0

#define IW_VERSION_STR "2019.0.0 Beta Update 1"

// Version of minimal compatible Intel IPP package
#define IW_MIN_COMPATIBLE_IPP_MAJOR  2017
#define IW_MIN_COMPATIBLE_IPP_MINOR  0
#define IW_MIN_COMPATIBLE_IPP_UPDATE 0

// Versions converted to single digits for comparison (e.g.: 20170101)
#define IPP_VERSION_COMPLEX           (IPP_VERSION_MAJOR*10000 + IPP_VERSION_MINOR*100 + IPP_VERSION_UPDATE)
#define IW_VERSION_COMPLEX            (IW_VERSION_MAJOR*10000 + IW_VERSION_MINOR*100 + IW_VERSION_UPDATE)
#define IW_MIN_COMPATIBLE_IPP_COMPLEX (IW_MIN_COMPATIBLE_IPP_MAJOR*10000 + IW_MIN_COMPATIBLE_IPP_MINOR*100 + IW_MIN_COMPATIBLE_IPP_UPDATE)

#endif
