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

#if !defined( __IPP_IWLL__ )
#define __IPP_IWLL__

extern "C" {
IW_DECL(IppStatus) llwiCopySplit(const void *pSrc, int srcStep, void* const pDstOrig[], int dstStep,
                                   IppiSize size, int typeSize, int channels, int partial);
IW_DECL(IppStatus) llwiCopyMerge(const void* const pSrc[], int srcStep, void *pDst, int dstStep,
                                   IppiSize size, int typeSize, int channels, int partial);
IW_DECL(IppStatus) llwiCopyChannel(const void *pSrc, int srcStep, int srcChannels, int srcChannel, void *pDst, int dstStep,
    int dstChannels, int dstChannel, IppiSize size, int typeSize);
IW_DECL(IppStatus) llwiCopyMask(const void *pSrc, int srcStep, void *pDst, int dstStep,
    IppiSize size, int typeSize, int channels, const Ipp8u *pMask, int maskStep);
IW_DECL(IppStatus) llwiSet(const double *pValue, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels);
IW_DECL(IppStatus) llwiSetMask(const double *pValue, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels, const Ipp8u *pMask, int maskStep);
IW_DECL(IppStatus) llwiCopyMakeBorder(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep,
    IwiSize size, IppDataType dataType, int channels, IwiBorderSize borderSize, IwiBorderType border, const Ipp64f *pBorderVal);
}

#endif
