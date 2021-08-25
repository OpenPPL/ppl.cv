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

#include "iw/iw_image_filter.h"
#include "iw_owni.h"

IW_DECL(IppStatus) llwiCanny(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType,
                               int channels, IppiDifferentialKernel kernel, IppiMaskSize kernelSize, IppNormType norm,
                               Ipp32f treshLow, Ipp32f treshHigh, IwiBorderType border, const Ipp64f *pBorderVal);

IW_DECL(IppStatus) llwiCannyDeriv(const void *pSrcDx, IwSize srcStepDx, const void *pSrcDy, IwSize srcStepDy, IppDataType srcDataType, void *pDst, IwSize dstStep,
                               IwiSize size, IppDataType dstDataType, int channels, IppNormType norm, Ipp32f treshLow, Ipp32f treshHigh);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterCanny
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterCanny(const IwiImage *pSrcImage, IwiImage *pDstImage, Ipp32f treshLow, Ipp32f treshHigh,
    const IwiFilterCannyParams *pAuxParams, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus            status;
    IwiFilterCannyParams auxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst)
        return ippStsInplaceModeNotSupportedErr;

    if(pSrcImage->m_dataType != pDstImage->m_dataType ||
        pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if(pAuxParams)
        auxParams = *pAuxParams;
    else
        iwiFilterCanny_SetDefaultParams(&auxParams);

    {
        const void *pSrc = pSrcImage->m_ptrConst;
        void       *pDst = pDstImage->m_ptr;
        IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

        // Long compatibility check
        {
            IppiSize _size;

            status = ownLongCompatCheckValue(pSrcImage->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            return llwiCanny(pSrc, (int)pSrcImage->m_step, pDst, (int)pDstImage->m_step, _size, pSrcImage->m_dataType,
                pSrcImage->m_channels, auxParams.kernel, auxParams.kernelSize, auxParams.norm, treshLow, treshHigh, border, pBorderVal);
        }
    }
}

IW_DECL(IppStatus) iwiFilterCannyDeriv(const IwiImage *pSrcImageDx, const IwiImage *pSrcImageDy, IwiImage *pDstImage,
    Ipp32f treshLow, Ipp32f treshHigh, const IwiFilterCannyDerivParams *pAuxParams)
{
    IppStatus                 status;
    IwiFilterCannyDerivParams auxParams;

    status = owniCheckImageRead(pSrcImageDx);
    if(status)
        return status;
    status = owniCheckImageRead(pSrcImageDy);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImageDx->m_ptrConst == pDstImage->m_ptrConst || pSrcImageDy->m_ptrConst == pDstImage->m_ptrConst)
        return ippStsInplaceModeNotSupportedErr;

    if(pSrcImageDx->m_dataType != pSrcImageDy->m_dataType ||
        pSrcImageDx->m_channels != pSrcImageDy->m_channels)
        return ippStsBadArgErr;

    if(pSrcImageDx->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if(pAuxParams)
        auxParams = *pAuxParams;
    else
        iwiFilterCannyDeriv_SetDefaultParams(&auxParams);

    {
        const void *pSrcX = pSrcImageDx->m_ptrConst;
        const void *pSrcY = pSrcImageDy->m_ptrConst;
        void       *pDst  = pDstImage->m_ptr;
        IwiSize     size  = owniGetMinSize(&pSrcImageDx->m_size, &pSrcImageDy->m_size);
                            owniGetMinSize(&size, &pDstImage->m_size);

        return llwiCannyDeriv(pSrcX, pSrcImageDx->m_step, pSrcY, pSrcImageDy->m_step, pSrcImageDx->m_dataType, pDst, pDstImage->m_step, size, pDstImage->m_dataType,
            pDstImage->m_channels, auxParams.norm, treshLow, treshHigh);
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiCanny(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType,
                               int channels, IppiDifferentialKernel kernel, IppiMaskSize kernelSize, IppNormType norm,
                               Ipp32f treshLow, Ipp32f treshHigh, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;

    Ipp8u *pTmpBuffer    = 0;
    int    tmpBufferSize = 0;

    for(;;)
    {
        status = ippiCannyBorderGetSize(size, kernel, kernelSize, dataType, &tmpBufferSize);
        if(status < 0)
            break;

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        switch(dataType)
        {
#if IW_ENABLE_DATA_TYPE_8U
        case ipp8u:
            switch(channels)
            {
#if IW_ENABLE_CHANNELS_C1
            case 1:  status = ippiCannyBorder_8u_C1R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, kernel, kernelSize, border, OWN_GET_BORDER_VAL(8u), treshLow, treshHigh, norm, pTmpBuffer); break;
#endif
            default: status = ippStsNumChannelsErr; break;
            }
            break;
#endif
        default: status = ippStsDataTypeErr; break;
        }
        if(status < 0)
            break;

        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}

IW_DECL(IppStatus) llwiCannyDeriv(const void *pSrcDx, IwSize srcStepDx, const void *pSrcDy, IwSize srcStepDy, IppDataType srcDataType, void *pDst, IwSize dstStep,
                               IwiSize size, IppDataType dstDataType, int channels, IppNormType norm, Ipp32f treshLow, Ipp32f treshHigh)
{
#if IPP_VERSION_COMPLEX >= 20170002
    IppStatus status;

    Ipp8u *pTmpBuffer    = 0;
    IwSize tmpBufferSize = 0;

    for(;;)
    {
        status = ippiCannyGetSize_L(size, &tmpBufferSize);
        if(status < 0)
            break;

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        switch(srcDataType)
        {
        case ipp16s:
            switch(dstDataType)
            {
            case ipp8u:
                switch(channels)
                {
                case 1:  status = ippiCanny_16s8u_C1R_L((Ipp16s*)pSrcDx, srcStepDx, (Ipp16s*)pSrcDy, srcStepDy, (Ipp8u*)pDst, dstStep, size, treshLow, treshHigh, norm, pTmpBuffer); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
            break;
        case ipp32f:
            switch(dstDataType)
            {
            case ipp8u:
                switch(channels)
                {
#if IPP_VERSION_COMPLEX >= 20180000
                case 1:  status = ippiCanny_32f8u_C1R_L((Ipp32f*)pSrcDx, srcStepDx, (Ipp32f*)pSrcDy, srcStepDy, (Ipp8u*)pDst, dstStep, size, treshLow, treshHigh, norm, pTmpBuffer); break;
#endif
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
            break;
        default: status = ippStsDataTypeErr; break;
        }
        if(status < 0)
            break;

        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
#else
    (void)pSrcDx; (void)srcStepDx; (void)pSrcDy; (void)srcStepDy; (void)srcDataType; (void)pDst; (void)dstStep;
    (void)size; (void)dstDataType; (void)channels; (void)norm; (void)treshLow; (void)treshHigh;
    return ippStsNotSupportedModeErr;
#endif
}
