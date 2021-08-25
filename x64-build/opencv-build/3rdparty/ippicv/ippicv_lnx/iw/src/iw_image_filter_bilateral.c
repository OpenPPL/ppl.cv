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

IW_DECL(IppStatus) llwiFilterBilateral(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep, IwiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal);

#if IW_ENABLE_THREADING_LAYER
IW_DECL(IppStatus) llwiFilterBilateral_TL(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep, IwiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal);
#endif

IW_DECL(IppStatus) llwiFilterBilateral_classic(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterBilateral
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterBilateral(const IwiImage *pSrcImage, IwiImage *pDstImage, int radius,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, const IwiFilterBilateralParams *pAuxParams,
    IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus                status;
    IwiFilterBilateralParams auxParams;

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
        iwiFilterBilateral_SetDefaultParams(&auxParams);

#if IW_ENABLE_THREADING_LAYER
    if(iwGetThreadsNum() > 1)
    {
        IwiSize size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);
        return llwiFilterBilateral_TL(pSrcImage->m_ptr, pSrcImage->m_step, pDstImage->m_ptr, pDstImage->m_step,
            size, pSrcImage->m_dataType, pSrcImage->m_channels, auxParams.filter, radius, auxParams.distMethod, valSquareSigma, posSquareSigma, border, pBorderVal);
    }
    else
#endif
    {
        const void *pSrc = pSrcImage->m_ptrConst;
        void       *pDst = pDstImage->m_ptr;
        IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(OWN_GET_PURE_BORDER(border) == ippBorderWrap)
                return ippStsNotSupportedModeErr;

            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi         dstRoi = pTile->m_dstRoi;
                IwiBorderSize  borderSize = iwiSizeSymToBorderSize(radius*2);

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;
                owniTile_CorrectBordersOverlap(&dstRoi, &size, &border, &borderSize, &borderSize, &pSrcImage->m_size);
                owniTile_GetTileBorder(&border, &dstRoi, &borderSize, &pSrcImage->m_size);

                pSrc = iwiImage_GetPtrConst(pSrcImage, dstRoi.y, dstRoi.x, 0);
                pDst = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi srcLim;
                IwiRoi dstLim;
                iwiTilePipeline_GetBoundedSrcRoi(pTile, &srcLim);
                iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pSrc   = iwiImage_GetPtrConst(pSrcImage, srcLim.y, srcLim.x, 0);
                pDst   = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);
                iwiTilePipeline_GetTileBorder(pTile, &border);

                size = owniGetMinSizeFromRect(&srcLim, &dstLim);
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        if(pSrcImage->m_dataType == ipp32f)
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

            return llwiFilterBilateral_classic(pSrc, (int)pSrcImage->m_step, pDst, (int)pDstImage->m_step,
                _size, pSrcImage->m_dataType, pSrcImage->m_channels, auxParams.filter, radius, auxParams.distMethod, valSquareSigma, posSquareSigma, border, pBorderVal);
        }
        else
        {
            return llwiFilterBilateral(pSrc, pSrcImage->m_step, pDst, pDstImage->m_step,
                size, pSrcImage->m_dataType, pSrcImage->m_channels, auxParams.filter, radius, auxParams.distMethod, valSquareSigma, posSquareSigma, border, pBorderVal);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilterBilateral(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep, IwiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;
    Ipp64f    borderVal[4] = {0};

    IppiFilterBilateralSpec *pSpec       = 0;
    IwSize                   specSize    = 0;

    Ipp8u    *pTmpBuffer    = 0;
    IwSize    tmpBufferSize = 0;

    for(;;)
    {
        status = ippiFilterBilateralBorderGetBufferSize_L(filter, size, radius, dataType, channels, distMethod, &specSize, &tmpBufferSize);
        if(status < 0)
            break;

        pSpec = (IppiFilterBilateralSpec*)ownSharedMalloc(specSize);
        if(!pSpec)
        {
            status = ippStsNoMemErr;
            break;
        }

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        status = ippiFilterBilateralBorderInit_L(filter, size, radius, dataType, channels, distMethod, valSquareSigma, posSquareSigma, pSpec);
        if(status < 0)
            break;

        switch(dataType)
        {
        case ipp8u:
            switch(channels)
            {
            case 1:  status = ippiFilterBilateralBorder_8u_C1R_L((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBilateralBorder_8u_C3R_L((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 3), pSpec, pTmpBuffer); break;
            default: status = ippStsNumChannelsErr; break;
            }
            break;
        default: status = ippStsDataTypeErr; break;
        }
        break;
    }

    if(pSpec)
        ownSharedFree(pSpec);
    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}

#if IW_ENABLE_THREADING_LAYER
IW_DECL(IppStatus) llwiFilterBilateral_TL(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep, IwiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;
    Ipp64f    borderVal[4] = {0};

    IppiFilterBilateralSpec_LT *pSpec       = 0;
    IwSize                      specSize    = 0;

    Ipp8u    *pTmpBuffer    = 0;
    IwSize    tmpBufferSize = 0;

    for(;;)
    {
        status = ippiFilterBilateralBorderGetBufferSize_LT(filter, size, radius, dataType, channels, distMethod, &specSize, &tmpBufferSize);
        if(status < 0)
            break;

        pSpec = (IppiFilterBilateralSpec_LT*)ownSharedMalloc(specSize);
        if(!pSpec)
        {
            status = ippStsNoMemErr;
            break;
        }

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        status = ippiFilterBilateralBorderInit_LT(filter, size, radius, dataType, channels, distMethod, valSquareSigma, posSquareSigma, pSpec);
        if(status < 0)
            break;

        switch(dataType)
        {
        case ipp8u:
            switch(channels)
            {
            case 1:  status = ippiFilterBilateralBorder_8u_C1R_LT((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBilateralBorder_8u_C3R_LT((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(8u, 3), pSpec, pTmpBuffer); break;
            default: status = ippStsNumChannelsErr; break;
            }
            break;
        default: status = ippStsDataTypeErr; break;
        }
        break;
    }

    if(pSpec)
        ownSharedFree(pSpec);
    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}
#endif

IW_DECL(IppStatus) llwiFilterBilateral_classic(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size,
    IppDataType dataType, int channels, IppiFilterBilateralType filter, int radius, IppiDistanceMethodType distMethod,
    Ipp32f valSquareSigma, Ipp32f posSquareSigma, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;
    Ipp64f    borderVal[4] = {0};

    IppiFilterBilateralSpec *pSpec       = 0;
    int                      specSize    = 0;

    Ipp8u    *pTmpBuffer    = 0;
    int       tmpBufferSize = 0;

    for(;;)
    {

        status = ippiFilterBilateralBorderGetBufferSize(filter, size, radius, dataType, channels, distMethod, &specSize, &tmpBufferSize);
        if(status < 0)
            break;

        pSpec = (IppiFilterBilateralSpec*)ownSharedMalloc(specSize);
        if(!pSpec)
        {
            status = ippStsNoMemErr;
            break;
        }

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        status = ippiFilterBilateralBorderInit(filter, size, radius, dataType, channels, distMethod, valSquareSigma, posSquareSigma, pSpec);
        if(status < 0)
            break;

        switch(dataType)
        {
        case ipp32f:
            switch(channels)
            {
            case 1:  status = ippiFilterBilateralBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(32f, 1), pSpec, pTmpBuffer); break;
            case 3:  status = ippiFilterBilateralBorder_32f_C3R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, border, OWN_GET_BORDER_VALP(32f, 3), pSpec, pTmpBuffer); break;
            default: status = ippStsNumChannelsErr; break;
            }
            break;
        default: status = ippStsDataTypeErr; break;
        }
        break;
    }

    if(pSpec)
        ownSharedFree(pSpec);
    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}
