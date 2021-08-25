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

IW_DECL(IppStatus) llwiFilterSobel(const void *pSrc, int srcStep, IppDataType srcType, void *pDst, int dstStep, IppDataType dstType,
                                     IppiSize size, int channels, IwiDerivativeType opType, IppiMaskSize kernelSize, IwiBorderType border, const Ipp64f *pBorderVal);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterSobel
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiFilterSobel(const IwiImage *pSrcImage, IwiImage *pDstImage, IwiDerivativeType opType,
                                  IppiMaskSize kernelSize, const IwiFilterSobelParams *pAuxParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst)
        return ippStsInplaceModeNotSupportedErr;

    if(pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    {
        const void *pSrc = pSrcImage->m_ptrConst;
        void       *pDst = pDstImage->m_ptr;
        IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);
        if(!size.width || !size.height)
            return ippStsNoOperation;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            IwiImage srcSubImage = *pSrcImage;
            IwiImage dstSubImage = *pDstImage;

            if(OWN_GET_PURE_BORDER(border) == ippBorderWrap)
                return ippStsNotSupportedModeErr;

            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi         dstRoi     = pTile->m_dstRoi;
                IwiBorderSize  borderSize = iwiSizeToBorderSize(iwiMaskToSize(kernelSize));

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;
                owniTile_CorrectBordersOverlap(&dstRoi, &size, &border, &borderSize, &borderSize, &pSrcImage->m_size);
                owniTile_GetTileBorder(&border, &dstRoi, &borderSize, &pSrcImage->m_size);

                iwiImage_RoiSet(&srcSubImage, dstRoi);
                iwiImage_RoiSet(&dstSubImage, dstRoi);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                iwiImage_RoiSet(&srcSubImage, pTile->m_boundSrcRoi);
                iwiImage_RoiSet(&dstSubImage, pTile->m_boundDstRoi);

                status = owniTilePipeline_ProcBorder(pTile, &srcSubImage, &border, pBorderVal);
                if(status < 0)
                    return status;
            }
            else
                return ippStsContextMatchErr;

            return iwiFilterSobel(&srcSubImage, &dstSubImage, opType, kernelSize, pAuxParams, border, pBorderVal, NULL);
        }

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

            return llwiFilterSobel(pSrc, (int)pSrcImage->m_step, pSrcImage->m_dataType, pDst, (int)pDstImage->m_step, pDstImage->m_dataType,
                _size, pSrcImage->m_channels, opType, kernelSize, border, pBorderVal);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiFilterSobel(const void *pSrc, int srcStep, IppDataType srcType, void *pDst, int dstStep, IppDataType dstType,
                                     IppiSize size, int channels, IwiDerivativeType opType, IppiMaskSize kernelSize, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;

    Ipp8u *pTmpBuffer    = 0;
    int    tmpBufferSize = 0;

    for(;;)
    {
        switch(opType)
        {
        case iwiDerivHorFirst:   status = ippiFilterSobelHorizBorderGetBufferSize(size, kernelSize, srcType, dstType, channels, &tmpBufferSize);         break;
        case iwiDerivHorSecond:  status = ippiFilterSobelHorizSecondBorderGetBufferSize(size, kernelSize, srcType, dstType, channels, &tmpBufferSize);   break;
        case iwiDerivVerFirst:   status = ippiFilterSobelVertBorderGetBufferSize(size, kernelSize, srcType, dstType, channels, &tmpBufferSize);          break;
        case iwiDerivVerSecond:  status = ippiFilterSobelVertSecondBorderGetBufferSize(size, kernelSize, srcType, dstType, channels, &tmpBufferSize);    break;
        case iwiDerivNVerFirst:  status = ippiFilterSobelNegVertBorderGetBufferSize(size, kernelSize, srcType, dstType, channels, &tmpBufferSize);       break;
        default:                 status = ippStsNotSupportedModeErr; break;
        }
        if(status < 0)
            break;

        pTmpBuffer = (Ipp8u*)ownSharedMalloc(tmpBufferSize);
        if(tmpBufferSize && !pTmpBuffer)
        {
            status = ippStsNoMemErr;
            break;
        }

        switch(opType)
        {
        case iwiDerivHorFirst:
            if(srcType == ipp8u && dstType == ipp16s)
                status = ippiFilterSobelHorizBorder_8u16s_C1R((Ipp8u*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(8u), pTmpBuffer);
            else if(srcType == ipp16s && dstType == ipp16s)
                status = ippiFilterSobelHorizBorder_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(16s), pTmpBuffer);
            else if(srcType == ipp32f && dstType == ipp32f)
                status = ippiFilterSobelHorizBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(32f), pTmpBuffer);
            else
                status = ippStsDataTypeErr;
            break;
        case iwiDerivHorSecond:
            if(srcType == ipp8u && dstType == ipp16s)
                status = ippiFilterSobelHorizSecondBorder_8u16s_C1R((Ipp8u*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(8u), pTmpBuffer);
            else if(srcType == ipp32f && dstType == ipp32f)
                status = ippiFilterSobelHorizSecondBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(32f), pTmpBuffer);
            else
                status = ippStsDataTypeErr;
            break;
        case iwiDerivVerFirst:
            if(srcType == ipp8u && dstType == ipp16s)
                status = ippiFilterSobelVertBorder_8u16s_C1R((Ipp8u*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(8u), pTmpBuffer);
            else if(srcType == ipp16s && dstType == ipp16s)
                status = ippiFilterSobelVertBorder_16s_C1R((Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(16s), pTmpBuffer);
            else if(srcType == ipp32f && dstType == ipp32f)
                status = ippiFilterSobelVertBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(32f), pTmpBuffer);
            else
                status = ippStsDataTypeErr;
            break;
        case iwiDerivVerSecond:
            if(srcType == ipp8u && dstType == ipp16s)
                status = ippiFilterSobelVertSecondBorder_8u16s_C1R((Ipp8u*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(8u), pTmpBuffer);
            else if(srcType == ipp32f && dstType == ipp32f)
                status = ippiFilterSobelVertSecondBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(32f), pTmpBuffer);
            else
                status = ippStsDataTypeErr;
            break;
        case iwiDerivNVerFirst:
            if(srcType == ipp8u && dstType == ipp16s)
                status = ippiFilterSobelNegVertBorder_8u16s_C1R((Ipp8u*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(8u), pTmpBuffer);
            else if(srcType == ipp32f && dstType == ipp32f)
                status = ippiFilterSobelNegVertBorder_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, kernelSize, border, OWN_GET_BORDER_VAL(32f), pTmpBuffer);
            else
                status = ippStsDataTypeErr;
            break;
        default:
            status = ippStsNotSupportedModeErr; break;
        }
        if(status < 0)
            break;

        break;
    }

    if(pTmpBuffer)
        ownSharedFree(pTmpBuffer);

    return status;
}
