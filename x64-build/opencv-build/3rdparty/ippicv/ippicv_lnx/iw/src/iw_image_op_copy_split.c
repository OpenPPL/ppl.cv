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

#include "iw/iw_image_op.h"
#include "iw_owni.h"

IW_DECL(IppStatus) llwiCopySplit(const void *pSrc, int srcStep, void* const pDstOrig[], int dstStep,
                                   IppiSize size, int typeSize, int channels, int partial);

IW_DECL(IppStatus) llwiCopyChannel(const void *pSrc, int srcStep, int srcChannels, int srcChannel, void *pDst, int dstStep,
    int dstChannels, int dstChannel, IppiSize size, int typeSize);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSplitChannels
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiSplitChannels(const IwiImage *pSrcImage, IwiImage* const pDstImage[], const IwiSplitChannelsParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;

    if(!pDstImage)
        return ippStsNullPtrErr;

    if(pSrcImage->m_channels == 1)
        return iwiCopy(pSrcImage, pDstImage[0], NULL, NULL, pTile);

    if(pSrcImage->m_channels > 4)
        return ippStsNumChannelsErr;

    {
        void*       pDst[4]       = {NULL};
        IwSize      dstStep[4]    = {0};
        int         dstPixSize[4] = {0};
        const void* pSrc          = pSrcImage->m_ptrConst;
        IwiSize     size          = pSrcImage->m_size;
        int         channels      = pSrcImage->m_channels;
        int         i;

        for(i = 0; i < pSrcImage->m_channels; i++)
        {
            if(pDstImage[i] && pDstImage[i]->m_ptr)
            {
                if(pSrcImage->m_ptrConst == pDstImage[i]->m_ptrConst)
                    return ippStsInplaceModeNotSupportedErr;

                if(pSrcImage->m_typeSize != pDstImage[i]->m_typeSize)
                    return ippStsBadArgErr;

                size          = owniGetMinSize(&size, &pDstImage[i]->m_size);
                pDst[i]       = pDstImage[i]->m_ptr;
                dstStep[i]    = pDstImage[i]->m_step;
                dstPixSize[i] = pDstImage[i]->m_typeSize*pDstImage[i]->m_channels;
                if(dstStep[i] != dstStep[0])
                    return ippStsStepErr;
                if(dstPixSize[i] != dstPixSize[0])
                    return ippStsBadArgErr;
            }
            else
                channels--;
        }
        if(!size.width || !size.height)
            return ippStsNoOperation;
        if(!channels)
            return ippStsNoOperation;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                pSrc = iwiImage_GetPtrConst(pSrcImage, dstRoi.y, dstRoi.x, 0);
                owniShiftPtrArr(pDst, pDst, dstStep, dstPixSize, pSrcImage->m_channels,dstRoi.x, dstRoi.y);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi srcLim;
                IwiRoi dstLim;
                iwiTilePipeline_GetBoundedSrcRoi(pTile, &srcLim);
                iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pSrc = iwiImage_GetPtrConst(pSrcImage, srcLim.y, srcLim.x, 0);
                owniShiftPtrArr(pDst, pDst, dstStep, dstPixSize, pSrcImage->m_channels, dstLim.x, dstLim.y);

                size = owniGetMinSizeFromRect(&srcLim, &dstLim);
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        {
            IppiSize _size;

            status = ownLongCompatCheckValue(pSrcImage->m_step, NULL);
            if(status < 0)
                return status;

            status = ownLongCompatCheckValue(pDstImage[0]->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            return llwiCopySplit(pSrc, (int)pSrcImage->m_step, pDst, (int)dstStep[0], _size, pSrcImage->m_typeSize, pSrcImage->m_channels, (channels != pSrcImage->m_channels)?1:0);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiCopySplit(const void *pSrc, int srcStep, void* const pDst[], int dstStep,
                                   IppiSize size, int typeSize, int channels, int partial)
{
    if(partial)
    {
        IppStatus status = ippStsNoErr;
        int       i = 0;
        for(i = 0; i < channels; i++)
        {
            if(pDst[i])
            {
                status = llwiCopyChannel(pSrc, srcStep, channels, i, pDst[i], dstStep, 1, 0, size, typeSize);
                if(status < 0)
                    return status;
            }
        }
        return status;
    }
    else
    {
        switch(typeSize)
        {
        case 1:
            switch(channels)
            {
            case 3:  return ippiCopy_8u_C3P3R((const Ipp8u*)pSrc, srcStep, (Ipp8u**)pDst, dstStep, size);
            case 4:  return ippiCopy_8u_C4P4R((const Ipp8u*)pSrc, srcStep, (Ipp8u**)pDst, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 2:
            switch(channels)
            {
            case 3:  return ippiCopy_16u_C3P3R((const Ipp16u*)pSrc, srcStep, (Ipp16u**)pDst, dstStep, size);
            case 4:  return ippiCopy_16u_C4P4R((const Ipp16u*)pSrc, srcStep, (Ipp16u**)pDst, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 4:
            switch(channels)
            {
            case 3:  return ippiCopy_32f_C3P3R((const Ipp32f*)pSrc, srcStep, (Ipp32f**)pDst, dstStep, size);
            case 4:  return ippiCopy_32f_C4P4R((const Ipp32f*)pSrc, srcStep, (Ipp32f**)pDst, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        default: return ippStsDataTypeErr;
        }
    }
}
