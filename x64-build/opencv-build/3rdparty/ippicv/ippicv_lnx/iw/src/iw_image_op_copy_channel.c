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

IW_DECL(IppStatus) llwiCopyChannel(const void *pSrc, int srcStep, int srcChannels, int srcChannel, void *pDst, int dstStep,
    int dstChannels, int dstChannel, IppiSize size, int typeSize);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCopyChannel
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiCopyChannel(const IwiImage *pSrcImage, int srcChannel, IwiImage *pDstImage, int dstChannel, const IwiCopyChannelParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst && srcChannel == dstChannel)
        return ippStsNoOperation;

    if(srcChannel >= pSrcImage->m_channels || srcChannel < 0 || dstChannel >= pDstImage->m_channels || dstChannel < 0)
        return ippStsBadArgErr;

    if(pSrcImage->m_channels == 1 && pDstImage->m_channels == 1)
        return iwiCopy(pSrcImage, pDstImage, NULL, NULL, pTile);

    if(pSrcImage->m_typeSize != pDstImage->m_typeSize)
        return ippStsBadArgErr;

    {
        const void *pSrc = pSrcImage->m_ptrConst;
        void       *pDst = pDstImage->m_ptr;
        IwiSize     size = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                pSrc = iwiImage_GetPtrConst(pSrcImage, dstRoi.y, dstRoi.x, 0);
                pDst = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi srcLim;
                IwiRoi dstLim;
                iwiTilePipeline_GetBoundedSrcRoi(pTile, &srcLim);
                iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pSrc = iwiImage_GetPtrConst(pSrcImage, srcLim.y, srcLim.x, 0);
                pDst = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);

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

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            return llwiCopyChannel(pSrc, (int)pSrcImage->m_step, pSrcImage->m_channels, srcChannel, pDst, (int)pDstImage->m_step, pDstImage->m_channels, dstChannel, _size, pSrcImage->m_typeSize);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiCopyChannel(const void *pSrc, int srcStep, int srcChannels, int srcChannel, void *pDst, int dstStep,
    int dstChannels, int dstChannel, IppiSize size, int typeSize)
{
    switch(typeSize)
    {
    case 1:
        switch(srcChannels)
        {
        case 1:
            switch(dstChannels)
            {
            case 3:  return ippiCopy_8u_C1C3R(((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_8u_C1C4R(((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 3:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_8u_C3C1R(((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            case 3:  return ippiCopy_8u_C3CR (((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 4:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_8u_C4C1R(((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_8u_C4CR (((const Ipp8u*)pSrc)+srcChannel, srcStep, ((Ipp8u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        default: return ippStsNumChannelsErr;
        }
    case 2:
        switch(srcChannels)
        {
        case 1:
            switch(dstChannels)
            {
            case 3:  return ippiCopy_16u_C1C3R(((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_16u_C1C4R(((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 3:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_16u_C3C1R(((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            case 3:  return ippiCopy_16u_C3CR (((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 4:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_16u_C4C1R(((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_16u_C4CR (((const Ipp16u*)pSrc)+srcChannel, srcStep, ((Ipp16u*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        default: return ippStsNumChannelsErr;
        }
    case 4:
        switch(srcChannels)
        {
        case 1:
            switch(dstChannels)
            {
            case 3:  return ippiCopy_32f_C1C3R(((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_32f_C1C4R(((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 3:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_32f_C3C1R(((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            case 3:  return ippiCopy_32f_C3CR (((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        case 4:
            switch(dstChannels)
            {
            case 1:  return ippiCopy_32f_C4C1R(((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            case 4:  return ippiCopy_32f_C4CR (((const Ipp32f*)pSrc)+srcChannel, srcStep, ((Ipp32f*)pDst)+dstChannel, dstStep, size);
            default: return ippStsNumChannelsErr;
            }
        default: return ippStsNumChannelsErr;
        }
    default: return ippStsDataTypeErr;
    }
}
