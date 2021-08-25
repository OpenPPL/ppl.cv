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

IW_DECL(IppStatus) llwiSwapChannels(const void *pSrc, int srcStep, int srcChannels, void *pDst, int dstStep,
    int dstChannels, IppiSize size, IppDataType dataType, const int *pDstOrder, double value, IwiChDescriptor chDesc);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiSwapChannels
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiSwapChannels(const IwiImage *pSrcImage, IwiImage *pDstImage, const int *pDstOrder, double value, const IwiSwapChannelsParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus             status;
    IwiSwapChannelsParams auxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_dataType != pDstImage->m_dataType)
        return ippStsBadArgErr;

    if(pSrcImage->m_channels == 1 && pDstImage->m_channels == 1)
        return iwiCopy(pSrcImage, pDstImage, NULL, NULL, pTile);

    if(!pDstOrder)
        return ippStsNullPtrErr;

    if(pAuxParams)
        auxParams = *pAuxParams;
    else
        iwiSwapChannels_SetDefaultParams(&auxParams);

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

            return llwiSwapChannels(pSrc, (int)pSrcImage->m_step, pSrcImage->m_channels, pDst, (int)pDstImage->m_step,
                pDstImage->m_channels, _size, pSrcImage->m_dataType, pDstOrder, value, auxParams.chDesc);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiSwapChannels(const void *pSrc, int srcStep, int srcChannels, void *pDst, int dstStep,
    int dstChannels, IppiSize size, IppDataType dataType, const int *pDstOrder, double value, IwiChDescriptor chDesc)
{
    int         depth  = iwTypeToSize(dataType);
    OwniChCodes chCode = owniChDescriptorToCode(chDesc, srcChannels, dstChannels);

    if(pSrc == pDst)
    {
        if(srcChannels != dstChannels)
            return ippStsNumChannelsErr;

        switch(chCode)
        {
#if IW_ENABLE_CHANNELS_C3
        case owniC3:
            switch(depth)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case 1:     return ippiSwapChannels_8u_C3IR((Ipp8u*)pSrc, srcStep, size, pDstOrder);
#endif
            default:    return ippStsDataTypeErr;
            }
#endif
#if IW_ENABLE_CHANNELS_C4
        case owniC4:
            switch(depth)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case 1:     return ippiSwapChannels_8u_C4IR((Ipp8u*)pSrc, srcStep, size, pDstOrder);
#endif
            default:    return ippStsDataTypeErr;
            }
#endif
#if IW_ENABLE_CHANNELS_AC4
        case owniC4M1110:
        {
            int tmpOrder[4];
            tmpOrder[0] = pDstOrder[0];
            tmpOrder[1] = pDstOrder[1];
            tmpOrder[2] = pDstOrder[2];
            tmpOrder[3] = 3;
            switch(depth)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case 1:
                        if(tmpOrder[0] == 3 || tmpOrder[1] == 3 || tmpOrder[2] == 3)
                            return ippStsChannelOrderErr;
                        return ippiSwapChannels_8u_C4IR((Ipp8u*)pSrc, srcStep, size, tmpOrder);
#endif
            default:    return ippStsDataTypeErr;
            }
        }
#endif
        default: return ippStsNumChannelsErr;
        }
    }
    else
    {
        switch(chCode)
        {
#if IW_ENABLE_CHANNELS_C3
        case owniC3:
            switch(depth)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case 1:     return ippiSwapChannels_8u_C3R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, pDstOrder);
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case 2:     return ippiSwapChannels_16u_C3R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pDstOrder);
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case 4:     return ippiSwapChannels_32f_C3R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, pDstOrder);
#endif
            default:    return ippStsDataTypeErr;
            }
#endif
#if IW_ENABLE_CHANNELS_C3 || IW_ENABLE_CHANNELS_C4
        case owniC3C4:
            switch(dataType)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case ipp8u:  return ippiSwapChannels_8u_C3C4R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, pDstOrder, ownCast_64f8u(value));
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case ipp16u: return ippiSwapChannels_16u_C3C4R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pDstOrder, ownCast_64f16u(value));
#endif
#if IW_ENABLE_DATA_TYPE_16S
            case ipp16s: return ippiSwapChannels_16s_C3C4R((const Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, pDstOrder, ownCast_64f16s(value));
#endif
#if IW_ENABLE_DATA_TYPE_32S
            case ipp32s: return ippiSwapChannels_32s_C3C4R((const Ipp32s*)pSrc, srcStep, (Ipp32s*)pDst, dstStep, size, pDstOrder, ownCast_64f32s(value));
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case ipp32f: return ippiSwapChannels_32f_C3C4R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, pDstOrder, ownCast_64f32f(value));
#endif
            default:     return ippStsDataTypeErr;
            }
        case owniC4C3:
            switch(depth)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case 1:     return ippiSwapChannels_8u_C4C3R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, pDstOrder);
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case 2:     return ippiSwapChannels_16u_C4C3R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pDstOrder);
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case 4:     return ippiSwapChannels_32f_C4C3R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, pDstOrder);
#endif
            default:    return ippStsDataTypeErr;
            }
#endif
#if IW_ENABLE_CHANNELS_C4
        case owniC4:
            switch(depth)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case 1:     return ippiSwapChannels_8u_C4R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, pDstOrder);
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case 2:     return ippiSwapChannels_16u_C4R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pDstOrder);
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case 4:     return ippiSwapChannels_32f_C4R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, pDstOrder);
#endif
            default:    return ippStsDataTypeErr;
            }
#endif
#if IW_ENABLE_CHANNELS_AC4
        case owniC4M1110:
            switch(depth)
            {
#if IW_ENABLE_DATA_TYPE_8U
            case 1:     return ippiSwapChannels_8u_AC4R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, pDstOrder);
#endif
#if IW_ENABLE_DATA_TYPE_16U
            case 2:     return ippiSwapChannels_16u_AC4R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, pDstOrder);
#endif
#if IW_ENABLE_DATA_TYPE_32F
            case 4:     return ippiSwapChannels_32f_AC4R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, pDstOrder);
#endif
            default:    return ippStsDataTypeErr;
            }
#endif
        default: return ippStsNumChannelsErr;
        }
    }
}
