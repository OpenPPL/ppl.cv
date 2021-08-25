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

#include "iw/iw_image_transform.h"
#include "iw_owni.h"

IW_DECL(IppStatus) llwiMirror_Wrap(const IwiImage *pSrcImage, IwiImage *pDstImage, IppiAxis axis, const IwiMirrorParams *pAuxParams, const IwiTile *pTile);

IW_DECL(IppStatus) llwiMirror(const void *pSrc, int srcStep, void *pDst, int dstStep,
                              IppiSize size, int typeSize, int channels, IppiAxis axis, IwiChDescriptor chDesc);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMirror
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiMirror(const IwiImage *pSrcImage, IwiImage *pDstImage, IppiAxis axis, const IwiMirrorParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus       status;
    IwiMirrorParams auxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_typeSize != pDstImage->m_typeSize ||
        pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if(pAuxParams)
        auxParams = *pAuxParams;
    else
        iwiMirror_SetDefaultParams(&auxParams);

    return llwiMirror_Wrap(pSrcImage, pDstImage, axis, &auxParams, pTile);
}

IW_DECL(IppStatus) iwiMirror_GetDstSize(IwiSize srcSize, IppiAxis axis, IwiSize *pDstSize)
{
    if(!pDstSize)
        return ippStsNullPtrErr;

    switch(axis)
    {
    case ippAxsHorizontal:
    case ippAxsVertical:
    case ippAxsBoth:
        *pDstSize = srcSize;
        break;
    case ippAxs45:
    case ippAxs135:
        pDstSize->width  = srcSize.height;
        pDstSize->height = srcSize.width;
        break;
    default:
        return ippStsBadArgErr;
    }

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiMirror_GetSrcRoi(IppiAxis axis, IwiSize dstSize, IwiRoi dstRoi, IwiRoi *pSrcRoi)
{
    if(!pSrcRoi)
        return ippStsNullPtrErr;

    *pSrcRoi = dstRoi;
    switch(axis)
    {
    case ippAxs135:
        pSrcRoi->x = dstSize.height - dstRoi.y - dstRoi.height;
        pSrcRoi->y = dstSize.width - dstRoi.x - dstRoi.width;
        pSrcRoi->width  = dstRoi.height;
        pSrcRoi->height = dstRoi.width;
        break;
    case ippAxs45:
        pSrcRoi->x = dstRoi.y;
        pSrcRoi->y = dstRoi.x;
        pSrcRoi->width  = dstRoi.height;
        pSrcRoi->height = dstRoi.width;
        break;
    case ippAxsHorizontal:
        pSrcRoi->y = dstSize.height - dstRoi.y - dstRoi.height;
        break;
    case ippAxsVertical:
        pSrcRoi->x = dstSize.width - dstRoi.x - dstRoi.width;
        break;
    case ippAxsBoth:
        pSrcRoi->y = dstSize.height - dstRoi.y - dstRoi.height;
        pSrcRoi->x = dstSize.width - dstRoi.x - dstRoi.width;
        break;
    default:
        return ippStsNotSupportedModeErr;
    }

    return ippStsNoErr;
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiMirror_Wrap(const IwiImage *pSrcImage, IwiImage *pDstImage, IppiAxis axis, const IwiMirrorParams *pAuxParams, const IwiTile *pTile)
{
    const void *pSrc    = pSrcImage->m_ptrConst;
    void       *pDst    = pDstImage->m_ptr;
    IwiSize     size    = pSrcImage->m_size;

    // Flip src size
    if(axis == ippAxs45 || axis == ippAxs135)
    {
        size.width  = pSrcImage->m_size.height;
        size.height = pSrcImage->m_size.width;
    }
    size = owniGetMinSize(&size, &pDstImage->m_size);

    if(pTile && pTile->m_initialized != ownTileInitNone)
    {
        IwiImage srcSubImage = *pSrcImage;
        IwiImage dstSubImage = *pDstImage;

        if(pTile->m_initialized == ownTileInitSimple)
        {
            IppStatus status;
            IwiSize   srcSize = size;
            IwiRoi    dstRoi  = pTile->m_dstRoi;
            IwiRoi    srcRoi;

            if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst)
                return ippStsInplaceModeNotSupportedErr;

            // Flip src size back
            if(axis == ippAxs45 || axis == ippAxs135)
            {
                srcSize.width  = size.height;
                srcSize.height = size.width;
            }

            status = iwiMirror_GetSrcRoi(axis, pDstImage->m_size, dstRoi, &srcRoi);
            if(status < 0)
                return status;

            if(!owniTile_BoundToSize(&srcRoi, &srcSize))
                return ippStsNoOperation;
            if(!owniTile_BoundToSize(&dstRoi, &size))
                return ippStsNoOperation;

            iwiImage_RoiSet(&srcSubImage, srcRoi);
            iwiImage_RoiSet(&dstSubImage, dstRoi);
        }
        else if(pTile->m_initialized == ownTileInitPipe)
        {
            iwiImage_RoiSet(&srcSubImage, pTile->m_boundSrcRoi);
            iwiImage_RoiSet(&dstSubImage, pTile->m_boundDstRoi);
        }
        else
            return ippStsContextMatchErr;

        return llwiMirror_Wrap(&srcSubImage, &dstSubImage, axis, pAuxParams, NULL);
    }

    // Long compatibility check
    {
        IppStatus status;
        IppiSize  _size;

        status = ownLongCompatCheckValue(pSrcImage->m_step, NULL);
        if(status < 0)
            return status;

        status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
        if(status < 0)
            return status;

        status = owniLongCompatCheckSize(size, &_size);
        if(status < 0)
            return status;

        return llwiMirror(pSrc, (int)pSrcImage->m_step, pDst, (int)pDstImage->m_step, _size, pSrcImage->m_typeSize, pSrcImage->m_channels, axis, pAuxParams->chDesc);
    }
}

IW_DECL(IppStatus) llwiMirror(const void *pSrc, int srcStep, void *pDst, int dstStep,
                              IppiSize size, int typeSize, int channels, IppiAxis axis, IwiChDescriptor chDesc)
{
    OwniChCodes chCode = owniChDescriptorToCode(chDesc, channels, channels);

    if(pSrc == pDst)
    {
        switch(typeSize)
        {
#if IW_ENABLE_DATA_DEPTH_8
        case 1:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        return ippiMirror_8u_C1IR((Ipp8u*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        return ippiMirror_8u_C3IR((Ipp8u*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        return ippiMirror_8u_C4IR((Ipp8u*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   return ippiMirror_8u_AC4IR((Ipp8u*)pSrc, srcStep, size, axis);
#endif
            default:            return ippStsNumChannelsErr;
            }
#endif
#if IW_ENABLE_DATA_DEPTH_16
        case 2:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        return ippiMirror_16u_C1IR((Ipp16u*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        return ippiMirror_16u_C3IR((Ipp16u*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        return ippiMirror_16u_C4IR((Ipp16u*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   return ippiMirror_16u_AC4IR((Ipp16u*)pSrc, srcStep, size, axis);
#endif
            default:            return ippStsNumChannelsErr;
            }
#endif
#if IW_ENABLE_DATA_DEPTH_32
        case 4:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        return ippiMirror_32f_C1IR((Ipp32f*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        return ippiMirror_32f_C3IR((Ipp32f*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        return ippiMirror_32f_C4IR((Ipp32f*)pSrc, srcStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   return ippiMirror_32f_AC4IR((Ipp32f*)pSrc, srcStep, size, axis);
#endif
            default:            return ippStsNumChannelsErr;
            }
#endif
        default: return ippStsDataTypeErr;
        }
    }
    else
    {
        switch(typeSize)
        {
#if IW_ENABLE_DATA_DEPTH_8
        case 1:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        return ippiMirror_8u_C1R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        return ippiMirror_8u_C3R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        return ippiMirror_8u_C4R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   return ippiMirror_8u_AC4R((Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, axis);
#endif
            default:            return ippStsNumChannelsErr;
            }
#endif
#if IW_ENABLE_DATA_DEPTH_16
        case 2:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        return ippiMirror_16u_C1R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        return ippiMirror_16u_C3R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        return ippiMirror_16u_C4R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   return ippiMirror_16u_AC4R((Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, axis);
#endif
            default:            return ippStsNumChannelsErr;
            }
#endif
#if IW_ENABLE_DATA_DEPTH_32
        case 4:
            switch(chCode)
            {
#if IW_ENABLE_CHANNELS_C1
            case owniC1:        return ippiMirror_32f_C1R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C3
            case owniC3:        return ippiMirror_32f_C3R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_C4
            case owniC4:        return ippiMirror_32f_C4R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, axis);
#endif
#if IW_ENABLE_CHANNELS_AC4
            case owniC4M1110:   return ippiMirror_32f_AC4R((Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, axis);
#endif
            default:            return ippStsNumChannelsErr;
            }
#endif
        default: return ippStsDataTypeErr;
        }
    }
}
