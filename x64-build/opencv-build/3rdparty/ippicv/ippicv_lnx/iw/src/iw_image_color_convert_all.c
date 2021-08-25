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

#include "iw/iw_image_color.h"
#include "iw/iw_image_op.h"
#include "iw_owni.h"

IW_DECL(int) iwiColorToChannels(IwiColorFmt color, int planeNum)
{
    if(IWI_COLOR_IS_PLANAR(color))
    {
        int planes = IWI_COLOR_GET_PLANES(color);
        if(planeNum < 0 || planeNum >= planes)
            return 0;

        switch(color)
        {
        default:
            return 0;
        }
    }
    else
        return IWI_COLOR_GET_CHANNELS(color);
}

IW_DECL(int) iwiColorToPlanes(IwiColorFmt color)
{
    if(IWI_COLOR_IS_PLANAR(color))
        return IWI_COLOR_GET_PLANES(color);
    else
        return 1;
}

IW_DECL(IwiSize) iwiColorGetPlaneSize(IwiColorFmt color, IwiSize origSize, int planeNum)
{
    if(IWI_COLOR_IS_PLANAR(color))
    {
        IwiSize   planeSize = {0,0};
        int       planes    = IWI_COLOR_GET_PLANES(color);
        if(planeNum < 0 || planeNum >= planes)
            return planeSize;

        switch(color)
        {
        default:
            return planeSize;
        }
    }
    else
        return origSize;
}

IW_DECL(IppStatus) llwiColorConvert_RGBA_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType);
IW_DECL(IppStatus) llwiColorConvert_RGBA_RGB(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType);
IW_DECL(IppStatus) llwiColorConvert_ARGB_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType);
IW_DECL(IppStatus) llwiColorConvert_RGB_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType);
IW_DECL(IppStatus) llwiColorConvert_BGRA_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType);
IW_DECL(IppStatus) llwiColorConvert_ABGR_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType);
IW_DECL(IppStatus) llwiColorConvert_BGR_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType);
IW_DECL(IppStatus) llwiColorConvert_Gray_RGBA(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType, const Ipp64f alphaVal);
IW_DECL(IppStatus) llwiColorConvert_Gray_ARGB(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType, const Ipp64f alphaVal);
IW_DECL(IppStatus) llwiColorConvert_Gray_RGB(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType);

IW_DECL(IppStatus) llwiSwapChannels(const void *pSrc, int srcStep, int srcChannels, void *pDst, int dstStep,
    int dstChannels, IppiSize size, IppDataType dataType, const int *pDstOrder, double value, IwiChDescriptor chDesc);

IW_DECL(IppStatus) llwiColorConvert(const void *pSrc, int srcStep, IwiColorFmt srcFormat, void *pDst, int dstStep, IwiColorFmt dstFormat,
                                           IppiSize size, IppDataType dataType, Ipp64f alphaVal);

IW_DECL(IppStatus) llwiColorConvert_Wrap(const IwiImage* const pSrcImage[], IwiColorFmt srcFormat, IwiImage* const pDstImage[], IwiColorFmt dstFormat,
    Ipp64f alphaVal, const IwiColorConvertParams *pAuxParams, const IwiTile *pTile);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiColorConvert
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiColorConvert(const IwiImage* const pSrcImage[], IwiColorFmt srcFormat, IwiImage* const pDstImage[], IwiColorFmt dstFormat,
    Ipp64f alphaVal, const IwiColorConvertParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;
    int       srcPlanes = iwiColorToPlanes(srcFormat);
    int       dstPlanes = iwiColorToPlanes(dstFormat);
    int       i;

    (void)pAuxParams;

    if(!pSrcImage || !pDstImage)
        return ippStsNullPtrErr;

    if(srcFormat == dstFormat)
    {
        status = ippStsNoErr;
        for(i = 0; i < srcPlanes; i++)
        {
            status = iwiCopy(pSrcImage[i], pDstImage[i], NULL, NULL, pTile);
            if(status < 0)
                return status;
        }
        return status;
    }

    for(i = 0; i < srcPlanes; i++)
    {
        status = owniCheckImageRead(pSrcImage[i]);
        if(status)
            return status;

        if(iwiColorToChannels(srcFormat, i) != pSrcImage[i]->m_channels)
            return ippStsBadArgErr;

        if(i)
        {
            if(pSrcImage[i-i]->m_dataType != pSrcImage[i]->m_dataType)
                return ippStsBadArgErr;
        }
    }

    for(i = 0; i < dstPlanes; i++)
    {
        status = owniCheckImageWrite(pDstImage[i]);
        if(status)
            return status;

        if(iwiColorToChannels(dstFormat, i) != pDstImage[i]->m_channels)
            return ippStsBadArgErr;

        if(i)
        {
            if(pDstImage[i-i]->m_dataType != pDstImage[i]->m_dataType)
                return ippStsBadArgErr;
        }
    }

    if(pSrcImage[0]->m_dataType != pDstImage[0]->m_dataType)
        return ippStsBadArgErr;

    return llwiColorConvert_Wrap(pSrcImage, srcFormat, pDstImage, dstFormat, alphaVal, pAuxParams, pTile);
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiColorConvert_Wrap(const IwiImage* const pSrcImage[], IwiColorFmt srcFormat, IwiImage* const pDstImage[], IwiColorFmt dstFormat,
    Ipp64f alphaVal, const IwiColorConvertParams *pAuxParams, const IwiTile *pTile)
{
    const void *pSrc = pSrcImage[0]->m_ptrConst;
    void       *pDst = pDstImage[0]->m_ptr;
    IwiSize     size = owniGetMinSize(&pSrcImage[0]->m_size, &pDstImage[0]->m_size);

    if(pTile && pTile->m_initialized != ownTileInitNone)
    {
        IwiImage srcSubImage = *pSrcImage[0];
        IwiImage dstSubImage = *pDstImage[0];
        const IwiImage* pSrcSubImages[] = {NULL, NULL, NULL, NULL};
        IwiImage* pDstSubImages[]       = {NULL, NULL, NULL, NULL};
        pSrcSubImages[0] = &srcSubImage;
        pDstSubImages[0] = &dstSubImage;

        if(pTile->m_initialized == ownTileInitSimple)
        {
            IwiRoi dstRoi = pTile->m_dstRoi;

            if(!owniTile_BoundToSize(&dstRoi, &size))
                return ippStsNoOperation;

            iwiImage_RoiSet(&srcSubImage, dstRoi);
            iwiImage_RoiSet(&dstSubImage, dstRoi);
        }
        else if(pTile->m_initialized == ownTileInitPipe)
        {
            iwiImage_RoiSet(&srcSubImage, pTile->m_boundSrcRoi);
            iwiImage_RoiSet(&dstSubImage, pTile->m_boundDstRoi);
        }
        else
            return ippStsContextMatchErr;

        return llwiColorConvert_Wrap(pSrcSubImages, srcFormat, pDstSubImages, dstFormat, alphaVal, pAuxParams, NULL);
    }

    // Long compatibility check
    {
        IppStatus status;
        IppiSize  _size;

        status = ownLongCompatCheckValue(pSrcImage[0]->m_step, NULL);
        if(status < 0)
            return status;

        status = ownLongCompatCheckValue(pDstImage[0]->m_step, NULL);
        if(status < 0)
            return status;

        status = owniLongCompatCheckSize(size, &_size);
        if(status < 0)
            return status;

        return llwiColorConvert(pSrc, (int)pSrcImage[0]->m_step, srcFormat, pDst, (int)pDstImage[0]->m_step, dstFormat, _size, pSrcImage[0]->m_dataType, alphaVal);
    }
}

const Ipp32s g_RgbBgrNoSwapOrder[4]  = {0, 1, 2, 3};                // Swap without actual swapping to emulate X->XA copy with alpha value
const Ipp32s g_RgbBgrSwapOrder[4]    = {2, 1, 0, 3};                // RGB-BGR and vice versa swapping

IW_DECL(IppStatus) llwiColorConvert(const void *pSrc, int srcStep, IwiColorFmt srcFormat, void *pDst, int dstStep, IwiColorFmt dstFormat,
                                           IppiSize size, IppDataType dataType, Ipp64f alphaVal)
{
    switch(srcFormat)
    {
    case iwiColorGray:
        switch(dstFormat)
        {
        case iwiColorRGB:
        case iwiColorBGR:   return llwiColorConvert_Gray_RGB (pSrc, srcStep, pDst, dstStep, size, dataType);
        case iwiColorRGBA:
        case iwiColorBGRA:  return llwiColorConvert_Gray_RGBA(pSrc, srcStep, pDst, dstStep, size, dataType, alphaVal);
        default:            return ippStsNotSupportedModeErr;
        }
    case iwiColorRGB:
        switch(dstFormat)
        {
        case iwiColorGray:  return llwiColorConvert_RGB_Gray(pSrc, srcStep, pDst, dstStep, size, dataType);
        case iwiColorBGR:   return llwiSwapChannels         (pSrc, srcStep, 3, pDst, dstStep, 3, size, dataType, g_RgbBgrSwapOrder, 0, iwiChDesc_None);
        case iwiColorRGBA:  return llwiSwapChannels         (pSrc, srcStep, 3, pDst, dstStep, 4, size, dataType, g_RgbBgrNoSwapOrder, alphaVal, iwiChDesc_None);
        case iwiColorBGRA:  return llwiSwapChannels         (pSrc, srcStep, 3, pDst, dstStep, 4, size, dataType, g_RgbBgrSwapOrder, alphaVal, iwiChDesc_None);
        default:            return ippStsNotSupportedModeErr;
        }
    case iwiColorBGR:
        switch(dstFormat)
        {
        case iwiColorGray:  return llwiColorConvert_BGR_Gray(pSrc, srcStep, pDst, dstStep, size, dataType);
        case iwiColorRGB:   return llwiSwapChannels         (pSrc, srcStep, 3, pDst, dstStep, 3, size, dataType, g_RgbBgrSwapOrder, 0, iwiChDesc_None);
        case iwiColorRGBA:  return llwiSwapChannels         (pSrc, srcStep, 3, pDst, dstStep, 4, size, dataType, g_RgbBgrSwapOrder, alphaVal, iwiChDesc_None);
        case iwiColorBGRA:  return llwiSwapChannels         (pSrc, srcStep, 3, pDst, dstStep, 4, size, dataType, g_RgbBgrNoSwapOrder, alphaVal, iwiChDesc_None);
        default:            return ippStsNotSupportedModeErr;
        }
    case iwiColorRGBA:
        switch(dstFormat)
        {
        case iwiColorGray:  return llwiColorConvert_RGBA_Gray(pSrc, srcStep, pDst, dstStep, size, dataType);
        case iwiColorRGB:   return llwiColorConvert_RGBA_RGB (pSrc, srcStep, pDst, dstStep, size, dataType);
        case iwiColorBGR:   return llwiSwapChannels          (pSrc, srcStep, 4, pDst, dstStep, 3, size, dataType, g_RgbBgrSwapOrder, 0, iwiChDesc_None);
        case iwiColorBGRA:  return llwiSwapChannels          (pSrc, srcStep, 4, pDst, dstStep, 4, size, dataType, g_RgbBgrSwapOrder, 0, iwiChDesc_None);
        default:            return ippStsNotSupportedModeErr;
        }
    case iwiColorBGRA:
        switch(dstFormat)
        {
        case iwiColorGray:  return llwiColorConvert_BGRA_Gray(pSrc, srcStep, pDst, dstStep, size, dataType);
        case iwiColorRGB:   return llwiSwapChannels          (pSrc, srcStep, 4, pDst, dstStep, 3, size, dataType, g_RgbBgrSwapOrder, 0, iwiChDesc_None);
        case iwiColorBGR:   return llwiColorConvert_RGBA_RGB (pSrc, srcStep, pDst, dstStep, size, dataType);
        case iwiColorRGBA:  return llwiSwapChannels          (pSrc, srcStep, 4, pDst, dstStep, 4, size, dataType, g_RgbBgrSwapOrder, 0, iwiChDesc_None);
        default:            return ippStsNotSupportedModeErr;
        }
    default: return ippStsNotSupportedModeErr;
    }
}
