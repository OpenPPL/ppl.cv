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

IW_DECL(IppStatus) llwiSetChannel(double value, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels, int channelNum);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSetChannel
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiSetChannel(double value, IwiImage *pDstImage, int channelNum, const IwiSetChannelParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pDstImage->m_channels == 1)
        return iwiSet(&value, 1, pDstImage, NULL, NULL, pTile);

    if(channelNum >= pDstImage->m_channels || channelNum < 0)
        return ippStsBadArgErr;

    {
        void*     pDst  = pDstImage->m_ptr;
        IwiSize   size  = pDstImage->m_size;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                pDst = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi dstLim; iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pDst = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);

                size.width  = dstLim.width;
                size.height = dstLim.height;
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        {
            IppiSize _size;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            return llwiSetChannel(value, pDst, (int)pDstImage->m_step, _size, pDstImage->m_dataType, pDstImage->m_channels, channelNum);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiSetChannel(double value, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels, int channelNum)
{
    switch(dataType)
    {
    case ipp8u:
        switch(channels)
        {
        case 3:  return ippiSet_8u_C3CR(ownCast_64f8u(value), ((Ipp8u*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_8u_C4CR(ownCast_64f8u(value), ((Ipp8u*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp8s:
        switch(channels)
        {
        case 3:  return ippiSet_8u_C3CR(ownCast_64f8s(value), ((Ipp8u*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_8u_C4CR(ownCast_64f8s(value), ((Ipp8u*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp16u:
        switch(channels)
        {
        case 3:  return ippiSet_16u_C3CR(ownCast_64f16u(value), ((Ipp16u*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_16u_C4CR(ownCast_64f16u(value), ((Ipp16u*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp16s:
        switch(channels)
        {
        case 3:  return ippiSet_16u_C3CR(ownCast_64f16s(value), ((Ipp16u*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_16u_C4CR(ownCast_64f16s(value), ((Ipp16u*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32u:
        switch(channels)
        {
        case 3:  return ippiSet_32s_C3CR(ownCast_64f32u(value), ((Ipp32s*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_32s_C4CR(ownCast_64f32u(value), ((Ipp32s*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32s:
        switch(channels)
        {
        case 3:  return ippiSet_32s_C3CR(ownCast_64f32s(value), ((Ipp32s*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_32s_C4CR(ownCast_64f32s(value), ((Ipp32s*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32f:
        switch(channels)
        {
        case 3:  return ippiSet_32f_C3CR(ownCast_64f32f(value), ((Ipp32f*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_32f_C4CR(ownCast_64f32f(value), ((Ipp32f*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    default: return ippStsDataTypeErr;
    }
}
