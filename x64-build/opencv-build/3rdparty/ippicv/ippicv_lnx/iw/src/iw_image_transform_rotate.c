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

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiRotate
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiRotate_GetDstSize(IwiSize srcSize, double angle, IwiSize *pDstSize)
{
    IppStatus  status;
    IppiRect   rect  = {0};

    double bound[2][2]  = {0};
    double coeffs[2][3] = {0};

    if(!pDstSize)
        return ippStsNullPtrErr;

    status = ippiGetRotateTransform(angle, 0, 0, coeffs);
    if(status < 0)
        return status;

    rect.width  = (int)srcSize.width;
    rect.height = (int)srcSize.height;
    status = ippiGetAffineBound(rect, bound, (const double (*)[3])coeffs);
    if(status < 0)
        return status;

    pDstSize->width  = (IwSize)(bound[1][0] - bound[0][0] + 1.5);
    pDstSize->height = (IwSize)(bound[1][1] - bound[0][1] + 1.5);

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiRotate(const IwiImage *pSrcImage, IwiImage *pDstImage, double angle, IppiInterpolationType interpolation, const IwiRotateParams *pAuxParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
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
        return ippStsNoOperation;

    if(pSrcImage->m_typeSize != pDstImage->m_typeSize ||
        pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if(!pSrcImage->m_size.width || !pSrcImage->m_size.height ||
        !pDstImage->m_size.width || !pDstImage->m_size.height)
        return ippStsNoOperation;

    for(;;)
    {
        IppiRect rect         = {0};
        double   bound[2][2]  = {0};
        double   coeffs[2][3] = {0};

        status = ippiGetRotateTransform(angle, 0, 0, coeffs);
        if(status < 0)
            break;

        rect.width  = (int)pSrcImage->m_size.width;
        rect.height = (int)pSrcImage->m_size.height;
        status = ippiGetAffineBound(rect, bound, (const double (*)[3])coeffs);
        if(status < 0)
            return status;

        coeffs[0][2] -= bound[0][0];
        coeffs[1][2] -= bound[0][1];

        status = iwiWarpAffine(pSrcImage, pDstImage, (const double (*)[3])coeffs, iwTransForward, interpolation, NULL, border, pBorderVal, pTile);
        break;
    }

    return status;
}
