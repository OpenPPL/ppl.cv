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

/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
const Ipp32f g_ColorGrayBGRCoeffs[3] = {0.114f, 0.587f, 0.299f};    // RGB coefficients in reverse order for BGR-Gray direct conversion

IW_DECL(IppStatus) llwiColorConvert_RGBA_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C4 || IW_ENABLE_CHANNELS_AC4
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiRGBToGray_8u_AC4C1R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiRGBToGray_16u_AC4C1R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_16S
    case ipp16s: return ippiRGBToGray_16s_AC4C1R((const Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiRGBToGray_32f_AC4C1R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size);
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_RGBA_RGB(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType)
{
    int depth = iwTypeToSize(dataType);
    switch(depth)
    {
#if IW_ENABLE_CHANNELS_C3 || IW_ENABLE_CHANNELS_C4 || IW_ENABLE_CHANNELS_AC4
#if IW_ENABLE_DATA_DEPTH_8
    case 1:  return ippiCopy_8u_AC4C3R((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_DEPTH_16
    case 2: return ippiCopy_16u_AC4C3R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_DEPTH_32
    case 4: return ippiCopy_32f_AC4C3R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size);
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_ARGB_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C4 || IW_ENABLE_CHANNELS_AC4
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiRGBToGray_8u_AC4C1R (((const Ipp8u*)pSrc)+1, srcStep, (Ipp8u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiRGBToGray_16u_AC4C1R(((const Ipp16u*)pSrc)+1, srcStep, (Ipp16u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_16S
    case ipp16s: return ippiRGBToGray_16s_AC4C1R(((const Ipp16s*)pSrc)+1, srcStep, (Ipp16s*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiRGBToGray_32f_AC4C1R(((const Ipp32f*)pSrc)+1, srcStep, (Ipp32f*)pDst, dstStep, size);
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_RGB_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C3
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiRGBToGray_8u_C3C1R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiRGBToGray_16u_C3C1R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_16S
    case ipp16s: return ippiRGBToGray_16s_C3C1R((const Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiRGBToGray_32f_C3C1R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size);
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_BGRA_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C4 || IW_ENABLE_CHANNELS_AC4
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiColorToGray_8u_AC4C1R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiColorToGray_16u_AC4C1R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_16S
    case ipp16s: return ippiColorToGray_16s_AC4C1R((const Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiColorToGray_32f_AC4C1R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_ABGR_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C4 || IW_ENABLE_CHANNELS_AC4
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiColorToGray_8u_AC4C1R (((const Ipp8u*)pSrc)+1, srcStep, (Ipp8u*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiColorToGray_16u_AC4C1R(((const Ipp16u*)pSrc)+1, srcStep, (Ipp16u*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_16S
    case ipp16s: return ippiColorToGray_16s_AC4C1R(((const Ipp16s*)pSrc)+1, srcStep, (Ipp16s*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiColorToGray_32f_AC4C1R(((const Ipp32f*)pSrc)+1, srcStep, (Ipp32f*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_BGR_Gray(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C3
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiColorToGray_8u_C3C1R ((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiColorToGray_16u_C3C1R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_16S
    case ipp16s: return ippiColorToGray_16s_C3C1R((const Ipp16s*)pSrc, srcStep, (Ipp16s*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiColorToGray_32f_C3C1R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, g_ColorGrayBGRCoeffs);
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_Gray_RGBA(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType, const Ipp64f alphaVal)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C4
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiGrayToRGB_8u_C1C4R((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, ownCast_64f8u(alphaVal));
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiGrayToRGB_16u_C1C4R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, ownCast_64f16u(alphaVal));
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiGrayToRGB_32f_C1C4R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, ownCast_64f32f(alphaVal));
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_Gray_ARGB(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType, const Ipp64f alphaVal)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C4
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiGrayToRGB_8u_C1C4R((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size, ownCast_64f8u(alphaVal));
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiGrayToRGB_16u_C1C4R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size, ownCast_64f16u(alphaVal));
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiGrayToRGB_32f_C1C4R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size, ownCast_64f32f(alphaVal));
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}

IW_DECL(IppStatus) llwiColorConvert_Gray_RGB(const void *pSrc, int srcStep, void *pDst, int dstStep, IppiSize size, IppDataType dataType)
{
    switch(dataType)
    {
#if IW_ENABLE_CHANNELS_C1 || IW_ENABLE_CHANNELS_C3
#if IW_ENABLE_DATA_TYPE_8U
    case ipp8u:  return ippiGrayToRGB_8u_C1C3R((const Ipp8u*)pSrc, srcStep, (Ipp8u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_16U
    case ipp16u: return ippiGrayToRGB_16u_C1C3R((const Ipp16u*)pSrc, srcStep, (Ipp16u*)pDst, dstStep, size);
#endif
#if IW_ENABLE_DATA_TYPE_32F
    case ipp32f: return ippiGrayToRGB_32f_C1C3R((const Ipp32f*)pSrc, srcStep, (Ipp32f*)pDst, dstStep, size);
#endif
#endif
    default:     return ippStsDataTypeErr;
    }
}
