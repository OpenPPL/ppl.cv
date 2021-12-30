#include "./intrinutils_avx.hpp"
#include "./internal_avx.hpp"
#include "ppl/common/sys.h"
#include "ppl/common/x86/sysinfo.h"
#include <vector>
#include <stdint.h>
#include <cstring>
#include <cassert>
#include<immintrin.h>
namespace ppl {
namespace cv {
namespace x86 {

template<typename _TpSrc, typename _TpDst> struct imageCrop
{
    imageCrop(float _scale):scale(_scale) {}
    void operator()(const _TpSrc* src, _TpDst* dst, int n) const
    {
        for(int i = 0; i < n; i++) {
            dst[i] = (_TpDst)(scale*src[i]);
        }
    }
    float scale;
};

template <>
    struct imageCrop<float, float>
    {
        imageCrop(float _scale):scale(_scale)
        {
            v_scale = _mm256_set1_ps(scale);
            core = 1;
            bSupportAVX = ppl::common::CpuSupports(ppl::common::ISA_X86_AVX);
        }

        void operator()(const float* src, float* dst, int n) const
        {
            int i=0;
            if (bSupportAVX) {
                for ( ; i <= n - 8; i += 8, src += 8)
                {
                    __m256 v_in = _mm256_loadu_ps(src);
                    __m256 v_out = _mm256_mul_ps(v_in, v_scale);

                    _mm256_storeu_ps((dst + i), v_out);
                }
            }

            for ( ; i < n; i++, src++)
            {
                dst[i] = scale*src[0];
            }
        }

        float scale;
        __m256 v_scale;
        __m256 v_zero;
        int core;
        bool bSupportAVX;
    };

template<>
void x86ImageCrop_avx<float, 1, float, 1, 1>(
    int p_y, int p_x, int inWidthStride, const float* inData,
        int outHeight, int outWidth, int outWidthStride, float* outData,
        float ratio){
    assert(inData != NULL);
    assert(outData != NULL);
    assert(outHeight != 0 && outWidth != 0 && inWidthStride != 0 && outWidthStride != 0);

    const float *src = inData + p_y * inWidthStride + p_x;
    float *dst = outData;

    imageCrop<float, float> s = imageCrop<float, float>(ratio);
    for(int i = 0; i<outHeight; i++)
    {
        s.operator()(src, dst, outWidth);
        src += inWidthStride;
        dst += outWidthStride;
    }
}
template<>
void x86ImageCrop_avx<float, 2, float, 2, 2>(
    int p_y, int p_x, int inWidthStride, const float* inData,
        int outHeight, int outWidth, int outWidthStride, float* outData,
        float ratio){
    assert(inData != NULL);
    assert(outData != NULL);
    assert(outHeight != 0 && outWidth != 0 && inWidthStride != 0 && outWidthStride != 0);

    const float *src = inData + p_y * inWidthStride + p_x*2;
    float *dst = outData;

    imageCrop<float, float> s = imageCrop<float, float>(ratio);
    for (int i = 0; i < outHeight; ++i)
    {
        s.operator()(src, dst, outWidth*2);
        src += inWidthStride;
        dst += outWidthStride;
    }
}

template<>
void x86ImageCrop_avx<float, 3, float, 3, 3>(
    int p_y, int p_x, int inWidthStride, const float* inData,
        int outHeight, int outWidth, int outWidthStride, float* outData,
        float ratio){
    assert(inData != NULL);
    assert(outData != NULL);
    assert(outHeight != 0 && outWidth != 0 && inWidthStride != 0 && outWidthStride != 0);

    const float *src = inData + p_y * inWidthStride + p_x*3;
    float *dst = outData;

    imageCrop<float, float> s = imageCrop<float, float>(ratio);
    for (int i = 0; i < outHeight; ++i)
    {
        s.operator()(src, dst, outWidth*3);
        src += inWidthStride;
        dst += outWidthStride;
    }
}
template<>
void x86ImageCrop_avx<float, 4, float, 4, 4>(
    int p_y, int p_x, int inWidthStride, const float* inData,
        int outHeight, int outWidth, int outWidthStride, float* outData,
        float ratio){
    assert(inData != NULL);
    assert(outData != NULL);
    assert(outHeight != 0 && outWidth != 0 && inWidthStride != 0 && outWidthStride != 0);

    const float *src = inData + p_y * inWidthStride + p_x*4;
    float *dst = outData;

    imageCrop<float, float> s = imageCrop<float, float>(ratio);
    for(int i = 0; i<outHeight; i++)
    {
        s.operator()(src, dst, outWidth*4);
        src += inWidthStride;
        dst += outWidthStride;
    }
}

}
}
}
