// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/cv/x86/guidedfilter.h"
#include "ppl/cv/x86/convertto.h"
#include "ppl/cv/x86/split.h"
#include "ppl/cv/x86/merge.h"
#include "ppl/cv/x86/boxfilter.h"
#include "ppl/cv/x86/arithmetic.h"
#include "ppl/cv/x86/setvalue.h"
#include "ppl/cv/types.h"
#include <string.h>
#include <cmath>

#include <limits.h>
#include <assert.h>
#include <algorithm>
#include <memory>
#include <ppl/common/sys.h>

#include <vector>
#include <stack>
namespace ppl {
namespace cv {
namespace x86 {

template <>
::ppl::common::RetCode GuidedFilter<float, 1, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inImage,
    int32_t guidedWidthStride,
    const float* guidedImage,
    int32_t outWidthStride,
    float* outImage,
    int32_t radius,
    float eps,
    BorderType border_type)
{
    if (nullptr == inImage || nullptr == outImage || nullptr == guidedImage) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width || guidedWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT && border_type != ppl::cv::BORDER_REFLECT101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const int32_t kernelSize = 2 * radius + 1;

    std::unique_ptr<float[]> mean_I(new float[height * width]);
    std::unique_ptr<float[]> mean_II(new float[height * width]);
    BoxFilter<float, 1>(height, width, guidedWidthStride, guidedImage, kernelSize, kernelSize, true, width, mean_I.get(), border_type);
    Mul<float, 1>(height, width, guidedWidthStride, guidedImage, guidedWidthStride, guidedImage, outWidthStride, outImage);
    BoxFilter<float, 1>(height, width, outWidthStride, outImage, kernelSize, kernelSize, true, width, mean_II.get(), border_type);
    Mls<float, 1>(height, width, width, mean_I.get(), width, mean_I.get(), width, mean_II.get());
    std::unique_ptr<float[]>& var_I = mean_II;
    SetTo<float, 1>(height, width, outWidthStride, outImage, eps);
    Add<float, 1>(height, width, width, var_I.get(), outWidthStride, outImage, width, var_I.get());

    std::unique_ptr<float[]> mean_P(new float[height * width]);
    BoxFilter<float, 1>(height, width, inWidthStride, inImage, kernelSize, kernelSize, true, width, mean_P.get(), border_type);
    Mul<float, 1>(height, width, inWidthStride, inImage, guidedWidthStride, guidedImage, outWidthStride, outImage);
    std::unique_ptr<float[]> mean_IP(new float[height * width]);
    BoxFilter<float, 1>(height, width, outWidthStride, outImage, kernelSize, kernelSize, true, width, mean_IP.get(), border_type);
    Mls<float, 1>(height, width, width, mean_I.get(), width, mean_P.get(), width, mean_IP.get());
    std::unique_ptr<float[]>& cov_Ip = mean_IP;

    std::unique_ptr<float[]>& a = cov_Ip;
    Div<float, 1>(height, width, width, cov_Ip.get(), width, var_I.get(), width, a.get());
    Mls<float, 1>(height, width, width, a.get(), width, mean_I.get(), width, mean_P.get());
    std::unique_ptr<float[]>& b = mean_P;

    BoxFilter<float, 1>(height, width, width, a.get(), kernelSize, kernelSize, true, width, var_I.get(), border_type);
    BoxFilter<float, 1>(height, width, width, b.get(), kernelSize, kernelSize, true, outWidthStride, outImage, border_type);

    Mla<float, 1>(height, width, guidedWidthStride, guidedImage, width, var_I.get(), outWidthStride, outImage);
    return ppl::common::RC_SUCCESS;
}

template <typename T>
class simpleMemoryPool {
public:
    // block_size is elements in one block
    simpleMemoryPool(int32_t block_size, int32_t max_block_num)
    {
        m_block_size    = block_size * sizeof(T);
        m_max_block_num = max_block_num;
        m_data          = (uint8_t*)ppl::common::AlignedAlloc((m_block_size * m_max_block_num), 64);
        m_allocated.resize(m_max_block_num, false);
        m_used_block_num     = 0;
        m_max_used_block_num = 0;
    }
    ~simpleMemoryPool()
    {
        if (m_data != nullptr) {
            ppl::common::AlignedFree(m_data);
        }
    }
    T* fastMalloc()
    {
        if (m_recently_used.empty() == true) {
            for (int32_t i = 0; i < m_max_block_num; i++) {
                if (m_allocated[i] == false) {
                    m_allocated[i] = true;
                    m_used_block_num++;
                    m_max_used_block_num = std::max(m_max_used_block_num, m_used_block_num);
                    return (T*)(m_data + i * m_block_size);
                }
            }
            return nullptr;
        }
        int32_t i = m_recently_used.top();
        m_recently_used.pop();
        m_allocated[i] = true;
        m_used_block_num++;
        m_max_used_block_num = std::max(m_max_used_block_num, m_used_block_num);
        return (T*)(m_data + i * m_block_size);
    }
    void fastFree(T* addr_f)
    {
        uint8_t* addr = (uint8_t*)addr_f;
        assert(addr >= m_data && addr < m_data + m_max_block_num * m_block_size);
        assert((addr - m_data) % m_block_size == 0);
        int32_t i      = (addr - m_data) / m_block_size;
        m_allocated[i] = false;
        m_used_block_num--;
        m_recently_used.push(i);
    }
    int32_t getMaxUsedBlockNum(void)
    {
        return m_max_used_block_num;
    }

private:
    int32_t m_block_size;
    int32_t m_max_block_num;
    uint8_t* m_data;
    std::vector<bool> m_allocated;
    std::stack<int32_t> m_recently_used;
    int32_t m_used_block_num;
    int32_t m_max_used_block_num;
};

template <>
::ppl::common::RetCode GuidedFilter<float, 3, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inImage,
    int32_t guidedWidthStride,
    const float* guidedImage,
    int32_t outWidthStride,
    float* outImage,
    int32_t radius,
    float eps,
    BorderType border_type)
{
    if (nullptr == inImage || nullptr == outImage || nullptr == guidedImage) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width || guidedWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT && border_type != ppl::cv::BORDER_REFLECT101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t kernelSize = 2 * radius + 1;

    simpleMemoryPool<float> mp(width * height, 25);

    float* guidedR = mp.fastMalloc();
    float* guidedG = mp.fastMalloc();
    float* guidedB = mp.fastMalloc();
    Split3Channels(height, width, guidedWidthStride, guidedImage, width, guidedR, guidedG, guidedB);

    float* meanGuidedR = mp.fastMalloc();
    float* meanGuidedG = mp.fastMalloc();
    float* meanGuidedB = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, guidedR, kernelSize, kernelSize, true, width, meanGuidedR, border_type);
    BoxFilter<float, 1>(height, width, width, guidedG, kernelSize, kernelSize, true, width, meanGuidedG, border_type);
    BoxFilter<float, 1>(height, width, width, guidedB, kernelSize, kernelSize, true, width, meanGuidedB, border_type);

    float* GuidedRxGuidedR = mp.fastMalloc();
    float* GuidedRxGuidedG = mp.fastMalloc();
    float* GuidedRxGuidedB = mp.fastMalloc();
    float* GuidedGxGuidedG = mp.fastMalloc();
    float* GuidedGxGuidedB = mp.fastMalloc();
    float* GuidedBxGuidedB = mp.fastMalloc();
    Mul<float, 1>(height, width, width, guidedR, width, guidedR, width, GuidedRxGuidedR);
    Mul<float, 1>(height, width, width, guidedR, width, guidedG, width, GuidedRxGuidedG);
    Mul<float, 1>(height, width, width, guidedR, width, guidedB, width, GuidedRxGuidedB);
    Mul<float, 1>(height, width, width, guidedG, width, guidedG, width, GuidedGxGuidedG);
    Mul<float, 1>(height, width, width, guidedG, width, guidedB, width, GuidedGxGuidedB);
    Mul<float, 1>(height, width, width, guidedB, width, guidedB, width, GuidedBxGuidedB);

    float* varGuidedRR = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedRxGuidedR, kernelSize, kernelSize, true, width, varGuidedRR, border_type);
    mp.fastFree(GuidedRxGuidedR);
    float* varGuidedRG = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedRxGuidedG, kernelSize, kernelSize, true, width, varGuidedRG, border_type);
    mp.fastFree(GuidedRxGuidedG);
    float* varGuidedRB = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedRxGuidedB, kernelSize, kernelSize, true, width, varGuidedRB, border_type);
    mp.fastFree(GuidedRxGuidedB);
    float* varGuidedGG = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedGxGuidedG, kernelSize, kernelSize, true, width, varGuidedGG, border_type);
    mp.fastFree(GuidedGxGuidedG);
    float* varGuidedGB = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedGxGuidedB, kernelSize, kernelSize, true, width, varGuidedGB, border_type);
    mp.fastFree(GuidedGxGuidedB);
    float* varGuidedBB = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedBxGuidedB, kernelSize, kernelSize, true, width, varGuidedBB, border_type);
    mp.fastFree(GuidedBxGuidedB);

    // stay here
    Mls<float, 1>(height, width, width, meanGuidedR, width, meanGuidedR, width, varGuidedRR);
    Mls<float, 1>(height, width, width, meanGuidedR, width, meanGuidedG, width, varGuidedRG);
    Mls<float, 1>(height, width, width, meanGuidedR, width, meanGuidedB, width, varGuidedRB);
    Mls<float, 1>(height, width, width, meanGuidedG, width, meanGuidedG, width, varGuidedGG);
    Mls<float, 1>(height, width, width, meanGuidedG, width, meanGuidedB, width, varGuidedGB);
    Mls<float, 1>(height, width, width, meanGuidedB, width, meanGuidedB, width, varGuidedBB);

    float* epsImage = mp.fastMalloc();
    SetTo<float, 1>(height, width, width, epsImage, eps);
    Add<float, 1>(height, width, width, varGuidedRR, width, epsImage, width, varGuidedRR);
    Add<float, 1>(height, width, width, varGuidedGG, width, epsImage, width, varGuidedGG);
    Add<float, 1>(height, width, width, varGuidedBB, width, epsImage, width, varGuidedBB);
    mp.fastFree(epsImage);

    float* invVarGuidedRR = mp.fastMalloc();
    float* invVarGuidedRG = mp.fastMalloc();
    float* invVarGuidedRB = mp.fastMalloc();
    float* invVarGuidedGG = mp.fastMalloc();
    float* invVarGuidedGB = mp.fastMalloc();
    float* invVarGuidedBB = mp.fastMalloc();
    Mul<float, 1>(height, width, width, varGuidedGG, width, varGuidedBB, width, invVarGuidedRR);
    Mul<float, 1>(height, width, width, varGuidedGB, width, varGuidedRB, width, invVarGuidedRG);
    Mul<float, 1>(height, width, width, varGuidedRG, width, varGuidedGB, width, invVarGuidedRB);
    Mul<float, 1>(height, width, width, varGuidedRR, width, varGuidedBB, width, invVarGuidedGG);
    Mul<float, 1>(height, width, width, varGuidedRB, width, varGuidedRG, width, invVarGuidedGB);
    Mul<float, 1>(height, width, width, varGuidedRR, width, varGuidedGG, width, invVarGuidedBB);

    Mls<float, 1>(height, width, width, varGuidedGB, width, varGuidedGB, width, invVarGuidedRR);
    Mls<float, 1>(height, width, width, varGuidedRG, width, varGuidedBB, width, invVarGuidedRG);
    mp.fastFree(varGuidedBB);
    Mls<float, 1>(height, width, width, varGuidedGG, width, varGuidedRB, width, invVarGuidedRB);
    mp.fastFree(varGuidedGG);
    Mls<float, 1>(height, width, width, varGuidedRB, width, varGuidedRB, width, invVarGuidedGG);
    Mls<float, 1>(height, width, width, varGuidedRR, width, varGuidedGB, width, invVarGuidedGB);
    mp.fastFree(varGuidedGB);
    Mls<float, 1>(height, width, width, varGuidedRG, width, varGuidedRG, width, invVarGuidedBB);

    float* convDet = mp.fastMalloc();
    Mul<float, 1>(height, width, width, invVarGuidedRR, width, varGuidedRR, width, convDet);
    mp.fastFree(varGuidedRR);
    Mla<float, 1>(height, width, width, invVarGuidedRG, width, varGuidedRG, width, convDet);
    mp.fastFree(varGuidedRG);
    Mla<float, 1>(height, width, width, invVarGuidedRB, width, varGuidedRB, width, convDet);
    mp.fastFree(varGuidedRB);

    Div<float, 1>(height, width, width, invVarGuidedRR, width, convDet, width, invVarGuidedRR);
    Div<float, 1>(height, width, width, invVarGuidedRG, width, convDet, width, invVarGuidedRG);
    Div<float, 1>(height, width, width, invVarGuidedRB, width, convDet, width, invVarGuidedRB);
    Div<float, 1>(height, width, width, invVarGuidedGG, width, convDet, width, invVarGuidedGG);
    Div<float, 1>(height, width, width, invVarGuidedGB, width, convDet, width, invVarGuidedGB);
    Div<float, 1>(height, width, width, invVarGuidedBB, width, convDet, width, invVarGuidedBB);
    mp.fastFree(convDet);

    float* srcR = mp.fastMalloc();
    float* srcG = mp.fastMalloc();
    float* srcB = mp.fastMalloc();
    Split3Channels(height, width, inWidthStride, inImage, width, srcR, srcG, srcB);
    float* srcArray[3] = {srcR, srcG, srcB};
    float* dstR        = mp.fastMalloc();
    float* dstG        = mp.fastMalloc();
    float* dstB        = mp.fastMalloc();
    float* dstArray[3] = {dstR, dstG, dstB};
    float* workspace   = nullptr;

    for (int32_t i = 0; i < 3; ++i) {
        float* curChannelImage            = srcArray[i];
        float* meanCurChannelImage        = mp.fastMalloc();
        float* varGuidedRxCurChannelImage = mp.fastMalloc();
        float* varGuidedGxCurChannelImage = mp.fastMalloc();
        float* varGuidedBxCurChannelImage = mp.fastMalloc();
        workspace                         = mp.fastMalloc();
        BoxFilter<float, 1>(height, width, width, curChannelImage, kernelSize, kernelSize, true, width, meanCurChannelImage, border_type);
        Mul<float, 1>(height, width, width, guidedR, width, curChannelImage, width, workspace);
        BoxFilter<float, 1>(height, width, width, workspace, kernelSize, kernelSize, true, width, varGuidedRxCurChannelImage, border_type);
        Mul<float, 1>(height, width, width, guidedG, width, curChannelImage, width, workspace);
        BoxFilter<float, 1>(height, width, width, workspace, kernelSize, kernelSize, true, width, varGuidedGxCurChannelImage, border_type);
        Mul<float, 1>(height, width, width, guidedB, width, curChannelImage, width, workspace);
        BoxFilter<float, 1>(height, width, width, workspace, kernelSize, kernelSize, true, width, varGuidedBxCurChannelImage, border_type);
        mp.fastFree(workspace);

        Mls<float, 1>(height, width, width, meanGuidedR, width, meanCurChannelImage, width, varGuidedRxCurChannelImage);
        Mls<float, 1>(height, width, width, meanGuidedG, width, meanCurChannelImage, width, varGuidedGxCurChannelImage);
        Mls<float, 1>(height, width, width, meanGuidedB, width, meanCurChannelImage, width, varGuidedBxCurChannelImage);

        float* workspaceR = mp.fastMalloc();
        float* workspaceG = mp.fastMalloc();
        float* workspaceB = mp.fastMalloc();
        Mul<float, 1>(height, width, width, invVarGuidedRR, width, varGuidedRxCurChannelImage, width, workspaceR);
        Mul<float, 1>(height, width, width, invVarGuidedRG, width, varGuidedRxCurChannelImage, width, workspaceG);
        Mul<float, 1>(height, width, width, invVarGuidedRB, width, varGuidedRxCurChannelImage, width, workspaceB);
        mp.fastFree(varGuidedRxCurChannelImage);

        Mla<float, 1>(height, width, width, invVarGuidedRG, width, varGuidedGxCurChannelImage, width, workspaceR);
        Mla<float, 1>(height, width, width, invVarGuidedGG, width, varGuidedGxCurChannelImage, width, workspaceG);
        Mla<float, 1>(height, width, width, invVarGuidedGB, width, varGuidedGxCurChannelImage, width, workspaceB);
        mp.fastFree(varGuidedGxCurChannelImage);

        Mla<float, 1>(height, width, width, invVarGuidedRB, width, varGuidedBxCurChannelImage, width, workspaceR);
        Mla<float, 1>(height, width, width, invVarGuidedGB, width, varGuidedBxCurChannelImage, width, workspaceG);
        Mla<float, 1>(height, width, width, invVarGuidedBB, width, varGuidedBxCurChannelImage, width, workspaceB);
        mp.fastFree(varGuidedBxCurChannelImage);

        Mls<float, 1>(height, width, width, workspaceR, width, meanGuidedR, width, meanCurChannelImage);
        Mls<float, 1>(height, width, width, workspaceG, width, meanGuidedG, width, meanCurChannelImage);
        Mls<float, 1>(height, width, width, workspaceB, width, meanGuidedB, width, meanCurChannelImage);

        workspace = mp.fastMalloc();
        BoxFilter<float, 1>(height, width, width, workspaceR, kernelSize, kernelSize, true, width, workspace, border_type);
        mp.fastFree(workspaceR);
        Mul<float, 1>(height, width, width, workspace, width, guidedR, width, dstArray[i]);
        BoxFilter<float, 1>(height, width, width, workspaceG, kernelSize, kernelSize, true, width, workspace, border_type);
        mp.fastFree(workspaceG);
        Mla<float, 1>(height, width, width, workspace, width, guidedG, width, dstArray[i]);
        BoxFilter<float, 1>(height, width, width, workspaceB, kernelSize, kernelSize, true, width, workspace, border_type);
        mp.fastFree(workspaceB);
        Mla<float, 1>(height, width, width, workspace, width, guidedB, width, dstArray[i]);
        BoxFilter<float, 1>(height, width, width, meanCurChannelImage, kernelSize, kernelSize, true, width, workspace, border_type);
        mp.fastFree(meanCurChannelImage);
        Add<float, 1>(height, width, width, workspace, width, dstArray[i], width, dstArray[i]);
        mp.fastFree(workspace);
    }
    Merge3Channels(height, width, width, dstArray[0], dstArray[1], dstArray[2], outWidthStride, outImage);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GuidedFilter<uint8_t, 3, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inImage,
    int32_t guidedWidthStride,
    const uint8_t* guidedImage,
    int32_t outWidthStride,
    uint8_t* outImage,
    int32_t radius,
    float eps,
    BorderType border_type)
{
    if (nullptr == inImage || nullptr == outImage || nullptr == guidedImage) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width || guidedWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT && border_type != ppl::cv::BORDER_REFLECT101) {
        return ppl::common::RC_INVALID_VALUE;
    }

    std::unique_ptr<float[]> srcImageFP32(new float[height * width * 3]);
    std::unique_ptr<float[]> guidedImageFP32(new float[height * width * 3]);
    std::unique_ptr<float[]> dstImageFP32(new float[height * width * 3]);
    ConvertTo<uint8_t, 3, float>(height, width, inWidthStride, inImage, 1.0f, width * 3, srcImageFP32.get());
    ConvertTo<uint8_t, 3, float>(height, width, guidedWidthStride, guidedImage, 1.0f, width * 3, guidedImageFP32.get());
    GuidedFilter<float, 3, 3>(height, width, inWidthStride, srcImageFP32.get(), guidedWidthStride, guidedImageFP32.get(), outWidthStride, dstImageFP32.get(), radius, eps, border_type);
    ConvertTo<float, 3, uint8_t>(height, width, width * 3, dstImageFP32.get(), 1.0f, outWidthStride, outImage);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GuidedFilter<uint8_t, 1, 1>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inImage,
    int32_t guidedWidthStride,
    const uint8_t* guidedImage,
    int32_t outWidthStride,
    uint8_t* outImage,
    int32_t radius,
    float eps,
    BorderType border_type)
{
    if (nullptr == inImage || nullptr == outImage || nullptr == guidedImage) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width || guidedWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT && border_type != ppl::cv::BORDER_REFLECT101) {
        return ppl::common::RC_INVALID_VALUE;
    }

    std::unique_ptr<float[]> srcImageFP32(new float[height * width * 1]);
    std::unique_ptr<float[]> guidedImageFP32(new float[height * width * 1]);
    std::unique_ptr<float[]> dstImageFP32(new float[height * width * 1]);
    ConvertTo<uint8_t, 1, float>(height, width, inWidthStride, inImage, 1.0f, width, srcImageFP32.get());
    ConvertTo<uint8_t, 1, float>(height, width, guidedWidthStride, guidedImage, 1.0f, width * 1, guidedImageFP32.get());
    GuidedFilter<float, 1, 1>(height, width, inWidthStride, srcImageFP32.get(), guidedWidthStride, guidedImageFP32.get(), outWidthStride, dstImageFP32.get(), radius, eps, border_type);
    ConvertTo<float, 1, uint8_t>(height, width, width, dstImageFP32.get(), 1.0f, outWidthStride, outImage);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GuidedFilter<float, 1, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const float* inImage,
    int32_t guidedWidthStride,
    const float* guidedImage,
    int32_t outWidthStride,
    float* outImage,
    int32_t radius,
    float eps,
    BorderType border_type)
{
    if (nullptr == inImage || nullptr == outImage || nullptr == guidedImage) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width || guidedWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT && border_type != ppl::cv::BORDER_REFLECT101) {
        return ppl::common::RC_INVALID_VALUE;
    }
    int32_t kernelSize = 2 * radius + 1;

    simpleMemoryPool<float> mp(width * height, 20);

    float* guidedR = mp.fastMalloc();
    float* guidedG = mp.fastMalloc();
    float* guidedB = mp.fastMalloc();
    Split3Channels(height, width, guidedWidthStride, guidedImage, width, guidedR, guidedG, guidedB);

    float* meanGuidedR = mp.fastMalloc();
    float* meanGuidedG = mp.fastMalloc();
    float* meanGuidedB = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, guidedR, kernelSize, kernelSize, true, width, meanGuidedR, border_type);
    BoxFilter<float, 1>(height, width, width, guidedG, kernelSize, kernelSize, true, width, meanGuidedG, border_type);
    BoxFilter<float, 1>(height, width, width, guidedB, kernelSize, kernelSize, true, width, meanGuidedB, border_type);

    float* GuidedRxGuidedR = mp.fastMalloc();
    float* GuidedRxGuidedG = mp.fastMalloc();
    float* GuidedRxGuidedB = mp.fastMalloc();
    float* GuidedGxGuidedG = mp.fastMalloc();
    float* GuidedGxGuidedB = mp.fastMalloc();
    float* GuidedBxGuidedB = mp.fastMalloc();
    Mul<float, 1>(height, width, width, guidedR, width, guidedR, width, GuidedRxGuidedR);
    Mul<float, 1>(height, width, width, guidedR, width, guidedG, width, GuidedRxGuidedG);
    Mul<float, 1>(height, width, width, guidedR, width, guidedB, width, GuidedRxGuidedB);
    Mul<float, 1>(height, width, width, guidedG, width, guidedG, width, GuidedGxGuidedG);
    Mul<float, 1>(height, width, width, guidedG, width, guidedB, width, GuidedGxGuidedB);
    Mul<float, 1>(height, width, width, guidedB, width, guidedB, width, GuidedBxGuidedB);

    float* varGuidedRR = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedRxGuidedR, kernelSize, kernelSize, true, width, varGuidedRR, border_type);
    mp.fastFree(GuidedRxGuidedR);
    float* varGuidedRG = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedRxGuidedG, kernelSize, kernelSize, true, width, varGuidedRG, border_type);
    mp.fastFree(GuidedRxGuidedG);
    float* varGuidedRB = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedRxGuidedB, kernelSize, kernelSize, true, width, varGuidedRB, border_type);
    mp.fastFree(GuidedRxGuidedB);
    float* varGuidedGG = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedGxGuidedG, kernelSize, kernelSize, true, width, varGuidedGG, border_type);
    mp.fastFree(GuidedGxGuidedG);
    float* varGuidedGB = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedGxGuidedB, kernelSize, kernelSize, true, width, varGuidedGB, border_type);
    mp.fastFree(GuidedGxGuidedB);
    float* varGuidedBB = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, GuidedBxGuidedB, kernelSize, kernelSize, true, width, varGuidedBB, border_type);
    mp.fastFree(GuidedBxGuidedB);

    // stay here
    Mls<float, 1>(height, width, width, meanGuidedR, width, meanGuidedR, width, varGuidedRR);
    Mls<float, 1>(height, width, width, meanGuidedR, width, meanGuidedG, width, varGuidedRG);
    Mls<float, 1>(height, width, width, meanGuidedR, width, meanGuidedB, width, varGuidedRB);
    Mls<float, 1>(height, width, width, meanGuidedG, width, meanGuidedG, width, varGuidedGG);
    Mls<float, 1>(height, width, width, meanGuidedG, width, meanGuidedB, width, varGuidedGB);
    Mls<float, 1>(height, width, width, meanGuidedB, width, meanGuidedB, width, varGuidedBB);

    float* epsImage = mp.fastMalloc();
    SetTo<float, 1>(height, width, width, epsImage, eps);
    Add<float, 1>(height, width, width, varGuidedRR, width, epsImage, width, varGuidedRR);
    Add<float, 1>(height, width, width, varGuidedGG, width, epsImage, width, varGuidedGG);
    Add<float, 1>(height, width, width, varGuidedBB, width, epsImage, width, varGuidedBB);
    mp.fastFree(epsImage);

    float* invVarGuidedRR = mp.fastMalloc();
    float* invVarGuidedRG = mp.fastMalloc();
    float* invVarGuidedRB = mp.fastMalloc();
    float* invVarGuidedGG = mp.fastMalloc();
    float* invVarGuidedGB = mp.fastMalloc();
    float* invVarGuidedBB = mp.fastMalloc();
    Mul<float, 1>(height, width, width, varGuidedGG, width, varGuidedBB, width, invVarGuidedRR);
    Mul<float, 1>(height, width, width, varGuidedGB, width, varGuidedRB, width, invVarGuidedRG);
    Mul<float, 1>(height, width, width, varGuidedRG, width, varGuidedGB, width, invVarGuidedRB);
    Mul<float, 1>(height, width, width, varGuidedRR, width, varGuidedBB, width, invVarGuidedGG);
    Mul<float, 1>(height, width, width, varGuidedRB, width, varGuidedRG, width, invVarGuidedGB);
    Mul<float, 1>(height, width, width, varGuidedRR, width, varGuidedGG, width, invVarGuidedBB);

    Mls<float, 1>(height, width, width, varGuidedGB, width, varGuidedGB, width, invVarGuidedRR);
    Mls<float, 1>(height, width, width, varGuidedRG, width, varGuidedBB, width, invVarGuidedRG);
    mp.fastFree(varGuidedBB);
    Mls<float, 1>(height, width, width, varGuidedGG, width, varGuidedRB, width, invVarGuidedRB);
    mp.fastFree(varGuidedGG);
    Mls<float, 1>(height, width, width, varGuidedRB, width, varGuidedRB, width, invVarGuidedGG);
    Mls<float, 1>(height, width, width, varGuidedRR, width, varGuidedGB, width, invVarGuidedGB);
    mp.fastFree(varGuidedGB);
    Mls<float, 1>(height, width, width, varGuidedRG, width, varGuidedRG, width, invVarGuidedBB);

    float* convDet = mp.fastMalloc();
    Mul<float, 1>(height, width, width, invVarGuidedRR, width, varGuidedRR, width, convDet);
    mp.fastFree(varGuidedRR);
    Mla<float, 1>(height, width, width, invVarGuidedRG, width, varGuidedRG, width, convDet);
    mp.fastFree(varGuidedRG);
    Mla<float, 1>(height, width, width, invVarGuidedRB, width, varGuidedRB, width, convDet);
    mp.fastFree(varGuidedRB);

    Div<float, 1>(height, width, width, invVarGuidedRR, width, convDet, width, invVarGuidedRR);
    Div<float, 1>(height, width, width, invVarGuidedRG, width, convDet, width, invVarGuidedRG);
    Div<float, 1>(height, width, width, invVarGuidedRB, width, convDet, width, invVarGuidedRB);
    Div<float, 1>(height, width, width, invVarGuidedGG, width, convDet, width, invVarGuidedGG);
    Div<float, 1>(height, width, width, invVarGuidedGB, width, convDet, width, invVarGuidedGB);
    Div<float, 1>(height, width, width, invVarGuidedBB, width, convDet, width, invVarGuidedBB);
    mp.fastFree(convDet);

    float* workspace = nullptr;

    float* meanImage        = mp.fastMalloc();
    float* varGuidedRxImage = mp.fastMalloc();
    float* varGuidedGxImage = mp.fastMalloc();
    float* varGuidedBxImage = mp.fastMalloc();
    workspace               = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, inWidthStride, inImage, kernelSize, kernelSize, true, width, meanImage, border_type);
    Mul<float, 1>(height, width, width, guidedR, inWidthStride, inImage, width, workspace);
    BoxFilter<float, 1>(height, width, width, workspace, kernelSize, kernelSize, true, width, varGuidedRxImage, border_type);
    Mul<float, 1>(height, width, width, guidedG, inWidthStride, inImage, width, workspace);
    BoxFilter<float, 1>(height, width, width, workspace, kernelSize, kernelSize, true, width, varGuidedGxImage, border_type);
    Mul<float, 1>(height, width, width, guidedB, inWidthStride, inImage, width, workspace);
    BoxFilter<float, 1>(height, width, width, workspace, kernelSize, kernelSize, true, width, varGuidedBxImage, border_type);
    mp.fastFree(workspace);

    Mls<float, 1>(height, width, width, meanGuidedR, width, meanImage, width, varGuidedRxImage);
    Mls<float, 1>(height, width, width, meanGuidedG, width, meanImage, width, varGuidedGxImage);
    Mls<float, 1>(height, width, width, meanGuidedB, width, meanImage, width, varGuidedBxImage);

    float* workspaceR = mp.fastMalloc();
    float* workspaceG = mp.fastMalloc();
    float* workspaceB = mp.fastMalloc();
    Mul<float, 1>(height, width, width, invVarGuidedRR, width, varGuidedRxImage, width, workspaceR);
    Mul<float, 1>(height, width, width, invVarGuidedRG, width, varGuidedRxImage, width, workspaceG);
    Mul<float, 1>(height, width, width, invVarGuidedRB, width, varGuidedRxImage, width, workspaceB);
    mp.fastFree(varGuidedRxImage);

    Mla<float, 1>(height, width, width, invVarGuidedRG, width, varGuidedGxImage, width, workspaceR);
    Mla<float, 1>(height, width, width, invVarGuidedGG, width, varGuidedGxImage, width, workspaceG);
    Mla<float, 1>(height, width, width, invVarGuidedGB, width, varGuidedGxImage, width, workspaceB);
    mp.fastFree(varGuidedGxImage);

    Mla<float, 1>(height, width, width, invVarGuidedRB, width, varGuidedBxImage, width, workspaceR);
    Mla<float, 1>(height, width, width, invVarGuidedGB, width, varGuidedBxImage, width, workspaceG);
    Mla<float, 1>(height, width, width, invVarGuidedBB, width, varGuidedBxImage, width, workspaceB);
    mp.fastFree(varGuidedBxImage);

    Mls<float, 1>(height, width, width, workspaceR, width, meanGuidedR, width, meanImage);
    Mls<float, 1>(height, width, width, workspaceG, width, meanGuidedG, width, meanImage);
    Mls<float, 1>(height, width, width, workspaceB, width, meanGuidedB, width, meanImage);

    workspace = mp.fastMalloc();
    BoxFilter<float, 1>(height, width, width, workspaceR, kernelSize, kernelSize, true, width, workspace, border_type);
    mp.fastFree(workspaceR);
    Mul<float, 1>(height, width, width, workspace, width, guidedR, outWidthStride, outImage);
    BoxFilter<float, 1>(height, width, width, workspaceG, kernelSize, kernelSize, true, width, workspace, border_type);
    mp.fastFree(workspaceG);
    Mla<float, 1>(height, width, width, workspace, width, guidedG, outWidthStride, outImage);
    BoxFilter<float, 1>(height, width, width, workspaceB, kernelSize, kernelSize, true, width, workspace, border_type);
    mp.fastFree(workspaceB);
    Mla<float, 1>(height, width, width, workspace, width, guidedB, outWidthStride, outImage);
    BoxFilter<float, 1>(height, width, width, meanImage, kernelSize, kernelSize, true, width, workspace, border_type);
    mp.fastFree(meanImage);
    Add<float, 1>(height, width, width, workspace, outWidthStride, outImage, outWidthStride, outImage);
    mp.fastFree(workspace);
    return ppl::common::RC_SUCCESS;
}

template <>
::ppl::common::RetCode GuidedFilter<uint8_t, 1, 3>(
    int32_t height,
    int32_t width,
    int32_t inWidthStride,
    const uint8_t* inImage,
    int32_t guidedWidthStride,
    const uint8_t* guidedImage,
    int32_t outWidthStride,
    uint8_t* outImage,
    int32_t radius,
    float eps,
    BorderType border_type)
{
    if (nullptr == inImage || nullptr == outImage || nullptr == guidedImage) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (width <= 0 || height <= 0 || inWidthStride < width || outWidthStride < width || guidedWidthStride <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (border_type != ppl::cv::BORDER_REFLECT && border_type != ppl::cv::BORDER_REFLECT101) {
        return ppl::common::RC_INVALID_VALUE;
    }

    std::unique_ptr<float[]> srcImageFP32(new float[height * width * 1]);
    std::unique_ptr<float[]> guidedImageFP32(new float[height * width * 3]);
    std::unique_ptr<float[]> dstImageFP32(new float[height * width * 1]);
    ConvertTo<uint8_t, 1, float>(height, width, inWidthStride, inImage, 1.0f, width, srcImageFP32.get());
    ConvertTo<uint8_t, 3, float>(height, width, guidedWidthStride, guidedImage, 1.0f, width * 3, guidedImageFP32.get());
    GuidedFilter<float, 1, 3>(height, width, inWidthStride, srcImageFP32.get(), guidedWidthStride, guidedImageFP32.get(), outWidthStride, dstImageFP32.get(), radius, eps, border_type);
    ConvertTo<float, 1, uint8_t>(height, width, width, dstImageFP32.get(), 1.0f, outWidthStride, outImage);
    return ppl::common::RC_SUCCESS;
}

}
}
} // namespace ppl::cv::x86
