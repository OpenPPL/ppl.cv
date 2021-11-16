/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "ppl/cv/cuda/mean.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

typedef unsigned char uchar;

template<typename T>
__host__ __device__ inline T divUp(T a, T b){
    return (a + b - 1) / b;
}

//subMeanDivVariance  Mean ImageToTensor
template<typename T>
__device__ T shfl_sum(T mySum, int offset, int warpSize);
template<>
__device__ double4 shfl_sum(double4 mySum, int offset, int warpSize) {
    double4 res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res.x += __shfl_down_sync(0xffffffff, mySum.x, offset, warpSize);
        res.y += __shfl_down_sync(0xffffffff, mySum.y, offset, warpSize);
        res.z += __shfl_down_sync(0xffffffff, mySum.z, offset, warpSize);
        res.w += __shfl_down_sync(0xffffffff, mySum.w, offset, warpSize);
    #else
        res.x += __shfl_down(mySum.x, offset);
        res.y += __shfl_down(mySum.y, offset);
        res.z += __shfl_down(mySum.z, offset);
        res.w += __shfl_down(mySum.w, offset);
    #endif
    return res;
}

template<>
__device__ double3 shfl_sum(double3 mySum, int offset, int warpSize) {
    double3 res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res.x += __shfl_down_sync(0xffffffff, mySum.x, offset, warpSize);
        res.y += __shfl_down_sync(0xffffffff, mySum.y, offset, warpSize);
        res.z += __shfl_down_sync(0xffffffff, mySum.z, offset, warpSize);
    #else
        res.x += __shfl_down(mySum.x, offset);
        res.y += __shfl_down(mySum.y, offset);
        res.z += __shfl_down(mySum.z, offset);
    #endif
    return res;
}

template<>
__device__ double shfl_sum(double mySum, int offset, int warpSize) {
    double res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res += __shfl_down_sync(0xffffffff, mySum, offset, warpSize);
    #else
        res += __shfl_down(mySum, offset);
    #endif
    return res;
}
template<>
__device__ float4 shfl_sum(float4 mySum, int offset, int warpSize) {
    float4 res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res.x += __shfl_down_sync(0xffffffff, mySum.x, offset, warpSize);
        res.y += __shfl_down_sync(0xffffffff, mySum.y, offset, warpSize);
        res.z += __shfl_down_sync(0xffffffff, mySum.z, offset, warpSize);
        res.w += __shfl_down_sync(0xffffffff, mySum.w, offset, warpSize);
    #else
        res.x += __shfl_down(mySum.x, offset);
        res.y += __shfl_down(mySum.y, offset);
        res.z += __shfl_down(mySum.z, offset);
        res.w += __shfl_down(mySum.w, offset);
    #endif
    return res;
}

template<>
__device__ float3 shfl_sum(float3 mySum, int offset, int warpSize) {
    float3 res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res.x += __shfl_down_sync(0xffffffff, mySum.x, offset, warpSize);
        res.y += __shfl_down_sync(0xffffffff, mySum.y, offset, warpSize);
        res.z += __shfl_down_sync(0xffffffff, mySum.z, offset, warpSize);
    #else
        res.x += __shfl_down(mySum.x, offset);
        res.y += __shfl_down(mySum.y, offset);
        res.z += __shfl_down(mySum.z, offset);
    #endif
    return res;
}

template<>
__device__ float shfl_sum(float mySum, int offset, int warpSize) {
    float res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res += __shfl_down_sync(0xffffffff, mySum, offset, warpSize);
    #else
        res += __shfl_down(mySum, offset);
    #endif
    return res;
}

template<>
__device__ uint4 shfl_sum(uint4 mySum, int offset, int warpSize) {
    uint4 res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res.x += __shfl_down_sync(0xffffffff, mySum.x, offset, warpSize);
        res.y += __shfl_down_sync(0xffffffff, mySum.y, offset, warpSize);
        res.z += __shfl_down_sync(0xffffffff, mySum.z, offset, warpSize);
        res.w += __shfl_down_sync(0xffffffff, mySum.w, offset, warpSize);
    #else
        res.x += __shfl_down(mySum.x, offset);
        res.y += __shfl_down(mySum.y, offset);
        res.z += __shfl_down(mySum.z, offset);
        res.w += __shfl_down(mySum.w, offset);
    #endif
    return res;
}

template<>
__device__ uint3 shfl_sum(uint3 mySum, int offset, int warpSize) {
    uint3 res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res.x += __shfl_down_sync(0xffffffff, mySum.x, offset, warpSize);
        res.y += __shfl_down_sync(0xffffffff, mySum.y, offset, warpSize);
        res.z += __shfl_down_sync(0xffffffff, mySum.z, offset, warpSize);
    #else
        res.x += __shfl_down(mySum.x, offset);
        res.y += __shfl_down(mySum.y, offset);
        res.z += __shfl_down(mySum.z, offset);
    #endif
    return res;
}

template<>
__device__ int shfl_sum(int mySum, int offset, int warpSize) {
    int res = mySum;
    #if __CUDACC_VER_MAJOR__ >= 9
        res += __shfl_down_sync(0xffffffff, mySum, offset, warpSize);
    #else
        res += __shfl_down(mySum, offset);
    #endif
    return res;
}


template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

inline  __device__ double4 operator+(double4 a, uint4 b)
{
    return make_double4(a.x + __uint2double_rn(b.x), a.y +
                        __uint2double_rn(b.y), a.z + __uint2double_rn(b.z),
                        a.w + __uint2double_rn(b.w));
}
inline  __device__ double3 operator+(double3 a, uint3 b)
{
    return make_double3(a.x + __uint2double_rn(b.x), a.y +
                        __uint2double_rn(b.y), a.z + __uint2double_rn(b.z));
}
inline  __device__ double4 operator+(double4 a, uchar4 b)
{
    return make_double4(a.x + __uint2double_rn(b.x), a.y +
                        __uint2double_rn(b.y), a.z + __uint2double_rn(b.z),
                        a.w + __uint2double_rn(b.w));
}
inline  __device__ double3 operator+(double3 a, uchar3 b)
{
    return make_double3(a.x + __uint2double_rn(b.x), a.y +
                        __uint2double_rn(b.y), a.z + __uint2double_rn(b.z));
}
inline  __device__ double4 operator+(double4 a, double4 b)
{
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline  __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline  __device__ float4 operator+(float4 a, uint4 b)
{
    return make_float4(a.x + __uint2float_rn(b.x), a.y + __uint2float_rn(b.y),
                       a.z + __uint2float_rn(b.z), a.w + __uint2float_rn(b.w));
}
inline  __device__ float3 operator+(float3 a, uint3 b)
{
    return make_float3(a.x + __uint2float_rn(b.x), a.y + __uint2float_rn(b.y),
                       a.z + __uint2float_rn(b.z));
}
inline  __device__ float4 operator+(float4 a, uchar4 b)
{
    return make_float4(a.x + __uint2float_rn(b.x), a.y + __uint2float_rn(b.y),
                       a.z + __uint2float_rn(b.z), a.w + __uint2float_rn(b.w));
}
inline  __device__ float3 operator+(float3 a, uchar3 b)
{
    return make_float3(a.x + __uint2float_rn(b.x), a.y + __uint2float_rn(b.y),
                       a.z + __uint2float_rn(b.z));
}
inline  __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline  __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline  __device__ uchar4 operator+(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline  __device__ uchar3 operator+(uchar3 a, uchar3 b)
{
    return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline  __device__ uint4 operator+(uint4 a, uchar4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline  __device__ uint3 operator+(uint3 a, uchar3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline  __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline  __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline  __device__ double4 operator*(double4 a, double4 b)
{
    return make_double4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline  __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline  __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline  __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline  __device__ uint4 operator*(uchar4 a, uchar4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline  __device__ uint3 operator*(uchar3 a, uchar3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline  __device__ uint4 operator*(uint4 a, uchar4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline  __device__ uint3 operator*(uint3 a, uchar3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline  __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline  __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}


template <typename T1, typename T2>
__global__ void
reduce_kernel_mask(const T1 *g_idata, T2 *g_odata, const uchar *mask,
                   int* tempMask, int height, int width, int mask_stride,
                   int in_stride, int meanSmemCount) {
    T2 *sdata = SharedMemory<T2>();
    int *scount = (int*)&sdata[meanSmemCount];

    int warpSize = 32;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_const = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_const;
    int rowStride = blockDim.y * gridDim.y;
    int colStride = blockDim.x * gridDim.x;

    T2 mySum = {0};
    int count = 0;
    while (row < height)
    {
        while(col < width) {
            int maskIndex = row * mask_stride + col;
            int inIndex = row * in_stride + col;
            if(mask[maskIndex]) {
                mySum = mySum + g_idata[inIndex];
                count += 1;
            }
            col += colStride;
        }
        row += rowStride;
        col = col_const;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    scount[tid] = count;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128)
    {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
        scount[tid] = count = count + scount[tid + 128];
    }

    __syncthreads();

    if (tid <  64)
    {
        sdata[tid] = mySum = mySum + sdata[tid +  64];
        scount[tid] = count = count + scount[tid + 64];
    }

    __syncthreads();

    if (tid < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        mySum = mySum + sdata[tid + 32];
        scount[tid] = count = count + scount[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum = shfl_sum(mySum, offset, warpSize);
            #if __CUDACC_VER_MAJOR__ >= 9
                count += __shfl_down_sync(0xffffffff, count, offset, warpSize);
            #else
                count += __shfl_down(count, offset);
            #endif
        }
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.y * gridDim.x + blockIdx.x] = mySum;
        tempMask[blockIdx.y * gridDim.x + blockIdx.x] = count;
    }
}

template <typename T1, typename T2>
__global__ void
reduce_kernel(const T1 *g_idata, T2 *g_odata, int height, int width,
              int width_stride) {
    T2 *sdata = SharedMemory<T2>();

    int warpSize = 32;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_const = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_const;
    int rowStride = blockDim.y * gridDim.y;
    int colStride = blockDim.x * gridDim.x;

    T2 mySum = {0};
    while (row < height)
    {
        while(col < width) {
            int index = row * width_stride + col;
            mySum = mySum + g_idata[index];
            col += colStride;
        }
        row += rowStride;
        col = col_const;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128)
    {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    __syncthreads();

    if (tid <  64)
    {
        sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

    if (tid < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        mySum = mySum + sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum = shfl_sum(mySum, offset, warpSize);
        }
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.y * gridDim.x + blockIdx.x] = mySum;
    }
}


template <typename T1, typename T2, int nc>
__global__ void
block_reduce_kernel(T1 *tempIdata, T2 *g_odata, int* tempMask, int height,
                    int width, int blockNum, bool channelWise) {
    int count = 0;
    if(tempMask == NULL) {
        count = height * width;
    }
    else {
        for(int i = 0; i < blockNum; i++) {
            count += tempMask[i];
        }
    }
    if(channelWise) {
        for(int i = 1; i < blockNum; i++) {
            tempIdata[threadIdx.x] += tempIdata[i * nc + threadIdx.x];
        }
        g_odata[threadIdx.x] = tempIdata[threadIdx.x] / count;
    }
    else {
        if (threadIdx.x == 0) {
            for(int i = 1; i < blockNum * nc; i++) {
                tempIdata[0] += tempIdata[i];
            }
            g_odata[0] = tempIdata[0] / (count * nc);
        }
    }

}


template <typename T1, typename T2>
__global__ void
reducemeanvar_kernel_mask(const T1 *g_idata, T2 *g_meandata, T2* g_vardata,
                          const uchar *mask, int* tempMask, int height,
                          int width, int mask_stride, int in_stride,
                          int meanSmemCount) {
    T2 *ssum = SharedMemory<T2>();
    T2 *sqsum = &ssum[meanSmemCount];
    int *scount = (int*)&ssum[meanSmemCount << 1];

    int warpSize = 32;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_const = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_const;
    int rowStride = blockDim.y * gridDim.y;
    int colStride = blockDim.x * gridDim.x;

    T2 mySum = {0};
    T2 myQSum = {0};
    int count = 0;
    while (row < height)
    {
        while(col < width) {
            int maskIndex = row * mask_stride + col;
            int inIndex = row * in_stride + col;
            if(mask[maskIndex]) {
                mySum = mySum + g_idata[inIndex];
                myQSum = myQSum + g_idata[inIndex] * g_idata[inIndex];
                count += 1;
            }
            col += colStride;
        }
        row += rowStride;
        col = col_const;
    }

    // each thread puts its local sum into shared memory
    ssum[tid] = mySum;
    sqsum[tid] = myQSum;
    scount[tid] = count;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128)
    {
        ssum[tid] = mySum = mySum + ssum[tid + 128];
        sqsum[tid] = myQSum = myQSum + sqsum[tid + 128];
        scount[tid] = count = count + scount[tid + 128];
    }

    __syncthreads();

    if (tid <  64)
    {
        ssum[tid] = mySum = mySum + ssum[tid +  64];
        sqsum[tid] = myQSum = myQSum + sqsum[tid + 64];
        scount[tid] = count = count + scount[tid + 64];
    }

    __syncthreads();

    if (tid < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        mySum = mySum + ssum[tid + 32];
        myQSum = myQSum + sqsum[tid + 32];
        count = count + scount[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum = shfl_sum(mySum, offset, warpSize);
            myQSum = shfl_sum(myQSum, offset, warpSize);
            #if __CUDACC_VER_MAJOR__ >= 9
                count += __shfl_down_sync(0xffffffff, count, offset, warpSize);
            #else
                count += __shfl_down(count, offset);
            #endif
        }
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_meandata[blockIdx.y * gridDim.x + blockIdx.x] = mySum;
        g_vardata[blockIdx.y * gridDim.x + blockIdx.x] = myQSum;
        tempMask[blockIdx.y * gridDim.x + blockIdx.x] = count;
    }
}

template <typename T1, typename T2>
__global__ void
reducemeanvar_kernel(const T1 *g_idata, T2 *g_meandata, T2 *g_vardata,
                     int height, int width, int width_stride,
                     int meanSmemCount) {
    T2 *ssum = SharedMemory<T2>();
    T2 *sqsum = &ssum[meanSmemCount];

    int warpSize = 32;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_const = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_const;
    int rowStride = blockDim.y * gridDim.y;
    int colStride = blockDim.x * gridDim.x;

    T2 mySum = {0};
    T2 myQSum = {0};
    while (row < height)
    {
        while(col < width) {
            int index = row * width_stride + col;
            mySum = mySum + g_idata[index];
            myQSum = myQSum + g_idata[index] * g_idata[index];
            col += colStride;
        }
        row += rowStride;
        col = col_const;
    }

    // each thread puts its local sum into shared memory
    ssum[tid] = mySum;
    sqsum[tid] = myQSum;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128)
    {
        ssum[tid] = mySum = mySum + ssum[tid + 128];
        sqsum[tid] = myQSum = myQSum + sqsum[tid + 128];
    }

    __syncthreads();

    if (tid <  64)
    {
        ssum[tid] = mySum = mySum + ssum[tid +  64];
        sqsum[tid] = myQSum = myQSum + sqsum[tid +  64];
    }

    __syncthreads();

    if (tid < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        mySum = mySum + ssum[tid + 32];
        myQSum = myQSum + sqsum[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum = shfl_sum(mySum, offset, warpSize);
            myQSum = shfl_sum(myQSum, offset, warpSize);
        }
    }

    // write result for this block to global mem
    if (tid == 0) {
        int outIndex = blockIdx.y * gridDim.x + blockIdx.x;
        g_meandata[outIndex] = mySum;
        g_vardata[outIndex] = myQSum;
    }
}


template <typename T1, typename T2, int nc>
__global__ void
block_reducemeanvar_kernel(T1 *tempMeanData, T1 *tempVarData, T2 *g_omeandata,
                           T2 * g_ovardata, int* tempMask, int height,
                           int width, int blockNum, bool channelWise) {
    int count = 0;
    if(tempMask == NULL) {
        count = height * width;
    }
    else {
        for(int i = 0; i < blockNum; i++) {
            count += tempMask[i];
        }
    }
    if(channelWise) {
        for(int i = 1; i < blockNum; i++) {
            tempMeanData[threadIdx.x] += tempMeanData[i * nc + threadIdx.x];
            tempVarData[threadIdx.x] += tempVarData[i * nc + threadIdx.x];
        }
        g_omeandata[threadIdx.x] = tempMeanData[threadIdx.x] / count;
        g_ovardata[threadIdx.x] = sqrt(max((double)(tempVarData[threadIdx.x] /
           count - g_omeandata[threadIdx.x] * g_omeandata[threadIdx.x]), 0.f));
    }
    else {
        for(int i = 1; i < blockNum * nc; i++) {
            tempMeanData[0] += tempMeanData[i];
            tempVarData[0] += tempVarData[i];
        }
        g_omeandata[0] = tempMeanData[0] / (count * nc);
        g_ovardata[0] = sqrt(max((double)(tempVarData[0] /
            (count * nc)- g_omeandata[0] * g_omeandata[0]), 0.f));
    }

}

#define maxBlocksReduce 8

template <>
RetCode Mean<float, 1>(cudaStream_t stream, int height, int width,
          int inWidthStride, const float* inData, float* outMeanData,
          int inMaskStride, const uchar* inMask, bool channelWise) {
    const int nc = 1;

    float* tempMean = NULL;
    int* tempMask = NULL;
    cudaMalloc(&tempMean,
               sizeof(float) * nc * maxBlocksReduce * maxBlocksReduce);
    cudaMalloc(&tempMask, sizeof(int) * maxBlocksReduce * maxBlocksReduce);

    int inVecWidthStride = inWidthStride / nc;
    const int threadx = 32;
    const int thready = 8;
    dim3 blockSize(threadx, thready);
    int calBlocksPerGridex = (width + threadx - 1)/threadx;
    int blocksPerGridx = calBlocksPerGridex > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridex;
    int calBlocksPerGridey = (height + thready - 1)/thready;
    int blocksPerGridy = calBlocksPerGridey > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridey;
    dim3 gridSize(blocksPerGridx, blocksPerGridy);
    if(inMask != NULL) {
        int smemSize = threadx * thready * (sizeof(float) * nc + sizeof(int));
        int meanSmemCount = threadx * thready;
        reduce_kernel_mask<float, float><<<gridSize, blockSize, smemSize>>>(
            inData, (float*)tempMean, inMask, tempMask, height, width,
            inVecWidthStride, inMaskStride, meanSmemCount);
    }
    else {
        int smemSize = threadx * thready * sizeof(float) * nc;
        reduce_kernel<float, float><<<gridSize, blockSize, smemSize>>>(inData,
            (float*)tempMean, height, width, inVecWidthStride);
    }

    int blockNum = blocksPerGridx * blocksPerGridy;
    int* tempMaskPtr = inMask == NULL ? NULL : tempMask;
    block_reduce_kernel<float, float, nc><<<1, nc>>>(tempMean, outMeanData,
        tempMaskPtr, height, width, blockNum, channelWise);

    cudaFree(tempMean);
    cudaFree(tempMask);

    return RC_SUCCESS;
}

template <>
RetCode Mean<float, 3>(cudaStream_t stream, int height, int width,
                       int inWidthStride, const float* inData,
                       float* outMeanData, int inMaskStride,
                       const uchar* inMask, bool channelWise) {
    const int nc = 3;

    float* tempMean = NULL;
    int* tempMask = NULL;
    cudaMalloc(&tempMean,
               sizeof(float) * nc * maxBlocksReduce * maxBlocksReduce);
    cudaMalloc(&tempMask, sizeof(int) * maxBlocksReduce * maxBlocksReduce);

    int inVecWidthStride = inWidthStride / nc;
    const int threadx = 32;
    const int thready = 8;
    dim3 blockSize(threadx, thready);
    int calBlocksPerGridex = (width + threadx - 1)/threadx;
    int blocksPerGridx = calBlocksPerGridex > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridex;
    int calBlocksPerGridey = (height + thready - 1)/thready;
    int blocksPerGridy = calBlocksPerGridey > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridey;
    dim3 gridSize(blocksPerGridx, blocksPerGridy);
    if(inMask != NULL) {
        int smemSize = threadx * thready * (sizeof(float) * nc + sizeof(int));
        int meanSmemCount = threadx * thready;
        reduce_kernel_mask<float3, float3><<<gridSize, blockSize, smemSize>>>(
            (float3*)inData, (float3*)tempMean, inMask, tempMask, height, width,
            inVecWidthStride, inMaskStride, meanSmemCount);
    }
    else {
        int smemSize = threadx * thready * sizeof(float) * nc;
        reduce_kernel<float3, float3><<<gridSize, blockSize, smemSize>>>(
            (float3*)inData, (float3*)tempMean, height, width, inVecWidthStride);
    }

    int blockNum = blocksPerGridx * blocksPerGridy;
    int* tempMaskPtr = inMask == NULL ? NULL : tempMask;
    block_reduce_kernel<float, float, nc><<<1, nc>>>(tempMean, outMeanData,
        tempMaskPtr, height, width, blockNum, channelWise);

    cudaFree(tempMean);
    cudaFree(tempMask);

    return RC_SUCCESS;
}

template <>
RetCode Mean<float, 4>(cudaStream_t stream, int height, int width,
                       int inWidthStride, const float* inData,
                       float* outMeanData, int inMaskStride,
                       const uchar* inMask, bool channelWise) {
    const int nc = 4;

    float* tempMean = NULL;
    int* tempMask = NULL;
    cudaMalloc(&tempMean, sizeof(float) * nc * maxBlocksReduce * maxBlocksReduce);
    cudaMalloc(&tempMask, sizeof(int) * maxBlocksReduce * maxBlocksReduce);

    int inVecWidthStride = inWidthStride / nc;
    const int threadx = 32;
    const int thready = 8;
    dim3 blockSize(threadx, thready);
    int calBlocksPerGridex = (width + threadx - 1)/threadx;
    int blocksPerGridx = calBlocksPerGridex > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridex;
    int calBlocksPerGridey = (height + thready - 1)/thready;
    int blocksPerGridy = calBlocksPerGridey > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridey;
    dim3 gridSize(blocksPerGridx, blocksPerGridy);
    if(inMask != NULL) {
        int smemSize = threadx * thready * (sizeof(float) * nc + sizeof(int));
        int meanSmemCount = threadx * thready;
        reduce_kernel_mask<float4, float4><<<gridSize, blockSize, smemSize>>>(
            (float4*)inData, (float4*)tempMean, inMask, tempMask, height, width,
            inVecWidthStride, inMaskStride, meanSmemCount);
    }
    else {
        int smemSize = threadx * thready * sizeof(float) * nc;
        reduce_kernel<float4, float4><<<gridSize, blockSize, smemSize>>>(
            (float4*)inData, (float4*)tempMean, height, width, inVecWidthStride);
    }

    int blockNum = blocksPerGridx * blocksPerGridy;
    int* tempMaskPtr = inMask == NULL ? NULL : tempMask;
    block_reduce_kernel<float, float, nc><<<1, nc>>>(tempMean, outMeanData,
        tempMaskPtr, height, width, blockNum, channelWise);

    cudaFree(tempMean);
    cudaFree(tempMask);

    return RC_SUCCESS;
}


template <>
RetCode Mean<uchar, 1>(cudaStream_t stream, int height, int width,
                       int inWidthStride, const uchar* inData,
                       float* outMeanData, int inMaskStride,
                       const uchar* inMask, bool channelWise) {
    const int nc = 1;

    float* tempMean = NULL;
    int* tempMask = NULL;
    cudaMalloc(&tempMean,
               sizeof(float) * nc * maxBlocksReduce * maxBlocksReduce);
    cudaMalloc(&tempMask, sizeof(int) * maxBlocksReduce * maxBlocksReduce);

    int inVecWidthStride = inWidthStride / nc;
    const int threadx = 32;
    const int thready = 8;
    dim3 blockSize(threadx, thready);
    int calBlocksPerGridex = (width + threadx - 1)/threadx;
    int blocksPerGridx = calBlocksPerGridex > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridex;
    int calBlocksPerGridey = (height + thready - 1)/thready;
    int blocksPerGridy = calBlocksPerGridey > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridey;
    dim3 gridSize(blocksPerGridx, blocksPerGridy);
    if(inMask != NULL) {
        int smemSize = threadx * thready * (sizeof(float) * nc + sizeof(int));
        int meanSmemCount = threadx * thready;
        reduce_kernel_mask<uchar, float><<<gridSize, blockSize, smemSize>>>(
            (uchar*)inData, (float*)tempMean, inMask, tempMask, height, width,
            inMaskStride, inVecWidthStride, meanSmemCount);
    }
    else {
        int smemSize = threadx * thready * sizeof(float) * nc;
        reduce_kernel<uchar, float><<<gridSize, blockSize, smemSize>>>(
            (uchar*)inData, (float*)tempMean, height, width, inVecWidthStride);
    }

    int blockNum = blocksPerGridx * blocksPerGridy;
    int* tempMaskPtr = inMask == NULL ? NULL : tempMask;
    block_reduce_kernel<float, float, nc><<<1, nc>>>((float*)tempMean,
        (float*)outMeanData, tempMaskPtr, height, width, blockNum, channelWise);

    cudaFree(tempMean);
    cudaFree(tempMask);

    return RC_SUCCESS;
}

template <>
RetCode Mean<uchar, 3>(cudaStream_t stream, int height, int width,
                       int inWidthStride, const uchar* inData,
                       float* outMeanData, int inMaskStride,
                       const uchar* inMask, bool channelWise) {
    const int nc = 3;

    float* tempMean = NULL;
    int* tempMask = NULL;
    cudaMalloc(&tempMean,
               sizeof(float) * nc * maxBlocksReduce * maxBlocksReduce);
    cudaMalloc(&tempMask, sizeof(int) * maxBlocksReduce * maxBlocksReduce);

    int inVecWidthStride = inWidthStride / nc;
    const int threadx = 32;
    const int thready = 8;
    dim3 blockSize(threadx, thready);
    int calBlocksPerGridex = (width + threadx - 1)/threadx;
    int blocksPerGridx = calBlocksPerGridex > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridex;
    int calBlocksPerGridey = (height + thready - 1)/thready;
    int blocksPerGridy = calBlocksPerGridey > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridey;
    dim3 gridSize(blocksPerGridx, blocksPerGridy);
    if(inMask != NULL) {
        int smemSize = threadx * thready * (sizeof(float) * nc + sizeof(int));
        int meanSmemCount = threadx * thready;
        reduce_kernel_mask<uchar3, float3><<<gridSize, blockSize, smemSize>>>(
            (uchar3*)inData, (float3*)tempMean, inMask, tempMask, height, width,
            inMaskStride, inVecWidthStride, meanSmemCount);
    }
    else {
        int smemSize = threadx * thready * sizeof(float) * nc;
        reduce_kernel<uchar3, float3><<<gridSize, blockSize, smemSize>>>(
            (uchar3*)inData, (float3*)tempMean, height, width, inVecWidthStride);
    }

    int blockNum = blocksPerGridx * blocksPerGridy;
    int* tempMaskPtr = inMask == NULL ? NULL : tempMask;
    block_reduce_kernel<float, float, nc><<<1, nc>>>((float*)tempMean,
        (float*)outMeanData, tempMaskPtr, height, width, blockNum, channelWise);

    cudaFree(tempMean);
    cudaFree(tempMask);

    return RC_SUCCESS;
}

template <>
RetCode Mean<uchar, 4>(cudaStream_t stream, int height, int width,
                       int inWidthStride, const uchar* inData,
                       float* outMeanData, int inMaskStride,
                       const uchar* inMask, bool channelWise) {
    const int nc = 4;

    float* tempMean = NULL;
    int* tempMask = NULL;
    cudaMalloc(&tempMean,
               sizeof(float) * nc * maxBlocksReduce * maxBlocksReduce);
    cudaMalloc(&tempMask, sizeof(int) * maxBlocksReduce * maxBlocksReduce);

    int inVecWidthStride = inWidthStride / nc;
    const int threadx = 32;
    const int thready = 8;
    dim3 blockSize(threadx, thready);
    int calBlocksPerGridex = (width + threadx - 1)/threadx;
    int blocksPerGridx = calBlocksPerGridex > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridex;
    int calBlocksPerGridey = (height + thready - 1)/thready;
    int blocksPerGridy = calBlocksPerGridey > maxBlocksReduce ?
                         maxBlocksReduce : calBlocksPerGridey;
    dim3 gridSize(blocksPerGridx, blocksPerGridy);
    if(inMask != NULL) {
        int smemSize = threadx * thready * (sizeof(float) * nc + sizeof(int));
        int meanSmemCount = threadx * thready;
        reduce_kernel_mask<uchar4, float4><<<gridSize, blockSize, smemSize>>>(
            (uchar4*)inData, (float4*)tempMean, inMask, tempMask, height, width,
            inMaskStride, inVecWidthStride, meanSmemCount);
    }
    else {
        int smemSize = threadx * thready * sizeof(float) * nc;
        reduce_kernel<uchar4, float4><<<gridSize, blockSize, smemSize>>>(
            (uchar4*)inData, (float4*)tempMean, height, width, inVecWidthStride);
    }

    int blockNum = blocksPerGridx * blocksPerGridy;
    int* tempMaskPtr = inMask == NULL ? NULL : tempMask;
    block_reduce_kernel<float, float, nc><<<1, nc>>>((float*)tempMean,
        (float*)outMeanData, tempMaskPtr, height, width, blockNum, channelWise);

    cudaFree(tempMean);
    cudaFree(tempMask);

    return RC_SUCCESS;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
