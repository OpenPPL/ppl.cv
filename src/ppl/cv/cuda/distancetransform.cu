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

#include "ppl/cv/cuda/distancetransform.h"

using namespace ppl::common;

namespace ppl {
namespace cv {
namespace cuda {

typedef unsigned char uchar;

__constant__ float mask5[5][5];
__constant__ float mask3[3][3];

/// surface reference
surface<void, 2> surfRef;
/// Device array binded to surface
cudaArray* cuInArray;

__global__
void initDT(int height, int width, int widthStride, const uchar *src) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * widthStride + col;

    float val = (float)height*width;
    if(row < height && col < width) {
        float data = 0.f;
        if(src[ind] != 0)
            data = val;

        surf2Dwrite(data, surfRef, col*4, row);
    }
}

__global__
void finalizeDT(int height, int width, int widthStride, float *dst) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = row * widthStride + col;

    if(row < height && col < width) {
        float data = 0.0f;
        surf2Dread(&data, surfRef, col * 4, row, cudaBoundaryModeClamp);
        //surf2Dwrite(data, surfRef, col*4, row);
        dst[ind] = data;
    }
}

__global__
void calcDT_3x3(int w, int h, int *done) {
    __shared__ int found;
    bool written = true;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    bool mainthread = (threadIdx.x + threadIdx.y == 0);

    if(row < h && col < w) {
        if(mainthread) {
            written = false;
            atomicExch(&found, 1);
        }
        __syncthreads();

        int inf = 2147483647;
        float data;
        float eps = 0;
        surf2Dread(&data, surfRef, col* 4, row);
        if(data > 0 || mainthread) {
            float newData, oldData, initData;
            newData = data;
            oldData = data;
            initData = data;
            while(found > 0) {
                if(mainthread) {
                    atomicExch(&found, 0);
                }
                __syncthreads();

                oldData = newData < initData ? newData : initData;
                newData = inf;

                surf2Dread(&data, surfRef, (col-1) * 4, row-1,
                           cudaBoundaryModeClamp);
                data += mask3[-1+1][-1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-1) * 4, row+0,
                           cudaBoundaryModeClamp);
                data += mask3[-1+1][0+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-1) * 4, row+1,
                           cudaBoundaryModeClamp);
                data += mask3[-1+1][1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+0) * 4, row-1,
                           cudaBoundaryModeClamp);
                data += mask3[0+1][-1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+0) * 4, row+1,
                           cudaBoundaryModeClamp);
                data += mask3[0+1][1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row-1,
                           cudaBoundaryModeClamp);
                data += mask3[1+1][-1+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row+0,
                           cudaBoundaryModeClamp);
                data += mask3[1+1][0+1];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row+1,
                           cudaBoundaryModeClamp);
                data += mask3[1+1][1+1];
                if(newData - data > eps) newData = data;

                if(newData < oldData) {
                    surf2Dwrite(newData, surfRef, col * 4, row);
                    atomicExch(&found, 1);
                }

                __syncthreads();
                if(mainthread && found > 0 && !written) {
                    atomicExch(done, 0);
                    written = true;
                }
            }
        }
    }
}

__global__
void calcDT_5x5(int w, int h, int *done) {
    __shared__ int found;
    bool written = true;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    bool mainthread = (threadIdx.x + threadIdx.y == 0);

    if(row < h && col < w) {
        if(mainthread) {
            written = false;
            atomicExch(&found, 1);
        }
        __syncthreads();

        int inf = 2147483647;
        float data;
        float eps = 0;
        surf2Dread(&data, surfRef, col* 4, row);

        if(data > 0 || mainthread) {
            float newData, oldData, initData;
            newData = data;
            oldData = data;
            initData = data;
            while(found > 0) {
                if(mainthread) {
                    atomicExch(&found, 0);
                }
                __syncthreads();

                oldData = newData < initData ? newData : initData;
                newData = inf;

                surf2Dread(&data, surfRef, (col-1) * 4, row-1,
                           cudaBoundaryModeClamp);
                data += mask5[-1+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-1) * 4, row+0,
                           cudaBoundaryModeClamp);
                data += mask5[-1+2][0+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-1) * 4, row+1,
                           cudaBoundaryModeClamp);
                data += mask5[-1+2][1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+0) * 4, row-1,
                           cudaBoundaryModeClamp);
                data += mask5[0+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+0) * 4, row+1,
                           cudaBoundaryModeClamp);
                data += mask5[0+2][1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row-1,
                           cudaBoundaryModeClamp);
                data += mask5[1+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row+0,
                           cudaBoundaryModeClamp);
                data += mask5[1+2][0+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row+1,
                           cudaBoundaryModeClamp);
                data += mask5[1+2][1+2];
                if(newData - data > eps) newData = data;

                /// for c's
                surf2Dread(&data, surfRef, (col+2) * 4, row-1,
                           cudaBoundaryModeClamp);
                data += mask5[2+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+2) * 4, row+1,
                           cudaBoundaryModeClamp);
                data += mask5[2+2][1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-2) * 4, row-1,
                           cudaBoundaryModeClamp);
                data += mask5[-2+2][-1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-2) * 4, row+1,
                           cudaBoundaryModeClamp);
                data += mask5[-2+2][1+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-1) * 4, row-2,
                           cudaBoundaryModeClamp);
                data += mask5[-1+2][-2+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row-2,
                           cudaBoundaryModeClamp);
                data += mask5[1+2][-2+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col-1) * 4, row+2,
                           cudaBoundaryModeClamp);
                data += mask5[-1+2][2+2];
                if(newData - data > eps) newData = data;

                surf2Dread(&data, surfRef, (col+1) * 4, row+2,
                           cudaBoundaryModeClamp);
                data += mask5[1+2][2+2];
                if(newData - data > eps) newData = data;

                if(newData < oldData) {
                    surf2Dwrite(newData, surfRef, col * 4, row);
                    atomicExch(&found, 1);
                }

                __syncthreads();
                if(mainthread && found > 0 && !written) {
                    atomicExch(done, 0);
                    written = true;
                }
            }
        }
    }
}

__global__
void computeCol(const uchar* src, int* tempOut, int inWidthStride, int width,
                int height) {
	extern __shared__ int imgCol []; // allocates shared memory
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;
	int row, rowi;
	int d;
	int value;
	for (row = threadIdx.x; row < height; row += blockDim.x) {
		imgCol[row] = src[row*inWidthStride+y]; // copy column to shared memory
	}
	__syncthreads();
	for(row = x; row < height; row += blockDim.x) {
		value = imgCol[row];
		if(value != 0) {
			value = width*width + height*height;
			d = 1;
			for(rowi = 1; rowi < height - row; rowi++) { // scan 1
				if(imgCol[row + rowi] == 0)
					value = min(value, d);
				d += 1 + 2 * rowi;
				if(d > value) break;
			}
			d = 1;
			for(rowi = 1; rowi <= row; rowi++) { // scan 2
				if(imgCol[row - rowi] == 0)
					value = min(value, d);
				d += 1 + 2 * rowi;
				if(d > value) break;
			}
		}
		tempOut[row *width + y] = value;
	}
}

__global__
void computeRow(int* tempOut, float* out, int outWidthStride, int width,
                int height) {
	extern __shared__ int imgRow[]; // allocates shared memory
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * width;
	int outY = blockIdx.y * outWidthStride;
 	int col, coli;
 	int value;
 	int d;
 	for(col = threadIdx.x; col < width; col += blockDim.x) {
 		imgRow[col] = tempOut[y + col]; // copy rows to shared memory
 	}
 	__syncthreads();
 	for(col = x; col < width; col += blockDim.x) {
 		value = imgRow[col];
 		if(value != 0) {
	 		d = 1;
			for(coli = 1; coli < width - col; coli++) { // scan 1
				value = min(value, imgRow[col + coli] + d);
				d += 1 + 2 * coli;
				if(d > value) break;
			}
	 		d = 1;
			for(coli = 1; coli <= col; coli++) { // scan 2
				value = min(value, imgRow[col - coli] + d);
				d += 1 + 2 * coli;
				if(d > value) break;
			}
		}
 		out[outY + col] = sqrt((double)value);
	}
}

template <>
RetCode DistanceTransform<float>(cudaStream_t stream,
                                int height,
                                int width,
                                int inWidthStride,
                                const uchar* inData,
                                int outWidthStride,
                                float* outData,
                                DistTypes distanceType,
                                DistanceTransformMasks maskSize) {
    int imageSize = height * width;

    if(maskSize == ppl::cv::DIST_MASK_PRECISE) {
        int MAXTH = 1024;

        int *devTemp;
        cudaMalloc((void **) &devTemp, imageSize * sizeof(int));

        int TH = MAXTH;
        if(height < TH) TH = height;
        int DH = (int) ceil(height/(float)TH);
        dim3 dimGrid(DH, width, 1);
        computeCol<<<dimGrid, TH, height*sizeof(int), stream>>>(inData, devTemp,
          inWidthStride, width, height);

        int TW = MAXTH;
        if(width < TW) TW = width;
        int DW = (int) ceil(width/(float)TW);
        dim3 dimGridr(DW, height, 1);
        computeRow<<<dimGridr, TW, width*sizeof(int), stream>>>(devTemp, outData,
          outWidthStride, width, height);
        cudaStreamSynchronize(stream);

        cudaFree(devTemp);
    }
    else {
        const int BX = 32;
        const int BY = 16;
        dim3 blockSize(BX,BY);
        dim3 gridSize;
        gridSize.x = (width + BX-1)/BX;
        gridSize.y = (height + BY-1)/BY;

        int* done_dev;
        cudaMalloc((void**)&done_dev, sizeof(int));
        int done_host = 1;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
          cudaChannelFormatKindFloat);
        cudaMallocArray(&cuInArray, &channelDesc, width, height,
          cudaArraySurfaceLoadStore);
        cudaBindSurfaceToArray(surfRef, cuInArray, channelDesc);
        initDT<<<gridSize, blockSize, 0, stream>>>(height, width, inWidthStride,
          inData);

        if(maskSize == ppl::cv::DIST_MASK_3) {
            float a, b;
            if(distanceType == ppl::cv::DIST_L1) {
            a = 1.0f;
            b = 2.0f;
            }
            else if(distanceType == ppl::cv::DIST_C) {
                a = 1.0f;
                b = 1.0f;
            }
            else {
                a = 0.955;
                b = 1.3693;
            }
            float mask_host[][3] = {{b,a,b}, {a,0,a}, {b,a,b}};
            cudaMemcpyToSymbolAsync( mask3, mask_host, sizeof(float)*3*3, 0,
              cudaMemcpyHostToDevice, stream);

            while(true) {
                cudaMemsetAsync(done_dev,1,sizeof(int), stream);
                calcDT_3x3<<<gridSize, blockSize, sizeof(int), stream>>>(width,
                  height,done_dev);
                cudaMemcpyAsync(&done_host, done_dev, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                if(done_host > 0) break;
            }
        }
        else {
            float a = 1;
            float b = 1.4;
            float c = 2.1969;
            float mask_host[][5] = { {-1,c,-1,c,-1}, {c,b,a,b,c},
                                {-1,a,0,a,-1},{c,b,a,b,c}, {-1,c,-1,c,-1}};
            cudaMemcpyToSymbolAsync( mask5, mask_host, sizeof(float)*5*5, 0,
              cudaMemcpyHostToDevice, stream);
            while(true) {
                cudaMemsetAsync(done_dev,1,sizeof(int), stream);
                calcDT_5x5<<<gridSize, blockSize, sizeof(int), stream>>>(width,
                  height,done_dev);
                cudaMemcpyAsync(&done_host, done_dev, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                if(done_host > 0) break;
            }
        }

        finalizeDT<<<gridSize, blockSize, 0, stream>>>(height, width,
          outWidthStride, outData);

        cudaFreeArray(cuInArray);
        cudaFree(done_dev);
    }

    return RC_SUCCESS;
}

}  // namespace cuda
}  // namespace cv
}  // namespace ppl
