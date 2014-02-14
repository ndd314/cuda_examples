/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#define WARPS_PER_BLOCK 8
#define ELEMENTS_PER_THREAD 4

#define THREADS_PER_BLOCK (WARPS_PER_BLOCK*32)
#define ELEMENTS_PER_BLOCK (ELEMENTS_PER_THREAD * THREADS_PER_BLOCK)
#define ELEMENTS_PER_WARP (ELEMENTS_PER_THREAD * 32)

#if __CUDA_ARCH__ < 300
#define PARALLEL_HISTS 64
#else
#define PARALLEL_HISTS 8
#endif


__device__ unsigned int __laneID()
{
    unsigned int ret;
    asm("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}


// This memory access pattern improves the 2D locality of pixels processed in each warp.
template<typename data2key_t>
class TwoDLocalMem
{
    public:
        __host__ __device__
        TwoDLocalMem(data2key_t _data, int _width, int _height) : data(_data), width(_width), height(_height)
        {
        };

        __host__
        dim3 block()
        {
            return dim3(8,4,WARPS_PER_BLOCK);
        }

        __host__
        dim3 grid()
        {
            return dim3((width+31)/32, (height+31)/32);
        }


        __device__ __forceinline__
        int warpID()
        {
            return threadIdx.z;
        }

        __device__ __forceinline__
        int linearThreadID()
        {
            return threadIdx.z * 32 + threadIdx.y * 8 + threadIdx.x;
        }

        __device__ __forceinline__
        unsigned int operator[](int element)
        {
            int blockID = threadIdx.z + WARPS_PER_BLOCK * element;
            int x = blockIdx.x * 32 + threadIdx.x + (blockID & 3) * 8;
            int y = blockIdx.y * 32 + threadIdx.y + (blockID >> 2) * 4;

            return data(x,y);
        }

        data2key_t data;
        int width;
        int height;
};



struct DirectHistogram
{

    __device__
    DirectHistogram(int *_bins)
    {
        bins = _bins;
    };

    __forceinline__
    __device__
    void add(int key, int count)
    {
        if (count > 0) atomicAdd(&bins[key], count);
    }

    __device__
    void flush() {}

    int *bins;
};


#if __CUDA_ARCH__ >= 200

__device__
unsigned int warpHistogramWarpReduce(unsigned int myKey, volatile int *iwarpKey, unsigned int *threadKey, const int laneID)
{
    volatile unsigned int *warpKey = (volatile unsigned int *)iwarpKey;
    unsigned int *pThreadKey = &threadKey[laneID];

    *pThreadKey = myKey;

    unsigned int myBallot;

    for (int i=0; i<32; ++i)
    {
        unsigned int *pCurrentKey;

        // Find master thread: all active threads write to same location, one will win
        if (i > 0)
        {
            *warpKey = (unsigned int)pThreadKey;
            pCurrentKey = (unsigned int *)*warpKey;
        }
        else
        {
            // Just use the first lane in the first iteration
            pCurrentKey = &threadKey[0];
        }

        // Current Key is the one from the master thread, compare against mine
        int currentKey = *pCurrentKey;

        // Cast vote over all threads and get the ballot
        unsigned int ballot = __ballot(myKey == currentKey);

        // Housekeeping
        if (myKey == currentKey)
        {
            myBallot = (((int)pCurrentKey) == ((int) pThreadKey)) ? ballot : 0;

            break;
        }
    }

    return __popc(myBallot);
}

#endif

__device__
unsigned int warpHistogramNaive(unsigned int myKey, volatile int *warpKey, unsigned int *threadKey, const int laneID)
{
    return 1;
}


template<class in_T>
__global__
void largeHistogramKernel(in_T keys, int *bins, int n_bins)
{

    __shared__ unsigned int threadKey[WARPS_PER_BLOCK * 32];
    __shared__ int warpKeys[WARPS_PER_BLOCK];

    const int laneID = __laneID();
    const int warpID = keys.warpID();
    const int histID = (blockIdx.y * gridDim.x + blockIdx.x) & (PARALLEL_HISTS-1);

    DirectHistogram histogram(&bins[histID * n_bins]);

    for (int i=0; i < ELEMENTS_PER_THREAD; ++ i)
    {
        int myKey = keys[i];

#if __CUDA_ARCH__ < 300 && __CUDA_ARCH__ >= 200
        int count = warpHistogramWarpReduce(myKey, &warpKeys[warpID], &threadKey[warpID*32], laneID);
#else
        // Kepler has significantly improved atomics throughput. This obsoletes the warp reduction step which is beneficial for Fermi.
        int count = warpHistogramNaive(myKey, &warpKeys[warpID], &threadKey[warpID*32], laneID);
#endif
        histogram.add(myKey, count);
    }

    histogram.flush();

}

__global__
void histogramReduction(int *out, int *bins, int bin_count)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < bin_count)
    {
        int sum = 0;

        for (int k=0; k < PARALLEL_HISTS; ++k)
        {
            sum += bins[idx + k * bin_count];
        }

        out[idx] = sum;
    }

}

int gpuHistogramTempSize(int n_bins)
{
    return n_bins * PARALLEL_HISTS * sizeof(int);
}

template<class data2key_t>
void gpuHistogram(int *output, int n_bins, data2key_t input, int width, int height, int *partial_histograms)
{

    cudaMemsetAsync(partial_histograms, 0, n_bins * PARALLEL_HISTS * sizeof(int));

    TwoDLocalMem<data2key_t> mem(input, width, height);
    largeHistogramKernel<<<mem.grid(), mem.block()>>>(mem, partial_histograms, n_bins);

    histogramReduction<<<(n_bins+511)/512,512>>>(output, partial_histograms, n_bins);
}