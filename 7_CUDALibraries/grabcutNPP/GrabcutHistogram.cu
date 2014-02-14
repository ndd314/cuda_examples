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
#include <nppi.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <assert.h>

#include "GPUHistogram.h"

#define INF (255.0f * 255.0f * 3 * 8 + 1)
#define _FIXED(x) rintf(1e1f * (x))

__global__
void loglikelihoodsKernel(float *loglikelihood, const int *fg_histogram, int fg_total, const int *bg_histogram, int bg_total, float alpha, int n_bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float fg_scale = 1.0f / (fg_total + n_bins);
    float bg_scale = 1.0f / (bg_total + n_bins);

    float fg_prob = (fg_histogram[i] + alpha) * fg_scale;
    float bg_prob = (bg_histogram[i] + alpha) * bg_scale;

    loglikelihood[i] = logf(fg_prob / bg_prob);
}

__device__
int getKeyFromColor(uchar4 c)
{
    int r_bin = c.x >> 3;
    int g_bin = c.y >> 3;
    int b_bin = c.z >> 3;

    int key = (b_bin << 10) | (g_bin << 5) | r_bin;

    return key;
}

__global__
void LikelihoodfromHistogramKernel(Npp32s *data, int data_pitch, const float *loglikelihood, const unsigned char *trimap, int trimap_pitch, const uchar4 *image, int image_pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {

        Npp32s *m_line = &data[y*data_pitch];
        const uchar4 *i_line = &image[y*image_pitch];
        const unsigned char *u_line = &trimap[y*trimap_pitch];

        float likelihood;

        switch (u_line[x])
        {

            case 2 :
                {
                    //FG
                    likelihood = INF;
                    break;
                }

            case 0 :
                {
                    //BG
                    likelihood = -INF;
                    break;
                }

            default:
                {
                    likelihood = loglikelihood[getKeyFromColor(i_line[x])];
                    break;
                }

        };

        m_line[x] =  _FIXED(likelihood);
    }
}

cudaError_t HistogramDataTerm(Npp32s *terminals, int terminal_pitch, int *histogram, float *loglikelihood, const uchar4 *image, int image_pitch, const unsigned char *trimap, int trimap_pitch, int width, int height)
{
    thrust::device_ptr<const int> fg_ptr(&histogram[32768]);
    thrust::device_ptr<const int> bg_ptr(histogram);

    int fg_total = thrust::reduce(fg_ptr, fg_ptr + 32768);
    int bg_total = thrust::reduce(bg_ptr, bg_ptr + 32768);

    loglikelihoodsKernel<<<32768/512,512>>>(loglikelihood, &histogram[32768], fg_total, histogram, bg_total, 1.0f, 32768);

    dim3 block(32,4);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

    LikelihoodfromHistogramKernel<<<grid, block>>>(terminals, terminal_pitch/4, loglikelihood, trimap, trimap_pitch, image, image_pitch/4, width, height);

    return cudaGetLastError();
}


struct KeyFromTrimap_t
{

    __host__ __device__
    KeyFromTrimap_t(const uchar4 *_image, int _image_pitch, const unsigned char *_user, int _user_pitch, int _width, int _height)
    {
        image = _image;
        image_pitch = _image_pitch;
        user = _user;
        user_pitch = _user_pitch;
        width = _width;
        height = _height;
    }

    __device__
    unsigned int operator()(int x, int y)
    {
        int key;

        if (x >= width || y >= height)
        {
            key = 65536;
        }
        else
        {
            int u = user[y*user_pitch+x];
            int isFG = (u == 1);
            key = (isFG << 15) | getKeyFromColor(image[y*image_pitch+x]);
        }

        return key;
    }

    const uchar4 *image;
    int image_pitch;
    const unsigned char *user;
    int user_pitch;
    int width;
    int height;
};

cudaError_t HistogramUpdate(int *histogram, int *histogram_temp, const uchar4 *image, int image_pitch, const unsigned char *trimap, int trimap_pitch, int width, int height)
{
    KeyFromTrimap_t keyFromTrimap(image, image_pitch/4, trimap, trimap_pitch, width, height);

    gpuHistogram(histogram, 65537, keyFromTrimap, width, height, histogram_temp);

    return cudaGetLastError();
}

int HistogramGetScratchSize()
{
    return gpuHistogramTempSize(65537);
}

