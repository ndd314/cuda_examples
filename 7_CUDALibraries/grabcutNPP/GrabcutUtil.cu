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

#include "nppi.h"

#define _FIXED(x) rintf(1e1f * (x))


template<class vec_a, class vec_b>
__device__
float vector_distance_2(vec_a a, vec_b b)
{
    return ((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
}


texture<uchar4, 2, cudaReadModeElementType> imageTex;

__global__
void MeanEdgeStrengthReductionKernel(int width, int height, float *scratch_mem)
{
    __shared__ volatile float s_sum[8][32];

    int y = blockIdx.y * 32 + threadIdx.y * 4;
    int x = blockIdx.x * 32 + threadIdx.x;

    float sum = 0.0f;

    for (int k=0; k < 4; ++k)
    {
        if ((x > 0) && (y > 0) && (x < width-1) && (y < height-1))
        {

            uchar4 pixel = tex2D(imageTex, x + 0.5f ,y + 0.5f);
            float3 center = make_float3(pixel.x, pixel.y, pixel.z);

            sum += vector_distance_2(center, tex2D(imageTex, x - 0.5f ,y + 1.5f));
            sum += vector_distance_2(center, tex2D(imageTex, x + 0.5f ,y + 1.5f));
            sum += vector_distance_2(center, tex2D(imageTex, x + 1.5f ,y + 1.5f));
            sum += vector_distance_2(center, tex2D(imageTex, x + 1.5f ,y + 0.5f));
        }

        ++y;
    }

    // Reduce for each global GMM element
    s_sum[threadIdx.y][threadIdx.x] = sum;

    // Warp Reductions
    sum += s_sum[threadIdx.y][(threadIdx.x + 16) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    sum += s_sum[threadIdx.y][(threadIdx.x + 8) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    sum += s_sum[threadIdx.y][(threadIdx.x + 4) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    sum += s_sum[threadIdx.y][(threadIdx.x + 2) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    sum += s_sum[threadIdx.y][(threadIdx.x + 1) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    __syncthreads();

    // Final Reduction
    if (threadIdx.y ==0 && threadIdx.x == 0)
    {
        for (int j=1; j<8; ++j)
        {
            sum += s_sum[j][0];
        }

        scratch_mem[blockIdx.y * gridDim.x + blockIdx.x] = sum / (4.0f * (width-2.0f) * (height-2.0f));
    }
}

__global__
void MeanEdgeStrengthFinalKernel(float *scratch_mem, int N)
{
    __shared__ volatile float s_sum[4][32];


    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int N_threads = blockDim.x * blockDim.y;

    float sum = idx < N ? scratch_mem[idx] : 0.0f;

    for (idx += N_threads; idx < N; idx += N_threads)
    {
        sum += scratch_mem[idx];
    }

    s_sum[threadIdx.y][threadIdx.x] = sum;

    // Warp Reduction
    sum += s_sum[threadIdx.y][(threadIdx.x + 16) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    sum += s_sum[threadIdx.y][(threadIdx.x + 8) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    sum += s_sum[threadIdx.y][(threadIdx.x + 4) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    sum += s_sum[threadIdx.y][(threadIdx.x + 2) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    sum += s_sum[threadIdx.y][(threadIdx.x + 1) & 31];
    s_sum[threadIdx.y][threadIdx.x] = sum;

    __syncthreads();

    if (threadIdx.y ==0 && threadIdx.x == 0)
    {
        for (int j=1; j<4; ++j)
        {
            sum += s_sum[j][0];
        }

        // Store beta
        scratch_mem[0] = 1.0f/(2.0f *  sum);
    }

}

__device__
Npp32f edge_weight(float3 zm, uchar4 zn, float alpha, float beta, float recp_dist)
{
    return recp_dist * alpha * expf(-beta * (vector_distance_2(zm, make_float3(zn.x, zn.y, zn.z)))) + 3.0f;
}


__global__
void EdgeCuesKernel(float alpha, const float *g_beta, Npp32s *g_left_transposed, Npp32s *g_right_transposed, Npp32s *g_top, Npp32s *g_bottom, Npp32s *g_topleft, Npp32s *g_topright, Npp32s *g_bottomleft, Npp32s *g_bottomright, int pitch, int transposed_pitch, int width, int height)
{

    __shared__ Npp32s s_right[32][33];

    int y0 = blockIdx.y * 32;
    int x0 = blockIdx.x * 32;

    int x = x0+threadIdx.x;

    const float beta = g_beta[0];

    for (int i=threadIdx.y; i < 32; i+=blockDim.y)
    {
        int y = y0 + i;

        if (x < width && y < height)
        {

            uchar4 pixel = tex2D(imageTex, x + 0.5f ,y + 0.5f);
            float3 center = make_float3(pixel.x, pixel.y, pixel.z);

            // Left/Right
            s_right[i][threadIdx.x] = edge_weight(center, tex2D(imageTex, x + 1.5f ,y + 0.5f), alpha, beta, 1.0f);

            // Top/Bottom
            Npp32s bottom;
            bottom = _FIXED(edge_weight(center, tex2D(imageTex, x + 0.5f ,y + 1.5f), alpha, beta, 1.0f));

            if (y < height -1)
            {
                g_bottom[y * pitch + x] = bottom;
                g_top[(y+1) * pitch + x] = bottom;
            }
            else
            {
                g_bottom[y * pitch + x] = 0;
                g_top[x] = 0;
            }

            // Bottomright
            Npp32s bottomright;
            bottomright = _FIXED(edge_weight(center, tex2D(imageTex, x + 1.5f ,y + 1.5f), alpha, beta, 1.0f / sqrtf(2.0f)));

            if (y < height-1 && x < width-1)
            {
                g_bottomright[y * pitch + x] = bottomright;
            }
            else
            {
                g_bottomright[y * pitch + x] = 0;
            }

            // Bottomleft
            Npp32s bottomleft;
            bottomleft = _FIXED(edge_weight(center, tex2D(imageTex, x - 0.5f ,y + 1.5f), alpha, beta, 1.0f / sqrtf(2.0f)));

            if (y < height-1 && x > 0)
            {
                g_bottomleft[y * pitch + x] = bottomleft;
            }
            else
            {
                g_bottomleft[y * pitch + x] = 0;
            }

            // topright
            Npp32s topright;
            topright = _FIXED(edge_weight(center, tex2D(imageTex, x + 1.5f ,y - 0.5f), alpha, beta, 1.0f / sqrtf(2.0f)));

            if (y > 0 && x < width-1)
            {
                g_topright[y * pitch + x] = topright;
            }
            else
            {
                g_topright[y * pitch + x] = 0;
            }

            // topleft
            Npp32s topleft;
            topleft = _FIXED(edge_weight(center, tex2D(imageTex, x - 0.5f ,y - 0.5f), alpha, beta, 1.0f / sqrtf(2.0f)));

            if (y > 0  && x > 0)
            {
                g_topleft[y * pitch + x] = topleft;
            }
            else
            {
                g_topleft[y * pitch + x] = 0;
            }

        }
    }

    __syncthreads();

    int y = y0 + threadIdx.x;

    for (int i=threadIdx.y; i < 32; i+=blockDim.y)
    {

        int x = x0 + i;

        if (x < width && y < height)
        {
            if (x < width - 1)
            {
                g_right_transposed[x * transposed_pitch +y] = s_right[threadIdx.x][i];
                g_left_transposed[(x+1) * transposed_pitch +y] = s_right[threadIdx.x][i];
            }
            else
            {
                g_right_transposed[x * transposed_pitch +y] = 0;
                g_left_transposed[y] = 0;
            }
        }
    }
}

cudaError_t EdgeCues(float alpha, const uchar4 *image, int image_pitch, Npp32s *left_transposed, Npp32s *right_transposed, Npp32s *top, Npp32s *bottom, Npp32s *topleft, Npp32s *topright, Npp32s *bottomleft, Npp32s *bottomright, int pitch, int transposed_pitch, int width, int height, float *scratch_mem)
{
    cudaError_t error;

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc<uchar4>();

    error = cudaBindTexture2D(NULL, imageTex, image, channelDesc, width, height,  image_pitch);

    if (error != cudaSuccess)
    {
        return error;
    }

    dim3 grid((width+31) / 32, (height+31) / 32);
    dim3 block(32,4);
    dim3 large_block(32,8);

    MeanEdgeStrengthReductionKernel<<<grid, large_block>>>(width, height, scratch_mem);
    MeanEdgeStrengthFinalKernel<<<1,block>>>(scratch_mem, grid.x *grid.y);

    EdgeCuesKernel<<<grid, block>>>(alpha , scratch_mem, left_transposed, right_transposed, top, bottom, topleft, topright, bottomleft, bottomright, pitch / 4, transposed_pitch/ 4, width, height);

    error = cudaUnbindTexture(imageTex);
    return error;
}


__global__
void SegmentationChangedKernel(int *g_changed, Npp8u *alpha_old, Npp8u *alpha_new, int alpha_pitch, int width, int height)
{
#if __CUDA_ARCH__ < 200
    __shared__ int s_changed;
    s_changed = 0;
    __syncthreads();
#endif

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    int changed = 0;

    for (int k=0; k < 4; ++k)
    {
        if (x < width && y < height)
        {
            // Check if the lsb is equal
            changed |= (alpha_old[y * alpha_pitch+x] ^ alpha_new[y * alpha_pitch+x]) & 1;
        }

        y += blockDim.y;
    }


#if __CUDA_ARCH__ < 200

    if (changed > 0)
    {
        s_changed = 1;
    }

    __syncthreads();

    if (threadIdx.y == 0 && s_changed > 0)
    {
        g_changed[0] = 1;
    }

#else

    if (__syncthreads_or(changed > 0))
    {
        if (threadIdx.y == 0)
        {
            g_changed[0] = 1;
        }
    }

#endif
}


cudaError_t SegmentationChanged(bool &result, int *d_changed, Npp8u *alpha_old, Npp8u *alpha_new, int alpha_pitch, int width, int height)
{
    cudaError_t error;
    dim3 grid((width+31) / 32, (height+31) / 32);
    dim3 block(32,8);

    error = cudaMemsetAsync(d_changed,0,4);

    if (error != cudaSuccess)
    {
        return error;
    }

    SegmentationChangedKernel<<<grid, block>>>(d_changed, alpha_old, alpha_new, alpha_pitch, width, height);

    int h_changed;
    error = cudaMemcpy(&h_changed, d_changed, 4, cudaMemcpyDeviceToHost);

    result = (h_changed != 0);
    return error;
}


struct boxfilter_functor
{
    __device__
    uchar4 operator()(const uchar4 &a, const uchar4 &b, const uchar4 &c, const uchar4 &d)
    {
        float4 r = make_float4(0.25f * a.x, 0.25f * a.y, 0.25f * a.z, 0.25f * a.w);

        r.x += 0.25f * b.x;
        r.y += 0.25f * b.y;
        r.z += 0.25f * b.z;
        r.w += 0.25f * b.w;
        r.x += 0.25f * c.x;
        r.y += 0.25f * c.y;
        r.z += 0.25f * c.z;
        r.w += 0.25f * c.w;
        r.x += 0.25f * d.x;
        r.y += 0.25f * d.y;
        r.z += 0.25f * d.z;
        r.w += 0.25f * d.w;

        return make_uchar4(rintf(r.x), rintf(r.y), rintf(r.z), rintf(r.w));
    }
};

struct maxfilter_functor
{
    __device__
    unsigned char operator()(const unsigned char &a, const unsigned char &b, const unsigned char &c, const unsigned char &d)
    {
        return max(max(max(a,b),c),d);
    }
};

struct minfilter_functor
{
    __device__
    unsigned char operator()(const unsigned char &a, const unsigned char &b, const unsigned char &c, const unsigned char &d)
    {
        return min(min(min(a,b),c),d);
    }
};


template<class T>
__device__
T clamp_read(int y0, int x0, const T *image, int pitch, int width, int height)
{
    int x = min(x0, width-1);
    int y = min(y0, height-1);

    return image[y * pitch + x];
}

template<class T, class functor_t>
__global__
void downscaleKernel(T *small_image, int small_pitch, int small_width, int small_height, const T *image, int pitch, int width, int height, functor_t functor)
{
    __shared__ T tile[16][64];

    int x0 = blockIdx.x * 64 + threadIdx.x;
    int y0 = blockIdx.y * 64 + threadIdx.y;

    int small_x0 = blockIdx.x * 32 + threadIdx.x;
    int small_y0 = blockIdx.y * 32 + threadIdx.y;


    for (int k=0; k < 4; ++k)
    {
        int y = y0 + k * 16;

        tile[threadIdx.y][threadIdx.x] = clamp_read(y,x0,image,pitch, width, height);
        tile[threadIdx.y][threadIdx.x+32] =  clamp_read(y,x0+32,image,pitch, width, height);
        tile[threadIdx.y+8][threadIdx.x+32] = clamp_read(y+8,x0+32,image,pitch, width, height);
        tile[threadIdx.y+8][threadIdx.x] = clamp_read(y+8,x0,image,pitch, width, height);

        __syncthreads();

        int small_y = small_y0 + k * 8;

        if (small_y < small_height && small_x0 < small_width)
        {
            small_image[small_y * small_pitch + small_x0] = functor(tile[2 * threadIdx.y][2 * threadIdx.x],
                                                                    tile[2 * threadIdx.y][2 * threadIdx.x+1],
                                                                    tile[2 * threadIdx.y+1][2 * threadIdx.x+1],
                                                                    tile[2 * threadIdx.y+1][2 * threadIdx.x]);
        }
    }
}

cudaError_t downscale(uchar4 *small_image, int small_pitch, int small_width, int small_height, const uchar4 *image, int pitch, int width, int height)
{

    dim3 grid((width + 63)/64, (height+63)/64);
    dim3 block(32,8);

    downscaleKernel<<<grid, block>>>(small_image, small_pitch/4, small_width, small_height, image, pitch/4, width, height, boxfilter_functor());

    return cudaGetLastError();
}

cudaError_t downscaleTrimap(unsigned char *small_image, int small_pitch, int small_width, int small_height, const unsigned char *image, int pitch, int width, int height)
{

    dim3 grid((width + 63)/64, (height+63)/64);
    dim3 block(32,8);

    downscaleKernel<<<grid, block>>>(small_image, small_pitch, small_width, small_height, image, pitch, width, height, maxfilter_functor());
    return cudaGetLastError();
}


__global__
void upsampleAlphaKernel(unsigned char *alpha, unsigned char *small_alpha, int alpha_pitch, int width, int height, int shift)
{
    int x = blockIdx.x * 128 + threadIdx.x * 4;
    int y0 = blockIdx.y * 32 + threadIdx.y;

    uchar4 *alpha4 = (uchar4 *) alpha;
    int alpha4_pitch = alpha_pitch / 4;

    for (int k=0; k<4; ++k)
    {
        int y = y0 + k*8;
        uchar4 output;

        if (x < width && y < height)
        {
            output.x = small_alpha[(y >> shift) * alpha_pitch + (x >> shift)];
            output.y = small_alpha[(y >> shift) * alpha_pitch + ((x+1) >> shift)];
            output.z = small_alpha[(y >> shift) * alpha_pitch + ((x+2) >> shift)];
            output.w = small_alpha[(y >> shift) * alpha_pitch + ((x+3) >> shift)];

            alpha4[y * alpha4_pitch + blockIdx.x * 32 + threadIdx.x] = output;
        }
    }
}

cudaError_t upsampleAlpha(unsigned char *alpha, unsigned char *small_alpha, int alpha_pitch, int width, int height, int small_width, int small_height)
{
    dim3 grid((width+127)/128, (height+31)/32);
    dim3 block(32,8);

    int factor = width / small_width;
    int shift = 0;

    while (factor > (1<<shift))
    {
        shift++;
    }

    upsampleAlphaKernel<<<grid, block>>>(alpha, small_alpha, alpha_pitch, width, height, shift);

    return cudaGetLastError();
}

__global__
void TrimapFromRectKernel(Npp8u *alpha, int alpha_pitch, NppiRect rect, int width, int height)
{

    Npp32u *alpha4 = (Npp32u *)alpha;
    int alpha4_pitch = alpha_pitch / 4;

    int x0 = blockIdx.x * 32 + threadIdx.x;
    int y0 = blockIdx.y * 32;

    int x = x0 * 4;

    for (int i=threadIdx.y; i<32; i+=blockDim.y)
    {

        int y = y0 + i;

        if (x< width && y < height)
        {

            if (y >= rect.y && y < (rect.y + rect.height))
            {
                int first_x = min(max(0, rect.x - x),4);
                int last_x = min(max(0,x - (rect.x + rect.width - 4)),4);

                unsigned int pattern = 0x001010101u;
                unsigned int mask    = 0x0ffffffffu;

                alpha4[y * alpha4_pitch + x0] = (pattern << (first_x *8)) & (mask >> (last_x*8));
            }
            else
            {
                alpha4[y * alpha4_pitch + x0] = 0;
            }

        }
    }

}

cudaError_t TrimapFromRect(Npp8u *alpha, int alpha_pitch, NppiRect rect, int width, int height)
{
    dim3 block(32,8);
    dim3 grid((width+(block.x*4)-1) / (block.x*4), (height+31) / 32);

    rect.y = height - 1 - (rect.y + rect.height - 1) ; // Flip horizontal (FreeImage inverts y axis)

    TrimapFromRectKernel<<<grid, block>>>(alpha, alpha_pitch, rect, width, height);

    return cudaGetLastError();
}

__device__
uchar4 filter(uchar4 color)
{
    float hue = 0.33f * (color.x + color.y + color.z);

    return make_uchar4(rintf(hue * 0.6f), rintf(hue * 0.3f), rintf(hue * 1.0f), color.w);
}


template<int mode>
__global__
void ApplyMatteKernel(uchar4 *result, int result_pitch, const uchar4 *image, int image_pitch, const unsigned char *matte, int matte_pitch, int width, int height)
{

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    for (int k=0; k < 4; ++k)
    {
        if (x < width && y < height)
        {
            uchar4 pixel = image[ y * image_pitch + x];

            if (mode == 1)
            {
                if (((matte[y * matte_pitch + x]) & 1) == 0)
                {
                    pixel = filter(pixel);
                }
            }

            if (mode == 2)
            {
                if (((matte[y * matte_pitch + x]) & 1)  == 0)
                {
                    pixel = make_uchar4(0,0,0,0);
                }
            }

            result[y * result_pitch + x] = pixel;
        }

        y += blockDim.y;
    }

}


cudaError_t ApplyMatte(int mode, uchar4 *result, int result_pitch, const uchar4 *image, int image_pitch, const unsigned char *matte, int matte_pitch, int width, int height)
{
    dim3 block(32,8);
    dim3 grid((width+31) / 32, (height+31) / 32);

    switch (mode)
    {
        case 0 :
            ApplyMatteKernel<0><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
            break;

        case 1 :
            ApplyMatteKernel<1><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
            break;

        case 2 :
            ApplyMatteKernel<2><<<grid, block>>>(result, result_pitch/4, image, image_pitch/4, matte, matte_pitch, width, height);
            break;

    }

    return cudaGetLastError();
}

__global__
void convertRGBToRGBAKernel(uchar4 *i4, int i4_pitch, uchar3 *i3, int i3_pitch, int width, int height)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    unsigned char *i3_linear = (unsigned char *) i3;
    unsigned char *i4_linear = (unsigned char *) i4;

    for (int k=0; k < 4; ++k)
    {
        if (x < width && y < height)
        {
            uchar3 *i3_line = (uchar3 *)(i3_linear + y*i3_pitch);
            uchar4 *i4_line = (uchar4 *)(i4_linear + y*i4_pitch);

            uchar3 pixel = i3_line[x];
            i4_line[x] = make_uchar4(pixel.x, pixel.y, pixel.z, 255);
        }

        y += blockDim.y;
    }
}


cudaError_t convertRGBToRGBA(uchar4 *i4, int i4_pitch, uchar3 *i3, int i3_pitch, int width, int height)
{
    dim3 block(32,8);
    dim3 grid((width + 31)/32, (height+31)/32);

    convertRGBToRGBAKernel<<<grid, block>>>(i4, i4_pitch, i3, i3_pitch, width, height);

    return cudaGetLastError();
}

