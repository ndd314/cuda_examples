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

#ifndef _GRABCUT_H_
#define _GRABCUT_H_

#include <cuda_runtime.h>
#include <nppi.h>

class GrabCut
{

    public:
        GrabCut(const uchar4 *image, int image_pitch, const unsigned char *trimap, int trimap_pitch, int width, int height);
        ~GrabCut();

        void computeSegmentationFromTrimap();
        void updateSegmentation();

        void updateImage(const uchar4 *image);

        void setNeighborhood(int n)
        {
            m_neighborhood = n;
            computeSegmentationFromTrimap();
        };

        const unsigned char *getAlpha() const
        {
            return d_alpha[current_alpha];
        }
        int getAlphaPitch() const
        {
            return (int)alpha_pitch;
        };

    private:
        void createSmallImage(int max_dim);
        void createSmallTrimap();

        uchar4 *d_image;
        size_t image_pitch;
        float edge_strength;

        uchar4 *d_small_image;
        size_t small_pitch;
        NppiSize small_size;

        const unsigned char *d_trimap;
        int trimap_pitch;

        unsigned char *d_small_trimap[2];
        size_t small_trimap_pitch[2];
        int small_trimap_idx;

        NppiSize size;

        Npp32s *d_terminals;
        Npp32s *d_left_transposed, *d_right_transposed;
        Npp32s *d_top, *d_bottom, *d_topleft, *d_topright, *d_bottomleft, *d_bottomright;
        size_t pitch, transposed_pitch;
        int m_neighborhood;

        unsigned char *d_alpha[2];
        size_t alpha_pitch;
        int current_alpha;

        Npp8u *d_scratch_mem;
        NppiGraphcutState *pState;

        float *d_gmm;
        size_t gmm_pitch;

        int *d_histogram;

        int gmms;
        int blocks;

        cudaEvent_t start, stop;
};


#endif //_GRABCUT_H_