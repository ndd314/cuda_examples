/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma warning(disable:4819)
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include "FreeImage.h"
#include "Exceptions.h"

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>      // CUDA Runtime
#include <npp.h>               // CUDA NPP Definitions

#include <helper_cuda.h>       // helper for CUDA Error handling and initialization
#include <helper_string.h>     // helper for string parsing

int  g_nDevice = -1;


inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

void printfNPPinfo(int argc, char *argv[])
{
    const char *sComputeCap[] =
    {
        "No CUDA Capable Device Found",
        "Compute 1.0", "Compute 1.1", "Compute 1.2", "Compute 1.3",
        "Compute 2.0", "Compute 2.1", "Compute 3.0", "Compute 3.5", NULL
    };

    const NppLibraryVersion *libVer   = nppGetLibVersion();
    NppGpuComputeCapability computeCap = nppGetGpuComputeCapability();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    if (computeCap != 0 && g_nDevice == -1)
    {
        printf("%s using GPU <%s> with %d SM(s) with", argv[0], nppGetGpuName(), nppGetGpuNumSMs(), sComputeCap[computeCap]);

        if (computeCap > 0)
        {
            printf(" %s\n", sComputeCap[computeCap]);
        }
        else
        {
            printf(" Unknwon Compute Capabilities\n");
        }
    }
    else
    {
        printf("%s\n", sComputeCap[computeCap]);
    }
}

// Error handler for FreeImage library.
//  In case this handler is invoked, it throws an NPP exception.
extern "C" void
FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
    throw npp::Exception(zMessage);
}

std::ostream &
operator <<(std::ostream &rOutputStream, const FIBITMAP &rBitmap)
{
    unsigned int nImageWidth    = FreeImage_GetWidth(const_cast<FIBITMAP *>(&rBitmap));
    unsigned int nImageHeight   = FreeImage_GetHeight(const_cast<FIBITMAP *>(&rBitmap));
    unsigned int nPitch         = FreeImage_GetPitch(const_cast<FIBITMAP *>(&rBitmap));
    unsigned int nBPP           = FreeImage_GetBPP(const_cast<FIBITMAP *>(&rBitmap));

    FREE_IMAGE_COLOR_TYPE eType = FreeImage_GetColorType(const_cast<FIBITMAP *>(&rBitmap));
    BITMAPINFO *pInfo          = FreeImage_GetInfo(const_cast<FIBITMAP *>(&rBitmap));

    rOutputStream << "Size  (" << FreeImage_GetWidth(const_cast<FIBITMAP *>(&rBitmap)) << ", "
                  << FreeImage_GetHeight(const_cast<FIBITMAP *>(&rBitmap)) << ")\n";
    rOutputStream << "Pitch "  << FreeImage_GetPitch(const_cast<FIBITMAP *>(&rBitmap)) << "\n";
    rOutputStream << "Type  ";

    switch (eType)
    {
        case FIC_MINISWHITE:
            rOutputStream << "FIC_MINISWHITE\n";
            break;

        case FIC_MINISBLACK:
            rOutputStream << "FIC_MINISBLACK\n";
            break;

        case FIC_RGB:
            rOutputStream << "FIC_RGB\n";
            break;

        case FIC_PALETTE:
            rOutputStream << "FIC_PALETTE\n";
            break;

        case FIC_RGBALPHA:
            rOutputStream << "FIC_RGBALPHA\n";
            break;

        case FIC_CMYK:
            rOutputStream << "FIC_CMYK\n";
            break;

        default:
            rOutputStream << "Unknown pixel format.\n";
    }

    rOutputStream << "BPP   " << nBPP << std::endl;

    return rOutputStream;
}

int
main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sFilename;
        char *filePath = sdkFindFilePath("Lena.pgm", argv[0]);

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            printf("Error unable to find Lena.pgm\n");
            exit(EXIT_FAILURE);
        }

        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        cudaDeviceInit(argc, (const char **)argv);

        printfNPPinfo(argc, argv);

        if (argc > 1)
        {
            sFilename = argv[1];
        }

        // if we specify the filename at the command line, then we only test sFilename
        // otherwise we will check both sFilename[0,1]
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "freeImageInteropNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "freeImageInteropNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_boxFilterFII.pgm";

        if (argc >= 3)
        {
            sResultFilename = argv[2];
        }

        FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());

        // no signature? try to guess the file format from the file extension
        if (eFormat == FIF_UNKNOWN)
        {
            eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
        }

        NPP_ASSERT(eFormat != FIF_UNKNOWN);
        // check that the plugin has reading capabilities ...
        FIBITMAP *pBitmap;

        if (FreeImage_FIFSupportsReading(eFormat))
        {
            pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
        }

        NPP_ASSERT(pBitmap != 0);
        // Dump the bitmap information to the console
        std::cout << (*pBitmap) << std::endl;
        // make sure this is an 8-bit single channel image
        NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_MINISBLACK);
        NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 8);

        unsigned int nImageWidth  = FreeImage_GetWidth(pBitmap);
        unsigned int nImageHeight = FreeImage_GetHeight(pBitmap);
        unsigned int nSrcPitch    = FreeImage_GetPitch(pBitmap);
        unsigned char *pSrcData  = FreeImage_GetBits(pBitmap);

        int nSrcPitchCUDA;
        Npp8u *pSrcImageCUDA = nppiMalloc_8u_C1(nImageWidth, nImageHeight, &nSrcPitchCUDA);
        NPP_ASSERT_NOT_NULL(pSrcImageCUDA);
        // copy image loaded via FreeImage to into CUDA device memory, i.e.
        // transfer the image-data up to the GPU's video-memory
        NPP_CHECK_CUDA(cudaMemcpy2D(pSrcImageCUDA, nSrcPitchCUDA, pSrcData, nSrcPitch,
                                    nImageWidth, nImageHeight, cudaMemcpyHostToDevice));

        // define size of the box filter
        const NppiSize  oMaskSize   = {7, 7};
        const NppiPoint oMaskAchnor = {0, 0};
        // compute maximal result image size
        const NppiSize  oSizeROI = {nImageWidth  - (oMaskSize.width - 1),
                                    nImageHeight - (oMaskSize.height - 1)
                                   };
        // allocate result image memory
        int nDstPitchCUDA;
        Npp8u *pDstImageCUDA = nppiMalloc_8u_C1(oSizeROI.width, oSizeROI.height, &nDstPitchCUDA);
        NPP_ASSERT_NOT_NULL(pDstImageCUDA);
        NPP_CHECK_NPP(nppiFilterBox_8u_C1R(pSrcImageCUDA, nSrcPitchCUDA, pDstImageCUDA, nDstPitchCUDA,
                                           oSizeROI, oMaskSize, oMaskAchnor));
        // create the result image storage using FreeImage so we can easily
        // save
        FIBITMAP *pResultBitmap = FreeImage_Allocate(oSizeROI.width, oSizeROI.height, 8 /* bits per pixel */);
        NPP_ASSERT_NOT_NULL(pResultBitmap);
        unsigned int nResultPitch   = FreeImage_GetPitch(pResultBitmap);
        unsigned char *pResultData = FreeImage_GetBits(pResultBitmap);

        NPP_CHECK_CUDA(cudaMemcpy2D(pResultData, nResultPitch, pDstImageCUDA, nDstPitchCUDA,
                                    oSizeROI.width, oSizeROI.height, cudaMemcpyDeviceToHost));
        // now save the result image
        bool bSuccess;
        bSuccess = FreeImage_Save(FIF_PGM, pResultBitmap, sResultFilename.c_str(), 0) == TRUE;
        NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");

        //free nppiImage
        nppiFree(pSrcImageCUDA);
        nppiFree(pDstImageCUDA);

        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}
