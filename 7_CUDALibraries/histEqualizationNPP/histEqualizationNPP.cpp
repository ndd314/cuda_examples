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

#pragma warning (disable:4819)

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>
#include <string.h>

#include <string>
#include <fstream>
#include <iostream>

#include <npp.h>

#include <helper_cuda.h>

#ifdef WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif

bool g_bQATest = false;
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
        printf("%s using GPU <%s> with %d SM(s) with", argv[0], nppGetGpuName(), nppGetGpuNumSMs());

        if (computeCap > 0)
        {
            printf(" %s\n", sComputeCap[computeCap]);
        }
        else
        {
            printf(" Unknown Compute Capabilities\n");
        }
    }
    else
    {
        printf("%s\n", sComputeCap[computeCap]);
    }
}

int main(int argc, char *argv[])
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

        cudaDeviceInit(argc, (const char **)argv);

        printfNPPinfo(argc, argv);

        if (g_bQATest == false && (g_nDevice == -1) && argc > 1)
        {
            sFilename = argv[1];
        }

        // if we specify the filename at the command line, then we only test sFilename.
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "histEqualizationNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "histEqualizationNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string dstFileName = sFilename;

        std::string::size_type dot = dstFileName.rfind('.');

        if (dot != std::string::npos)
        {
            dstFileName = dstFileName.substr(0, dot);
        }

        dstFileName += "_histEqualization.pgm";

        if (argc >= 3 && !g_bQATest)
        {
            dstFileName = argv[2];
        }

        npp::ImageCPU_8u_C1 oHostSrc;
        npp::loadImage(sFilename, oHostSrc);
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        //
        // allocate arrays for histogram and levels
        //

        const int binCount = 255;
        const int levelCount = binCount + 1; // levels array has one more element

        Npp32s *histDevice = 0;
        Npp32s *levelsDevice = 0;

        NPP_CHECK_CUDA(cudaMalloc((void **)&histDevice,   binCount   * sizeof(Npp32s)));
        NPP_CHECK_CUDA(cudaMalloc((void **)&levelsDevice, levelCount * sizeof(Npp32s)));

        //
        // compute histogram
        //

        NppiSize oSizeROI = {oDeviceSrc.width(), oDeviceSrc.height()}; // full image
        // create device scratch buffer for nppiHistogram
        int nDeviceBufferSize;
        nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, levelCount ,&nDeviceBufferSize);
        Npp8u *pDeviceBuffer;
        NPP_CHECK_CUDA(cudaMalloc((void **)&pDeviceBuffer, nDeviceBufferSize));

        // compute levels values on host
        Npp32s levelsHost[levelCount];
        NPP_CHECK_NPP(nppiEvenLevelsHost_32s(levelsHost, levelCount, 0, levelCount));
        // compute the histogram
        NPP_CHECK_NPP(nppiHistogramEven_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI,
                                               histDevice, levelCount, 0, binCount,
                                               pDeviceBuffer));
        // copy histogram and levels to host memory
        Npp32s histHost[binCount];
        NPP_CHECK_CUDA(cudaMemcpy(histHost, histDevice, binCount * sizeof(Npp32s), cudaMemcpyDeviceToHost));

        Npp32s  lutHost[binCount + 1];

        // fill LUT
        {
            Npp32s *pHostHistogram = histHost;
            Npp32s totalSum = 0;

            for (; pHostHistogram < histHost + binCount; ++pHostHistogram)
            {
                totalSum += *pHostHistogram;
            }

            NPP_ASSERT(totalSum == oSizeROI.width * oSizeROI.height);

            if (totalSum == 0)
            {
                totalSum = 1;
            }

            float multiplier = 1.0f / float(totalSum) * 0xFF;

            Npp32s runningSum = 0;
            Npp32s *pLookupTable = lutHost;

            for (pHostHistogram = histHost; pHostHistogram < histHost + binCount; ++pHostHistogram)
            {
                *pLookupTable = (Npp32s)(runningSum * multiplier + 0.5f);
                pLookupTable++;
                runningSum += *pHostHistogram;
            }

            lutHost[binCount] = 0xFF; // last element is always 1
        }

        //
        // apply LUT transformation to the image
        //
        // Create a device image for the result.
        npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.size());

#if CUDART_VERSION >= 5000
        // Note for CUDA 5.0, that nppiLUT_Linear_8u_C1R requires these pointers to be in GPU device memory
        Npp32s  *lutDevice  = 0;
        Npp32s  *lvlsDevice = 0;

        NPP_CHECK_CUDA(cudaMalloc((void **)&lutDevice,    sizeof(Npp32s) * (binCount + 1)));
        NPP_CHECK_CUDA(cudaMalloc((void **)&lvlsDevice,   sizeof(Npp32s) * (binCount + 1)));

        NPP_CHECK_CUDA(cudaMemcpy(lutDevice , lutHost,    sizeof(Npp32s) * (binCount+1), cudaMemcpyHostToDevice));
        NPP_CHECK_CUDA(cudaMemcpy(lvlsDevice, levelsHost, sizeof(Npp32s) * (binCount+1), cudaMemcpyHostToDevice));

        NPP_CHECK_NPP(nppiLUT_Linear_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                            oDeviceDst.data(), oDeviceDst.pitch(),
                                            oSizeROI,
                                            lutDevice, // value and level arrays are in GPU device memory
                                            lvlsDevice,
                                            binCount+1));

        NPP_CHECK_CUDA(cudaFree(lutDevice));
        NPP_CHECK_CUDA(cudaFree(lvlsDevice));
#else
        NPP_CHECK_NPP(nppiLUT_Linear_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                            oDeviceDst.data(), oDeviceDst.pitch(),
                                            oSizeROI,
                                            lutHost, // value and level arrays are in host memory
                                            levelsHost,
                                            binCount+1));
#endif

        // copy the result image back into the storage that contained the
        // input image
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        // save the result
        npp::saveImage(dstFileName.c_str(), oHostDst);

        std::cout << "Saved image file " << dstFileName << std::endl;
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

    return 0;
}

