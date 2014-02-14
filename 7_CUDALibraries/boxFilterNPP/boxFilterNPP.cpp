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

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>


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

        // if we specify the filename at the command line, then we only test sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "boxFilterNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">" << std::endl;
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

        sResultFilename += "_boxFilter.pgm";

        if (argc >= 3 && !g_bQATest)
        {
            sResultFilename = argv[2];
        }

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);
        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        // create struct with box-filter mask size
        NppiSize oMaskSize = {5, 5};
        // create struct with ROI size given the current mask
        NppiSize oSizeROI = {oDeviceSrc.width() - oMaskSize.width + 1, oDeviceSrc.height() - oMaskSize.height + 1};
        // allocate device image of appropriatedly reduced size
        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
        // set anchor point inside the mask to (0, 0)
        NppiPoint oAnchor = {0, 0};
        // run box filter
        NppStatus eStatusNPP;
        eStatusNPP = nppiFilterBox_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                          oDeviceDst.data(), oDeviceDst.pitch(),
                                          oSizeROI, oMaskSize, oAnchor);
        NPP_ASSERT(NPP_NO_ERROR == eStatusNPP);
        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;

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
        return -1;
    }

    return 0;
}
