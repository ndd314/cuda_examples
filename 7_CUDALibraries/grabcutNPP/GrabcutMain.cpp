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

#pragma warning(disable:4819)

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif


#ifdef WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "helper_string.h"

#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <cuda_gl_interop.h>

#include <npp.h>

#include "GrabCut.h"
#include "FreeImage.h"

#ifndef min
#define min(a,b) ((a < b) ? a:b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a:b)
#endif

#define MAX_IMAGE_SIZE 1024.0f

const char *default_image = "flower.ppm";
const char *gold_image = "flower_gold.png";

bool g_bQATest = false;
bool g_bDisplay = true;
int  g_nDevice = 0;
std::string sFilename, sReferenceFile;

const int handle_radius = 4;
int mouseDown_x, mouseDown_y;
int active_handle = -1;
NppiRect rect;

int display_mode = 1;

GLuint pbo;
GLuint texture;

int width, height;
uchar4 *d_image;
size_t image_pitch;

unsigned char *d_trimap;
size_t trimap_pitch;

GrabCut *grabcut;
int neighborhood = 8;

struct cudaGraphicsResource *pbo_resource;

void initGL(int *argc, char **argv, int w, int h);
void saveResult(const char *filename);
bool verifyResult(const char *filename);

// Functions from GrabcutUtil.cu
cudaError_t TrimapFromRect(Npp8u *alpha, int alpha_pitch, NppiRect rect, int width, int height);
cudaError_t ApplyMatte(int mode, uchar4 *result, int result_pitch, const uchar4 *image, int image_pitch, const unsigned char *matte, int matte_pitch, int width, int height);

inline int cudaDeviceInit()
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = g_nDevice;

    if (dev < 0)
    {
        dev = 0;
    }

    if (dev > deviceCount-1)
    {
        std::cerr << std::endl << ">> %d CUDA capable GPU device(s) detected. <<" << deviceCount << std::endl;
        std::cerr <<">> cudaDeviceInit (-device=" << dev << ") is not a valid GPU device. <<" << std::endl << std::endl;
        return -dev;
    }
    else
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;
    }

    if (g_bDisplay)
    {
        checkCudaErrors(cudaGLSetGLDevice(dev));
    }
    else
    {
        checkCudaErrors(cudaSetDevice(dev));
    }

    return dev;
}

void parseCommandLineArguments(int argc, char *argv[])
{
    if (argc >= 2)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "nodisplay"))
        {
            g_bDisplay = false;
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            g_nDevice = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            char* pFilename = 0;
            getCmdLineArgumentString(argc, (const char **)argv, "input", &pFilename);
            if( pFilename ) sFilename = pFilename;
        }
        if (checkCmdLineFlag(argc, (const char **)argv, "verify"))
        {
            char* pReferenceFile = 0;
            getCmdLineArgumentString(argc, (const char **)argv, "verify", &pReferenceFile);
            if( pReferenceFile) sReferenceFile = pReferenceFile;
            g_bQATest = true;
            g_bDisplay = false;
        }
    }
}

NppGpuComputeCapability printfNPPinfo(char *argv[])
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

    return computeCap;
}


int main(int argc, char **argv)
{

    printf("%s Starting...\n\n", argv[0]);

    // Parse the command line arguments for proper configuration
    parseCommandLineArguments(argc, argv);

    if (sFilename.length() == 0 || g_bQATest)
    {
        char *filePath = sdkFindFilePath(default_image, argv[0]);

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            printf("Was unable to find %s from %s\n", default_image, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    cudaDeviceInit();

    if (printfNPPinfo(argv) < NPP_CUDA_1_1)
    {
        printf("Insufficient Compute Capability (must be >= 1.1)\n");
        exit(EXIT_SUCCESS);
    }

    // if we specify the filename at the command line, then we only test sFilename
    int file_errors = 0;

    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
        std::cout << "Opened: <" << sFilename.data() << "> successfully!" << std::endl;
        file_errors = 0;
        infile.close();
    }
    else
    {
        std::cout << "Unable to open: <" << sFilename.data() << ">" << std::endl;
        file_errors++;
        infile.close();
    }

    if (file_errors > 0)
    {
        exit(EXIT_FAILURE);
    }

    // Load the image file
    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());

    if (eFormat == FIF_UNKNOWN)
    {
        eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
    }

    FIBITMAP *pBitmap = NULL;
    FIBITMAP *p4Bitmap= NULL;

    if (FreeImage_FIFSupportsReading(eFormat))
    {
        pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
    }


    width = FreeImage_GetWidth(pBitmap);;
    height = FreeImage_GetHeight(pBitmap);

    if (width > MAX_IMAGE_SIZE || height > MAX_IMAGE_SIZE)
    {

        float scale_factor = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height);

        FIBITMAP *pResampled = FreeImage_Rescale(pBitmap, (int)(scale_factor * width), (int)(scale_factor * height), FILTER_BICUBIC);
        width = FreeImage_GetWidth(pResampled);
        height = FreeImage_GetHeight(pResampled);

        p4Bitmap = FreeImage_ConvertTo32Bits(pResampled);

        FreeImage_Unload(pResampled);

    }
    else
    {
        p4Bitmap = FreeImage_ConvertTo32Bits(pBitmap);
    }

    if (!g_bQATest && g_bDisplay)
    {
        initGL(&argc, argv, width, height);
    }

    checkCudaErrors(cudaMallocPitch(&d_image, &image_pitch, width * sizeof(uchar4), height));
    checkCudaErrors(cudaMemcpy2D(d_image, image_pitch, FreeImage_GetBits(p4Bitmap) , FreeImage_GetPitch(p4Bitmap), width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

    FreeImage_Unload(p4Bitmap);
    FreeImage_Unload(pBitmap);

    checkCudaErrors(cudaMallocPitch(&d_trimap, &trimap_pitch, width, height));

    // Setup GrabCut
    grabcut = new GrabCut(d_image, (int) image_pitch, d_trimap, (int) trimap_pitch, width, height);

    // Default selection rectangle
    rect.x = (int) ceil(width * 0.1);
    rect.y = (int) ceil(height * 0.1);
    rect.width = width - 2 * rect.x;
    rect.height = height - 2 * rect.y;

    checkCudaErrors(TrimapFromRect(d_trimap, (int) trimap_pitch, rect, width, height));

    grabcut->computeSegmentationFromTrimap();

    if (!g_bQATest && g_bDisplay)
    {
        glutMainLoop();
    }

    int qaStatus = EXIT_SUCCESS;

    if (g_bQATest)
    {
        qaStatus = verifyResult(sdkFindFilePath((char *)sReferenceFile.c_str(), argv[0])) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // Cleanup
    delete grabcut;

    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_trimap));

    exit(qaStatus);
}

void plotUserRect()
{
    glEnable(GL_COLOR_LOGIC_OP);
    glLogicOp(GL_XOR);

    NppiRect fliprect = rect;
    fliprect.y = height - 1 - (rect.y + rect.height - 1) ;

    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_LINE_LOOP);
    glVertex2i(fliprect.x, fliprect.y);
    glVertex2i(fliprect.x + fliprect.width, fliprect.y);
    glVertex2i(fliprect.x + fliprect.width, fliprect.y + fliprect.height);
    glVertex2i(fliprect.x, fliprect.y + fliprect.height);
    glEnd();
    glDisable(GL_COLOR_LOGIC_OP);

    // Handles
    glColor3f(0.75f, 0.75f, 0.75f);

    int center_x = fliprect.x + fliprect.width / 2;
    int center_y = fliprect.y + fliprect.height / 2;

    glRecti(fliprect.x - handle_radius, fliprect.y - handle_radius, fliprect.x + handle_radius, fliprect.y + handle_radius);
    glRecti(fliprect.x - handle_radius, fliprect.y + fliprect.height - handle_radius, fliprect.x + handle_radius, fliprect.y + fliprect.height + handle_radius);

    glRecti(fliprect.x + fliprect.width - handle_radius, fliprect.y - handle_radius, fliprect.x + fliprect.width + handle_radius, fliprect.y + handle_radius);
    glRecti(fliprect.x + fliprect.width - handle_radius, fliprect.y + fliprect.height - handle_radius, fliprect.x + fliprect.width + handle_radius, fliprect.y + fliprect.height + handle_radius);

    glRecti(center_x - handle_radius, fliprect.y - handle_radius, center_x + handle_radius, fliprect.y + handle_radius);
    glRecti(center_x - handle_radius, fliprect.y + fliprect.height - handle_radius, center_x + handle_radius, fliprect.y + fliprect.height + handle_radius);

    glRecti(fliprect.x - handle_radius, center_y - handle_radius, fliprect.x + handle_radius, center_y + handle_radius);
    glRecti(fliprect.x + fliprect.width - handle_radius, center_y - handle_radius, fliprect.x + fliprect.width + handle_radius, center_y + handle_radius);
}

void display()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &pbo_resource, 0));
    uchar4 *pbo_pointer;
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &pbo_pointer, &num_bytes,
                                                         pbo_resource));

    checkCudaErrors(ApplyMatte(display_mode, pbo_pointer, width*4, d_image, (int) image_pitch, grabcut->getAlpha(), grabcut->getAlphaPitch(), width, height));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &pbo_resource, 0));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    width, height,
                    GL_BGRA, GL_UNSIGNED_BYTE, NULL);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // Plot  Image

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(0.0, 0.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f((GLfloat)width, 0.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f((GLfloat)width, (GLfloat)height, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(0, (GLfloat)height, 0.5);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    // Plot User Rectangle
    if (display_mode == 1)
    {
        plotUserRect();
    }

    glutSwapBuffers();
}

void reshape(int w, int h)
{
    glViewport(0,0,w,h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

}

bool inside_rect(int x, int y, int x1, int y1, int x2, int y2)
{
    return(x >= x1 && x <= x2 && y >= y1 && y <= y2);
}

void mouseClick(int button, int state, int x, int y)
{
    if (display_mode != 1)
    {
        return;
    }

    if (button == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_DOWN)
        {
            mouseDown_x = x;
            mouseDown_y = y;

            int center_x = rect.x + rect.width / 2;
            int center_y = rect.y + rect.height / 2;

            active_handle = -1;

            if (inside_rect(x, y, rect.x - handle_radius, rect.y - handle_radius, rect.x + handle_radius, rect.y + handle_radius))
            {
                active_handle = 0;
            }

            if (inside_rect(x, y, rect.x - handle_radius, rect.y + rect.height - handle_radius, rect.x + handle_radius, rect.y + rect.height + handle_radius))
            {
                active_handle = 6;
            }

            if (inside_rect(x, y, rect.x + rect.width - handle_radius, rect.y - handle_radius, rect.x + rect.width + handle_radius, rect.y + handle_radius))
            {
                active_handle = 2;
            }

            if (inside_rect(x, y, rect.x + rect.width - handle_radius, rect.y + rect.height - handle_radius, rect.x + rect.width + handle_radius, rect.y + rect.height + handle_radius))
            {
                active_handle = 4;
            }

            if (inside_rect(x, y, center_x - handle_radius, rect.y - handle_radius, center_x + handle_radius, rect.y + handle_radius))
            {
                active_handle = 1;
            }

            if (inside_rect(x, y, center_x - handle_radius, rect.y + rect.height - handle_radius, center_x + handle_radius, rect.y + rect.height + handle_radius))
            {
                active_handle = 5;
            }

            if (inside_rect(x, y, rect.x - handle_radius, center_y - handle_radius, rect.x + handle_radius, center_y + handle_radius))
            {
                active_handle = 7;
            }

            if (inside_rect(x, y, rect.x + rect.width - handle_radius, center_y - handle_radius, rect.x + rect.width + handle_radius, center_y + handle_radius))
            {
                active_handle = 3;
            }

            if (active_handle == -1 && inside_rect(x,y, rect.x-handle_radius, rect.y-handle_radius, rect.x+rect.width+handle_radius, rect.height+rect.height+handle_radius) && !inside_rect(x,y, rect.x+handle_radius, rect.y+handle_radius, rect.x+rect.width-handle_radius, rect.y+rect.height-handle_radius))
            {
                active_handle = 8;
            }
        }
        else
        {
            if (active_handle >= 0)
            {
                active_handle = -1;
                checkCudaErrors(TrimapFromRect(d_trimap, (int) trimap_pitch, rect, width, height));
                grabcut->computeSegmentationFromTrimap();
                glutPostRedisplay();
            }
        }
    }
}

void mouseMotion(int x, int y)
{

    if (active_handle >= 0)
    {
        if (active_handle >= 0 && active_handle <= 2)
        {
            y = max(y, 0);
            int diff = y - mouseDown_y;
            rect.y += diff;
            rect.height -= diff;
            mouseDown_y = y;
        }

        if (active_handle >= 2 && active_handle <=4)
        {
            x = min(x, width-1);
            int diff = x - mouseDown_x;
            rect.width += diff;
            mouseDown_x = x;
        }

        if (active_handle >=4 && active_handle <= 6)
        {
            y = min(y, height-1);

            int diff = y - mouseDown_y;
            rect.height += diff;
            mouseDown_y = y;
        }

        if ((active_handle >= 6 && active_handle <= 7) || active_handle == 0)
        {
            x = max(x, 0);

            int diff = x - mouseDown_x;
            rect.x += diff;
            rect.width -= diff;
            mouseDown_x = x;
        }

        if (active_handle == 8)
        {
            rect.x += x - mouseDown_x;
            rect.y += y - mouseDown_y;

            mouseDown_x = x;
            mouseDown_y = y;
        }

        glutPostRedisplay();
    }

}

void keyboard(unsigned char key, int x, int y)
{
    if (key == 27)
    {
        exit(EXIT_SUCCESS);
    }

    if (key == ' ')
    {
        display_mode = (display_mode + 1) % 3;
        glutPostRedisplay();
    }

    if (key == 's')
    {
        saveResult("Result.png");
    }

    if (key == 'n')
    {
        neighborhood = neighborhood == 4 ? 8 : 4;
        grabcut->setNeighborhood(neighborhood);
        glutPostRedisplay();
    }

}

void initGL(int *argc, char **argv, int w, int h)
{

    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(w, h);
    glutCreateWindow("NPP GrabCut - <space> to toggle view, <n> to toggle neighborhood, <s> to save result");
    glewInit();

    glViewport(0, 0, w,h);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouseClick);
    glutMotionFunc(mouseMotion);
    glutKeyboardFunc(keyboard);

    // Create Texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glBufferData(GL_ARRAY_BUFFER, w * h * 4, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&pbo_resource, pbo, cudaGraphicsMapFlagsNone));
}


void saveResult(const char *filename)
{
    uchar4 *d_result;
    size_t result_pitch;

    checkCudaErrors(cudaMallocPitch(&d_result, &result_pitch, width*4, height));

    ApplyMatte(2, d_result, (int) result_pitch, d_image, (int) image_pitch, grabcut->getAlpha(), grabcut->getAlphaPitch(), width, height);

    FIBITMAP *h_Image = FreeImage_Allocate(width, height, 32);

    checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(h_Image) , FreeImage_GetPitch(h_Image), d_result, result_pitch, width * 4, height, cudaMemcpyDeviceToHost));

    FreeImage_Save(FIF_PNG, h_Image, filename, 0);

    FreeImage_Unload(h_Image);

    checkCudaErrors(cudaFree(d_result));

    printf("Saved result as %s\n", filename);
}

bool verifyResult(const char *filename)
{
    uchar4 *d_result;
    size_t result_pitch;

    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(filename);
    FIBITMAP *goldImage = FreeImage_Load(eFormat, filename);

    if (goldImage == 0)
    {
        printf("Could not open gold image: %s\n", filename);
        return false;
    }

    if (FreeImage_GetHeight(goldImage) != (unsigned int)height || FreeImage_GetWidth(goldImage) != (unsigned int)width)
    {
        printf("Gold image size != result image size\n");
        FreeImage_Unload(goldImage);
        return false;
    }

    checkCudaErrors(cudaMallocPitch(&d_result, &result_pitch, width*4, height));

    ApplyMatte(2, d_result, (int) result_pitch, d_image, (int) image_pitch, grabcut->getAlpha(), grabcut->getAlphaPitch(), width, height);

    FIBITMAP *h_Image = FreeImage_Allocate(width, height, 32);
    checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(h_Image) , FreeImage_GetPitch(h_Image), d_result, result_pitch, width * 4, height, cudaMemcpyDeviceToHost));


    bool result = true;

    int bytespp = FreeImage_GetLine(h_Image) / FreeImage_GetWidth(h_Image);

    for (int y = 0; y < height; y++)
    {
        BYTE *goldBits = FreeImage_GetScanLine(goldImage, y);
        BYTE *resultBits = FreeImage_GetScanLine(h_Image, y);

        for (int x = 0; x < width * bytespp; x++)
        {
            if (goldBits[x] != resultBits[x])
            {
                result = false;
            }
        }
    }
    printf("Checking grabcut results with reference file <%s>\n", filename);
    printf("Images %s\n", result ? "Match!" : "Mismatched!");

    FreeImage_Unload(h_Image);
    FreeImage_Unload(goldImage);
    checkCudaErrors(cudaFree(d_result));

    return result;
}


