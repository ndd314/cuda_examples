////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA SDK samples
#include <timer.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

const char *sSDKsample = "Matrix Multiply CUBLAS (no-qatest)";

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions (in addition to helper_cuda.h)

void inline checkError(cublasStatus_t status, const char *msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("%s", msg);
        exit(EXIT_FAILURE);
    }
}
// end of CUDA Helper Functions

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiplyCUBLAS(int argc, char **argv, int block_size, dim3 dimsA, dim3 dimsB)
{
    // allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = dimsA.y * dimsB.x;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_C      = (float *) malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // create and start timer
    printf("Computing result using CUBLAS cublasSgmemm()...");

    // execute the kernel
    int nIter = 300;

    // CUBLAS version 2.0 API interface
    {
        cublasHandle_t handle;
        checkError(cublasCreate(&handle), "cublasCreate() error!\n");
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        //Perform warmup operation with cublas
        cublasStatus_t ret =
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        dimsB.x, dimsA.y, dimsA.x, &alpha, d_B,
                        dimsB.x, d_A,     dimsA.x, &beta,  d_C, dimsA.x);
        checkError(ret, "cublas Sgemm returned an error!\n");

        // Start Timing (CUBLAS)
        StartTimer();

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        dimsB.x, dimsA.y, dimsA.x, &alpha, d_B,
                        dimsB.x, d_A,     dimsA.x, &beta,  d_C, dimsA.x);
        }

        // check if kernel execution generated and error
        getLastCudaError("CUBLAS Kernel execution failed");
        cudaDeviceSynchronize();

        double dSeconds = GetTimer()/((double)nIter * 1000.0);
        double dNumOps = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
        double gflops = 1.0e-9 * dNumOps/dSeconds;

        printf("done.\n");

        //Log througput, etc
        printf("CUBLAS= %.4f GFlop/s, Time= %.2f(ms), Size = %.0f Ops\n",
               gflops, dSeconds*1000., dNumOps);

        // copy result from device to host
        checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
    }

    printf("Checking CUBLAS computed result for correctness: ");
    bool correct = true;

    for (int i = 0; i < (int)size_C; i++)
    {
        if (fabs(h_C[i] - (dimsA.x * valB)) > 1e-5)
        {
            printf("Error! Matrix[%05d]=%f error term is > 1e-5\n", i, h_C[i]);
            correct = false;
        }
    }

    printf("%s\n", correct ? "OK" : "FAIL");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    cudaDeviceReset();

    if (correct)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[%s] - Starting...\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("       widthA heightA (Width x Height of Matrix A)\n");
        printf("       widthB heightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");
        exit(EXIT_SUCCESS);
    }

    // Find the best possible CUDA capable GPU with the highest perforance
    int devID = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    // use a larger block size for Fermi and above
    int block_size = (props.major < 2) ? 16 : 32;

    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

    if (argc > 2)   // argument 2 must be width of matrix A
    {
        dimsA.x = atoi(argv[2]);
    }

    if (argc > 3)   // argument 3 must be height of matrix A
    {
        dimsA.y = atoi(argv[3]);
    }

    if (argc > 4)   // argument 4 must be width of matrix B
    {
        dimsB.x = atoi(argv[4]);
    }

    if (argc > 5)   // argument 5 must be width of matrix B
    {
        dimsB.y = atoi(argv[5]);
    }

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d) MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    return matrixMultiplyCUBLAS(argc, argv, block_size, dimsA, dimsB);
}
