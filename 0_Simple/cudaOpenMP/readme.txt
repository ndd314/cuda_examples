This sample demonstrates how to use OpenMP for multiGPU programming with CUDA.  Note that a single CPU thread can interact only with one CUDA device.  Thus, to take advantage of multiple CUDA devices in a system, one must write a multi-threaded CPU application as well.  OpenMP provides a simple and easy to learn API for creating and managing CPU threads.  OpenMP standard is supported by compilers on multiple platforms and operating systems (gcc 4.2 or later is required for OpenMP).

The sample detects and displays the CUDA devices available on the system.  As many CPU threads as there are CUDA devices are created, one per CUDA device.  It is possible to create more CPU threads than there are CUDa devices (see comments in the code), though in that case some GPUs will be given work by multiple CPU threads.  Note that all variables declared inside an omp parallel scope are by default local to each thread.

The sample uses multiple GPUs to increment an array by a constant.  The array size is proportional to the number of detected CUDA devices.  Thus, if the number of CPU threads is greater than the number of GPUs, make sure that the number of CPU threads divides the total input size (and that each kernel launch gets a multiple of 128 input elements, since threadblock is hardcoded to be 128 threads).

C++ compilers that support OpenMP:
 MS Visual Studio v8 (Professional and Team editions) and later,
 Intel C++
 GCC 4.2 and later