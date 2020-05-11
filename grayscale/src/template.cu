////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
 * example application.
 * Host code.
 */

 // includes, system
 #include <stdlib.h>
 #include <stdio.h>
 #include <string.h>
 #include <math.h>
 #include "lodepng.h"
 
 // includes CUDA
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 
 // includes, project
 #include <helper_cuda.h>
 #include <helper_functions.h> // helper functions for SDK examples
 
 ////////////////////////////////////////////////////////////////////////////////
 // declaration, forward
 void runTest(int argc, char** argv, int index);
 
 #define COUNT 5
 int BLOCKSIZE = 1;
 
 extern "C" void computeGrayScale(unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* out, unsigned int ARRAYSIZE);
 
 __global__ void testGrayScale(unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* out, unsigned int ARRAYSIZE) {
	 int idx = blockIdx.x * blockDim.x + threadIdx.x;
	 if (idx < ARRAYSIZE) {
		 out[idx] = 0.2126 * r[idx] + 0.7152 * g[idx] + 0.0722 * b[idx];
	 }
 }
 
 float cpuResults[COUNT];
 float gpuCalcResults[COUNT];
 float gpuTotalResults[COUNT];
 
 unsigned height, width;
 unsigned char* r, * g, * b, * a, * out;
 
 ////////////////////////////////////////////////////////////////////////////////
 // Program main
 ////////////////////////////////////////////////////////////////////////////////
 void encodeOneStep(const char* filename) {
	 /*Encode the image*/
	 unsigned char* image = (unsigned char*)malloc(4 * width * height * sizeof(unsigned char));
	 for (int y = 0; y < height; y++) {
		 for (int x = 0; x < width; x++) {
			 /*get RGBA components*/
			 image[4 * y * width + 4 * x + 0] = out[y * width + x]; /*red*/
			 image[4 * y * width + 4 * x + 1] = out[y * width + x]; /*green*/
			 image[4 * y * width + 4 * x + 2] = out[y * width + x]; /*blue*/
			 image[4 * y * width + 4 * x + 3] = a[y * width + x]; /*alpha*/
		 }
	 }
 
	 unsigned error = lodepng_encode32_file(filename, image, width, height);
 
	 /*if there's an error, display it*/
	 if (error)
		 printf("error %u: %s\n", error, lodepng_error_text(error));
	 free(r);
	 free(g);
	 free(b);
	 free(a);
	 free(image);
 
 }
 
 void decodeOneStep(const char* filename) {
	 unsigned error;
	 unsigned char* image = 0;
 
	 error = lodepng_decode32_file(&image, &width, &height, filename);
	 if (error)
		 printf("error %u: %s\n", error, lodepng_error_text(error));
 
	 r = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	 g = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	 b = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	 a = (unsigned char*)malloc(width * height * sizeof(unsigned char));
 
	 for (int y = 0; y < height; y++) {
		 for (int x = 0; x < width; x++) {
			 /*get RGBA components*/
			 r[y * width + x] = image[4 * y * width + 4 * x + 0]; /*red*/
			 g[y * width + x] = image[4 * y * width + 4 * x + 1]; /*green*/
			 b[y * width + x] = image[4 * y * width + 4 * x + 2]; /*blue*/
			 a[y * width + x] = image[4 * y * width + 4 * x + 3]; /*alpha*/
		 }
	 }
	 free(image);
 }
 
 int main(int argc, char** argv) {
	 const char* filename = argc > 1 ? argv[1] : "original.png";
 
	 decodeOneStep(filename);
 
	 out = (unsigned char*)malloc(width * height * sizeof(unsigned char));
 
	 FILE* file;
	 char* str = (char*)malloc(50);
	 snprintf(str, 50, "opgave-3-t.csv");
	 file = fopen(str, "w");
	 free(str);
 
	 fprintf(file, "BLOCKSIZE;CPU_TIME;GPU_CALCULATION;GPU_TOTAL\n");
	 while (BLOCKSIZE <= 1024) {
		 float cpuAvg = 0;
		 float gpuCalcAvg = 0;
		 float gpuTotalAvg = 0;
		 for (int i = 0; i < COUNT; i++) {
			 runTest(argc, argv, i);
			 cpuAvg += cpuResults[i];
			 gpuCalcAvg += gpuCalcResults[i];
			 gpuTotalAvg += gpuTotalResults[i];
		 }
 
		 cpuAvg = cpuAvg / COUNT;
		 gpuCalcAvg = gpuCalcAvg / COUNT;
		 gpuTotalAvg = gpuTotalAvg / COUNT;
		 fprintf(file, "%i;%f;%f;%f\n", BLOCKSIZE, cpuAvg, gpuCalcAvg, gpuTotalAvg);
		 printf("%i;%f;%f;%f\n", BLOCKSIZE, cpuAvg, gpuCalcAvg, gpuTotalAvg);
		 BLOCKSIZE++;
	 }
 
	 encodeOneStep("grayscale-t.png");
 
	 free(out);
	 fclose(file);
 
	 return 0;
 }
 
 ////////////////////////////////////////////////////////////////////////////////
 //! Run a simple test for CUDA
 ////////////////////////////////////////////////////////////////////////////////
 void runTest(int argc, char** argv, int index) { 
	 unsigned int ARRAYSIZE = height * width;
	 unsigned int nBlocks = ARRAYSIZE / BLOCKSIZE + (ARRAYSIZE % BLOCKSIZE == 0 ? 0 : 1);
 
	 //declare variables
	 unsigned char* out_host;
	 unsigned char* r_dev, * g_dev, * b_dev, * out_dev;
	 //allocate arrays on host
	 out_host = (unsigned char*)malloc(ARRAYSIZE * sizeof(unsigned char));
 
	 for (unsigned int i = 0; i < ARRAYSIZE; i++) {
		 out_host[i] = 0;
	 }
 
	 StopWatchInterface* timerCPU = 0;
	 cudaEvent_t startTotal, startCalc, stopTotal, stopCalc;
	 float timeTotal = 0;
	 float timeCalc = 0;
	 cudaEventCreate(&startTotal);
	 cudaEventCreate(&stopTotal);
	 cudaEventCreate(&startCalc);
	 cudaEventCreate(&stopCalc);
	 sdkCreateTimer(&timerCPU);
 
	 //printf("Starting GPU...\n");
	 cudaEventRecord(startTotal);
 
	 //allocate arrays on device
	 cudaMalloc((void**)&r_dev, ARRAYSIZE * sizeof(unsigned char));
	 cudaMalloc((void**)&g_dev, ARRAYSIZE * sizeof(unsigned char));
	 cudaMalloc((void**)&b_dev, ARRAYSIZE * sizeof(unsigned char));
	 cudaMalloc((void**)&out_dev, ARRAYSIZE * sizeof(unsigned char));
 
	 //Step 1: Copy data to GPU memory
 
	 cudaMemcpy(r_dev, r, ARRAYSIZE * sizeof(unsigned char),
		 cudaMemcpyHostToDevice);
	 cudaMemcpy(g_dev, g, ARRAYSIZE * sizeof(unsigned char),
		 cudaMemcpyHostToDevice);
	 cudaMemcpy(b_dev, b, ARRAYSIZE * sizeof(unsigned char),
		 cudaMemcpyHostToDevice);
	 cudaMemcpy(out_dev, out_host, ARRAYSIZE * sizeof(unsigned char),
		 cudaMemcpyHostToDevice);
 
	 //Step 2 & 3: RUN
	 cudaEventRecord(startCalc);
	 testGrayScale << < nBlocks, BLOCKSIZE >> > (r_dev, g_dev, b_dev, out_dev, ARRAYSIZE);
	 cudaEventRecord(stopCalc);
	 cudaEventSynchronize(stopCalc);
 
 
 
	 // check if kernel execution generated and error
	 getLastCudaError("Kernel execution failed");
 
	 //Step 4: Retrieve result
	 cudaMemcpy(out_host, out_dev, ARRAYSIZE * sizeof(unsigned char),
		 cudaMemcpyDeviceToHost);
 
	 cudaEventRecord(stopTotal);
	 cudaEventSynchronize(stopTotal);
	 cudaEventElapsedTime(&timeCalc, startCalc, stopCalc);
	 cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
 
	 //printf("Starting CPU...\n");
 
	 sdkStartTimer(&timerCPU);
 
	 computeGrayScale(r, g, b, out_host, ARRAYSIZE);
 
	 sdkStopTimer(&timerCPU);
 
	 for (unsigned int i = 0; i < ARRAYSIZE; i++) {
		 out[i] = out_host[i];
	 }
 
	 // RESULTS
	 cpuResults[index] = sdkGetTimerValue(&timerCPU);
	 gpuCalcResults[index] = timeCalc;
	 gpuTotalResults[index] = timeTotal;
 
	 //rest of program (Other 4 steps go here)
	 //end of  program
	 //cleanup: VERY IMPORTANT!!!
	 sdkDeleteTimer(&timerCPU);
	 free(out_host);
	 cudaFree(r_dev);
	 cudaFree(g_dev);
	 cudaFree(b_dev);
	 cudaFree(out_dev);
 }
 