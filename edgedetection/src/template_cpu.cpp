/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 ////////////////////////////////////////////////////////////////////////////////
 // export C interface
#include <math.h>

extern "C" void computeGrayScale(unsigned char* r, unsigned char* g, unsigned char* b, unsigned int ARRAYSIZE);

extern "C" void computeEdges(unsigned char* in, unsigned char* out, unsigned int width, unsigned int height);

void computeGrayScale(unsigned char* r, unsigned char* g, unsigned char* b, unsigned int ARRAYSIZE)
{
	for (unsigned int i = 0; i < ARRAYSIZE; i++)
	{
		/*get RGBA components*/
		float tmp = 0.2126 * r[i] / 255 + 0.7152 * g[i] / 255 + 0.0722 * b[i] / 255;
		r[i] = 255 * tmp;
		g[i] = 255 * tmp;
		b[i] = 255 * tmp;
	}
}

void computeEdges(unsigned char* in, unsigned char* out, unsigned int width, unsigned int height)
{

	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float tmpVert = -1 * in[(y - 1) * width + x - 1] + 0 * in[(y - 1) * width + x] + 1 * in[(y - 1) * width + x + 1];
			tmpVert += -2 * in[y * width + x - 1] + 0 * in[y * width + x] + 2 * in[y * width + x + 1];
			tmpVert += -1 * in[(y + 1) * width + x - 1] + 0 * in[(y + 1) * width + x] + 1 * in[(y + 1) * width + x + 1];

			float tmpHor = 1 * in[(y - 1) * width + x - 1] + 2 * in[(y - 1) * width + x] + 1 * in[(y - 1) * width + x + 1];
			tmpHor += 0 * in[y * width + x - 1] + 0 * in[y * width + x] + 0 * in[y * width + x + 1];
			tmpHor += -1 * in[(y + 1) * width + x - 1] + -2 * in[(y + 1) * width + x] + -1 * in[(y + 1) * width + x + 1];

			float mag = sqrt(tmpVert * tmpVert + tmpHor * tmpHor);

			if (mag > 255)
			{
				mag = 255;
			}
			out[y * width + x] = mag;
		}
	}
}
