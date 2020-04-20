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
extern "C" void computeGrayScale(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char* out, unsigned int ARRAYSIZE);

void computeGrayScale(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char* out, unsigned int ARRAYSIZE)
{
	for (unsigned int i = 0; i < ARRAYSIZE; i++)
	{
		/*get RGBA components*/
		float tmp = 0.2126 * r[i] / 255 + 0.7152 * g[i] / 255 + 0.0722 * b[i] / 255;
		out[i] = 255 * tmp;
	}
}

