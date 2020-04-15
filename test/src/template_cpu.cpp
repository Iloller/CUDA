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
extern "C" void computeSwap(int* in, int* out, int length);

void computeSwap(int* in, int* out, int length) {

	for (unsigned int i = 0; i < length; i++) {
		out[length-1-i] = in[i];
	}
}
