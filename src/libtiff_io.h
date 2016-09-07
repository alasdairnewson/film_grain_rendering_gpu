
#ifndef LIBTIFF_IO_H
#define LIBTIFF_IO_H

	#include <iostream>
	#include <cmath>
	#include <cstring>
	#include <tiffio.h>
	#include "matrix.h"

	#define MAX_CHANNELS 3

	float* read_tiff_image(const char* inputFile, uint32 *width,
		uint32 *height, uint32 *nChannels);

	int write_tiff_image(float* inputImg, unsigned int n, unsigned int m,
	unsigned int nChannels, const char* outputFile);
#endif
