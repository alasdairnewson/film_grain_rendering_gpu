

#ifndef FILM_GRAIN_RENDERING_H
#define FILM_GRAIN_RENDERING_H
	#include <vector>
	#include <cmath>
	#include <iostream> 
	#include <sys/time.h>
	#include <fstream>
	#include <cuda.h>
	
	#include "matrix.h"

	#define PIXEL_WISE 0
	#define GRAIN_WISE 1

	#define MAX_GREY_LEVEL 255
	#define EPSILON_GREY_LEVEL 0.1

	const float pi = 3.14159265358979323846f;


	template <typename T>
	struct filmGrainOptionsStruct {
		T muR;
		T sigmaR;
		T sigmaFilter;
		unsigned int NmonteCarlo;
		unsigned int algorithmID;
		float s;
		float xA;
		float yA;
		float xB;
		float yB;
		unsigned int mOut;
		unsigned int nOut;
		unsigned int grainSeed;
	};

	struct vec2d
	{
		float x;
		float y;
	};


	float* film_grain_rendering_pixel_wise_cuda(const float *src_im, int widthIn, int heightIn,
		int widthOut, int heightOut, filmGrainOptionsStruct<float> filmGrainOptions);
#endif
