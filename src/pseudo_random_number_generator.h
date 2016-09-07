

#ifndef PSEUDO_RANDOM_NUMBER_GENERATOR_H
#define PSEUDO_RANDOM_NUMBER_GENERATOR_H

	#include <cmath>
	#include <math.h>
	#include <iostream>

	const float pi = 3.14159265358979323846f;
	struct noise_prng
	{
		unsigned int state;
	};

	struct vec2d
	{
		float x;
		float y;
	};


	unsigned int wang_hash(unsigned int seed);
	unsigned int cellseed(unsigned int x, unsigned int y, unsigned int offset);
	void mysrand(noise_prng *p, const unsigned int seed);
	unsigned int myrand(noise_prng *p);
	float myrand_uniform_0_1(noise_prng *p);
	float myrand_gaussian_0_1(noise_prng *p);
	float my_rand_exponential(noise_prng *p, const float lambda);
	unsigned int my_rand_poisson(noise_prng *p, const float lambda, float prodIn=-1);

#endif