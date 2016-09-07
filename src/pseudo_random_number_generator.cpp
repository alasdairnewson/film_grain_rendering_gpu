

#include "pseudo_random_number_generator.h"

/* 
 * From http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
 * Same strategy as in Gabor noise by example
 * Apply hashtable to create cellseed
 * Use a linear congruential generator as fast PRNG
 */

/**
 * 
 */
 /**
* @brief Produce random seed
*
* @param input seed
* @return output, modified seed
*/
unsigned int wang_hash(unsigned int seed)
{
  seed = (seed ^ 61u) ^ (seed >> 16u);
  seed *= 9u;
  seed = seed ^ (seed >> 4u);
  seed *= 668265261u;
  seed = seed ^ (seed >> 15u);
  return(seed);
}

/**
 * 
 */
 /**
* @brief Generate unique seed for a a cell, given the coordinates of the cell
*
* @param (x,y) : input coordinates of the cell
* @param (x,y) : constant offset to change seeds for cells
* @return seed for this cell
*/
unsigned int cellseed(unsigned int x, unsigned int y, unsigned int offset)
{
  const unsigned int period = 65536u; // 65536 = 2^16
  unsigned int s = (( y % period) * period + (x % period)) + offset;
  if (s == 0u) s = 1u;
  return(s);
}

/**
 * 
 */
 /**
* @brief Initialise internal state of pseudo-random number generator
*
* @param : pointer to the state of the pseudo-random number generator
* @param : seed for initialisation
* @return : void
*/
void mysrand(noise_prng *p, const unsigned int seed)
{
    unsigned int s=seed;
    p->state = wang_hash(s);
}

/**
 * 
 */
 /**
* @brief Produce a pseudo-random number and increment the internal state of the pseudo-random number generator
*
* @param : pointer to the state of the pseudo-random number generator
* @return : random integer from 0 to max_unsigned_int (4294967295)
*/
unsigned int myrand(noise_prng *p)
{
// linear congruential generator: procudes correlated output. Similar patterns are visible
// p.state = 1664525u * p.state + 1013904223u;
// Xorshift algorithm from George Marsaglia's paper
  p->state ^= (p->state << 13u);
  p->state ^= (p->state >> 17u);
  p->state ^= (p->state << 5u);  
  return(p->state);
}

/**
 * 
 */
 /**
* @brief Produce uniform random number in the interval [0;1]
*
* @param : pointer to the state of the pseudo-random number generator
* @return : random floating point number in the interval [0;1]
*/
float myrand_uniform_0_1(noise_prng *p)
{
    return(((float) myrand(p)) / ((float) 4294967295u));
}

/**
 * 
 */
 /**
* @brief Produce random number following a standard normal distribution
*
* @param : pointer to the state of the pseudo-random number generator
* @return : random number following a standard normal distribution
*/
float myrand_gaussian_0_1(noise_prng *p)
{
    /* Box-Muller method for generating standard Gaussian variate */
    float u = myrand_uniform_0_1(p);
    float v = myrand_uniform_0_1(p);
    return( sqrt(-2.0 * log(u)) * cos(2.0 * pi * v) );
}

/**
 * 
 */
 /**
* @brief Produce pair of random numbers following a standard normal distribution
*
* @param : pointer to the state of the pseudo-random number generator
* @return : pair of random numbers following a standard normal distribution
*/
vec2d myrand_gaussian_vec2d(noise_prng *p)
{
    // Box-Muller method for generating standard Gaussian variate
    float u = myrand_uniform_0_1(p);
    float v = myrand_uniform_0_1(p);

	vec2d randOut;
	//sqrt(-2.0 * log(u)) * vec2(cos(2.0 * pi * v), sin(2.0 * pi * v)))
	randOut.x = sqrt(-2.0 * log(u)) * cos(2.0 * pi * v);
	randOut.y = sqrt(-2.0 * log(u)) * sin(2.0 * pi * v);
    return(randOut);
}


/**
 * 
 */
 /**
* @brief Produce a random number following a Poisson distribution
*
* @param : pointer to the state of the pseudo-random number generator
* @param : lambda, parameter of the Poisson distribution
* @param : optional value so that exp(-lambda) need not be recalculated each time we call the Poisson random number generator
* @return : random number following a poisson distribution
*/
unsigned int my_rand_poisson(noise_prng *p, const float lambda, float prodIn)
{
    /* Inverse transform sampling */
    float u=myrand_uniform_0_1(p);
    unsigned int x=0u;
    float prod;
    if (prodIn <= 0)
      prod = exp(-lambda); /* this should be passed as an argument if used extensively with the same value lambda */
    else
      prod = prodIn;
    float sum = prod;
    while ( (u>sum) && (x<floor(10000.0f*lambda)))
    {
			x = x + 1u;
			prod = prod * lambda /((float) x);
			sum = sum + prod;
    }
    return(x);
}