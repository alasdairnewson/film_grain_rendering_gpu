
#include "libtiff_io.h"
#include "film_grain_rendering.h"


/* Texture declaration */
texture<float4,2,cudaReadModeElementType> tex_src_im;
texture <float,1,cudaReadModeElementType> tex_lambda_list;
texture <float,1,cudaReadModeElementType> tex_exp_lambda_list;


#define CUDA_CALL(x) do { if((x) != cudaSuccess ) {printf(" Error at %s :% d \n" , __FILE__ , __LINE__ ); printf(" Error type : %s\n",cudaGetErrorString(x));return(NULL);}} while (0)

/**
 * This functions interlace_rgb_image takes an deinterlaced rgb image im and transforms it into an interlaced rgb image.
 * 
 */
__host__ void interlace_rgb_image(float *im, size_t w, size_t h)
{
    float *temp;
    unsigned int i;
    temp = (float *) malloc( 3*w*h*sizeof(float));
    size_t imsize = w*h;
    
    for(i=0; i<imsize; i++)
    {
        temp[3*i] = im[i];
        temp[3*i+1] = im[i+imsize];
        temp[3*i+2] = im[i+2*imsize];
    }
    for(i=0; i<3*imsize; i++)
    {
        im[i] = temp[i];
    }
    
    free(temp);
}


/**
 * Add alpha channel to float interlaced rgb image.
 */
__host__ float4* add_interlaced_alpha_channel(const float *im, size_t w, size_t h)
{
    
    float4* out;
    unsigned int i;
    if(NULL == (out = (float4 *) malloc( w*h*sizeof(float4)))) 
        printf("error in float4 array allocation\n");
    size_t imsize = w*h;
    
    for(i=0; i<imsize; i++)
    {
        (out[i]).x = im[i];
        (out[i]).y = im[i];
        (out[i]).z = im[i];
        (out[i]).w = 255.;
    }
    return(out);
}




/*********************************************/
/*****   PSEUDO-RANDOM NUMBER GENERATOR   ****/
/*********************************************/

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
__device__ unsigned int wang_hash(unsigned int seed)
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
__device__ unsigned int cellseed(unsigned int x, unsigned int y, unsigned int offset)
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
__device__ void mysrand(unsigned int  *p, const unsigned int seed)
{
    unsigned int s=seed;
    *p = wang_hash(s);
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
__device__ unsigned int myrand(unsigned int  *p)
{
// linear congruential generator: procudes correlated output. Similar patterns are visible
// p.state = 1664525u * p.state + 1013904223u;
// Xorshift algorithm from George Marsaglia's paper
  *p ^= (*p << 13u);
  *p ^= (*p >> 17u);
  *p ^= (*p << 5u);  
  return(*p);
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
__device__ float myrand_uniform_0_1(unsigned int *p)
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
__device__ float myrand_gaussian_0_1(unsigned int  *p)
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
__device__ vec2d myrand_gaussian_vec2d(unsigned int  *p)
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
__device__ unsigned int my_rand_poisson(unsigned int *prngstate, float lambda, float expLambda)
{
	// Inverse transform sampling
	float u=myrand_uniform_0_1(prngstate);
	unsigned int x = 0u;
	//float prod = exp(-lambda); // this should be passed as an argument if used extensively with the same value lambda
	float prod = expLambda;
	float sum = prod;
	while ( (u>sum) && (x<floor(10000.0f*lambda)))
	{
		x = x + 1u;
		prod = prod * lambda /((float) x);
		sum = sum + prod;
	}
	return(x);

    //return unsigned(floor(lambda + 0.5 + (sqrt(lambda) * myrand_gaussian_0_1(prngstate))));
}


/*********************************************/
/*********     GRAIN RENDERING     ***********/
/*********************************************/

/**
 * 
 */
 /**
* @brief Square distance 
*
* @param lambda parameter of the Poisson process
* @param x1, y1 : x, y coordinates of the first point
* @param x2, y2 : x, y coordinates of the second point
* @return squared Euclidean distance
*/
__device__ float sqDistance(const float x1, const float y1, const float x2, const float y2)
{
	return((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

/**
 * 
 */
 /**
* @brief Render one pixel in the pixel-wise algorithm
*
* @param imgIn input image
* @param yOut, xOut : coordinates of the output pixel
* @param mIn, nIn : input image size
* @param mOut, nOut : output image size
* @param offset : offset to put into the pseudo-random number generator
* @param nMonteCarlo : number of iterations in the Monte Carlo simulation
* @param grainRadius : average grain radius
* @param grainSigma : standard deviation of the grain radius
* @param sigmaFilter : standard deviation of the blur kernel
* @param (xA,yA), (xB,yB) : limits of image to render
* @return output value of the pixel
*/
__global__ void kernel( float *out_im, size_t out_w, size_t out_h, 
                        size_t src_w, size_t src_h,
                        float grainRadius, float grainSigma, int nMonteCarlo, float s, float sigmaFilter, unsigned int offsetRand, float xA, float yA, float xB, float yB)
{
    // map from blockIdx to pixel position
    float x = (float)(threadIdx.x + blockIdx.x * blockDim.x);
    float y = (float)(threadIdx.y + blockIdx.y * blockDim.y);

	float normalQuantile = 3.0902;//2.3263;	//standard normal quantile for alpha=0.999
	float logNormalQuantile;
	float grainRadiusSq = grainRadius*grainRadius;
	float currRadius,currGrainRadiusSq;
	float mu, sigma, sigmaSq;
	float maxRadius = grainRadius;



    float ag = 1/ceil(1/grainRadius);
    float sX = ((float)(out_w-1))/((float)(xB-xA)); 
    float sY = ((float)(out_h-1))/((float)(yB-yA));

	//calculate the mu and sigma for the lognormal distribution
	if (grainSigma > 0.0)
	{
		sigma = sqrt(log( (grainSigma/grainRadius)*(grainSigma/grainRadius) + (float)1.0));
		sigmaSq = sigma*sigma;
		mu = log(grainRadius)-sigmaSq/((float)2.0);
		logNormalQuantile = exp(mu + sigma*normalQuantile);
		maxRadius = logNormalQuantile;
	}

    if( (x<out_w) && (y<out_h) )
    {
		float pixOut=0.0, u;
		//unsigned int offsetRand = 2;

		//conversion from output grid (xOut,yOut) to input grid (xIn,yIn)
		//we inspect the middle of the output pixel (1/2)
		//the size of a pixel is (xB-xA)/nOut
		x = xA + (x+(float)0.5 ) * ((xB-xA)/((float)out_w));
		y = yA + (y+(float)0.5 ) * ((yB-yA)/((float)out_h));

		// Simulate Poisson process on the 4 neighborhood cells of (x,y)
		unsigned int pMonteCarlo;
		unsigned int p;
		mysrand(&pMonteCarlo, ((unsigned int)2016)*(offsetRand));

		for (int i=0; i<nMonteCarlo; i++)
		{

			float xGaussian = myrand_gaussian_0_1(&pMonteCarlo);
			float yGaussian = myrand_gaussian_0_1(&pMonteCarlo);

			xGaussian = x + sigmaFilter*(xGaussian)/sX;
			yGaussian = y + sigmaFilter*(yGaussian)/sY;

			// Compute the Poisson parameters for the pixel that contains (x,y)
			/*float4 src_im_sxsy = tex2D(tex_src_im, (int)max(floor(xGaussian),0.0), (int)max(floor(yGaussian),0.0));
			u = src_im_sxsy.x;
			u = u/(uMax+epsilon);
			lambda = -((ag*ag)/( pi*(grainRadiusSq + grainSigma*grainSigma) )) * log(1.0f-u);*/

			//determine the bounding boxes around the current shifted pixel
			int minX = floor( (xGaussian - maxRadius)/ag);
			int maxX = floor( (xGaussian + maxRadius)/ag);
			int minY = floor( (yGaussian - maxRadius)/ag);
			int maxY = floor( (yGaussian + maxRadius)/ag);

			bool ptCovered = false; // used to break all for loops

			for(int ncx = minX; ncx <= maxX; ncx++) // x-cell number
			{
				if(ptCovered == true)
				break;
				for(int ncy = minY; ncy <= maxY; ncy++) // y-cell number
				{
					if(ptCovered == true)
						break;
					float cellCornerX = ag*((float)ncx);
					float cellCornerY = ag*((float)ncy);

					unsigned int seed = cellseed(ncx, ncy, offsetRand);
					mysrand(&p,seed);


					// Compute the Poisson parameters for the pixel that contains (x,y)
					float4 src_im_sxsy = tex2D(tex_src_im, (int)max(floor(xGaussian),0.0), (int)max(floor(yGaussian),0.0));
					u = src_im_sxsy.x;
					int uInd = (int)floor( u*( (float)MAX_GREY_LEVEL + (float)EPSILON_GREY_LEVEL) );
					float currLambda = tex1D(tex_lambda_list,uInd);
					float currExpLambda = tex1D(tex_exp_lambda_list,uInd);

					/*float currLambda = lambda;
					float currExpLambda = exp(-lambda);
					if((floor(cellCornerX) != floor(xGaussian)) || (floor(cellCornerY) != floor(yGaussian)))
					{
						float4 src_im_temp =
						tex2D(tex_src_im, (int)max(floor(cellCornerX),0.0), (int)max(floor(cellCornerY),0.0));
						// Compute the Poisson parameters for the pixel that contains (x,y)
						u = src_im_temp.x;
						u = u/(uMax+epsilon);
						currLambda = -((ag*ag)/( pi*(grainRadiusSq + grainSigma*grainSigma))) * log(1.0f-u);
						currLambda = exp(-lambda);
					}*/

					unsigned int Ncell = my_rand_poisson(&p, currLambda,currExpLambda);


					for(unsigned int k=0; k<Ncell; k++)
					{
						//draw the grain centre
						float xCentreGrain = cellCornerX + ag*myrand_uniform_0_1(&p);
						float yCentreGrain = cellCornerY + ag*myrand_uniform_0_1(&p);

						//draw the grain radius
						if (grainSigma>0.0)
						{
							//draw a random Gaussian radius, and convert it to log-normal
							currRadius = (float)fmin((float)exp(mu + sigma*myrand_gaussian_0_1(&p)),maxRadius);//myrand_uniform_0_1(&p);//
							currGrainRadiusSq = currRadius*currRadius;
						}
						else
							currGrainRadiusSq = grainRadiusSq;

						// test distance
						if(sqDistance(xCentreGrain, yCentreGrain, xGaussian, yGaussian) < currGrainRadiusSq)
						{
							pixOut = pixOut+(float)1.0;
							ptCovered = true;
							break;
						}
					}
				} 	//end ncy
			}		//end ncx
			ptCovered = false;
		}		//end monte carlo

		// store output
		pixOut = pixOut/((float)nMonteCarlo);//lambda;//

		// map from blockIdx to pixel position
		x =  (threadIdx.x + blockIdx.x * blockDim.x);
		y =  (threadIdx.y + blockIdx.y * blockDim.y);

		int offset = (int)(x + y * out_w);

		out_im[offset] = pixOut;
    }
}


/**
 * 
 */
 /**
* @brief Film grain rendering using the pixel wise algorithm 
*
* @param lambda parameter of the Poisson process
* @param r average grain radius
* @param stdGrain standard deviation of the grain radii
* @param distributionType 'constant' or 'log-normal' grain radii
* @param xPixel x coordinate of the cell
* @param yPixel y coordinate of the cell
* @return list of grains : [xCentre yCentre radius]
*/

float* film_grain_rendering_pixel_wise_cuda(const float *src_im, int widthIn, int heightIn,
  int widthOut, int heightOut, filmGrainOptionsStruct<float> filmGrainOptions)
{

    //display the parameters
    std::cout<< "image size : " << widthIn << " x " << heightIn << std::endl;
    std::cout<< "------------------" << std::endl;
    std::cout<< "muR : " << filmGrainOptions.muR << std::endl;
    std::cout<< "sigmaR : " << filmGrainOptions.sigmaR << std::endl;
    std::cout<< "zoom, s : " << filmGrainOptions.s << std::endl;
    std::cout<< "sigmaFilter : " <<  filmGrainOptions.sigmaFilter << std::endl;
    std::cout<< "NmonteCarlo : " << filmGrainOptions.NmonteCarlo << std::endl;
    std::cout<< "xA : " << filmGrainOptions.xA << std::endl;
    std::cout<< "yA : " << filmGrainOptions.yA << std::endl;
    std::cout<< "xB : " << filmGrainOptions.xB << std::endl;
    std::cout<< "yB : " << filmGrainOptions.yB << std::endl;
    std::cout<< "randomizeSeed : " << filmGrainOptions.randomizeSeed << std::endl;
    std::cout<< "------------------" << std::endl;

    printf("Output image size is %d x %d\n", widthOut, heightOut);


    /* copy src image on device */
    /* add unnecessary alpha channel to fit the float4 format of texture memory */
    float4* src_imf4;

    src_imf4 = add_interlaced_alpha_channel(src_im, widthIn, heightIn);

    /* copy input float4 texture on device texture memory */
    cudaArray* dev_src_im;
    cudaChannelFormatDesc descchannel;      
    descchannel=cudaCreateChannelDesc<float4>();
    CUDA_CALL( cudaMallocArray(&dev_src_im, &descchannel, widthIn, heightIn) );
    CUDA_CALL( cudaMemcpyToArray(dev_src_im,0,0,src_imf4,
                                 sizeof(float4)*widthIn*heightIn,
                                 cudaMemcpyHostToDevice) );
    tex_src_im.filterMode=cudaFilterModePoint;
    tex_src_im.addressMode[0]=cudaAddressModeClamp;
    tex_src_im.addressMode[1]=cudaAddressModeClamp;
    CUDA_CALL( cudaBindTextureToArray(tex_src_im,dev_src_im) );

	/* pre-calculate the Gaussian , lambda, and exp(-lambda) */
	//pre-calculate lambda and exp(-lambda) for each possible grey-level
	float *lambdaList = new float[ MAX_GREY_LEVEL ];
	float *expLambdaList = new float[ MAX_GREY_LEVEL ];
	for (int i=0; i<=MAX_GREY_LEVEL; i++)
	{
		float u = ((float)i)/( (float) ( (float)MAX_GREY_LEVEL + (float)EPSILON_GREY_LEVEL) );
		float ag = 1/ceil(1/(filmGrainOptions.muR));
		float lambdaTemp = -((ag*ag) /
			( pi*( (filmGrainOptions.muR) * (filmGrainOptions.muR) +
				(filmGrainOptions.sigmaR) * (filmGrainOptions.sigmaR)))) * log(1.0f-u);
		lambdaList[i] = lambdaTemp;
		expLambdaList[i] = exp(-lambdaTemp);
	}
    cudaArray* dev_lambda_list, *dev_exp_lambda_list;
    cudaChannelFormatDesc descchannel1D;      
    descchannel1D=cudaCreateChannelDesc<float>();
    CUDA_CALL( cudaMallocArray(&dev_lambda_list, &descchannel1D, MAX_GREY_LEVEL+1) );
    CUDA_CALL( cudaMallocArray(&dev_exp_lambda_list, &descchannel1D, MAX_GREY_LEVEL+1) );
    CUDA_CALL( cudaMemcpyToArray(dev_lambda_list,0,0,lambdaList,
                                 sizeof(float)*(MAX_GREY_LEVEL+1),
                                 cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpyToArray(dev_exp_lambda_list,0,0,expLambdaList,
                             sizeof(float)*(MAX_GREY_LEVEL+1),
                             cudaMemcpyHostToDevice) );
    tex_lambda_list.filterMode=cudaFilterModePoint;
    tex_lambda_list.addressMode[0]=cudaAddressModeClamp;
    tex_lambda_list.addressMode[1]=cudaAddressModeClamp;
    CUDA_CALL( cudaBindTextureToArray(tex_lambda_list,dev_lambda_list) );
    //exp(-lambda)
    tex_exp_lambda_list.filterMode=cudaFilterModePoint;
    tex_exp_lambda_list.addressMode[0]=cudaAddressModeClamp;
    tex_exp_lambda_list.addressMode[1]=cudaAddressModeClamp;
    CUDA_CALL( cudaBindTextureToArray(tex_exp_lambda_list,dev_exp_lambda_list) );

    /* allocate memory for output image on host and device */
    float *out_im;
    float *dev_out_im;
    out_im = (float *) calloc(widthOut*heightOut, sizeof(float));
    CUDA_CALL( cudaMalloc( (void**)&dev_out_im, widthOut*heightOut*sizeof(float) ) );
                              
    /* Execution of kernel on device */
    int nbthreads = 28;
    dim3 blocks( (widthOut+nbthreads-1)/nbthreads, (heightOut+nbthreads-1)/nbthreads); //blocks(1,1);//;
    dim3 threads(nbthreads,nbthreads); //threads(1,1);//
	printf("blocks.x : %d\n",blocks.x);
	printf("blocks.y : %d\n",blocks.y);
	printf("blocks.z : %d\n",blocks.z);    
	/* start cuda event for computation time */
    cudaEvent_t     start, stop;
    CUDA_CALL( cudaEventCreate( &start ) );
    CUDA_CALL( cudaEventCreate( &stop ) );
    CUDA_CALL( cudaEventRecord( start, 0 ) );

	//unsigned int seed = time(NULL);
	struct timeval time; 
	gettimeofday(&time,NULL);
        unsigned int seed;
        // RS EDIT: add flag to decide whether to randomize grain seed or not
        if (filmGrainOptions.randomizeSeed > 0) {
            seed = (unsigned int) ( (time.tv_sec * 1000) + (time.tv_usec / 1000) );
        } else {
	    seed = 1;
        }	
	//std::cout<< "seed : " << seed << std::endl;
    
	kernel<<<blocks,threads>>>(dev_out_im, widthOut, heightOut, 
                           widthIn, heightIn,
                          filmGrainOptions.muR,filmGrainOptions.sigmaR,
                          filmGrainOptions.NmonteCarlo,
                          filmGrainOptions.s, filmGrainOptions.sigmaFilter,seed,
                          filmGrainOptions.xA,filmGrainOptions.yA,
                          filmGrainOptions.xB, filmGrainOptions.yB);
	
    /* get stop time, and display the timing results */
    CUDA_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_CALL( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Time to generate:  %3.1f ms; Framerate: %3.2f images/sec\n", elapsedTime, 1000./elapsedTime );
	//printf( "elapsed time : %2.3f\n", elapsedTime);
    CUDA_CALL( cudaEventDestroy( start ) );
    CUDA_CALL( cudaEventDestroy( stop ) );
    
    /* copy output image */
    CUDA_CALL( cudaMemcpy( out_im, dev_out_im, widthOut*heightOut*sizeof(float),
                              cudaMemcpyDeviceToHost ) );

	float* imgOut = (float *) calloc(widthOut*heightOut, sizeof(float));
	memcpy(imgOut,out_im,widthOut*heightOut * sizeof(float));

    /* free memory */
    CUDA_CALL( cudaUnbindTexture( tex_src_im ) );
    CUDA_CALL( cudaUnbindTexture( tex_lambda_list ) );
    CUDA_CALL( cudaUnbindTexture( tex_exp_lambda_list ) );
    CUDA_CALL( cudaFree(dev_out_im) );
    CUDA_CALL( cudaFreeArray(dev_src_im) );
    free(src_imf4);
	free(out_im);

	return(imgOut);
}
