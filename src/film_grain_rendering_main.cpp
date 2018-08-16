
#include <string>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <algorithm> 
#include "matrix.h"
#include "io_png.h"
#include "libtiff_io.h"
#include "film_grain_rendering.h"


static void show_help();

/// help on usage of inpainting code
static void show_help() {
    std::cerr <<"\nFilm grain synthesis.\n"
              << "Usage: " << " synthesise_film_grain imgIn.tiff imgNameOut.tiff imgPathOut  [options]\n\n"
              << "Options (default values in parentheses)\n"
              << "-r : Grain size (0.05)\n"
			  << "-sigmaR : Grain standard deviation (0.0)\n"
  			  << "-NmonteCarlo : Number of Monte Carlo simulations\n"
			  << "-zoom : zoom of output image\n"
			  << "-sigmaFilter : Standard deviation of Gaussian filter (0.8)\n"
			  << "-color : whether color grain is activated (0 : black-and-white, 1 : color, default : 1)"
			  << "-xA \n"
			  << "-yA \n"
			  << "-xB \n"
			  << "-yB \n"
			  << "-mOut : output resolution, number of rows\n"
			  << "-nOut : output resolution, number of columns\n"
              << std::endl;
}

/**
 * 
 */
 /**
* @brief Generate local Boolean model information 
*
* @param lambda parameter of the Poisson process
* @param r average grain radius
* @param stdGrain standard deviation of the grain radii
* @param distributionType 'constant' or 'log-normal' grain radii
* @param xPixel x coordinate of the cell
* @param yPixel y coordinate of the cell
* @return list of grains : [xCentre yCentre radius]
*/
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}


/**
 * 
 */
 /**
* @brief Check for input parameter
*
* @param beginning of input command chain
* @param end of input command chain
* @return whether the parameter exists or not
*/
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

/**
 * 
 */
 /**
* @brief Get file exension of file name
*
* @param File name
* @return File extension
*/
std::string getFileExt(const std::string& s)
{
	size_t i = s.rfind('.', s.length());
	if (i != std::string::npos)
	{
		return(s.substr(i+1, s.length() - i));
	}
	else
		return("");
}


/**
 * 
 */
 /**
* @brief Get file name, without extension
*
* @param Input file name
* @return File name without extension
*/
std::string getFileName(const std::string& s)
{
	size_t i = s.rfind('.', s.length());
	if (i != std::string::npos)
	{
		return(s.substr(0, i));
	}
	else
		return(s);
}

/**
 * 
 */
 /**
* @brief Get current directory
*
* @return Current directory name
*/
std::string get_curr_dir() {
	size_t maxBufSize = 1024;
	char buf[maxBufSize];
	char* charTemp = getcwd(buf,maxBufSize);
	std::string currDir(charTemp);
	return(currDir);
}

/**
 * 
 */
 /**
* @brief Write the output to a .tiff or .png image.
*
* @param imgOut output image to write
* @param fileNameOut output file name
* @param filmGrainParams parameters of the film grain rendering algorithm
* @param nChannels number of colour channels in the output image
* @return 0 if write success, -1 if failure
*/
int write_output_image(float *imgOut, const std::string fileNameOut,
	filmGrainOptionsStruct<float> filmGrainParams, unsigned int nChannels)
{
	/*buffers for output file name*/
	//mean radius
	char bufferR [50];
	sprintf (bufferR, "%0.3f", filmGrainParams.muR);
	std::string strR(bufferR);
	
	//standard deviation of the radius
	char bufferStd [50];
	sprintf (bufferStd, "%0.4f", filmGrainParams.sigmaR);
	std::string strStd(bufferStd);

	//standard deviation of the blur kernel
	char bufferSigmaConv [50];
	sprintf (bufferSigmaConv, "%0.4f", filmGrainParams.sigmaFilter);
	std::string strSigmaConv(bufferSigmaConv);

	//output zoom factor
	char bufferZoom [50];
	sprintf (bufferZoom, "%0.4f", filmGrainParams.s);
	std::string sZoomStr(bufferZoom);

	//number of Monte Carlo iterations
	char bufferNmonteCarlo [50];
	sprintf (bufferNmonteCarlo, "%05d", filmGrainParams.NmonteCarlo);
	std::string sNmonteCarloStr(bufferNmonteCarlo);

	std::string outputExtension(getFileExt(fileNameOut));

	//write output image
	std::string fileNameOutFull = (char*)( (getFileName(fileNameOut) + "." + outputExtension) .c_str());
	std::cout << "output file name : " << fileNameOutFull << std::endl;

	if (strcmp((const char*)(outputExtension.c_str()),"tiff")==0 )	//tiff files
	{
		if (write_tiff_image(imgOut, (unsigned int)filmGrainParams.nOut,
			(unsigned int) filmGrainParams.mOut, nChannels, (const char*)fileNameOutFull.c_str()) ==-1)
		{
			std::cout<< "Error, could not write the image file." << std::endl;
			return(-1);
		}
	}
	else if (strcmp((const char*)(outputExtension.c_str()),"png")==0 ||
		strcmp((const char*)(outputExtension.c_str()),"")==0)	//png files
	{
		if(0 != io_png_write_f32(fileNameOutFull.c_str(), imgOut,
			filmGrainParams.nOut, filmGrainParams.mOut, nChannels))
		{
			std::cout<< "Error, could not write the image file." << std::endl;
			return(-1);
		}
	}
	else
	{
		std::cout<< "Error, unknown output file extension." << std::endl;
		return(-1);
	}

	return(0);
}


/**
* @brief main function call
*/
int main(int argc, char* argv[])
{

	if(argc < 3) {
        show_help();
        return -1;
    }
	
	//get file names
	std::string fileNameIn(argv[1]);
	std::string fileNameOut(argv[2]);

	float grainSize, sigmaR, s, sigmaFilter, xA, yA, xB, yB;
	int mOut, nOut;
	unsigned int NmonteCarlo;
        int randSeed; // RS
	int colourActivated;

	/**************************************************/
	/*************   READ INPUT IMAGE   ***************/
	/**************************************************/
	float *imgInFloat;
	size_t widthIn, heightIn, nChannels;

	//check the extension of the input file
	if (strcmp((const char*)(getFileExt(fileNameIn).c_str()),"tiff")==0)
	{
		uint32 widthTemp, heightTemp, nChannelsTemp;
		imgInFloat = read_tiff_image((const char*)((fileNameIn).c_str()),
			&widthTemp, &heightTemp, &nChannelsTemp);
		widthIn = (size_t)widthTemp;
		heightIn = (size_t)heightTemp;
		nChannels = (size_t)nChannelsTemp;
		if (imgInFloat == NULL)
		{
			std::cout<< "Error, could not read the image file." << std::endl;
			return(-1);
		}
	}
	else if (strcmp((const char*)(getFileExt(fileNameIn).c_str()),"png")==0)
	{
		imgInFloat = io_png_read_f32((const char*)(fileNameIn.c_str()),
			&widthIn, &heightIn, &nChannels);
	}
	else
	{
		std::cout << "Unable to read the input image." << std::endl;
		return(-1);
	}

	//show help
	if(cmdOptionExists(argv, argv+argc, "-h"))
	{
		show_help();
		return(-1);
	}

	/**************************************************/
	/*************   GET INPUT OPTIONS   **************/
	/**************************************************/

        // RS rand seed
        if(cmdOptionExists(argv, argv+argc, "-seed"))
        {
            randSeed = (int)atof(getCmdOption(argv, argv+argc, "-seed"));
        }
        else
            randSeed = 1;

	//grain size
	if(cmdOptionExists(argv, argv+argc, "-r"))
	{
			grainSize = (float)atof(getCmdOption(argv, argv + argc, "-r"));
	}
	else
		grainSize = 0.1;
	if(cmdOptionExists(argv, argv+argc, "-sigmaR"))
	{
		sigmaR = (float)atof(getCmdOption(argv, argv + argc, "-sigmaR"));
	}
	else
		sigmaR = 0.0;
	
	//zoom
	if(cmdOptionExists(argv, argv+argc, "-zoom"))
		s =  (float)atof(getCmdOption(argv, argv + argc, "-zoom"));
	else
		s = 1.0;

	//filter standard deviation
	if(cmdOptionExists(argv, argv+argc, "-sigmaFilter"))
		sigmaFilter = (float)atof(getCmdOption(argv, argv + argc, "-sigmaFilter"));
	else
		sigmaFilter = 0.8;

	//number of Monte Carlo iterations
	if(cmdOptionExists(argv, argv+argc, "-NmonteCarlo"))
		NmonteCarlo = (unsigned int)atoi(getCmdOption(argv, argv + argc, "-NmonteCarlo"));
	else
		NmonteCarlo = 800;

	//colour image
	if(cmdOptionExists(argv, argv+argc, "-color"))
		colourActivated = (int)atoi(getCmdOption(argv, argv + argc, "-color"));
	else
		colourActivated = 1;
	if(nChannels == 1)
		colourActivated = 0;

	//image limits and output resolution
	//NOTE : these are not taken into account if the "zoom" parameter is set
	//by default, these are set to the input image limits
	//xA
	if(cmdOptionExists(argv, argv+argc, "-xA"))
		xA = atof(getCmdOption(argv, argv + argc, "-xA"));
	else
		xA = 0;
	//yA
	if(cmdOptionExists(argv, argv+argc, "-yA"))
		yA = atof(getCmdOption(argv, argv + argc, "-yA"));
	else
		yA = 0;
	//xB
	if(cmdOptionExists(argv, argv+argc, "-xB"))
		xB = atof(getCmdOption(argv, argv + argc, "-xB"));
	else
		xB = widthIn;
	//yB
	if(cmdOptionExists(argv, argv+argc, "-yB"))
		yB = atof(getCmdOption(argv, argv + argc, "-yB"));
	else
		yB = heightIn;

	//mOut
	if(!cmdOptionExists(argv, argv+argc, "-zoom") && cmdOptionExists(argv, argv+argc, "-mOut"))
		mOut = atoi(getCmdOption(argv, argv + argc, "-mOut"));
	else if (cmdOptionExists(argv, argv+argc, "-zoom"))
		mOut = (int)floor(s * (int)heightIn);
	else
		mOut = (int)heightIn;
	//nOut
	if(!cmdOptionExists(argv, argv+argc, "-zoom") && cmdOptionExists(argv, argv+argc, "-nOut"))
		nOut = atoi(getCmdOption(argv, argv + argc, "-nOut"));
	else if (cmdOptionExists(argv, argv+argc, "-zoom"))
		nOut = (int)floor(s * (int)widthIn);
	else
		nOut = (int)widthIn;
	//check the xA, yA, xB, yB parameters
	if ( xA <0 || yA<0 || xB>(widthIn) || yB>(heightIn))
	{
		std::cout << "Error, the (xA,yA), (xB,yB) parameters are out of bounds." << std::endl;
		std::cout << "xA : " << xA << ", yA : " << yA <<std::endl;
		std::cout << "xB : " << xB << ", yB : " << yB <<std::endl;
		return(-1);
	}
		
	//create film grain options structure
	filmGrainOptionsStruct<float> filmGrainParams;

	filmGrainParams.muR = grainSize;
	filmGrainParams.sigmaR = sigmaR;
	filmGrainParams.s = ((float)mOut)/((float)(heightIn)); 	//zoom
	filmGrainParams.sigmaFilter = (float)sigmaFilter;
	filmGrainParams.NmonteCarlo = NmonteCarlo; 	//number of monte carlo iterations
	filmGrainParams.xA = xA;
	filmGrainParams.yA = yA;
	filmGrainParams.xB = xB;
	filmGrainParams.yB = yB;
	filmGrainParams.mOut = mOut;
	filmGrainParams.nOut = nOut;
	
	//display parameters
	std::cout<< "Input image size : " << widthIn << " x " << heightIn << std::endl;
	std::cout<< "grainRadius : " << filmGrainParams.muR << std::endl;
	std::cout<< "sigmaR : " << filmGrainParams.sigmaR << std::endl;
	std::cout<< "sigmaFilter : " <<  filmGrainParams.sigmaFilter << std::endl;
	std::cout<< "NmonteCarlo : " << filmGrainParams.NmonteCarlo << std::endl;
	if (colourActivated == 0)
		std::cout<< "black and white" << std::endl;
	else
		std::cout<< "colour" << std::endl;
	if(cmdOptionExists(argv, argv+argc, "-zoom"))
		std::cout<< "zoom : " <<  filmGrainParams.s << std::endl;
	std::cout<< "xA : " << filmGrainParams.xA << std::endl;
	std::cout<< "yA : " << filmGrainParams.yA << std::endl;
	std::cout<< "xB : " << filmGrainParams.xB << std::endl;
	std::cout<< "yB : " << filmGrainParams.yB << std::endl;
	std::cout<< "mOut : " << filmGrainParams.mOut << std::endl;
	std::cout<< "nOut : " << filmGrainParams.nOut << std::endl;

	/**************************************************/
	/*****  TIME AND CARRY OUT GRAIN RENDERING   ******/
	/**************************************************/

	struct timeval start, end;
	gettimeofday(&start, NULL);

	//create output float image
	float *imgOut = new float[(mOut) * (nOut) * ( (unsigned int)MAX_CHANNELS)];

	matrix<float> *imgIn = new matrix<float>();
	imgIn->allocate_memory((int)heightIn, (int)widthIn);
	//create pseudo-random number generator for the colour seeding

	//execute the film grain synthesis
	std::cout << "***************************" << std::endl;
	for (unsigned int colourChannel=0; colourChannel< ( (unsigned int)MAX_CHANNELS); colourChannel++)
	{
		float *imgOutTemp;
		//copy memory
		for (unsigned int i=0; i<(unsigned int)heightIn; i++)
		{
			for (unsigned int j=0; j<(unsigned int)widthIn; j++)
			{
				imgIn->set_value(i,j,(float)imgInFloat[ (int) i*((unsigned int)widthIn) + (int)j +
				colourChannel*((unsigned int)widthIn)*((unsigned int)heightIn)]);
			}
		}
		//normalise input image
                imgIn->divide((float)(MAX_GREY_LEVEL+EPSILON_GREY_LEVEL)); // RS

		filmGrainParams.grainSeed = (unsigned int)randSeed; // RS //(unsigned int)myrand(&pSeedColour);
		
		/***************************************/
		/**   carry out film grain synthesis  **/
		/***************************************/
		imgOutTemp = film_grain_rendering_pixel_wise_cuda(imgIn->get_ptr(),
			imgIn->get_ncols(), imgIn->get_nrows(), nOut, mOut, filmGrainParams);
		
		//put the output image back to [0, 255]
                for (int i=0; i<(mOut*nOut); i++)
			imgOutTemp[i] = imgOutTemp[i] * ((float)(MAX_GREY_LEVEL+EPSILON_GREY_LEVEL));

		if (colourActivated>0) 	//colour grain
		{
			for ( unsigned int i=0; i<(unsigned int)mOut ; i++)
				for ( unsigned int j=0; j< ((unsigned int)nOut); j++)
					imgOut[ j + i*(nOut) +
						colourChannel*(mOut) * (nOut)] =
						imgOutTemp[  j + i*(nOut) ];
			delete imgOutTemp;
		}
		else 	//black-and-white
		{
			for ( unsigned int i=0; i<(unsigned int)mOut ; i++)
				for ( unsigned int j=0; j< (unsigned int)nOut; j++)
				{
					imgOut[ j + i*(nOut) ] = 
						imgOutTemp[  j + i*(nOut) ];		//red
					imgOut[ j + i*(nOut) + (mOut) * (nOut)] =
							imgOutTemp[  j + i*(nOut) ]; 	//green
					imgOut[ j + i*(nOut) + 2*(mOut) * (nOut)] = 
						imgOutTemp[  j + i*(nOut) ];	//blue
				}
			delete imgOutTemp;std::cout << "Here" <<std::endl;
			break;
		}
	}
	write_output_image(imgOut, fileNameOut, filmGrainParams,(unsigned int)MAX_CHANNELS);

	delete imgInFloat;
	delete imgIn;
	delete imgOut;
	
	gettimeofday(&end, NULL);
	double elapsedTime = (end.tv_sec  - start.tv_sec) + 
			 (end.tv_usec - start.tv_usec) / 1.e6;		
	std::cout << "time elapsed : " << elapsedTime << std::endl;	
	std::cout << "***************************" << std::endl << std::endl << std::endl;

	return(0);

}
