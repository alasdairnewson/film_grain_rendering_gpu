
#include "libtiff_io.h"

float * read_tiff_image(const char* inputFile, uint32 *widthOut, uint32 *heightOut, uint32 *nChannels)
{

	TIFF *tifFile=TIFFOpen(inputFile, "r");
	//get height and width of image
	TIFFGetField(tifFile, TIFFTAG_IMAGEWIDTH, widthOut);           // uint32 width;
	TIFFGetField(tifFile, TIFFTAG_IMAGELENGTH, heightOut);        // uint32 height;
	TIFFGetField(tifFile, TIFFTAG_IMAGEDEPTH, nChannels);
	*nChannels = (int)fmax((float)*nChannels,(float)1.0);
	*nChannels = (int)fmin((float)*nChannels,(float)MAX_CHANNELS);

	std::cout << "Image size : " << *widthOut << " x " <<
			*heightOut << " x " << *nChannels << std::endl;
	uint32 width = *widthOut;
	uint32 height = *heightOut;

	//reserve temporary space for the image
	uint32 npixels=(uint32)(width*height);
	uint32* raster=(uint32*) _TIFFmalloc(npixels *sizeof(uint32));

	float *outputImg = new float[npixels*(*nChannels)];
	//read the image
	if (raster != NULL)
	{
		if (TIFFReadRGBAImage(tifFile, width, height, raster, 0) != 0)
		{
			//copy image information into the matrix
			for (uint32 i=0; i<height; i++)
				for (uint32 j=0; j<width; j++)
					for (uint32 c=0; c<(*nChannels); c++)
					{
						int iRaster = height-i-1;	//note, the libtiff stores the image from the bottom left as the origin
						switch(c)
						{
							case 0:
								outputImg[j + i*width] = (float)TIFFGetR(raster[ iRaster*width + j]);
								break;

							case 1:
								outputImg[j + i*width + width*height] =
								(float)TIFFGetG(raster[ iRaster*width + j]);
								break;
						
							case 2:
								outputImg[j + i*width + 2*width*height] =
								(float)TIFFGetB(raster[ iRaster*width + j]);
								break;
							default:
  								std::cout << "Error in reading the tiff file, too many channels." << std::endl;
  								break;
						}
					}
		}
		else
		{
			std::cout << "Error reading the image file with TIFFReadRGBAImage" << std::endl;
		}
    }
	else
	{
		std::cout << "Error, could not read the input image." << std::endl;
		std::cout << "File name : " << inputFile << std::endl;
		TIFFClose(tifFile);
		return(NULL);
	}
    _TIFFfree(raster);

	//close image
	TIFFClose(tifFile);
	return(outputImg);
}


int write_tiff_image(float* inputImg, unsigned int n, unsigned int m,
	unsigned int nChannels, const char* outputFile)
{
	uint32 width,height;
	width = (uint32)n;
	height = (uint32)m;

	//parameters
	uint32 samplePerPixel = nChannels;

	TIFF *tifFile= TIFFOpen(outputFile, "w");
	//set parameters of image
	TIFFSetField (tifFile, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
	TIFFSetField(tifFile, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
	TIFFSetField(tifFile, TIFFTAG_SAMPLESPERPIXEL, samplePerPixel);   // set number of channels per pixel
	TIFFSetField(tifFile, TIFFTAG_BITSPERSAMPLE, 8);
	TIFFSetField(tifFile, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
	//   Some other essential fields to set that you do not have to understand for now.
	TIFFSetField(tifFile, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tifFile, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

	//create a temporary buffer to write the image info to
	tsize_t lineBytes = (tsize_t) (samplePerPixel * width);  // length in memory of one row of pixel in the image.
	unsigned char *buf = NULL;        // buffer used to store the row of pixel information for writing to file
	unsigned char *bufChar;
	//    Allocating memory to store the pixels of current row
	if (TIFFScanlineSize(tifFile) == lineBytes)
	{
		buf = (unsigned char *)_TIFFmalloc(lineBytes);
		bufChar = new unsigned char[lineBytes];
	}
	else
	{
		buf = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(tifFile));
		bufChar = new unsigned char[TIFFScanlineSize(tifFile)];
	}

	//std::cout << "line size : " << TIFFScanlineSize(tifFile)<< std::endl;
	//std::cout << "line bytes : " << lineBytes << std::endl;
	//write the image
	 // We set the strip size of the file to be size of one row of pixels
	TIFFSetField(tifFile, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tifFile, width*samplePerPixel));
	//std::cout << "width*samplePerPixel : " << width*samplePerPixel << std::endl;

	//Now writing image to the file one strip at a time
	for (uint32 i = 0; i < height; i++)
	{
		//copy the image information into a temporary buffer
		for (uint32 j=0; j<width; j++)
		{	//the tiff stores the image info in the following order : RGB
			for (uint32 c=0; c<nChannels; c++)
				bufChar[j*nChannels+c] = (unsigned char)(round( inputImg[j + i*width + c*width*height]) );
		}
		std::memcpy(buf, bufChar, lineBytes);	//copy information to special Tiff buffer
		//std::cout << " " << (*inputImg->get_ptr(i,0)) << std::endl;
		if (TIFFWriteScanline(tifFile, buf, i, 0) < 0)
		{
			std::cout << "Error in writing the image file." << std::endl;
			return(-1);
		}
	}

    if (buf)
    	_TIFFfree(buf);
	delete bufChar;

	//close image
	TIFFClose(tifFile);
	return(0);
}

