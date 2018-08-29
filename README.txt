                      ━━━━━━━━
                       README
                      ━━━━━━━━


Table of Contents
─────────────────

1 About
2 Overview
3 Requirement
4 Compilation
5 Usage
.. 5.1 Film grain rendering
..... 5.1.1 Description


1 About
═══════

  Author : Alasdair Newson <alasdairnewson.work@gmail.com>,
	   Julie Delon <julie.delon@parisdescartes.fr> and
	   Bruno Galerne <bruno.galerne@parisdescartes.fr>
  Copyright: (C) 2016 IPOL Image Processing On Line [http://www.ipol.im/]
  Licence : GPL V3+

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.


2 Overview
══════════

  This source code provides an implementation of the film grain rendering algorithm of Newson et al.


  The 'bin/film_grain_rendering_main' program reads an input image, and adds film grain noise/texture
  to this image. Only PNG or TIFF (black and white or colour) images are handled.


3 Requirement
═════════════

  The code is written in UTF8 C++, and should compile on any system with
  an UTF8 C++ compiler.

  The libpng and libtiff header and libraries are required on the system
  for compilation and execution. On Linux, just use your package manager
  to install it:
  ┌────
  │ sudo apt-get install libpng
  │ sudo apt-get install libtiff
  └────


  For more information, see [http://www.libpng.org/pub/png/libpng.html]
  and [http://www.libtiff.org/].


4 Compilation
═════════════

  To compile the code, use the provided makefile, with the command 'make'. The
  makefile will produce a program called : 'bin/film_grain_rendering_main'.

  It is possible to compile the program using OpenMP with the command
  'make OMP=1'.

  The 'film_grain_rendering_main' program is used to render an input image with film grain.


5 Usage
═══════

5.1 Film grain rendering
──────────

5.1.1 Description
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  The 'film_grain_synthesis_main' program can be called, for example, in the following manner :
  ┌────
  │ bin/film_grain_rendering_main input.tiff output.png -r "0.1" -grainSigma "0.0" -filterSigma "0.8" -zoom "2.0" -algorithmID "0" -NmonteCarlo "100"
  └────
  input.tiff : input image. Can be a tiff or png image.
  output.tiff : output, rendered image. A tiff image.
  Parameters [default value]
  -r : average grain radius [0.1]
  -grainSigma : standard deviation of the grain radii [0.0]
  -filterSigma : standard deviation (in pixels) of the Gaussian filter applied to the continuous grain model [0.8]
  -zoom : zoom coefficient which increases the resolution of the output image with respect to the input image resolution [1.0]
  -algorithmID : identifier of the algorithm used. Can be equal to 0 (pixel-wise algorithm) or 1 (grain-wise algorithm) [0]
  -NmonteCarlo : number of Monte Carlo simulations. This influences the quality of the result (the higher the number, the better the quality,
  but the longer it takes) [800]

  #Further parameters concerning zoom and resolution. These parameters are to be specified if the user wishes to zoom on a specific region of the input image and only carry out the rendering in this region. In this case, you should also specify the number of pixels in the x and y directions (nX, nY).
  #Note, if the 'zoom' parameter is activated, then the following parameters are ignored, as they may potentially be incompatible with the zoom.

  -nX : number of pixels in the x direction (number of columns). By default, this is set to the same number as in the input image.
  -nY : number of pixels in the y direction (number of rows). By default, this is set to the same number as in the input image.
  -xA : x coordinate of the upper left corner of the rectangle of the region in which the film grain rendering is carried out [0.0]
  -yA : y coordinate of the upper left corner of the rectangle of the region in which the film grain rendering is carried out [0.0]
  -xB : x coordinate of the bottom right corner of the rectangle of the region in which the film grain rendering is carried out [nX]
  -yB : y coordinate of the bottom right corner of the rectangle of the region in which the film grain rendering is carried out [nY]

6 Bugs Report
═════════════
You can report any bug with the github interface :

