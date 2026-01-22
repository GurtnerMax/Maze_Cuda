#pragma once

#include <iostream>


#include "OpencvTools.h"

//#include "cudas.h"
#include "builtin_types.h"//cuda

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Image
    {
	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	Image(std::string filename);

	virtual ~Image();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	cv::Mat mat_RGBA();
	cv::Mat mat_BGR();

	uchar4* uchar4_RGBA();
	uchar3* uchar3_BGR();

	int w();
	int h();

	std::string getFileName();

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Input
	std::string filename;

	// Outputs
	cv::Mat matBGR;
	cv::Mat matRGBA;

	uchar4* ptrImageRGBA;
	uchar3* ptrImageBGR;

	int width;
	int height;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
