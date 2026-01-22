#pragma once

#include <iostream>

#include "ColorRGB_01.h"

#include "Image_GPU.h"
#include "Animable_I.h"

using std::cout;
using std::endl;
using std::string;
using std::to_string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

template<typename T, ColorModel ColorModel, int PixelFormat>
class ImageFromAnimable: public Image_A<T, ColorModel, PixelFormat, DomaineMath>
    {
	/*--------------------------------------*\
	|*		Constructeur		*|
	 \*-------------------------------------*/

    public:

	ImageFromAnimable(Animable_I<T>* ptrAnimable , ColorRGB_01 colorTitreRGB = ColorRGB_01(1.0f, 0.0f, 0.0f)) :
		Image_A<T, ColorModel, PixelFormat, DomaineMath>(ptrAnimable->getW(), //
			ptrAnimable->getH(), //
			ptrAnimable->getDomaineMathInit()), ptrAnimable(ptrAnimable), //
		colorTitreRGB(colorTitreRGB)
	    {
	    // rien
	    }

	virtual ~ImageFromAnimable(void)
	    {
	    delete ptrAnimable; // TODO discutable, not that problem with shared_ptr
	    }

	/*--------------------------------------*\
	    |*		Methode			*|
	 \*-------------------------------------*/

    public:

	Animable_I<T>* getAnimable()
	    {
	    return ptrAnimable;
	    }

	/*----------------*\
	    |*  Override	*|
	 \*---------------*/

	/**
	 * Override, call periodicly by the api
	 */
	virtual void fillImage(T* ptrPixel , int w , int h , const DomaineMath& domaineMath)
	    {
	    ptrAnimable->process(ptrPixel, w, h, domaineMath);
	    }

	/**
	 * Override, call periodicly by the api
	 */
	virtual void animationStep(bool& isNeedUpdateView)
	    {
	    ptrAnimable->animationStep();
	    isNeedUpdateView = true;
	    }

	/**
	 * Override, call periodicly by the api
	 */
	virtual void paintPrimitives(Graphic2D& graphic2D)
	    {
	    graphic2D.setFont(TIMES_ROMAN_24);

	    float r = colorTitreRGB.r;
	    float g = colorTitreRGB.g;
	    float b = colorTitreRGB.b;

	    graphic2D.setColorRGB(r, g, b);

	    // Top : Para Animation
		{
		float t = ptrAnimable->getAnimationPara();

		string message = "t = " + to_string(t);

		graphic2D.drawTitleTop(message);
		}

	    // Bottom : Title
		{
		string title = ptrAnimable->getTitle();

		graphic2D.drawTitleBottom(title);
		}
	    }

	/*--------------------------------------*\
	|*		Attribut		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	Animable_I<T>* ptrAnimable;
	ColorRGB_01 colorTitreRGB;

    };

typedef ImageFromAnimable<uchar4, RGBA, GL_UNSIGNED_BYTE> ImageAnimable_RGBA_uchar4;
typedef ImageFromAnimable<uchar3, RGB, GL_UNSIGNED_BYTE> ImageAnimable_RGB_uchar3;
typedef ImageFromAnimable<uchar, GRAY, GL_UNSIGNED_BYTE> ImageAnimable_GRAY_uchar;

typedef ImageFromAnimable<uchar4, HSBA, GL_UNSIGNED_BYTE> ImageAnimable_HSBA_uchar4;
typedef ImageFromAnimable<uchar3, HSB, GL_UNSIGNED_BYTE> ImageAnimable_HSB_uchar3;
typedef ImageFromAnimable<unsigned char, HUE, GL_UNSIGNED_BYTE> ImageAnimable_HUE_uchar;

typedef ImageFromAnimable<float4, RGBA, GL_FLOAT> ImageAnimable_RGBA_float4;
typedef ImageFromAnimable<float3, RGB, GL_FLOAT> ImageAnimable_RGB_float3;
typedef ImageFromAnimable<float, GRAY, GL_FLOAT> ImageAnimable_GRAY_float;

typedef ImageFromAnimable<float4, HSBA, GL_FLOAT> ImageAnimable_HSBA_float4;
typedef ImageFromAnimable<float3, HSB, GL_FLOAT> ImageAnimable_HSB_float3;
typedef ImageFromAnimable<float, HUE, GL_FLOAT> ImageAnimable_HUE_float;

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
