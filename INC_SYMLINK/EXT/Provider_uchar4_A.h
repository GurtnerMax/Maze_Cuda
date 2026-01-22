#pragma once

#include "cudas.h"

#include "ProviderUse_I.h"
#include "Provider_I.h"


/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Provider_uchar4_A: public Provider_I<uchar4>, ProviderUse_I
    {

	/*----------------------------------------------------------------------*\
	|*			Constructeur 					*|
	 \*---------------------------------------------------------------------*/

    public:

	Provider_uchar4_A();

	virtual ~Provider_uchar4_A();

	/*----------------------------------------------------------------------*\
	|*			Methode 					*|
	 \*---------------------------------------------------------------------*/

    public:

	/*---------------------*\
	|*	Virtual 	*|
	 \*--------------------*/

	virtual Grid grid()=0;

	virtual Animable_I<uchar4>* createAnimable(const Grid& grid , bool isVerbose = false)=0;

	/*---------------------*\
	|*	Override 	*|
	 \*--------------------*/

	/**
	 * Override
	 */
	virtual Animable_I<uchar4>* createAnimable(bool isVerbose = true);

	/**
	 * Override
	 */
	virtual Image_I* createImageGL();

	/**
	 * Override
	 */
	virtual Use_I* createUse(const Grid& grid , bool isVerbose = false);

	/**
	 * Override
	 */
	virtual Use_I* createUse(bool isVerbose = true);

	/*---------------------*\
	|*	a deriver 	*|
	 \*--------------------*/

	virtual ColorRGB_01 colorTitle();

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

