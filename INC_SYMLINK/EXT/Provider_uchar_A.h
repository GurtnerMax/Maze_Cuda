#pragma once

#include "cudas.h"

#include "ProviderUse_I.h"

#include "Provider_I.h"


/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Provider_uchar_A: public Provider_I<uchar>, ProviderUse_I
    {
	/*----------------------------------------------------------------------*\
	|*			Constructeur 					*|
	 \*---------------------------------------------------------------------*/

    public:

	Provider_uchar_A(bool loadOnlyOneImage = false);

	virtual ~Provider_uchar_A();

	/*----------------------------------------------------------------------*\
	|*			Methode 					*|
	 \*---------------------------------------------------------------------*/

    public:

	/*---------------------*\
	|*	Virtual 	*|
	 \*--------------------*/

	virtual Grid grid()=0;

	virtual Animable_I<uchar>* createAnimable(const Grid& grid , bool isVerbose = false)=0;

	/*---------------------*\
	|*	Override 	*|
	 \*--------------------*/

	/**
	 * Override
	 */
	virtual Animable_I<uchar>* createAnimable(bool isVerbose = true);

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

	/*----------------------------------------------------------------------*\
	 |*			attribut 					*|
	 \*---------------------------------------------------------------------*/

    protected:

	// Inputs
	bool loadOnlyOneImage;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

