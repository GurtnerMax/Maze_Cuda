#pragma once

#include "CudaContext.h"

#include "LaunchModeImage.h"
#include "Args.h"

#define MainImage(name) int (*name)(const Args& args)

class CudaContextImage: public CudaContext
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/
    public:

	/**
	 * set public attributs
	 */
	CudaContextImage();

	virtual ~CudaContextImage();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    protected:

	/**
	 * redefinition
	 */
	virtual int launch();

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    public:

	// Inputs
	LaunchModeImage launchMode;
	Args args;

	// syntaxe 1
	MainImage(mainImage);

	// syntaxe 2
//	int (*mainImage)(const Args& args);
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

