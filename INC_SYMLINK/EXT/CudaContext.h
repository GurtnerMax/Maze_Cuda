#pragma once

#include "LaunchModeMOO.h"
#include "CudaContextSimple.h"

#ifndef CBI_MACRO_CUDA_CONTEXT_DEFINE
#define CBI_MACRO_CUDA_CONTEXT_DEFINE

#define MainUse(name) int (*name)()
#define MainTest(name) int (*name)()
#define MainBrutforce(name) int (*name)()
#define MainBenchmark(name) int (*name)()

#endif

class CudaContext: public CudaContextSimple
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/
    public:

	/**
	 * see public attributs
	 */
	CudaContext();

	virtual ~CudaContext();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    protected:

	/**
	 * Override
	 */
	virtual int launch();

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    public:

	// Inputs
	LaunchModeMOO launchMode;

	// syntaxe 1
	MainUse(mainUse);
	MainTest(mainTest);
	MainBenchmark (mainBenchmark);
	MainBrutforce(mainBrutforce);

	// syntaxe 2
//	int (*mainUse)();
//	int (*mainTest)();
//	int (*mainBenchmark)();
//	int (*mainBrutforce)();

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

