#pragma once

#include "DeviceInfo.h"
#include "DeviceDriver.h"
#include "Args.h"

//#ifndef CBI_MACRO_CUDA_CONTEXT_SIMPLE_DEFINE
//#define CBI_MACRO_CUDA_CONTEXT_SIMPLE_DEFINE

//#define MainCoreSimple(name) int (*name)(const Args& args)

//#endif

class CudaContextSimple
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/
    public:

	/**
	 * see public attributs
	 */
	CudaContextSimple();

	virtual ~CudaContextSimple();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	int process();

    protected:

	virtual int launch();

    private:

	void checkCompatibility();

	void driver();

	void deviceQuery();

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    public:

	// Inputs
	int deviceId;

	DeviceDriver deviceDriver;
	DeviceInfo deviceInfo;

	// syntaxe 1
	//MainCoreSimple(mainCoreSimple);

	// syntaxe 2
	int (*mainCore)(const Args& args);

	Args args;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

