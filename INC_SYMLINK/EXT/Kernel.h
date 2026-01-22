#pragma once

//#include <cuda_runtime.h>
#include "cudas.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Kernel
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*--------------*\
	|* Wrapper      *|
	 \*-------------*/

	/**
	 * wrapper for cudaDeviceSynchronize().
	 * idem Stream::synchronize()
	 */
	static void synchronize();

	/*--------------*\
		|* Tools      *|
	 \*-------------*/

	/**
	 * message = nameKernel by example
	 */
	static void lastCudaError(const char* message = NULL);

	static void lastCudaError(std::string message="");

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

