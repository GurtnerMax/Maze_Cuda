#pragma once

//#include <cuda_runtime.h>
#include "cudas.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Stream
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*--------------*\
	|* wrapper      *|
	 \*-------------*/

	/**
	 * wrapper for cudaDeviceSynchronize().
	 * idem Kernel::synchronize()
	 */
	static void synchronize(void);

	/*--------------*\
	|* stream      *|
	 \*-------------*/

	static void synchronize(cudaStream_t stream);

	static void create(cudaStream_t* ptrStream);

	static void destroy(cudaStream_t stream);

	/*--------------*\
	|* event      *|
	 \*-------------*/

	static void eventCreate(cudaEvent_t* ptrEvent);

	static void eventDestroy(cudaEvent_t event);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

