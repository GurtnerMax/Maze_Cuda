#pragma once

//#include <cuda_runtime.h>
#include "cudas.h"
#include "CMLink.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class CM
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*--------------*\
	|* Hardware      *|
	 \*-------------*/

	/**
	 * ko
	 */
	static int getMaxKO(int idDevice);

	/**
	 * ko
	 */
	static int getMaxKO();

	/*--------------*\
	|* Tools      *|
	 \*-------------*/

	/**
	 * octets
	 */
	static void assertSize(size_t sizeTabCM);

	/*--------------*\
	|* wrapper      *|
	 \*-------------*/

	template<typename T>
	static void memcpyToCM(T* ptrCM , T* ptrSrc , size_t sizeOctet);

	//template<typename T>
	//static void memcpyAsyncToCM(T* ptrCM , T* ptrSrc , size_t sizeOctet , cudaStream_t stream);

    };

#include "CM_MemoryManagement.h"
// contain static methode with template
// pas code completion!!
// include car template

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

