#pragma once

#include "cudas.h"
#include "HostMemoryType.h"

class HM
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*--------------*\
	 |*   malloc  *|
	 \*-------------*/

	/**
	 * <pre>
	 * idem cudaHostMalloc  but with hostMemoryType replace by native int of cuda (allow combinaison)
	 *
	 * cudaHostAllocDefault
	 * cudaHostAllocPortable
	 * cudaHostAllocMapped
	 * cudaHostAllocWriteCombined
	 *
	 * http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g15a3871f15f8c38f5b7190946845758c.html#g15a3871f15f8c38f5b7190946845758c
	 * </pre>
	 */
	template<typename T>
	static void malloc(T** ptrHM , size_t sizeOctet , int hostMemoryType);

	/**
	 * use DMA to copy Host to Device
	 */
	template<typename T>
	static void malloc(T** ptrHM , size_t sizeOctet , HostMemoryType hostMemoryType=HostMemoryType::DEFAULT);

	/*--------------*\
	 |*   free  *|
	 \*-------------*/

	template<typename T>
	static void free(T* ptrHM);

	/*--------------*\
	|*   tools  *|
	 \*-------------*/

	static int toNvidiaType(HostMemoryType hostMemoryType);

    };

#include "HM_MemoryManagement.h"
// contain static methode with template
// pas code completion!!
// include car template

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
