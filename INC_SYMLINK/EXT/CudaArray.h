#pragma once

#include "cudas.h"

#include "CudaArrayType.h"

/*----------------------------------------------------------------------*\
 |*			Classe	 					*|
 \*---------------------------------------------------------------------*/

class CudaArray
    {

    public:

	/*--------------------------------------*\
	|*		malloc	generic		*|
	 \*-------------------------------------*/

	/**
	 * CudaArrayType (facultatif):
	 *
	 * 	- DEFAULT         	: cudaArrayDefault 		: Default
	 * 	- SURFACE_LOAD_STORE 	: cudaArraySurfaceLoadStore     : Allocates an array that can be read from or written to using a surface reference
	 * 	- TEXTURE_GATHER    	: cudaArrayTextureGather        : This flag indicates that texture gather operations will be performed on the array (gather= rassembler)
	 */
	static void malloc(cudaArray** ptrptrCudaArray , uint w , uint h , cudaChannelFormatDesc channelDescription , CudaArrayType type =
		CudaArrayType::DEFAULT_CUARRAY);

	/**
	 * exemple : CudaArray::malloc<uchar>(&ptrCudaArray, w, h);
	 *
	 * CudaArrayType (facultatif):
	 *
	 * 	- DEFAULT         	: cudaArrayDefault 		: Default
	 * 	- SURFACE_LOAD_STORE 	: cudaArraySurfaceLoadStore     : Allocates an array that can be read from or written to using a surface reference
	 * 	- TEXTURE_GATHER    	: cudaArrayTextureGather        : This flag indicates that texture gather operations will be performed on the array (gather= rassembler)
	 */
	template<typename T>
	static void malloc(cudaArray** ptrptrCudaArray , uint w , uint h , CudaArrayType type = CudaArrayType::DEFAULT_CUARRAY)
	    {
	    cudaChannelFormatDesc channelDescription = cudaCreateChannelDesc<T>(); //T meme type que cudaArray

	    CudaArray::malloc(ptrptrCudaArray, w, h, channelDescription, type);
	    }

	/*--------------------------------------*\
	|*	malloc	type predefini		*|
	 \*-------------------------------------*/

	static void malloc_double(cudaArray** ptrptrCudaArray , uint w , uint h , CudaArrayType type = CudaArrayType::DEFAULT_CUARRAY);
	static void malloc_float(cudaArray** ptrptrCudaArray , uint w , uint h , CudaArrayType type = CudaArrayType::DEFAULT_CUARRAY);

	static void malloc_long(cudaArray** ptrptrCudaArray , uint w , uint h , CudaArrayType type = CudaArrayType::DEFAULT_CUARRAY);
	static void malloc_int(cudaArray** ptrptrCudaArray , uint w , uint h , CudaArrayType type = CudaArrayType::DEFAULT_CUARRAY);

	static void malloc_uchar(cudaArray** ptrptrCudaArray , uint w , uint h , CudaArrayType type = CudaArrayType::DEFAULT_CUARRAY);

	/*--------------------------------------*\
	 |*		free			*|
	 \*-------------------------------------*/

	static void free(cudaArray& cudaArray);

	static void free(cudaArray* ptrCudaArray);

	/*--------------------------------------*\
	|*		copy			*|
	 \*-------------------------------------*/

	template<typename T>
	static void memcpyHtoD(cudaArray* ptrCudaArray , T* tabHost , uint w , uint h)
	    {
	    size_t widthOctet = w * sizeof(T);
	    size_t heightOctet = h * sizeof(T);
	    size_t pitch = widthOctet; // pitch = width + padding (pour passer a un elemnet la ligne du dessous)

	    HANDLE_ERROR(cudaMemcpy2DToArray(ptrCudaArray, 0, 0, tabHost, pitch, widthOctet, heightOctet, cudaMemcpyHostToDevice));
	    }

	template<typename T> // TODO a tester
	static void memcpyDtoD(cudaArray* ptrCudaArray , T* tabSrc , uint w , uint h)
	    {
	    size_t widthOctet = w * sizeof(T);
	    size_t heightOctet = h * sizeof(T);
	    size_t pitch = widthOctet; // pitch = width + padding (pour passer a un elemnet la ligne du dessous)

	    HANDLE_ERROR(cudaMemcpy2DToArray(ptrCudaArray, 0, 0, tabSrc, pitch, widthOctet, heightOctet, cudaMemcpyDeviceToDevice));
	    }

	template<typename T> // TODO a tester
	static void memcpyDtoH(T* tabSrc , cudaArray* ptrCudaArray , uint w , uint h)
	    {
	    size_t widthOctet = w * sizeof(T);
	    size_t heightOctet = h * sizeof(T);
	    size_t pitch = widthOctet; // pitch = width + padding (pour passer a un elemnet la ligne du dessous)

	    HANDLE_ERROR(cudaMemcpy2DToArray(ptrCudaArray, 0, 0, tabSrc, pitch, widthOctet, heightOctet, cudaMemcpyDeviceToHost));
	    }

	/*--------------------------------------*\
	|*		private			*|
	 \*-------------------------------------*/

	static uint toNvidiaType(CudaArrayType type);
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
