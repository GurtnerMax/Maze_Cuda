#pragma once

//#include <cuda_runtime.h>
#include "cudas.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class GM
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*--------------*\
	|*   get 	 *|
	 \*-------------*/

	/**
	 * Go
	 */
	static int getMaxGo(int idDevice);

	/**
	 * GO
	 */
	static int getMaxGo();

	/*--------------*\
	|*   malloc	  *|
	 \*-------------*/

	template<typename T>
	static void malloc(T** ptrGM , size_t sizeOctet);

	/**
	 * malloc et memClear
	 */
	template<typename T>
	static void malloc0(T** ptrGM , size_t sizeOctet);

	static void mallocFloat(float** ptrGM , float initValue);
	static void mallocFloat0(float** ptrGM);

	static void mallocDouble(double** ptrGM , double initValue);
	static void mallocDouble0(double** ptrGM);

	static void mallocInt(int** ptrGM , int initValue);
	static void mallocInt0(int** ptrGM);

	static void mallocLong(long** ptrGM , long initValue);
	static void mallocLong0(long** ptrGM);

	static void mallocUchar(uchar** ptrGM , uchar initValue);
	static void mallocUchar0(uchar** ptrGM);

	/*--------------*\
	|*   free	  *|
	 \*-------------*/

	template<typename T>
	static void free(T* ptrGM);

	/**
	 * with 0
	 */
	template<typename T>
	static void memclear(T* ptrGM , size_t sizeOctet);

	/*--------------*\
	|*   copy	  *|
	 \*-------------*/

	/**
	 * for fastest copy use in host-side : cudaHostMalloc
	 */
	template<typename T>
	static void memcpyHToD(T* ptrGM , T* ptr , size_t sizeOctet);

	/**
	 * for fastest copy use in host-side : cudaHostMalloc : HM::malloc(...)
	 */
	template<typename T>
	static void memcpyDToH(T* ptr , T* ptrGM , size_t sizeOctet);

	template<typename T>
	static void memcpyDToD(T* ptrGM , T* ptrDevSrc , size_t sizeOctet);

	/*--------------*\
	|*	type     *|
	 \*-------------*/

	static void memcpyDToH_float(float* ptr , float* ptrGM);
	static void memcpyDToH_double(double* ptr , double* ptrGM);
	static void memcpyDToH_int(int* ptr , int* ptrGM);
	static void memcpyDToH_long(long* ptr , long* ptrGM);

	/*--------------*\
	|*	Async     *|
	 \*-------------*/

	/**
	 * for fastest copy use in host-side : cudaHostMalloc : HM::malloc(...)
	 */
	template<typename T>
	static void memcpyAsyncHToD(T* ptrDestGM , T* ptrSrc , size_t sizeOctet , cudaStream_t stream);

	/**
	 * for fastest copy use in host-side : cudaHostMalloc : HM::malloc(...)
	 */
	template<typename T>
	static void memcpyAsyncDToH(T* ptrDest , T* ptrDestGM , size_t sizeOctet , cudaStream_t stream);

	/**
	 * for fastest copy use in host-side : cudaHostMalloc : HM::malloc(...)
	 */
	template<typename T>
	static void memcpyAsyncDToD(T* ptrDestGM , T* ptrSrcGM , size_t sizeOctet , cudaStream_t stream);

	/*--------------*\
	|*	p2p     *|
	 \*-------------*/

	/**
	 * Use before once GM::p2pEnableAll()
	 */
	template<typename T>
	static void memcpyPeerDToD(T* ptrDestGM , int destDevice , T* ptrSrcGM , int srcDevice , size_t sizeOctet);

	/**
	 * Use before once GM::p2pEnableAll()
	 */
	template<typename T>
	static void memcpyAsyncPeerDToD(T* ptrDestGM , int destDevice , T* ptrSrcGM , int srcDevice , size_t sizeOctet,cudaStream_t stream=0);

	/**
	 * Opengl Warning : to be used after HANDLE_ERROR(cudaGLSetGLDevice(deviceId));
	 */
	static void p2pEnableALL();

	static void printP2PmatrixCompatibility();

	/**
	 * use delete[] ptrMatrixP2PCompatibility after usage
	 * raw major vectoriser
	 */
	static int* p2pMatrixCompatibility();

    };

#include "GM_MemoryManagement.cpp.h"
// contain static methode with template
// pas code completion!!
// include car template

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

