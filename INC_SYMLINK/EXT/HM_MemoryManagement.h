#pragma once

#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------*\
 |*   malloc  *|
 \*-------------*/

template<typename T>
void HM::malloc(T** ptrHM , size_t sizeOctet , int hostMemoryType)
    {
    T* ptr = NULL;

    HANDLE_ERROR(cudaHostAlloc(&ptr, sizeOctet, hostMemoryType));
    //HANDLE_ERROR(cudaAllocHost(&ptr, sizeOctet, hostMemoryType));

    *ptrHM = ptr;
    }

template<typename T>
void HM::malloc(T** ptrHM , size_t sizeOctet , HostMemoryType hostMemoryType)
    {
    malloc(ptrHM, sizeOctet, toNvidiaType(hostMemoryType));
    }

/*--------------*\
 |*   free  *|
 \*-------------*/

template<typename T>
void HM::free(T* ptrHM)
    {
    //HANDLE_ERROR(cudaHostFree(ptrHM));
    HANDLE_ERROR(cudaFreeHost(ptrHM));
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

