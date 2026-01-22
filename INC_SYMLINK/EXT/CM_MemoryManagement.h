#pragma once

#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/


/**
 * <pre>
 * T* ptrSymbolCM;
 * HANDLE_ERROR(cudaGetSymbolAddress((void**)&ptrSymbolCM, ptrCM));
 *
 *  Device::memcpyToCM( ptrSymbolCM , ptrSrc ,sizeOctet)
 * </pre>
 */
template<typename T>
void CM::memcpyToCM(T* ptrSymbolCM , T* ptrSrc , size_t sizeOctet)
    {
    std::cout << "[ConstantMemory::memcpyToCM] : Warning !" << std::endl;
    std::cout << std::endl;
    std::cout << "\t If you meet problem with this methode, use instead : either-either :" << std::endl;
    std::cout << std::endl;
    std::cout << "Version 1 :\n" << std::endl;
    std::cout << std::endl;
    std::cout << "\tT* ptrSymbolCM;" << std::endl;
    std::cout << "\tHANDLE_ERROR(cudaGetSymbolAddress((void**)&ptrSymbolCM, ptrCM)); " << std::endl; // on recuper le symbole du ptr de la cm sur le device
    std::cout << "\tDevice::memcpyToCM(ptrSymbolCM ,ptrSrc ,sizeOctet)" << std::endl;
    std::cout << std::endl;
    std::cout << "Version 2 :\n" << std::endl;
    std::cout << std::endl;
    std::cout << "\tHANDLE_ERROR(cudaMemcpyToSymbol(ptrCM, ptrSrc, sizeOctet,cudaMemcpyHostToDevice));" << std::endl;
    std::cout << std::endl;

    // v1: ko :
    // 		On arrive pas passer la cm qui est un symbole a cette methode! crash a l'appel!
    // 		Il faut recuperer le symbole avant!
	{
	// T* ptrSymbolCM;
	// HANDLE_ERROR(cudaGetSymbolAddress((void**)&ptrSymbolCM, ptrCM)); // on recuper le symbole du ptr de la cm sur le device
	// HANDLE_ERROR(cudaMemcpy(ptrSymbolCM, ptrSrc, sizeOctet,  cudaMemcpyHostToDevice));
	}

// v2
	{
	HANDLE_ERROR(cudaMemcpy(ptrSymbolCM, ptrSrc, sizeOctet, cudaMemcpyHostToDevice));
	}
    }
/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

