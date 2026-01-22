#pragma once

#include "Reducer.cu.h"

/*----------------------------------------------------------------------*\
 |*			Tools	 					*|
 \*---------------------------------------------------------------------*/

#ifndef MIN
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#endif

// existe deja
//
//__device__ float min(float a , float b)
//    {
//    return MIN(a, b);
//    }

template<typename T>
__device__ void minAtomicT(T* ptrA , T b)
    {
    atomicMin(ptrA,b);
    }

/*----------------------------------------------------------------------*\
 |*			Classe	 					*|
 \*---------------------------------------------------------------------*/

class ReducerMin
    {
    public:

	/**
	 * Hypothese:
	 *
	 * 		(H1)	Operator > is define in classe T
	 *
	 * 		(H2) 	T own a atomiMin operator, else specify one with version 3
	 *
	 * Usage example :
	 *
	 * 		Version1:
	 *
	 * 			ReductionMin::reduce(tabSm,ptrResultGM);
	 *
	 * 		Version2:
	 *
	 *
	 * 			#define XXX		// avec XXX REDUCER_OPTIMISER ou REDUCER_OPTIMISER_64
	 * 			#include "ReducerMin.h
	 * 			ReducerMin::reduce(tabSm,ptrResultGM); // contrainte :  db.x>=64 avec REDUCER_OPTIMISER_64
	 *
	 * 		Version 3 :
	 *
	 *			#include "Lock.cu.h"
	 * 			static __device__ int volatile mutexReducerMinT=0;
	 *			template<typename T>
	 *			__device__ void maxAtomicT(T* ptrA , T b)
	 *					{
	 *					Lock locker(&mutexReducerMinT);
	 *					locker.lock();
	 *					*ptrA = MAX(*ptrA, b);
	 *					locker.unlock();
	 *					}
	 *
	 *			#define XXX		// avec XXX REDUCER_OPTIMISER ou REDUCER_OPTIMISER_64
	 * 			#include "ReducerMin.h
	 * 			ReducerMin::reduce(minAtomic,tabSm,ptrResultGM); // contrainte :  db.x>=64 avec REDUCER_OPTIMISER_64
	 *
	 *			__device__ void minAtomic(float* ptrA , float b)
	 *				{
	 *			         atomicMin(ptrA,b);	// if exist with your version of cuda and of hardware
	 *				}
	 * Doc :
	 * 		see ReducerAdd.h
	 */
	template <typename T>
	static __device__ void reduce(AtomicOp(ATOMIC_OP), T* tabSM, T* ptrResultGM)
	    {
	    Reducer::reduce(min, ATOMIC_OP, tabSM, ptrResultGM);
	    }

	template <typename T>
	static __device__ void reduce(T* tabSM, T* ptrResultGM)
	    {
	    reduce(minAtomicT,tabSM,ptrResultGM);
	    }

    private:

	/*--------------------------------------*\
	|*	Private				*|
	 \*-------------------------------------*/

    };

/*----------------------------------------------------------------------*\
|*			End	 					*|
 \*---------------------------------------------------------------------*/
