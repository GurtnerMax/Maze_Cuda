#pragma once

#include "Reducer.cu.h"

/*----------------------------------------------------------------------*\
 |*			Tools	 					*|
 \*---------------------------------------------------------------------*/

#ifndef MAX
#define MAX(X,Y) ((X)>(Y)?(X):(Y))
#endif

// existe deja
//
//__device__ float max(float a , float b)
//    {
//    return MAX(a, b);
//    }


template<typename T>
__device__ void maxAtomicT(T* ptrA , T b)
    {
    atomicMax(ptrA,b);
    }

/*----------------------------------------------------------------------*\
 |*			Classe	 					*|
 \*---------------------------------------------------------------------*/

class ReducerMax
    {
    public:

	/**
	 * Hypothese:
	 *
	 * 		(H1)	Operator > is define in classe T
	 *
	 * 		(H2) 	T own a atomiMax operator, else specify one with version 3
	 *
	 * Optimisation
	 *
	 * 		(O1)	Specify your own atomicMax operator 	(else un general lock is use)
	 *
	 * Usage example :
	 *
	 * 		Version1:
	 *
	 * 			ReductionMax::reduce(tabSm,ptrResultGM);
	 *
	 * 		Version2:
	 *
	 *
	 * 			#define XXX		// avec XXX REDUCER_OPTIMISER ou REDUCER_OPTIMISER_64
	 * 			#include "ReducerMax.h
	 * 			ReducerMax::reduce(tabSm,ptrResultGM); // contrainte :  db.x>=64 avec REDUCER_OPTIMISER_64
	 *
	 * 		Version 3 :
	 *
	 *			#include "Lock.cu.h"
	 *			static __device__ int volatile mutexReducerMaxT=0;
	 *			template<typename T>
	 *			__device__ void maxAtomicT(T* ptrA , T b)
	 *					{
	 *					Lock locker(&mutexReducerMaxT);
	 *					locker.lock();
	 *					*ptrA = MAX(*ptrA, b);
	 *					locker.unlock();
	 *					}
	 *
	 *			#define XXX		// avec XXX REDUCER_OPTIMISER ou REDUCER_OPTIMISER_64
	 * 			#include "ReducerMax.h
	 * 			ReducerMax::reduce(maxAtomic,tabSm,ptrResultGM); // contrainte :  db.x>=64 avec REDUCER_OPTIMISER_64
	 *
	 *
	 *			__device__ void maxAtomic(float* ptrA , float b)
	 *				{
	 *			         atomicMax(ptrA,b);	// if exist with your version of cuda and of hardware
	 *				}

	 * Doc :
	 * 		see ReducerAdd.h
	 */
	template <typename T>
	static __device__ void reduce(AtomicOp(ATOMIC_OP), T* tabSM, T* ptrResultGM)
	    {
	    Reducer::reduce(max, ATOMIC_OP, tabSM, ptrResultGM);
	    }

	template <typename T>
	static __device__ void reduce(T* tabSM, T* ptrResultGM)
	    {
	    reduce(maxAtomicT,tabSM,ptrResultGM);
	    }

    private:

	/*--------------------------------------*\
	|*	Private				*|
	 \*-------------------------------------*/

    };

/*----------------------------------------------------------------------*\
|*			End	 					*|
 \*---------------------------------------------------------------------*/
