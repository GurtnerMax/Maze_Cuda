#pragma once

#include "Lock.cu.h"

#ifndef MIN
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#endif

__device__ int volatile mutexReductionMinArgT = 0;	//variable global

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

// TODO a tester
class ReducerMinArg
    {
    public:

	/**
	 * Hypothese:
	 *
	 * 	(H1) 	On suppose que T est un type sur lequel < est defini
	 *
	 *  Usage example :
	 *
	 * 		Version1:
	 *
	 * 			ReducerMinArg::reduce(tabMinSM,tabArgminSM,ptrMinGM,ptrArgMinGM);
	 *
	 * 		Version2:
	 *
	 * 			#define XXX		// avec REDUCER_MINARG_OPTIMISER ou REDUCER_MINARG_OPTIMISER_64
	 * 			#include "ReducerMinArg.h
	 * 			ReducerMinArg::reduce(tabMinSM,tabArgminSM,ptrMinGM,ptrArgMinGM);
	 *
	 * Doc :
	 * 		see ReducerAdd.h
	 */
	template <typename T>
	static __device__ void reduce(T* tabMinSM, int* tabArgminSM, T* ptrMinGM,int* ptrArgMinGM)
	    {
	    //tabSM=tabBlock

#ifdef REDUCER_MINARG_OPTIMISER
	    reductionIntraBlock_v1(tabMinSM,tabArgminSM);
#elif defined(REDUCER_MINARG_OPTIMISER_64)
	    reductionIntraBlock_v2(tabMinSM,tabArgminSM);
#else
	    reductionIntraBlock_v0(tabMinSM,tabArgminSM);
#endif

	    //__syncthreads();

	    reductionInterBlock(tabMinSM,tabArgminSM, ptrMinGM,ptrArgMinGM);
	    }

    private:

	/*--------------------------------------*\
	|*	reductionIntraBlock		*|
	 \*-------------------------------------*/

	/**
	 * used dans une boucle in reductionIntraBlock
	 */
	template <typename T>
	static __device__ void ecrasementV0(T* tabMinSM, int* tabArgminSM,int middle)
	    {
	    // tidLocal=threadIdx.x
	    if (threadIdx.x < middle)
		{
		const int jump=threadIdx.x + middle;
		if(tabMinSM[threadIdx.x]> tabMinSM[jump])
		    {
		    tabMinSM[threadIdx.x]=tabMinSM[jump];
		    tabArgminSM[threadIdx.x]=jump;
		    }
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v0(T* tabMinSM, int* tabArgminSM)
	    {
	    int middle = blockDim.x / 2;

	    while (middle >= 1)
		{
		ecrasementV0(tabMinSM,tabArgminSM,middle);

		middle>>=1; // middle /= 2;

		__syncthreads();
		}
	    }

	/**
	 * used dans une boucle in reductionIntraBlock
	 */
	template <typename T>
	static __device__ void ecrasementV3(T* tabMinSM, int* tabArgminSM,int middle)
	    {
	    //if (threadIdx.x < middle) // plus besoin a cause du break dans reductionIntraBlock after __syncthreads
		{
		const int jump=threadIdx.x + middle;
		if(tabMinSM[threadIdx.x]> tabMinSM[jump])
		    {
		    tabMinSM[threadIdx.x]=tabMinSM[jump];
		    tabArgminSM[threadIdx.x]=jump;
		    }
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v1(T* tabMinSM,int* tabArgminSM)
	    {
	    int middle = blockDim.x / 2;

	    if (threadIdx.x < middle)
		{
		while (middle >= 1)
		    {
		    ecrasementV3(tabMinSM,tabArgminSM,middle);

		    middle>>=1; // middle /= 2;

		    __syncthreads();

		    if(threadIdx.x >= middle)// after __syncthreads();
			{
			break;
			}
		    }
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v2(T* tabMinSM,int* tabArgminSM)
	    {
	    // tidLocal=threadIdx.x

	    int middle = blockDim.x / 2;

	    if (threadIdx.x < middle)
		{

		//a 64 on ne divise plus
		// et on a besoin de 32 thread pour finir de reduire le 64 premieres cases
		while (middle >= 64)
		    {
		    ecrasementV3(tabMinSM,tabArgminSM,middle);

		    middle>>=1; //middle /= 2;

		    __syncthreads();

		    if(threadIdx.x >= middle)// after __syncthreads();
			{
			break;
			}
		    }

		// Dans le cas midlde=32, la semantique est dans les 64 premieres cases.
		// Utilisation des 32 thread d'un warp pour finir la reduction des 64 premieres cases, sans synchronisation
		if(threadIdx.x<32)
		    {
		    // no __syncthreads() necessary after each of the following lines as long as  we acces the data via a pointer decalred as volatile
		    // because the 32 thread in each warp execute in a locked-step with each other
		    volatile T* ptrDataMin=tabMinSM;

		    if(tabMinSM[threadIdx.x]> tabMinSM[threadIdx.x+32])
			{
			ptrDataMin[threadIdx.x]=tabMinSM[threadIdx.x+32];
			tabArgminSM[threadIdx.x+32]=threadIdx.x+32;
			}

		    if(tabMinSM[threadIdx.x]> tabMinSM[threadIdx.x+16])
			{
			ptrDataMin[threadIdx.x]=tabMinSM[threadIdx.x+16];
			tabArgminSM[threadIdx.x+16]=threadIdx.x+16;
			}

		    if(tabMinSM[threadIdx.x]> tabMinSM[threadIdx.x+8])
			{
			ptrDataMin[threadIdx.x]=tabMinSM[threadIdx.x+8];
			tabArgminSM[threadIdx.x+8]=threadIdx.x+8;
			}

		    if(tabMinSM[threadIdx.x]> tabMinSM[threadIdx.x+4])
			{
			ptrDataMin[threadIdx.x]=tabMinSM[threadIdx.x+4];
			tabArgminSM[threadIdx.x+4]=threadIdx.x+4;
			}

		    if(tabMinSM[threadIdx.x]> tabMinSM[threadIdx.x+2])
			{
			ptrDataMin[threadIdx.x]=tabMinSM[threadIdx.x+2];
			tabArgminSM[threadIdx.x+2]=threadIdx.x+2;
			}

		    if(tabMinSM[threadIdx.x]> tabMinSM[threadIdx.x+1])
			{
			ptrDataMin[threadIdx.x]=tabMinSM[threadIdx.x+1];
			tabArgminSM[threadIdx.x+1]=threadIdx.x+1;
			}
		    }
		}
	    }

	/*--------------------------------------*\
	|*	reductionInterblock		*|
	 \*-------------------------------------*/

	template <typename T>
	static __device__ void reductionInterBlock(T* tabMinSM, int* tabArgminSM,T* ptrMinGM,int* ptrArgMinGM)
	    {
	    if (threadIdx.x == 0)
		{
		Lock locker(&mutexReductionMinArgT);
		locker.lock();

		// section critique
		    {
		    if(tabMinSM[0]< *ptrMinGM)
			{
			*ptrMinGM=tabMinSM[0];
			*ptrArgMinGM=tabArgminSM[0];
			}
		    }

		locker.unlock();
		}
	    }

    };

/*----------------------------------------------------------------------*\
|*			End	 					*|
 \*---------------------------------------------------------------------*/
