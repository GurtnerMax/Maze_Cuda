#pragma once

#include "Lock.cu.h"

#ifndef MAX
#define MAX(X,Y) ((X)>(Y)?(X):(Y))
#endif

__device__ int volatile mutexReductionMaxArgT = 0;	//variable global

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

// TODO a tester
class ReducerMaxArg
    {
    public:

	/**
	 * Hypothese:
	 *
	 * 	(H1) 	On suppose que T est un type  sur le quel > est defini
	 *
	 *  Usage example :
	 *
	 * 		Version1:
	 *
	 * 			ReducerMaxArg::reduce(tabMaxSM,tabArgmaxSM,ptrMaxGM,ptrArgMaxGM);
	 *
	 * 		Version2:
	 *
	 * 			#define XXX 		// avec XXX = REDUCER_MAXARG_OPTIMISER ou REDUCER_MAXARG_OPTIMISER_64
	 * 			#include "ReducerMaxArg.h
	 * 			ReducerMaxArg::reduce(tabMaxSM,tabArgmaxSM,ptrMaxGM,ptrArgMaxGM);
	 *
	 * Doc :
	 * 		see ReducerAdd.h
	 */
	template <typename T>
	static __device__ void reduce(T* tabMaxSM, int* tabArgmaxSM, T* ptrMaxGM,int* ptrArgMaxGM)
	    {
	    //tabSM=tabBlock

#ifdef REDUCER_MAXARG_OPTIMISER
	    reductionIntraBlock_v1(tabMaxSM,tabArgmaxSM);
#elif defined(REDUCER_MAXARG_OPTIMISER_64)
	    reductionIntraBlock_v2(tabMaxSM,tabArgmaxSM);
#else
	    reductionIntraBlock_v0(tabMaxSM,tabArgmaxSM);
#endif

	    //__syncthreads();

	    reductionInterBlock(tabMaxSM,tabArgmaxSM, ptrMaxGM,ptrArgMaxGM);
	    }

    private:

	/*--------------------------------------*\
	|*	reductionIntraBlock		*|
	 \*-------------------------------------*/

	/**
	 * used dans une boucle in reductionIntraBlock
	 */
	template <typename T>
	static __device__ void ecrasementV0(T* tabMaxSM, int* tabArgmaxSM,int middle)
	    {
	    // tidLocal=threadIdx.x
	    if (threadIdx.x < middle)
		{
		const int jump=threadIdx.x + middle;
		if(tabMaxSM[threadIdx.x]< tabMaxSM[jump])
		    {
		    tabMaxSM[threadIdx.x]=tabMaxSM[jump];
		    tabArgmaxSM[threadIdx.x]=jump;
		    }
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v0(T* tabMaxSM, int* tabArgmaxSM)
	    {
	    int middle = blockDim.x / 2;

	    while (middle >= 1)
		{
		ecrasementV0(tabMaxSM,tabArgmaxSM,middle);

		middle>>=1; // middle /= 2;

		__syncthreads();
		}
	    }

	/**
	 * used dans une boucle in reductionIntraBlock
	 */
	template <typename T>
	static __device__ void ecrasementV3(T* tabMaxSM, int* tabArgmaxSM,int middle)
	    {
	    //if (threadIdx.x < middle) // plus besoin a cause du break dans reductionIntraBlock after __syncthreads
		{
		const int jump=threadIdx.x + middle;
		if(tabMaxSM[threadIdx.x]< tabMaxSM[jump])
		    {
		    tabMaxSM[threadIdx.x]=tabMaxSM[jump];
		    tabArgmaxSM[threadIdx.x]=jump;
		    }
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v1(T* tabMaxSM,int* tabArgmaxSM)
	    {
	    int middle = blockDim.x / 2;

	    if (threadIdx.x < middle)
		{
		while (middle >= 1)
		    {
		    ecrasementV3(tabMaxSM,tabArgmaxSM,middle);

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
	static __device__ void reductionIntraBlock_v2(T* tabMaxSM,int* tabArgmaxSM)
	    {
	    // tidLocal=threadIdx.x

	    int middle = blockDim.x / 2;

	    if (threadIdx.x < middle)
		{

		//a 64 on ne divise plus
		// et on a besoin de 32 thread pour finir de reduire le 64 premieres cases
		while (middle >= 64)
		    {
		    ecrasementV3(tabMaxSM,tabArgmaxSM,middle);

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
		    volatile T* ptrDataMax=tabMaxSM;

		    if(ptrDataMax[threadIdx.x]< ptrDataMax[threadIdx.x+32])
			{
			ptrDataMax[threadIdx.x]=ptrDataMax[threadIdx.x+32];
			tabArgmaxSM[threadIdx.x+32]=threadIdx.x+32;
			}

		    if(ptrDataMax[threadIdx.x]< ptrDataMax[threadIdx.x+16])
			{
			ptrDataMax[threadIdx.x]=ptrDataMax[threadIdx.x+16];
			tabArgmaxSM[threadIdx.x+16]=threadIdx.x+16;
			}

		    if(ptrDataMax[threadIdx.x]< ptrDataMax[threadIdx.x+8])
			{
			ptrDataMax[threadIdx.x]=ptrDataMax[threadIdx.x+8];
			tabArgmaxSM[threadIdx.x+8]=threadIdx.x+8;
			}

		    if(ptrDataMax[threadIdx.x]< ptrDataMax[threadIdx.x+4])
			{
			ptrDataMax[threadIdx.x]=ptrDataMax[threadIdx.x+4];
			tabArgmaxSM[threadIdx.x+4]=threadIdx.x+4;
			}

		    if(ptrDataMax[threadIdx.x]< ptrDataMax[threadIdx.x+2])
			{
			ptrDataMax[threadIdx.x]=ptrDataMax[threadIdx.x+2];
			tabArgmaxSM[threadIdx.x+2]=threadIdx.x+2;
			}

		    if(ptrDataMax[threadIdx.x]< ptrDataMax[threadIdx.x+1])
			{
			ptrDataMax[threadIdx.x]=ptrDataMax[threadIdx.x+1];
			tabArgmaxSM[threadIdx.x+1]=threadIdx.x+1;
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
		Lock locker(&mutexReductionMaxArgT);
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
