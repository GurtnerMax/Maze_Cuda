#pragma once

#include "Lock.cu.h"

/*----------------------------------------------------------------------*\
 |*			prt fonction / reduction			*|
 \*---------------------------------------------------------------------*/

#define BinaryOperator(name) T (*name)(T, T)
#define AtomicOp(name) void (*name)(T*, T)

/*----------------------------------------------------------------------*\
 |*			Note Implementation				*|
 \*---------------------------------------------------------------------*/

// 	Avec
//
//		#define BinaryOperator void (*OP)(T*, T) // update T*
//
// 	On obtient les memes performance, mais on ne peut plus utiliser la version optimiser avec le volatile!
//
/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

class Reducer
    {
    public:

	/**
	 * Hypothese:
	 *
	 * 	(H1) 	BinaryOperator un operateur binaire sur des element de Type T
	 * 	(H2)	AtomicOp permet de realiser des operations atomics
	 *
	 * Usage example :
	 *
	 * 		Version1:
	 *
	 * 			__device__ int add(int x, int y){return x+y;}
	 * 			__device__ void addAtomic(int* ptrX, int y){atomicAdd(ptrX,y);}
	 *
	 * 			Reducer::reduce(add,addAtomic,tabSm,ptrResultGM);
	 *
	 * 		Version2:
	 *
	 * 			__device__ int add(int x, int y){return x+y;}
	 *
	 *			#include "Lock.cu.h"
	 * 			__device__ int volatile mutex = 0;	//variable global
	 * 			__device__ void addAtomic(int* ptrX, int y) // 10x plus lent que version1, mais plus flexible
	 * 				{
	 * 				Lock locker(&mutex);
	 * 				locker.lock();
	 * 				(*ptrX)+=y;
	 * 				locker.unlock();
	 * 				}
	 *
	 * 			Reducer::reduce(add,addAtomic,tabSm,ptrResultGM);
	 *
	 * 		Version3:
	 *
	 * 			__device__ int add(int x, int y){return x+y;}
	 * 			__device__ void addAtomic(int* x, int y){atomicAdd(x,y);}
	 *
	 * 			#define XXX		// avec XXX REDUCER_OPTIMISER ou REDUCER_OPTIMISER_64
	 * 			#include "Reducer.h
	 * 			Reducer::reduce(add,addAtomic,tabSm,ptrResultGM); // contrainte :  db.x>=64 avec REDUCER_OPTIMISER_64
	 *
	 * Doc :
	 * 		see ReducerAdd.h
	 */
	template <typename T>
	static __device__ void reduce(BinaryOperator(OP) ,AtomicOp(ATOMIC_OP), T* tabSM, T* ptrResultGM)
	    {
	    //tabSM=tabBlock

#ifdef REDUCER_OPTIMISER
	    reductionIntraBlock_v1(OP,tabSM);
#elif defined(REDUCER_OPTIMISER_64)
	    reductionIntraBlock_v2(OP,tabSM);
#else
	    reductionIntraBlock_v0(OP,tabSM);
#endif

	    __syncthreads();

	    reductionInterBlock(ATOMIC_OP,tabSM, ptrResultGM);
	    }

    private:

	/*--------------------------------------*\
	|*	reductionIntraBlock		*|
	 \*-------------------------------------*/

	/**
	 * used dans une boucle in reductionIntraBlock
	 */
	template <typename T>
	static __device__ void ecrasement(BinaryOperator(OP),T* tabSM, int middle)
	    {
	    // tidLocal=threadIdx.x
	    if (threadIdx.x < middle)
		{
		tabSM[threadIdx.x] =OP(tabSM[threadIdx.x], tabSM[threadIdx.x + middle]);
		//OP(&tabSM[threadIdx.x], tabSM[threadIdx.x + middle]); // pas mieux
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v0(BinaryOperator(OP),T* tabSM)
	    {
	    int middle = blockDim.x / 2;

	    while (middle >= 1)
		{
		ecrasement(OP,tabSM,middle);

		middle>>=1;	   // middle /= 2;

		__syncthreads();
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v1(BinaryOperator(OP),T* tabSM)
	    {
	    int middle = blockDim.x / 2;

	    if (threadIdx.x < middle)
		{
		while (middle >= 1)
		    {
		    // ecrasement
		    //if (threadIdx.x < middle) // plus besoin a cause du break ci-dessous, et du if ci-dessus
			{
			tabSM[threadIdx.x] =OP(tabSM[threadIdx.x], tabSM[threadIdx.x + middle]);
			}

		    middle>>=1;	   // middle /= 2;

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
	static __device__ void reductionIntraBlock_v2(BinaryOperator(OP),T* tabSM)
	    {
	    // tidLocal=threadIdx.x

	    int middle = blockDim.x / 2;

	    if (threadIdx.x < middle)
		{

		// a 64 on ne divise plus
		// et on a besoin de 32 thread pour finir de reduire le 64 premieres cases
		while (middle >= 64)
		    {
		    // ecrasement
		    // if (threadIdx.x < middle) // plus besoin a cause du break ci-dessous, et du if ci-dessus
			{
			tabSM[threadIdx.x] =OP(tabSM[threadIdx.x], tabSM[threadIdx.x + middle]);
			}

		    middle>>=1; //middle /= 2;

		    __syncthreads();

		    if(threadIdx.x >= middle)// after __syncthreads();
			{
			break;
			}
		    }

		// now midlde=32

		// Dans le cas midlde=32, la semantique est dans les 64 premieres cases.
		// Utilisation des 32 thread d'un warp pour finir la reduction des 64 premieres cases, sans synchronisation
		if(threadIdx.x<32)
		    {
		    // no __syncthreads() necessary after each of the following lines as long as  we acces the data via a pointer decalred as volatile
		    // because the 32 therad in each warp execute in a locked-step with each other
		    volatile T* ptrData=tabSM;

		    ptrData[threadIdx.x]=OP(ptrData[threadIdx.x],ptrData[threadIdx.x+32]);//  each thread of the warp execute this line in the same time. Aucun thread ne peut prendre de l'avance! A la fine de cette ligne, semantique dans les 32 premieres cases

		    //if(threadIdx.x<16)		// pas necessaire car lecture(#) de ptrData[16] avant écriture(##) dans ptrData[16] ecriture ptrData[0]+=ptrData[16] (action #)  et ptrData[16]+=ptrData[32] (action ##)
		    ptrData[threadIdx.x]=OP(ptrData[threadIdx.x],ptrData[threadIdx.x+16]);//  Apres cette ligne semantique dans les 16 premières cases. Seuls les 16 premiers threads sont utiles

		    //if(threadIdx.x<8)
		    ptrData[threadIdx.x]=OP(ptrData[threadIdx.x],ptrData[threadIdx.x+8]);//  Apres cette ligne semantique dans les 8 premières cases. Seuls les 8 premiers threads sont utiles

		    //if(threadIdx.x<4)
		    ptrData[threadIdx.x]=OP(ptrData[threadIdx.x],ptrData[threadIdx.x+4]);// ...

		    //if(threadIdx.x<2)
		    ptrData[threadIdx.x]=OP(ptrData[threadIdx.x],ptrData[threadIdx.x+2]);

		    //if(threadIdx.x<1)
		    ptrData[threadIdx.x]=OP(ptrData[threadIdx.x],ptrData[threadIdx.x+1]);
		    }
		}
	    }

	/*--------------------------------------*\
	|*	reductionInterblock		*|
	 \*-------------------------------------*/

	template <typename T>
	static __device__ void reductionInterBlock(AtomicOp(ATOMIC_OP), T* tabSM, T* ptrResultGM)
	    {
	    if(threadIdx.x==0)
		{
		ATOMIC_OP(ptrResultGM,tabSM[0]);
		}
	    }

    };

/*----------------------------------------------------------------------*\
|*			End	 					*|
 \*---------------------------------------------------------------------*/
