#pragma once

#include "Lock.cu.h"

/*----------------------------------------------------------------------*\
 |*			Tools	 					*|
 \*---------------------------------------------------------------------*/

#define AtomicOp(name) void (*name)(T*, T)

template<typename T>
__device__ void atomicAddNatif(T* ptrA , T b)
    {
    atomicAdd(ptrA,b);
    }


/*----------------------------------------------------------------------*\
 |*			Classe	 					*|
 \*---------------------------------------------------------------------*/

class ReducerAdd
    {
    public:

	/**
	 * Hypothese:
	 *
	 *	(H0)	On suppose que T est un type ou + existe
	 *
	 * 	(H1) 	On suppose que T est un type simple sur lequel atomicAdd existe,
	 * 		sinon utiliser la version 3 ou on peut specifer un atomicAdd
	 *
	 * Usage example :
	 *
	 * 		Version1:
	 *
	 * 			ReducerAdd::reduce(tabSm,ptrResultGM);
	 *
	 * 		Version2:
	 *
	 * 			#define XXX		// avec XXX = REDUCER_ADD_OPTIMISER ou  REDUCER_ADD_OPTIMISER_64
	 * 			#include "ReducerAdd.h
	 * 			ReducerAdd::reduce(tabSm,ptrResultGM);//dg.x puissance de 2, >=64 pour REDUCER_ADD_OPTIMISER_64
	 *
	 * 		Version3:
	 *
	 *			#include "Lock.cu.h"
	 *			__device__ int volatile mutexAddDouble = 0;	//variable global
	 *			__device__ void addAtomicDouble(double* ptrX , double y) // 10x plus lent que version1, mais plus flexible
	 *				{
	 * 				Lock locker(&mutexAddDouble);
	 * 				locker.lock();
	 *				(*ptrX) += y;
	 *				locker.unlock();
	 *				}
	 *
	 * 			ReducerAdd::reduce(addAtomicDouble,tabSm,ptrResultGM);
	 * Contraintes :
	 *
	 * 	(C1) 	|tabSM| puissance de 2,  comme 64,128,256,512,1024 (>=64 pour REDUCER_ADD_OPTIMISER_64)
	 * 	(C2)	|ThreadByBlock|=|tabSM|
	 * 	(C3)	Reduction intra-thread laisser a l'utilsiateur (ie remplissage de tabSM)
	 * 	(C4)	Grid 1D
	 *
	 * Warning :
	 *
	 * 	(W1)	ptrResultGM n'est pas un tableau, mais un scalair contenant le resultat final
	 * 	(W2)	Oubliez pas le MM pour  ptrResultGM
	 * 	(W3)	Oubliez pas l'initialisation de ptrResultGM
	 * 		Exemples :
	 * 			adition 	: a zero avec un Device::memclear
	 * 			multiplication  : a 1 avec un Device:memcopyHtoD
	 */
	template <typename T>
	static __device__ void reduce(AtomicOp(ATOMIC_OP),T* tabSM, T* ptrResultGM)
	    {
	    //tabSM=tabBlock

#ifdef REDUCER_ADD_OPTIMISER
	    reductionIntraBlock_v1(tabSM);
#elif defined(REDUCER_ADD_OPTIMISER_64)
	    reductionIntraBlock_v2(tabSM);
#else
	    reductionIntraBlock_v0(tabSM);
#endif

	    //__syncthreads();

	    reductionInterBlock(ATOMIC_OP,tabSM, ptrResultGM);
	    }

	template <typename T>
	static __device__ void reduce(T* tabSM, T* ptrResultGM)
	    {
	    reduce(atomicAddNatif,tabSM,ptrResultGM);
	    }

    private:

	/*--------------------------------------*\
	|*	reductionIntraBlock		*|
	 \*-------------------------------------*/

	/**
	 * used dans une boucle in reductionIntraBlock
	 */
	template <typename T>
	static __device__ void ecrasement(T* tabSM, int middle)
	    {
	    // tidLocal=threadIdx.x
	    if (threadIdx.x < middle)
		{
		tabSM[threadIdx.x] += tabSM[threadIdx.x + middle];
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v0(T* tabSM)
	    {
	    int middle = blockDim.x / 2;

	    while (middle >= 1)
		{
		ecrasement(tabSM,middle);

		middle>>=1; // middle /= 2;

		__syncthreads();
		}
	    }

	/**
	 * Sur place, le resultat est dans tabSM[0]
	 */
	template <typename T>
	static __device__ void reductionIntraBlock_v1(T* tabSM)
	    {
	    int middle = blockDim.x / 2;

	    if (threadIdx.x < middle)
		{
		while (middle >= 1)
		    {
		    // ecrasement
		    //if (threadIdx.x < middle) // plus besoin a cause du break ci-dessous, et du if ci-dessus
			{
			tabSM[threadIdx.x] += tabSM[threadIdx.x + middle];
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
	static __device__ void reductionIntraBlock_v2(T* tabSM)
	    {
	    // tidLocal=threadIdx.x

	    int middle = blockDim.x / 2;

	    if (threadIdx.x < middle)
		{

		//a 64 on ne divise plus
		// et on a besoin de 32 thread pour finir de reduire le 64 premieres cases
		while (middle >= 64)
		    {
		    // ecrasement
		    // if (threadIdx.x < middle) // plus besoin a cause du break ci-dessous, et du if ci-dessus
			{
			tabSM[threadIdx.x] += tabSM[threadIdx.x + middle];
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

		    ptrData[threadIdx.x]+=ptrData[threadIdx.x+32];//  each thread of the warp execute this line in the same time. Aucun thread ne peut prendre de l'avance! A la fine de cette ligne, semantique dans les 32 premieres cases

		    //if(threadIdx.x<16)		// pas necessaire car lecture(#) de ptrData[16] avant écriture(##) dans ptrData[16] ecriture ptrData[0]+=ptrData[16] (action #)  et ptrData[16]+=ptrData[32] (action ##)
		    ptrData[threadIdx.x]+=ptrData[threadIdx.x+16];//  Apres cette ligne semantique dans les 16 premières cases. Seuls les 16 premiers threads sont utiles

		    //if(threadIdx.x<8)
		    ptrData[threadIdx.x]+=ptrData[threadIdx.x+8];//  Apres cette ligne semantique dans les 8 premières cases. Seuls les 8 premiers threads sont utiles

		    //if(threadIdx.x<4)
		    ptrData[threadIdx.x]+=ptrData[threadIdx.x+4];// ...

		    //if(threadIdx.x<2)
		    ptrData[threadIdx.x]+=ptrData[threadIdx.x+2];

		    //if(threadIdx.x<1)
		    ptrData[threadIdx.x]+=ptrData[threadIdx.x+1];
		    }
		}
	    }

	/*--------------------------------------*\
	|*	reductionInterblock		*|
	 \*-------------------------------------*/

	template <typename T>
	static __device__ void reductionInterBlock(AtomicOp(ATOMIC_OP),T* tabSM, T* ptrResultGM)
	    {
	    if (threadIdx.x == 0)
		{
		//atomicAdd(ptrResultGM, tabSM[0]); // autant d'acces que de block
		ATOMIC_OP(ptrResultGM, tabSM[0]);// autant d'acces que de block
		}
	    }

    };

/*----------------------------------------------------------------------*\
|*			End	 					*|
 \*---------------------------------------------------------------------*/
