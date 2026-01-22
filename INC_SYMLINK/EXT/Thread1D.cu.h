#pragma once

class Thread1D
    {

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*------------------*\
	|*	tid Global   *|
	 \*-----------------*/

	/**
	 * i in [0,nbThreadX-1]
	 * i=threadIdx.y + (blockDim.y * blockIdx.y)
	 */
	__device__
	inline static int tid()
	    {
	    return threadIdx.x + (blockDim.x * blockIdx.x);
	    }

	/*------------------*\
	|*	tid Local   *|
	 \*-----------------*/

	/**
	 * output in [0,nbThreadBlock()[
	 * return threadIdx;
	 */
	__device__
	inline static int tidLocalBlock()
	    {
	    return threadIdx.x;
	    }

	/**
	 * idem tidLocalBlock
	 */
	__device__
	inline static int tidBlock()
	    {
	    return threadIdx.x;
	    }

	/**
	 * idem tidLocalBlock
	 */
	__device__
	inline static int tidLocal()
	    {
	    return threadIdx.x;
	    }

	/*------------------*\
	|*	nbThread    *|
	 \*-------------------*/

	__device__
	inline static int nbThread()
	    {
	    return gridDim.x * blockDim.x;
	    }

	__device__
	inline static int nbThreadBlock()
	    {
	    return blockDim.x;
	    }

	/**
	 * idem nbThreadBlock
	 */
	__device__
	inline static int nbThreadLocal()
	    {
	    return blockDim.x ;
	    }

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
