#pragma once

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Thread2D_long
    {

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*------------------*\
	|*	tid Global   *|
	 \*-----------------*/

	/**
	 * Best performance with entrelacement pattern! Minimize memory transaction!
	 * Warning: If 2d is not required, use grid1D cause process tid required less aritmetical operation
	 *
	 *  ----------------------------------
	 *  | 0  1  2 |  6  7  8 | 12 13 14 |
	 *  | 3  4  5 |  9 10 11 | 15 16 17 |
	 *  ----------------------------------
	 *  | 18 19 20 | 24 25 26 | 30 31 32 |
	 *  | 21 22 23 | 27 28 29 | 33 34 35 |
	 *  ----------------------------------
	 */
	__device__
	inline static long tid()
	    {
	    return ((long)blockDim.x * (long)blockDim.y) * (((long)gridDim.x * (long)blockIdx.y) + (long)blockIdx.x) + ((long)blockDim.x * (long)threadIdx.y)
		    + (long)threadIdx.x;
	    // Note (blockDimk.x*threadIdx.y)+threadIdx.x = tidLocal
	    }

	/**
	 * tidRawMajor : Usefull for tiling gm to sm !
	 *
	 *  ----------------------------------
	 *  | 0  1  2 |  3  4  5 |  6  7  8 |
	 *  | 9 10 11 | 12 13 14 | 15 16 17 |
	 *  ----------------------------------
	 *  | 18 19 20 | 21 22 23 | 24 25 26 |
	 *  | 27 28 29 | 30 31 32 | 33 34 35 |
	 *  ----------------------------------
	 */
	__device__
	inline static long tidRawMajor()
	    {
	    // idem tidColMajor x<-->y
	    return ((long)threadIdx.x + ((long)blockDim.x * (long)blockIdx.x))
		    + (((long)threadIdx.y + ((long)blockDim.y * (long)blockIdx.y)) * ((long)gridDim.x * (long)blockDim.x));
	    }

	/**
	 *  tidColMajor : Usefull for tiling sm to gm ! (transpose algo)
	 *
	 *  ----------------------------------
	 *  | 0  4  8 | 12 16 20 | 24 28 32 |
	 *  | 1  5  9 | 13 17 21 | 25 29 33 |
	 *  ----------------------------------
	 *  | 2  6 10 | 14 18 22 | 26 30 34 |
	 *  | 3  7 11 | 15 19 23 | 27 31 35 |
	 *  ----------------------------------
	 */
	__device__
	inline static long tidColMajor()
	    {
	    // idem tidRawMajor x<-->y
	    return (long)threadIdx.y + ((long)blockDim.y * (long)blockIdx.y)
		    + (((long)threadIdx.x + ((long)blockDim.x * (long)blockIdx.x)) * ((long)gridDim.y * (long)blockDim.y));
	    }

	/*------------------*\
	|*	tid Local   *|
	 \*-----------------*/

	/**
	 * output in [0,nbThreadBlock()[
	 * return threadIdx.x+blockDim.x*threadIdx.y;
	 */
	__device__
	inline static long tidLocalBlock()
	    {
	    return threadIdx.x + blockDim.x * threadIdx.y;
	    }

	/**
	 * idem tidLocalBlock
	 */
	__device__
	inline static int tidBlock()
	    {
	    return threadIdx.x + blockDim.x * threadIdx.y;
	    }

	/**
	 * idem tidLocalBlock
	 */
	__device__
	inline static int tidLocal()
	    {
	    return threadIdx.x + blockDim.x * threadIdx.y;
	    }

	/*------------------*\
	|*	nbThread    *|
	 \*-------------------*/

	__device__
	inline static long nbThread()
	    {
	    return ((long)gridDim.x * (long)gridDim.y) * ((long)blockDim.x * (long)blockDim.y);
	    }

	__device__
	inline static long nbThreadX()
	    {
	    return (long)gridDim.x * (long)blockDim.x;
	    }

	__device__
	inline static long nbThreadY()
	    {
	    return (long)gridDim.y * (long)blockDim.y;
	    }

	/**
	 * idem nbThreadLocal
	 */
	__device__
	inline static int nbThreadBlock()
	    {
	    return blockDim.x * blockDim.y;
	    }

	/**
	 * idem nbThreadBlock
	 */
	__device__
	inline static int nbThreadLocal()
	    {
	    return blockDim.x * blockDim.y;
	    }

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
