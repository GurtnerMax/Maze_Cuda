#pragma once

#include "cudas.h"

#include "Surface.h"
#include "CudaArray.h"

/**
 * With a cudaSurfaceObject_t, we can :
 *
 *	write:		surf2Dwrite(dataS, surf2d, j, i, boundaryMode ); // boundaryMode facultatif
 *
 *	read:		surf2Dread(&dataS, surf2d, j, i, boundaryMode ); // boundaryMode facultatif
 *			T data=surf2Dread<T>( surf2d, j, i, boundaryMode ); // boundaryMode facultatif
 *
 * BoundaryMode (for out-of-bounds addresses):
 *
 *		- cudaBoundaryModeZero
 *		- cudaBoundaryModeClamp   (pour lire en dehors)
 *		- cudaBoundaryModeTrap
 * Warning:
 *		(W1)	(j,i) et non (i,j)
 *		(W2) 	A la creation du cudaArray, utiliser : cudaArraySurfaceLoadStore (mais semble marcher meme avec les autres, bizarre?)
 *		(W3) 	surface ideme texture, but no interpolation or data conversion
 *
 * Interresting:
 *
 * 		A same cudaArray can be use with a texture and a surface
 *
 * Background:
 *
 * 		cudaArray
 */
template<class T>
class Surface2D
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	Surface2D(int w , int h , CudaArrayType cudaArrayType = CudaArrayType::DEFAULT_CUARRAY) : //
		w(w), //
		h(h), //
		cudaArrayType(cudaArrayType)
	    {
	    create();
	    }

	virtual ~Surface2D()
	    {
	    Surface::destroy(surfCuda);
	    CudaArray::free(ptrCudaArray);
	    }

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

	/**
	 * meme output cudaSurfaceObject_t que l'attribut surfCuda
	 */
	cudaTextureObject_t* memcpyHtoD(T* tabHost) // TODO a tester
	    {
	    CudaArray::memcpyHtoD(ptrCudaArray, tabHost, w, h);

	    return &surfCuda;
	    }

	/**
	 * meme output cudaSurfaceObject_t que l'attribut surfCuda
	 */
	cudaTextureObject_t* memcpyDtoD(T* tabSource) // TODO a tester
	    {
	    CudaArray::memcpyDtoD(ptrCudaArray, tabSource, w, h);

	    return &surfCuda;
	    }

	/**
	 * meme output cudaSurfaceObject_t que l'attribut surfCuda
	 */
	cudaTextureObject_t* memcpyDtoH(T* tabHost) // TODO a tester
	    {
	    CudaArray::memcpyDtoH(tabHost, ptrCudaArray, w, h);

	    return &surfCuda;
	    }

    public:

	/*--------------------------------------*\
	|*		private			*|
	 \*-------------------------------------*/

    private:

	void create()
	    {
	    CudaArray::malloc<T>(&ptrCudaArray, w, h, cudaArrayType);

	    Surface::create<T>(&surfCuda, &ressourceDescription, ptrCudaArray, w, h);
	    }

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int w;
	int h;
	CudaArrayType cudaArrayType;

	// Tools cudarrays
	cudaArray* ptrCudaArray;
	cudaResourceDesc ressourceDescription;

    public:

	// outputs
	cudaSurfaceObject_t surfCuda;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
