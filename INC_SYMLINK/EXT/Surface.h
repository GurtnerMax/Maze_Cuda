#pragma once

#include "cudas.h"
#include "RessourceDescription.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

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
 *		- cudaBoundaryModeClamp  (pour lire en dehors)
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
class Surface
    {

    public:

	/*--------------------------------------*\
	|*		custom			*|
	 \*-------------------------------------*/

	static void create(cudaSurfaceObject_t* ptrSurf , cudaResourceDesc& ressourceDescription);

	/*--------------------------------------*\
	|*		predefini		*|
	 \*-------------------------------------*/

	template<typename T>
	static void create( //
		cudaSurfaceObject_t* ptrSurf , 			// output
		cudaResourceDesc* ptrRessourceDescription , 	// output
		//
		cudaArray* ptrCudaArray , 			// input
		uint w , 					// input
		uint h)						// input
	    {
	    RessourceDescription::setup<T>(ptrRessourceDescription, ptrCudaArray, w, h);
	    Surface::create(ptrSurf, *ptrRessourceDescription);
	    }

	/*--------------------------------------*\
	|*		destroy			*|
	 \*-------------------------------------*/

	static void destroy(cudaSurfaceObject_t& surf);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
