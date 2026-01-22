#pragma once

/*----------------------------------------------------------------------*\
 |*			Enum	 					*|
 \*---------------------------------------------------------------------*/

/**
 * 	- cudaArrayDefault          : Default
 * 	- cudaArraySurfaceLoadStore : Allocates an array that can be read from or written to using a surface reference
 * 	- cudaArrayTextureGather    : This flag indicates that texture gather operations will be performed on the array (gather= rassembler)
 */
enum CudaArrayType
    {
    DEFAULT_CUARRAY,
    SURFACE_LOAD_STORE, //
    TEXTURE_GATHER
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
