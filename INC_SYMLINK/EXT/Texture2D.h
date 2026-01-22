#pragma once

#include "cudas.h"

#include "CudaArray.h"
#include "Textures.h"
#include "TextureDescription.h"

/**
 * with cudaArray
 */
template<class T>
class Texture2D
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	Texture2D(int w , int h , cudaTextureDesc& textureDescription) : //
		w(w), //
		h(h), //
		textureDescription(textureDescription)
	    {
	    create();
	    }

	Texture2D(int w , int h) : //
		w(w), //
		h(h)
	    {
	    TextureDescription::setup<T>(&textureDescription, AdressMode::DEFAUT_ADRESS_MODE);

	    create();
	    }

	virtual ~Texture2D()
	    {
	    Textures::destroy(texCuda);
	    CudaArray::free(ptrCudaArray);
	    }

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/**
	 * meme output cudaTextureObject_t que l'attribut public texCuda
	 */
	cudaTextureObject_t* memcpyHtoD(T* tabHost)
	    {
	    CudaArray::memcpyHtoD(ptrCudaArray, tabHost, w, h);

	    return &texCuda;
	    }

	/**
	 * meme output cudaTextureObject_t que l'attribut public texCuda
	 */
	cudaTextureObject_t* memcpyDtoD(T* tabSource)
	    {
	    CudaArray::memcpyDtoD(ptrCudaArray, tabSource, w, h);

	    return &texCuda;
	    }

	/**
	 * meme output cudaTextureObject_t que l'attribut public texCuda
	 */
	cudaTextureObject_t* memcpyDtoH(T* tabSource) // TODO a tester
	    {
	    CudaArray::memcpyDtoH(tabSource, ptrCudaArray, w, h);

	    return &texCuda;
	    }

	/*------------------*\
	 |*  private    *|
	 \*-----------------*/

    private:

	void create()
	    {
	    CudaArray::malloc<T>(&ptrCudaArray, w, h);

	    Textures::create<T>(&texCuda, &ressourceDescription, textureDescription, ptrCudaArray, w, h); // textureDescription INPUT
	    }

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int w;
	int h;

	// Tools cudarrays
	cudaArray* ptrCudaArray;
	cudaResourceDesc ressourceDescription;

	// Tools texture
	cudaTextureDesc textureDescription;

    public:
	// outputs
	cudaTextureObject_t texCuda;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
