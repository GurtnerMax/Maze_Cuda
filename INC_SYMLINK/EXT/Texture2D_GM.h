#pragma once

#include "cudas.h"

#include "Textures.h"
#include "TextureDescription.h"

template<class T>
class Texture2D_GM
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	Texture2D_GM(int w , int h , cudaTextureDesc& textureDescription) : //
		w(w), //
		h(h), //
		textureDescription(textureDescription), //
		isFirstTime(true)
	    {
	    // create() // pas possible ici car tabGM pas encore connu
	    }

	Texture2D_GM(int w , int h) : //
		w(w), //
		h(h), //
		isFirstTime(true)
	    {
	    // create() // pas possible ici car tabGM pas encore connu

	    TextureDescription::setup<T>(&textureDescription, AdressMode::DEFAUT_ADRESS_MODE);
	    }

	virtual ~Texture2D_GM()
	    {
	    Textures::destroy(texCuda);
	    }

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/**
	 * return meme objet que attribut public tex2d
	 */
	cudaTextureObject_t* see(T* tabGM)
	    {
	    if (isFirstTime)
		{
		isFirstTime = false;

		this->tabGM = tabGM;

		create();
		}

	    return &texCuda;
	    }

	/*--------------------------------------*\
	 |*		private			*|
	 \*-------------------------------------*/

    private:

	void create()
	    {
	    // Input  : textureDescription
	    // Output : ressourceDescription (create defaut by Textures)
	    Textures::create<T>(&texCuda, &ressourceDescription, textureDescription, tabGM, w, h);
	    }

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int w;
	int h;
	T* tabGM;

	// Tools
	cudaResourceDesc ressourceDescription;
	cudaTextureDesc textureDescription;
	bool isFirstTime;

    public:
	// outputs
	cudaTextureObject_t texCuda;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
