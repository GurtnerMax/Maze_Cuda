#pragma once

#include "cudas.h"

#include "AdressMode.h"
#include "RessourceDescription.h"
#include "TextureDescription.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Textures // TODO why copile pas avec name Texture??
    {

    public:

	/*--------------------------------------*\
	|*		custom			*|
	 \*-------------------------------------*/

	static void create(cudaTextureObject_t* ptrTex , cudaResourceDesc& ressourceDescription , cudaTextureDesc& textureDescription);

	/*--------------------------------------*\
	|*		predefini		*|
	 \*-------------------------------------*/

	/*-------------*\
	|* cudaArray	*|
	 \*------------*/

	/**
	 * adressMode:
	 *
	 * 	- DEFAUT_ADRESS_MODE	: cudaAddressModeWrap
	 * 	- BORDER		: cudaAddressModeBorder
	 * 	- MIRROR		: cudaAddressModeMirror 	(modulo)
	 * 	- WRAP 			: cudaAddressModeWrap		DEFAULT
	 * 	- CLAMP			: cudaAddressModeClamp 		(nearest)
	 */
	template<typename T>
	static void create( //
		cudaTextureObject_t* ptrTex , 			// output
		cudaResourceDesc* ptrRessourceDescription , 	// output
		cudaTextureDesc* ptrTextureDescription , 	// output
		//
		cudaArray* ptrCudaArray , 				// Input
		uint w , 						// Input
		uint h , 						// Input
		AdressMode adressMode = AdressMode::DEFAUT_ADRESS_MODE)	// Input
	    {
	    RessourceDescription::setup<T>(ptrRessourceDescription, ptrCudaArray, w, h);
	    TextureDescription::setup<T>(ptrTextureDescription, adressMode);

	    Textures::create(ptrTex, *ptrRessourceDescription, *ptrTextureDescription);
	    }


	template<typename T>
	static void create( //
		cudaTextureObject_t* ptrTex , 			// output
		cudaResourceDesc* ptrRessourceDescription , 	// output
		//
		cudaTextureDesc& textureDescription , 			// Input
		cudaArray* ptrCudaArray , 				// Input
		uint w , 						// Input
		uint h ) 						// Input
	    {
	    RessourceDescription::setup<T>(ptrRessourceDescription, ptrCudaArray, w, h);

	    Textures::create(ptrTex, *ptrRessourceDescription, textureDescription);
	    }

	/*-------------*\
	|*	gm	*|
	 \*------------*/

	/**
	 * adressMode:
	 *
	 * 	- DEFAUT_ADRESS_MODE	: cudaAddressModeWrap
	 * 	- BORDER		: cudaAddressModeBorder
	 * 	- MIRROR		: cudaAddressModeMirror 	(modulo)
	 * 	- WRAP 			: cudaAddressModeWrap		DEFAULT
	 * 	- CLAMP			: cudaAddressModeClamp 		(nearest)
	 */
	template<typename T>
	static void create( //
		cudaTextureObject_t* ptrTex , 				// output
		cudaResourceDesc* ptrRessourceDescription , 		// output
		cudaTextureDesc* ptrTextureDescription ,		// output
		//
		T* tabGM , 						// Input
		uint w , 						// Input
		uint h , 						// Input
		AdressMode adressMode = AdressMode::DEFAUT_ADRESS_MODE)	// Input
	    {
	    RessourceDescription::setup<T>(ptrRessourceDescription, tabGM, w, h);
	    TextureDescription::setup<T>(ptrTextureDescription, adressMode);

	    Textures::create(ptrTex, *ptrRessourceDescription, *ptrTextureDescription);
	    }


	template<typename T>
	static void create( //
		cudaTextureObject_t* ptrTex , 				// output
		cudaResourceDesc* ptrRessourceDescription , 		// output
		//
		cudaTextureDesc& textureDescription ,			// Input
		T* tabGM , 						// Input
		uint w , 						// Input
		uint h ) 						// Input
	    {
	    RessourceDescription::setup<T>(ptrRessourceDescription, tabGM, w, h);

	    Textures::create(ptrTex, *ptrRessourceDescription, textureDescription);
	    }

	/*--------------------------------------*\
	|*		destroy			*|
	 \*-------------------------------------*/

	static void destroy(cudaTextureObject_t& tex);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
