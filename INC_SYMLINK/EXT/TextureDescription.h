#pragma once

#include <assert.h>

#include "cudas.h"

#include "AdressMode.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class TextureDescription
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/**
	 * adressMode:
	 *
	 * 	- DEFAULT	: cudaAddressModeWrap
	 * 	- BORDER	: cudaAddressModeBorder
	 * 	- MIRROR	: cudaAddressModeMirror 	(modulo)
	 * 	- WRAP 		: cudaAddressModeWrap		DEFAULT
	 * 	- CLAMP		: cudaAddressModeClamp 		(nearest)
	 */
	template<typename T>
	static void setup( //
		cudaTextureDesc* ptrTextureDescription , //
		AdressMode adressMode=AdressMode::DEFAUT_ADRESS_MODE)
	    {
	    // Init
		{
		memset(ptrTextureDescription, 0, sizeof(*ptrTextureDescription));
		}

	    // Description
	    // https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__TYPES.html
	    // https://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf
		{
		ptrTextureDescription->addressMode[0] = toNvidiaType(adressMode);
		ptrTextureDescription->addressMode[1] = toNvidiaType(adressMode);
		ptrTextureDescription->filterMode = cudaFilterModePoint; 		//  cudaFilterModePoint cudaFilterModeLinear
		ptrTextureDescription->normalizedCoords = false;
		// ptrTextureDescription.readMode = cudaReadModeElementType; 	// cudaReadModeElementType cudaReadModeNormalizedFloat
		}
	    }

    private:

	static cudaTextureAddressMode toNvidiaType(AdressMode adressMode)
	    {
	    switch (adressMode)
		{
	    case DEFAUT_ADRESS_MODE:
		return cudaAddressModeWrap;

	    case BORDER:
		return cudaAddressModeBorder;

	    case MIRROR:
		return cudaAddressModeMirror;

	    case WRAP:
		return cudaAddressModeClamp;

	    case CLAMP:
		return cudaAddressModeWrap;

	    default:
		{
		assert(false);
		return cudaAddressModeWrap;
		}
		}
	    }

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
