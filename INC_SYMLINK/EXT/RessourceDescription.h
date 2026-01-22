#pragma once

#include "cudas.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class RessourceDescription
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	template<typename T>
	static void setup(cudaResourceDesc* ptrRessourceDescription , cudaArray* ptrCudaArray , uint w , uint h)
	    {
	    // Init
		{
		memset(ptrRessourceDescription, 0, sizeof(*ptrRessourceDescription));
		}

	    // Description
		{
		ptrRessourceDescription->resType = cudaResourceTypeArray; 		// Type cudaArray  	[WARNING]
		ptrRessourceDescription->res.pitch2D.devPtr = ptrCudaArray; 		// tableau des valeurs 	[WARNING]

		ptrRessourceDescription->res.pitch2D.width = w;				// largeur du tableau
		ptrRessourceDescription->res.pitch2D.height = h;			// hauteur du tableau
		ptrRessourceDescription->res.pitch2D.desc = cudaCreateChannelDesc<T>();	// T = type data
		ptrRessourceDescription->res.pitch2D.pitchInBytes = w * sizeof(T);	// taille d'une ligne
		}
	    }

	template<typename T>
	static void setup(cudaResourceDesc* ptrRessourceDescription , T* ptrTabGM , uint w , uint h)
	    {
	    // Init
		{
		memset(ptrRessourceDescription, 0, sizeof(*ptrRessourceDescription));
		}

	    // Description
		{
		ptrRessourceDescription->resType = cudaResourceTypePitch2D; 		// Type tab2D GM	[WARNING]
		ptrRessourceDescription->res.pitch2D.devPtr = ptrTabGM; 		// tableau des valeurs 	[WARNING]

		ptrRessourceDescription->res.pitch2D.width = w;				// largeur du tableau
		ptrRessourceDescription->res.pitch2D.height = h;			// hauteur du tableau
		ptrRessourceDescription->res.pitch2D.desc = cudaCreateChannelDesc<T>();	// T = type de la GM
		ptrRessourceDescription->res.pitch2D.pitchInBytes = w * sizeof(T);	// taille d'une ligne
		}
	    }
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
