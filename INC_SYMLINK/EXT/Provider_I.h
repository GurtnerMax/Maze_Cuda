#pragma once

#include "Grid.h"
#include "Image_I.h"
#include "ProviderUse_I.h"

#include "Animable_I.h"
#include "ImageFromAnimable.h" // utile pour implementation

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

template<typename T>
class Provider_I //: public ProviderUse_I
    {
	/*--------------------------------------*\
	   |*		Constructeur		*|
	 \*-------------------------------------*/

    public:

	virtual ~Provider_I()
	    {
	    //Rien
	    }

	/*--------------------------------------*\
	     |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	virtual Animable_I<T>* createAnimable(bool isVerbose = true)=0;

	virtual Animable_I<T>* createAnimable(const Grid& grid , bool isVerbose = false)=0;

	virtual Image_I* createImageGL()=0;

//	    /**
//	     * override
//	     */
//	    Use_I* createUse(const Grid& grid) // serait bien mais probleme de compilation, serprent qui se mort la queue avec ProviderUse_I.h (template, .h only)
//	        {
//	        return new UseFromProvider<T>(this, grid);
//	        }
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
