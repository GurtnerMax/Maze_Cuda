#pragma once

#include "Provider_I.h"
#include "AnimatorImage.h"

//using namespace gpu;

/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

template<typename T>
class BenchmarkImage
    {

	/*------------------------------------------------------------*\
        |*			Methodes 			      *|
	 \*------------------------------------------------------------*/

    public:

	static void run(Provider_I<T>* ptrProvider , double durationMaxS)
	    {
	    const bool IS_ANIMABLE_VERBOSITY = false;
	    const bool IS_FPS_VERBOSITY = true;

	    Animable_I<T>* ptrAnimable = ptrProvider->createAnimable(IS_ANIMABLE_VERBOSITY);

	    AnimatorImage<T> animator(ptrAnimable, IS_FPS_VERBOSITY, durationMaxS);
	    animator.run();

	    delete ptrAnimable;
	    }

	/*------------------------------------------------------------*\
        |*			Attributs			      *|
	 \*------------------------------------------------------------*/

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
