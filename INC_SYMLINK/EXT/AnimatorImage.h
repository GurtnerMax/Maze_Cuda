#pragma once

#include <iostream>

#include "RunnableFromAnimable.h"
#include "Animable_I.h"
#include "FpsCalculator.h"

using std::string;
using std::cout;
using std::cerr;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

template<typename T>
class AnimatorImage
    {
	/*------------------------------------------------------------*\
	    |*			Constructors 			      *|
	 \*------------------------------------------------------------*/

    public:

	AnimatorImage(Animable_I<T>* ptrAnimable , bool isVerboseEnable = true , double nbSecondesProcessingMax = 8) :
		ptrAnimable(ptrAnimable), //
		isVerboseEnable(isVerboseEnable), //
		nbSecondesProcessingMax(nbSecondesProcessingMax)
	    {
	    // rien
	    }

	virtual ~AnimatorImage()
	    {
	    delete ptrFpsCalculator;
	    }

	/*------------------------------------------------------------*\
        |*			Methodes 			      *|
	 \*------------------------------------------------------------*/

    public:

	/**
	 * return fpsMedian
	 */
	long run()
	    {
	    gpu::RunnableFromAnimable<T> runnableFromAnimable(ptrAnimable);
	    RunnableGPU_I* ptrRunnableGPU = (RunnableGPU_I*)&runnableFromAnimable;

	    this->ptrFpsCalculator = new FpsCalculator(ptrRunnableGPU, isVerboseEnable, nbSecondesProcessingMax);

	    long fpsMedian = ptrFpsCalculator->run();
	    this->ptrFpsCalculator->print();

	    return fpsMedian;
	    }

	/*------------------------------------------------------------*\
        |*			Attributs			      *|
	 \*------------------------------------------------------------*/

    private:

	// Input
	Animable_I<T>* ptrAnimable;

	double nbSecondesProcessingMax;
	bool isVerboseEnable;

	// Tools
	FpsCalculator* ptrFpsCalculator;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
