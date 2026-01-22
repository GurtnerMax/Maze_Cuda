#pragma once

#include <iostream>
#include <string.h>

#include "Runnable_I.h"
#include "FpsCalculator.h"

using std::string;
using std::cout;
using std::cerr;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

namespace cpu
    {
    class Animator
	{
	    /*--------------------------------------*\
	    |*		Constructor			*|
	     \*-------------------------------------*/

	public:

	    Animator(Runnable_I* ptrRunnable , //
		    bool isVerboseEnable = true , //
		    double nbSecondesProcessingMax = 8);

	    virtual ~Animator(void);

	    /*--------------------------------------*\
	     |*		Methodes		*|
	     \*-------------------------------------*/

	public:

	    /**
	     * return fpsMedian
	     */
	    long run();

	    /*--------------*\
	    |*	get	*|
	     \*-------------*/

	    int getFpsGlobal();

	    int getFpsMean();
	    int getFpsMedian();
	    int getFpsMin();
	    int getFpsMax();
	    int getFpsStd();

	    //double getDurationMedianIteration1_MS();

	    Runnable_I* getRunnable() const;

	    /*--------------------------------------*\
	     |*		Attributs		*|
	     \*-------------------------------------*/

	private:

	    // Inputs
	    Runnable_I* ptrRunnable;
	    bool isVerboseEnable;
	    string title;
	    double nbSecondesProcessingMax;

	    // Tools
	    FpsCalculator* ptrFpsCalculator;
	};

    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

