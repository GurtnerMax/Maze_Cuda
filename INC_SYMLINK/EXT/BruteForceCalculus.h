#pragma once

#include "ForceBrutOutput.h"
#include "FpsCalculator.h"
#include "RunnableGPU_I.h"
#include "GridMaillage.h"
#include "Iterator.h"
#include "ProviderUse_I.h"

using std::to_string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

namespace gpu
    {

    class BruteForceCalculus
	{
	    /*--------------------------------------*\
	     |*		Constructor		*|
	     \*-------------------------------------*/

	public:

	    BruteForceCalculus(ProviderUse_I* ptrProviderUse , //
		    GridMaillage* ptrGridMaillage ,  //
		    double durationMaxS = 0.5 , //
		    bool verboseBruteForceEnable = true);

	    virtual ~BruteForceCalculus();

	    /*--------------------------------------*\
	     |*		Methodes		*|
	     \*-------------------------------------*/

	public:

	    ForceBrutOutput* run();

	private:

	    long fps(Grid* ptrGrid);

	    double estimationDurationS();

	    double estimationDurationS(Grid* ptrGrid);

	    void createTitle();

	    std::string titleProgress();

	    /*--------------------------------------*\
	     |*		Attributs		*|
	     \*-------------------------------------*/

	private:

	    // Inputs
	    ProviderUse_I* ptrProviderUse;
	    GridMaillage* ptrGridMaillage;
	    bool verboseBruteForceEnable;
	    bool verboseFpsCalculator;
	    double durationMaxS;
	    int maillageSize;

	    // Tools
	    GridFps* tabGridFps;
	    std::string title;

	    // Outputs
	    ForceBrutOutput* ptrOutput;
	};
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
