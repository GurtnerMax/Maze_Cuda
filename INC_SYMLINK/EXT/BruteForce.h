#pragma once

#include "ProviderUse_I.h"
#include "Matlab.h"
#include "GridMaillage.h"

/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

class BruteForce
    {

	/*------------------------------------------------------------*\
        |*			Methodes 			      *|
	 \*------------------------------------------------------------*/

    public:

	static void run(ProviderUse_I* ptrProviderUse , GridMaillage* ptrGridMaillage, Matlab* ptrMatlab , const PlotType& plotType , double durationMaxS = 0.5);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
