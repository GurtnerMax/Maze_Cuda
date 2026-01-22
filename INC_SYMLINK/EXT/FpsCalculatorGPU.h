#pragma once

#include "FpsCalculator.h"
#include "RunnableGPU_I.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class FpsCalculatorGPU: public FpsCalculator
    {
	/*--------------------------------------*\
	    |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	FpsCalculatorGPU(RunnableGPU_I* ptrRunnable , bool isVerboseEnable = true , double nbSecondesProcessingMax = 8);

	FpsCalculatorGPU(RunnableGPU_I* ptrRunnable , double nbSecondesProcessingMax);

	virtual ~FpsCalculatorGPU();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    protected:

	/**
	 * override
	 */
	double animerLotS(long nbIterationByLot);

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

