#pragma once

#include <iostream>
#include <iomanip>
#include <assert.h>

#include "Use_I.h"
#include "Grid.h"
#include "RunnableGPU_I.h"

#include "RunnableFromAnimable.h"

#include "Provider_I.h"
//using namespace gpu;

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::setprecision;
//using gpu::RunnableFromAnimable;


/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

template<typename T>
class UseFromProvider: public Use_I
    {
	/*------------------------------------------------------------*\
        |*			Constructors 			      *|
	 \*------------------------------------------------------------*/

    public:

	UseFromProvider(Provider_I<T>* ptrProvider , const Grid& grid, bool isVerbose)
	    {
	    this->ptrAnimable = ptrProvider->createAnimable(grid,isVerbose);
	    this->ptrRunnableGPU = (RunnableGPU_I*)(new gpu::RunnableFromAnimable<T>(ptrAnimable));

	    assert(filter(grid));
	    }

	UseFromProvider(Provider_I<T>* ptrProvider, bool isVerbose)
	    {
	    this->ptrAnimable = ptrProvider->createAnimable(isVerbose);
	    this->ptrRunnableGPU = (RunnableGPU_I*)(new gpu::RunnableFromAnimable<T>(ptrAnimable));
	    }

	virtual ~UseFromProvider()
	    {
	    delete ptrRunnableGPU;
	    delete ptrAnimable;
	    }

	/*------------------------------------------------------------*\
        |*			Methodes 			      *|
	 \*------------------------------------------------------------*/

    public:

	/**
	 * Override
	 */
	bool isOk(bool isVerbose)
	    {
	    this->ptrRunnableGPU->run();

	    return true;
	    }

	/*--------------*\
	|*	get	*|
	 \*-------------*/

	/**
	 * Override
	 * Mandatory : user don't delete yourself RunnableGPU_I
	 */
	RunnableGPU_I* getRunnableGPU()
	    {
	    return this->ptrRunnableGPU;
	    }

    private:

	/*------------------------------------------------------------*\
        |*			Attributs			      *|
	 \*------------------------------------------------------------*/

    private:

	// Input

	// Tools
	Animable_I<T>* ptrAnimable;

	// Output
	RunnableGPU_I* ptrRunnableGPU;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
