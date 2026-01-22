#pragma once

#include <iostream>
#include <string.h>

#include "RunnableGPU_I.h"
#include "Animable_I.h"
#include "cudas.h"
#include "GM.h"
#include "Kernel.h"

using std::string;
using std::cout;
using std::cerr;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

namespace gpu
    {
    template<typename T>
    class RunnableFromAnimable: public RunnableGPU_I
	{
	    /*------------------------------------------------------------*\
	    |*			Constructors 			      *|
	     \*------------------------------------------------------------*/

	public:

	    RunnableFromAnimable(Animable_I<T>* ptrAnimable) : //
		    ptrAnimable(ptrAnimable), //
		    domaineMath(ptrAnimable->getDomaineMathInit()), //
		    w(ptrAnimable->getW()), //
		    h(ptrAnimable->getH())
		{
		// MM
		    {
		    size_t sizeGM = w * h * sizeof(T);

		    GM::malloc0(&ptrImageGM, sizeGM);
		    }

		}

	    virtual ~RunnableFromAnimable()
		{
		GM::free(ptrImageGM);
		}

	    /*------------------------------------------------------------*\
        |*			Methodes 			      *|
	     \*------------------------------------------------------------*/

	public:

	    /**
	     * Override
	     */
	    void run()
		{
		ptrAnimable->process(ptrImageGM, w, h, domaineMath);
		ptrAnimable->animationStep();
		}

	    /**
	     * Override
	     */
	    inline string getTitle()
		{
		return ptrAnimable->getTitle();
		}

	    /**
	     * Override
	     */
	    inline Grid getGrid()
		{
		return ptrAnimable->getGrid();
		}

	    /**
	     * Override
	     */
	    inline dim3 getDG()
		{
		return ptrAnimable->getDg();
		}

	    /**
	     * Override
	     */
	    inline dim3 getDB()
		{
		return ptrAnimable->getDb();
		}

	    /**
	     * Override
	     */
	    inline double getInputGO()
		{
		return ptrAnimable->getInputGO();
		}

	    /**
	     * Override
	     */
	    inline  double getOutputGO()
		{
		return ptrAnimable->getOutputGO();
		}

	    /**
	     * Override
	     */
	    inline bool isVerbosity()
		{
		return ptrAnimable->isVerbosity();
		}

	    /*------------------------------------------------------------*\
        |*			Attributs			      *|
	     \*------------------------------------------------------------*/

	private:

	    // Input
	    Animable_I<T>* ptrAnimable;

	    int w;
	    int h;
	    DomaineMath domaineMath;

	    // Tools
	    T* ptrImageGM;

	};

    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
