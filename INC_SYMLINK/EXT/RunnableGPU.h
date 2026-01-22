#pragma once

#include <iostream>
#include <string.h>

#include "Grid.h"
#include "cudas.h"

#include "RunnableGPU_I.h"

/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

class RunnableGPU: public RunnableGPU_I
    {
	/*------------------------------------------------------------*\
        |*			Constructors 			      *|
	 \*------------------------------------------------------------*/

    public:

	RunnableGPU(const Grid& grid , std::string title, bool isVerbose=false);

	virtual ~RunnableGPU();

	/*------------------------------------------------------------*\
        |*			Methodes 			      *|
	 \*------------------------------------------------------------*/

    public:

	/**
	 * override
	 */
	std::string getTitle();

	/**
	 * override
	 */
	Grid getGrid();

	/**
	 * override
	 */
	dim3 getDG();

	/**
	 * override
	 */
	dim3 getDB();

	/**
	 * override
	 */
	bool isVerbosity();

	/*------------------------------------------------------------*\
        |*			Attributs			      *|
	 \*------------------------------------------------------------*/

    protected:

	// Input
	Grid grid;
	dim3 dg;
	dim3 db;
	std::string title;
	bool isVerbose;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
