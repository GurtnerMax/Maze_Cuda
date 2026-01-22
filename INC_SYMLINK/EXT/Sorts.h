#pragma once

#include "GridFps.h"

/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

class Sorts
    {

	/*------------------------------------------------------------*\
        |*			Methodes 			      *|
	 \*------------------------------------------------------------*/

    public:

	/**
	 * in place, decrease
	 */
	static void sort(GridFps* tabGridFps , int n);

    private:

	/**
	 * in place, decrease
	 */
	static void cbi(GridFps* tabGridFps , int n);

	/**
	 * in place, decrease
	 */
	static void stdv1(GridFps* tabGridFps , int n);

	/**
	 * in place, decrease
	 */
	static void stdv2(GridFps* tabGridFps , int n);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
