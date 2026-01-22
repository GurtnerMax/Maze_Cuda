#pragma once

#include <iostream>
#include <string>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class TabTools
    {
    public:

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

	TabTools();
	virtual ~TabTools();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	static double reduction(double* tab , int n);

	static void init(double* tab , int n , double a);

	static double mean(double* tab , int n);

	static double median(double* tab , int n);

	static double std(double* tab , int n , double mean);

	static void minMax(double* tab , int n , double* ptrMin, double* ptrMax);

	static void print(double* tab , int n , std::string title = "");

	/**
	 * in place
	 */
	static void sort(double* tab , int n);

    private:

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
