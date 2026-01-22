#pragma once

#include <iostream>
#include <vector>

#include "cpptest.h"
#include "Grid.h"
#include "Maths.h"
#include "Chrono.h"
#include "ProviderUse_I.h"
#include "FilterGrid.h"
#include "FilterFromUse.h"

using std::string;

/*----------------------------------------------------------------------*\
 |*			Filter	 					*|
 \*---------------------------------------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Classe	 					*|
 \*---------------------------------------------------------------------*/

class TestCuda_A: public Test::Suite
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	TestCuda_A(ProviderUse_I* ptrProviderUse);

	virtual ~TestCuda_A();

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    public:

	TestCuda_A* enableVerbose(bool isVerbose);
	TestCuda_A* enableChrono(bool isChrono);

	/**
	 * <pre>
	 * filter	 : 	ptr de function:
	 * 	- Input  :	const Grid& grid
	 * 	- Output :	bool
	 *
	 * true => keep the grid
	 * </pre>
	 */
	void addFilter(FilterGrid* ptrFilterGrid);

    protected:

	virtual void allTests()=0;

	void test(const Grid& grid);

	/**
	 * keep the grid
	 */
	bool filter(const Grid& grid);

    private:

	/**
	 * Override
	 */
	void beforeRun();

	void show(const Grid& grid , bool isOk , Chrono* ptrChrono);

	/*--------------------------------------*\
	|*		Attribut		*|
	 \*-------------------------------------*/

    private:

	bool isChrono;

	// Tools
	std::vector<FilterGrid*> listFilter;
	FilterFromUse* ptrFilterFromUse;
	Use_I* ptrUse;

    protected:

	// Input
	ProviderUse_I* ptrProviderUse;

	// Enable
	string title;
	bool isVerbose;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

