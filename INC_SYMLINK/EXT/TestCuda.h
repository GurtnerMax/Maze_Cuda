#pragma once

#include "TestCuda_A.h"
#include "Iterator.h"
#include "Filter1D.h"

#include <iostream>
#include <string.h>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class TestCuda: public TestCuda_A
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	TestCuda(ProviderUse_I* ptrProviderUse);

	virtual ~TestCuda();

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    public:

	/*--------------*\
	|*	Config	 *|
	 \*--------------*/

	/**
	 * for  testGrid
	 */
	TestCuda* setDgx(Iterator iteratorDgx);

	/**
	 * for  testGrid
	 */
	TestCuda* setDbx(Iterator iteratorDbx);

	/**
	 * pour test simple
	 */
	TestCuda* setDgx(int dgxTestSimple);

	void enableTestMultiDevice(bool isEnable);

	void enableTestMonoThread(bool isEnable);

	void enableTestMonoBlock(bool isEnable);

	void enableTestPerformance(bool isEnable , long thresholdPerformanceFps);

	/*--------------*\
	|*	get	 *|
	 \*--------------*/

	Iterator* getIteratorDgx();

	Iterator* getIteratorDbx();

	/**
	 * pour test simple
	 */
	int getDgx();

    protected:

	/**
	 * Override
	 */
	virtual void allTests();

	// Tools
	void showTitle(string title2);

    private:

	// Simple

	void testSimple();

	void testDB2();

	void testDB4();

	void testDB8();

	void testDB16();

	void testDB32();

	void testDB64();

	void testDB128();

	void testDB256();

	void testDB512();

	void testDB1024();

	void testDBX(int dbx); // Tools

	// grid

	void testGrid();

	void testMultiDevice();

	// special

	void testMonoThread();

	void testMonoBlock();

	// tools
	void sleepMS(long DurationMS);

    protected:

	void testPerformance();

	void testPerformance(Grid grid,long thresholdPerformanceFps);

	void testGridTools(Iterator* ptrIteratorDGX,Iterator* ptrIteratorDBX,std::string title); // tools

	/*--------------------------------------*\
	|*		Attribut		*|
	 \*-------------------------------------*/

    protected:

	// Input
	bool isMultiDeviceEnable;
	bool isMonoThreadEnable;
	bool isMonoBlockEnable;
	bool isPerformanceEnable;
	long thresholdPerformanceFps;

	Iterator iteratorDgx;
	Iterator iteratorDbx;

	// Tools
	int dgxTestSimple;

    private:

	//Filter1D* ptrFilter1D;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

