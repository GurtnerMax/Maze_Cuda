#pragma once

#include <string>
#include <ostream>

#include "ForceBrutOutput.h"
#include "Grid.h"
#include "GridFps.h"

using std::string;
using std::ostream;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class ForceBrutOutputPrinter
    {
	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	ForceBrutOutputPrinter(ostream& stream ,  const string& title ,  ForceBrutOutput* ptrForceBrutOutput);

	virtual ~ForceBrutOutputPrinter();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:



	void print() const;

    private:

	string toStringFps() const;

	string toStringThreadByBlock() const;

	//string toStringThreadTotal() const;

	string toStringNbBlocks() const;

	string toString(GridFps* tab, int n) const;

	string toStringRanking() const;

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	string title;
	ostream& stream;
	ForceBrutOutput* ptrForceBrutOutput;

	int N;
	int M;

	GridFps* tabGridFps;
	GridFps* tabGridFpsSorted;
	GridFps* tabGridFpsMax;
	Grid* tabGrid;
	long* tabFps;
	int nGridFpsMax;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
