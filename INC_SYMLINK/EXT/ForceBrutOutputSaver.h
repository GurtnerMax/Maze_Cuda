#pragma once

#include <string>
#include <ostream>

#include "ForceBrutOutput.h"

using std::string;
using std::ostream;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class ForceBrutOutputSaver
    {
	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	ForceBrutOutputSaver(string file ,string folder, const string& title ,  ForceBrutOutput* ptrForceBrutOutput);

	virtual ~ForceBrutOutputSaver();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	void save() const;

    private:

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	string title;
	string file;
	string folder;
	ForceBrutOutput* ptrForceBrutOutput;

	int N;
	int M;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
