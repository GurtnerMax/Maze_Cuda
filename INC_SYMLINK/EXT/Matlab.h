#pragma once

#include <iostream>
#include <vector>
#include <string>

using std::string;
using std::ostream;
using std::vector;

enum PlotType
    {
    SURFACE,
    CURVE,
    ALL_GRAPHE
    };

class Matlab
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	Matlab();

	~Matlab();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	void record(string title , PlotType plotType=PlotType::ALL_GRAPHE);

	void play();

	/*--------------------------------------*\
	 |*		friend			*|
	 \*-------------------------------------*/

	friend ostream& operator <<(ostream& stream , const Matlab& matlab);

	/*--------------------------------------*\
	|*		private		*|
	 \*-------------------------------------*/

    private:

	void createInputMatlab();

	void callMatlab();

	void callMatlabLinux();

	void callMatlabWin();

	static string toString(PlotType plotType);

	static string* toArray(const vector<string>& list);

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	vector<string> listTitle;
	vector<PlotType> listOption;

	// Tools
	string inputMatlab;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

