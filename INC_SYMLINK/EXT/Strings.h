#pragma once

#include <string>

using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Strings
    {
    public:

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

	static string toString(int number);
	static string toString(unsigned int number);
	static string toString(long number);
	static string toString(float number);
	static string toString(double number);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

