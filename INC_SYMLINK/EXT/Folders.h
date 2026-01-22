#pragma once

#include <iostream>
#include <string.h>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Folders
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/
    public:

	static void mkdirP(std::string folder);

    private:

	static void mkdirP_win(std::string folder);
	static void mkdirP_linux(std::string folder);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
