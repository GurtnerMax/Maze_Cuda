#pragma once

#include <limits.h>
#include <float.h>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/**
 * http://pubs.opengroup.org/onlinepubs/009695399/basedefs/limits.h.html
 */
class Limits
    {
    public:

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

	/**
	 * DEPRECATED
	 */
	static void rappelTypeSize();

	/**
	 * Be careful between linux and windows limits!
	 */
	static void show();

    private:

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    public:

	static const short MAX_SHRT = SHRT_MAX;

	static const int MAX_INT = INT_MAX;

	static const long MAX_LONG = LONG_MAX;

	static const long long MAX_LLONG = LONG_MAX;

	// TODO ok en c++17, mais pas c++14
//	inline static const double MAX_DOUBLE=DBL_MAX;
//	inline static const float MAX_FLOAT=FLT_MAX;

// en attendant
	static const double MAX_DOUBLE;
	static const float MAX_FLOAT;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
